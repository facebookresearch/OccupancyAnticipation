#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC

import cv2
import math
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from occant_baselines.rl.policy import (
    Mapper,
    GlobalPolicy,
    LocalPolicy,
    HeuristicLocalPolicy,
)
from occant_baselines.rl.planner import (
    AStarPlannerVector,
    AStarPlannerSequential,
)
from occant_utils.common import (
    spatial_transform_map,
    convert_world2map,
    convert_map2world,
    subtract_pose,
    add_pose,
    crop_map,
)
from einops import rearrange, reduce, asnumpy


class ActiveNeuralSLAMExplorer:
    def __init__(self, config, projection_unit):
        self.config = config
        torch.manual_seed(config.pyt_random_seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.mapper = Mapper(config.MAPPER, projection_unit)
        self.global_policy = GlobalPolicy(config.GLOBAL_POLICY)
        self.local_policy = LocalPolicy(config.LOCAL_POLICY)
        self.planner = AStarPlannerVector(config.PLANNER)
        self.nplanners = self.config.PLANNER.nplanners
        self.planning_step = int(self.config.planning_step / config.MAPPER.map_scale)
        self.goal_success_radius = self.config.goal_success_radius
        self.goal_interval = self.config.goal_interval

        self.states = {
            # Planning states
            "current_global_goals": None,
            "current_local_goals": None,
            "prev_dist2localgoal": None,
            "dist2localgoal": None,
            # Global map reward states
            "prev_map_states": None,
        }
        self._debug = False

    def act(
        self,
        observations,
        prev_observations,
        prev_state_estimates,
        ep_time,
        masks,
        deterministic=False,
    ):
        t = ep_time  # (bs, 1)
        ep_step = ep_time[0].item()
        M = prev_state_estimates["map_states"].shape[2]
        s = self.mapper.map_config["scale"]
        device = observations["rgb"].device
        SAMPLE_GLOBAL_GOAL_FLAG = ep_step % self.goal_interval == 0

        assert M % 2 == 1  # The code is currently tested for odd M only

        # Update map and pose
        mapper_inputs = self._create_mapper_inputs(
            observations, prev_observations, prev_state_estimates
        )
        mapper_outputs = self.mapper(mapper_inputs)

        # Precompute useful variables
        rgb_at_t = observations["rgb"]
        pose_hat_at_t = mapper_outputs["xt_hat"]
        map_at_t = mapper_outputs["mt"]
        map_states = map_at_t
        recurrent_hidden_states = prev_state_estimates[
            "recurrent_hidden_states"
        ]  # (bs, 256)
        visited_states = prev_state_estimates["visited_states"]  # (bs, 1, M, M)

        # Agent's map position
        agent_position_at_t = pose_hat_at_t[:, :2]
        agent_map_xy = convert_world2map(agent_position_at_t, (M, M), s)
        agent_map_xy = torch.clamp(agent_map_xy, 0, M - 1)

        # Debugging
        if self._debug:
            mt_img = self.generate_visualization(map_at_t)
            cv2.imshow(
                "Full map", np.flip(mt_img.reshape(-1, *mt_img.shape[2:]), axis=2)
            )
            cv2.waitKey(30)

        # Update state visitation
        visited_states = self._update_state_visitation(visited_states, agent_map_xy)

        # Update status for current local goals
        if ep_step > 0:
            self.states["prev_dist2localgoal"] = self.states["dist2localgoal"]
            # This is computed here to check for goal completion. It may be updated later.
            self.states["dist2localgoal"] = (
                torch.norm(agent_map_xy - self.states["current_local_goals"], dim=1) * s
            )

        # Compute rewards for previous action
        local_rewards = self._compute_local_rewards(agent_map_xy, ep_step, s)

        # Sample a new goal if needed
        if SAMPLE_GLOBAL_GOAL_FLAG:
            global_policy_inputs = self._create_global_policy_inputs(
                map_states, visited_states, agent_map_xy
            )

            (
                global_value,
                global_action,
                global_action_log_probs,
                _,
            ) = self.global_policy.act(global_policy_inputs, None, None, None)

            # Convert action to location (row-major format)
            global_action_map_x = torch.fmod(
                global_action.squeeze(1), self.global_policy.G
            ).float()  # (bs, )
            global_action_map_y = (
                global_action.squeeze(1) / self.global_policy.G
            ).float()  # (bs, )

            # Convert to MxM map coordinates
            global_action_map_x = (
                global_action_map_x * M / self.global_policy.G
            ).long()
            global_action_map_y = (
                global_action_map_y * M / self.global_policy.G
            ).long()

            global_action_map_xy = torch.stack(
                [global_action_map_x, global_action_map_y], dim=1
            )  # (bs, 2)

            # Set the goal
            self.states[
                "current_global_goals"
            ] = global_action_map_xy  # (bs, 2) --- (x, y) in map image coordinates
        else:
            global_policy_inputs = None
            global_value, global_action, global_action_log_probs = None, None, None

        if SAMPLE_GLOBAL_GOAL_FLAG:
            # Condition 1: A new global goal was sampled
            SAMPLE_LOCAL_GOAL_FLAGS = [1 for _ in range(self.nplanners)]
            self.states["current_local_goals"] = torch.zeros(self.nplanners, 2).to(
                device
            )
        else:
            # Condition 2: The previous local goal was reached
            SAMPLE_LOCAL_GOAL_FLAGS = asnumpy(
                (self.states["dist2localgoal"] < self.goal_success_radius).float()
            ).tolist()

        # Execute planner and compute local goals
        self._compute_plans_and_local_goals(
            map_states, agent_map_xy, SAMPLE_LOCAL_GOAL_FLAGS
        )

        # Recompute dist2localgoal to account for any changes in local goals
        self.states["dist2localgoal"] = (
            torch.norm(agent_map_xy - self.states["current_local_goals"], dim=1) * s
        )

        # Follow the plan with the local policy
        relative_goals = self._compute_relative_local_goals(pose_hat_at_t, M, s)

        local_policy_inputs = {
            "rgb_at_t": rgb_at_t,
            "goal_at_t": relative_goals,
            "t": t,
        }

        (
            local_value,
            local_action,
            local_action_log_probs,
            recurrent_hidden_states,
        ) = self.local_policy.act(
            local_policy_inputs, recurrent_hidden_states, None, masks
        )

        state_estimates = {
            "recurrent_hidden_states": recurrent_hidden_states,
            "map_states": map_states,
            "visited_states": visited_states,
            "pose_estimates": pose_hat_at_t,
        }
        local_policy_outputs = {
            "values": local_value,
            "actions": local_action,
            "action_log_probs": local_action_log_probs,
        }
        global_policy_outputs = {
            "values": global_value,
            "actions": global_action,
            "action_log_probs": global_action_log_probs,
        }
        rewards = {
            "local_rewards": local_rewards,
        }

        return (
            mapper_inputs,
            local_policy_inputs,
            global_policy_inputs,
            mapper_outputs,
            local_policy_outputs,
            global_policy_outputs,
            state_estimates,
            rewards,
        )

    def _process_maps(self, maps):
        """
        Inputs:
            maps - (bs, 2, M, M) --- 1st channel is prob of obstacle present
                                 --- 2nd channel is prob of being explored
        """
        obstacle_mask = (maps[:, 0] > self.config.thresh_obstacle) & (
            maps[:, 1] > self.config.thresh_explored
        )
        final_maps = asnumpy(obstacle_mask.float())
        kernel = np.ones((3, 3))
        for i in range(final_maps.shape[0]):
            final_maps[i] = cv2.dilate(final_maps[i], kernel, iterations=1)

        return final_maps

    def _sample_random_explored(self, agent_map, agent_pos, d_thresh=30):
        """
        Inputs:
            agent_map - (2, M, M) --- 1st channel is prob of obstacle present
                                  --- 2nd channel is prob of being explored

        Sampled random explored locations within a distance d_thresh from the agent_pos.
        """
        free_mask = (agent_map[0] <= self.config.thresh_obstacle) & (
            agent_map[1] > self.config.thresh_explored
        )
        final_maps = asnumpy(free_mask.float())
        kernel = np.ones((3, 3))
        valid_locs = np.where(final_maps > 0)
        if valid_locs[0].shape[0] == 0:
            rand_x = agent_pos[0] + random.randint(-15, 15)
            rand_y = agent_pos[1] + random.randint(-15, 15)
        else:
            explored_locations = np.array(list(zip(*valid_locs)))  # (N, 2)
            explored_locations = np.flip(explored_locations, axis=1)
            # Pick some random explored location near agent's position
            dist2explored = np.linalg.norm(
                np.array(agent_pos)[np.newaxis, :] - explored_locations, axis=1
            )  # (N, )
            valid_explored_locations = explored_locations[dist2explored < d_thresh]
            if valid_explored_locations.shape[0] == 0:
                rand_x = agent_pos[0] + random.randint(-15, 15)
                rand_y = agent_pos[1] + random.randint(-15, 15)
            else:
                rand_x, rand_y = np.random.permutation(valid_explored_locations)[0]
                rand_x = float(rand_x.item())
                rand_y = float(rand_y.item())
        return (rand_x, rand_y)

    def _has_reached_goal(agent_position, goal_position):
        if agent_position is None or goal_position is None:
            return False
        if np.linalg.norm(agent_position - goal_position) < self.goal_success_radius:
            return True
        else:
            return False

    def to(self, device):
        self.mapper.to(device)
        self.global_policy.to(device)
        self.local_policy.to(device)

    def train(self):
        self.mapper.train()
        self.global_policy.train()
        self.local_policy.train()

    def eval(self):
        self.mapper.eval()
        self.global_policy.eval()
        self.local_policy.eval()

    def parameters(self):
        return chain(
            self.mapper.parameters(),
            self.global_policy.parameters(),
            self.local_policy.parameters(),
        )

    def state_dict(self):
        return {
            "mapper": self.mapper.state_dict(),
            "global_policy": self.global_policy.state_dict(),
            "local_policy": self.local_policy.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.mapper.load_state_dict(state_dict["mapper"])
        self.global_policy.load_state_dict(state_dict["global_policy"])
        self.local_policy.load_state_dict(state_dict["local_policy"])

    def reset(self):
        for k in self.states:
            self.states[k] = None

    def get_states(self):
        return copy.deepcopy(self.states)

    def update_states(self, states):
        self.states = states

    def _compute_relative_local_goals(self, agent_world_pose, M, s):
        """
        Converts a local goal (x, y) position in the map to egocentric coordinates▫
        relative to agent's current pose.
        """
        local_map_goals = self.states["current_local_goals"]
        local_world_goals = convert_map2world(local_map_goals, (M, M), s)  # (bs, 2)
        # Concatenate dummy directions to goal
        local_world_goals = torch.cat(
            [
                local_world_goals,
                torch.zeros(self.nplanners, 1).to(agent_world_pose.device),
            ],
            dim=1,
        )
        relative_goals = subtract_pose(agent_world_pose, local_world_goals)[:, :2]

        return relative_goals

    def _compute_local_rewards(
        self, agent_map_xy, ep_step, s,
    ):
        local_rewards = torch.zeros(self.nplanners, 1).to(agent_map_xy.device)
        if ep_step > 0:
            # Reward the local agent based on the reduction in distance to the (older) target
            pd2g = self.states["prev_dist2localgoal"]
            d2g = self.states["dist2localgoal"]
            if self.config.local_reward_type == "diff":
                local_rewards = (pd2g - d2g).unsqueeze(1)
            elif self.config.local_reward_type == "exp":
                local_rewards = torch.exp(
                    -d2g / self.config.local_reward_temperature
                ).unsqueeze(1)
            else:
                raise ValueError(
                    f"ActiveNeuralMappingNet - _compute_local_rewards(): Reward type {self.config.local_reward_type} is not defined!"
                )

        return local_rewards

    def _create_mapper_inputs(
        self, observations, prev_observations, prev_state_estimates
    ):
        rgb_at_t_1 = prev_observations["rgb"]  # (bs, H, W, C)
        depth_at_t_1 = prev_observations["depth"]  # (bs, H, W, 1)
        ego_map_gt_at_t_1 = prev_observations["ego_map_gt"]  # (bs, Hby2, Wby2, 2)
        pose_at_t_1 = prev_observations["pose"]  # (bs, 3)
        rgb_at_t = observations["rgb"]  # (bs, H, W, C)
        depth_at_t = observations["depth"]  # (bs, H, W, 1)
        ego_map_gt_at_t = observations["ego_map_gt"]  # (bs, Hby2, Wby2, 2)
        pose_at_t = observations["pose"]  # (bs, 3)
        # This happens only for a baseline
        if "ego_map_gt_anticipated" in prev_observations:
            ego_map_gt_anticipated_at_t_1 = prev_observations["ego_map_gt_anticipated"]
            ego_map_gt_anticipated_at_t = observations["ego_map_gt_anticipated"]
        else:
            ego_map_gt_anticipated_at_t_1 = None
            ego_map_gt_anticipated_at_t = None
        pose_hat_at_t_1 = prev_state_estimates["pose_estimates"]  # (bs, 3)
        map_at_t_1 = prev_state_estimates["map_states"]  # (bs, 2, M, M)

        mapper_inputs = {
            "rgb_at_t_1": rgb_at_t_1,
            "depth_at_t_1": depth_at_t_1,
            "ego_map_gt_at_t_1": ego_map_gt_at_t_1,
            "ego_map_gt_anticipated_at_t_1": ego_map_gt_anticipated_at_t_1,
            "pose_at_t_1": pose_at_t_1,
            "pose_hat_at_t_1": pose_hat_at_t_1,
            "map_at_t_1": map_at_t_1,
            "rgb_at_t": rgb_at_t,
            "depth_at_t": depth_at_t,
            "ego_map_gt_at_t": ego_map_gt_at_t,
            "ego_map_gt_anticipated_at_t": ego_map_gt_anticipated_at_t,
            "pose_at_t": pose_at_t,
        }

        return mapper_inputs

    def _update_state_visitation(self, visited_states, agent_map_xy):
        """
        visited_states - (bs, 1, V, V) tensor with 0s for unvisited locations, 1s for visited locations
        agent_map_xy - (bs, 2) agent's current position on the map
        """
        agent_map_x = agent_map_xy[:, 0].long()  # (bs, )
        agent_map_y = agent_map_xy[:, 1].long()  # (bs, )
        visited_states[:, 0, agent_map_y, agent_map_x] = 1

        return visited_states

    def _compute_plans_and_local_goals(
        self, map_states, agent_map_xy, SAMPLE_LOCAL_GOAL_FLAGS
    ):
        """
        map_states - (bs, 2, V, V) tensor
        agent_map_xy - (bs, 2) agent's current position on the map
        """
        # Make a plan from agent position to the goal
        maps_np = self._process_maps(map_states)
        goal_positions_np = asnumpy(self.states["current_global_goals"]).astype(
            np.int32
        )  # (bs, 2)
        current_positions_np = asnumpy(agent_map_xy).astype(np.int32)  # (bs, 2)
        plans = self.planner.plan(
            maps_np, current_positions_np, goal_positions_np, SAMPLE_LOCAL_GOAL_FLAGS
        )

        # Pick a local goal
        for i in range(self.nplanners):
            if SAMPLE_LOCAL_GOAL_FLAGS[i] != 1:
                continue
            path_x, path_y = plans[i]

            # If planning failed, sample random local goal.
            if path_x is None:
                goal_x, goal_y = self._sample_random_explored(
                    map_states[i], current_positions_np[i].tolist()
                )
            else:
                delta = min(self.planning_step, len(path_x) - 1)
                # The path is in reverse order
                goal_x, goal_y = path_x[-delta], path_y[-delta]

            self.states["current_local_goals"][i, 0] = goal_x
            self.states["current_local_goals"][i, 1] = goal_y

    def _create_global_policy_inputs(self, map_states, visited_states, agent_map_xy):
        """
        map_states     - (bs, 2, V, V) - map occupancy, explored states
        visited_states - (bs, 1, V, V) - agent visitation status on the map
        agent_map_xy   - (bs, 2) - agent's XY position on the map
        """
        agent_map_x = agent_map_xy[:, 0].long()  # (bs, )
        agent_map_y = agent_map_xy[:, 1].long()  # (bs, )
        agent_position_onehot = torch.zeros_like(visited_states)
        agent_position_onehot[:, 0, agent_map_y, agent_map_x] = 1
        h_t = torch.cat(
            [map_states, visited_states, agent_position_onehot], dim=1
        )  # (bs, 4, M, M)

        global_policy_inputs = {
            "pose_in_map_at_t": agent_map_xy,
            "map_at_t": h_t,
        }

        return global_policy_inputs


class ActiveNeuralSLAMNavigator:
    def __init__(self, config, projection_unit):
        self.config = config
        self.mapper = Mapper(config.MAPPER, projection_unit)
        self.local_policy = LocalPolicy(config.LOCAL_POLICY)
        self.planner = AStarPlannerVector(config.PLANNER)
        self.nplanners = self.config.PLANNER.nplanners
        self.planning_step = int(self.config.planning_step / config.MAPPER.map_scale)
        self.goal_success_radius = self.config.goal_success_radius
        self.stop_action_id = self.config.stop_action_id
        self.left_action_id = self.config.left_action_id

        self.states = {
            # Planning states
            "current_global_goals": None,
            "current_local_goals": None,
            "dist2localgoal": None,
        }
        self._debug = False

    def act(
        self,
        observations,
        prev_observations,
        prev_state_estimates,
        ep_time,
        masks,
        deterministic=False,
    ):
        t = ep_time  # (bs, 1)
        bs = ep_time.shape[0]
        ep_step = ep_time[0].item()
        M = prev_state_estimates["map_states"].shape[2]
        s = self.mapper.map_config["scale"]
        device = observations["rgb"].device

        assert M % 2 == 1  # The code is currently tested for odd M only

        # Update map and pose
        mapper_inputs = self._create_mapper_inputs(
            observations, prev_observations, prev_state_estimates
        )
        mapper_outputs = self.mapper(mapper_inputs)

        # Precompute useful variables
        rgb_at_t = observations["rgb"]
        pose_hat_at_t = mapper_outputs["xt_hat"]
        map_at_t = mapper_outputs["mt"]
        map_states = map_at_t
        recurrent_hidden_states = prev_state_estimates[
            "recurrent_hidden_states"
        ]  # (bs, 256)

        # Agent's map position
        agent_position_at_t = pose_hat_at_t[:, :2]
        agent_map_xy = convert_world2map(agent_position_at_t, (M, M), s)
        agent_map_xy = torch.clamp(agent_map_xy, 0, M - 1)

        if self._debug:
            mt_img = self.generate_visualization(map_at_t)
            cv2.imshow(
                "Full map", np.flip(mt_img.reshape(-1, *mt_img.shape[2:]), axis=2)
            )
            cv2.waitKey(30)

        # Update goal location
        # Convention for pointgoal: x is forward, y is rightward in the start-pose coordinates.
        goal_at_t_polar = observations["pointgoal"]  # (bs, 2) --- (rho, phi)
        goal_at_t = self._convert_polar2cartesian(goal_at_t_polar)
        goal_map_xy = convert_world2map(goal_at_t, (M, M), s)
        self.states["current_global_goals"] = goal_map_xy
        goal_at_t_aug = torch.cat([goal_at_t, torch.zeros(bs, 1).to(device)], dim=1)
        relative_goal_at_t = subtract_pose(pose_hat_at_t, goal_at_t_aug)[
            :, :2
        ]  # (x, y)

        # Update status for current local goals
        if ep_step > 0:
            self.states["dist2localgoal"] = (
                torch.norm(agent_map_xy - self.states["current_local_goals"], dim=1) * s
            )

        if ep_step == 0:
            SAMPLE_LOCAL_GOAL_FLAGS = [1.0 for _ in range(bs)]
            self.states["current_local_goals"] = torch.zeros(self.nplanners, 2).to(
                device
            )
        elif ep_step % 10 == 0:
            SAMPLE_LOCAL_GOAL_FLAGS = [1.0 for _ in range(bs)]
        else:
            SAMPLE_LOCAL_GOAL_FLAGS = (
                (self.states["dist2localgoal"] < self.goal_success_radius)
                .float()
                .detach()
                .cpu()
            )

        # Execute planner and compute local goals
        self._compute_plans_and_local_goals(
            map_states, agent_map_xy, SAMPLE_LOCAL_GOAL_FLAGS
        )

        # Recompute dist2localgoal to account for any changes in local goals
        self.states["dist2localgoal"] = (
            torch.norm(agent_map_xy - self.states["current_local_goals"], dim=1) * s
        )

        # Follow the plan with the local policy
        relative_goals = self._compute_relative_local_goals(pose_hat_at_t, M, s)

        local_policy_inputs = {
            "rgb_at_t": rgb_at_t,
            "goal_at_t": relative_goals,
            "t": t,
        }

        (
            local_value,
            local_action,
            local_action_log_probs,
            recurrent_hidden_states,
        ) = self.local_policy.act(
            local_policy_inputs, recurrent_hidden_states, None, masks
        )

        # Rotate in place for 10 time-steps to allow better map initialization for planner
        if ep_step < 10:
            local_action.fill_(self.left_action_id)

        # Check if the goal was reached
        reached_goal_flag = (
            torch.norm(relative_goal_at_t, dim=1) < self.goal_success_radius
        )
        if self._debug:
            print(f"Distance to goal: {torch.norm(relative_goal_at_t, dim=1)}")

        local_action[reached_goal_flag] = self.stop_action_id

        state_estimates = {
            "recurrent_hidden_states": recurrent_hidden_states,
            "map_states": map_states,
            "pose_estimates": pose_hat_at_t,
        }
        local_policy_outputs = {
            "values": local_value,
            "actions": local_action,
            "action_log_probs": local_action_log_probs,
        }

        return (
            mapper_inputs,
            local_policy_inputs,
            mapper_outputs,
            local_policy_outputs,
            state_estimates,
        )

    def _process_maps(self, maps, agent_pos, goals):
        """
        Inputs:
            maps - (bs, 2, M, M) --- 1st channel is prob of obstacle present
                                 --- 2nd channel is prob of being explored
        """
        obstacle_mask = (maps[:, 0] > self.config.thresh_obstacle) & (
            maps[:, 1] > self.config.thresh_explored
        )
        final_maps = asnumpy(obstacle_mask.float())
        kernel = np.ones((3, 3))
        for i in range(final_maps.shape[0]):
            final_maps[i] = cv2.dilate(final_maps[i], kernel, iterations=1)
            # Ensure that the goal point is not set to be occupied
            goal_x, goal_y = goals[i, 0], goals[i, 1]
            final_maps[i][
                (goal_y - 3) : (goal_y + 4), (goal_x - 3) : (goal_x + 4)
            ] = 0.0
            # Ensure that the agent's position is not set to be occupied
            agent_x, agent_y = agent_pos[i, 0], agent_pos[i, 1]
            final_maps[i][
                (agent_y - 2) : (agent_y + 3), (agent_x - 2) : (agent_x + 3)
            ] = 0.0

        return final_maps

    def to(self, device):
        self.mapper.to(device)
        self.local_policy.to(device)

    def train(self):
        self.mapper.train()
        self.local_policy.train()

    def eval(self):
        self.mapper.eval()
        self.local_policy.eval()

    def parameters(self):
        return chain(self.mapper.parameters(), self.local_policy.parameters())

    def state_dict(self):
        return {
            "mapper": self.mapper.state_dict(),
            "local_policy": self.local_policy.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.mapper.load_state_dict(state_dict["mapper"])
        self.local_policy.load_state_dict(state_dict["local_policy"])

    def reset(self):
        for k in self.states:
            self.states[k] = None

    def get_states(self):
        return copy.deepcopy(self.states)

    def update_states(self, states):
        self.states = states

    def _convert_polar2cartesian(self, coors):
        r = coors[:, 0]
        phi = -coors[:, 1]
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        return torch.stack([x, y], dim=1)

    def _compute_relative_local_goals(self, agent_world_pose, M, s):
        """
        Converts a local goal (x, y) position in the map to egocentric coordinates▫
        relative to agent's current pose.
        """

        local_map_goals = self.states["current_local_goals"]
        local_world_goals = convert_map2world(local_map_goals, (M, M), s)  # (bs, 2)
        # Concatenate dummy directions to goal
        local_world_goals = torch.cat(
            [
                local_world_goals,
                torch.zeros(self.nplanners, 1).to(agent_world_pose.device),
            ],
            dim=1,
        )
        relative_goals = subtract_pose(agent_world_pose, local_world_goals)[:, :2]

        return relative_goals

    def _sample_random_explored(self, agent_map, agent_pos, d_thresh=30):
        """
        Inputs:
            agent_map - (2, M, M) --- 1st channel is prob of obstacle present
                                  --- 2nd channel is prob of being explored

        Sampled random explored locations within a distance d_thresh from the agent_pos.
        """
        # Classify the unseen regions as free space as well.
        free_mask = (agent_map[0] <= self.config.thresh_obstacle) & (
            agent_map[1] > self.config.thresh_explored
        )
        free_mask = free_mask | (agent_map[1] <= self.config.thresh_explored)
        final_maps = asnumpy(free_mask.float())
        kernel = np.ones((3, 3))
        valid_locs = np.where(final_maps > 0)
        if valid_locs[0].shape[0] == 0:
            rand_x = agent_pos[0] + random.randint(-15, 15)
            rand_y = agent_pos[1] + random.randint(-15, 15)
        else:
            explored_locations = np.array(list(zip(*valid_locs)))  # (N, 2)
            explored_locations = np.flip(explored_locations, axis=1)
            # Pick some random explored location near agent's position
            dist2explored = np.linalg.norm(
                np.array(agent_pos)[np.newaxis, :] - explored_locations, axis=1
            )  # (N, )
            valid_explored_locations = explored_locations[dist2explored < d_thresh]
            if valid_explored_locations.shape[0] == 0:
                rand_x = agent_pos[0] + random.randint(-15, 15)
                rand_y = agent_pos[1] + random.randint(-15, 15)
            else:
                rand_x, rand_y = np.random.permutation(valid_explored_locations)[0]
                rand_x = float(rand_x.item())
                rand_y = float(rand_y.item())
        return (rand_x, rand_y)

    def _create_mapper_inputs(
        self, observations, prev_observations, prev_state_estimates
    ):
        rgb_at_t_1 = prev_observations["rgb"]  # (bs, H, W, C)
        depth_at_t_1 = prev_observations["depth"]  # (bs, H, W, 1)
        ego_map_gt_at_t_1 = prev_observations["ego_map_gt"]  # (bs, Hby2, Wby2, 2)
        pose_at_t_1 = prev_observations["pose"]  # (bs, 3)
        rgb_at_t = observations["rgb"]  # (bs, H, W, C)
        depth_at_t = observations["depth"]  # (bs, H, W, 1)
        ego_map_gt_at_t = observations["ego_map_gt"]  # (bs, Hby2, Wby2, 2)
        pose_at_t = observations["pose"]  # (bs, 3)
        # This happens only for a baseline
        if "ego_map_gt_anticipated" in prev_observations:
            ego_map_gt_anticipated_at_t_1 = prev_observations["ego_map_gt_anticipated"]
            ego_map_gt_anticipated_at_t = observations["ego_map_gt_anticipated"]
        else:
            ego_map_gt_anticipated_at_t_1 = None
            ego_map_gt_anticipated_at_t = None
        pose_hat_at_t_1 = prev_state_estimates["pose_estimates"]  # (bs, 3)
        map_at_t_1 = prev_state_estimates["map_states"]  # (bs, 2, M, M)

        mapper_inputs = {
            "rgb_at_t_1": rgb_at_t_1,
            "depth_at_t_1": depth_at_t_1,
            "ego_map_gt_at_t_1": ego_map_gt_at_t_1,
            "ego_map_gt_anticipated_at_t_1": ego_map_gt_anticipated_at_t_1,
            "pose_at_t_1": pose_at_t_1,
            "pose_hat_at_t_1": pose_hat_at_t_1,
            "map_at_t_1": map_at_t_1,
            "rgb_at_t": rgb_at_t,
            "depth_at_t": depth_at_t,
            "ego_map_gt_at_t": ego_map_gt_at_t,
            "ego_map_gt_anticipated_at_t": ego_map_gt_anticipated_at_t,
            "pose_at_t": pose_at_t,
        }

        return mapper_inputs

    def _compute_plans_and_local_goals(
        self, map_states, agent_map_xy, SAMPLE_LOCAL_GOAL_FLAGS
    ):
        """
        map_states - (bs, 2, V, V) tensor
        agent_map_xy - (bs, 2) agent's current position on the map
        """
        # Make a plan from agent position to the goal
        goal_positions_np = asnumpy(self.states["current_global_goals"]).astype(
            np.int32
        )  # (bs, 2)
        current_positions_np = asnumpy(agent_map_xy).astype(np.int32)  # (bs, 2)
        maps_np = self._process_maps(
            map_states, current_positions_np, goal_positions_np
        )
        plans = self.planner.plan(
            maps_np, current_positions_np, goal_positions_np, SAMPLE_LOCAL_GOAL_FLAGS
        )

        # Pick a local goal to reach with the local policy
        for i in range(self.nplanners):
            if SAMPLE_LOCAL_GOAL_FLAGS[i] != 1:
                continue

            path_x, path_y = plans[i]

            # If planning failed, sample random goal
            if path_x is None:
                goal_x, goal_y = self._sample_random_explored(
                    map_states[i], current_positions_np[i].tolist()
                )
            else:
                delta = min(self.planning_step, len(path_x) - 1)
                # The path is in reverse order
                goal_x, goal_y = path_x[-delta], path_y[-delta]

            self.states["current_local_goals"][i, 0] = goal_x
            self.states["current_local_goals"][i, 1] = goal_y
