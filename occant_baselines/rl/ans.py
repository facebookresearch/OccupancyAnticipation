#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC

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
from itertools import chain
from einops import rearrange, reduce, asnumpy


class ActiveNeuralSLAMBase(ABC):
    def __init__(self, config, projection_unit):
        self.config = config
        self.mapper = Mapper(config.MAPPER, projection_unit)
        if config.LOCAL_POLICY.use_heuristic_policy:
            self.local_policy = HeuristicLocalPolicy(config.LOCAL_POLICY)
        else:
            self.local_policy = LocalPolicy(config.LOCAL_POLICY)
        if config.PLANNER.nplanners > 1:
            self.planner = AStarPlannerVector(config.PLANNER)
        else:
            self.planner = AStarPlannerSequential(config.PLANNER)
        self.nplanners = self.config.PLANNER.nplanners
        self.planning_step_mts = self.config.planning_step
        self.goal_success_radius = self.config.goal_success_radius
        # Set random seed generators
        self._py_rng = random.Random()
        self._py_rng.seed(self.config.pyt_random_seed)
        self._npy_rng = np.random.RandomState()
        self._npy_rng.seed(self.config.pyt_random_seed)
        # Define states
        self._create_agent_states()

    def _create_agent_states(self):
        raise NotImplementedError

    def act(
        self,
        observations,
        prev_observations,
        prev_state_estimates,
        ep_time,
        masks,
        deterministic=False,
    ):
        raise NotImplementedError

    def _process_maps(self, maps, goals=None):
        """
        Inputs:
            maps - (bs, 2, M, M) --- 1st channel is prob of obstacle present
                                 --- 2nd channel is prob of being explored
        """
        map_scale = self.mapper.map_config["scale"]
        # Compute a map with ones for obstacles and zeros for the rest
        obstacle_mask = (maps[:, 0] > self.config.thresh_obstacle) & (
            maps[:, 1] > self.config.thresh_explored
        )
        final_maps = obstacle_mask.float()  # (bs, M, M)
        # Post-process map based on previously visited locations
        final_maps[self.states["visited_map"] == 1] = 0
        # Post-process map based on previously collided regions
        if self.states["collision_map"] is not None:
            final_maps[self.states["collision_map"] == 1] = 1
        # Set small regions around the goal location to be zeros
        if goals is not None:
            lfs = int(self.config.PLANNER.local_free_size / map_scale)
            for i in range(final_maps.shape[0]):
                goal_x = int(goals[i, 0].item())
                goal_y = int(goals[i, 1].item())
                final_maps[
                    i,
                    (goal_y - lfs) : (goal_y + lfs + 1),
                    (goal_x - lfs) : (goal_x + lfs + 1),
                ] = 0.0

        return final_maps

    def _sample_random_explored(
        self, agent_map_orig, agent_pos, map_scale, d_thresh=1.5
    ):
        """
        Inputs:
            agent_map - (2, M, M) --- 1st channel is prob of obstacle present
                                  --- 2nd channel is prob of being explored

        Sampled random explored locations within a distance d_thresh meters from the agent_pos.
        """
        # Crop a small region around the agent position
        range_xy = int(d_thresh / map_scale)
        H, W = agent_map_orig.shape[1:]
        start_x = np.clip(int(agent_pos[0] - range_xy), 0, W - 1).item()
        end_x = np.clip(int(agent_pos[0] + range_xy), 0, W - 1).item()
        start_y = np.clip(int(agent_pos[1] - range_xy), 0, H - 1).item()
        end_y = np.clip(int(agent_pos[1] + range_xy), 0, H - 1).item()
        agent_map = agent_map_orig[:, start_y : (end_y + 1), start_x : (end_x + 1)]
        if agent_map.shape[1] == 0 or agent_map.shape[2] == 0:
            return agent_pos
        free_mask = (agent_map[0] <= self.config.thresh_obstacle) & (
            agent_map[1] > self.config.thresh_explored
        )
        valid_locs = torch.nonzero(free_mask)  # (N, 2)
        if valid_locs.shape[0] == 0:
            rand_x = agent_pos[0] + self._py_rng.randint(
                -(range_xy // 2), range_xy // 2
            )
            rand_y = agent_pos[1] + self._py_rng.randint(
                -(range_xy // 2), range_xy // 2
            )
        else:
            rand_idx = self._npy_rng.randint(0, valid_locs.shape[0])
            rand_y, rand_x = valid_locs[rand_idx]
            rand_x = float(rand_x.item()) + start_x
            rand_y = float(rand_y.item()) + start_y
        return (rand_x, rand_y)

    def _sample_random_near_agent(
        self, agent_map_orig, agent_pos, map_scale, d_thresh=1.5
    ):
        """
        Inputs:
            agent_map - (2, M, M) --- 1st channel is prob of obstacle present
                                  --- 2nd channel is prob of being explored

        Sampled random location within a distance d_thresh meters from the agent_pos.
        """
        # Crop a small 3m x 3m region around the agent position
        range_xy = int(d_thresh / map_scale)
        rand_x = agent_pos[0] + self._py_rng.randint(-range_xy, range_xy)
        rand_y = agent_pos[1] + self._py_rng.randint(-range_xy, range_xy)
        return (rand_x, rand_y)

    def _sample_random_towards_goal(
        self, agent_map_orig, agent_pos, goal_pos, map_scale, d_thresh=1.5
    ):
        """
        Inputs:
            agent_map - (2, M, M) --- 1st channel is prob of obstacle present
                                  --- 2nd channel is prob of being explored

        Sampled random location within a distance d_thresh meters from the agent_pos.
        """
        # Crop a small 3m x 3m region around the agent position
        range_xy = int(d_thresh / map_scale)
        # Bias sampling towards the goal
        goal_rel_x = goal_pos[0] - agent_pos[0]
        goal_rel_y = goal_pos[1] - agent_pos[1]
        if goal_rel_x >= 0:
            start_x = 0
            end_x = range_xy
        else:
            start_x = -range_xy
            end_x = -1
        if goal_rel_y >= 0:
            start_y = 0
            end_y = range_xy
        else:
            start_y = -range_xy
            end_y = -1
        rand_x = agent_pos[0] + self._py_rng.randint(start_x, end_x + 1)
        rand_y = agent_pos[1] + self._py_rng.randint(start_y, end_y + 1)
        return (rand_x, rand_y)

    def _sample_random_explored_towards_goal(
        self, agent_map_orig, agent_pos, goal_pos, map_scale, d_thresh=0.5
    ):
        """
        Inputs:
            agent_map - (2, M, M) --- 1st channel is prob of obstacle present
                                  --- 2nd channel is prob of being explored

        Sampled random explored locations within a distance d_thresh meters from the agent_pos.
        """
        # Crop a small 3m x 3m region around the agent position
        range_xy = int(d_thresh / map_scale)
        H, W = agent_map_orig.shape[1:]
        start_x = np.clip(int(agent_pos[0] - range_xy), 0, W - 1).item()
        end_x = np.clip(int(agent_pos[0] + range_xy), 0, W - 1).item()
        start_y = np.clip(int(agent_pos[1] - range_xy), 0, H - 1).item()
        end_y = np.clip(int(agent_pos[1] + range_xy), 0, H - 1).item()
        agent_map = agent_map_orig[:, start_y : (end_y + 1), start_x : (end_x + 1)]
        if agent_map.shape[1] == 0 or agent_map.shape[2] == 0:
            return agent_pos
        free_mask = (agent_map[0] <= self.config.thresh_obstacle) & (
            agent_map[1] > self.config.thresh_explored
        )
        valid_locs = torch.nonzero(free_mask)  # (N, 2)
        if valid_locs.shape[0] == 0:
            rand_x = agent_pos[0] + self._py_rng.randint(-range_xy, range_xy)
            rand_y = agent_pos[1] + self._py_rng.randint(-range_xy, range_xy)
        else:
            goal_x_trans = goal_pos[0] - start_x
            goal_y_trans = goal_pos[1] - start_y
            dist2goal = (valid_locs[:, 1] - goal_x_trans) ** 2 + (
                valid_locs[:, 0] - goal_y_trans
            ) ** 2
            # Sort based on distance to goal
            _, sort_idxes = torch.sort(dist2goal)
            # Sample from top-10 explored points closest to the goal
            rand_idx = self._npy_rng.randint(0, min(10, dist2goal.shape[0]))
            rand_y, rand_x = valid_locs[sort_idxes[rand_idx].item()]
            rand_x = float(rand_x.item()) + start_x
            rand_y = float(rand_y.item()) + start_y
        return (rand_x, rand_y)

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
        return chain(self.mapper.parameters(), self.local_policy.parameters(),)

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

    def _compute_relative_local_goals(self, agent_world_pose, M, s):
        """
        Converts a local goal (x, y) position in the map to egocentric coordinates
        relative to agent's current pose.
        """
        local_map_goals = self.states["curr_local_goals"]
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
        action_at_t_1 = observations["prev_actions"]
        # This happens only for a baseline
        if "ego_map_gt_anticipated" in prev_observations:
            ego_map_gt_anticipated_at_t_1 = prev_observations["ego_map_gt_anticipated"]
            ego_map_gt_anticipated_at_t = observations["ego_map_gt_anticipated"]
        else:
            ego_map_gt_anticipated_at_t_1 = None
            ego_map_gt_anticipated_at_t = None
        pose_hat_at_t_1 = prev_state_estimates["pose_estimates"]  # (bs, 3)
        map_at_t_1 = prev_state_estimates["map_states"]  # (bs, 2, M, M)
        pose_gt_at_t_1 = prev_observations.get("pose_gt", None)
        pose_gt_at_t = observations.get("pose_gt", None)

        mapper_inputs = {
            "rgb_at_t_1": rgb_at_t_1,
            "depth_at_t_1": depth_at_t_1,
            "ego_map_gt_at_t_1": ego_map_gt_at_t_1,
            "ego_map_gt_anticipated_at_t_1": ego_map_gt_anticipated_at_t_1,
            "pose_at_t_1": pose_at_t_1,
            "pose_gt_at_t_1": pose_gt_at_t_1,
            "pose_hat_at_t_1": pose_hat_at_t_1,
            "map_at_t_1": map_at_t_1,
            "rgb_at_t": rgb_at_t,
            "depth_at_t": depth_at_t,
            "ego_map_gt_at_t": ego_map_gt_at_t,
            "ego_map_gt_anticipated_at_t": ego_map_gt_anticipated_at_t,
            "pose_at_t": pose_at_t,
            "pose_gt_at_t": pose_gt_at_t,
            "action_at_t_1": action_at_t_1,
        }
        return mapper_inputs

    def _compute_plans(
        self,
        global_map,
        agent_map_xy,
        goal_map_xy,
        sample_goal_flags,
        cache_map=False,
        crop_map_flag=True,
    ):
        """
        global_map - (bs, 2, V, V) tensor
        agent_map_xy - (bs, 2) agent's current position on the map
        goal_map_xy - (bs, 2) goal's current position on the map
        sample_goal_flags - list of zeros and ones should a new goal be sampled?
        """
        # ==================== Process the map to get planner map =====================
        s = self.mapper.map_config["scale"]
        # Processed map has zeros for free-space and ones for obstacles
        global_map_proc = self._process_maps(
            global_map, goal_map_xy
        )  # (bs, M, M) tensor
        # =================== Crop a local region around agent, goal ==================
        if crop_map_flag:
            # Determine crop size
            abs_diff_xy = torch.abs(agent_map_xy - goal_map_xy)
            S = int(torch.max(abs_diff_xy).item())
            # Add a small buffer space around the agent location
            buffer_size = int(3.0 / s)  # 3 meters buffer space in total
            S += buffer_size
            old_center_xy = (agent_map_xy + goal_map_xy) / 2
            # Crop a SxS region centered around old_center_xy
            # Note: The out-of-bound regions will be zero-padded by default. In this case,
            # since zeros correspond to free-space, that is not a problem.
            cropped_global_map = crop_map(
                global_map_proc.unsqueeze(1), old_center_xy, S
            ).squeeze(
                1
            )  # (bs, S, S)
            # Add zero padding to ensure the plans don't fail due to cropping
            pad_size = 5
            cropped_global_map = F.pad(
                cropped_global_map, (pad_size, pad_size, pad_size, pad_size)
            )
            S += pad_size * 2
            # Transform to new coordinates
            new_center_xy = torch.ones_like(old_center_xy) * (S / 2)
            new_agent_map_xy = agent_map_xy + (new_center_xy - old_center_xy)
            new_goal_map_xy = goal_map_xy + (new_center_xy - old_center_xy)
        else:
            cropped_global_map = global_map_proc  # (bs, M, M)
            old_center_xy = (agent_map_xy + goal_map_xy) / 2
            new_center_xy = old_center_xy
            new_agent_map_xy = agent_map_xy
            new_goal_map_xy = goal_map_xy
            S = cropped_global_map.shape[1]

        if cache_map:
            self._cropped_global_map = cropped_global_map
        # Clip points to ensure they are within map limits
        new_agent_map_xy = torch.clamp(new_agent_map_xy, 0, S - 1)
        new_goal_map_xy = torch.clamp(new_goal_map_xy, 0, S - 1)
        # Convert to numpy
        agent_map_xy_np = asnumpy(new_agent_map_xy).astype(np.int32)
        goal_map_xy_np = asnumpy(new_goal_map_xy).astype(np.int32)
        global_map_np = asnumpy(cropped_global_map)
        # =================== Plan path from agent to goal positions ==================
        plans = self.planner.plan(
            global_map_np, agent_map_xy_np, goal_map_xy_np, sample_goal_flags
        )  # List of tuple of lists
        # Convert plans back to original coordinates
        final_plans = []
        for i in range(len(plans)):
            plan_x, plan_y = plans[i]
            # Planning failure
            if plan_x is None:
                final_plans.append((plan_x, plan_y))
                continue
            offset_x = int((old_center_xy[i, 0] - new_center_xy[i, 0]).item())
            offset_y = int((old_center_xy[i, 1] - new_center_xy[i, 1]).item())
            final_plan_x, final_plan_y = [], []
            for px, py in zip(plan_x, plan_y):
                final_plan_x.append(px + offset_x)
                final_plan_y.append(py + offset_y)
            final_plans.append((final_plan_x, final_plan_y))
        return final_plans

    def _compute_plans_and_local_goals(
        self, global_map, agent_map_xy, SAMPLE_LOCAL_GOAL_FLAGS,
    ):
        raise NotImplementedError

    def _compute_dist2localgoal(self, global_map, map_xy, local_goal_xy):
        """
        global_map - (bs, 2, V, V) tensor
        map_xy - (bs, 2) agent's current position on the map
        local_goal_xy - (bs, 2) local goal position on the map
        """
        sample_goal_flags = [1.0 for _ in range(self.nplanners)]
        plans = self._compute_plans(
            global_map, map_xy, local_goal_xy, sample_goal_flags
        )
        dist2localgoal = []
        # Compute distance to local goal
        for i in range(self.nplanners):
            path_x, path_y = plans[i]
            # If planning failed, return euclidean distance
            if path_x is None:
                d2l = torch.norm(map_xy[i] - local_goal_xy[i])
            else:
                path_xy = np.array([path_x, path_y]).T  # (n, 2)
                d2l = np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1).sum().item()
            dist2localgoal.append(d2l * self.mapper.map_config["scale"])
        dist2localgoal = torch.Tensor(dist2localgoal).to(global_map.device)
        return dist2localgoal

    def _compute_local_map_crop(self, global_map, global_pose):
        local_crop_size = self.config.LOCAL_POLICY.embed_map_size
        exp_crop_size = int(1.5 * local_crop_size)
        cropped_map = crop_map(
            global_map, self.states["curr_map_position"], exp_crop_size
        )
        global_heading = global_pose[:, 2]  # (bs, ) pose in radians
        rotation_params = torch.stack(
            [
                torch.zeros_like(global_heading),
                torch.zeros_like(global_heading),
                global_heading,
            ],
            dim=1,
        )
        rotated_map = spatial_transform_map(cropped_map, rotation_params)
        center_locs = torch.zeros_like(self.states["curr_map_position"])
        center_locs[:, 0] = rotated_map.shape[3] // 2
        center_locs[:, 1] = rotated_map.shape[2] // 2
        rotated_map = crop_map(rotated_map, center_locs, local_crop_size)
        return rotated_map


class ActiveNeuralSLAMExplorer(ActiveNeuralSLAMBase):
    def __init__(self, config, projection_unit):
        super().__init__(config, projection_unit)
        self.global_policy = GlobalPolicy(config.GLOBAL_POLICY)
        self.goal_interval = self.config.goal_interval

    def _create_agent_states(self):
        self.states = {
            # Planning states
            "curr_global_goals": None,
            "curr_local_goals": None,
            "prev_dist2localgoal": None,
            "curr_dist2localgoal": None,
            "prev_map_position": None,
            "curr_map_position": None,
            "local_path_length": None,
            "local_shortest_path_length": None,
            # Heuristics for navigation
            "collision_map": None,
            "visited_map": None,
            "col_width": None,
            "sample_random_explored_timer": None,
            # Global map reward states
            "prev_map_states": None,
        }

    def act(
        self,
        observations,
        prev_observations,
        prev_state_estimates,
        ep_time,
        masks,
        deterministic=False,
    ):
        # ============================ Set useful variables ===========================
        ep_step = ep_time[0].item()
        M = prev_state_estimates["map_states"].shape[2]
        s = self.mapper.map_config["scale"]
        device = observations["rgb"].device
        assert M % 2 == 1, "The code is tested only for odd map sizes!"
        # =================== Update states from current observation ==================
        # Update map, pose and visitation map
        mapper_inputs = self._create_mapper_inputs(
            observations, prev_observations, prev_state_estimates
        )
        mapper_outputs = self.mapper(mapper_inputs)
        global_map = mapper_outputs["mt"]
        global_pose = mapper_outputs["xt_hat"]
        map_xy = convert_world2map(global_pose[:, :2], (M, M), s)
        map_xy = torch.clamp(map_xy, 0, M - 1)
        visited_states = self._update_state_visitation(
            prev_state_estimates["visited_states"], map_xy
        )  # (bs, 1, M, M)
        # Update local ANM state variables
        curr_map_position = map_xy
        if ep_step > 0:
            # Compute state updates
            prev_dist2localgoal = self.states["curr_dist2localgoal"]
            curr_dist2localgoal = self._compute_dist2localgoal(
                global_map, map_xy, self.states["curr_local_goals"],
            )
            prev_map_position = self.states["curr_map_position"]
            prev_step_size = (
                torch.norm(curr_map_position - prev_map_position, dim=1).unsqueeze(1)
                * s
            )
            # Update the state variables
            self.states["prev_dist2localgoal"] = prev_dist2localgoal
            self.states["curr_dist2localgoal"] = curr_dist2localgoal
            self.states["prev_map_position"] = prev_map_position
            self.states["local_path_length"] += prev_step_size
        self.states["curr_map_position"] = curr_map_position
        # Initialize collision and visited maps at t=0
        if ep_step == 0:
            self.states["collision_map"] = torch.zeros(self.nplanners, M, M).to(device)
            self.states["visited_map"] = torch.zeros(self.nplanners, M, M).to(device)
            self.states["col_width"] = torch.ones(self.nplanners)
            # Monitors number of steps elapsed since last call to sample random explored
            self.states["sample_random_explored_timer"] = torch.zeros(self.nplanners)
        if ep_step > 0:
            # Update collision maps
            forward_step = self.config.LOCAL_POLICY.AGENT_DYNAMICS.forward_step
            for i in range(self.nplanners):
                prev_action_i = observations["prev_actions"][i, 0].item()
                # If not forward action, skip
                if prev_action_i != 0:
                    continue
                x1, y1 = asnumpy(self.states["prev_map_position"][i]).tolist()
                x2, y2 = asnumpy(self.states["curr_map_position"][i]).tolist()
                t2 = global_pose[i, 2].item() - math.pi / 2
                if abs(x1 - x2) < 1 and abs(y1 - y2) < 1:
                    self.states["col_width"][i] += 2
                    self.states["col_width"][i] = min(self.states["col_width"][i], 9)
                else:
                    self.states["col_width"][i] = 1
                dist_trav_i = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * s
                # Add an obstacle infront of the agent if a collision happens
                if dist_trav_i < 0.7 * forward_step:  # Collision
                    length = 2
                    width = int(self.states["col_width"][i].item())
                    buf = 3
                    cmH, cmW = self.states["collision_map"][i].shape
                    for j in range(length):
                        for k in range(width):
                            wx = (
                                x2
                                + ((j + buf) * math.cos(t2))
                                + ((k - width / 2) * math.sin(t2))
                            )
                            wy = (
                                y2
                                + ((j + buf) * math.sin(t2))
                                - ((k - width / 2) * math.cos(t2))
                            )
                            wx, wy = int(wx), int(wy)
                            if wx < 0 or wx >= cmW or wy < 0 or wy >= cmH:
                                continue
                            self.states["collision_map"][i, wy, wx] = 1
        # Update visitation maps
        for i in range(self.nplanners):
            mx, my = asnumpy(self.states["curr_map_position"][i]).tolist()
            mx, my = int(mx), int(my)
            self.states["visited_map"][i, my - 2 : my + 3, mx - 2 : mx + 3] = 1
        # ===================== Compute rewards for previous action ===================
        local_rewards = self._compute_local_rewards(ep_step, s).to(device)
        # ====================== Global policy action selection =======================
        SAMPLE_GLOBAL_GOAL_FLAG = ep_step % self.goal_interval == 0
        # Sample global goal if needed
        if SAMPLE_GLOBAL_GOAL_FLAG:
            global_policy_inputs = self._create_global_policy_inputs(
                global_map, visited_states, self.states["curr_map_position"]
            )
            (
                global_value,
                global_action,
                global_action_log_probs,
                _,
            ) = self.global_policy.act(global_policy_inputs, None, None, None)
            # Convert action to location (row-major format)
            G = self.global_policy.G
            global_action_map_x = torch.fmod(
                global_action.squeeze(1), G
            ).float()  # (bs, )
            global_action_map_y = (global_action.squeeze(1) / G).float()  # (bs, )
            # Convert to MxM map coordinates
            global_action_map_x = global_action_map_x * M / G
            global_action_map_y = global_action_map_y * M / G
            global_action_map_xy = torch.stack(
                [global_action_map_x, global_action_map_y], dim=1
            )
            # Set the goal (bs, 2) --- (x, y) in map image coordinates
            self.states["curr_global_goals"] = global_action_map_xy
            # Update the current map to prev_map_states in order to facilitate
            # future reward computation.
            self.states["prev_map_states"] = global_map.detach()
        else:
            global_policy_inputs = None
            global_value = None
            global_action = None
            global_action_log_probs = None
        # ======================= Local policy action selection =======================
        # Initialize states at t=0
        if ep_step == 0:
            self.states["curr_local_goals"] = torch.zeros(self.nplanners, 2).to(device)
            self.states["local_shortest_path_length"] = torch.zeros(
                self.nplanners, 1
            ).to(device)
            self.states["local_path_length"] = torch.zeros(self.nplanners, 1).to(device)
        # Should local goals be sampled now?
        if SAMPLE_GLOBAL_GOAL_FLAG:
            # Condition 1: A new global goal was sampled
            SAMPLE_LOCAL_GOAL_FLAGS = [1 for _ in range(self.nplanners)]
        else:
            # Condition 2 (a): The previous local goal was reached
            prev_goal_reached = (
                self.states["curr_dist2localgoal"] < self.goal_success_radius
            )
            # Condition 2 (b): The previous local goal is occupied.
            goals = self.states["curr_local_goals"].long().to(device)
            prev_gcells = global_map[
                torch.arange(0, goals.shape[0]).long(), :, goals[:, 1], goals[:, 0]
            ]
            prev_goal_occupied = (prev_gcells[:, 0] > self.config.thresh_obstacle) & (
                prev_gcells[:, 1] > self.config.thresh_explored
            )
            SAMPLE_LOCAL_GOAL_FLAGS = asnumpy(
                (prev_goal_reached | prev_goal_occupied).float()
            ).tolist()
        # Execute planner and compute local goals
        self._compute_plans_and_local_goals(
            global_map, self.states["curr_map_position"], SAMPLE_LOCAL_GOAL_FLAGS
        )
        # Update state variables to account for new local goals
        self.states["curr_dist2localgoal"] = self._compute_dist2localgoal(
            global_map,
            self.states["curr_map_position"],
            self.states["curr_local_goals"],
        )
        # Sample action with local policy
        local_masks = 1 - torch.Tensor(SAMPLE_LOCAL_GOAL_FLAGS).to(device).unsqueeze(1)
        recurrent_hidden_states = prev_state_estimates["recurrent_hidden_states"]
        relative_goals = self._compute_relative_local_goals(global_pose, M, s)
        local_policy_inputs = {
            "rgb_at_t": observations["rgb"],
            "goal_at_t": relative_goals,
            "t": ep_time,
        }
        outputs = self.local_policy.act(
            local_policy_inputs,
            recurrent_hidden_states,
            None,
            local_masks,
            deterministic=deterministic,
        )
        (
            local_value,
            local_action,
            local_action_log_probs,
            recurrent_hidden_states,
        ) = outputs
        # If imitation learning is used, also sample the ground-truth action to take
        if "gt_global_map" in observations.keys():
            gt_global_map = observations["gt_global_map"]  # (bs, 2, M, M)
            gt_global_pose = observations["pose_gt"]  # (bs, 3)
            relative_goals_aug = torch.cat(
                [relative_goals, torch.zeros_like(relative_goals[:, 0:1])], dim=1
            )
            gt_goals = add_pose(gt_global_pose, relative_goals_aug)  # (bs, 3)
            gt_actions = self._compute_gt_local_action(
                gt_global_map, gt_global_pose, gt_goals, M, s
            )  # (bs, 1)
            gt_actions = gt_actions.to(device)
        else:
            gt_actions = torch.zeros_like(local_action)

        # ============================== Create output dicts ==========================
        state_estimates = {
            "recurrent_hidden_states": recurrent_hidden_states,
            "map_states": mapper_outputs["mt"],
            "visited_states": visited_states,
            "pose_estimates": global_pose,
        }
        local_policy_outputs = {
            "values": local_value,
            "actions": local_action,
            "action_log_probs": local_action_log_probs,
            "local_masks": local_masks,
            "gt_actions": gt_actions,
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

    def _compute_gt_local_action(
        self, global_map, agent_world_xyt, goal_world_xyt, M, s
    ):
        """
        Estimate the shortest-path action from agent position to goal position.
        """
        agent_map_xy = convert_world2map(agent_world_xyt, (M, M), s)
        goal_map_xy = convert_world2map(goal_world_xyt, (M, M), s)
        # ============ Crop a region covering agent_map_xy and goal_map_xy ============
        abs_delta_xy = torch.abs(agent_map_xy - goal_map_xy)
        S = max(int(torch.max(abs_delta_xy).item()), 80)
        old_center_xy = (agent_map_xy + goal_map_xy) // 2  # (bs, 2)
        cropped_global_map = crop_map(global_map, old_center_xy, int(S))
        global_map_np = self._process_maps(cropped_global_map)  # (bs, M, M)
        new_center_xy = torch.zeros_like(old_center_xy)
        new_center_xy[:, 0] = global_map_np.shape[2] // 2
        new_center_xy[:, 1] = global_map_np.shape[1] // 2
        # ================ Transform points to new coordinate system ==================
        new_agent_map_xy = agent_map_xy + new_center_xy - old_center_xy
        new_goal_map_xy = goal_map_xy + new_center_xy - old_center_xy
        map_xy_np = asnumpy(new_agent_map_xy).astype(np.int32)  # (bs, 2)
        goal_xy_np = asnumpy(new_goal_map_xy).astype(np.int32)
        # Ensure that points do not go outside map limits
        map_W = global_map_np.shape[2]
        map_H = global_map_np.shape[1]
        map_xy_np[:, 0] = np.clip(map_xy_np[:, 0], 0, map_W - 1)
        map_xy_np[:, 1] = np.clip(map_xy_np[:, 1], 0, map_H - 1)
        goal_xy_np[:, 0] = np.clip(goal_xy_np[:, 0], 0, map_W - 1)
        goal_xy_np[:, 1] = np.clip(goal_xy_np[:, 1], 0, map_H - 1)
        sample_flag = [1.0 for _ in range(self.nplanners)]
        # Compute plan
        plans_xy = self.planner.plan(global_map_np, map_xy_np, goal_xy_np, sample_flag)
        # ===================== Sample an action to a nearby point ====================
        # 0 is forward, 1 is left, 2 is right
        gt_actions = []
        forward_step = self.config.LOCAL_POLICY.AGENT_DYNAMICS.forward_step
        turn_angle = math.radians(self.config.LOCAL_POLICY.AGENT_DYNAMICS.turn_angle)
        for i in range(self.nplanners):
            path_x, path_y = plans_xy[i]
            # If planning failed, sample random action
            if path_x is None:
                gt_actions.append(random.randint(0, 2))
                continue
            # Plan to navigate to a point 0.5 meters away
            dl = min(int(0.5 / s), len(path_x) - 1)
            # The path is in reverse order
            goal_x, goal_y = path_x[-dl], path_y[-dl]
            agent_x, agent_y = map_xy_np[i].tolist()
            # Decide action
            agent_heading = agent_world_xyt[i, 2].item() - math.pi / 2
            reqd_heading = math.atan2(goal_y - agent_y, goal_x - agent_x)
            diff_angle = reqd_heading - agent_heading
            diff_angle = math.atan2(math.sin(diff_angle), math.cos(diff_angle))
            # Move forward if facing the correct direction
            if abs(diff_angle) < 1.5 * turn_angle:
                gt_actions.append(0)
            # Turn left if the goal is to the left
            elif diff_angle < 0:
                gt_actions.append(1)
            # Turn right otherwise
            else:
                gt_actions.append(2)

        return torch.Tensor(gt_actions).unsqueeze(1).long()

    def _compute_local_rewards(
        self, ep_step, s,
    ):
        local_rewards = torch.zeros(self.nplanners, 1)
        if ep_step == 0:
            return local_rewards
        # Reward reduction in geodesic distance to the target
        p_d2g = self.states["prev_dist2localgoal"]
        c_d2g = self.states["curr_dist2localgoal"]
        local_rewards += (p_d2g - c_d2g).unsqueeze(1).cpu()
        # Add slack reward
        local_rewards += self.config.local_slack_reward
        # If local goal is reached, then provide an SPL reward
        success_flag = (c_d2g < self.goal_success_radius).float().unsqueeze(1)
        L = self.states["local_shortest_path_length"]
        P = self.states["local_path_length"]
        spl = success_flag * L / (torch.max(L, P) + 1e-8)
        local_rewards += spl.cpu()
        # Store variables for debugging
        self._spl_reward = spl
        return local_rewards

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
        self, global_map, agent_map_xy, SAMPLE_LOCAL_GOAL_FLAGS,
    ):
        """
        global_map - (bs, 2, V, V) tensor
        agent_map_xy - (bs, 2) agent's current position on the map
        """
        s = self.mapper.map_config["scale"]
        goal_map_xy = self.states["curr_global_goals"]
        plans_xy = self._compute_plans(
            global_map,
            agent_map_xy,
            goal_map_xy,
            SAMPLE_LOCAL_GOAL_FLAGS,
            cache_map=True,
            crop_map_flag=self.config.crop_map_for_planning,
        )
        # ========================= Sample a local goal =======================
        # Sample a local goal and measure the shortest path distance to that
        # goal according to the current map.
        # Update step counts for sample_random_explored calls
        self.states["sample_random_explored_timer"] += 1
        # Pick a local goal to reach with the local policy
        for i in range(self.nplanners):
            if SAMPLE_LOCAL_GOAL_FLAGS[i] != 1:
                continue
            path_x, path_y = plans_xy[i]
            # If planning failed, sample random local goal.
            if path_x is None:
                # Note: This is an expensive call, especially when the map is messy
                # and planning keeps failing. Call sample_random_explored only after
                # so many steps elapsed since the last call.
                if self.config.recovery_heuristic == "random_explored_towards_goal":
                    if self.states["sample_random_explored_timer"][i].item() > 10:
                        goal_x, goal_y = self._sample_random_explored_towards_goal(
                            global_map[i],
                            asnumpy(agent_map_xy[i]).tolist(),
                            asnumpy(goal_map_xy[i]).tolist(),
                            s,
                        )
                        # Reset count
                        self.states["sample_random_explored_timer"][i] = 0
                    else:
                        goal_x, goal_y = self._sample_random_towards_goal(
                            global_map[i],
                            asnumpy(agent_map_xy[i]).tolist(),
                            asnumpy(goal_map_xy[i]).tolist(),
                            s,
                        )
                        goal_x, goal_y = asnumpy(agent_map_xy[i]).tolist()
                elif self.config.recovery_heuristic == "random_explored":
                    if self.states["sample_random_explored_timer"][i].item() > 10:
                        goal_x, goal_y = self._sample_random_explored(
                            global_map[i], asnumpy(agent_map_xy[i]).tolist(), s
                        )
                        # Reset count
                        self.states["sample_random_explored_timer"][i] = 0
                    else:
                        goal_x, goal_y = self._sample_random_near_agent(
                            global_map[i], asnumpy(agent_map_xy[i]).tolist(), s
                        )
                        goal_x, goal_y = asnumpy(agent_map_xy[i]).tolist()
                else:
                    raise ValueError
                # When planning fails, default to euclidean distance
                curr_x, curr_y = agent_map_xy[i].tolist()
                splength = (
                    math.sqrt((goal_x - curr_x) ** 2 + (goal_y - curr_y) ** 2) * s
                )
            else:
                dl = min(int(self.planning_step_mts / s), len(path_x) - 1)
                # The path is in reverse order
                goal_x, goal_y = path_x[-dl], path_y[-dl]
                sp_xy = np.array([path_x[-dl:], path_y[-dl:]]).T  # (dl, 2)
                splength = (
                    np.linalg.norm(sp_xy[:-1] - sp_xy[1:], axis=1).sum().item() * s
                )
            # Ensure goals are within map bounds
            goal_x = np.clip(goal_x, 0, global_map[i].shape[-1] - 1).item()
            goal_y = np.clip(goal_y, 0, global_map[i].shape[-2] - 1).item()
            # Set the local goals as well as the corresponding path length
            # measures
            self.states["curr_local_goals"][i, 0] = goal_x
            self.states["curr_local_goals"][i, 1] = goal_y
            self.states["local_shortest_path_length"][i] = splength
            self.states["local_path_length"][i] = 0.0

    def _create_global_policy_inputs(self, global_map, visited_states, map_xy):
        """
        global_map     - (bs, 2, V, V) - map occupancy, explored states
        visited_states - (bs, 1, V, V) - agent visitation status on the map
        map_xy   - (bs, 2) - agent's XY position on the map
        """
        agent_map_x = map_xy[:, 0].long()  # (bs, )
        agent_map_y = map_xy[:, 1].long()  # (bs, )
        agent_position_onehot = torch.zeros_like(visited_states)
        agent_position_onehot[:, 0, agent_map_y, agent_map_x] = 1
        h_t = torch.cat(
            [global_map, visited_states, agent_position_onehot], dim=1
        )  # (bs, 4, M, M)

        global_policy_inputs = {
            "pose_in_map_at_t": map_xy,
            "map_at_t": h_t,
        }

        return global_policy_inputs


class ActiveNeuralSLAMNavigator(ActiveNeuralSLAMBase):
    def __init__(self, config, projection_unit):
        super().__init__(config, projection_unit)
        self.stop_action_id = self.config.stop_action_id
        self.left_action_id = self.config.left_action_id

    def _create_agent_states(self):
        self.states = {
            # Planning states
            "curr_global_goals": None,
            "curr_ego_world_goals": None,
            "curr_local_goals": None,
            "curr_dist2localgoal": None,
            "curr_map_position": None,
            "prev_map_position": None,
            # Heuristics for navigation
            "collision_map": None,
            "visited_map": None,
            "col_width": None,
            "sample_random_explored_timer": None,
        }

    def act(
        self,
        observations,
        prev_observations,
        prev_state_estimates,
        ep_time,
        masks,
        deterministic=False,
    ):
        # ============================ Set useful variables ===========================
        ep_step = ep_time[0].item()
        M = prev_state_estimates["map_states"].shape[2]
        s = self.mapper.map_config["scale"]
        device = observations["rgb"].device
        assert M % 2 == 1, "The code is tested only for odd map sizes!"
        # =================== Update states from current observation ==================
        # Update map and pose
        mapper_inputs = self._create_mapper_inputs(
            observations, prev_observations, prev_state_estimates
        )
        mapper_outputs = self.mapper(mapper_inputs)
        global_map = mapper_outputs["mt"]
        global_pose = mapper_outputs["xt_hat"]
        map_xy = convert_world2map(global_pose[:, :2], (M, M), s)
        map_xy = torch.clamp(map_xy, 0, M - 1)
        # Update local ANM state variables
        curr_map_position = map_xy
        if ep_step > 0:
            self.states["prev_map_position"] = self.states["curr_map_position"]
        self.states["curr_map_position"] = curr_map_position
        # Update goal location
        # Convention for pointgoal: x is forward, y is rightward in the
        # start-pose coordinates.
        global_goal_polar = observations["pointgoal"]  # (bs, 2) --- (rho, phi)
        global_goal = self._convert_polar2cartesian(global_goal_polar)
        map_goal_xy = convert_world2map(global_goal, (M, M), s)
        self.states["curr_global_goals"] = map_goal_xy
        # Compute relative location of goal in the world relative to agent
        global_goal_aug = torch.cat(
            [global_goal, torch.zeros(self.nplanners, 1).to(device)], dim=1
        )
        curr_ego_world_goals = subtract_pose(global_pose, global_goal_aug)[
            :, :2
        ]  # (bs, 2)
        self.states["curr_ego_world_goals"] = curr_ego_world_goals
        # Initialize collision and visited maps at t=0
        if ep_step == 0:
            self.states["collision_map"] = torch.zeros(self.nplanners, M, M).to(device)
            self.states["visited_map"] = torch.zeros(self.nplanners, M, M).to(device)
            self.states["col_width"] = torch.ones(self.nplanners)
            # Monitors number of steps elapsed since last call to sample random explored
            self.states["sample_random_explored_timer"] = torch.zeros(self.nplanners)
        if ep_step > 0:
            # Compute state updates
            curr_dist2localgoal = self._compute_dist2localgoal(
                global_map, map_xy, self.states["curr_local_goals"],
            )
            # Update the state variables
            self.states["curr_dist2localgoal"] = curr_dist2localgoal
            # Update collision maps
            forward_step = self.config.LOCAL_POLICY.AGENT_DYNAMICS.forward_step
            for i in range(self.nplanners):
                prev_action_i = observations["prev_actions"][i, 0].item()
                # If not forward action, skip
                if prev_action_i != 0:
                    continue
                x1, y1 = asnumpy(self.states["prev_map_position"][i]).tolist()
                x2, y2 = asnumpy(self.states["curr_map_position"][i]).tolist()
                t2 = global_pose[i, 2].item() - math.pi / 2
                if abs(x1 - x2) < 1 and abs(y1 - y2) < 1:
                    self.states["col_width"][i] += 2
                    self.states["col_width"][i] = min(self.states["col_width"][i], 9)
                else:
                    self.states["col_width"][i] = 1
                dist_trav_i = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * s
                # Add an obstacle infront of the agent if a collision happens
                if dist_trav_i < 0.7 * forward_step:  # Collision
                    length = 2
                    width = int(self.states["col_width"][i].item())
                    buf = 3
                    cmH, cmW = self.states["collision_map"][i].shape
                    for j in range(length):
                        for k in range(width):
                            wx = (
                                x2
                                + ((j + buf) * math.cos(t2))
                                + ((k - width / 2) * math.sin(t2))
                            )
                            wy = (
                                y2
                                + ((j + buf) * math.sin(t2))
                                - ((k - width / 2) * math.cos(t2))
                            )
                            wx, wy = int(wx), int(wy)
                            if wx < 0 or wx >= cmW or wy < 0 or wy >= cmH:
                                continue
                            self.states["collision_map"][i, wy, wx] = 1
        # Update visitation maps
        for i in range(self.nplanners):
            mx, my = asnumpy(self.states["curr_map_position"][i]).tolist()
            mx, my = int(mx), int(my)
            self.states["visited_map"][i, my - 2 : my + 3, mx - 2 : mx + 3] = 1
        # ======================= Local policy action selection =======================
        # Initialize states at t=0
        if ep_step == 0:
            self.states["curr_local_goals"] = torch.zeros(self.nplanners, 2).to(device)
        # Should local goals be sampled now?
        if ep_step == 0:
            # Condition 1: The very first time-step.
            SAMPLE_LOCAL_GOAL_FLAGS = [1 for _ in range(self.nplanners)]
        elif ep_step % 25 == 0:
            # Condition 2: Re-sample a local goal after every 25 steps.
            # Prevents the agent from getting stuck at a local goal.
            SAMPLE_LOCAL_GOAL_FLAGS = [1.0 for _ in range(self.nplanners)]
        else:
            # Condition 3: (a) The previous local goal was reached.
            prev_goal_reached = (
                self.states["curr_dist2localgoal"] < self.goal_success_radius
            )
            # Condition 3: (b) The previous local goal is occupied.
            goals = self.states["curr_local_goals"].long().to(device)
            prev_gcells = global_map[
                torch.arange(0, goals.shape[0]).long(), :, goals[:, 1], goals[:, 0]
            ]
            prev_goal_occupied = (prev_gcells[:, 0] > self.config.thresh_obstacle) & (
                prev_gcells[:, 1] > self.config.thresh_explored
            )
            SAMPLE_LOCAL_GOAL_FLAGS = asnumpy(
                (prev_goal_reached | prev_goal_occupied).float()
            ).tolist()
        # Execute planner and compute local goals
        self._compute_plans_and_local_goals(
            global_map, self.states["curr_map_position"], SAMPLE_LOCAL_GOAL_FLAGS
        )
        # Update state variables to account for new local goals
        self.states["curr_dist2localgoal"] = self._compute_dist2localgoal(
            global_map,
            self.states["curr_map_position"],
            self.states["curr_local_goals"],
        )
        # Sample action with local policy
        local_masks = 1 - torch.Tensor(SAMPLE_LOCAL_GOAL_FLAGS).to(device).unsqueeze(1)
        recurrent_hidden_states = prev_state_estimates["recurrent_hidden_states"]
        relative_goals = self._compute_relative_local_goals(global_pose, M, s)
        local_policy_inputs = {
            "rgb_at_t": observations["rgb"],
            "goal_at_t": relative_goals,
            "t": ep_time,
        }
        outputs = self.local_policy.act(
            local_policy_inputs,
            recurrent_hidden_states,
            None,
            local_masks,
            deterministic=deterministic,
        )
        (
            local_value,
            local_action,
            local_action_log_probs,
            recurrent_hidden_states,
        ) = outputs
        # Overwrite local policy action in certain cases.
        # (1) Rotate in place for 3 time-steps to allow better map
        # initialization for planner.
        if ep_step < 3:
            local_action.fill_(self.left_action_id)
        # (2) If goal was reached, execute STOP action.
        # Check if the goal was reached
        reached_goal_flag = (
            torch.norm(self.states["curr_ego_world_goals"], dim=1)
            < self.goal_success_radius
        )
        local_action[reached_goal_flag] = self.stop_action_id
        state_estimates = {
            "recurrent_hidden_states": recurrent_hidden_states,
            "map_states": mapper_outputs["mt"],
            "pose_estimates": global_pose,
        }
        local_policy_outputs = {
            "values": local_value,
            "actions": local_action,
            "action_log_probs": local_action_log_probs,
            "local_masks": local_masks,
            "global_map_proc": self._cropped_global_map,
        }

        return (
            mapper_inputs,
            local_policy_inputs,
            mapper_outputs,
            local_policy_outputs,
            state_estimates,
        )

    def _convert_polar2cartesian(self, coors):
        r = coors[:, 0]
        phi = -coors[:, 1]
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        return torch.stack([x, y], dim=1)

    def _compute_plans_and_local_goals(
        self, global_map, agent_map_xy, SAMPLE_LOCAL_GOAL_FLAGS,
    ):
        """
        global_map - (bs, 2, V, V) tensor
        agent_map_xy - (bs, 2) agent's current position on the map
        """
        s = self.mapper.map_config["scale"]
        goal_map_xy = self.states["curr_global_goals"]
        plans = self._compute_plans(
            global_map,
            agent_map_xy,
            goal_map_xy,
            SAMPLE_LOCAL_GOAL_FLAGS,
            cache_map=True,
            crop_map_flag=self.config.crop_map_for_planning,
        )
        # ========================= Sample a local goal =======================
        # Update step counts for sample_random_explored calls
        self.states["sample_random_explored_timer"] += 1
        # Pick a local goal to reach with the local policy
        for i in range(self.nplanners):
            if SAMPLE_LOCAL_GOAL_FLAGS[i] != 1:
                continue
            path_x, path_y = plans[i]
            # If planning failed, sample random goal
            if path_x is None:
                # Note: This is an expensive call, especially when the map is messy
                # and planning keeps failing. Call sample_random_explored only after
                # so many steps elapsed since the last call.
                if self.config.recovery_heuristic == "random_explored_towards_goal":
                    if self.states["sample_random_explored_timer"][i].item() > 10:
                        goal_x, goal_y = self._sample_random_explored_towards_goal(
                            global_map[i],
                            asnumpy(agent_map_xy[i]).tolist(),
                            asnumpy(goal_map_xy[i]).tolist(),
                            s,
                        )
                        # Reset count
                        self.states["sample_random_explored_timer"][i] = 0
                    else:
                        goal_x, goal_y = self._sample_random_towards_goal(
                            global_map[i],
                            asnumpy(agent_map_xy[i]).tolist(),
                            asnumpy(goal_map_xy[i]).tolist(),
                            s,
                        )
                        goal_x, goal_y = asnumpy(agent_map_xy[i]).tolist()
                elif self.config.recovery_heuristic == "random_explored":
                    if self.states["sample_random_explored_timer"][i].item() > 10:
                        goal_x, goal_y = self._sample_random_explored(
                            global_map[i], asnumpy(agent_map_xy[i]).tolist(), s
                        )
                        # Reset count
                        self.states["sample_random_explored_timer"][i] = 0
                    else:
                        goal_x, goal_y = self._sample_random_near_agent(
                            global_map[i], asnumpy(agent_map_xy[i]).tolist(), s
                        )
                        goal_x, goal_y = asnumpy(agent_map_xy[i]).tolist()
                else:
                    raise ValueError
            else:
                delta = min(int(self.planning_step_mts / s), len(path_x) - 1)
                # The path is in reverse order
                goal_x, goal_y = path_x[-delta], path_y[-delta]
            # Ensure goals are within map bounds
            goal_x = np.clip(goal_x, 0, global_map[i].shape[-1] - 1).item()
            goal_y = np.clip(goal_y, 0, global_map[i].shape[-2] - 1).item()
            # Update local goals
            self.states["curr_local_goals"][i, 0] = goal_x
            self.states["curr_local_goals"][i, 1] = goal_y
