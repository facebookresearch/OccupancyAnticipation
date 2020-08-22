#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import time
import json
import h5py
import glob
import itertools

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import math
import numpy as np
import gym.spaces as spaces
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.core.spaces import EmptySpace, ActionSpace
from habitat_extensions.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from occant_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from occant_baselines.common.rollout_storage import (
    RolloutStorageExtended,
    MapLargeRolloutStorageMP,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.rl.ppo import PPO
from occant_baselines.supervised.imitation import Imitation
from occant_baselines.supervised.map_update import MapUpdate
from occant_baselines.rl.ans import ActiveNeuralSLAMExplorer
from occant_baselines.rl.policy_utils import OccupancyAnticipationWrapper
from occant_utils.visualization import generate_topdown_allocentric_map
from occant_utils.common import (
    add_pose,
    convert_world2map,
    convert_gt2channel_to_gtrgb,
)
from occant_utils.metrics import (
    measure_pose_estimation_performance,
    measure_area_seen_performance,
    measure_anticipation_reward,
    measure_map_quality,
    TemporalMetric,
)
from occant_baselines.models.mapnet import DepthProjectionNet
from occant_baselines.models.occant import OccupancyAnticipator
from einops import rearrange, asnumpy


@baseline_registry.register_trainer(name="occant_exp")
class OccAntExpTrainer(BaseRLTrainer):
    r"""Trainer class for Occupancy Anticipated based exploration algorithm.
    """
    supported_tasks = ["Exp-v0"]
    frozen_mapper_types = ["ans_depth", "occant_ground_truth"]

    def __init__(self, config=None):
        if config is not None:
            self._synchronize_configs(config)
        super().__init__(config)

        # Set pytorch random seed for initialization
        torch.manual_seed(config.PYT_RANDOM_SEED)

        self.mapper = None
        self.local_actor_critic = None
        self.global_actor_critic = None
        self.ans_net = None
        self.planner = None
        self.mapper_agent = None
        self.local_agent = None
        self.global_agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

    def _synchronize_configs(self, config):
        r"""Matches configs for different parts of the model as well as the simulator.
        """
        config.defrost()
        config.RL.ANS.PLANNER.nplanners = config.NUM_PROCESSES
        config.RL.ANS.MAPPER.thresh_explored = config.RL.ANS.thresh_explored
        config.RL.ANS.pyt_random_seed = config.PYT_RANDOM_SEED
        config.RL.ANS.OCCUPANCY_ANTICIPATOR.pyt_random_seed = config.PYT_RANDOM_SEED
        # Compute the EGO_PROJECTION options based on the
        # depth sensor information and agent parameters.
        map_size = config.RL.ANS.MAPPER.map_size
        map_scale = config.RL.ANS.MAPPER.map_scale
        min_depth = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        hfov = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV
        width = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
        height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT
        hfov_rad = np.radians(float(hfov))
        vfov_rad = 2 * np.arctan((height / width) * np.tan(hfov_rad / 2.0))
        vfov = np.degrees(vfov_rad).item()
        camera_height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION[1]
        height_thresholds = [0.2, 1.5]
        # Set the EGO_PROJECTION options
        ego_proj_config = config.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION
        ego_proj_config.local_map_shape = (2, map_size, map_size)
        ego_proj_config.map_scale = map_scale
        ego_proj_config.min_depth = min_depth
        ego_proj_config.max_depth = max_depth
        ego_proj_config.hfov = hfov
        ego_proj_config.vfov = vfov
        ego_proj_config.camera_height = camera_height
        ego_proj_config.height_thresholds = height_thresholds
        config.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION = ego_proj_config
        # Set the GT anticipation options
        wall_fov = config.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.wall_fov
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.WALL_FOV = wall_fov
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.MAP_SIZE = map_size
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.MAP_SCALE = map_scale
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.MAX_SENSOR_RANGE = -1
        # Set the correct image scaling values
        config.RL.ANS.MAPPER.image_scale_hw = config.RL.ANS.image_scale_hw
        config.RL.ANS.LOCAL_POLICY.image_scale_hw = config.RL.ANS.image_scale_hw
        # Set the agent dynamics for the local policy
        config.RL.ANS.LOCAL_POLICY.AGENT_DYNAMICS.forward_step = (
            config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE
        )
        config.RL.ANS.LOCAL_POLICY.AGENT_DYNAMICS.turn_angle = (
            config.TASK_CONFIG.SIMULATOR.TURN_ANGLE
        )
        # Enable global_maps measure if imitation learning is used for local policy
        if config.RL.ANS.LOCAL_POLICY.learning_algorithm == "il":
            if "GT_GLOBAL_MAP" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
                config.TASK_CONFIG.TASK.MEASUREMENTS.append("GT_GLOBAL_MAP")
            config.TASK_CONFIG.TASK.GT_GLOBAL_MAP.MAP_SIZE = (
                config.RL.ANS.overall_map_size
            )
            config.TASK_CONFIG.TASK.GT_GLOBAL_MAP.MAP_SCALE = (
                config.RL.ANS.MAPPER.map_scale
            )
        config.freeze()

    def _setup_actor_critic_agent(self, ppo_cfg: Config, ans_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params
            ans_cfg: config node for ActiveNeuralSLAM model

        Returns:
            None
        """
        try:
            os.mkdir(self.config.TENSORBOARD_DIR)
        except:
            pass
        logger.add_filehandler(os.path.join(self.config.TENSORBOARD_DIR, "run.log"))

        occ_cfg = ans_cfg.OCCUPANCY_ANTICIPATOR
        mapper_cfg = ans_cfg.MAPPER
        # Create occupancy anticipation model
        occupancy_model = OccupancyAnticipator(occ_cfg)
        occupancy_model = OccupancyAnticipationWrapper(
            occupancy_model, mapper_cfg.map_size, (128, 128)
        )
        # Create ANS model
        self.ans_net = ActiveNeuralSLAMExplorer(ans_cfg, occupancy_model)
        self.mapper = self.ans_net.mapper
        self.local_actor_critic = self.ans_net.local_policy
        self.global_actor_critic = self.ans_net.global_policy
        # Create depth projection model to estimate visible occupancy
        self.depth_projection_net = DepthProjectionNet(
            ans_cfg.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION
        )
        # Set to device
        self.mapper.to(self.device)
        self.local_actor_critic.to(self.device)
        self.global_actor_critic.to(self.device)
        self.depth_projection_net.to(self.device)
        # ============================== Create agents ================================
        # Mapper agent
        self.mapper_agent = MapUpdate(
            self.mapper,
            lr=mapper_cfg.lr,
            eps=mapper_cfg.eps,
            label_id=mapper_cfg.label_id,
            max_grad_norm=mapper_cfg.max_grad_norm,
            pose_loss_coef=mapper_cfg.pose_loss_coef,
            occupancy_anticipator_type=ans_cfg.OCCUPANCY_ANTICIPATOR.type,
            freeze_projection_unit=mapper_cfg.freeze_projection_unit,
            num_update_batches=mapper_cfg.num_update_batches,
            batch_size=mapper_cfg.map_batch_size,
            mapper_rollouts=self.mapper_rollouts,
        )
        # Local policy
        if ans_cfg.LOCAL_POLICY.use_heuristic_policy:
            self.local_agent = None
        elif ans_cfg.LOCAL_POLICY.learning_algorithm == "rl":
            self.local_agent = PPO(
                actor_critic=self.local_actor_critic,
                clip_param=ppo_cfg.clip_param,
                ppo_epoch=ppo_cfg.ppo_epoch,
                num_mini_batch=ppo_cfg.num_mini_batch,
                value_loss_coef=ppo_cfg.value_loss_coef,
                entropy_coef=ppo_cfg.local_entropy_coef,
                lr=ppo_cfg.local_policy_lr,
                eps=ppo_cfg.eps,
                max_grad_norm=ppo_cfg.max_grad_norm,
            )
        else:
            self.local_agent = Imitation(
                actor_critic=self.local_actor_critic,
                lr=ppo_cfg.local_policy_lr,
                eps=ppo_cfg.eps,
                max_grad_norm=ppo_cfg.max_grad_norm,
            )
        # Global policy
        self.global_agent = PPO(
            actor_critic=self.global_actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )
        if ans_cfg.model_path != "":
            self.resume_checkpoint(ans_cfg.model_path)

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "mapper_state_dict": self.mapper_agent.state_dict(),
            "local_state_dict": self.local_agent.state_dict(),
            "global_state_dict": self.global_agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def resume_checkpoint(self, path=None):
        r"""If an existing checkpoint already exists, resume training.
        """
        checkpoints = glob.glob(f"{self.config.CHECKPOINT_FOLDER}/*.pth")
        ppo_cfg = self.config.RL.PPO
        if path is None:
            if len(checkpoints) == 0:
                num_updates_start = 0
                count_steps = 0
                count_checkpoints = 0
            else:
                # Load lastest checkpoint
                last_ckpt = sorted(checkpoints, key=lambda x: int(x.split(".")[1]))[-1]
                checkpoint_path = last_ckpt
                # Restore checkpoints to models
                ckpt_dict = self.load_checkpoint(checkpoint_path)
                self.mapper_agent.load_state_dict(ckpt_dict["mapper_state_dict"])
                self.local_agent.load_state_dict(ckpt_dict["local_state_dict"])
                self.global_agent.load_state_dict(ckpt_dict["global_state_dict"])
                self.mapper = self.mapper_agent.mapper
                self.local_actor_critic = self.local_agent.actor_critic
                self.global_actor_critic = self.global_agent.actor_critic
                # Set the logging counts
                ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
                num_updates_start = ckpt_dict["extra_state"]["update"] + 1
                count_steps = ckpt_dict["extra_state"]["step"]
                count_checkpoints = ckpt_id + 1
                print(f"Resuming checkpoint {last_ckpt} at {count_steps} frames")
        else:
            print(f"Loading pretrained model!")
            # Restore checkpoints to models
            ckpt_dict = self.load_checkpoint(path)
            self.mapper_agent.load_state_dict(ckpt_dict["mapper_state_dict"])
            self.local_agent.load_state_dict(ckpt_dict["local_state_dict"])
            self.global_agent.load_state_dict(ckpt_dict["global_state_dict"])
            self.mapper = self.mapper_agent.mapper
            self.local_actor_critic = self.local_agent.actor_critic
            self.global_actor_critic = self.global_agent.actor_critic
            num_updates_start = 0
            count_steps = 0
            count_checkpoints = 0

        return num_updates_start, count_steps, count_checkpoints

    def _create_mapper_rollout_inputs(
        self, prev_batch, batch,
    ):
        ans_cfg = self.config.RL.ANS
        mapper_rollout_inputs = {
            "rgb_at_t_1": prev_batch["rgb"],
            "depth_at_t_1": prev_batch["depth"],
            "ego_map_gt_at_t_1": prev_batch["ego_map_gt"],
            "pose_at_t_1": prev_batch["pose"],
            "pose_gt_at_t_1": prev_batch["pose_gt"],
            "rgb_at_t": batch["rgb"],
            "depth_at_t": batch["depth"],
            "ego_map_gt_at_t": batch["ego_map_gt"],
            "pose_at_t": batch["pose"],
            "pose_gt_at_t": batch["pose_gt"],
            "ego_map_gt_anticipated_at_t": batch["ego_map_gt_anticipated"],
            "action_at_t_1": batch["prev_actions"],
        }
        if ans_cfg.OCCUPANCY_ANTICIPATOR.type == "baseline_gt_anticipation":
            mapper_rollout_inputs["ego_map_gt_anticipated_at_t_1"] = prev_batch[
                "ego_map_gt_anticipated"
            ]

        return mapper_rollout_inputs

    def _convert_actions_to_delta(self, actions):
        """actions -> torch Tensor
        """
        sim_cfg = self.config.TASK_CONFIG.SIMULATOR
        delta_xyt = torch.zeros(self.envs.num_envs, 3, device=self.device)
        # Forward step
        act_mask = actions.squeeze(1) == 0
        delta_xyt[act_mask, 0] = sim_cfg.FORWARD_STEP_SIZE
        # Turn left
        act_mask = actions.squeeze(1) == 1
        delta_xyt[act_mask, 2] = math.radians(-sim_cfg.TURN_ANGLE)
        # Turn right
        act_mask = actions.squeeze(1) == 2
        delta_xyt[act_mask, 2] = math.radians(sim_cfg.TURN_ANGLE)
        return delta_xyt

    def _compute_global_metric(self, ground_truth_states, mapper_outputs):
        """Estimates global reward metric for the current states.
        """
        if self.config.RL.ANS.reward_type == "area_seen":
            global_reward_metric = measure_area_seen_performance(
                ground_truth_states["visible_occupancy"], reduction="none"
            )["area_seen"]
        else:
            global_reward_metric = measure_anticipation_reward(
                mapper_outputs["mt"], ground_truth_states["environment_layout"]
            )
        return global_reward_metric.unsqueeze(1)

    def _collect_rollout_step(
        self,
        batch,
        prev_batch,
        episode_step_count,
        state_estimates,
        ground_truth_states,
        masks,
        mapper_rollouts,
        local_rollouts,
        global_rollouts,
        current_local_episode_reward,
        current_global_episode_reward,
        running_episode_stats,
        statistics_dict,
    ):
        pth_time = 0.0
        env_time = 0.0

        device = self.device
        ppo_cfg = self.config.RL.PPO
        ans_cfg = self.config.RL.ANS
        sim_cfg = self.config.TASK_CONFIG.SIMULATOR

        NUM_LOCAL_STEPS = ppo_cfg.num_local_steps

        self.ans_net.eval()

        for t in range(NUM_LOCAL_STEPS):

            # print(f'===> Local time: {t}, episode time: {episode_step_count[0].item()}')
            # ---------------------------- sample actions -----------------------------
            t_sample_action = time.time()

            with torch.no_grad():
                (
                    mapper_inputs,
                    local_policy_inputs,
                    global_policy_inputs,
                    mapper_outputs,
                    local_policy_outputs,
                    global_policy_outputs,
                    state_estimates,
                    intrinsic_rewards,
                ) = self.ans_net.act(
                    batch,
                    prev_batch,
                    state_estimates,
                    episode_step_count,
                    masks,
                    deterministic=ans_cfg.LOCAL_POLICY.deterministic_flag,
                )

            pth_time += time.time() - t_sample_action

            # -------------------- update global rollout stats ------------------------
            t_update_stats = time.time()

            if t == 0:
                # Sanity check
                assert global_policy_inputs is not None

                global_reward_metric = self._compute_global_metric(
                    ground_truth_states, mapper_outputs
                )
                # Update reward for previous global_policy action
                if global_rollouts.step == 0:
                    global_rewards = torch.zeros(self.envs.num_envs, 1)
                else:
                    global_rewards = (
                        global_reward_metric
                        - ground_truth_states["prev_global_reward_metric"]
                    ).cpu()
                ground_truth_states["prev_global_reward_metric"].copy_(
                    global_reward_metric
                )
                global_rollouts.rewards[global_rollouts.step - 1].copy_(
                    global_rewards * ppo_cfg.global_reward_scale
                )
                global_rollouts.insert(
                    global_policy_inputs,
                    None,
                    global_policy_outputs["actions"],
                    global_policy_outputs["action_log_probs"],
                    global_policy_outputs["values"],
                    torch.zeros_like(global_rewards),
                    masks.to(device),
                )
                current_global_episode_reward += global_rewards

            pth_time += time.time() - t_update_stats

            # --------------------- update mapper rollout stats -----------------------
            t_update_stats = time.time()

            mapper_rollout_inputs = self._create_mapper_rollout_inputs(
                prev_batch, batch
            )
            mapper_rollouts.insert(mapper_rollout_inputs)

            pth_time += time.time() - t_update_stats

            # ------------------ update local_policy rollout stats --------------------
            t_update_stats = time.time()

            # Assign local rewards to previous local action
            local_rewards = (
                intrinsic_rewards["local_rewards"]
                + batch["collision_sensor"] * ans_cfg.local_collision_reward
            ).cpu()
            current_local_episode_reward += local_rewards
            # The intrinsic rewards correspond to the previous action, not
            # the one executed currently.
            if local_rollouts.step > 0:
                local_rollouts.rewards[local_rollouts.step - 1].copy_(
                    local_rewards * ppo_cfg.local_reward_scale
                )
            # Update local_rollouts
            if ans_cfg.LOCAL_POLICY.learning_algorithm == "rl":
                local_policy_actions = local_policy_outputs["actions"]
            else:
                local_policy_actions = local_policy_outputs["gt_actions"]

            local_rollouts.insert(
                local_policy_inputs,
                state_estimates["recurrent_hidden_states"],
                local_policy_actions,
                local_policy_outputs["action_log_probs"],
                local_policy_outputs["values"],
                torch.zeros_like(local_rewards),
                local_policy_outputs["local_masks"].to(device),
            )

            pth_time += time.time() - t_update_stats

            # ---------------------- execute environment action -----------------------
            t_step_env = time.time()

            actions = local_policy_outputs["actions"]
            outputs = self.envs.step([a[0].item() for a in actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            env_time += time.time() - t_step_env

            # -------------------- update ground-truth states -------------------------
            t_update_stats = time.time()

            masks.copy_(
                torch.tensor(
                    [[0.0] if done else [1.0] for done in dones], dtype=torch.float
                )
            )
            # Sanity check
            assert episode_step_count[0].item() <= self.config.T_EXP - 1
            assert not dones[0], "DONE must not be called during training"

            del prev_batch
            prev_batch = batch
            batch = self._prepare_batch(
                observations, prev_batch=prev_batch, device=device, actions=actions
            )

            # Update visible occupancy
            ground_truth_states["visible_occupancy"] = self.mapper.ext_register_map(
                ground_truth_states["visible_occupancy"],
                rearrange(batch["ego_map_gt"], "b h w c -> b c h w"),
                batch["pose_gt"],
            )
            ground_truth_states["pose"].copy_(batch["pose_gt"])
            # Update ground_truth world layout that is provided only at episode start
            # to avoid data transfer bottlenecks
            if episode_step_count[0].item() == 0 and "gt_global_map" in infos[0].keys():
                environment_layout = np.stack(
                    [info["gt_global_map"] for info in infos], axis=0
                )
                environment_layout = rearrange(environment_layout, "b h w c -> b c h w")
                environment_layout = torch.Tensor(environment_layout)
                ground_truth_states["environment_layout"].copy_(environment_layout)

            # The ground_truth world layout is used to generate ground_truth action
            # labels for local policy during imitation.
            if ans_cfg.LOCAL_POLICY.learning_algorithm == "il":
                batch["gt_global_map"] = ground_truth_states["environment_layout"]

            pth_time += time.time() - t_update_stats

            episode_step_count += 1

        return (
            pth_time,
            env_time,
            self.envs.num_envs * NUM_LOCAL_STEPS,
            prev_batch,
            batch,
            state_estimates,
            ground_truth_states,
        )

    def _supplementary_rollout_update(
        self,
        batch,
        prev_batch,
        episode_step_count,
        state_estimates,
        ground_truth_states,
        masks,
        local_rollouts,
        global_rollouts,
        update_option="local",
    ):
        """
        Since the inputs for local, global policies are obtained only after
        a forward pass, it will not be possible to update the rollouts immediately
        after self.envs.step() . This causes a delay of 1 step in the rollout
        updates for local, global policies. To account for this, perform this
        supplementary rollout update just before updating the corresponding policy.
        """
        pth_time = 0.0
        env_time = 0.0
        ppo_cfg = self.config.RL.PPO
        ans_cfg = self.config.RL.ANS

        t_sample_action = time.time()

        # Copy states before sampling actions
        ans_states_copy = self.ans_net.get_states()

        self.ans_net.eval()

        with torch.no_grad():
            (
                mapper_inputs,
                local_policy_inputs,
                global_policy_inputs,
                mapper_outputs,
                local_policy_outputs,
                _,
                _,
                intrinsic_rewards,
            ) = self.ans_net.act(
                batch,
                prev_batch,
                state_estimates,
                episode_step_count,
                masks,
                deterministic=ans_cfg.LOCAL_POLICY.deterministic_flag,
            )

        self.ans_net.train()

        # Restore states
        self.ans_net.update_states(ans_states_copy)

        pth_time += time.time() - t_sample_action

        t_update_stats = time.time()

        # Update local_rollouts
        if update_option == "local":
            for k, v in local_policy_inputs.items():
                local_rollouts.observations[k][local_rollouts.step].copy_(v)
            local_rewards = intrinsic_rewards["local_rewards"].cpu()
            local_rollouts.rewards[local_rollouts.step - 1].copy_(
                local_rewards * ppo_cfg.local_reward_scale
            )
            local_masks = local_policy_outputs["local_masks"]
            local_rollouts.masks[local_rollouts.step].copy_(local_masks)

        # Update global_rollouts if available
        if update_option == "global":
            # Sanity check
            assert episode_step_count[0].item() % ans_cfg.goal_interval == 0
            assert global_policy_inputs is not None
            for k, v in global_policy_inputs.items():
                global_rollouts.observations[k][global_rollouts.step].copy_(v)
            global_reward_metric = self._compute_global_metric(
                ground_truth_states, mapper_outputs
            )
            global_rewards = (
                global_reward_metric - ground_truth_states["prev_global_reward_metric"]
            ).cpu()
            global_rollouts.rewards[global_rollouts.step - 1].copy_(
                global_rewards * self.config.RL.PPO.global_reward_scale
            )
            global_rollouts.masks[global_rollouts.step].copy_(masks)

        pth_time += time.time() - t_update_stats

        return pth_time

    def _update_mapper_agent(self, mapper_rollouts):
        t_update_model = time.time()

        losses = self.mapper_agent.update(mapper_rollouts)

        return time.time() - t_update_model, losses

    def _update_local_agent(self, local_rollouts):
        t_update_model = time.time()

        ppo_cfg = self.config.RL.PPO

        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in local_rollouts.observations.items()
            }
            next_local_value = self.local_actor_critic.get_value(
                last_observation,
                local_rollouts.recurrent_hidden_states[-1],
                local_rollouts.prev_actions[-1],
                local_rollouts.masks[-1],
            ).detach()

        local_rollouts.compute_returns(
            next_local_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        (
            local_value_loss,
            local_action_loss,
            local_dist_entropy,
        ) = self.local_agent.update(local_rollouts)

        update_metrics = {
            "value_loss": local_value_loss,
            "action_loss": local_action_loss,
            "dist_entropy": local_dist_entropy,
        }

        local_rollouts.after_update()

        return time.time() - t_update_model, update_metrics

    def _update_global_agent(self, global_rollouts):
        t_update_model = time.time()

        ppo_cfg = self.config.RL.PPO

        with torch.no_grad():
            last_observation = {
                k: v[-1].to(self.device)
                for k, v in global_rollouts.observations.items()
            }
            next_global_value = self.global_actor_critic.get_value(
                last_observation,
                None,
                global_rollouts.prev_actions[-1].to(self.device),
                global_rollouts.masks[-1].to(self.device),
            ).detach()

        global_rollouts.compute_returns(
            next_global_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        (
            global_value_loss,
            global_action_loss,
            global_dist_entropy,
        ) = self.global_agent.update(global_rollouts)

        update_metrics = {
            "value_loss": global_value_loss,
            "action_loss": global_action_loss,
            "dist_entropy": global_dist_entropy,
        }

        global_rollouts.after_update()

        return time.time() - t_update_model, update_metrics

    def _assign_devices(self):
        # Assign devices for the simulator
        if len(self.config.SIMULATOR_GPU_IDS) > 0:
            devices = self.config.SIMULATOR_GPU_IDS
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            devices = [int(dev) for dev in visible_devices]
            # Devices need to be indexed between 0 to N-1
            devices = [dev for dev in range(len(devices))]
            if len(devices) > 1:
                devices = devices[1:]
        else:
            devices = None
        return devices

    def _create_mapper_rollouts(self, ppo_cfg, ans_cfg):
        M = ans_cfg.overall_map_size
        V = ans_cfg.MAPPER.map_size
        s = ans_cfg.MAPPER.map_scale
        imH, imW = ans_cfg.image_scale_hw
        mapper_observation_space = {
            "rgb_at_t_1": spaces.Box(
                low=0.0, high=255.0, shape=(imH, imW, 3), dtype=np.float32
            ),
            "depth_at_t_1": spaces.Box(
                low=0.0, high=255.0, shape=(imH, imW, 1), dtype=np.float32
            ),
            "ego_map_gt_at_t_1": spaces.Box(
                low=0.0, high=1.0, shape=(V, V, 2), dtype=np.float32
            ),
            "pose_at_t_1": spaces.Box(
                low=-100000.0, high=100000.0, shape=(3,), dtype=np.float32
            ),
            "pose_gt_at_t_1": spaces.Box(
                low=-100000.0, high=100000.0, shape=(3,), dtype=np.float32
            ),
            "pose_hat_at_t_1": spaces.Box(
                low=-100000.0, high=100000.0, shape=(3,), dtype=np.float32
            ),
            "rgb_at_t": spaces.Box(
                low=0.0, high=255.0, shape=(imH, imW, 3), dtype=np.float32
            ),
            "depth_at_t": spaces.Box(
                low=0.0, high=255.0, shape=(imH, imW, 1), dtype=np.float32
            ),
            "ego_map_gt_at_t": spaces.Box(
                low=0.0, high=1.0, shape=(V, V, 2), dtype=np.float32
            ),
            "pose_at_t": spaces.Box(
                low=-100000.0, high=100000.0, shape=(3,), dtype=np.float32
            ),
            "pose_gt_at_t": spaces.Box(
                low=-100000.0, high=100000.0, shape=(3,), dtype=np.float32
            ),
            "ego_map_gt_anticipated_at_t": self.envs.observation_spaces[0].spaces[
                "ego_map_gt_anticipated"
            ],
            "action_at_t_1": spaces.Box(low=0, high=4, shape=(1,), dtype=np.int32),
        }
        if ans_cfg.OCCUPANCY_ANTICIPATOR.type == "baseline_gt_anticipation":
            mapper_observation_space[
                "ego_map_gt_anticipated_at_t_1"
            ] = self.envs.observation_spaces[0].spaces["ego_map_gt_anticipated"]
        mapper_observation_space = spaces.Dict(mapper_observation_space)
        # Multiprocessing manager
        mapper_manager = mp.Manager()
        mapper_device = self.device
        if ans_cfg.MAPPER.use_data_parallel and len(ans_cfg.MAPPER.gpu_ids) > 0:
            mapper_device = ans_cfg.MAPPER.gpu_ids[0]
        mapper_rollouts = MapLargeRolloutStorageMP(
            ans_cfg.MAPPER.replay_size,
            mapper_observation_space,
            mapper_device,
            mapper_manager,
        )

        return mapper_rollouts

    def _create_global_rollouts(self, ppo_cfg, ans_cfg):
        M = ans_cfg.overall_map_size
        G = ans_cfg.GLOBAL_POLICY.map_size
        global_observation_space = spaces.Dict(
            {
                "pose_in_map_at_t": spaces.Box(
                    low=-100000.0, high=100000.0, shape=(2,), dtype=np.float32
                ),
                "map_at_t": spaces.Box(
                    low=0.0, high=1.0, shape=(4, M, M), dtype=np.float32
                ),
            }
        )
        global_action_space = ActionSpace(
            {
                f"({x[0]}, {x[1]})": EmptySpace()
                for x in itertools.product(range(G), range(G))
            }
        )
        global_rollouts = RolloutStorageExtended(
            ppo_cfg.num_global_steps,
            self.envs.num_envs,
            global_observation_space,
            global_action_space,
            1,
            enable_recurrence=False,
            delay_observations_entry=True,
            delay_masks_entry=True,
            enable_memory_efficient_mode=True,
        )
        return global_rollouts

    def _create_local_rollouts(self, ppo_cfg, ans_cfg):
        imH, imW = ans_cfg.image_scale_hw
        local_action_space = ActionSpace(
            {
                "move_forward": EmptySpace(),
                "turn_left": EmptySpace(),
                "turn_right": EmptySpace(),
            }
        )
        local_observation_space = {
            "rgb_at_t": spaces.Box(
                low=0.0, high=255.0, shape=(imH, imW, 3), dtype=np.float32
            ),
            "goal_at_t": spaces.Box(
                low=-100000.0, high=100000.0, shape=(2,), dtype=np.float32
            ),
            "t": spaces.Box(low=0.0, high=100000.0, shape=(1,), dtype=np.float32),
        }

        local_observation_space = spaces.Dict(local_observation_space)
        local_rollouts = RolloutStorageExtended(
            ppo_cfg.num_local_steps,
            self.envs.num_envs,
            local_observation_space,
            local_action_space,
            ans_cfg.LOCAL_POLICY.hidden_size,
            enable_recurrence=True,
            delay_observations_entry=True,
            delay_masks_entry=True,
        )
        return local_rollouts

    def _prepare_batch(self, observations, prev_batch=None, device=None, actions=None):
        imH, imW = self.config.RL.ANS.image_scale_hw
        device = self.device if device is None else device
        batch = batch_obs(observations, device=device)
        if batch["rgb"].size(1) != imH or batch["rgb"].size(2) != imW:
            rgb = rearrange(batch["rgb"], "b h w c -> b c h w")
            rgb = F.interpolate(rgb, (imH, imW), mode="bilinear")
            batch["rgb"] = rearrange(rgb, "b c h w -> b h w c")
        if batch["depth"].size(1) != imH or batch["depth"].size(2) != imW:
            depth = rearrange(batch["depth"], "b h w c -> b c h w")
            depth = F.interpolate(depth, (imH, imW), mode="nearest")
            batch["depth"] = rearrange(depth, "b c h w -> b h w c")
        # Compute ego_map_gt from depth
        ego_map_gt_b = self.depth_projection_net(
            rearrange(batch["depth"], "b h w c -> b c h w")
        )
        batch["ego_map_gt"] = rearrange(ego_map_gt_b, "b c h w -> b h w c")
        if actions is None:
            # Initialization condition
            # If pose estimates are not available, set the initial estimate to zeros.
            if "pose" not in batch:
                # Set initial pose estimate to zero
                batch["pose"] = torch.zeros(self.envs.num_envs, 3).to(self.device)
            batch["prev_actions"] = torch.zeros(self.envs.num_envs, 1).to(self.device)
        else:
            # Rollouts condition
            # If pose estimates are not available, compute them from action taken.
            if "pose" not in batch:
                assert prev_batch is not None
                actions_delta = self._convert_actions_to_delta(actions)
                batch["pose"] = add_pose(prev_batch["pose"], actions_delta)
            batch["prev_actions"] = actions

        return batch

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            devices=self._assign_devices(),
        )

        ppo_cfg = self.config.RL.PPO
        ans_cfg = self.config.RL.ANS
        mapper_cfg = self.config.RL.ANS.MAPPER
        occ_cfg = self.config.RL.ANS.OCCUPANCY_ANTICIPATOR
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self.mapper_rollouts = self._create_mapper_rollouts(ppo_cfg, ans_cfg)
        self._setup_actor_critic_agent(ppo_cfg, ans_cfg)
        logger.info(
            "mapper_agent number of parameters: {}".format(
                sum(param.numel() for param in self.mapper_agent.parameters())
            )
        )
        logger.info(
            "local_agent number of parameters: {}".format(
                sum(param.numel() for param in self.local_agent.parameters())
            )
        )
        logger.info(
            "global_agent number of parameters: {}".format(
                sum(param.numel() for param in self.global_agent.parameters())
            )
        )
        mapper_rollouts = self.mapper_rollouts
        global_rollouts = self._create_global_rollouts(ppo_cfg, ans_cfg)
        local_rollouts = self._create_local_rollouts(ppo_cfg, ans_cfg)
        global_rollouts.to(self.device)
        local_rollouts.to(self.device)
        # ===================== Create statistics buffers =====================
        statistics_dict = {}
        # Mapper statistics
        statistics_dict["mapper"] = defaultdict(
            lambda: deque(maxlen=ppo_cfg.loss_stats_window_size)
        )
        # Local policy statistics
        local_episode_rewards = torch.zeros(self.envs.num_envs, 1)
        statistics_dict["local_policy"] = defaultdict(
            lambda: deque(maxlen=ppo_cfg.loss_stats_window_size)
        )
        window_local_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_local_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)
        # Global policy statistics
        global_episode_rewards = torch.zeros(self.envs.num_envs, 1)
        statistics_dict["global_policy"] = defaultdict(
            lambda: deque(maxlen=ppo_cfg.loss_stats_window_size)
        )
        window_global_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_global_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)
        # Overall count statistics
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        # ==================== Measuring memory consumption ===================
        total_memory_size = 0
        print("=================== Mapper rollouts ======================")
        for k, v in mapper_rollouts.observations.items():
            mem = v.element_size() * v.nelement() * 1e-9
            print(f"key: {k:<40s}, memory: {mem:>10.4f} GB")
            total_memory_size += mem
        print(f"Total memory: {total_memory_size:>10.4f} GB")

        total_memory_size = 0
        print("================== Local policy rollouts =====================")
        for k, v in local_rollouts.observations.items():
            mem = v.element_size() * v.nelement() * 1e-9
            print(f"key: {k:<40s}, memory: {mem:>10.4f} GB")
            total_memory_size += mem
        print(f"Total memory: {total_memory_size:>10.4f} GB")

        total_memory_size = 0
        print("================== Global policy rollouts ====================")
        for k, v in global_rollouts.observations.items():
            mem = v.element_size() * v.nelement() * 1e-9
            print(f"key: {k:<40s}, memory: {mem:>10.4f} GB")
            total_memory_size += mem
        print(f"Total memory: {total_memory_size:>10.4f} GB")
        # Resume checkpoint if available
        (
            num_updates_start,
            count_steps_start,
            count_checkpoints,
        ) = self.resume_checkpoint()
        count_steps = count_steps_start

        imH, imW = ans_cfg.image_scale_hw
        M = ans_cfg.overall_map_size
        # ==================== Create state variables =================
        state_estimates = {
            # Agent's pose estimate
            "pose_estimates": torch.zeros(self.envs.num_envs, 3).to(self.device),
            # Agent's map
            "map_states": torch.zeros(self.envs.num_envs, 2, M, M).to(self.device),
            "recurrent_hidden_states": torch.zeros(
                1, self.envs.num_envs, ans_cfg.LOCAL_POLICY.hidden_size
            ).to(self.device),
            "visited_states": torch.zeros(self.envs.num_envs, 1, M, M).to(self.device),
        }
        ground_truth_states = {
            # To measure area seen
            "visible_occupancy": torch.zeros(
                self.envs.num_envs, 2, M, M, device=self.device
            ),
            "pose": torch.zeros(self.envs.num_envs, 3, device=self.device),
            "prev_global_reward_metric": torch.zeros(
                self.envs.num_envs, 1, device=self.device
            ),
        }
        if (
            ans_cfg.reward_type == "map_accuracy"
            or ans_cfg.LOCAL_POLICY.learning_algorithm == "il"
        ):
            ground_truth_states["environment_layout"] = torch.zeros(
                self.envs.num_envs, 2, M, M
            ).to(self.device)
        masks = torch.zeros(self.envs.num_envs, 1)
        episode_step_count = torch.zeros(self.envs.num_envs, 1, device=self.device)

        # ==================== Reset the environments =================
        observations = self.envs.reset()
        batch = self._prepare_batch(observations)
        prev_batch = batch
        # Update visible occupancy
        ground_truth_states["visible_occupancy"] = self.mapper.ext_register_map(
            ground_truth_states["visible_occupancy"],
            rearrange(batch["ego_map_gt"], "b h w c -> b c h w"),
            batch["pose_gt"],
        )
        ground_truth_states["pose"].copy_(batch["pose_gt"])

        current_local_episode_reward = torch.zeros(self.envs.num_envs, 1)
        current_global_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            local_reward=torch.zeros(self.envs.num_envs, 1),
            global_reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        # Useful variables
        NUM_MAPPER_STEPS = ans_cfg.MAPPER.num_mapper_steps
        NUM_LOCAL_STEPS = ppo_cfg.num_local_steps
        NUM_GLOBAL_STEPS = ppo_cfg.num_global_steps
        GLOBAL_UPDATE_INTERVAL = NUM_GLOBAL_STEPS * ans_cfg.goal_interval
        NUM_GLOBAL_UPDATES_PER_EPISODE = self.config.T_EXP // GLOBAL_UPDATE_INTERVAL
        NUM_GLOBAL_UPDATES = (
            self.config.NUM_EPISODES
            * NUM_GLOBAL_UPDATES_PER_EPISODE
            // self.config.NUM_PROCESSES
        )
        # Sanity checks
        assert (
            NUM_MAPPER_STEPS % NUM_GLOBAL_STEPS == 0
        ), "Mapper steps must be a multiple of global steps"
        assert (
            NUM_LOCAL_STEPS == ans_cfg.goal_interval
        ), "Local steps must be same as subgoal sampling interval"
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(num_updates_start, NUM_GLOBAL_UPDATES):
                for step in range(NUM_GLOBAL_STEPS):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                        prev_batch,
                        batch,
                        state_estimates,
                        ground_truth_states,
                    ) = self._collect_rollout_step(
                        batch,
                        prev_batch,
                        episode_step_count,
                        state_estimates,
                        ground_truth_states,
                        masks,
                        mapper_rollouts,
                        local_rollouts,
                        global_rollouts,
                        current_local_episode_reward,
                        current_global_episode_reward,
                        running_episode_stats,
                        statistics_dict,
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                    # Useful flags
                    FROZEN_MAPPER = (
                        True
                        if mapper_cfg.ignore_pose_estimator
                        and (
                            occ_cfg.type in self.frozen_mapper_types
                            or mapper_cfg.freeze_projection_unit
                        )
                        else False
                    )
                    UPDATE_MAPPER_FLAG = (
                        True
                        if episode_step_count[0].item() % NUM_MAPPER_STEPS == 0
                        else False
                    )
                    UPDATE_LOCAL_FLAG = True

                    # ------------------------ update mapper --------------------------
                    if UPDATE_MAPPER_FLAG:
                        (
                            delta_pth_time,
                            update_metrics_mapper,
                        ) = self._update_mapper_agent(mapper_rollouts)

                        for k, v in update_metrics_mapper.items():
                            statistics_dict["mapper"][k].append(v)

                    pth_time += delta_pth_time

                    # -------------------- update local policy ------------------------
                    if UPDATE_LOCAL_FLAG:
                        delta_pth_time = self._supplementary_rollout_update(
                            batch,
                            prev_batch,
                            episode_step_count,
                            state_estimates,
                            ground_truth_states,
                            masks,
                            local_rollouts,
                            global_rollouts,
                            update_option="local",
                        )

                        # Sanity check
                        assert local_rollouts.step == local_rollouts.num_steps

                        pth_time += delta_pth_time
                        (
                            delta_pth_time,
                            update_metrics_local,
                        ) = self._update_local_agent(local_rollouts)

                        for k, v in update_metrics_local.items():
                            statistics_dict["local_policy"][k].append(v)

                    # -------------------------- log statistics -----------------------
                    for k, v in statistics_dict.items():
                        logger.info(
                            "=========== {:20s} ============".format(k + " stats")
                        )
                        for kp, vp in v.items():
                            if len(vp) > 0:
                                writer.add_scalar(f"{k}/{kp}", np.mean(vp), count_steps)
                                logger.info(f"{kp:25s}: {np.mean(vp).item():10.5f}")

                    for k, v in running_episode_stats.items():
                        window_episode_stats[k].append(v.clone())

                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
                        )
                        for k, v in window_episode_stats.items()
                    }
                    deltas["count"] = max(deltas["count"], 1.0)

                    writer.add_scalar(
                        "local_reward",
                        deltas["local_reward"] / deltas["count"],
                        count_steps,
                    )
                    writer.add_scalar(
                        "global_reward",
                        deltas["global_reward"] / deltas["count"],
                        count_steps,
                    )
                    fps = (count_steps - count_steps_start) / (time.time() - t_start)
                    writer.add_scalar("fps", fps, count_steps)

                    if update > 0:
                        logger.info("update: {}\tfps: {:.3f}\t".format(update, fps))

                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                            "frames: {}".format(update, env_time, pth_time, count_steps)
                        )

                        logger.info(
                            "Average window size: {}  {}".format(
                                len(window_episode_stats["count"]),
                                "  ".join(
                                    "{}: {:.3f}".format(k, v / deltas["count"])
                                    for k, v in deltas.items()
                                    if k != "count"
                                ),
                            )
                        )

                    pth_time += delta_pth_time

                # At episode termination, manually set masks to zeros.
                if episode_step_count[0].item() == self.config.T_EXP:
                    masks.fill_(0)

                # -------------------- update global policy -----------------------
                self._supplementary_rollout_update(
                    batch,
                    prev_batch,
                    episode_step_count,
                    state_estimates,
                    ground_truth_states,
                    masks,
                    local_rollouts,
                    global_rollouts,
                    update_option="global",
                )

                # Sanity check
                assert global_rollouts.step == NUM_GLOBAL_STEPS

                (delta_pth_time, update_metrics_global,) = self._update_global_agent(
                    global_rollouts
                )

                for k, v in update_metrics_global.items():
                    statistics_dict["global_policy"][k].append(v)

                pth_time += delta_pth_time

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(step=count_steps, update=update),
                    )
                    count_checkpoints += 1

                # Manually enforce episode termination criterion
                if episode_step_count[0].item() == self.config.T_EXP:

                    # Update episode rewards
                    running_episode_stats["local_reward"] += (
                        1 - masks
                    ) * current_local_episode_reward
                    running_episode_stats["global_reward"] += (
                        1 - masks
                    ) * current_global_episode_reward
                    running_episode_stats["count"] += 1 - masks

                    current_local_episode_reward *= masks
                    current_global_episode_reward *= masks

                    # Measure accumulative error in pose estimates
                    pose_estimation_metrics = measure_pose_estimation_performance(
                        state_estimates["pose_estimates"], ground_truth_states["pose"]
                    )
                    for k, v in pose_estimation_metrics.items():
                        statistics_dict["mapper"]["episode_" + k].append(v)

                    observations = self.envs.reset()
                    batch = self._prepare_batch(observations)
                    prev_batch = batch
                    # Reset episode step counter
                    episode_step_count.fill_(0)
                    # Reset states
                    for k in ground_truth_states.keys():
                        ground_truth_states[k].fill_(0)
                    for k in state_estimates.keys():
                        state_estimates[k].fill_(0)
                    # Update visible occupancy
                    ground_truth_states[
                        "visible_occupancy"
                    ] = self.mapper.ext_register_map(
                        ground_truth_states["visible_occupancy"],
                        rearrange(batch["ego_map_gt"], "b h w c -> b c h w"),
                        batch["pose_gt"],
                    )
                    ground_truth_states["pose"].copy_(batch["pose_gt"])

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO
        ans_cfg = config.RL.ANS

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if "COLLISION_SENSOR" not in config.TASK_CONFIG.TASK.SENSORS:
            config.TASK_CONFIG.TASK.SENSORS.append("COLLISION_SENSOR")
        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_EXP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self.mapper_rollouts = None
        self._setup_actor_critic_agent(ppo_cfg, ans_cfg)

        self.mapper_agent.load_state_dict(ckpt_dict["mapper_state_dict"])
        if self.local_agent is not None:
            self.local_agent.load_state_dict(ckpt_dict["local_state_dict"])
            self.local_actor_critic = self.local_agent.actor_critic
        else:
            self.local_actor_critic = self.ans_net.local_policy
        self.global_agent.load_state_dict(ckpt_dict["global_state_dict"])
        self.mapper = self.mapper_agent.mapper
        self.global_actor_critic = self.global_agent.actor_critic

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        assert (
            self.envs.num_envs == 1
        ), "Number of environments needs to be 1 for evaluation"

        # Set models to evaluation
        self.mapper.eval()
        self.local_actor_critic.eval()
        self.global_actor_critic.eval()

        M = ans_cfg.overall_map_size
        V = ans_cfg.MAPPER.map_size
        s = ans_cfg.MAPPER.map_scale
        imH, imW = ans_cfg.image_scale_hw

        # Define metric accumulators
        mapping_metrics = defaultdict(lambda: TemporalMetric())
        pose_estimation_metrics = defaultdict(lambda: TemporalMetric())

        # Environment statistics
        episode_statistics = []
        episode_visualization_maps = []

        times_per_episode = deque(maxlen=100)

        for ep in range(number_of_eval_episodes):
            if ep == 0:
                observations = self.envs.reset()
            current_episodes = self.envs.current_episodes()
            batch = self._prepare_batch(observations)
            prev_batch = batch
            state_estimates = {
                "pose_estimates": torch.zeros(self.envs.num_envs, 3).to(self.device),
                "map_states": torch.zeros(self.envs.num_envs, 2, M, M).to(self.device),
                "recurrent_hidden_states": torch.zeros(
                    1, self.envs.num_envs, ans_cfg.LOCAL_POLICY.hidden_size
                ).to(self.device),
                "visited_states": torch.zeros(self.envs.num_envs, 1, M, M).to(
                    self.device
                ),
            }
            ground_truth_states = {
                "visible_occupancy": torch.zeros(self.envs.num_envs, 2, M, M).to(
                    self.device
                ),
                "pose": torch.zeros(self.envs.num_envs, 3).to(self.device),
                "environment_layout": None,
            }

            # Reset ANS states
            self.ans_net.reset()

            current_episode_reward = torch.zeros(
                self.envs.num_envs, 1, device=self.device
            )

            prev_actions = torch.zeros(
                self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long,
            )

            masks = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device)

            # Visualization stuff
            gt_agent_poses_over_time = [[] for _ in range(self.config.NUM_PROCESSES)]
            pred_agent_poses_over_time = [[] for _ in range(self.config.NUM_PROCESSES)]
            gt_map_agent = asnumpy(
                convert_world2map(ground_truth_states["pose"], (M, M), s)
            )
            pred_map_agent = asnumpy(
                convert_world2map(state_estimates["pose_estimates"], (M, M), s)
            )
            pred_map_agent = np.concatenate(
                [pred_map_agent, asnumpy(state_estimates["pose_estimates"][:, 2:3]),],
                axis=1,
            )
            for i in range(self.config.NUM_PROCESSES):
                gt_agent_poses_over_time[i].append(gt_map_agent[i])
                pred_agent_poses_over_time[i].append(pred_map_agent[i])

            ep_start_time = time.time()
            for ep_step in range(self.config.T_EXP):
                ep_time = torch.zeros(
                    self.config.NUM_PROCESSES, 1, device=self.device
                ).fill_(ep_step)

                prev_pose_hat = state_estimates["pose_estimates"]
                with torch.no_grad():
                    (
                        mapper_inputs,
                        local_policy_inputs,
                        global_policy_inputs,
                        mapper_outputs,
                        local_policy_outputs,
                        global_policy_outputs,
                        state_estimates,
                        intrinsic_rewards,
                    ) = self.ans_net.act(
                        batch,
                        prev_batch,
                        state_estimates,
                        ep_time,
                        masks,
                        deterministic=ans_cfg.LOCAL_POLICY.deterministic_flag,
                    )

                    actions = local_policy_outputs["actions"]
                    prev_actions.copy_(actions)
                curr_pose_hat = state_estimates["pose_estimates"]

                # Update GT estimates at t = ep_step
                ground_truth_states["pose"] = batch["pose_gt"]
                ground_truth_states[
                    "visible_occupancy"
                ] = self.ans_net.mapper.ext_register_map(
                    ground_truth_states["visible_occupancy"],
                    batch["ego_map_gt"].permute(0, 3, 1, 2),
                    batch["pose_gt"],
                )

                # Visualization stuff
                gt_map_agent = asnumpy(
                    convert_world2map(ground_truth_states["pose"], (M, M), s)
                )
                gt_map_agent = np.concatenate(
                    [gt_map_agent, asnumpy(ground_truth_states["pose"][:, 2:3])],
                    axis=1,
                )
                pred_map_agent = asnumpy(
                    convert_world2map(state_estimates["pose_estimates"], (M, M), s)
                )
                pred_map_agent = np.concatenate(
                    [
                        pred_map_agent,
                        asnumpy(state_estimates["pose_estimates"][:, 2:3]),
                    ],
                    axis=1,
                )
                for i in range(self.config.NUM_PROCESSES):
                    gt_agent_poses_over_time[i].append(gt_map_agent[i])
                    pred_agent_poses_over_time[i].append(pred_map_agent[i])

                outputs = self.envs.step([a[0].item() for a in actions])

                observations, _, dones, infos = [list(x) for x in zip(*outputs)]

                if ep_step == 0:
                    environment_layout = np.stack(
                        [info["gt_global_map"] for info in infos], axis=0
                    )  # (bs, M, M, 2)
                    environment_layout = rearrange(
                        environment_layout, "b h w c -> b c h w"
                    )  # (bs, 2, M, M)
                    environment_layout = torch.Tensor(environment_layout).to(
                        self.device
                    )
                    ground_truth_states["environment_layout"] = environment_layout
                    # Update environment statistics
                    for i in range(self.envs.num_envs):
                        episode_statistics.append(infos[i]["episode_statistics"])

                if ep_step == self.config.T_EXP - 1:
                    assert dones[0]

                prev_batch = batch
                batch = self._prepare_batch(observations, prev_batch, actions=actions)

                masks = torch.tensor(
                    [[0.0] if done else [1.0] for done in dones],
                    dtype=torch.float,
                    device=self.device,
                )

                next_episodes = self.envs.current_episodes()
                envs_to_pause = []
                n_envs = self.envs.num_envs

                if ep_step == 0 or (ep_step + 1) % 50 == 0:
                    curr_all_metrics = {}
                    # Compute accumulative pose estimation error
                    pose_hat_final = state_estimates["pose_estimates"]  # (bs, 3)
                    pose_gt_final = ground_truth_states["pose"]  # (bs, 3)
                    curr_pose_estimation_metrics = measure_pose_estimation_performance(
                        pose_hat_final, pose_gt_final, reduction="sum",
                    )
                    for k, v in curr_pose_estimation_metrics.items():
                        pose_estimation_metrics[k].update(
                            v, self.envs.num_envs, ep_step
                        )
                    curr_all_metrics.update(curr_pose_estimation_metrics)

                    # Compute map quality
                    curr_map_quality_metrics = measure_map_quality(
                        state_estimates["map_states"],
                        ground_truth_states["environment_layout"],
                        s,
                        entropy_thresh=1.0,
                        reduction="sum",
                        apply_mask=True,
                    )
                    for k, v in curr_map_quality_metrics.items():
                        mapping_metrics[k].update(v, self.envs.num_envs, ep_step)
                    curr_all_metrics.update(curr_map_quality_metrics)

                    # Compute area seen
                    curr_area_seen_metrics = measure_area_seen_performance(
                        ground_truth_states["visible_occupancy"], s, reduction="sum"
                    )
                    for k, v in curr_area_seen_metrics.items():
                        mapping_metrics[k].update(v, self.envs.num_envs, ep_step)
                    curr_all_metrics.update(curr_area_seen_metrics)

                    # Debug stuff
                    if (ep_step + 1) == self.config.T_EXP:
                        times_per_episode.append(time.time() - ep_start_time)
                        mins_per_episode = np.mean(times_per_episode).item() / 60.0
                        eta_completion = mins_per_episode * (
                            number_of_eval_episodes - ep - 1
                        )
                        logger.info(
                            f"====> episode {ep}/{number_of_eval_episodes} done"
                        )
                        logger.info(
                            f"Time per episode: {mins_per_episode:.3f} mins"
                            f"\tETA: {eta_completion:.3f} mins"
                        )
                        for k, v in curr_all_metrics.items():
                            logger.info(f"{k:30s}: {v/self.envs.num_envs:8.3f}")

                for i in range(n_envs):
                    if (
                        len(self.config.VIDEO_OPTION) > 0
                        or self.config.SAVE_STATISTICS_FLAG
                    ):
                        # episode ended
                        if masks[i].item() == 0:
                            episode_visualization_maps.append(rgb_frames[i][-1])
                            video_metrics = {}
                            for k in ["area_seen", "mean_iou", "map_accuracy"]:
                                video_metrics[k] = curr_all_metrics[k]
                            if len(self.config.VIDEO_OPTION) > 0:
                                generate_video(
                                    video_option=self.config.VIDEO_OPTION,
                                    video_dir=self.config.VIDEO_DIR,
                                    images=rgb_frames[i],
                                    episode_id=current_episodes[i].episode_id,
                                    checkpoint_idx=checkpoint_index,
                                    metrics=video_metrics,
                                    tb_writer=writer,
                                )

                                rgb_frames[i] = []
                        # episode continues
                        elif (
                            len(self.config.VIDEO_OPTION) > 0
                            or ep_step == self.config.T_EXP - 2
                        ):
                            frame = observations_to_image(
                                observations[i], infos[i], observation_size=300
                            )
                            # Add ego_map_gt to frame
                            ego_map_gt_i = asnumpy(batch["ego_map_gt"][i])  # (2, H, W)
                            ego_map_gt_i = convert_gt2channel_to_gtrgb(ego_map_gt_i)
                            ego_map_gt_i = cv2.resize(ego_map_gt_i, (300, 300))
                            frame = np.concatenate([frame, ego_map_gt_i], axis=1)
                            # Generate ANS specific visualizations
                            environment_layout = asnumpy(
                                ground_truth_states["environment_layout"][i]
                            )  # (2, H, W)
                            visible_occupancy = asnumpy(
                                ground_truth_states["visible_occupancy"][i]
                            )  # (2, H, W)
                            curr_gt_poses = gt_agent_poses_over_time[i]
                            anticipated_occupancy = asnumpy(
                                state_estimates["map_states"][i]
                            )  # (2, H, W)
                            curr_pred_poses = pred_agent_poses_over_time[i]

                            H = frame.shape[0]
                            visible_occupancy_vis = generate_topdown_allocentric_map(
                                environment_layout,
                                visible_occupancy,
                                curr_gt_poses,
                                thresh_explored=ans_cfg.thresh_explored,
                                thresh_obstacle=ans_cfg.thresh_obstacle,
                            )
                            visible_occupancy_vis = cv2.resize(
                                visible_occupancy_vis, (H, H)
                            )
                            anticipated_occupancy_vis = generate_topdown_allocentric_map(
                                environment_layout,
                                anticipated_occupancy,
                                curr_pred_poses,
                                thresh_explored=ans_cfg.thresh_explored,
                                thresh_obstacle=ans_cfg.thresh_obstacle,
                            )
                            anticipated_occupancy_vis = cv2.resize(
                                anticipated_occupancy_vis, (H, H)
                            )
                            anticipated_action_map = generate_topdown_allocentric_map(
                                environment_layout,
                                anticipated_occupancy,
                                curr_pred_poses,
                                zoom=False,
                                thresh_explored=ans_cfg.thresh_explored,
                                thresh_obstacle=ans_cfg.thresh_obstacle,
                            )
                            global_goals = self.ans_net.states["curr_global_goals"]
                            local_goals = self.ans_net.states["curr_local_goals"]
                            if global_goals is not None:
                                cX = int(global_goals[i, 0].item())
                                cY = int(global_goals[i, 1].item())
                                anticipated_action_map = cv2.circle(
                                    anticipated_action_map,
                                    (cX, cY),
                                    10,
                                    (255, 0, 0),
                                    -1,
                                )
                            if local_goals is not None:
                                cX = int(local_goals[i, 0].item())
                                cY = int(local_goals[i, 1].item())
                                anticipated_action_map = cv2.circle(
                                    anticipated_action_map,
                                    (cX, cY),
                                    10,
                                    (0, 255, 255),
                                    -1,
                                )
                            anticipated_action_map = cv2.resize(
                                anticipated_action_map, (H, H)
                            )

                            maps_vis = np.concatenate(
                                [
                                    visible_occupancy_vis,
                                    anticipated_occupancy_vis,
                                    anticipated_action_map,
                                    np.zeros_like(anticipated_action_map),
                                ],
                                axis=1,
                            )
                            frame = np.concatenate([frame, maps_vis], axis=0)

                            rgb_frames[i].append(frame)

                # done-if
            # done-for

        num_frames_per_process = (
            (checkpoint_index + 1)
            * self.config.CHECKPOINT_INTERVAL
            * self.config.T_EXP
            / self.config.RL.PPO.num_global_steps
        )

        if checkpoint_index == 0:
            try:
                eval_ckpt_idx = self.config.EVAL_CKPT_PATH_DIR.split("/")[-1].split(
                    "."
                )[1]
                logger.add_filehandler(
                    f"{self.config.TENSORBOARD_DIR}/results_ckpt_final_{eval_ckpt_idx}.txt"
                )
            except:
                logger.add_filehandler(
                    f"{self.config.TENSORBOARD_DIR}/results_ckpt_{checkpoint_index}.txt"
                )
        else:
            logger.add_filehandler(
                f"{self.config.TENSORBOARD_DIR}/results_ckpt_{checkpoint_index}.txt"
            )

        logger.info(
            f"======= Evaluating over {number_of_eval_episodes} episodes ============="
        )

        logger.info(f"=======> Mapping metrics")
        for k, v in mapping_metrics.items():
            metric_all_times = v.get_metric()
            for kp in sorted(list(metric_all_times.keys())):
                vp = metric_all_times[kp]
                logger.info(f"{k}: {kp},{vp}")
            writer.add_scalar(
                f"mapping_evaluation/{k}", v.get_last_metric(), num_frames_per_process,
            )

        logger.info(f"=======> Pose-estimation metrics")
        for k, v in pose_estimation_metrics.items():
            metric_all_times = v.get_metric()
            for kp in sorted(list(metric_all_times.keys())):
                vp = metric_all_times[kp]
                logger.info(f"{k}: {kp},{vp}")
            writer.add_scalar(
                f"pose_estimation_evaluation/{k}",
                v.get_last_metric(),
                num_frames_per_process,
            )

        if self.config.SAVE_STATISTICS_FLAG:
            # Logging results individually per episode
            per_episode_metrics = {}
            for k, v in mapping_metrics.items():
                per_episode_metrics["mapping/" + k] = v.metric_list
            for k, v in pose_estimation_metrics.items():
                per_episode_metrics["pose_estimation/" + k] = v.metric_list

            per_episode_statistics = []
            for i in range(num_eval_episodes):
                stats = {}
                for k, v in per_episode_metrics.items():
                    stats[k] = {}
                    for t in v.keys():
                        stats[k][t] = v[t][i]
                stats["episode_statistics"] = episode_statistics[i]
                per_episode_statistics.append(stats)

            per_episode_maps = np.stack(episode_visualization_maps, axis=0)
            json_save_path = f"{self.config.TENSORBOARD_DIR}/statistics.json"
            h5py_save_path = f"{self.config.TENSORBOARD_DIR}/visualized_maps.h5"
            json.dump(per_episode_statistics, open(json_save_path, "w"))
            h5file = h5py.File(h5py_save_path, "w")
            h5file.create_dataset("maps", data=per_episode_maps)
            h5file.close()

        self.envs.close()
