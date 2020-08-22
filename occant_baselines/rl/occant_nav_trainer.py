#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import time
import copy
import random

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import math
import numpy as np
import torch
import torch.nn.functional as F

from habitat import Config, logger
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from occant_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import batch_obs
from occant_baselines.rl.ans import ActiveNeuralSLAMNavigator
from occant_baselines.rl.policy_utils import OccupancyAnticipationWrapper
from occant_utils.common import add_pose
from occant_utils.metrics import Metric
from occant_baselines.models.mapnet import DepthProjectionNet
from occant_baselines.models.occant import OccupancyAnticipator
from einops import rearrange


@baseline_registry.register_trainer(name="occant_nav")
class OccAntNavTrainer(BaseRLTrainer):
    r"""Trainer class for Occupancy Anticipated based navigation.
    This only evaluates the transfer performance of a pre-trained model.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        if config is not None:
            self._synchronize_configs(config)
        super().__init__(config)

        # Set pytorch random seed for initialization
        random.seed(config.PYT_RANDOM_SEED)
        np.random.seed(config.PYT_RANDOM_SEED)
        torch.manual_seed(config.PYT_RANDOM_SEED)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.mapper = None
        self.local_actor_critic = None
        self.ans_net = None
        self.planner = None
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
        self.ans_net = ActiveNeuralSLAMNavigator(ans_cfg, occupancy_model)
        self.mapper = self.ans_net.mapper
        self.local_actor_critic = self.ans_net.local_policy
        # Create depth projection model to estimate visible occupancy
        self.depth_projection_net = DepthProjectionNet(
            ans_cfg.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION
        )
        # Set to device
        self.mapper.to(self.device)
        self.local_actor_critic.to(self.device)
        self.depth_projection_net.to(self.device)

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

    def _remap_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Converts actions of exploration agent to actions for navigation.
        Remapping:
            0 -> 1 (forward)
            1 -> 2 (turn left)
            2 -> 3 (turn right)
            3 -> 0 (stop)
        """
        actions_rmp = torch.remainder(actions + 1, 4).long()
        return actions_rmp

    def _prepare_batch(self, observations, device=None, actions=None):
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
        # Add previous action to batch as well
        batch["prev_actions"] = self.prev_actions
        # Add a rough pose estimate if GT pose is not available
        if "pose" not in batch:
            if self.prev_batch is None:
                # Set initial pose estimate to zero
                batch["pose"] = torch.zeros(self.envs.num_envs, 3).to(self.device)
            else:
                actions_delta = self._convert_actions_to_delta(self.prev_actions)
                batch["pose"] = add_pose(self.prev_batch["pose"], actions_delta)

        return batch

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        raise NotImplementedError

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

        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg, ans_cfg)

        # Convert the state_dict of mapper_agent to mapper
        mapper_dict = {
            k.replace("mapper.", ""): v
            for k, v in ckpt_dict["mapper_state_dict"].items()
        }
        # Converting the state_dict of local_agent to just the local_policy.
        local_dict = {
            k.replace("actor_critic.", ""): v
            for k, v in ckpt_dict["local_state_dict"].items()
        }
        # Strict = False is set to ignore to handle the case where
        # pose_estimator is not required.
        self.mapper.load_state_dict(mapper_dict, strict=False)
        self.local_actor_critic.load_state_dict(local_dict)

        # Set models to evaluation
        self.mapper.eval()
        self.local_actor_critic.eval()

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

        M = ans_cfg.overall_map_size
        V = ans_cfg.MAPPER.map_size
        s = ans_cfg.MAPPER.map_scale
        imH, imW = ans_cfg.image_scale_hw

        assert (
            self.envs.num_envs == 1
        ), "Number of environments needs to be 1 for evaluation"

        # Define metric accumulators
        # Navigation metrics
        navigation_metrics = {
            "success_rate": Metric(),
            "spl": Metric(),
            "distance_to_goal": Metric(),
            "time": Metric(),
            "softspl": Metric(),
        }
        per_difficulty_navigation_metrics = {
            "easy": {
                "success_rate": Metric(),
                "spl": Metric(),
                "distance_to_goal": Metric(),
                "time": Metric(),
                "softspl": Metric(),
            },
            "medium": {
                "success_rate": Metric(),
                "spl": Metric(),
                "distance_to_goal": Metric(),
                "time": Metric(),
                "softspl": Metric(),
            },
            "hard": {
                "success_rate": Metric(),
                "spl": Metric(),
                "distance_to_goal": Metric(),
                "time": Metric(),
                "softspl": Metric(),
            },
        }

        times_per_episode = deque()
        times_per_step = deque()
        # Define a simple function to return episode difficulty based on
        # the geodesic distance
        def classify_difficulty(gd):
            if gd < 5.0:
                return "easy"
            elif gd < 10.0:
                return "medium"
            else:
                return "hard"

        eval_start_time = time.time()
        # Reset environments only for the very first batch
        observations = self.envs.reset()
        for ep in range(number_of_eval_episodes):
            # ============================== Reset agent ==============================
            # Reset agent states
            state_estimates = {
                "pose_estimates": torch.zeros(self.envs.num_envs, 3).to(self.device),
                "map_states": torch.zeros(self.envs.num_envs, 2, M, M).to(self.device),
                "recurrent_hidden_states": torch.zeros(
                    1, self.envs.num_envs, ans_cfg.LOCAL_POLICY.hidden_size
                ).to(self.device),
            }
            # Reset ANS states
            self.ans_net.reset()
            self.not_done_masks = torch.zeros(self.envs.num_envs, 1, device=self.device)
            self.prev_actions = torch.zeros(self.envs.num_envs, 1, device=self.device)
            self.prev_batch = None
            self.ep_time = torch.zeros(self.envs.num_envs, 1, device=self.device)
            # =========================== Episode loop ================================
            ep_start_time = time.time()
            current_episodes = self.envs.current_episodes()
            for ep_step in range(self.config.T_MAX):
                step_start_time = time.time()
                # ============================ Action step ============================
                batch = self._prepare_batch(observations)
                if self.prev_batch is None:
                    self.prev_batch = copy.deepcopy(batch)

                prev_pose_estimates = state_estimates["pose_estimates"]
                with torch.no_grad():
                    (
                        _,
                        _,
                        mapper_outputs,
                        local_policy_outputs,
                        state_estimates,
                    ) = self.ans_net.act(
                        batch,
                        self.prev_batch,
                        state_estimates,
                        self.ep_time,
                        self.not_done_masks,
                        deterministic=ans_cfg.LOCAL_POLICY.deterministic_flag,
                    )
                    actions = local_policy_outputs["actions"]
                    # Make masks not done till reset (end of episode)
                    self.not_done_masks = torch.ones(
                        self.envs.num_envs, 1, device=self.device
                    )
                    self.prev_actions.copy_(actions)

                if ep_step == 0:
                    state_estimates["pose_estimates"].copy_(prev_pose_estimates)

                self.ep_time += 1
                # Update prev batch
                for k, v in batch.items():
                    self.prev_batch[k].copy_(v)

                # Remap actions from exploration to navigation agent.
                actions_rmp = self._remap_actions(actions)

                # =========================== Environment step ========================
                outputs = self.envs.step([a[0].item() for a in actions_rmp])

                observations, _, dones, infos = [list(x) for x in zip(*outputs)]

                times_per_step.append(time.time() - step_start_time)
                # ============================ Process metrics ========================
                if dones[0]:
                    times_per_episode.append(time.time() - ep_start_time)
                    mins_per_episode = np.mean(times_per_episode).item() / 60.0
                    eta_completion = mins_per_episode * (
                        number_of_eval_episodes - ep - 1
                    )
                    secs_per_step = np.mean(times_per_step).item()
                    for i in range(self.envs.num_envs):
                        episode_id = int(current_episodes[i].episode_id)
                        curr_metrics = {
                            "spl": infos[i]["spl"],
                            "softspl": infos[i]["softspl"],
                            "success_rate": infos[i]["success"],
                            "time": ep_step + 1,
                            "distance_to_goal": infos[i]["distance_to_goal"],
                        }
                        # Estimate difficulty of episode
                        episode_difficulty = classify_difficulty(
                            current_episodes[i].info["geodesic_distance"]
                        )
                        for k, v in curr_metrics.items():
                            navigation_metrics[k].update(v, 1.0)
                            per_difficulty_navigation_metrics[episode_difficulty][
                                k
                            ].update(v, 1.0)

                        logger.info(f"====> {ep}/{number_of_eval_episodes} done")
                        for k, v in curr_metrics.items():
                            logger.info(f"{k:25s} : {v:10.3f}")
                        logger.info("{:25s} : {:10d}".format("episode_id", episode_id))
                        logger.info(f"Time per episode: {mins_per_episode:.3f} mins")
                        logger.info(f"Time per step: {secs_per_step:.3f} secs")
                        logger.info(f"ETA: {eta_completion:.3f} mins")

                    # For navigation, terminate episode loop when dones is called
                    break
            # done-for

        if checkpoint_index == 0:
            try:
                checkpoint_index = self.config.EVAL_CKPT_PATH_DIR.split("/")[-1].split(
                    "."
                )[1]
                logger.add_filehandler(
                    f"{self.config.TENSORBOARD_DIR}/navigation_results_ckpt_final_{eval_ckpt_idx}.txt"
                )
            except:
                logger.add_filehandler(
                    f"{self.config.TENSORBOARD_DIR}/navigation_results_ckpt_{checkpoint_index}.txt"
                )
        else:
            logger.add_filehandler(
                f"{self.config.TENSORBOARD_DIR}/navigation_results_ckpt_{checkpoint_index}.txt"
            )

        logger.info(
            f"======= Evaluating over {number_of_eval_episodes} episodes ============="
        )

        logger.info(f"=======> Navigation metrics")
        for k, v in navigation_metrics.items():
            logger.info(f"{k}: {v.get_metric():.3f}")
            writer.add_scalar(f"navigation/{k}", v.get_metric(), checkpoint_index)

        for diff, diff_metrics in per_difficulty_navigation_metrics.items():
            logger.info(f"=============== {diff:^10s} metrics ==============")
            for k, v in diff_metrics.items():
                logger.info(f"{k}: {v.get_metric():.3f}")
                writer.add_scalar(
                    f"{diff}_navigation/{k}", v.get_metric(), checkpoint_index
                )

        total_eval_time = (time.time() - eval_start_time) / 60.0
        logger.info(f"Total evaluation time: {total_eval_time:.3f} mins")
        self.envs.close()
