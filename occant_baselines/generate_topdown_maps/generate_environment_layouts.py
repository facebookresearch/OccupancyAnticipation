#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import json
import tqdm
import gzip
import argparse
import numpy as np

import torch

import habitat
import habitat_extensions

from habitat.utils.visualizations import maps

from einops import rearrange, asnumpy
from occant_baselines.rl.policy import Mapper
from occant_baselines.config.default import get_config


class DummyRLEnv(habitat.RLEnv):
    def __init__(self, config, dataset=None, env_ind=0):
        super(DummyRLEnv, self).__init__(config, dataset)
        self._env_ind = env_ind

    def get_reward_range(self):
        return -1.0, 1.0

    def get_reward(self, observations):
        return 0.0

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def get_env_ind(self):
        return self._env_ind


def colorize_ego_map(ego_map):
    """
    ego_map - (V, V, 2) array where 1st channel represents prob(occupied space) an
              d 2nd channel represents prob(explored space)
    """
    explored_mask = ego_map[..., 1] > 0.5
    occupied_mask = np.logical_and(ego_map[..., 0] > 0.5, explored_mask)
    free_space_mask = np.logical_and(ego_map[..., 0] <= 0.5, explored_mask)
    unexplored_mask = ego_map[..., 1] <= 0.5

    ego_map_color = np.zeros((*ego_map.shape[:2], 3), np.uint8)

    # White unexplored map
    ego_map_color[unexplored_mask, 0] = 255
    ego_map_color[unexplored_mask, 1] = 255
    ego_map_color[unexplored_mask, 2] = 255

    # Blue occupied map
    ego_map_color[occupied_mask, 0] = 0
    ego_map_color[occupied_mask, 1] = 0
    ego_map_color[occupied_mask, 2] = 255

    # Green free space map
    ego_map_color[free_space_mask, 0] = 0
    ego_map_color[free_space_mask, 1] = 255
    ego_map_color[free_space_mask, 2] = 0

    return ego_map_color


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass


def main(args):

    config = get_config()

    mapper_config = config.RL.ANS.MAPPER
    mapper_config.defrost()
    mapper_config.map_size = 65
    mapper_config.map_scale = 0.05
    mapper_config.freeze()

    mapper = Mapper(mapper_config, None)

    M = args.global_map_size

    config_path = args.config_path
    save_dir = args.save_dir
    safe_mkdir(save_dir)

    config = habitat_extensions.get_extended_config(config_path)

    dataset_path = config.DATASET.DATA_PATH.replace("{split}", config.DATASET.SPLIT)
    with gzip.open(dataset_path, "rt") as fp:
        dataset = json.load(fp)

    num_episodes = len(dataset["episodes"])

    env = DummyRLEnv(config=config)
    env.seed(1234)
    device = torch.device("cuda:0")

    for i in tqdm.tqdm(range(num_episodes)):
        _ = env.reset()

        # Initialize a global map for the episode
        global_map = torch.zeros(1, 2, M, M).to(device)

        grid_size = config.TASK.GT_EGO_MAP.MAP_SCALE
        coordinate_max = maps.COORDINATE_MAX
        coordinate_min = maps.COORDINATE_MIN
        resolution = (coordinate_max - coordinate_min) / grid_size
        grid_resolution = (int(resolution), int(resolution))

        top_down_map = maps.get_topdown_map(env.habitat_env.sim, grid_resolution, 20000)

        map_w, map_h = top_down_map.shape

        intervals = (max(int(1.0 / grid_size), 1), max(int(1.0 / grid_size), 1))
        x_vals = np.arange(0, map_w, intervals[0], dtype=int)
        y_vals = np.arange(0, map_h, intervals[1], dtype=int)
        coors = np.stack(np.meshgrid(x_vals, y_vals), axis=2)  # (H, W, 2)
        coors = coors.reshape(-1, 2)  # (H*W, 2)
        map_vals = top_down_map[coors[:, 0], coors[:, 1]]
        valid_coors = coors[map_vals > 0]

        real_x_vals = coordinate_max - valid_coors[:, 0] * grid_size
        real_z_vals = coordinate_min + valid_coors[:, 1] * grid_size
        start_y = env.habitat_env.sim.get_agent_state().position[1]

        for i in range(real_x_vals.shape[0]):
            for theta in np.arange(-np.pi, np.pi, np.pi / 3.0):
                position = [
                    real_x_vals[i].item(),
                    start_y.item(),
                    real_z_vals[i].item(),
                ]
                rotation = [
                    0.0,
                    np.sin(theta / 2).item(),
                    0.0,
                    np.cos(theta / 2).item(),
                ]

                sim_obs = env.habitat_env.sim.get_observations_at(
                    position, rotation, keep_agent_at_new_pose=True
                )
                episode = env.habitat_env.current_episode
                obs = env.habitat_env.task.sensor_suite.get_observations(
                    observations=sim_obs, episode=episode, task=env.habitat_env.task
                )
                ego_map_gt = torch.Tensor(obs["ego_map_gt"]).to(device)
                ego_map_gt = rearrange(ego_map_gt, "h w c -> () c h w")
                pose_gt = torch.Tensor(obs["pose_gt"]).unsqueeze(0).to(device)
                global_map = mapper.ext_register_map(global_map, ego_map_gt, pose_gt)

        # Save data
        global_map_np = asnumpy(rearrange(global_map, "b c h w -> b h w c")[0])
        episode_id = env.habitat_env.current_episode.episode_id
        np.save(os.path.join(save_dir, f"episode_id_{episode_id}.npy"), global_map_np)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--global-map-size", type=int, required=True)

    args = parser.parse_args()

    main(args)
