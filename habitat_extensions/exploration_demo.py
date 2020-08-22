#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import pdb
import math
import numpy as np

import habitat

from habitat_extensions.config import get_extended_config
from habitat_extensions.utils import observations_to_image


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


config = get_extended_config("habitat_extensions/config/exploration_gibson.yaml")

env = DummyRLEnv(config=config)
env.seed(1234)

obs = env.reset()

"""
Action space:
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STOP = 3
"""

action = 0
count = 0
H, W = 100, 100
while True:
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

    ego_map = cv2.resize(colorize_ego_map(obs["ego_map_gt"]), (W, H))
    ego_map_anticipated = cv2.resize(
        colorize_ego_map(obs["ego_map_gt_anticipated"]), (W, H)
    )
    if info["gt_global_map"] is not None:
        gt_global_map = colorize_ego_map(info["gt_global_map"])
        gt_global_map = cv2.resize(gt_global_map, (W, H))

    vis_image_top = observations_to_image(obs, info, W)
    vis_image_bot = np.concatenate(
        [ego_map, ego_map_anticipated, gt_global_map], axis=1
    )
    vis_image = np.concatenate([vis_image_top, vis_image_bot], axis=0)

    cv2.imshow("Image", np.flip(vis_image, axis=2))
    key = cv2.waitKey(0)

    if "w" == chr(key & 255):
        action = 0
    elif "a" == chr(key & 255):
        action = 1
    elif "d" == chr(key & 255):
        action = 2
    else:
        break
