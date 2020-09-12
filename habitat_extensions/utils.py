#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

from scipy import stats
from typing import Dict, List, Optional, Tuple

from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import draw_collision

cv2 = try_cv2_import()


def truncated_normal_noise_distr(mu, var, width):
    """
    Returns a truncated normal distribution.
    mu - mean of gaussian
    var - variance of gaussian
    width - how much of the normal to sample on either sides of 0
    """
    lower = -width
    upper = width
    sigma = math.sqrt(var)

    X = stats.truncnorm(lower, upper, loc=mu, scale=sigma)

    return X


def observations_to_image(
    observation: Dict, info: Dict, observation_size: Optional[int] = None
) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    if "rgb" in observation:
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()
        if observation_size is None:
            observation_size = observation["rgb"].shape[0]
        else:
            scale = observation_size / rgb.shape[0]
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale)
        egocentric_view.append(rgb)

    # draw depth map if observation has depth info
    if "depth" in observation:
        depth_map = observation["depth"].squeeze() * 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()
        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        if observation_size is None:
            observation_size = depth_map.shape[0]
        else:
            scale = observation_size / depth_map.shape[0]
            depth_map = cv2.resize(depth_map, None, fx=scale, fy=scale)
        egocentric_view.append(depth_map)

    # add image goal if observation has image_goal info
    if "imagegoal" in observation:
        observation_size = observation["imagegoal"].shape[0]
        rgb = observation["imagegoal"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view.append(rgb)

    assert len(egocentric_view) > 0, "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    if "top_down_map_exp" in info:
        info["top_down_map"] = info["top_down_map_exp"]

    if "top_down_map" in info:
        top_down_map = topdown_to_image(info["top_down_map"])
        scale = observation_size / top_down_map.shape[0]
        top_down_map = cv2.resize(top_down_map, None, fx=scale, fy=scale)
        frame = np.concatenate((egocentric_view, top_down_map), axis=1)

    return frame


def topdown_to_image(topdown_info: np.ndarray) -> np.ndarray:
    r"""Convert topdown map to an RGB image.
    """
    top_down_map = topdown_info["map"]
    fog_of_war_mask = topdown_info["fog_of_war_mask"]
    top_down_map = maps.colorize_topdown_map(top_down_map, fog_of_war_mask)
    map_agent_pos = topdown_info["agent_map_coord"]

    # Add zero padding
    min_map_size = 200
    if top_down_map.shape[0] != top_down_map.shape[1]:
        H = top_down_map.shape[0]
        W = top_down_map.shape[1]
        if H > W:
            pad_value = (H - W) // 2
            padding = ((0, 0), (pad_value, pad_value), (0, 0))
            map_agent_pos = (map_agent_pos[0], map_agent_pos[1] + pad_value)
        else:
            pad_value = (W - H) // 2
            padding = ((pad_value, pad_value), (0, 0), (0, 0))
            map_agent_pos = (map_agent_pos[0] + pad_value, map_agent_pos[1])
        top_down_map = np.pad(
            top_down_map, padding, mode="constant", constant_values=255
        )

    if top_down_map.shape[0] < min_map_size:
        H, W = top_down_map.shape[:2]
        top_down_map = cv2.resize(top_down_map, (min_map_size, min_map_size))
        map_agent_pos = (
            int(map_agent_pos[0] * min_map_size // H),
            int(map_agent_pos[1] * min_map_size // W),
        )
    top_down_map = maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=topdown_info["agent_angle"],
        agent_radius_px=top_down_map.shape[0] // 16,
    )

    return top_down_map
