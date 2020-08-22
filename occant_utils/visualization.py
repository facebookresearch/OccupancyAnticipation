#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import math
import numpy as np


EXPLORED_COLOR = (220, 183, 226)
GT_OBSTACLE_COLOR = (204, 204, 204)
CORRECT_OBSTACLE_COLOR = (51, 102, 0)
FALSE_OBSTACLE_COLOR = (102, 204, 0)
TRAJECTORY_COLOR = (0, 0, 0)


def draw_triangle(image, position, theta, radius=30, color=(0, 0, 0)):
    """
    position - center position of the triangle in image coordinates
    theta - direction in radians measured from -Y to X. for example, 0 rads is facing upward.
    """
    x, y = position

    r_1 = radius
    theta_1 = theta
    coor_1 = (x + r_1 * math.sin(theta_1), y - r_1 * math.cos(theta_1))

    r_2 = 0.5 * radius
    theta_2 = theta + math.radians(130)
    coor_2 = (x + r_2 * math.sin(theta_2), y - r_2 * math.cos(theta_2))

    r_3 = 0.5 * radius
    theta_3 = theta - math.radians(130)
    coor_3 = (x + r_3 * math.sin(theta_3), y - r_3 * math.cos(theta_3))

    triangle_contour = np.array([coor_1, coor_2, coor_3])

    image = cv2.drawContours(image, [triangle_contour.astype(np.int32)], 0, color, -1)

    return image


def generate_topdown_allocentric_map(
    global_map,
    pred_coverage_map,
    agent_positions,
    thresh_explored=0.6,
    thresh_obstacle=0.6,
    zoom=True,
    draw_trajectory=False,
    draw_agent=True,
):
    """
    Inputs:
        global_map        - (2, H, W) numpy array
        pred_coverage_map - (2, H, W) numpy array
        agent_positions   - (T, 3) numpy array --- (x, y, theta) map pose
    """
    H, W = global_map.shape[1:]
    colored_map = np.ones((H, W, 3), np.uint8) * 255
    global_obstacle_map = (global_map[0] == 1) & (global_map[1] == 1)

    # First show explored regions
    explored_map = pred_coverage_map[1] >= thresh_explored
    colored_map[explored_map, :] = np.array(EXPLORED_COLOR)

    # Show GT obstacles in explored regions
    gt_obstacles_in_explored_map = global_obstacle_map & explored_map
    colored_map[gt_obstacles_in_explored_map, :] = np.array(GT_OBSTACLE_COLOR)

    # Show correctly predicted obstacles in dark green
    pred_obstacles = (pred_coverage_map[0] >= thresh_obstacle) & explored_map
    correct_pred_obstacles = pred_obstacles & gt_obstacles_in_explored_map
    colored_map[correct_pred_obstacles, :] = np.array(CORRECT_OBSTACLE_COLOR)

    # Show in-correctly predicted obstacles in light green
    false_pred_obstacles = pred_obstacles & ~gt_obstacles_in_explored_map
    colored_map[false_pred_obstacles, :] = np.array(FALSE_OBSTACLE_COLOR)

    # Draw trajectory
    if draw_trajectory:
        agent_positions_subsampled = [
            agent_positions[i] for i in range(0, len(agent_positions), 20)
        ]
        for pose in agent_positions_subsampled:
            x, y = pose[:2]
            colored_map = cv2.circle(colored_map, (x, y), 2, TRAJECTORY_COLOR, -1)

    if draw_agent:
        colored_map = draw_triangle(
            colored_map,
            agent_positions[-1][:2].tolist(),
            agent_positions[-1][2].item(),
            radius=15,
            color=TRAJECTORY_COLOR,
        )

    if zoom:
        # Add an initial padding to ensure a non-zero boundary.
        global_occ_map = np.pad(global_map[0], 5, mode="constant", constant_values=1.0)
        # Zoom into the map based on extents in global_map
        global_map_ysum = (1 - global_occ_map).sum(axis=0)  # (W, )
        global_map_xsum = (1 - global_occ_map).sum(axis=1)  # (H, )
        x_start = W
        x_end = 0
        y_start = H
        y_end = 0
        for i in range(W - 1):
            if global_map_ysum[i] == 0 and global_map_ysum[i + 1] > 0:
                x_start = min(x_start, i)
            if global_map_ysum[i] > 0 and global_map_ysum[i + 1] == 0:
                x_end = max(x_end, i)

        for i in range(H - 1):
            if global_map_xsum[i] == 0 and global_map_xsum[i + 1] > 0:
                y_start = min(y_start, i)
            if global_map_xsum[i] > 0 and global_map_xsum[i + 1] == 0:
                y_end = max(y_end, i)

        # Remove the initial padding
        x_start = max(x_start - 5, 0)
        y_start = max(y_start - 5, 0)
        x_end = max(x_end - 5, 0)
        y_end = max(y_end - 5, 0)

        # Some padding
        x_start = max(x_start - 5, 0)
        x_end = min(x_end + 5, W - 1)
        x_width = x_end - x_start + 1
        y_start = max(y_start - 5, 0)
        y_end = min(y_end + 5, H - 1)
        y_width = y_end - y_start + 1
        max_width = max(x_width, y_width)

        colored_map = colored_map[
            y_start : (y_start + max_width), x_start : (x_start + max_width)
        ]

    return colored_map
