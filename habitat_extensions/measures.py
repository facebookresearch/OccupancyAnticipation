#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type, Union

import math
import numpy as np

from habitat.config import Config
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
)
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Simulator,
)
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat_extensions.geometry_utils import (
    quaternion_xyzw_to_wxyz,
    compute_heading_from_quaternion,
)
from habitat.utils.visualizations import fog_of_war, maps
from habitat.tasks.nav.nav import TopDownMap

cv2 = try_cv2_import()


MAP_THICKNESS_SCALAR: int = 1250


@registry.register_measure
class TopDownMapExp(TopDownMap):
    r"""Top Down Map measure
    """

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "top_down_map_exp"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        # draw source last to avoid overlap
        if self._config.DRAW_SOURCE:
            self._draw_point(episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR)


@registry.register_measure
class GTGlobalMap(Measure):
    r"""GTGlobalMap

    returns the ground-truth global map w.r.t agent's starting pose as origin.
    At t=0, the agent is standing at the center of the map facing upward.
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self.current_episode = None
        self.current_episode_time = 0
        # Map statistics
        self.map_size = config.MAP_SIZE
        self.map_scale = config.MAP_SCALE
        self._num_samples = config.NUM_TOPDOWN_MAP_SAMPLE_POINTS
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        resolution = (self._coordinate_max - self._coordinate_min) / self.map_scale
        self._map_resolution = (int(resolution), int(resolution))

        super().__init__()

    def get_original_map(self):
        top_down_map = maps.get_topdown_map(
            self._sim, self._map_resolution, self._num_samples, False,
        )
        return top_down_map

    def _get_mesh_occupancy(self):
        agent_position = self.current_episode.start_position
        agent_rotation = quaternion_xyzw_to_wxyz(self.current_episode.start_rotation)
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )

        # The map size here represents size around the agent, not infront.
        mrange = int(self.map_size * 1.5 / 2.0)

        # Add extra padding if map range is coordinates go out of bounds
        y_start = a_y - mrange
        y_end = a_y + mrange
        x_start = a_x - mrange
        x_end = a_x + mrange

        x_l_pad, y_l_pad, x_r_pad, y_r_pad = 0, 0, 0, 0

        H, W = self._top_down_map.shape
        if x_start < 0:
            x_l_pad = int(-x_start)
            x_start += x_l_pad
            x_end += x_l_pad
        if x_end >= W:
            x_r_pad = int(x_end - W + 1)
        if y_start < 0:
            y_l_pad = int(-y_start)
            y_start += y_l_pad
            y_end += y_l_pad
        if y_end >= H:
            y_r_pad = int(y_end - H + 1)

        ego_map = np.pad(self._top_down_map, ((y_l_pad, y_r_pad), (x_l_pad, x_r_pad)))
        ego_map = ego_map[y_start : (y_end + 1), x_start : (x_end + 1)]

        if ego_map.shape[0] == 0 or ego_map.shape[1] == 0:
            ego_map = np.zeros((2 * mrange + 1, 2 * mrange + 1), dtype=np.uint8)

        # Rotate to get egocentric map
        # Negative since the value returned is clockwise rotation about Y,
        # but we need anti-clockwise rotation
        agent_heading = -compute_heading_from_quaternion(agent_rotation)
        agent_heading = math.degrees(agent_heading)

        half_size = ego_map.shape[0] // 2
        center = (half_size, half_size)
        M = cv2.getRotationMatrix2D(center, agent_heading, scale=1.0)

        ego_map = cv2.warpAffine(
            ego_map,
            M,
            (ego_map.shape[1], ego_map.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(1,),
        )

        ego_map = ego_map.astype(np.float32)
        mrange = int(self.map_size / 2.0)
        start_coor = half_size - mrange
        end_coor = int(start_coor + self.map_size - 1)
        ego_map = ego_map[start_coor : (end_coor + 1), start_coor : (end_coor + 1)]

        # This map is currently 0 if occupied and 1 if unoccupied. Flip it.
        ego_map = 1.0 - ego_map

        # Flip the x axis because to_grid() flips the conventions
        ego_map = np.flip(ego_map, axis=1)

        # Append explored status in the 2nd channel
        ego_map = np.stack([ego_map, np.ones_like(ego_map)], axis=2)

        return ego_map

    def _process_gt(self, ego_map):
        """
        Remove unnecessary occupied space for out-of-bound regions in the ego_map.

        ego_map - (H, W, 2) with ones and zeros in channel 0 and only ones in channel 1
        """
        occ_map = np.round(ego_map[..., 0] * 255.0).astype(
            np.uint8
        )  # 255 for occupied, 0 for free
        # Flood fill from all four corners of the map
        H, W = occ_map.shape
        cv2.floodFill(occ_map, None, (0, 0), 125)
        cv2.floodFill(occ_map, None, (0, H - 1), 125)
        cv2.floodFill(occ_map, None, (W - 1, 0), 125)
        cv2.floodFill(occ_map, None, (W - 1, H - 1), 125)
        # Occupied regions within the environment
        int_occ_mask = occ_map == 255
        # Expand occupied regions around free space
        free_map = 1 - ego_map[..., 0]
        kernel = np.ones((5, 5), np.float32)
        exp_free_mask = cv2.dilate(free_map, kernel, iterations=2) > 0
        explored_mask = exp_free_mask | int_occ_mask
        """
        NOTE: The above still leaves out objects that are along the edges of
        the walls. Only their edges with free-space will be covered.
        Add all occupied regions that were seen by an oracle agent covering
        the entire environment to fill these in.
        """
        if self._seen_area_map is not None:
            explored_mask = explored_mask | (self._seen_area_map[..., 1] > 0.5)
        ego_map[..., 1] = explored_mask.astype(np.float32)

        return ego_map

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "gt_global_map"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self.current_episode = episode
        self.current_episode_time = 0
        if self._config.ENVIRONMENT_LAYOUTS_PATH != "":
            path = f"{self._config.ENVIRONMENT_LAYOUTS_PATH}/episode_id_{episode.episode_id}.npy"
            self._seen_area_map = np.load(path)
        else:
            self._seen_area_map = None

        self._top_down_map = self.get_original_map().T
        self._metric = self._get_mesh_occupancy()
        # Process the map to remove unnecessary occupied regions
        self._metric = self._process_gt(self._metric)

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        self.current_episode_time += 1
        if self.current_episode_time > 1:
            self._metric = None  # Return only at t=1 to save data transfer costs


@registry.register_measure
class PathLength(Measure):
    r"""Path length so far in the episode.
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._previous_position = None
        self._agent_episode_distance = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "path_length"

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._agent_episode_distance = 0.0
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = self._agent_episode_distance
