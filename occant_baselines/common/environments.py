#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Type

import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_env(name="ExpRLEnv")
class ExpRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG

        self._previous_action = None
        self._episode_distance_covered = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            -1.0,
            +1.0,
        )

    def get_reward(self, observations):
        reward = 0
        return reward

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        metrics = self.habitat_env.get_metrics()
        episode_statistics = {
            "episode_id": self.habitat_env.current_episode.episode_id,
            "scene_id": self.habitat_env.current_episode.scene_id,
        }
        metrics["episode_statistics"] = episode_statistics
        return metrics
