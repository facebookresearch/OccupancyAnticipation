#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat_extensions.config.default import get_extended_config
from habitat_extensions.registration import (
    _try_register_exp_task,
    _try_register_explorationdatasetv1,
)
from habitat_extensions import measures, sensors

_try_register_exp_task()
_try_register_explorationdatasetv1()

__all__ = [
    "get_extended_config",
]
