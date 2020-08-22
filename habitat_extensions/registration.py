#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.core.dataset import Dataset


def _try_register_exp_task():
    try:
        from habitat_extensions.exploration_task import ExplorationTask

        has_exptask = True
    except ImportError as e:
        has_exptask = False
        exptask_import_error = e

    if has_exptask:
        from habitat_extensions.exploration_task import ExplorationTask
    else:

        @registry.register_task(name="Exp-v0")
        class ExplorationTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise exptask_import_error


def _try_register_explorationdatasetv1():
    try:
        from habitat_extensions.exploration_dataset import ExplorationDatasetV1

        has_exploration = True
    except ImportError as e:
        has_exploration = False
        exploration_import_error = e

    if has_exploration:
        from habitat_extensions.exploration_dataset import ExplorationDatasetV1
    else:

        @registry.register_dataset(name="Exploration-v1")
        class ExplorationDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise exploration_import_error
