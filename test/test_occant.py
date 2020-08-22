#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import habitat

from occant_baselines.models.occant import (
    ANSRGB,
    ANSDepth,
    OccAntRGB,
    OccAntDepth,
    OccAntRGBD,
    OccAntGroundTruth,
    OccupancyAnticipator,
)

from occant_baselines.config.default import get_config


TASK_CONFIG = "occant_baselines/config/ppo_exploration.yaml"


def test_ans_rgb():
    bs = 16
    V = 128

    cfg = get_config(TASK_CONFIG)
    occ_cfg = cfg.RL.ANS.OCCUPANCY_ANTICIPATOR

    net = ANSRGB(occ_cfg)

    batch = {
        "rgb": torch.rand(bs, 3, V, V),
        "depth": torch.rand(bs, 1, V, V),
        "ego_map_gt": torch.rand(bs, 2, V, V),
        "ego_map_gt_anticipated": torch.rand(bs, 2, V, V),
    }

    y = net(batch)

    assert "occ_estimate" in y.keys()


def test_ans_depth():
    bs = 16
    V = 128

    cfg = get_config(TASK_CONFIG)
    occ_cfg = cfg.RL.ANS.OCCUPANCY_ANTICIPATOR

    net = ANSDepth(occ_cfg)

    batch = {
        "rgb": torch.rand(bs, 3, V, V),
        "depth": torch.rand(bs, 1, V, V),
        "ego_map_gt": torch.rand(bs, 2, V, V),
        "ego_map_gt_anticipated": torch.rand(bs, 2, V, V),
    }

    y = net(batch)

    assert "occ_estimate" in y.keys()
    assert torch.all(y["occ_estimate"] == batch["ego_map_gt"])


def test_occant_rgb():
    bs = 16
    V = 128

    cfg = get_config(TASK_CONFIG)
    occ_cfg = cfg.RL.ANS.OCCUPANCY_ANTICIPATOR

    net = OccAntRGB(occ_cfg)

    batch = {
        "rgb": torch.rand(bs, 3, V, V),
        "depth": torch.rand(bs, 1, V, V),
        "ego_map_gt": torch.rand(bs, 2, V, V),
        "ego_map_gt_anticipated": torch.rand(bs, 2, V, V),
    }

    y = net(batch)

    assert "occ_estimate" in y.keys()
    assert "depth_proj_estimate" in y.keys()


def test_occant_depth():
    bs = 16
    V = 128

    cfg = get_config(TASK_CONFIG)
    occ_cfg = cfg.RL.ANS.OCCUPANCY_ANTICIPATOR

    net = OccAntDepth(occ_cfg)

    batch = {
        "rgb": torch.rand(bs, 3, V, V),
        "depth": torch.rand(bs, 1, V, V),
        "ego_map_gt": torch.rand(bs, 2, V, V),
        "ego_map_gt_anticipated": torch.rand(bs, 2, V, V),
    }

    y = net(batch)

    assert "occ_estimate" in y.keys()


def test_occant_rgbd():
    bs = 16
    V = 128

    cfg = get_config(TASK_CONFIG)
    occ_cfg = cfg.RL.ANS.OCCUPANCY_ANTICIPATOR

    net = OccAntRGBD(occ_cfg)

    batch = {
        "rgb": torch.rand(bs, 3, V, V),
        "depth": torch.rand(bs, 1, V, V),
        "ego_map_gt": torch.rand(bs, 2, V, V),
        "ego_map_gt_anticipated": torch.rand(bs, 2, V, V),
    }

    y = net(batch)

    assert "occ_estimate" in y.keys()


def test_occant_ground_truth():
    bs = 16
    V = 128

    cfg = get_config(TASK_CONFIG)
    occ_cfg = cfg.RL.ANS.OCCUPANCY_ANTICIPATOR

    net = OccAntGroundTruth(occ_cfg)

    batch = {
        "rgb": torch.rand(bs, 3, V, V),
        "depth": torch.rand(bs, 1, V, V),
        "ego_map_gt": torch.rand(bs, 2, V, V),
        "ego_map_gt_anticipated": torch.rand(bs, 2, V, V),
    }

    y = net(batch)

    assert "occ_estimate" in y.keys()
    assert torch.all(batch["ego_map_gt_anticipated"] == y["occ_estimate"])


def test_occupancy_anticipator():
    model_types = [
        "ans_rgb",
        "ans_depth",
        "occant_rgb",
        "occant_depth",
        "occant_rgbd",
        "occant_ground_truth",
    ]

    bs = 16
    V = 128

    cfg = get_config(TASK_CONFIG)
    occ_cfg = cfg.RL.ANS.OCCUPANCY_ANTICIPATOR

    batch = {
        "rgb": torch.rand(bs, 3, V, V),
        "depth": torch.rand(bs, 1, V, V),
        "ego_map_gt": torch.rand(bs, 2, V, V),
        "ego_map_gt_anticipated": torch.rand(bs, 2, V, V),
    }

    for model_type in model_types:
        occ_cfg_copy = occ_cfg.clone()
        occ_cfg_copy.defrost()
        occ_cfg_copy.type = model_type
        occ_cfg_copy.freeze()

        net = OccAntGroundTruth(occ_cfg_copy)

        y = net(batch)

        assert "occ_estimate" in y.keys()


if __name__ == "__main__":
    test_ans_rgb()
    test_ans_depth()
    test_occant_rgb()
    test_occant_depth()
    test_occant_rgbd()
    test_occant_ground_truth()
    test_occupancy_anticipator()
