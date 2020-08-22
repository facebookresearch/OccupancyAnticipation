#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import habitat

from occant_baselines.models.unet import (
    UNetEncoder,
    UNetDecoder,
    MiniUNetEncoder,
    LearnedRGBProjection,
    MergeMultimodal,
    ResNetRGBEncoder,
)


def test_unet_encoder():
    n_channels = 2
    nsf = 16
    bs = 16
    V = 128

    net = UNetEncoder(n_channels, nsf)

    x = torch.randn(bs, n_channels, V, V)
    y = net(x)

    assert True


def test_unet_decoder():
    n_channels = 2
    n_classes = 2
    nsf = 16
    bs = 16
    V = 128

    unet_encoder = UNetEncoder(n_channels, nsf)
    unet_decoder = UNetDecoder(n_classes, nsf)

    x = torch.randn(bs, n_channels, V, V)
    y = unet_encoder(x)
    z = unet_decoder(y)

    assert True


def test_mini_unet_encoder():
    n_channels = 192
    feat_size = 512
    bs = 16
    V = 64

    mini_unet_encoder = MiniUNetEncoder(n_channels, feat_size)

    x = torch.randn(bs, n_channels, V, V)
    y = mini_unet_encoder(x)

    assert True


def test_learned_rgb_projection():
    infeats = 192
    bs = 16
    V = 64

    net = LearnedRGBProjection(infeats=infeats)

    x = torch.randn(bs, infeats, V, V)
    y = net(x)

    assert True


def test_merge_multimodal():
    nfeats = 512
    bs = 16
    V = 64
    nmodes = 2

    net = MergeMultimodal(nfeats, nmodes=nmodes)

    x1 = torch.randn(bs, nfeats, V, V)
    x2 = torch.randn(bs, nfeats, V, V)

    y = net(x1, x2)

    assert True


def test_resnet_rgb_encoder():
    resnet_types = ["resnet18", "resnet50"]

    bs = 16
    V = 128
    n_channels = 3

    x = torch.randn(bs, n_channels, V, V)

    for resnet_type in resnet_types:
        net = ResNetRGBEncoder(resnet_type=resnet_type)
        y = net(x)

    assert True


if __name__ == "__main__":

    test_unet_encoder()
    test_unet_decoder()
    test_mini_unet_encoder()
    test_learned_rgb_projection()
    test_merge_multimodal()
    test_resnet_rgb_encoder()
