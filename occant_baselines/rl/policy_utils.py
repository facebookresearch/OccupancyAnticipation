#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from occant_utils.common import (
    padded_resize,
    process_image,
    init,
)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class OccupancyAnticipationWrapper(nn.Module):
    def __init__(self, model, V, input_hw):
        super().__init__()
        self.main = model
        self.V = V
        self.input_hw = input_hw
        self.keys_to_interpolate = [
            "ego_map_hat",
            "occ_estimate",
            "depth_proj_estimate",  # specific to RGB Model V2
        ]

    def forward(self, x):
        x["rgb"] = padded_resize(x["rgb"], self.input_hw[0])
        if "ego_map_gt" in x:
            x["ego_map_gt"] = F.interpolate(x["ego_map_gt"], size=self.input_hw)
        x_full = self.main(x)
        for k in x_full.keys():
            if k in self.keys_to_interpolate:
                x_full[k] = F.interpolate(
                    x_full[k], size=(self.V, self.V), mode="bilinear"
                )
        return x_full


class BucketingEmbedding(nn.Module):
    def __init__(self, min_val, max_val, count, dim, use_log_scale=False):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.count = count
        self.dim = dim
        self.use_log_scale = use_log_scale
        if self.use_log_scale:
            self.min_val = torch.log2(torch.Tensor([self.min_val])).item()
            self.max_val = torch.log2(torch.Tensor([self.max_val])).item()
        self.main = nn.Embedding(count, dim)

    def forward(self, x):
        """
        x - (bs, ) values
        """
        if self.use_log_scale:
            x = torch.log2(x)
        x = self.count * (x - self.min_val) / (self.max_val - self.min_val)
        x = torch.clamp(x, 0, self.count - 1).long()
        return self.main(x)

    def get_class(self, x):
        """
        x - (bs, ) values
        """
        if self.use_log_scale:
            x = torch.log2(x)
        x = self.count * (x - self.min_val) / (self.max_val - self.min_val)
        x = torch.clamp(x, 0, self.count - 1).long()
        return x


class PoseEstimator(nn.Module):
    def __init__(
        self,
        ego_input_size,
        inputs,
        n_pose_layers=1,
        n_ensemble_layers=1,
        input_shape=(90, 160),
    ):
        """Assumes that map inputs are input_size x input_size tensors.
        RGB inputs are 90x160 in size. Depth inputs are 90x160 in size.
        """
        super().__init__()
        self.inputs = inputs
        assert len(inputs) > 0
        feat_size = 0
        imH, imW = input_shape
        if "rgb" in inputs:
            (
                rgb_encoder,
                rgb_projector,
                rgb_predictor,
            ) = self._get_simple_pose_predictor(6, (imH, imW), n_pose_layers)
            self.rgb_encoder = rgb_encoder
            self.rgb_projector = rgb_projector
            self.rgb_predictor = rgb_predictor
            feat_size += 1024
        if "depth" in inputs:
            (
                depth_encoder,
                depth_projector,
                depth_predictor,
            ) = self._get_simple_pose_predictor(2, (imH, imW), n_pose_layers)
            feat_size += 1024
            self.depth_encoder = depth_encoder
            self.depth_projector = depth_projector
            self.depth_predictor = depth_predictor
        if "ego_map" in inputs:
            V = ego_input_size
            (
                ego_map_encoder,
                ego_map_projector,
                ego_map_predictor,
            ) = self._get_simple_pose_predictor(4, (V, V), n_pose_layers)
            feat_size += 1024
            self.ego_map_encoder = ego_map_encoder
            self.ego_map_projector = ego_map_projector
            self.ego_map_predictor = ego_map_predictor
        if len(self.inputs) > 1:
            self.ensemble_attention = self._get_ensemble_attention(
                n_ensemble_layers, feat_size, len(self.inputs)
            )

    def _get_simple_pose_predictor(
        self, n_channels, input_shape, n_pose_layers,
    ):
        encoder = self._get_cnn(n_channels, n_pose_layers)
        encoder_output_size = encoder(torch.randn(1, n_channels, *input_shape)).shape[1]
        projector = nn.Sequential(
            nn.Linear(encoder_output_size, 1024), nn.ReLU(), nn.Dropout(),
        )
        predictor = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 3))
        return encoder, projector, predictor

    def _get_ensemble_attention(self, n_ensemble_layers, feat_size, n_modes):
        layers = [
            nn.Linear(feat_size, 128),
            nn.ReLU(),
        ]
        for i in range(n_ensemble_layers):
            layers += [
                nn.Linear(128, 128),
                nn.ReLU(),
            ]
        layers += [
            nn.Linear(128, n_modes),
            nn.Softmax(dim=1),
        ]

        return nn.Sequential(*layers)

    def forward(self, pose_inputs):
        feats = []
        preds = []
        outputs = {}
        if "rgb" in self.inputs:
            st_1 = pose_inputs["rgb_t_1"]
            st = pose_inputs["rgb_t"]
            st_encoded = self.rgb_encoder(torch.cat([st_1, st], dim=1))
            st_feats = self.rgb_projector(st_encoded)
            pose_rgb = self.rgb_predictor(st_feats)
            feats.append(st_feats)
            preds.append(pose_rgb)
            outputs["pose_rgb"] = pose_rgb
        if "depth" in self.inputs:
            dt_1 = pose_inputs["depth_t_1"]
            dt = pose_inputs["depth_t"]
            dt_encoded = self.depth_encoder(torch.cat([dt_1, dt], dim=1))
            dt_feats = self.depth_projector(dt_encoded)
            pose_depth = self.depth_predictor(dt_feats)
            feats.append(dt_feats)
            preds.append(pose_depth)
            outputs["pose_depth"] = pose_depth
        if "ego_map" in self.inputs:
            pt_1 = pose_inputs["ego_map_t_1"]
            pt = pose_inputs["ego_map_t"]
            pt_encoded = self.ego_map_encoder(torch.cat([pt_1, pt], dim=1))
            pt_feats = self.ego_map_projector(pt_encoded)
            pose_ego_map = self.ego_map_predictor(pt_feats)
            feats.append(pt_feats)
            preds.append(pose_ego_map)
            outputs["pose_ego_map"] = pose_ego_map

        if len(self.inputs) > 1:
            feats = torch.cat(feats, dim=1)
            ensemble_weights = self.ensemble_attention(feats)  # (bs, n)
            stacked_poses = torch.stack(preds, dim=1)  # (bs, n, 3)
            pose = (ensemble_weights.unsqueeze(2) * stacked_poses).sum(dim=1)
            outputs["pose"] = pose
        else:
            outputs["pose"] = preds[0]

        return outputs

    def _get_cnn(self, n_channels, n_layers):
        cnn_layers = [
            nn.Conv2d(n_channels, 64, (4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 32, (4, 4), stride=(2, 2)),
            nn.ReLU(),
        ]
        for i in range(n_layers):
            cnn_layers += [
                nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
            ]
        cnn_layers += [
            nn.Conv2d(32, 16, (3, 3), stride=(1, 1)),
            nn.ReLU(),
            Rearrange("b c h w -> b (c h w)"),
        ]
        custom_cnn = nn.Sequential(*cnn_layers)

        return custom_cnn


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(
        self,
        recurrent,
        embedding_buckets,
        hidden_size=256,
        img_mean=[0, 0, 0],
        img_std=[1, 1, 1],
        input_shape=(90, 160),
    ):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        self.img_mean = img_mean
        self.img_std = img_std
        imH, imW = input_shape

        embedding_size = 0
        # Assumes input size is (imH, imW)
        self.rgb_encoder = nn.Sequential(
            init_(nn.Conv2d(3, 2, 3, padding=1)),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            init_(nn.Conv2d(2, 1, 3, padding=1)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            init_(nn.Conv2d(1, 1, 3, padding=1)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            Flatten(),
        )
        rgb_encoder_output = self.rgb_encoder(torch.randn(1, 3, imH, imW))
        embedding_size += rgb_encoder_output.shape[1]
        embedding_size += self._create_embeddings(embedding_buckets)

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.fuse_embedding = init_(nn.Linear(embedding_size, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def _create_embeddings(self, embedding_buckets):
        embedding_size = 0
        self.embedding_buckets = embedding_buckets
        self.distance_encoder = BucketingEmbedding(
            embedding_buckets.DISTANCE.min,
            embedding_buckets.DISTANCE.max,
            embedding_buckets.DISTANCE.count,
            embedding_buckets.DISTANCE.dim,
            embedding_buckets.DISTANCE.use_log_scale,
        )
        embedding_size += embedding_buckets.DISTANCE.dim

        self.angle_encoder = BucketingEmbedding(
            embedding_buckets.ANGLE.min,
            embedding_buckets.ANGLE.max,
            embedding_buckets.ANGLE.count,
            embedding_buckets.ANGLE.dim,
            embedding_buckets.ANGLE.use_log_scale,
        )
        embedding_size += embedding_buckets.ANGLE.dim

        self.time_encoder = BucketingEmbedding(
            embedding_buckets.TIME.min,
            embedding_buckets.TIME.max,
            embedding_buckets.TIME.count,
            embedding_buckets.TIME.dim,
            embedding_buckets.TIME.use_log_scale,
        )
        embedding_size += embedding_buckets.TIME.dim

        return embedding_size

    def forward(self, inputs, rnn_hxs, masks):
        x_rgb = inputs["rgb_at_t"]
        x_rgb = process_image(x_rgb, self.img_mean, self.img_std)
        x_goal = inputs["goal_at_t"]
        x_time = inputs["t"].squeeze(1)

        x_rho = torch.norm(x_goal, dim=1)
        x_phi = torch.atan2(x_goal[:, 1], x_goal[:, 0])

        x_rho_emb = self.distance_encoder(x_rho)
        x_phi_emb = self.angle_encoder(x_phi)
        x_time_emb = self.time_encoder(x_time)
        x_rgb_emb = self.rgb_encoder(x_rgb)

        embeddings = [x_rgb_emb, x_rho_emb, x_phi_emb, x_time_emb]

        x = self.fuse_embedding(torch.cat(embeddings, dim=1))

        x, rnn_hxs = self._forward_gru(x, rnn_hxs.squeeze(0), masks)
        rnn_hxs = rnn_hxs.unsqueeze(0)

        return self.critic_linear(x), x, rnn_hxs
