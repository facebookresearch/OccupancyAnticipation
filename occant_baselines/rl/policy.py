#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces
from occant_utils.common import (
    add_pose,
    crop_map,
    subtract_pose,
    process_image,
    transpose_image,
    bottom_row_padding,
    bottom_row_cropping,
    spatial_transform_map,
)
from occant_utils.common import (
    FixedCategorical,
    Categorical,
    init,
)
from occant_baselines.rl.policy_utils import (
    CNNBase,
    Flatten,
    PoseEstimator,
)
from einops import rearrange
from einops.layers.torch import Rearrange


EPS_MAPPER = 1e-8


class GlobalPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.G = config.map_size

        self.actor = nn.Sequential(  # (8, G, G)
            nn.Conv2d(8, 8, 3, padding=1),  # (8, G, G)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, padding=1),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 5, padding=2),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 2, 5, padding=2),  # (2, G, G)
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 1, 5, padding=2),  # (1, G, G)
            Flatten(),  # (G*G, )
        )

        self.critic = nn.Sequential(  # (8, G, G)
            nn.Conv2d(8, 8, 3, padding=1),  # (8, G, G)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, padding=1),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 5, padding=2),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 2, 5, padding=2),  # (2, G, G)
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 1, 5, padding=2),  # (1, G, G)
            Flatten(),
            nn.Linear(self.G * self.G, 1),
        )

        if config.use_data_parallel:
            self.actor = nn.DataParallel(
                self.actor, device_ids=config.gpu_ids, output_device=config.gpu_ids[0],
            )
            self.critic = nn.DataParallel(
                self.critic, device_ids=config.gpu_ids, output_device=config.gpu_ids[0],
            )

    def forward(self, inputs):
        raise NotImplementedError

    def _get_h12(self, inputs):
        x = inputs["pose_in_map_at_t"]
        h = inputs["map_at_t"]

        h_1 = crop_map(h, x[:, :2], self.G)
        h_2 = F.adaptive_max_pool2d(h, (self.G, self.G))

        h_12 = torch.cat([h_1, h_2], dim=1)

        return h_12

    def act(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False):
        """
        Note: inputs['pose_in_map_at_t'] must obey the following conventions:
              origin at top-left, downward Y and rightward X in the map coordinate system.
        """
        M = inputs["map_at_t"].shape[2]
        h_12 = self._get_h12(inputs)
        action_logits = self.actor(h_12)
        dist = FixedCategorical(logits=action_logits)
        value = self.critic(h_12)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, prev_actions, masks):
        h_12 = self._get_h12(inputs)
        value = self.critic(h_12)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, prev_actions, masks, action):
        h_12 = self._get_h12(inputs)
        action_logits = self.actor(h_12)
        dist = FixedCategorical(logits=action_logits)
        value = self.critic(h_12)

        action_log_probs = dist.log_probs(action)

        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class LocalPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nactions = config.nactions
        self.hidden_size = config.hidden_size
        embedding_buckets = config.EMBEDDING_BUCKETS

        self.base = CNNBase(
            True,
            embedding_buckets,
            hidden_size=self.hidden_size,
            img_mean=config.NORMALIZATION.img_mean,
            img_std=config.NORMALIZATION.img_std,
            input_shape=config.image_scale_hw,
        )

        self.dist = Categorical(self.hidden_size, self.nactions)
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, prev_actions, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, prev_actions, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, prev_actions, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class HeuristicLocalPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, inputs, rnn_hxs, prev_actions, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False):
        goal_xy = inputs["goal_at_t"]
        goal_phi = torch.atan2(goal_xy[:, 1], goal_xy[:, 0])

        turn_angle = math.radians(self.config.AGENT_DYNAMICS.turn_angle)
        fwd_action_flag = torch.abs(goal_phi) <= 0.9 * turn_angle
        turn_left_flag = ~fwd_action_flag & (goal_phi < 0)
        turn_right_flag = ~fwd_action_flag & (goal_phi > 0)

        action = torch.zeros_like(goal_xy)[:, 0:1]
        action[fwd_action_flag] = 0
        action[turn_left_flag] = 1
        action[turn_right_flag] = 2
        action = action.long()

        return None, action, None, rnn_hxs

    def load_state_dict(self, *args, **kwargs):
        pass


class Mapper(nn.Module):
    def __init__(self, config, projection_unit):
        super().__init__()
        self.config = config
        self.map_config = {"size": config.map_size, "scale": config.map_scale}
        V = self.map_config["size"]
        s = self.map_config["scale"]
        self.img_mean_t = rearrange(
            torch.Tensor(self.config.NORMALIZATION.img_mean), "c -> () c () ()"
        )
        self.img_std_t = rearrange(
            torch.Tensor(self.config.NORMALIZATION.img_std), "c -> () c () ()"
        )
        self.projection_unit = projection_unit
        self.pose_estimation_unit_part_1 = nn.Sequential(  # (4, V, V)
            nn.Conv2d(4, 2, 3, stride=1, padding=1),  # (2, V, V)
            nn.ReLU(),
            nn.Conv2d(2, 2, 5, stride=1, padding=2),  # (2, V, V)
            nn.ReLU(),
            nn.Conv2d(2, 1, 5, stride=1, padding=2),  # (1, V, V)
            nn.ReLU(),
            Rearrange("b c h w -> b (c h w)"),
        )
        self.pose_estimation_unit_part_2 = nn.Sequential(
            nn.Linear(V * V, 32), nn.LeakyReLU(0.2), nn.Linear(32, 3),
        )
        if self.config.freeze_projection_unit:
            for p in self.projection_unit.parameters():
                p.requires_grad = False

        if self.config.use_data_parallel:
            self.projection_unit = nn.DataParallel(self.projection_unit)

        # Cache to store pre-computed information
        self._cache = {}

    def forward(self, x, masks=None):
        outputs = self.predict_deltas(x, masks=masks)
        mt_1 = x["map_at_t_1"]
        if masks is not None:
            mt_1 = mt_1 * masks.view(-1, 1, 1, 1)
        with torch.no_grad():
            mt = self._register_map(mt_1, outputs["pt"], outputs["xt_hat"])
        outputs["mt"] = mt

        return outputs

    def predict_deltas(self, x, masks=None):
        # Transpose multichannel inputs
        st_1 = process_image(x["rgb_at_t_1"], self.img_mean_t, self.img_std_t)
        dt_1 = transpose_image(x["depth_at_t_1"])
        ego_map_gt_at_t_1 = transpose_image(x["ego_map_gt_at_t_1"])
        st = process_image(x["rgb_at_t"], self.img_mean_t, self.img_std_t)
        dt = transpose_image(x["depth_at_t"])
        ego_map_gt_at_t = transpose_image(x["ego_map_gt_at_t"])
        # This happens only for a baseline
        if (
            "ego_map_gt_anticipated_at_t_1" in x
            and x["ego_map_gt_anticipated_at_t_1"] is not None
        ):
            ego_map_gt_anticipated_at_t_1 = transpose_image(
                x["ego_map_gt_anticipated_at_t_1"]
            )
            ego_map_gt_anticipated_at_t = transpose_image(
                x["ego_map_gt_anticipated_at_t"]
            )
        else:
            ego_map_gt_anticipated_at_t_1 = None
            ego_map_gt_anticipated_at_t = None
        # Compute past and current egocentric maps
        bs = st_1.size(0)
        pu_inputs_t_1 = {
            "rgb": st_1,
            "depth": dt_1,
            "ego_map_gt": ego_map_gt_at_t_1,
            "ego_map_gt_anticipated": ego_map_gt_anticipated_at_t_1,
        }
        pu_inputs_t = {
            "rgb": st,
            "depth": dt,
            "ego_map_gt": ego_map_gt_at_t,
            "ego_map_gt_anticipated": ego_map_gt_anticipated_at_t,
        }
        pu_inputs = self._safe_cat(pu_inputs_t_1, pu_inputs_t)
        pu_outputs = self.projection_unit(pu_inputs)
        pu_outputs_t = {k: v[bs:] for k, v in pu_outputs.items()}
        pt_1, pt = pu_outputs["occ_estimate"][:bs], pu_outputs["occ_estimate"][bs:]
        # Compute relative pose
        dx = subtract_pose(x["pose_at_t_1"], x["pose_at_t"])
        # Estimate pose
        dx_hat = dx
        xt_hat = x["pose_at_t"]
        all_pose_outputs = None
        if not self.config.ignore_pose_estimator:
            all_pose_outputs = {}
            pose_inputs = {}
            if "rgb" in self.config.pose_predictor_inputs:
                pose_inputs["rgb_t_1"] = st_1
                pose_inputs["rgb_t"] = st
            if "depth" in self.config.pose_predictor_inputs:
                pose_inputs["depth_t_1"] = dt_1
                pose_inputs["depth_t"] = dt
            if "ego_map" in self.config.pose_predictor_inputs:
                pose_inputs["ego_map_t_1"] = pt_1
                pose_inputs["ego_map_t"] = pt
            if self.config.detach_map:
                for k in pose_inputs.keys():
                    pose_inputs[k] = pose_inputs[k].detach()
            n_pose_inputs = self._transform_observations(pose_inputs, dx)
            pose_map_inputs = torch.cat(
                [n_pose_inputs["ego_map_t_1"], n_pose_inputs["ego_map_t"]], 1
            )
            pose_features = self.pose_estimation_unit_part_1(pose_map_inputs)
            pose_outputs = {"pose": self.pose_estimation_unit_part_2(pose_features)}
            dx_hat = add_pose(dx, pose_outputs["pose"])
            all_pose_outputs["pose_outputs"] = pose_outputs
            # Estimate global pose
            xt_hat = add_pose(x["pose_hat_at_t_1"], dx_hat)
        # Zero out pose prediction based on the mask
        if masks is not None:
            xt_hat = xt_hat * masks
            dx_hat = dx_hat * masks
        outputs = {
            "pt": pt,
            "dx_hat": dx_hat,
            "xt_hat": xt_hat,
            "all_pu_outputs": pu_outputs_t,
            "all_pose_outputs": all_pose_outputs,
        }
        if "ego_map_hat" in pu_outputs_t:
            outputs["ego_map_hat_at_t"] = pu_outputs_t["ego_map_hat"]
        return outputs

    def _bottom_row_spatial_transform(self, p, dx, invert=False):
        """
        Inputs:
            p - (bs, 2, V, V) local map
            dx - (bs, 3) egocentric transformation --- (dx, dy, dtheta)

        NOTE: The agent stands at the central column of the last row in the
        ego-centric map and looks forward. But the rotation happens about the
        center of the map.  To handle this, first zero-pad pt_1 and then crop
        it after transforming.

        Conventions:
            The origin is at the bottom-center of the map.
            X is upward with agent's forward direction
            Y is rightward with agent's rightward direction
        """
        V = p.shape[2]
        p_pad = bottom_row_padding(p)
        p_trans_pad = self._spatial_transform(p_pad, dx, invert=invert)
        # Crop out the original part
        p_trans = bottom_row_cropping(p_trans_pad, V)

        return p_trans

    def _spatial_transform(self, p, dx, invert=False):
        """
        Applies the transformation dx to image p.
        Inputs:
            p - (bs, 2, H, W) map
            dx - (bs, 3) egocentric transformation --- (dx, dy, dtheta)

        Conventions:
            The origin is at the center of the map.
            X is upward with agent's forward direction
            Y is rightward with agent's rightward direction

        Note: These denote transforms in an agent's position. Not the image directly.
        For example, if an agent is moving upward, then the map will be moving downward.
        To disable this behavior, set invert=False.
        """
        s = self.map_config["scale"]
        # Convert dx to map image coordinate system with X as rightward and Y as downward
        dx_map = torch.stack(
            [(dx[:, 1] / s), -(dx[:, 0] / s), dx[:, 2]], dim=1
        )  # anti-clockwise rotation
        p_trans = spatial_transform_map(p, dx_map, invert=invert)

        return p_trans

    def _register_map(self, m, p, x):
        """
        Given the locally computed map, register it to the global map based
        on the current position.

        Inputs:
            m - (bs, F, M, M) global map
            p - (bs, F, V, V) local map
            x - (bs, 3) in global coordinates
        """
        V = self.map_config["size"]
        s = self.map_config["scale"]
        M = m.shape[2]
        Vby2 = (V - 1) // 2 if V % 2 == 1 else V // 2
        Mby2 = (M - 1) // 2 if M % 2 == 1 else M // 2
        # The agent stands at the bottom-center of the egomap and looks upward
        left_h_pad = Mby2 - V + 1
        right_h_pad = M - V - left_h_pad
        left_w_pad = Mby2 - Vby2
        right_w_pad = M - V - left_w_pad
        # Add zero padding to p so that it matches size of global map
        p_pad = F.pad(
            p, (left_w_pad, right_w_pad, left_h_pad, right_h_pad), "constant", 0
        )
        # Register the local map
        p_reg = self._spatial_transform(p_pad, x)
        # Aggregate
        m_updated = self._aggregate(m, p_reg)

        return m_updated

    def _aggregate(self, m, p_reg):
        """
        Inputs:
            m - (bs, 2, M, M) - global map
            p_reg - (bs, 2, M, M) - registered egomap
        """
        reg_type = self.config.registration_type
        beta = self.config.map_registration_momentum
        if reg_type == "max":
            m_updated = torch.max(m, p_reg)
        elif reg_type == "overwrite":
            # Overwrite only the currently explored regions
            mask = (p_reg[:, 1] > self.config.thresh_explored).float()
            mask = mask.unsqueeze(1)
            m_updated = m * (1 - mask) + p_reg * mask
        elif reg_type == "moving_average":
            mask_unexplored = (
                (p_reg[:, 1] <= self.config.thresh_explored).float().unsqueeze(1)
            )
            mask_unfilled = (m[:, 1] == 0).float().unsqueeze(1)
            m_ma = p_reg * (1 - beta) + m * beta
            m_updated = (
                m * mask_unexplored
                + m_ma * (1.0 - mask_unexplored) * (1.0 - mask_unfilled)
                + p_reg * (1.0 - mask_unexplored) * mask_unfilled
            )
        elif reg_type == "entropy_moving_average":
            explored_mask = (p_reg[:, 1] > self.config.thresh_explored).float()
            log_p_reg = torch.log(p_reg + EPS_MAPPER)
            log_1_p_reg = torch.log(1 - p_reg + EPS_MAPPER)
            entropy = -p_reg * log_p_reg - (1 - p_reg) * log_1_p_reg
            entropy_mask = (entropy.mean(dim=1) < self.config.thresh_entropy).float()
            explored_mask = explored_mask * entropy_mask
            unfilled_mask = (m[:, 1] == 0).float()
            m_updated = m
            # For regions that are unfilled, write as it is
            mask = unfilled_mask * explored_mask
            mask = mask.unsqueeze(1)
            m_updated = m_updated * (1 - mask) + p_reg * mask
            # For regions that are filled, do a moving average
            mask = (1 - unfilled_mask) * explored_mask
            mask = mask.unsqueeze(1)
            p_reg_ma = (p_reg * (1 - beta) + m_updated * beta) * mask
            m_updated = m_updated * (1 - mask) + p_reg_ma * mask
        else:
            raise ValueError(
                f"Mapper: registration_type: {self.config.registration_type} not defined!"
            )

        return m_updated

    def ext_register_map(self, m, p, x):
        return self._register_map(m, p, x)

    def _transform_observations(self, inputs, dx):
        """Converts observations from t-1 to coordinate frame for t.
        """
        # ====================== Transform past egocentric map ========================
        if "ego_map_t_1" in inputs:
            ego_map_t_1 = inputs["ego_map_t_1"]
            ego_map_t_1_trans = self._bottom_row_spatial_transform(
                ego_map_t_1, dx, invert=True
            )
            inputs["ego_map_t_1"] = ego_map_t_1_trans
        occ_cfg = self.projection_unit.main.config
        # ========================= Transform rgb and depth ===========================
        if "depth_t_1" in inputs:
            device = inputs["depth_t_1"].device
            depth_t_1 = inputs["depth_t_1"]
            if "K" not in self._cache.keys():
                # Project images from previous camera pose to current camera pose
                # Compute intrinsic camera matrix
                hfov = math.radians(occ_cfg.EGO_PROJECTION.hfov)
                vfov = math.radians(occ_cfg.EGO_PROJECTION.vfov)
                K = torch.Tensor(
                    [
                        [1 / math.tan(hfov / 2.0), 0.0, 0.0, 0.0],
                        [0.0, 1 / math.tan(vfov / 2.0), 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ).to(
                    device
                )  # (4, 4)
                self._cache["K"] = K.cpu()
            else:
                K = self._cache["K"].to(device)
            H, W = depth_t_1.shape[2:]
            min_depth = occ_cfg.EGO_PROJECTION.min_depth
            max_depth = occ_cfg.EGO_PROJECTION.max_depth
            depth_t_1_unnorm = depth_t_1 * (max_depth - min_depth) + min_depth
            if "xs" not in self._cache.keys():
                xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))
                xs = torch.Tensor(xs.reshape(1, H, W)).to(device).unsqueeze(0)
                ys = torch.Tensor(ys.reshape(1, H, W)).to(device).unsqueeze(0)
                self._cache["xs"] = xs.cpu()
                self._cache["ys"] = ys.cpu()
            else:
                xs = self._cache["xs"].to(device)
                ys = self._cache["ys"].to(device)
            # Unproject
            # negate depth as the camera looks along -Z
            xys = torch.stack(
                [
                    xs * depth_t_1_unnorm,
                    ys * depth_t_1_unnorm,
                    -depth_t_1_unnorm,
                    torch.ones_like(depth_t_1_unnorm),
                ],
                dim=4,
            )  # (bs, 1, H, W, 4)
            # Points in the target (camera 2)
            xys = rearrange(xys, "b () h w f -> b (h w) f")
            if "invK" not in self._cache.keys():
                invK = torch.inverse(K)
                self._cache["invK"] = invK.cpu()
            else:
                invK = self._cache["invK"].to(device)
            xy_c2 = torch.matmul(xys, invK.unsqueeze(0))
            # ================ Camera 2 --> Camera 1 transformation ===============
            # We need the target to source transformation to warp from camera 1
            # to camera 2. In dx, dx[:, 0] is -Z, dx[:, 1] is X and dx[:, 2] is
            # rotation from -Z to X.
            translation = torch.stack(
                [dx[:, 1], torch.zeros_like(dx[:, 1]), -dx[:, 0]], dim=1
            )  # (bs, 3)
            T_world_camera2 = torch.zeros(xy_c2.shape[0], 4, 4).to(device)
            # Right-hand-rule rotation about Y axis
            cos_theta = torch.cos(-dx[:, 2])
            sin_theta = torch.sin(-dx[:, 2])
            T_world_camera2[:, 0, 0].copy_(cos_theta)
            T_world_camera2[:, 0, 2].copy_(sin_theta)
            T_world_camera2[:, 1, 1].fill_(1.0)
            T_world_camera2[:, 2, 0].copy_(-sin_theta)
            T_world_camera2[:, 2, 2].copy_(cos_theta)
            T_world_camera2[:, :3, 3].copy_(translation)
            T_world_camera2[:, 3, 3].fill_(1.0)
            # Transformation matrix from camera 2 --> world.
            T_camera1_camera2 = T_world_camera2  # (bs, 4, 4)
            xy_c1 = torch.matmul(
                T_camera1_camera2, xy_c2.transpose(1, 2)
            )  # (bs, 4, HW)
            # Convert camera coordinates to image coordinates
            xy_newimg = torch.matmul(K, xy_c1)  # (bs, 4, HW)
            xy_newimg = xy_newimg.transpose(1, 2)  # (bs, HW, 4)
            xys_newimg = xy_newimg[:, :, :2] / (
                -xy_newimg[:, :, 2:3] + 1e-8
            )  # (bs, HW, 2)
            # Flip back to y-down to match array indexing
            xys_newimg[:, :, 1] *= -1  # (bs, HW, 2)
            # ================== Apply warp to RGB, Depth images ==================
            sampler = rearrange(xys_newimg, "b (h w) f -> b h w f", h=H, w=W)
            depth_t_1_trans = F.grid_sample(depth_t_1, sampler, padding_mode="zeros")
            inputs["depth_t_1"] = depth_t_1_trans
            if "rgb_t_1" in inputs:
                rgb_t_1 = inputs["rgb_t_1"]
                rgb_t_1_trans = F.grid_sample(rgb_t_1, sampler, padding_mode="zeros")
                inputs["rgb_t_1"] = rgb_t_1_trans

        return inputs

    def _safe_cat(self, d1, d2):
        """Given two dicts of tensors with same keys, the values are
        concatenated if not None.
        """
        d = {}
        for k, v1 in d1.items():
            d[k] = None if v1 is None else torch.cat([v1, d2[k]], 0)
        return d


class MapperDataParallelWrapper(Mapper):
    def forward(self, *args, method_name="predict_deltas", **kwargs):
        if method_name == "predict_deltas":
            outputs = self.predict_deltas(*args, **kwargs)
        elif method_name == "estimate_ego_map":
            outputs = self._estimate_ego_map(*args, **kwargs)

        return outputs

    def _estimate_ego_map(self, x):
        return self.projection_unit(x)
