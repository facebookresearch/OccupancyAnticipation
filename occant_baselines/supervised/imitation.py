#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim

EPS_PPO = 1e-5


def flatten_two(x):
    return x.view(-1, *x.shape[2:])


def unflatten_two(x, sh1, sh2):
    return x.view(sh1, sh2, *x.shape[1:])


class Imitation(nn.Module):
    def __init__(
        self, actor_critic, lr=None, eps=None, max_grad_norm=None,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, actor_critic.parameters())),
            lr=lr,
            eps=eps,
        )
        self.device = next(actor_critic.parameters()).device

    def forward(self, *x):
        raise NotImplementedError

    def update(self, rollouts):

        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        value_loss_epoch = 0.0

        T, N = rollouts.num_steps, rollouts.rewards.size(1)
        obs_batch = {k: flatten_two(v[:-1]) for k, v in rollouts.observations.items()}
        # These are assumed to be the GT actions to imitate
        il_actions_batch = flatten_two(rollouts.actions)
        prev_actions_batch = flatten_two(rollouts.prev_actions)
        masks_batch = flatten_two(rollouts.masks[:-1])
        recurrent_hidden_states_batch = rollouts.recurrent_hidden_states[0]

        # Reshape to do in a single forward pass for all steps
        (
            _,
            il_action_log_probs,  # (T * N, 1)
            dist_entropy,
            _,
        ) = self.actor_critic.evaluate_actions(
            obs_batch,
            recurrent_hidden_states_batch,
            prev_actions_batch,
            masks_batch,
            il_actions_batch,
        )

        # Maximize the probability of selecting GT actions
        action_loss = -il_action_log_probs.mean()

        self.optimizer.zero_grad()
        total_loss = action_loss

        self.before_backward(total_loss)
        total_loss.backward()
        self.after_backward(total_loss)

        self.before_step()
        self.optimizer.step()
        self.after_step()

        action_loss_epoch += action_loss.item()
        dist_entropy_epoch += dist_entropy.item()

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

    def after_step(self):
        pass

    def load_state_dict(self, loaded_state_dict):
        """Intelligent state dict assignment. Load state-dict only for keys
        that are available and have matching parameter sizes.
        """
        src_state_dict = self.state_dict()
        matching_state_dict = {}
        offending_keys = []
        for k, v in loaded_state_dict.items():
            if k in src_state_dict.keys() and v.shape == src_state_dict[k].shape:
                matching_state_dict[k] = v
            else:
                offending_keys.append(k)
        src_state_dict.update(matching_state_dict)
        super().load_state_dict(src_state_dict)
        if len(offending_keys) > 0:
            for k in offending_keys:
                print(k)
