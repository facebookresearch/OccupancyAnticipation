#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import torch
from habitat_baselines.common.rollout_storage import RolloutStorage


class RolloutStorageExtended(RolloutStorage):
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        enable_recurrence=True,
        delay_observations_entry=False,
        delay_masks_entry=False,
        enable_memory_efficient_mode=False,
    ):

        super().__init__(
            num_steps,
            num_envs,
            observation_space,
            action_space,
            recurrent_hidden_state_size,
            num_recurrent_layers=num_recurrent_layers,
        )
        self.enable_recurrence = enable_recurrence
        # This delays observation writing by 1 step.  This is necessary
        # if a forward pass through the policy is needed to actually
        # generate the required observations for the policy.
        self.delay_observations_entry = delay_observations_entry
        # This delays mask writing by 1 step.  This is necessary
        # if a forward pass through the policy is needed to actually
        # generate the required masks for the policy.
        self.delay_masks_entry = delay_masks_entry
        # If efficient mode is enabled, the rollout memories are stored on CPU
        # and ported to GPU during update.
        self.enable_memory_efficient_mode = enable_memory_efficient_mode

        if not self.enable_recurrence:
            self.recurrent_hidden_states = None

    def to(self, device):
        if self.enable_memory_efficient_mode:
            self._device = device
            return None

        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        if self.enable_recurrence:
            self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
    ):
        for sensor in observations:
            if self.delay_observations_entry:
                # Special behavior
                self.observations[sensor][self.step].copy_(observations[sensor])
            else:
                # The default behavior
                self.observations[sensor][self.step + 1].copy_(observations[sensor])
        if self.enable_recurrence:
            self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        if self.delay_masks_entry:
            # Special behavior
            self.masks[self.step].copy_(masks)
        else:
            # The default behavior
            self.masks[self.step + 1].copy_(masks)

        self.step = self.step + 1

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][self.step])

        if self.enable_recurrence:
            self.recurrent_hidden_states[0].copy_(
                self.recurrent_hidden_states[self.step]
            )
        self.masks[0].copy_(self.masks[self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if self.enable_memory_efficient_mode:
            next_value = next_value.cpu()
        if use_gae:
            self.value_preds[self.step] = next_value
            gae = 0
            for step in reversed(range(self.step)):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[self.step] = next_value
            for step in reversed(range(self.step)):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            if self.enable_recurrence:
                recurrent_hidden_states_batch = []
            else:
                recurrent_hidden_states_batch = None
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                if self.enable_recurrence:
                    recurrent_hidden_states_batch.append(
                        self.recurrent_hidden_states[0, :, ind]
                    )

                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )

                adv_targ.append(advantages[: self.step, ind])

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(observations_batch[sensor], 1)

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            if self.enable_recurrence:
                recurrent_hidden_states_batch = torch.stack(
                    recurrent_hidden_states_batch, 1
                )

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = self._flatten_helper(T, N, adv_targ)

            if self.enable_memory_efficient_mode:
                for sensor in observations_batch:
                    observations_batch[sensor] = observations_batch[sensor].to(
                        self._device
                    )
                actions_batch = actions_batch.to(self._device)
                prev_actions_batch = prev_actions_batch.to(self._device)
                value_preds_batch = value_preds_batch.to(self._device)
                return_batch = return_batch.to(self._device)
                masks_batch = masks_batch.to(self._device)
                old_action_log_probs_batch = old_action_log_probs_batch.to(self._device)
                adv_targ = adv_targ.to(self._device)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])


class MapLargeRolloutStorage:
    r"""Class for storing rollout information for training map prediction model.
    Stores information over several episodes as an online dataset for training.
    """

    def __init__(
        self, replay_size, observation_space, device,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                replay_size, *observation_space.spaces[sensor].shape
            )

        self.replay_size = replay_size
        self.step = 0
        self.memory_filled = False
        self.device = device

    def insert(self, observations):
        for sensor in observations:
            # The default behavior
            bs = observations[sensor].shape[0]
            if self.step + bs < self.replay_size:
                self.observations[sensor][self.step : (self.step + bs)].copy_(
                    observations[sensor]
                )
            else:
                self.memory_filled = True
                n1 = self.replay_size - self.step
                n2 = bs - n1
                self.observations[sensor][self.step :].copy_(observations[sensor][:n1])
                self.observations[sensor][:n2].copy_(observations[sensor][n1:])

        self.step = (self.step + bs) % self.replay_size

    def sample(self, batch_size):
        if self.memory_filled:
            ridx = torch.randint(0, self.replay_size - batch_size, (1,)).item()
        elif self.step > batch_size:
            ridx = torch.randint(0, self.step - batch_size, (1,)).item()
        else:
            return None

        random_batch = {}
        for sensor in self.observations:
            random_batch[sensor] = self.observations[sensor][
                ridx : (ridx + batch_size)
            ].to(self.device)

        return random_batch

    def get_memory_size(self):
        memory_size = self.replay_size if self.memory_filled else self.step
        return memory_size


class MapLargeRolloutStorageMP(MapLargeRolloutStorage):
    r"""Class for storing rollout information for training map prediction
    model. Stores information over several episodes as an online dataset for
    training.  Shares tensors and other variables across processes for
    torch.multiprocessing.
    """

    def __init__(
        self, replay_size, observation_space, device, mp_manager,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                replay_size, *observation_space.spaces[sensor].shape
            )

        self.replay_size = replay_size
        # Re-define step and memory_filled as multiprocessing Values that are
        # shared across processes. Create setters and getters for step and
        # memory_filled to keep rest of the pipeline fixed.
        self._step = mp_manager.Value("step", 0)
        self._memory_filled = mp_manager.Value("memory_filled", False)
        self.device = device
        # Share the tensor memory
        self.share_memory()

    @property
    def step(self):
        return self._step.value

    @step.setter
    def step(self, _step):
        self._step.value = _step

    @property
    def memory_filled(self):
        return self._memory_filled.value

    @memory_filled.setter
    def memory_filled(self, _memory_filled):
        self._memory_filled.value = _memory_filled

    def share_memory(self):
        for sensor in self.observations.keys():
            self.observations[sensor].share_memory_()
