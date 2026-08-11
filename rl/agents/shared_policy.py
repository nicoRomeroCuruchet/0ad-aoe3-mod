"""Shared per-villager actor with a centralized critic for team environments.

A joint MLP gives every villager its own output weights, so one villager can sit
idle forever while another works: nothing ties their behaviour together. This
policy runs *one* small network over each villager's observation slice and
concatenates the results, so two villagers in the same situation must act the
same. Differences can only come from the slice itself -- own position, the
relational features, and the agent id.

The critic stays centralized: it reads the whole joint state, which is only used
for learning, never for acting.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy


ACTION_VALUES_PER_VILLAGER = 3


class SharedVillagerExtractor(nn.Module):
    """Per-villager trunk plus shared action head, and a joint value trunk."""

    def __init__(
        self,
        villager_count: int,
        slice_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.villager_count = villager_count
        self.slice_dim = slice_dim
        self.latent_dim_pi = ACTION_VALUES_PER_VILLAGER * villager_count
        self.latent_dim_vf = hidden_dim

        self.villager_trunk = nn.Sequential(
            nn.Linear(slice_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.villager_head = nn.Linear(hidden_dim, ACTION_VALUES_PER_VILLAGER)
        self.value_trunk = nn.Sequential(
            nn.Linear(villager_count * slice_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the same weights to every villager slice."""

        batch = features.shape[0]
        slices = features.reshape(batch, self.villager_count, self.slice_dim)
        latent = self.villager_trunk(slices)
        actions = self.villager_head(latent)
        return actions.reshape(batch, self.latent_dim_pi)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Value the whole team from the joint state."""

        return self.value_trunk(features)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class SharedVillagerActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy whose actor is one network shared by every villager."""

    def __init__(
        self,
        *args: Any,
        hidden_dim: int = 64,
        m1_checkpoint: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.m1_checkpoint = m1_checkpoint
        super().__init__(*args, **kwargs)

    def _villager_shape(self) -> tuple[int, int]:
        shape = self.observation_space.shape
        if shape is None or len(shape) != 2:
            raise ValueError(
                "the shared villager policy needs a (villagers, features) "
                f"observation space, got {shape}"
            )
        return int(shape[0]), int(shape[1])

    def _build_mlp_extractor(self) -> None:
        villager_count, slice_dim = self._villager_shape()
        self.mlp_extractor = SharedVillagerExtractor(
            villager_count=villager_count,
            slice_dim=slice_dim,
            hidden_dim=self.hidden_dim,
        )

    def _build(self, lr_schedule: Any) -> None:
        super()._build(lr_schedule)
        # The extractor already emits one action mean per villager through the
        # shared head. SB3's own action_net would mix villagers back together
        # and destroy the weight sharing, so it is removed.
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        if self.m1_checkpoint is not None:
            initialize_from_m1(self, self.m1_checkpoint)
            print(f"m1_transfer: warm-started from {self.m1_checkpoint}", flush=True)


class M1TransferError(ValueError):
    """Raised when an M1 checkpoint cannot initialize the shared network."""


def initialize_from_m1(
    policy: SharedVillagerActorCriticPolicy,
    checkpoint_path: str,
    *,
    loader: Any = None,
) -> None:
    """Warm-start the shared villager network from a trained M1 policy.

    M1 learned the whole gather skill for one villager: walk to a tree, gather,
    stay quiet while UnitAI hauls. Its actor has the same shape as one villager's
    network here, and slice indices 0-9 are M1's observation in M1's order, so
    its weights copy straight into the input prefix. The weights for the
    appended inputs -- agent id and the relational block -- start at zero, so at
    step one the policy behaves exactly like trained M1 and training only has to
    learn what the new inputs mean.

    Loading a checkpoint deserializes pickled objects; only pass files you trust.
    """

    if loader is None:  # pragma: no cover - exercised live, stubbed in tests
        from stable_baselines3 import PPO

        def loader(path: str) -> Any:
            return PPO.load(path, device="cpu")

    source = loader(checkpoint_path).policy
    extractor = policy.mlp_extractor
    source_trunk = source.mlp_extractor.policy_net
    target_trunk = extractor.villager_trunk

    m1_inputs = source_trunk[0].weight.shape[1]
    if m1_inputs > extractor.slice_dim:
        raise M1TransferError(
            f"M1 observation is {m1_inputs} wide but a villager slice is only "
            f"{extractor.slice_dim}"
        )
    if source_trunk[0].weight.shape[0] != target_trunk[0].weight.shape[0]:
        raise M1TransferError("M1 hidden width does not match the villager trunk")
    if source.action_net.weight.shape[0] != ACTION_VALUES_PER_VILLAGER:
        raise M1TransferError("M1 action head does not emit three values")

    with torch.no_grad():
        target_trunk[0].weight.zero_()
        target_trunk[0].weight[:, :m1_inputs] = source_trunk[0].weight
        target_trunk[0].bias.copy_(source_trunk[0].bias)
        target_trunk[2].weight.copy_(source_trunk[2].weight)
        target_trunk[2].bias.copy_(source_trunk[2].bias)
        extractor.villager_head.weight.copy_(source.action_net.weight)
        extractor.villager_head.bias.copy_(source.action_net.bias)
        # One villager's exploration spread, repeated for every villager.
        policy.log_std.copy_(source.log_std.repeat(extractor.villager_count))
