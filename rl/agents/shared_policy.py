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

    def __init__(self, *args: Any, hidden_dim: int = 64, **kwargs: Any) -> None:
        self.hidden_dim = hidden_dim
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
