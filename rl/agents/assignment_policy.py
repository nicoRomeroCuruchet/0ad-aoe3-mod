"""Centralized, permutation-equivariant policy for team assignments."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    MultiCategoricalDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy

from rl.gather.assignment_actions import (
    joint_assignment_action_count,
    joint_assignment_from_index,
)


CORE_WIDTH = 11
ACTOR_CORE_WIDTH = 10
RESOURCE_WIDTH = 5
CARRIED_RESOURCE_INDEX = 5
GATHER_CYCLE_ACTIVE_INDEX = 9
CURRENT_TARGET_OFFSET = 3
RESOURCE_REMAINING_OFFSET = 4


def _shared_encoder(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
    )


def _score_head(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    )


class TeamAssignmentDistribution(MultiCategoricalDistribution):
    """Native PPO categoricals with a hierarchical deterministic mode."""

    def mode(self) -> torch.Tensor:
        """Choose ``NO_CLICK/CLICK`` first, then ``tree | CLICK``.

        The flat categories still represent the exact joint mixture used for
        stochastic sampling and PPO log-probabilities. Only its deterministic
        summary is hierarchical, so click probability is not fragmented across
        the resource categories when compared with ``NO_CLICK``.
        """

        actions = []
        for categorical in self.distribution:
            probabilities = categorical.probs
            tree = probabilities[:, 1:].argmax(dim=1) + 1
            click_wins = probabilities[:, 1:].sum(dim=1) > probabilities[:, 0]
            actions.append(torch.where(click_wins, tree, torch.zeros_like(tree)))
        return torch.stack(actions, dim=1)


class JointAssignmentDistribution(CategoricalDistribution):
    """Categorical matching distribution with a hierarchical deterministic mode.

    Sampling, entropy, and log probabilities remain the native SB3 categorical
    behavior.  Only ``mode`` first marginalizes the binary click/no-click
    pattern across villagers, then chooses a specific matching.  A flat argmax
    otherwise compares the one all-NO_CLICK label against many compatible
    productive matchings and can select silence even when their combined mass
    says to act.
    """

    def __init__(
        self,
        action_dim: int,
        click_patterns: torch.Tensor,
    ) -> None:
        super().__init__(action_dim)
        patterns = torch.as_tensor(click_patterns, dtype=torch.long).detach().cpu()
        if patterns.shape != (action_dim,):
            raise ValueError("click_patterns must contain one value per action")
        if torch.any(patterns < 0):
            raise ValueError("click_patterns must be non-negative")
        self._click_patterns = patterns

    def mode(self) -> torch.Tensor:
        """Select a click/no-click pattern by mass, then its best matching."""

        if self.distribution is None:
            raise RuntimeError("distribution parameters have not been set")
        probabilities = self.distribution.probs
        click_patterns = self._click_patterns.to(probabilities.device)
        pattern_choices = int(click_patterns.max().item()) + 1
        pattern_mass = torch.zeros(
            (probabilities.shape[0], pattern_choices),
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        pattern_mass.scatter_add_(
            1,
            click_patterns.unsqueeze(0).expand(probabilities.shape[0], -1),
            probabilities,
        )
        selected_pattern = pattern_mass.argmax(dim=1)
        in_selected_pattern = (
            click_patterns.unsqueeze(0) == selected_pattern.unsqueeze(1)
        )
        selected_matching = probabilities.masked_fill(
            ~in_selected_pattern,
            -torch.inf,
        ).argmax(dim=1)

        # Preserve a neutral, untrained policy's all-NO_CLICK action.  Without
        # this exact-tie fallback, action-table combinatorics alone would make
        # a 3-click bundle the deterministic default rather than letting PPO
        # learn when the team should begin work.
        valid = probabilities > 0.0
        log_probabilities = self.distribution.logits
        maximum = log_probabilities.masked_fill(~valid, -torch.inf).amax(dim=1)
        minimum = log_probabilities.masked_fill(~valid, torch.inf).amin(dim=1)
        exactly_neutral = maximum == minimum
        flat_mode = probabilities.argmax(dim=1)
        return torch.where(exactly_neutral, flat_mode, selected_matching)


class TeamAssignmentExtractor(nn.Module):
    """Share villager/tree scorers while using pooled whole-team context."""

    def __init__(
        self,
        villager_count: int,
        resource_count: int,
        slice_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        expected_slice_dim = CORE_WIDTH + RESOURCE_WIDTH * resource_count
        if villager_count <= 0 or resource_count <= 0:
            raise ValueError("villager_count and resource_count must be positive")
        if slice_dim != expected_slice_dim:
            raise ValueError(
                f"slice_dim must be {expected_slice_dim} for {resource_count} resources"
            )
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.villager_count = villager_count
        self.resource_count = resource_count
        self.slice_dim = slice_dim
        self.hidden_dim = hidden_dim
        self.latent_dim_pi = villager_count * (resource_count + 1)
        self.latent_dim_vf = hidden_dim

        self.villager_encoder = _shared_encoder(ACTOR_CORE_WIDTH, hidden_dim)
        self.resource_encoder = _shared_encoder(RESOURCE_WIDTH, hidden_dim)
        self.no_click_head = _score_head(4 * hidden_dim, hidden_dim)
        self.tree_head = _score_head(5 * hidden_dim, hidden_dim)
        self.value_trunk = _shared_encoder(2 * hidden_dim, hidden_dim)

    def _embeddings(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.shape[0]
        slices = features.reshape(
            batch_size,
            self.villager_count,
            self.slice_dim,
        )
        # Index ten is an arbitrary roster ID. Assignment behavior must depend
        # on state, not on which stable slot happened to receive a villager.
        villager_features = slices[:, :, :ACTOR_CORE_WIDTH]
        resource_features = slices[:, :, CORE_WIDTH:].reshape(
            batch_size,
            self.villager_count,
            self.resource_count,
            RESOURCE_WIDTH,
        )
        return (
            self.villager_encoder(villager_features),
            self.resource_encoder(resource_features),
        )

    def _local_actor_and_critic(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return local ``NO_CLICK/tree`` scores plus a pooled critic latent."""

        villagers, pairs = self._embeddings(features)
        global_villager = villagers.mean(dim=1)
        own_resources = pairs.mean(dim=2)
        global_resource = pairs.mean(dim=(1, 2))
        per_resource_team = pairs.mean(dim=1)

        villager_context = torch.cat(
            (
                villagers,
                global_villager[:, None, :].expand_as(villagers),
                own_resources,
                global_resource[:, None, :].expand_as(villagers),
            ),
            dim=-1,
        )
        no_click = self.no_click_head(villager_context)

        expanded_villagers = villagers[:, :, None, :].expand(
            -1, -1, self.resource_count, -1
        )
        expanded_team = global_villager[:, None, None, :].expand_as(expanded_villagers)
        expanded_resource_team = per_resource_team[:, None, :, :].expand_as(
            expanded_villagers
        )
        expanded_global_resource = global_resource[:, None, None, :].expand_as(
            expanded_villagers
        )
        tree_context = torch.cat(
            (
                expanded_villagers,
                expanded_team,
                pairs,
                expanded_resource_team,
                expanded_global_resource,
            ),
            dim=-1,
        )
        tree_logits = self.tree_head(tree_context).squeeze(-1)
        actor = torch.cat((no_click, tree_logits), dim=2)

        critic_context = torch.cat((global_villager, global_resource), dim=-1)
        critic = self.value_trunk(critic_context)
        return actor, critic

    def _latents(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor, critic = self._local_actor_and_critic(features)
        return actor.reshape(features.shape[0], self.latent_dim_pi), critic

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self._latents(features)[0]

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self._latents(features)[1]

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._latents(features)


def _assignment_shape(
    observation_space: spaces.Space,
    action_space: spaces.Space,
) -> tuple[int, int, int]:
    if not isinstance(action_space, spaces.MultiDiscrete):
        raise ValueError("TeamAssignmentPolicy requires a MultiDiscrete action space")
    if not isinstance(observation_space, spaces.Box):
        raise ValueError("TeamAssignmentPolicy requires a Box observation space")
    shape = observation_space.shape
    if shape is None or len(shape) != 2:
        raise ValueError("observation space must have shape (villagers, features)")

    villager_count, slice_dim = (int(shape[0]), int(shape[1]))
    if villager_count <= 0:
        raise ValueError("observation space must contain at least one villager")
    nvec = np.asarray(action_space.nvec)
    if nvec.shape != (villager_count,):
        raise ValueError("action space must contain one category per villager")
    starts = getattr(action_space, "start", None)
    if starts is not None and np.any(np.asarray(starts) != 0):
        raise ValueError("assignment action categories must start at zero")
    if np.any(nvec != nvec[0]) or int(nvec[0]) <= 1:
        raise ValueError("every villager must have the same resource categories")

    resource_count = int(nvec[0]) - 1
    expected_slice_dim = CORE_WIDTH + RESOURCE_WIDTH * resource_count
    if slice_dim != expected_slice_dim:
        raise ValueError(f"observation slices must be {expected_slice_dim} values wide")
    return villager_count, resource_count, slice_dim


class TeamAssignmentActorCriticPolicy(ActorCriticPolicy):
    """PPO policy producing native ``NO_CLICK/tree-slot`` categoricals."""

    def __init__(
        self,
        *args: Any,
        hidden_dim: int = 64,
        no_click_probability: float = 0.8,
        **kwargs: Any,
    ) -> None:
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, Integral)
            or hidden_dim <= 0
        ):
            raise ValueError("hidden_dim must be a positive integer")
        if (
            isinstance(no_click_probability, bool)
            or not isinstance(no_click_probability, Real)
            or not math.isfinite(float(no_click_probability))
            or not 0.0 < float(no_click_probability) < 1.0
        ):
            raise ValueError("no_click_probability must be finite and between 0 and 1")

        observation_space = kwargs.get("observation_space")
        action_space = kwargs.get("action_space")
        if observation_space is None and len(args) >= 1:
            observation_space = args[0]
        if action_space is None and len(args) >= 2:
            action_space = args[1]
        self._team_shape = _assignment_shape(observation_space, action_space)
        self.hidden_dim = int(hidden_dim)
        self.no_click_probability = float(no_click_probability)
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        return {
            **super()._get_constructor_parameters(),
            "hidden_dim": self.hidden_dim,
            "no_click_probability": self.no_click_probability,
        }

    def _build_mlp_extractor(self) -> None:
        villager_count, resource_count, slice_dim = self._team_shape
        self.mlp_extractor = TeamAssignmentExtractor(
            villager_count=villager_count,
            resource_count=resource_count,
            slice_dim=slice_dim,
            hidden_dim=self.hidden_dim,
        )

    def _build(self, lr_schedule: Any) -> None:
        if not isinstance(self.action_dist, MultiCategoricalDistribution):
            raise TypeError("unexpected assignment action distribution")
        self.action_dist = TeamAssignmentDistribution(
            list(self.action_dist.action_dims)
        )
        super()._build(lr_schedule)
        # The extractor already emits correctly ordered categorical logits.
        # A second linear layer would mix villager and tree slots together and
        # destroy both permutation guarantees.
        self.action_net = nn.Identity()
        self._initialize_assignment_scores()
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _initialize_assignment_scores(self) -> None:
        extractor = self.mlp_extractor
        if not isinstance(extractor, TeamAssignmentExtractor):
            raise TypeError("unexpected assignment extractor")
        no_click_output = extractor.no_click_head[-1]
        tree_output = extractor.tree_head[-1]
        self.init_weights(no_click_output, gain=0.01)
        self.init_weights(tree_output, gain=0.01)
        tree_probability = (1.0 - self.no_click_probability) / extractor.resource_count
        no_click_bias = math.log(self.no_click_probability / tree_probability)
        with torch.no_grad():
            no_click_output.bias.fill_(no_click_bias)
            tree_output.bias.zero_()


# Concise config-facing name used by ``rl.agents.sb3._resolve_policy``.
TeamAssignmentPolicy = TeamAssignmentActorCriticPolicy


class JointAssignmentExtractor(TeamAssignmentExtractor):
    """Score each valid whole-team matching from shared local pair scores.

    The action table is a fixed mapping, not an observation feature: it merely
    removes physically impossible duplicate-tree command bundles.  The policy
    still scores every villager/tree pairing from the current observation.
    """

    def __init__(
        self,
        villager_count: int,
        resource_count: int,
        slice_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(
            villager_count=villager_count,
            resource_count=resource_count,
            slice_dim=slice_dim,
            hidden_dim=hidden_dim,
        )
        action_count = joint_assignment_action_count(
            villager_count=villager_count,
            resource_count=resource_count,
        )
        categories = np.stack(
            [
                joint_assignment_from_index(
                    index,
                    villager_count=villager_count,
                    resource_count=resource_count,
                )
                for index in range(action_count)
            ],
            axis=0,
        )
        self.register_buffer(
            "assignment_categories",
            torch.as_tensor(categories, dtype=torch.long),
            persistent=True,
        )
        self.latent_dim_pi = action_count

    def _latents(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_scores, critic = self._local_actor_and_critic(features)
        # [actions, villagers] -> [batch, villagers, actions], then pick the
        # selected local category for each villager and sum it. This produces
        # one Categorical logit per complete, collision-free command bundle.
        categories = self.assignment_categories.transpose(0, 1).unsqueeze(0)
        indexes = categories.expand(local_scores.shape[0], -1, -1)
        joint_scores = local_scores.gather(2, indexes).sum(dim=1)
        valid = self._valid_action_mask(features)
        # ``finfo.min`` underflows to exactly zero probability in PyTorch's
        # categorical softmax, while retaining finite logits for the standard
        # SB3 distribution implementation. All-NO_CLICK is always valid.
        joint_scores = joint_scores.masked_fill(
            ~valid,
            torch.finfo(joint_scores.dtype).min,
        )
        return joint_scores, critic

    def _valid_action_mask(self, features: torch.Tensor) -> torch.Tensor:
        """Mask commands that would disturb or double-book live work.

        This is a state-derived *physical* constraint, not a hidden target:
        active-cycle state, current-target indicators, and remaining wood are
        all present in the M2 observation. It blocks another villager from
        claiming an occupied tree and blocks vanished slots, but it leaves
        ``NO_CLICK`` versus an interrupting click as a learned decision.
        """

        batch_size = features.shape[0]
        slices = features.reshape(
            batch_size,
            self.villager_count,
            self.slice_dim,
        )
        carrying = slices[:, :, CARRIED_RESOURCE_INDEX] > 0.0
        cycle_active = slices[:, :, GATHER_CYCLE_ACTIVE_INDEX] > 0.5
        # A chopping villager reserves its observed target tree. Its own
        # re-issued gather command is allowed; any other worker claiming that
        # tree would exceed MaxGatherers=1. Carrying villagers have already
        # left the tree to haul, so its gathering slot is free.
        resource_features = slices[:, :, CORE_WIDTH:].reshape(
            batch_size,
            self.villager_count,
            self.resource_count,
            RESOURCE_WIDTH,
        )
        active_targets = (
            (cycle_active & ~carrying)[:, :, None]
            & (resource_features[:, :, :, CURRENT_TARGET_OFFSET] > 0.5)
        )
        reserved_trees = active_targets.any(dim=1)
        # Resource state is repeated in each villager's relational block.
        # A tree that 0 A.D. removed remains a stable slot, but it cannot be a
        # valid fresh command target. ``any`` makes this robust to a malformed
        # observation while preserving valid availability from the real env.
        available_trees = (
            resource_features[:, :, :, RESOURCE_REMAINING_OFFSET] > 0.0
        ).any(dim=1)
        tree_categories = torch.arange(
            1,
            self.resource_count + 1,
            device=features.device,
            dtype=self.assignment_categories.dtype,
        )
        candidate_tree_use = (
            self.assignment_categories[None, :, :, None]
            == tree_categories[None, None, None, :]
        )
        candidate_uses_reserved_tree = (
            candidate_tree_use
            & reserved_trees[:, None, None, :]
            & ~active_targets[:, None, :, :]
        ).any(dim=(2, 3))
        candidate_uses_depleted_tree = (
            candidate_tree_use & ~available_trees[:, None, None, :]
        ).any(dim=(2, 3))
        return ~(
            candidate_uses_reserved_tree
            | candidate_uses_depleted_tree
        )


def _joint_assignment_shape(
    observation_space: spaces.Space,
    action_space: spaces.Space,
) -> tuple[int, int, int]:
    if not isinstance(action_space, spaces.Discrete):
        raise ValueError("JointAssignmentPolicy requires a Discrete action space")
    if int(action_space.start) != 0:
        raise ValueError("joint assignment action categories must start at zero")
    if not isinstance(observation_space, spaces.Box):
        raise ValueError("JointAssignmentPolicy requires a Box observation space")
    shape = observation_space.shape
    if shape is None or len(shape) != 2:
        raise ValueError("observation space must have shape (villagers, features)")

    villager_count, slice_dim = (int(shape[0]), int(shape[1]))
    if villager_count <= 0 or slice_dim <= CORE_WIDTH:
        raise ValueError("observation space must contain complete villager slices")
    relational_width = slice_dim - CORE_WIDTH
    if relational_width % RESOURCE_WIDTH:
        raise ValueError("observation slices must contain complete resource blocks")
    resource_count = relational_width // RESOURCE_WIDTH
    if resource_count <= 0:
        raise ValueError("observation slices must contain at least one resource")
    expected_actions = joint_assignment_action_count(
        villager_count=villager_count,
        resource_count=resource_count,
    )
    if int(action_space.n) != expected_actions:
        raise ValueError(
            "joint action space does not match the observation's villager/tree shape"
        )
    return villager_count, resource_count, slice_dim


class JointAssignmentActorCriticPolicy(ActorCriticPolicy):
    """PPO policy over whole-team collision-free assignment bundles.

    It uses SB3's normal categorical PPO distribution. The structured
    extractor supplies its logits by summing shared villager/tree scores over
    every valid matching, which creates competition between villagers without
    bespoke probability, entropy, or replay-buffer code.
    """

    def __init__(
        self,
        *args: Any,
        hidden_dim: int = 64,
        initial_no_click_logit: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, Integral)
            or hidden_dim <= 0
        ):
            raise ValueError("hidden_dim must be a positive integer")
        if (
            isinstance(initial_no_click_logit, bool)
            or not isinstance(initial_no_click_logit, Real)
            or not math.isfinite(float(initial_no_click_logit))
        ):
            raise ValueError("initial_no_click_logit must be finite")

        observation_space = kwargs.get("observation_space")
        action_space = kwargs.get("action_space")
        if observation_space is None and len(args) >= 1:
            observation_space = args[0]
        if action_space is None and len(args) >= 2:
            action_space = args[1]
        self._team_shape = _joint_assignment_shape(observation_space, action_space)
        self.hidden_dim = int(hidden_dim)
        self.initial_no_click_logit = float(initial_no_click_logit)
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        return {
            **super()._get_constructor_parameters(),
            "hidden_dim": self.hidden_dim,
            "initial_no_click_logit": self.initial_no_click_logit,
        }

    def _build_mlp_extractor(self) -> None:
        villager_count, resource_count, slice_dim = self._team_shape
        self.mlp_extractor = JointAssignmentExtractor(
            villager_count=villager_count,
            resource_count=resource_count,
            slice_dim=slice_dim,
            hidden_dim=self.hidden_dim,
        )

    def _build(self, lr_schedule: Any) -> None:
        if not isinstance(self.action_dist, CategoricalDistribution):
            raise TypeError("unexpected joint assignment action distribution")
        super()._build(lr_schedule)
        extractor = self.mlp_extractor
        if not isinstance(extractor, JointAssignmentExtractor):
            raise TypeError("unexpected joint assignment extractor")
        villager_count = extractor.assignment_categories.shape[1]
        villager_bits = 1 << torch.arange(
            villager_count,
            device=extractor.assignment_categories.device,
            dtype=torch.long,
        )
        self.action_dist = JointAssignmentDistribution(
            int(self.action_space.n),
            ((extractor.assignment_categories != 0).to(torch.long) * villager_bits)
            .sum(dim=1),
        )
        # The extractor already emits one score for each valid matching. A
        # learned output layer would mix action identities and erase its
        # resource/villager-equivariant construction.
        self.action_net = nn.Identity()
        self._initialize_joint_scores()
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _initialize_joint_scores(self) -> None:
        extractor = self.mlp_extractor
        if not isinstance(extractor, JointAssignmentExtractor):
            raise TypeError("unexpected joint assignment extractor")
        no_click_output = extractor.no_click_head[-1]
        tree_output = extractor.tree_head[-1]
        with torch.no_grad():
            # Start exactly neutral across valid team bundles. Feasibility is
            # architectural (the table and state mask); the desirable full
            # four-tree assignment remains something PPO has to discover.
            no_click_output.weight.zero_()
            no_click_output.bias.fill_(self.initial_no_click_logit)
            tree_output.weight.zero_()
            tree_output.bias.zero_()


# Concise config-facing name used by ``rl.agents.sb3._resolve_policy``.
JointAssignmentPolicy = JointAssignmentActorCriticPolicy
