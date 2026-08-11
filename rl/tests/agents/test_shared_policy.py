import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl.agents.shared_policy import (
    SharedVillagerActorCriticPolicy,
    SharedVillagerExtractor,
)


def _policy(villager_count=4, slice_dim=31):
    return SharedVillagerActorCriticPolicy(
        observation_space=spaces.Box(-1.0, 1.0, shape=(villager_count, slice_dim), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, shape=(3 * villager_count,), dtype=np.float32),
        lr_schedule=lambda _progress: 3e-4,
    )


def test_identical_slices_produce_identical_actions():
    policy = _policy()
    one_slice = torch.randn(1, 31)
    observation = one_slice.repeat(1, 4).reshape(1, 4, 31)

    with torch.no_grad():
        actions, _values, _log_prob = policy(observation, deterministic=True)

    per_villager = actions.reshape(4, 3)
    for villager in range(1, 4):
        # Same weights, same input -> same output. A joint MLP cannot promise this.
        assert torch.allclose(per_villager[0], per_villager[villager], atol=1e-6)


def test_action_shape_matches_three_values_per_villager():
    policy = _policy()
    observation = torch.randn(5, 4, 31)

    with torch.no_grad():
        actions, values, _log_prob = policy(observation)

    assert actions.shape == (5, 12)
    assert values.shape == (5, 1)


def test_the_actor_head_is_shared_not_per_slot():
    extractor = SharedVillagerExtractor(villager_count=4, slice_dim=31)

    names = {name for name, _ in extractor.named_parameters()}

    # One trunk and one head, not four of each.
    assert any(name.startswith("villager_trunk") for name in names)
    assert any(name.startswith("villager_head") for name in names)
    head_out = extractor.villager_head.out_features
    assert head_out == 3


def test_actor_parameter_count_is_independent_of_team_size():
    small = SharedVillagerExtractor(villager_count=2, slice_dim=31)
    large = SharedVillagerExtractor(villager_count=8, slice_dim=31)

    def actor_parameters(module):
        return sum(
            parameter.numel()
            for name, parameter in module.named_parameters()
            if name.startswith(("villager_trunk", "villager_head"))
        )

    assert actor_parameters(small) == actor_parameters(large)


def test_a_different_slice_changes_only_that_villager():
    policy = _policy()
    observation = torch.zeros(1, 4, 31)

    with torch.no_grad():
        baseline, _values, _log_prob = policy(observation, deterministic=True)
        observation[0, 2, 0] = 1.0
        changed, _values, _log_prob = policy(observation, deterministic=True)

    baseline = baseline.reshape(4, 3)
    changed = changed.reshape(4, 3)
    assert not torch.allclose(baseline[2], changed[2])
    for villager in (0, 1, 3):
        assert torch.allclose(baseline[villager], changed[villager], atol=1e-6)


def test_policy_rejects_a_flat_observation_space():
    with pytest.raises(ValueError, match="villagers, features"):
        SharedVillagerActorCriticPolicy(
            observation_space=spaces.Box(-1.0, 1.0, shape=(124,), dtype=np.float32),
            action_space=spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32),
            lr_schedule=lambda _progress: 3e-4,
        )


def test_sb3_resolves_the_policy_name():
    from rl.agents.sb3 import _resolve_policy

    assert _resolve_policy("SharedVillagerPolicy") is SharedVillagerActorCriticPolicy
    assert _resolve_policy("MlpPolicy") == "MlpPolicy"
