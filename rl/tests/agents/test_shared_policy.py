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


class _FakeM1Policy(torch.nn.Module):
    def __init__(self, obs_dim=10, hidden=64):
        super().__init__()
        self.mlp_extractor = torch.nn.Module()
        self.mlp_extractor.policy_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
        )
        self.action_net = torch.nn.Linear(hidden, 3)
        self.log_std = torch.nn.Parameter(torch.full((3,), -0.5))


class _FakeM1Model:
    def __init__(self, policy):
        self.policy = policy


def _loader(policy):
    return lambda _path: _FakeM1Model(policy)


def test_m1_weights_land_in_the_input_prefix_and_new_inputs_start_at_zero():
    from rl.agents.shared_policy import initialize_from_m1

    policy = _policy()
    m1 = _FakeM1Policy()

    initialize_from_m1(policy, "unused", loader=_loader(m1))

    first = policy.mlp_extractor.villager_trunk[0]
    assert torch.allclose(first.weight[:, :10], m1.mlp_extractor.policy_net[0].weight)
    # Agent id and relational block contribute nothing until training moves them.
    assert torch.count_nonzero(first.weight[:, 10:]) == 0
    assert torch.allclose(first.bias, m1.mlp_extractor.policy_net[0].bias)


def test_m1_hidden_layer_and_action_head_copy_exactly():
    from rl.agents.shared_policy import initialize_from_m1

    policy = _policy()
    m1 = _FakeM1Policy()

    initialize_from_m1(policy, "unused", loader=_loader(m1))

    assert torch.allclose(
        policy.mlp_extractor.villager_trunk[2].weight,
        m1.mlp_extractor.policy_net[2].weight,
    )
    assert torch.allclose(policy.mlp_extractor.villager_head.weight, m1.action_net.weight)
    assert torch.allclose(policy.mlp_extractor.villager_head.bias, m1.action_net.bias)


def test_log_std_is_repeated_once_per_villager():
    from rl.agents.shared_policy import initialize_from_m1

    policy = _policy(villager_count=4)

    initialize_from_m1(policy, "unused", loader=_loader(_FakeM1Policy()))

    assert policy.log_std.shape == (12,)
    assert torch.allclose(policy.log_std, torch.full((12,), -0.5))


def test_a_warm_started_policy_reproduces_m1_on_the_prefix():
    from rl.agents.shared_policy import initialize_from_m1

    policy = _policy()
    m1 = _FakeM1Policy()
    initialize_from_m1(policy, "unused", loader=_loader(m1))

    core = torch.randn(1, 10)
    slice_values = torch.cat([core, torch.randn(1, 21)], dim=1)
    observation = slice_values.repeat(1, 4).reshape(1, 4, 31)

    with torch.no_grad():
        actions, _values, _log_prob = policy(observation, deterministic=True)
        expected = m1.action_net(m1.mlp_extractor.policy_net(core))

    # The appended inputs are ignored at initialization, so M1's behaviour is
    # reproduced exactly whatever they contain.
    assert torch.allclose(actions.reshape(4, 3)[0], expected[0], atol=1e-6)


def test_transfer_rejects_a_wider_m1_observation():
    from rl.agents.shared_policy import M1TransferError, initialize_from_m1

    policy = _policy(slice_dim=8)

    with pytest.raises(M1TransferError, match="only 8"):
        initialize_from_m1(policy, "unused", loader=_loader(_FakeM1Policy(obs_dim=10)))
