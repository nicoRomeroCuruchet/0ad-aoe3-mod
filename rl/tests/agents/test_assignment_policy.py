import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    MultiCategoricalDistribution,
)

from rl.agents.assignment_policy import (
    JointAssignmentActorCriticPolicy,
    JointAssignmentDistribution,
    TeamAssignmentActorCriticPolicy,
    TeamAssignmentDistribution,
    TeamAssignmentExtractor,
)
from rl.gather.assignment_actions import joint_assignment_from_index


CORE_WIDTH = 11
RESOURCE_WIDTH = 5


def _policy(
    villager_count: int = 4,
    resource_count: int = 4,
    **policy_kwargs,
) -> TeamAssignmentActorCriticPolicy:
    slice_dim = CORE_WIDTH + RESOURCE_WIDTH * resource_count
    return TeamAssignmentActorCriticPolicy(
        observation_space=spaces.Box(
            -1.0,
            1.0,
            shape=(villager_count, slice_dim),
            dtype=np.float32,
        ),
        action_space=spaces.MultiDiscrete(
            np.full(villager_count, resource_count + 1, dtype=np.int64)
        ),
        lr_schedule=lambda _progress: 3e-4,
        **policy_kwargs,
    )


def _joint_policy(
    **policy_kwargs,
) -> JointAssignmentActorCriticPolicy:
    """Build the fixed 4-villager/4-tree M2 joint-action policy."""

    return JointAssignmentActorCriticPolicy(
        observation_space=spaces.Box(
            -1.0,
            1.0,
            shape=(4, CORE_WIDTH + 4 * RESOURCE_WIDTH),
            dtype=np.float32,
        ),
        # There are sum(C(4, k) * P(4, k), k=0..4) = 209 partial,
        # injective assignments, including the all-NO_CLICK action.
        action_space=spaces.Discrete(209),
        lr_schedule=lambda _progress: 3e-4,
        **policy_kwargs,
    )


def _logits(
    policy: TeamAssignmentActorCriticPolicy,
    observation: torch.Tensor,
) -> torch.Tensor:
    distribution = policy.get_distribution(observation)
    assert isinstance(distribution, MultiCategoricalDistribution)
    assert distribution.distribution is not None
    return torch.stack(
        [categorical.logits for categorical in distribution.distribution],
        dim=1,
    )


def _probabilities(
    policy: TeamAssignmentActorCriticPolicy,
    observation: torch.Tensor,
) -> torch.Tensor:
    distribution = policy.get_distribution(observation)
    assert isinstance(distribution, MultiCategoricalDistribution)
    assert distribution.distribution is not None
    return torch.stack(
        [categorical.probs for categorical in distribution.distribution],
        dim=1,
    )


def _permute_resources(
    observation: torch.Tensor,
    resource_order: torch.Tensor,
) -> torch.Tensor:
    core = observation[:, :, :CORE_WIDTH]
    resources = observation[:, :, CORE_WIDTH:].reshape(
        observation.shape[0],
        observation.shape[1],
        len(resource_order),
        RESOURCE_WIDTH,
    )
    reordered = resources[:, :, resource_order, :].reshape(
        observation.shape[0], observation.shape[1], -1
    )
    return torch.cat((core, reordered), dim=-1)


def test_policy_emits_one_native_categorical_assignment_per_villager():
    torch.manual_seed(7)
    policy = _policy()
    observation = torch.rand(6, 4, 31) * 2.0 - 1.0

    with torch.no_grad():
        actions, values, log_probability = policy(observation)

    assert isinstance(policy.action_dist, MultiCategoricalDistribution)
    assert actions.shape == (6, 4)
    assert actions.dtype == torch.int64
    assert values.shape == (6, 1)
    assert log_probability.shape == (6,)
    for action in actions.cpu().numpy():
        assert policy.action_space.contains(action)


def test_joint_policy_emits_a_single_categorical_over_all_209_matchings():
    torch.manual_seed(31)
    policy = _joint_policy()
    observation = torch.rand(6, 4, 31) * 2.0 - 1.0

    with torch.no_grad():
        distribution = policy.get_distribution(observation)
        actions, values, log_probability = policy(observation)

    assert isinstance(policy.action_dist, CategoricalDistribution)
    assert isinstance(distribution, CategoricalDistribution)
    assert isinstance(distribution, JointAssignmentDistribution)
    assert distribution.distribution is not None
    assert distribution.distribution.logits.shape == (6, 209)
    assert actions.shape == (6,)
    assert actions.dtype == torch.int64
    assert values.shape == (6, 1)
    assert log_probability.shape == (6,)
    assert torch.all((actions >= 0) & (actions < 209))


def test_joint_policy_scales_to_the_six_tree_m2_action_table():
    policy = JointAssignmentActorCriticPolicy(
        observation_space=spaces.Box(
            -1.0,
            1.0,
            shape=(4, CORE_WIDTH + 6 * RESOURCE_WIDTH),
            dtype=np.float32,
        ),
        action_space=spaces.Discrete(1045),
        lr_schedule=lambda _progress: 3e-4,
    )
    observation = torch.zeros(2, 4, CORE_WIDTH + 6 * RESOURCE_WIDTH)
    for resource in range(6):
        observation[:, :, CORE_WIDTH + resource * RESOURCE_WIDTH + 4] = 1.0

    with torch.no_grad():
        distribution = policy.get_distribution(observation)
        actions, _values, _log_probability = policy(observation)

    assert isinstance(distribution, CategoricalDistribution)
    assert distribution.distribution is not None
    assert distribution.distribution.logits.shape == (2, 1045)
    assert torch.all((actions >= 0) & (actions < 1045))


def test_joint_policy_deterministic_actions_always_decode_to_injective_matchings():
    torch.manual_seed(37)
    policy = _joint_policy()
    observations = torch.rand(8, 4, 31) * 2.0 - 1.0

    with torch.no_grad():
        actions, _values, _log_probability = policy(
            observations,
            deterministic=True,
        )

    for action in actions.cpu().numpy():
        assignment = joint_assignment_from_index(
            int(action),
            villager_count=4,
            resource_count=4,
        )
        selected_trees = assignment[assignment > 0]
        assert assignment.shape == (4,)
        assert len(selected_trees) == len(set(selected_trees.tolist()))


def test_joint_deterministic_mode_marginalizes_click_pattern_before_bundle_choice():
    """One no-op label must not beat the mass of many equivalent bundles."""

    categories = torch.as_tensor(
        np.stack(
            [
                joint_assignment_from_index(
                    index,
                    villager_count=4,
                    resource_count=4,
                )
                for index in range(209)
            ]
        ),
        dtype=torch.long,
    )
    click_patterns = (
        (categories != 0).to(torch.long)
        * (1 << torch.arange(4, dtype=torch.long))
    ).sum(dim=1)
    distribution = JointAssignmentDistribution(209, click_patterns)
    logits = torch.full((1, 209), -50.0)
    logits[0, 0] = 0.0
    full_clicks = (categories != 0).all(dim=1)
    three_clicks = (categories != 0).sum(dim=1) == 3
    # All three-click labels together have more mass than full-click labels,
    # but that mass is split across four different villagers-to-click masks.
    # The global controller must choose the strongest *mask* before assigning
    # trees, rather than merely choosing a click count.
    logits[0, full_clicks] = -2.0
    logits[0, three_clicks] = -2.2
    distribution.proba_distribution(logits)

    assert distribution.distribution is not None
    probabilities = distribution.distribution.probs
    # The all-no-click *label* is likelier than any individual 3-click label.
    assert probabilities[0, 0] > probabilities[0, three_clicks].max()
    # But the full-team click mask has the most aggregate compatible mass, so
    # deterministic evaluation must enter that expert.
    action = int(distribution.mode().item())
    assignment = joint_assignment_from_index(
        action,
        villager_count=4,
        resource_count=4,
    )
    assert np.count_nonzero(assignment) == 4


def test_joint_deterministic_mode_stays_silent_when_no_click_mass_wins():
    categories = torch.as_tensor(
        np.stack(
            [
                joint_assignment_from_index(
                    index,
                    villager_count=4,
                    resource_count=4,
                )
                for index in range(209)
            ]
        ),
        dtype=torch.long,
    )
    click_patterns = (
        (categories != 0).to(torch.long)
        * (1 << torch.arange(4, dtype=torch.long))
    ).sum(dim=1)
    distribution = JointAssignmentDistribution(209, click_patterns)
    logits = torch.full((1, 209), -50.0)
    logits[0, 0] = 0.0
    distribution.proba_distribution(logits)

    assert int(distribution.mode().item()) == 0


def test_joint_hierarchical_mode_keeps_native_categorical_training_math():
    categories = torch.as_tensor(
        np.stack(
            [
                joint_assignment_from_index(
                    index,
                    villager_count=4,
                    resource_count=4,
                )
                for index in range(209)
            ]
        ),
        dtype=torch.long,
    )
    logits = torch.randn(3, 209)
    click_patterns = (
        (categories != 0).to(torch.long)
        * (1 << torch.arange(4, dtype=torch.long))
    ).sum(dim=1)
    distribution = JointAssignmentDistribution(209, click_patterns).proba_distribution(
        logits
    )
    reference = CategoricalDistribution(209).proba_distribution(logits)
    actions = torch.tensor([0, 37, 208], dtype=torch.int64)

    torch.testing.assert_close(
        distribution.log_prob(actions),
        reference.log_prob(actions),
    )
    torch.testing.assert_close(distribution.entropy(), reference.entropy())
    torch.manual_seed(27)
    sample = distribution.sample()
    torch.manual_seed(27)
    reference_sample = reference.sample()
    torch.testing.assert_close(sample, reference_sample)


def _joint_probabilities(
    policy: JointAssignmentActorCriticPolicy,
    observation: torch.Tensor,
) -> torch.Tensor:
    distribution = policy.get_distribution(observation)
    assert isinstance(distribution, CategoricalDistribution)
    assert distribution.distribution is not None
    return distribution.distribution.probs


def _joint_indices_matching(predicate) -> list[int]:
    return [
        index
        for index in range(209)
        if predicate(
            joint_assignment_from_index(
                index,
                villager_count=4,
                resource_count=4,
            )
        )
    ]


def test_joint_policy_masks_commands_that_reassign_another_worker_to_active_tree():
    policy = _joint_policy()
    observation = torch.zeros(1, 4, 31)
    # Villager 0 is already chopping TREE_2. Its own click/no-click timing is
    # learned, but another villager cannot claim its occupied tree.
    observation[0, 0, 9] = 1.0
    observation[0, 0, CORE_WIDTH + 2 * RESOURCE_WIDTH + 3] = 1.0

    probabilities = _joint_probabilities(policy, observation)
    invalid = _joint_indices_matching(
        lambda assignment: any(int(assignment[villager]) == 3 for villager in range(1, 4))
    )
    valid = _joint_indices_matching(
        lambda assignment: all(int(assignment[villager]) != 3 for villager in range(1, 4))
    )

    assert invalid
    assert valid
    torch.testing.assert_close(
        probabilities[0, invalid],
        torch.zeros(len(invalid)),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        probabilities[0, valid].sum(),
        torch.tensor(1.0),
        atol=1e-6,
        rtol=0.0,
    )


def test_joint_policy_leaves_carrying_no_click_as_a_learned_choice():
    policy = _joint_policy()
    observation = torch.zeros(1, 4, 31)
    for villager in range(4):
        for resource in range(4):
            observation[
                0,
                villager,
                CORE_WIDTH + resource * RESOURCE_WIDTH + 4,
            ] = 1.0
    # Carrying wood makes clicking harmful, but it is not an impossible engine
    # command. M2 must learn NO_CLICK here rather than receive that behavior
    # for free from a policy-side mask.
    observation[0, 2, 5] = 1.0

    probabilities = _joint_probabilities(policy, observation)
    clicking = _joint_indices_matching(lambda assignment: assignment[2] != 0)
    no_click = _joint_indices_matching(lambda assignment: assignment[2] == 0)

    assert clicking
    assert no_click
    assert probabilities[0, clicking].sum() > 0.0
    assert probabilities[0, no_click].sum() > 0.0


def test_joint_policy_masks_categories_for_a_depleted_fixed_tree_slot():
    policy = _joint_policy()
    observation = torch.zeros(1, 4, 31)
    # Tree slots 0, 2, and 3 remain harvestable; slot 1 has been destroyed.
    # Each relational row carries the same global remaining amount.
    for villager in range(4):
        for resource in (0, 2, 3):
            observation[
                0,
                villager,
                CORE_WIDTH + resource * RESOURCE_WIDTH + 4,
            ] = 1.0

    probabilities = _joint_probabilities(policy, observation)
    invalid = _joint_indices_matching(
        lambda assignment: any(int(choice) == 2 for choice in assignment)
    )
    valid = _joint_indices_matching(
        lambda assignment: all(int(choice) != 2 for choice in assignment)
    )

    assert invalid
    assert valid
    torch.testing.assert_close(
        probabilities[0, invalid],
        torch.zeros(len(invalid)),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        probabilities[0, valid].sum(),
        torch.tensor(1.0),
        atol=1e-6,
        rtol=0.0,
    )


def test_joint_policy_starts_neutral_so_it_does_not_bake_in_a_solution():
    policy = _joint_policy()
    observation = torch.zeros(1, 4, 31)
    for villager in range(4):
        for resource in range(4):
            observation[
                0,
                villager,
                CORE_WIDTH + resource * RESOURCE_WIDTH + 4,
            ] = 1.0

    probabilities = _joint_probabilities(policy, observation)
    action, _state = policy.predict(
        observation.numpy()[0],
        deterministic=True,
    )

    # The action table encodes physical feasibility, but it must not embed a
    # preferred matching. PPO explores all valid bundles stochastically and
    # must learn which coordinated action advances the task.
    torch.testing.assert_close(
        probabilities,
        torch.full((1, 209), 1.0 / 209.0),
        atol=1e-6,
        rtol=0.0,
    )
    assert int(action) == 0


def test_joint_policy_save_load_preserves_nondefault_hidden_width(tmp_path):
    torch.manual_seed(41)
    policy = _joint_policy(hidden_dim=32)
    checkpoint = tmp_path / "joint_assignment_policy.pt"
    observation = np.zeros((4, 31), dtype=np.float32)

    policy.save(checkpoint)
    restored = JointAssignmentActorCriticPolicy.load(checkpoint)

    assert restored.hidden_dim == 32
    assert restored.mlp_extractor.hidden_dim == 32
    original_action, _state = policy.predict(observation, deterministic=True)
    restored_action, _state = restored.predict(observation, deterministic=True)
    np.testing.assert_array_equal(restored_action, original_action)
    assignment = joint_assignment_from_index(
        int(restored_action),
        villager_count=4,
        resource_count=4,
    )
    selected_trees = assignment[assignment > 0]
    assert len(selected_trees) == len(set(selected_trees.tolist()))


def test_hierarchical_mode_does_not_change_stochastic_distribution_math():
    torch.manual_seed(9)
    policy = _policy()
    observation = torch.rand(3, 4, 31) * 2.0 - 1.0
    distribution = policy.get_distribution(observation)
    assert isinstance(distribution, TeamAssignmentDistribution)
    logits = torch.cat(
        [categorical.logits for categorical in distribution.distribution],
        dim=1,
    )
    reference = MultiCategoricalDistribution([5, 5, 5, 5]).proba_distribution(
        logits
    )
    actions = torch.tensor(
        [[0, 1, 2, 3], [4, 3, 2, 1], [1, 1, 0, 4]],
        dtype=torch.int64,
    )

    torch.testing.assert_close(
        distribution.log_prob(actions),
        reference.log_prob(actions),
    )
    torch.testing.assert_close(distribution.entropy(), reference.entropy())
    torch.manual_seed(23)
    sample = distribution.sample()
    torch.manual_seed(23)
    reference_sample = reference.sample()
    torch.testing.assert_close(sample, reference_sample)


def test_extractor_emits_all_no_click_and_tree_logits_without_a_slot_head():
    extractor = TeamAssignmentExtractor(
        villager_count=3,
        resource_count=5,
        slice_dim=CORE_WIDTH + 5 * RESOURCE_WIDTH,
    )
    features = torch.randn(2, 3 * (CORE_WIDTH + 5 * RESOURCE_WIDTH))

    actor_latent, critic_latent = extractor(features)

    assert actor_latent.shape == (2, 3 * 6)
    assert critic_latent.shape == (2, extractor.latent_dim_vf)
    assert extractor.latent_dim_pi == 3 * 6


def test_initial_distribution_prefers_no_click_without_preferring_a_tree():
    policy = _policy(no_click_probability=0.8)
    observation = torch.zeros(2, 4, 31)

    with torch.no_grad():
        probabilities = _probabilities(policy, observation)

    torch.testing.assert_close(
        probabilities[:, :, 0],
        torch.full((2, 4), 0.8),
        atol=0.02,
        rtol=0.0,
    )
    torch.testing.assert_close(
        probabilities[:, :, 1:],
        torch.full((2, 4, 4), 0.05),
        atol=0.01,
        rtol=0.0,
    )


def test_deterministic_prediction_uses_click_mass_before_choosing_a_tree():
    policy = _policy()
    extractor = policy.mlp_extractor
    with torch.no_grad():
        extractor.no_click_head[-1].weight.zero_()
        extractor.no_click_head[-1].bias.fill_(np.log(0.4))
        extractor.tree_head[-1].weight.zero_()
        extractor.tree_head[-1].bias.fill_(np.log(0.15))

    # No individual tree beats NO_CLICK (0.15 < 0.40), but the click expert
    # owns 0.60 total probability and therefore wins the coarse gate.
    action, _state = policy.predict(
        np.zeros((4, 31), dtype=np.float32),
        deterministic=True,
    )
    with torch.no_grad():
        forward_action, _value, _log_probability = policy(
            torch.zeros(1, 4, 31),
            deterministic=True,
        )

    np.testing.assert_array_equal(action, np.ones(4, dtype=np.int64))
    np.testing.assert_array_equal(
        forward_action.cpu().numpy(),
        np.ones((1, 4), dtype=np.int64),
    )


def test_deterministic_prediction_stays_silent_when_no_click_mass_wins():
    policy = _policy()
    extractor = policy.mlp_extractor
    with torch.no_grad():
        extractor.no_click_head[-1].weight.zero_()
        extractor.no_click_head[-1].bias.fill_(np.log(0.6))
        extractor.tree_head[-1].weight.zero_()
        extractor.tree_head[-1].bias.fill_(np.log(0.1))

    action, _state = policy.predict(
        np.zeros((4, 31), dtype=np.float32),
        deterministic=True,
    )

    np.testing.assert_array_equal(action, np.zeros(4, dtype=np.int64))


def test_direct_policy_save_preserves_custom_architecture_parameters(tmp_path):
    policy = _policy(hidden_dim=32, no_click_probability=0.7)
    extractor = policy.mlp_extractor
    with torch.no_grad():
        extractor.no_click_head[-1].weight.zero_()
        extractor.no_click_head[-1].bias.fill_(np.log(0.4))
        extractor.tree_head[-1].weight.zero_()
        extractor.tree_head[-1].bias.fill_(np.log(0.15))
    checkpoint = tmp_path / "assignment_policy.pt"

    policy.save(checkpoint)
    restored = TeamAssignmentActorCriticPolicy.load(checkpoint)

    assert restored.hidden_dim == 32
    assert restored.no_click_probability == pytest.approx(0.7)
    assert restored.mlp_extractor.hidden_dim == 32
    assert isinstance(restored.action_dist, TeamAssignmentDistribution)
    with torch.no_grad():
        expected = _probabilities(policy, torch.zeros(1, 4, 31))
        actual = _probabilities(restored, torch.zeros(1, 4, 31))
    torch.testing.assert_close(actual, expected)
    action, _state = restored.predict(
        np.zeros((4, 31), dtype=np.float32),
        deterministic=True,
    )
    np.testing.assert_array_equal(action, np.ones(4, dtype=np.int64))


def test_permuting_villagers_permutes_actor_rows():
    torch.manual_seed(11)
    policy = _policy()
    observation = torch.rand(3, 4, 31) * 2.0 - 1.0
    villager_order = torch.tensor([2, 0, 3, 1])

    with torch.no_grad():
        original = _logits(policy, observation)
        permuted = _logits(policy, observation[:, villager_order, :])

    torch.testing.assert_close(
        permuted,
        original[:, villager_order, :],
        atol=1e-6,
        rtol=1e-6,
    )


def test_permuting_tree_blocks_permutes_only_tree_logits():
    torch.manual_seed(13)
    policy = _policy()
    observation = torch.rand(3, 4, 31) * 2.0 - 1.0
    resource_order = torch.tensor([2, 0, 3, 1])

    with torch.no_grad():
        original = _logits(policy, observation)
        permuted = _logits(
            policy,
            _permute_resources(observation, resource_order),
        )

    torch.testing.assert_close(
        permuted[:, :, 0],
        original[:, :, 0],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        permuted[:, :, 1:],
        original[:, :, 1:][:, :, resource_order],
        atol=1e-6,
        rtol=1e-6,
    )


def test_critic_is_invariant_to_villager_and_tree_order():
    torch.manual_seed(17)
    policy = _policy()
    observation = torch.rand(3, 4, 31) * 2.0 - 1.0
    villager_order = torch.tensor([2, 0, 3, 1])
    resource_order = torch.tensor([2, 0, 3, 1])
    reordered = _permute_resources(
        observation[:, villager_order, :],
        resource_order,
    )

    with torch.no_grad():
        original_value = policy.predict_values(observation)
        reordered_value = policy.predict_values(reordered)

    torch.testing.assert_close(
        reordered_value,
        original_value,
        atol=1e-6,
        rtol=1e-6,
    )


def test_a_teammate_can_change_another_villagers_assignment_logits():
    policy = _policy()
    with torch.no_grad():
        for parameter in policy.mlp_extractor.parameters():
            parameter.fill_(0.05)

    observation = torch.zeros(1, 4, 31)
    changed_observation = observation.clone()
    changed_observation[0, 1, 0] = 0.75

    with torch.no_grad():
        original = _logits(policy, observation)
        changed = _logits(policy, changed_observation)

    assert not torch.allclose(changed[:, 0, :], original[:, 0, :])


def test_actor_does_not_use_the_arbitrary_agent_id_feature():
    torch.manual_seed(19)
    policy = _policy()
    observation = torch.rand(2, 4, 31) * 2.0 - 1.0
    changed_ids = observation.clone()
    changed_ids[:, :, 10] = torch.tensor([1.0, -1.0, 0.5, -0.5])

    with torch.no_grad():
        original = _logits(policy, observation)
        changed = _logits(policy, changed_ids)

    torch.testing.assert_close(changed, original, atol=0.0, rtol=0.0)


def test_policy_rejects_a_continuous_action_space():
    with pytest.raises(ValueError, match="MultiDiscrete"):
        TeamAssignmentActorCriticPolicy(
            observation_space=spaces.Box(
                -1.0,
                1.0,
                shape=(4, 31),
                dtype=np.float32,
            ),
            action_space=spaces.Box(
                -1.0,
                1.0,
                shape=(4,),
                dtype=np.float32,
            ),
            lr_schedule=lambda _progress: 3e-4,
        )


def test_sb3_resolves_the_assignment_policy_name():
    from rl.agents.sb3 import _resolve_policy

    assert _resolve_policy("TeamAssignmentPolicy") is TeamAssignmentActorCriticPolicy


@pytest.mark.parametrize(
    ("observation_shape", "action_categories"),
    [
        ((124,), (5, 5, 5, 5)),
        ((4, 30), (5, 5, 5, 5)),
        ((4, 31), (5, 5, 5)),
        ((4, 31), (5, 4, 5, 5)),
    ],
    ids=[
        "flat-observation",
        "invalid-resource-width",
        "wrong-villager-count",
        "unequal-resource-counts",
    ],
)
def test_policy_rejects_incompatible_observation_and_action_shapes(
    observation_shape: tuple[int, ...],
    action_categories: tuple[int, ...],
):
    with pytest.raises(ValueError):
        TeamAssignmentActorCriticPolicy(
            observation_space=spaces.Box(
                -1.0,
                1.0,
                shape=observation_shape,
                dtype=np.float32,
            ),
            action_space=spaces.MultiDiscrete(
                np.asarray(action_categories, dtype=np.int64)
            ),
            lr_schedule=lambda _progress: 3e-4,
        )
