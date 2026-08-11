import pytest

from rl.gather.reward import (
    TeamRewardScales,
    TeamRewardTerms,
    VillagerRewardInputs,
    compose_team_reward,
)


SCALES = TeamRewardScales(
    distance_shaping_scale=0.02,
    carried_resource_delta_reward_scale=0.2,
    click_gather_cycle_penalty=1.0,
    carrying_no_click_reward=0.1,
)


def _villager(closed=0.0, carried_delta=0.0, interrupted=False, holding=False):
    return VillagerRewardInputs(
        distance_closed_m=closed,
        carried_resource_delta=carried_delta,
        disrupted_while_busy=interrupted,
        carrying_without_command=holding,
    )


def test_stock_delta_passes_through_unscaled():
    terms = compose_team_reward(20.0, (_villager(), _villager()), SCALES)

    assert terms.stock_delta == pytest.approx(20.0)
    assert terms.total() == pytest.approx(20.0)


def test_shaping_is_averaged_over_villagers_not_summed():
    villagers = (_villager(closed=100.0), _villager(closed=100.0))

    terms = compose_team_reward(0.0, villagers, SCALES)

    # 0.02 * 100 = 2.0 each; averaged, not summed.
    assert terms.distance_shaping == pytest.approx(2.0)
    assert terms.total() == pytest.approx(2.0)


def test_carried_delta_is_averaged_and_never_negative():
    villagers = (_villager(carried_delta=20.0), _villager(carried_delta=-5.0))

    terms = compose_team_reward(0.0, villagers, SCALES)

    # 0.2 * 20 = 4.0 for the first, 0.0 for the second, averaged over two.
    assert terms.carried_delta == pytest.approx(2.0)


def test_interruption_penalty_is_averaged_and_subtracted():
    villagers = (_villager(interrupted=True), _villager(), _villager(), _villager())

    terms = compose_team_reward(0.0, villagers, SCALES)

    assert terms.click_penalty == pytest.approx(0.25)
    assert terms.total() == pytest.approx(-0.25)


def test_holding_a_load_untouched_is_paid_and_averaged():
    villagers = (_villager(holding=True), _villager(), _villager(), _villager())

    terms = compose_team_reward(0.0, villagers, SCALES)

    assert terms.carrying_no_click == pytest.approx(0.025)
    assert terms.total() == pytest.approx(0.025)


def test_terms_sum_into_the_total():
    villagers = (_villager(closed=50.0, carried_delta=20.0, interrupted=True),)

    terms = compose_team_reward(20.0, villagers, SCALES)

    assert isinstance(terms, TeamRewardTerms)
    assert terms.total() == pytest.approx(20.0 + 1.0 + 4.0 - 1.0)


def test_composition_requires_at_least_one_villager():
    with pytest.raises(ValueError, match="at least one villager"):
        compose_team_reward(0.0, (), SCALES)
