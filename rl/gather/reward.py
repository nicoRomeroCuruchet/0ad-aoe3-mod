"""Reward term composition for team gather environments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TeamRewardScales:
    """Config-owned weights for every shaping term."""

    distance_shaping_scale: float
    carried_resource_delta_reward_scale: float
    click_gather_cycle_penalty: float
    carrying_no_click_reward: float = 0.0


@dataclass(frozen=True, slots=True)
class VillagerRewardInputs:
    """One villager's contribution to this step's shaping."""

    distance_closed_m: float = 0.0
    carried_resource_delta: float = 0.0
    disrupted_while_busy: bool = False
    carrying_without_command: bool = False


@dataclass(frozen=True, slots=True)
class TeamRewardTerms:
    """The reward broken into terms, for logging and for ``--verbose``."""

    stock_delta: float
    distance_shaping: float
    carried_delta: float
    click_penalty: float
    carrying_no_click: float = 0.0

    def total(self) -> float:
        """Sum the terms the environment actually returns."""

        return (
            self.stock_delta
            + self.distance_shaping
            + self.carried_delta
            + self.carrying_no_click
            - self.click_penalty
        )


def compose_team_reward(
    stock_delta: float,
    villagers: tuple[VillagerRewardInputs, ...],
    scales: TeamRewardScales,
) -> TeamRewardTerms:
    """Combine the team stock signal with averaged per-villager shaping.

    Shaping is averaged rather than summed: the team stock delta does not grow
    with the squad size, so summing would let shaping dominate as villagers are
    added. The M1 ablation showed a policy farming shaping instead of gathering
    once shaping outweighed the real signal.

    ``carrying_without_command`` pays a villager for being left alone while it
    holds a load, because that silence is exactly what lets UnitAI finish the
    haul. Without it the cheapest behaviour is to chop one load and then keep
    issuing commands forever, which collects the carried-resource shaping and
    never delivers.
    """

    if not villagers:
        raise ValueError("team reward needs at least one villager")
    count = len(villagers)
    distance_shaping = (
        sum(villager.distance_closed_m for villager in villagers)
        * scales.distance_shaping_scale
        / count
    )
    carried_delta = (
        sum(max(0.0, villager.carried_resource_delta) for villager in villagers)
        * scales.carried_resource_delta_reward_scale
        / count
    )
    click_penalty = (
        sum(1.0 for villager in villagers if villager.disrupted_while_busy)
        * scales.click_gather_cycle_penalty
        / count
    )
    carrying_no_click = (
        sum(1.0 for villager in villagers if villager.carrying_without_command)
        * scales.carrying_no_click_reward
        / count
    )
    return TeamRewardTerms(
        stock_delta=float(stock_delta),
        distance_shaping=float(distance_shaping),
        carried_delta=float(carried_delta),
        click_penalty=float(click_penalty),
        carrying_no_click=float(carrying_no_click),
    )
