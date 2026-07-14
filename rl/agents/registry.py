"""Explicit registry connecting config names to agent implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .base import AgentSpec, Policy, Trainer
from .baselines import GatherOraclePolicy, RandomPolicy
from .sb3 import SB3Policy, SB3SACTrainer, load_sb3_sac_policy


class UnknownAgentError(ValueError):
    """Raised when a config names an agent absent from the registry."""


class AgentCapabilityError(ValueError):
    """Raised when an agent does not support the requested operation."""


PolicyFactory = Callable[[AgentSpec, Any, int], Policy]
TrainerFactory = Callable[[], Trainer]
PolicyLoader = Callable[[AgentSpec, str | Path], Policy]
PolicySaver = Callable[[Policy, str | Path], None]


@dataclass(frozen=True, slots=True)
class AgentRegistration:
    """Capabilities supplied by one named agent adapter."""

    policy_factory: PolicyFactory | None = None
    trainer_factory: TrainerFactory | None = None
    loader: PolicyLoader | None = None
    saver: PolicySaver | None = None


def _require_no_parameters(agent: AgentSpec) -> None:
    if agent.parameters:
        unexpected = ", ".join(sorted(agent.parameters))
        raise ValueError(
            f"agent '{agent.name}' has unexpected parameters: {unexpected}"
        )


def _build_random(agent: AgentSpec, env: Any, seed: int) -> Policy:
    _require_no_parameters(agent)
    if not hasattr(env, "action_space"):
        raise TypeError("random policy requires an environment action_space")
    return RandomPolicy(env.action_space, seed=seed)


def _build_oracle(agent: AgentSpec, env: Any, seed: int) -> Policy:
    del env, seed
    _require_no_parameters(agent)
    return GatherOraclePolicy()


def _load_sb3(agent: AgentSpec, path: str | Path) -> Policy:
    del agent
    return load_sb3_sac_policy(path)


def _save_sb3(policy: Policy, path: str | Path) -> None:
    if not isinstance(policy, SB3Policy):
        raise TypeError("'sb3_sac' serializer requires an SB3Policy")
    policy.save(path)


_AGENTS: Mapping[str, AgentRegistration] = MappingProxyType(
    {
        "oracle": AgentRegistration(policy_factory=_build_oracle),
        "random": AgentRegistration(policy_factory=_build_random),
        "sb3_sac": AgentRegistration(
            trainer_factory=SB3SACTrainer,
            loader=_load_sb3,
            saver=_save_sb3,
        ),
    }
)


def available_agents() -> tuple[str, ...]:
    """Return stable names accepted in experiment config files."""

    return tuple(sorted(_AGENTS))


def _registration(agent: AgentSpec) -> AgentRegistration:
    try:
        return _AGENTS[agent.name]
    except KeyError as error:
        choices = ", ".join(available_agents())
        raise UnknownAgentError(
            f"unknown agent '{agent.name}'; available agents: {choices}",
        ) from error


def build_policy(agent: AgentSpec, env: Any, *, seed: int) -> Policy:
    """Construct a policy that does not require training or a checkpoint."""

    factory = _registration(agent).policy_factory
    if factory is None:
        raise AgentCapabilityError(
            f"agent '{agent.name}' cannot be built directly; train or load it",
        )
    return factory(agent, env, seed)


def build_trainer(agent: AgentSpec) -> Trainer:
    """Construct the trainer registered for an agent."""

    factory = _registration(agent).trainer_factory
    if factory is None:
        raise AgentCapabilityError(f"agent '{agent.name}' cannot be trained")
    return factory()


def load_policy(agent: AgentSpec, path: str | Path) -> Policy:
    """Load a checkpoint using the agent's registered serializer."""

    loader = _registration(agent).loader
    if loader is None:
        raise AgentCapabilityError(f"agent '{agent.name}' cannot load checkpoints")
    return loader(agent, path)


def save_policy(agent: AgentSpec, policy: Policy, path: str | Path) -> None:
    """Save a policy using the matching registered serializer."""

    saver = _registration(agent).saver
    if saver is None:
        raise AgentCapabilityError(f"agent '{agent.name}' cannot save checkpoints")
    saver(policy, path)
