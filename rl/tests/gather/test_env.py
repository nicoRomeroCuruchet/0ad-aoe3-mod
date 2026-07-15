import sys
from types import ModuleType

import numpy as np
import pytest

from rl.gather.env import ZeroADGatherEnv


def test_injected_backend_supports_reset_and_step_without_zero_ad(fake_backend):
    game, actions = fake_backend
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=2,
        reach_threshold=80.0,
        sim_steps_per_action=3,
        save_replay=True,
        game=game,
        actions=actions,
    )

    observation, info = env.reset(seed=7)

    assert info == {}
    assert observation.dtype == np.float32
    assert np.allclose(observation, [-1.0, -1.0, 0.0, -1.0, 0.5])
    assert game.reset_calls == [("scenario contents", True)]
    assert game.step_calls == [None]

    observation, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.0], dtype=np.float32)
    )

    assert np.allclose(observation, [-0.75, -1.0, 0.0, -1.0, 0.375])
    assert reward == 25.0
    assert terminated is True
    assert truncated is False
    assert info == {"distance": 75.0}
    assert actions.walk_calls == [([game.initial_state.villager], 100.0, 100.0)]
    command = ("walk", (game.initial_state.villager,), 100.0, 100.0)
    assert game.step_calls == [None, [command], None, None]


def test_horizon_truncates_an_unfinished_episode(fake_backend):
    game, actions = fake_backend
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=1,
        reach_threshold=1.0,
        sim_steps_per_action=1,
        game=game,
        actions=actions,
    )
    env.reset()

    _, _, terminated, truncated, _ = env.step([0.0, 0.0])

    assert terminated is False
    assert truncated is True


def test_close_releases_an_injected_backend_once(fake_backend):
    game, actions = fake_backend
    env = ZeroADGatherEnv("scenario contents", game=game, actions=actions)

    env.close()
    env.close()

    assert game.close_calls == 1


def test_default_backend_reports_how_to_run_without_zero_ad(monkeypatch):
    monkeypatch.setitem(sys.modules, "zero_ad", None)

    with pytest.raises(ModuleNotFoundError, match="inject both game and actions"):
        ZeroADGatherEnv("scenario contents")


def test_default_backend_preserves_zero_ad_client_construction(monkeypatch):
    created_uris = []
    default_game = object()
    default_actions = object()
    zero_ad = ModuleType("zero_ad")

    def create_game(uri):
        created_uris.append(uri)
        return default_game

    zero_ad.ZeroAD = create_game
    zero_ad.actions = default_actions
    monkeypatch.setitem(sys.modules, "zero_ad", zero_ad)

    env = ZeroADGatherEnv("scenario contents", uri="http://localhost:7000")

    assert created_uris == ["http://localhost:7000"]
    assert env.game is default_game
    assert env.actions is default_actions
