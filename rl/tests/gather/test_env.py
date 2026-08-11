import sys
from dataclasses import replace
from types import ModuleType

import numpy as np
import pytest

from rl.gather.env import ZeroADGatherEnv
from rl.gather.zero_ad_client import BatchedZeroAD


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
    assert game.step_calls == []

    observation, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.0], dtype=np.float32)
    )

    assert np.allclose(observation, [-0.75, -1.0, 0.0, -1.0, 0.375])
    assert reward == 25.0
    assert terminated is True
    assert truncated is False
    assert info == {
        "command": "walk",
        "distance": 75.0,
        "reward_mode": "distance_delta",
    }
    assert actions.walk_calls == [([game.initial_state.villager], 100.0, 100.0)]
    command = ("walk", (game.initial_state.villager,), 100.0, 100.0)
    assert game.step_calls == [[command], None, None]


def test_step_uses_backend_batch_capability(fake_backend):
    game, actions = fake_backend
    batch_calls = []

    def step_many(commands=None, *, turns):
        batch_calls.append((commands, turns))
        game.current_state = game.state_after_action
        return game.current_state

    game.step_many = step_many
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=2,
        reach_threshold=80.0,
        sim_steps_per_action=3,
        game=game,
        actions=actions,
    )
    env.reset()

    observation, reward, terminated, _, _ = env.step([0.0, 0.0])

    command = ("walk", (game.initial_state.villager,), 100.0, 100.0)
    assert batch_calls == [([command], 3)]
    assert game.step_calls == []
    assert np.allclose(observation, [-0.75, -1.0, 0.0, -1.0, 0.375])
    assert reward == 25.0
    assert terminated is True


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


def test_stock_delta_reward_emits_gather_when_target_hits_resource(fake_backend):
    game, actions = fake_backend
    game.stock_values = [10.0, 14.0]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=3,
        gather_command_distance=12.0,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        stock_success_threshold=3.0,
        game=game,
        actions=actions,
    )

    observation, info = env.reset()

    assert np.allclose(observation, [-1.0, -1.0, 0.0, -1.0, 0.5])
    assert info["resource_stock"] == 10.0
    assert info["episode_resource_stock_delta"] == 0.0

    observation, reward, terminated, truncated, info = env.step([0.0, -1.0])

    assert np.allclose(observation, [-0.75, -1.0, 0.0, -1.0, 0.375])
    assert reward == 4.0
    assert terminated is True
    assert truncated is False
    assert actions.gather_calls == [
        ([game.initial_state.villager], game.initial_state.resource)
    ]
    assert actions.walk_calls == []
    assert info == {
        "click_requested": True,
        "command": "gather",
        "distance": 75.0,
        "episode_resource_stock_delta": 4.0,
        "gather_cycle_active": False,
        "gather_cycle_finished": "stock_delta",
        "gather_requested": True,
        "return_resource_requested": False,
        "resource_stock": 14.0,
        "resource_stock_delta": 4.0,
        "reward_mode": "stock_delta",
        "target_hits_dropsite": False,
        "target_hits_resource": True,
    }


def test_stock_delta_reward_no_click_preserves_active_gather_cycle(fake_backend):
    game, actions = fake_backend
    game.stock_values = [10.0, 10.0, 14.0]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=3,
        gather_command_distance=12.0,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        stock_success_threshold=3.0,
        agent_controls_click=True,
        click_action_threshold=0.0,
        game=game,
        actions=actions,
    )
    env.reset()

    _, reward, terminated, truncated, info = env.step([0.0, -1.0, 1.0])

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["command"] == "gather"

    _, reward, terminated, truncated, info = env.step([1.0, 1.0, -1.0])

    assert reward == 4.0
    assert terminated is True
    assert truncated is False
    assert info["command"] == "no_click"
    assert info["click_requested"] is False
    assert info["click_signal"] == -1.0
    assert info["gather_cycle_active"] is False
    assert info["gather_cycle_finished"] == "stock_delta"
    assert info["gather_requested"] is False
    assert info["target_hits_resource"] is False
    assert actions.gather_calls == [
        ([game.initial_state.villager], game.initial_state.resource)
    ]
    assert actions.walk_calls == []
    gather_command = ("gather", (game.initial_state.villager,), game.initial_state.resource)
    assert game.step_calls == [[gather_command], None]


def test_stock_delta_reward_uses_agent_click_signal(fake_backend):
    game, actions = fake_backend
    game.stock_values = [10.0, 10.0, 14.0]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=3,
        gather_command_distance=12.0,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        stock_success_threshold=3.0,
        agent_controls_click=True,
        click_action_threshold=0.0,
        game=game,
        actions=actions,
    )
    env.reset()

    _, reward, terminated, truncated, info = env.step([0.0, -1.0, -0.5])

    assert env.action_space.shape == (3,)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["command"] == "no_click"
    assert info["click_requested"] is False
    assert info["gather_requested"] is False
    assert info["click_signal"] == -0.5
    assert info["target_hits_resource"] is True
    assert actions.gather_calls == []
    assert actions.walk_calls == []

    _, reward, terminated, truncated, info = env.step([0.0, -1.0, 0.5])

    assert reward == 4.0
    assert terminated is True
    assert truncated is False
    assert info["command"] == "gather"
    assert info["click_requested"] is True
    assert info["gather_cycle_active"] is False
    assert info["gather_cycle_finished"] == "stock_delta"
    assert info["gather_requested"] is True
    assert info["click_signal"] == 0.5
    assert info["target_hits_resource"] is True
    assert actions.gather_calls == [
        ([game.initial_state.villager], game.initial_state.resource)
    ]


def test_resource_state_observation_includes_carried_wood_and_stock(fake_backend):
    game, actions = fake_backend
    game.stock_values = [300.0, 320.0]
    game.carried_values = [0.0, [{"type": "wood.tree", "amount": 10.0}]]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=3,
        gather_command_distance=12.0,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        stock_success_threshold=100.0,
        resource_state_observation=True,
        carried_resource_observation_scale=20.0,
        stock_observation_scale=1000.0,
        game=game,
        actions=actions,
    )

    observation, info = env.reset()

    assert env.observation_space.shape == (7,)
    assert env.observation_labels[-2:] == ("carried_wood_norm", "stock_wood_norm")
    assert np.allclose(observation, [-1.0, -1.0, 0.0, -1.0, 0.5, 0.0, 0.3])
    assert info["resource_stock"] == 300.0
    assert len(game.evaluate_calls) == 1

    observation, reward, terminated, truncated, info = env.step([0.0, -1.0])

    assert np.allclose(observation, [-0.75, -1.0, 0.0, -1.0, 0.375, 0.5, 0.32])
    assert reward == 20.0
    assert terminated is False
    assert truncated is False
    assert info["resource_stock"] == 320.0
    assert len(game.evaluate_calls) == 2


def test_lifecycle_observation_exposes_dropsite_and_active_gather_cycle(
    fake_backend,
):
    game, actions = fake_backend
    game.stock_values = [300.0, 300.0, 300.0]
    game.carried_values = [0.0, 0.0, 20.0]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=3,
        gather_command_distance=12.0,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        stock_success_threshold=100.0,
        agent_controls_click=True,
        resource_state_observation=True,
        lifecycle_state_observation=True,
        carried_resource_observation_scale=20.0,
        stock_observation_scale=1000.0,
        game=game,
        actions=actions,
    )

    observation, _ = env.reset()

    assert env.observation_space.shape == (10,)
    assert env.observation_labels[-3:] == (
        "dropsite_x_norm",
        "dropsite_z_norm",
        "gather_cycle_active",
    )
    assert np.allclose(
        observation,
        [-1.0, -1.0, 0.0, -1.0, 0.5, 0.0, 0.3, -1.0, 0.0, 0.0],
    )

    observation, _, _, _, info = env.step([0.0, -1.0, 1.0])

    assert info["gather_cycle_active"] is True
    assert observation[-1] == 1.0
    assert np.allclose(observation[-3:-1], [-1.0, 0.0])

    observation, _, _, _, info = env.step([1.0, 1.0, 1.0])

    assert info["gather_cycle_active"] is False
    assert observation[-1] == 0.0


def test_lifecycle_observation_requires_resource_state_and_a_dropsite(fake_backend):
    game, actions = fake_backend

    with pytest.raises(ValueError, match="resource_state_observation"):
        ZeroADGatherEnv(
            "scenario contents",
            reward_mode="stock_delta",
            lifecycle_state_observation=True,
            game=game,
            actions=actions,
        )

    with pytest.raises(ValueError, match="stock_delta"):
        ZeroADGatherEnv(
            "scenario contents",
            resource_state_observation=True,
            lifecycle_state_observation=True,
            game=game,
            actions=actions,
        )

    game.initial_state = replace(game.initial_state, storehouse=None)
    game.current_state = game.initial_state
    env = ZeroADGatherEnv(
        "scenario contents",
        reward_mode="stock_delta",
        resource_state_observation=True,
        lifecycle_state_observation=True,
        game=game,
        actions=actions,
    )

    with pytest.raises(RuntimeError, match="storehouse"):
        env.reset()


def test_lifecycle_observation_clears_active_cycle_after_deposit(fake_backend):
    game, actions = fake_backend
    game.stock_values = [300.0, 300.0, 320.0]
    game.carried_values = [0.0, 20.0, 0.0]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=3,
        gather_command_distance=12.0,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        stock_success_threshold=100.0,
        agent_controls_click=True,
        resource_state_observation=True,
        lifecycle_state_observation=True,
        game=game,
        actions=actions,
    )
    env.reset()

    observation, _, _, _, _ = env.step([0.0, -1.0, 1.0])
    assert observation[-1] == 1.0

    observation, reward, _, _, info = env.step([0.0, -1.0, -1.0])

    assert reward == 20.0
    assert info["gather_cycle_finished"] == "stock_delta"
    assert observation[-1] == 0.0


def test_stock_delta_reward_unlocks_gather_cycle_after_stock_increase(fake_backend):
    game, actions = fake_backend
    game.stock_values = [10.0, 10.0, 14.0, 14.0]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=5,
        gather_command_distance=12.0,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        stock_success_threshold=100.0,
        agent_controls_click=True,
        click_action_threshold=0.0,
        game=game,
        actions=actions,
    )
    env.reset()

    _, reward, terminated, truncated, info = env.step([0.0, -1.0, 1.0])

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["command"] == "gather"
    assert info["gather_cycle_active"] is True

    _, reward, terminated, truncated, info = env.step([1.0, 1.0, -1.0])

    assert reward == 4.0
    assert terminated is False
    assert truncated is False
    assert info["command"] == "no_click"
    assert info["click_requested"] is False
    assert info["gather_cycle_active"] is False
    assert info["gather_cycle_finished"] == "stock_delta"

    _, reward, terminated, truncated, info = env.step([1.0, 1.0, 1.0])

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["command"] == "walk"
    assert actions.walk_calls == [([game.state_after_action.villager], 200.0, 200.0)]


def test_stock_delta_reward_allows_agent_to_interrupt_gather_cycle(fake_backend):
    game, actions = fake_backend
    game.stock_values = [10.0, 10.0, 10.0]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=5,
        gather_command_distance=12.0,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        stock_success_threshold=100.0,
        agent_controls_click=True,
        click_action_threshold=0.0,
        game=game,
        actions=actions,
    )
    env.reset()

    env.step([0.0, -1.0, 1.0])

    _, reward, terminated, truncated, info = env.step([1.0, 1.0, 1.0])

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["command"] == "walk"
    assert info["click_requested"] is True
    assert info["click_while_gather_cycle_active"] is True
    assert info["gather_cycle_active"] is False
    assert info["gather_cycle_finished"] == "interrupted_by_click"
    assert actions.walk_calls == [([game.state_after_action.villager], 200.0, 200.0)]


def test_stock_delta_reward_clicking_dropsite_returns_resource(fake_backend):
    game, actions = fake_backend
    game.stock_values = [10.0, 10.0, 10.0]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=5,
        gather_command_distance=12.0,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        stock_success_threshold=100.0,
        agent_controls_click=True,
        click_action_threshold=0.0,
        game=game,
        actions=actions,
    )
    env.reset()
    env.step([0.0, -1.0, 1.0])

    _, reward, terminated, truncated, info = env.step([-1.0, 0.0, 1.0])

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["command"] == "return_resource"
    assert info["click_requested"] is True
    assert info["return_resource_requested"] is True
    assert info["target_hits_dropsite"] is True
    assert "click_while_gather_cycle_active" not in info
    assert info["gather_cycle_active"] is True
    assert actions.returnresource_calls == [
        ([game.state_after_action.villager], game.initial_state.storehouse)
    ]
    assert actions.walk_calls == []


def test_stock_delta_reward_shapes_carried_wood_and_no_click(fake_backend):
    game, actions = fake_backend
    game.stock_values = [10.0, 10.0, 10.0]
    game.carried_values = [0.0, 0.0, 10.0]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=5,
        gather_command_distance=12.0,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        stock_success_threshold=100.0,
        agent_controls_click=True,
        click_action_threshold=0.0,
        carried_resource_delta_reward_scale=0.2,
        gather_cycle_no_click_reward=0.02,
        carrying_no_click_reward=0.05,
        game=game,
        actions=actions,
    )
    env.reset()

    _, reward, terminated, truncated, info = env.step([0.0, -1.0, 1.0])

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["command"] == "gather"
    assert info["carried_resource"] == 0.0
    assert info["carried_resource_delta"] == 0.0

    _, reward, terminated, truncated, info = env.step([1.0, 1.0, -1.0])

    assert reward == pytest.approx(2.07)
    assert terminated is False
    assert truncated is False
    assert info["command"] == "no_click"
    assert info["carried_resource"] == 10.0
    assert info["carried_resource_delta"] == 10.0
    assert info["carried_resource_delta_reward"] == 2.0
    assert info["gather_cycle_no_click_reward"] == 0.02
    assert info["carrying_no_click_reward"] == 0.05


def test_stock_delta_reward_penalizes_click_during_gather_cycle(fake_backend):
    game, actions = fake_backend
    game.stock_values = [10.0, 10.0, 10.0]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=5,
        gather_command_distance=12.0,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        stock_success_threshold=100.0,
        agent_controls_click=True,
        click_action_threshold=0.0,
        click_gather_cycle_penalty=1.0,
        game=game,
        actions=actions,
    )
    env.reset()
    env.step([0.0, -1.0, 1.0])

    _, reward, terminated, truncated, info = env.step([1.0, 1.0, 1.0])

    assert reward == -1.0
    assert terminated is False
    assert truncated is False
    assert info["command"] == "walk"
    assert info["click_while_gather_cycle_active"] is True
    assert info["click_gather_cycle_penalty"] == 1.0


def test_stock_delta_reward_walks_until_near_resource(fake_backend):
    game, actions = fake_backend
    game.stock_values = [10.0, 10.0]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=3,
        reach_threshold=1.0,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        game=game,
        actions=actions,
    )
    env.reset()

    _, reward, terminated, truncated, info = env.step([0.0, 0.0])

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["command"] == "walk"
    assert actions.walk_calls == [([game.initial_state.villager], 100.0, 100.0)]
    assert actions.gather_calls == []


def test_stock_delta_reward_can_shape_distance_and_gather_range(fake_backend):
    game, actions = fake_backend
    game.stock_values = [10.0, 10.0]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=3,
        gather_command_distance=80.0,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        distance_shaping_scale=0.02,
        gather_ready_reward=0.25,
        game=game,
        actions=actions,
    )
    env.reset()

    _, reward, terminated, truncated, info = env.step([0.0, 0.0])

    assert reward == 0.75
    assert terminated is False
    assert truncated is False
    assert info["resource_stock_delta"] == 0.0
    assert info["distance_shaping_reward"] == 0.5
    assert info["gather_ready_reward"] == 0.25
    assert info["command"] == "walk"


@pytest.mark.parametrize(
    ("post_stock", "post_carried", "expected_reward"),
    ((10.0, 5.0, 0.5), (17.0, 0.0, 7.5)),
)
def test_stock_delta_distance_shaping_follows_carried_wood_to_dropsite(
    fake_backend,
    post_stock,
    post_carried,
    expected_reward,
):
    game, actions = fake_backend
    game.state_after_action = replace(
        game.state_after_action,
        villager=replace(
            game.state_after_action.villager,
            coordinates=(0.0, 25.0),
        ),
    )
    game.stock_values = [10.0, post_stock]
    game.carried_values = [5.0, post_carried]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=3,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        stock_success_threshold=100.0,
        distance_shaping_scale=0.02,
        game=game,
        actions=actions,
    )
    env.reset()

    _, reward, terminated, truncated, info = env.step([0.0, 0.0])

    assert reward == expected_reward
    assert terminated is False
    assert truncated is False
    assert info["resource_stock_delta"] == post_stock - 10.0
    assert info["carried_resource"] == post_carried
    assert info["distance_shaping_reward"] == 0.5


def test_stock_delta_distance_shaping_is_zero_without_a_dropsite(fake_backend):
    game, actions = fake_backend
    game.initial_state = replace(game.initial_state, storehouse=None)
    game.current_state = game.initial_state
    game.state_after_action = replace(game.state_after_action, storehouse=None)
    game.stock_values = [10.0, 10.0]
    game.carried_values = [5.0, 5.0]
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=3,
        sim_steps_per_action=1,
        reward_mode="stock_delta",
        stock_success_threshold=100.0,
        distance_shaping_scale=0.02,
        game=game,
        actions=actions,
    )
    env.reset()

    _, reward, _, _, info = env.step([0.0, 0.0])

    assert reward == 0.0
    assert info["distance_shaping_reward"] == 0.0


def test_step_recovery_truncates_interrupted_episode(fake_backend):
    game, actions = fake_backend
    replacement_game = type(game)(game.initial_state, game.state_after_action)
    replacement_actions = type(actions)()
    game.step = lambda commands=None: (_ for _ in ()).throw(RuntimeError("gone"))
    created = []

    def backend_factory(uri):
        created.append(uri)
        return replacement_game, replacement_actions

    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        backend_retries=1,
        backend_retry_delay=0.0,
        game=game,
        actions=actions,
        backend_factory=backend_factory,
    )

    observation, reward, terminated, truncated, info = env.step([0.0, 0.0])

    assert created == ["http://localhost:6000"]
    assert np.allclose(observation, [-1.0, -1.0, 0.0, -1.0, 0.5])
    assert reward == 0.0
    assert terminated is False
    assert truncated is True
    assert info["interrupted"] is True


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


def test_default_backend_wraps_the_zero_ad_client_for_batched_steps(monkeypatch):
    created_uris = []
    default_game = object()
    default_actions = object()
    zero_ad = ModuleType("zero_ad")

    def create_game(uri):
        created_uris.append(uri)
        return default_game

    zero_ad.ZeroAD = create_game
    zero_ad.GameState = object
    zero_ad.actions = default_actions
    monkeypatch.setitem(sys.modules, "zero_ad", zero_ad)

    env = ZeroADGatherEnv("scenario contents", uri="http://localhost:7000")

    assert created_uris == ["http://localhost:7000"]
    assert isinstance(env.game, BatchedZeroAD)
    assert env.game._game is default_game
    assert env.actions is default_actions


def test_sim_frame_observer_reports_every_simulation_turn(fake_backend):
    game, actions = fake_backend
    batch_calls = []

    def step_many(commands=None, *, turns):
        batch_calls.append((commands, turns))
        game.current_state = game.state_after_action
        return game.current_state

    game.step_many = step_many
    ticks = []
    env = ZeroADGatherEnv(
        "scenario contents",
        map_size_m=200.0,
        horizon=2,
        reach_threshold=80.0,
        sim_steps_per_action=3,
        game=game,
        actions=actions,
        sim_frame_observer=lambda: ticks.append(len(game.step_calls)),
    )
    env.reset()

    env.step([0.0, 0.0])

    command = ("walk", (game.initial_state.villager,), 100.0, 100.0)
    # Recording gives up the batched fast path so every turn can be captured.
    assert batch_calls == []
    assert game.step_calls == [[command], None, None]
    assert ticks == [1, 2, 3]


def test_sim_frame_observer_must_be_callable(fake_backend):
    game, actions = fake_backend

    with pytest.raises(ValueError, match="sim_frame_observer must be callable"):
        ZeroADGatherEnv(
            "scenario contents",
            map_size_m=200.0,
            horizon=2,
            reach_threshold=80.0,
            sim_steps_per_action=1,
            game=game,
            actions=actions,
            sim_frame_observer="not callable",
        )


def test_team_snapshot_expression_names_every_entity_once():
    from rl.gather.env import team_snapshot_expression

    expression = team_snapshot_expression(1, (11, 12), (21, 22), "wood")

    assert "GetResourceCounts" in expression
    assert "[11,12]" in expression.replace(" ", "")
    assert "[21,22]" in expression.replace(" ", "")
    assert '"wood"' in expression


def test_parse_team_snapshot_orders_values_by_entity_id():
    from rl.gather.env import parse_team_snapshot

    payload = {
        "stock": {"wood": 300.0},
        "carried": {"12": 20.0, "11": 0.0},
        "remaining": {"22": 50.0, "21": 200.0},
    }

    stock, carried, remaining = parse_team_snapshot(payload, "wood", (11, 12), (21, 22))

    assert stock == 300.0
    assert carried == (0.0, 20.0)
    assert remaining == (200.0, 50.0)


def test_parse_team_snapshot_defaults_missing_entities_to_zero():
    from rl.gather.env import parse_team_snapshot

    payload = {"stock": {"wood": 10.0}, "carried": {}, "remaining": {}}

    stock, carried, remaining = parse_team_snapshot(payload, "wood", (11,), (21,))

    assert stock == 10.0
    assert carried == (0.0,)
    assert remaining == (0.0,)


def test_parse_team_snapshot_rejects_a_malformed_payload():
    from rl.gather.env import parse_team_snapshot

    with pytest.raises(TypeError, match="team snapshot"):
        parse_team_snapshot(["not", "a", "mapping"], "wood", (11,), (21,))


def test_parse_team_snapshot_requires_every_section():
    from rl.gather.env import parse_team_snapshot

    with pytest.raises(TypeError, match="missing 'remaining'"):
        parse_team_snapshot(
            {"stock": {"wood": 1.0}, "carried": {}},
            "wood",
            (11,),
            (21,),
        )
