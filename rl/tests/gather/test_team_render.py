import numpy as np
import pytest

from rl.experiments.evaluation import DecisionRecord
from rl.gather.engine_observer import EngineObserverUnavailable
from rl.gather.rollout_recording import AgentViewRolloutRecorder
from rl.gather.team_render import TeamRenderState, render_team_observer_frame


def _state(*, carried=(0.0, 20.0), targets=((300.0, 100.0),)):
    return TeamRenderState(
        villager_xz=((100.0, 100.0), (200.0, 140.0)),
        resource_xz=((300.0, 100.0), (320.0, 200.0)),
        resource_remaining=(200.0, 0.0),
        carried=carried,
        dropsite_xz=(60.0, 160.0),
        targets_xz=targets,
    )


def test_team_schematic_returns_a_ppm_frame_of_the_requested_size():
    frame = render_team_observer_frame(_state(), size=128)

    assert frame.width == 128
    assert frame.height == 128
    assert frame.ppm.startswith(b"P6\n128 128\n255\n")


def test_team_schematic_changes_for_loads_and_assignment_targets():
    empty = render_team_observer_frame(
        _state(carried=(0.0, 0.0), targets=()),
        size=128,
    )
    active = render_team_observer_frame(_state(), size=128)

    assert empty.ppm != active.ppm


def test_team_schematic_validates_parallel_state_lengths():
    with pytest.raises(ValueError, match="resource_remaining"):
        TeamRenderState(
            villager_xz=((100.0, 100.0),),
            resource_xz=((300.0, 100.0),),
            resource_remaining=(),
            carried=(0.0,),
            dropsite_xz=(60.0, 160.0),
        )


def test_rollout_fallback_uses_the_environments_team_schematic(tmp_path):
    expected = render_team_observer_frame(_state(), size=64)

    class TeamEnv:
        def __init__(self):
            self.actions = []

        def capture_agent_frame(self):
            raise EngineObserverUnavailable("not rendered")

        def capture_schematic_frame(self, action):
            self.actions.append(np.asarray(action).copy())
            return expected

    env = TeamEnv()
    recorder = AgentViewRolloutRecorder(
        tmp_path,
        env,
        capture_attempts=1,
        fallback_to_schematic=True,
    )
    record = DecisionRecord(
        episode=0,
        step=0,
        observation=np.zeros((2, 21), dtype=np.float32),
        action=np.array([1, 0], dtype=np.int64),
    )

    frame, source = recorder._capture_frame(record)

    assert frame is expected
    assert source == "schematic"
    np.testing.assert_array_equal(env.actions, [record.action])
