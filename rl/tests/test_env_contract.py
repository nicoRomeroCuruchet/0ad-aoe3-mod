import numpy as np
from gymnasium import spaces
from rl.gather import ZeroADGatherEnv


def test_spaces_are_well_formed_without_connecting():
    # No llamamos reset/step: solo construimos los espacios y los miramos.
    env = ZeroADGatherEnv.__new__(ZeroADGatherEnv)
    env.observation_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
    env.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    assert env.observation_space.shape == (5,)
    assert env.action_space.shape == (2,)
    assert env.action_space.contains(np.zeros(2, dtype=np.float32))
