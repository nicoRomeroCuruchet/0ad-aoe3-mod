from rl.gather.zero_ad_client import BatchedZeroAD


class FakeAPI:
    def __init__(self):
        self.calls = []

    def post(self, route, data):
        self.calls.append((route, data))
        return b'{"mapSize": 512, "entities": {}}'


class FakeGame:
    def __init__(self):
        self.api = FakeAPI()
        self.current_state = None
        self.player_id = 1


class FakeGameState:
    def __init__(self, data, game):
        self.data = data
        self.game = game


def test_batched_zero_ad_posts_commands_once_and_updates_current_state():
    game = FakeGame()
    client = BatchedZeroAD(game, FakeGameState)

    state = client.step_many([{"type": "walk", "x": 4}], turns=3)

    assert game.api.calls == [
        ("step_n?turns=3", '1;{"type": "walk", "x": 4}')
    ]
    assert state.data == {"mapSize": 512, "entities": {}}
    assert state.game is game
    assert game.current_state is state
