import pytest

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


def test_step_many_falls_back_when_the_server_lacks_step_n(capsys):
    from urllib.error import HTTPError

    from rl.gather.zero_ad_client import BatchedZeroAD

    class RefusingApi:
        def post(self, path, data):
            raise HTTPError(path, 404, "Not Found", {}, None)

    class StockGame:
        player_id = 1

        def __init__(self):
            self.api = RefusingApi()
            self.steps = []

        def step(self, actions=None):
            self.steps.append(actions)
            return "state"

    game = StockGame()
    client = BatchedZeroAD(game, lambda payload, game: payload)

    state = client.step_many([{"type": "walk"}], turns=3)

    assert state == "state"
    assert game.steps == [[{"type": "walk"}], None, None]
    assert client.batching_supported is False
    assert "step_n unavailable" in capsys.readouterr().out


def test_step_many_keeps_batching_after_a_transient_server_error():
    from urllib.error import HTTPError

    from rl.gather.zero_ad_client import BatchedZeroAD

    class FlakyApi:
        def post(self, path, data):
            raise HTTPError(path, 500, "Server Error", {}, None)

    class StockGame:
        player_id = 1
        api = FlakyApi()

    client = BatchedZeroAD(StockGame(), lambda payload, game: payload)

    with pytest.raises(HTTPError):
        client.step_many([], turns=2)

    assert client.batching_supported is True
