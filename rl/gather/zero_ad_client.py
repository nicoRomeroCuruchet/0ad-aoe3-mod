"""Small extensions for the pinned ``zero_ad`` client."""

from __future__ import annotations

import json
from itertools import cycle
from typing import Any, Callable
from urllib.error import HTTPError


MAX_BATCHED_TURNS = 10_000


class BatchedZeroAD:
    """Delegate to ``zero_ad.ZeroAD`` and add one-request multi-turn stepping."""

    # `step_n` ships with this repo's engine patch. A stock 0 A.D. build answers
    # with a client error, and then every turn is stepped one at a time instead.
    UNSUPPORTED_STATUS = frozenset({400, 404, 405, 501})

    def __init__(self, game: Any, game_state_factory: Callable[[Any, Any], Any]):
        self._game = game
        self._game_state_factory = game_state_factory
        self._batching_supported = True

    @property
    def batching_supported(self) -> bool:
        """False once the server has refused the batched endpoint."""

        return self._batching_supported

    def _step_one_at_a_time(self, actions, turns: int):
        state = self._game.step(actions or None)
        for _ in range(turns - 1):
            state = self._game.step()
        return state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._game, name)

    def step_many(self, actions=None, *, turns: int, player=None):
        if (
            isinstance(turns, bool)
            or not isinstance(turns, int)
            or not 1 <= turns <= MAX_BATCHED_TURNS
        ):
            raise ValueError(f"turns must be an integer in [1, {MAX_BATCHED_TURNS}]")
        if actions is None:
            actions = []
        player_ids = (
            cycle([self._game.player_id]) if player is None else cycle(player)
        )
        commands = zip(player_ids, actions, strict=False)
        post_data = "\n".join(
            f"{player_id};{json.dumps(action)}"
            for player_id, action in commands
            if action is not None
        )
        if not self._batching_supported:
            return self._step_one_at_a_time(actions, turns)
        try:
            state_json = self._game.api.post(f"step_n?turns={turns}", post_data)
        except HTTPError as error:
            if error.code not in self.UNSUPPORTED_STATUS:
                raise
            print(
                "step_n unavailable on this server; stepping one turn at a time",
                flush=True,
            )
            self._batching_supported = False
            return self._step_one_at_a_time(actions, turns)
        state = self._game_state_factory(json.loads(state_json), self._game)
        self._game.current_state = state
        return state
