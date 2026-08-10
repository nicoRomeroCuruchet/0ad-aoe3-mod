"""Small extensions for the pinned ``zero_ad`` client."""

from __future__ import annotations

import json
from itertools import cycle
from typing import Any, Callable


MAX_BATCHED_TURNS = 10_000


class BatchedZeroAD:
    """Delegate to ``zero_ad.ZeroAD`` and add one-request multi-turn stepping."""

    def __init__(self, game: Any, game_state_factory: Callable[[Any, Any], Any]):
        self._game = game
        self._game_state_factory = game_state_factory

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
        state_json = self._game.api.post(f"step_n?turns={turns}", post_data)
        state = self._game_state_factory(json.loads(state_json), self._game)
        self._game.current_state = state
        return state
