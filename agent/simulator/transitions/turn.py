"""Turn lifecycle transition: step / hour / day advancement.

Verified against the official environment: after each step,

    next_step = step + 1
    day       = next_step // turns_per_day
    hour      = next_step %  turns_per_day

The full end-of-day refresh (plant/animal daily updates, weed spawns, worker
drop-to-shed, hand reset) is deliberately deferred to a later phase; the
required transitions all stay within a single day.
"""

from __future__ import annotations

from dataclasses import replace

from ...state import GameState
from ..game_config import GameConfig


class TurnTransition:
    """Advances the absolute turn counter exactly as the official environment."""

    def __init__(self, config: GameConfig) -> None:
        self._config = config

    def advance(self, state: GameState) -> GameState:
        """Return ``state`` advanced by exactly one turn."""
        turns_per_day = self._config.turns_per_day
        next_step = state.step + 1
        return replace(
            state,
            step=next_step,
            day=next_step // turns_per_day,
            hour=next_step % turns_per_day,
        )
