"""Episode termination and terminal reward.

The official Kaggriculture episode runs for a fixed number of steps (720 by
default); when it ends, each player's reward is their farm money. This module
encodes that using the configurable ``episode_steps`` (from ``GameConfig``)
rather than hard-coding 720 anywhere in the search layer.
"""

from __future__ import annotations

from ..simulator import GameConfig
from .search_state import SearchState


class Terminal:
    """Determines episode end and the terminal value for a player."""

    def __init__(self, config: GameConfig) -> None:
        self._episode_steps = config.episode_steps

    def is_terminal(self, state: SearchState) -> bool:
        """Whether the episode has reached its configured length."""
        return state.game.step >= self._episode_steps

    def value(self, state: SearchState, player: int) -> float:
        """The terminal reward for ``player`` (their money at the end)."""
        return float(state.game.players[player].farm.money)
