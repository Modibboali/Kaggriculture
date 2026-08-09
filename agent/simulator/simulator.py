"""Simulator facade: the entry point for lightweight forward simulation.

``Simulator`` satisfies the differential framework's ``Simulator`` protocol
(``apply(state, actions: tuple[TurnAction, TurnAction]) -> GameState``) while
also accepting a single ``TurnAction`` for single-agent rollouts:

    simulator.apply(state, turn_action)          # current player acts
    simulator.apply(state, (p0_action, p1_action))  # simultaneous move

The simulator implements only the first verified transition layer: PASS,
movement, BUY_SEED, PLANT, and WATER. Unsupported actions are silent no-ops,
exactly as the official environment treats illegal actions.
"""

from __future__ import annotations

from ..actions import TurnAction
from ..state import GameState
from .game_config import GameConfig
from .transition_engine import TransitionEngine


class Simulator:
    """Forward-simulates the verified transition layer of Kaggriculture."""

    def __init__(self, config: GameConfig | None = None) -> None:
        self._engine = TransitionEngine(config)

    @property
    def engine(self) -> TransitionEngine:
        """The underlying transition engine."""
        return self._engine

    def apply(
        self, state: GameState, action: TurnAction | tuple[TurnAction, TurnAction]
    ) -> GameState:
        """Return the state after one turn of ``action``."""
        return self._engine.apply(state, action)
