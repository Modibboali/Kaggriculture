"""The MCTS <-> simulator boundary.

The simulator is the single source of truth for Kaggriculture transitions.
This adapter is the only place the search layer touches the simulator, keeping
MCTS completely free of game rules and of the Kaggle environment.

``transition(state, action)`` applies one ``TurnAction`` for the acting player
(``current_player``; the opponent passes during search, which is the standard
single-agent simplification for a first baseline) and returns a new
``SearchState``. The parent state is never mutated.
"""

from __future__ import annotations

from ..actions import TurnAction
from ..simulator import Simulator, TransitionEngine
from .search_state import SearchState


class SimulatorAdapter:
    """Deterministic, immutable forward-model interface used by MCTS."""

    def __init__(
        self,
        engine: TransitionEngine | Simulator | None = None,
        *,
        count_transitions: bool = False,
    ) -> None:
        if engine is None:
            engine = TransitionEngine()
        self._engine = engine
        self._counting = count_transitions
        self._transitions = 0

    @property
    def engine(self) -> TransitionEngine | Simulator:
        """The underlying simulator (kept opaque to the search layer)."""
        return self._engine

    @property
    def transitions(self) -> int:
        """Number of transitions applied (when ``count_transitions`` is on)."""
        return self._transitions

    def transition(self, state: SearchState, action: TurnAction) -> SearchState:
        """Apply ``action`` for the acting player; returns a new state."""
        if self._counting:
            self._transitions += 1
        return SearchState(self._engine.apply(state.game, action))
