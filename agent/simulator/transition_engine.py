"""The transition engine: composes the per-domain transitions.

One call to :meth:`TransitionEngine.apply` reproduces exactly one turn of the
official environment's ``step()`` pipeline, in the same order:

1. farmer / hand movement (both players),
2. farmer / hand farming actions — PLANT / WATER (both players),
3. market order processing (per-unit lockstep),
4. town consumption for the *current* step,
5. turn advance (step / hour / day).

Only the transitions in the first verified layer are wired in; everything else
(harvest, fertilizer, digging, structures, animals, daily end-of-day refresh)
is intentionally omitted.
"""

from __future__ import annotations

from ..actions import TurnAction
from ..state import GameState
from .game_config import GameConfig
from .transitions.farming import FarmingTransition
from .transitions.market import MarketTransition
from .transitions.movement import MovementTransition
from .transitions.turn import TurnTransition

_Action = TurnAction | tuple[TurnAction, TurnAction]


class TransitionEngine:
    """Applies one turn to a ``GameState`` via the transition handlers."""

    def __init__(self, config: GameConfig | None = None) -> None:
        self._config = config if config is not None else GameConfig()
        self._turn = TurnTransition(self._config)
        self._movement = MovementTransition(self._config)
        self._farming = FarmingTransition(self._config)
        self._market = MarketTransition(self._config)

    @property
    def config(self) -> GameConfig:
        """The configuration this engine simulates with."""
        return self._config

    def apply(self, state: GameState, action: _Action) -> GameState:
        """Advance ``state`` by exactly one turn.

        Accepts either a single ``TurnAction`` (applied to the current player,
        the other player passes) or a ``(player0, player1)`` pair for
        simultaneous-move simulations.
        """
        action0, action1 = self._split_actions(state, action)

        state = self._movement.apply(state, 0, action0)
        state = self._movement.apply(state, 1, action1)
        state = self._farming.apply(state, 0, action0)
        state = self._farming.apply(state, 1, action1)
        state = self._market.process_orders(
            state, action0.market_actions, action1.market_actions
        )
        state = self._market.town_consume(state)
        return self._turn.advance(state)

    @staticmethod
    def _split_actions(state: GameState, action: _Action) -> tuple[TurnAction, TurnAction]:
        if isinstance(action, tuple):
            return action[0], action[1]
        if state.current_player == 0:
            return action, TurnAction()
        return TurnAction(), action
