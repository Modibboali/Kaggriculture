"""The transition engine: composes the per-domain transitions.

One call to :meth:`TransitionEngine.apply` reproduces exactly one turn of the
official environment's ``step()`` pipeline, in the same order:

1. farmer / hand movement (both players),
2. farmer / hand farming actions — PLANT / WATER (both players),
3. farmer / hand HARVEST (both players),
4. farmer / hand PICKUP / DROP (both players),
5. farmer / hand FERTILIZE (both players),
6. farmer / hand DIG (both players),
7. market order processing (per-unit lockstep, including atomic BUY_LAND),
8. town consumption for the *current* step,
9. per-step crop decay (``_decay_plants``),
10. end-of-day transition when ``(step + 1) % turns_per_day == 0``,
11. turn advance (step / hour / day).

Structures, animals, workers and HIRE remain intentionally unimplemented (out
of scope); unrecognized actions are silent no-ops.
"""

from __future__ import annotations

from ..actions import TurnAction
from ..state import GameState
from .game_config import GameConfig
from .transitions.animals import AnimalTransition
from .transitions.crop_lifecycle import CropLifecycleTransition
from .transitions.dig import DigTransition
from .transitions.end_of_day import EndOfDayProcessor
from .transitions.farming import FarmingTransition
from .transitions.fertilize import FertilizeTransition
from .transitions.harvest import HarvestTransition
from .transitions.items import PickupDropTransition
from .transitions.land import LandTransition
from .transitions.market import MarketTransition
from .transitions.movement import MovementTransition
from .transitions.structure import StructureTransition
from .transitions.turn import TurnTransition
from .transitions.workers import WorkerTransition

_Action = TurnAction | tuple[TurnAction, TurnAction]


class TransitionEngine:
    """Applies one turn to a ``GameState`` via the transition handlers."""

    def __init__(self, config: GameConfig | None = None) -> None:
        self._config = config if config is not None else GameConfig()
        self._turn = TurnTransition(self._config)
        self._movement = MovementTransition(self._config)
        self._farming = FarmingTransition(self._config)
        self._harvest = HarvestTransition(self._config)
        self._items = PickupDropTransition(self._config)
        self._fertilize = FertilizeTransition(self._config)
        self._dig = DigTransition(self._config)
        self._structure = StructureTransition(self._config)
        self._animals = AnimalTransition(self._config)
        self._land = LandTransition(self._config)
        self._workers = WorkerTransition(self._config)
        self._market = MarketTransition(self._config, self._land, self._workers)
        self._crop_lifecycle = CropLifecycleTransition(self._config)
        self._end_of_day = EndOfDayProcessor(self._config)

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
        state = self._harvest.apply(state, 0, action0)
        state = self._harvest.apply(state, 1, action1)
        state = self._items.apply(state, 0, action0)
        state = self._items.apply(state, 1, action1)
        state = self._fertilize.apply(state, 0, action0)
        state = self._fertilize.apply(state, 1, action1)
        state = self._dig.apply(state, 0, action0)
        state = self._dig.apply(state, 1, action1)
        state = self._structure.apply(state, 0, action0)
        state = self._structure.apply(state, 1, action1)
        state = self._animals.apply(state, 0, action0)
        state = self._animals.apply(state, 1, action1)
        state = self._market.process_orders(
            state, action0.market_actions, action1.market_actions
        )
        state = self._market.town_consume(state)
        state = self._crop_lifecycle.decay(state)
        if (state.step + 1) % self._config.turns_per_day == 0:
            state = self._end_of_day.process(state)
        return self._turn.advance(state)

    @staticmethod
    def _split_actions(state: GameState, action: _Action) -> tuple[TurnAction, TurnAction]:
        if isinstance(action, tuple):
            return action[0], action[1]
        if state.current_player == 0:
            return action, TurnAction()
        return TurnAction(), action
