"""Top-level legal action generator.

``LegalActionGenerator`` is the composition root: it delegates each action
family to a specialized generator and combines the results. All legality is
derived from the acting player's ``GameState``; no heuristics, ranking, or
simulation happen anywhere in this package.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..actions import Action
from ..state import GameState, Worker
from .animal_generator import AnimalGenerator
from .farm_generator import FarmGenerator
from .inventory_generator import InventoryGenerator
from .market_generator import MarketGenerator
from .movement_generator import MovementGenerator


@dataclass(frozen=True, slots=True)
class WorkerActionSet:
    """A hired hand and the legal actions available to it this turn."""

    worker: Worker
    actions: tuple[Action, ...]


@dataclass(frozen=True, slots=True)
class LegalActions:
    """All legal atomic actions for the acting player this turn."""

    farmer_actions: tuple[Action, ...]
    worker_actions: tuple[WorkerActionSet, ...]
    market_actions: tuple[Action, ...]


class LegalActionGenerator:
    """Generates every legal atomic action for the current player.

    Specialized generators are injected so callers (and tests) can substitute
    or extend individual action families without touching the composition.
    """

    def __init__(
        self,
        *,
        movement: MovementGenerator | None = None,
        farm: FarmGenerator | None = None,
        animal: AnimalGenerator | None = None,
        inventory: InventoryGenerator | None = None,
        market: MarketGenerator | None = None,
    ) -> None:
        self._movement = movement or MovementGenerator()
        self._farm = farm or FarmGenerator()
        self._animal = animal or AnimalGenerator()
        self._inventory = inventory or InventoryGenerator()
        self._market = market or MarketGenerator()

    def _unit_actions(self, state: GameState, worker: Worker) -> tuple[Action, ...]:
        """All legal actions for one unit of the acting player."""
        player = state.current_player_state
        farm = player.farm
        return (
            self._movement.generate()
            + self._farm.generate(farm, worker.position, player.seeds, worker.inventory)
            + self._animal.generate(farm, worker.position, worker.inventory)
            + self._inventory.generate(
                farm, worker.position, worker.inventory, player.inventory
            )
        )

    def generate_farmer_actions(self, state: GameState) -> tuple[Action, ...]:
        """Every legal atomic action for the acting player's main farmer."""
        return self._unit_actions(state, state.current_player_state.farm.farmer)

    def generate_worker_actions(self, state: GameState) -> tuple[WorkerActionSet, ...]:
        """Every legal atomic action for each hired hand, one set per hand."""
        player = state.current_player_state
        return tuple(
            WorkerActionSet(worker=worker, actions=self._unit_actions(state, worker))
            for worker in player.farm.workers
        )

    def generate_market_actions(self, state: GameState) -> tuple[Action, ...]:
        """Every legal market order for the acting player."""
        player = state.current_player_state
        return self._market.generate(player, state.market)

    def generate_actions(self, state: GameState) -> LegalActions:
        """All legal atomic actions for the acting player, bundled."""
        return LegalActions(
            farmer_actions=self.generate_farmer_actions(state),
            worker_actions=self.generate_worker_actions(state),
            market_actions=self.generate_market_actions(state),
        )
