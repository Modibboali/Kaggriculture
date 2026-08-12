"""Phase-aware action priority model (deterministic, mechanics-grounded).

``ActionPriorityModel`` scores candidate actions for the acting player so the
search can expand the most promising actions first and rollouts can follow a
cash-conversion chain. The score is a sum of interpretable components:

    priority(state, action) =
        phase_priority(action_type, phase)      # 0..100 per-phase base table
      * time_feasibility(action, state)         # 1.0 realizable, else 0.05
      + immediate_value(state, action)          # small, state-dependent bonus
      - action_cost(state, action)              # small penalty for money spent

* ``phase_priority`` — a per-phase base table encoding what is worth doing in
  each phase (all derived from the horizon; long-horizon investments are
  deprioritised in CASH_CONVERSION / PRODUCTION).
* ``time_feasibility`` — from :meth:`can_realize`, which uses the simulator's
  own growth tables (``first_yield_day``) to decide whether an action's
  downstream benefit can complete before the terminal. This is the
  *realizability filter*; it never inspects the opponent or future states.
* ``immediate_value`` / ``action_cost`` — small additive terms that rank
  otherwise-equal actions by real economic value (market prices, yields, costs).

Nothing here hard-codes a wheat-only strategy: crop selection uses the full
``config.crops`` table and current market prices.
"""

from __future__ import annotations

from typing import Callable

from ..actions import (
    BuyAnimalAction,
    BuyLandAction,
    BuyProductAction,
    BuySeedAction,
    MovementAction,
    PlantAction,
    SellAction,
    TurnAction,
)
from ..actions import ActionType
from ..simulator import GameConfig
from ..state import ItemType
from .evaluation import EvaluationConfig, horizon_days, horizon_remaining
from .phase import GamePhase, phase_for
from .search_state import SearchState


def farmer_type(action: TurnAction) -> ActionType:
    """The ActionType of a TurnAction (market action wins over farmer action)."""
    if action.market_actions:
        return action.market_actions[0].action_type
    return action.farmer_action.action_type


# Base per-phase priorities (higher = preferred). DEVELOPMENT matches the
# existing heuristic; PRODUCTION focuses on crop cash-cycles and deprioritises
# long-horizon investments; CASH_CONVERSION focuses on harvesting/selling.
_DEVELOPMENT: dict[ActionType, float] = {
    ActionType.HARVEST: 100,
    ActionType.COLLECT_FERTILIZER: 95,
    ActionType.WATER: 90,
    ActionType.FEED: 85,
    ActionType.CARE: 80,
    ActionType.PLANT: 75,
    ActionType.SELL: 70,
    ActionType.FERTILIZE: 65,
    ActionType.BUY_SEED: 60,
    ActionType.BUY_LAND: 55,
    ActionType.HIRE: 50,
    ActionType.BUY_ANIMAL: 50,
    ActionType.PLACE: 45,
    ActionType.BUILD_COOP: 40,
    ActionType.BUILD_PASTURE: 40,
    ActionType.PICKUP: 35,
    ActionType.DROP: 35,
    ActionType.DIG: 30,
    ActionType.NORTH: 5,
    ActionType.SOUTH: 5,
    ActionType.EAST: 5,
    ActionType.WEST: 5,
    ActionType.PASS: 0,
    ActionType.BUY_PRODUCT: 55,
}

_PRODUCTION: dict[ActionType, float] = {
    ActionType.SELL: 100,
    ActionType.HARVEST: 98,
    ActionType.COLLECT_FERTILIZER: 95,
    ActionType.WATER: 90,
    ActionType.FEED: 82,
    ActionType.PLANT: 80,
    ActionType.CARE: 78,
    ActionType.BUY_SEED: 70,
    ActionType.FERTILIZE: 68,
    ActionType.PICKUP: 40,
    ActionType.DROP: 40,
    ActionType.DIG: 30,
    ActionType.PLACE: 28,
    ActionType.BUY_PRODUCT: 66,
    ActionType.BUY_ANIMAL: 18,
    ActionType.BUILD_COOP: 12,
    ActionType.BUILD_PASTURE: 12,
    ActionType.HIRE: 10,
    ActionType.BUY_LAND: 6,
    ActionType.NORTH: 5,
    ActionType.SOUTH: 5,
    ActionType.EAST: 5,
    ActionType.WEST: 5,
    ActionType.PASS: 0,
}

_CASH_CONVERSION: dict[ActionType, float] = {
    ActionType.SELL: 100,
    ActionType.HARVEST: 98,
    ActionType.COLLECT_FERTILIZER: 95,
    ActionType.WATER: 85,
    ActionType.PICKUP: 45,
    ActionType.DROP: 45,
    ActionType.DIG: 30,
    ActionType.FERTILIZE: 25,
    ActionType.PLANT: 20,
    ActionType.BUY_SEED: 15,
    ActionType.PLACE: 8,
    ActionType.FEED: 8,
    ActionType.CARE: 8,
    ActionType.BUY_PRODUCT: 14,
    ActionType.NORTH: 5,
    ActionType.SOUTH: 5,
    ActionType.EAST: 5,
    ActionType.WEST: 5,
    ActionType.BUILD_COOP: 3,
    ActionType.BUILD_PASTURE: 3,
    ActionType.BUY_ANIMAL: 3,
    ActionType.HIRE: 2,
    ActionType.BUY_LAND: 1,
    ActionType.PASS: 0,
}

_PHASE_PRIORITIES: dict[GamePhase, dict[ActionType, float]] = {
    GamePhase.DEVELOPMENT: _DEVELOPMENT,
    GamePhase.PRODUCTION: _PRODUCTION,
    GamePhase.CASH_CONVERSION: _CASH_CONVERSION,
    GamePhase.TERMINAL: {},
}

# Actions that are immediately realisable (no downstream dependency).
_IMMEDIATE_TYPES = frozenset(
    {
        ActionType.SELL,
        ActionType.HARVEST,
        ActionType.WATER,
        ActionType.PICKUP,
        ActionType.DROP,
        ActionType.COLLECT_FERTILIZER,
        ActionType.FEED,
        ActionType.CARE,
        ActionType.DIG,
        ActionType.PASS,
        ActionType.NORTH,
        ActionType.SOUTH,
        ActionType.EAST,
        ActionType.WEST,
    }
)


def _farming_only(action: TurnAction) -> bool:
    """Mode B filter: keep only immediate crop production / cash conversion."""
    return farmer_type(action) in {
        ActionType.PLANT,
        ActionType.WATER,
        ActionType.HARVEST,
        ActionType.FERTILIZE,
        ActionType.DIG,
        ActionType.PICKUP,
        ActionType.DROP,
        ActionType.SELL,
        ActionType.BUY_SEED,
        ActionType.BUY_PRODUCT,
        ActionType.PASS,
        ActionType.NORTH,
        ActionType.SOUTH,
        ActionType.EAST,
        ActionType.WEST,
    }


class ActionPriorityModel:
    """Deterministic, phase-aware action prioritisation + realizability."""

    def __init__(
        self,
        config: GameConfig,
        eval_config: EvaluationConfig | None = None,
    ) -> None:
        self._config = config
        self._eval = eval_config if eval_config is not None else EvaluationConfig()

    # -- public API ---------------------------------------------------------

    def phase(self, state: SearchState) -> GamePhase:
        h_steps = horizon_remaining(state, self._config.episode_steps)
        h_days = horizon_days(state, self._config)
        return phase_for(h_steps, h_days, self._config)

    def can_realize(self, action: TurnAction, state: SearchState) -> bool:
        """Whether ``action``'s downstream benefit can occur before terminal.

        Uses only the acting player's own state and the simulator's growth
        tables; never inspects the opponent or future states.
        """
        h_steps = horizon_remaining(state, self._config.episode_steps)
        h_days = h_steps // self._config.turns_per_day
        if h_steps <= 0:
            return False
        at = farmer_type(action)
        if at in _IMMEDIATE_TYPES:
            return True

        if at == ActionType.PLANT:
            crop = action.farmer_action.crop  # type: ignore[attr-defined]
            return h_days >= self._config.crops[crop].first_yield_day + 1
        if at == ActionType.BUY_SEED:
            crop = action.market_actions[0].crop  # type: ignore[attr-defined]
            return h_days >= self._config.crops[crop].first_yield_day + 2
        if at == ActionType.FERTILIZE:
            return h_days >= 1
        if at == ActionType.BUILD_COOP or at == ActionType.BUILD_PASTURE:
            return h_days >= self._min_animal_cash_days()
        if at == ActionType.BUY_ANIMAL:
            animal = action.market_actions[0].animal  # type: ignore[attr-defined]
            return h_days >= self._config.animals[animal].first_yield_day + 2
        if at == ActionType.PLACE:
            return True  # placing an owned animal; value flows via the evaluator
        if at == ActionType.HIRE:
            return h_days >= self._shortest_crop_cash_days()
        if at == ActionType.BUY_LAND:
            return h_days >= self._longest_crop_cash_days()
        return True

    def priority(self, state: SearchState, action: TurnAction) -> float:
        """Interpretable score for ``action`` at ``state`` (higher = better)."""
        phase = self.phase(state)
        base = _PHASE_PRIORITIES.get(phase, {}).get(farmer_type(action), 0.0)
        feasibility = 1.0 if self.can_realize(action, state) else 0.05
        return (
            base * feasibility
            + self._immediate_value(state, action)
            - self._action_cost(state, action)
        )

    def rank(self, state: SearchState, actions: list[TurnAction]) -> tuple[TurnAction, ...]:
        """Actions sorted so the *best* is expanded first by MCTS.

        MCTS pops untried actions from the end of its list, so the highest-
        priority action must be last: we sort ascending by priority.
        """
        return tuple(sorted(actions, key=lambda a: self.priority(state, a)))

    def filter_realizable(self, state: SearchState, actions: list[TurnAction]) -> tuple[TurnAction, ...]:
        """Keep only actions whose downstream benefit can complete (Mode D)."""
        return tuple(a for a in actions if self.can_realize(a, state))

    @staticmethod
    def farming_only_filter(state: SearchState, actions: list[TurnAction]) -> tuple[TurnAction, ...]:
        """Mode B filter: crop production / cash conversion actions only."""
        del state
        return tuple(a for a in actions if _farming_only(a))

    # -- helpers ------------------------------------------------------------

    def _shortest_crop_cash_days(self) -> int:
        return min(s.first_yield_day for s in self._config.crops.values()) + 2

    def _longest_crop_cash_days(self) -> int:
        return max(s.first_yield_day for s in self._config.crops.values()) + 2

    def _min_animal_cash_days(self) -> int:
        return min(a.first_yield_day for a in self._config.animals.values()) + 2

    def _market_price(self, state: SearchState, item: ItemType) -> int:
        return state.game.market.price(item)

    def _immediate_value(self, state: SearchState, action: TurnAction) -> float:
        """Small additive bonuses that rank ready-to-execute actions."""
        at = farmer_type(action)
        game = state.game
        player_state = game.players[game.current_player]
        farm = player_state.farm
        tile = farm.tile_at(farm.farmer.position)

        if at == ActionType.SELL:
            item = action.market_actions[0].item  # type: ignore[attr-defined]
            price = self._market_price(state, item)
            return min(12.0, price / 20.0)
        if at == ActionType.HARVEST:
            from ..state import CoopTile, PastureTile, PlantTile

            if isinstance(tile, PlantTile):
                product = tile.plant.crop.produce
                return min(12.0, self._market_price(state, product) / 20.0)
            if isinstance(tile, (CoopTile, PastureTile)) and tile.animal is not None:
                animal_spec = self._config.animals[tile.animal.animal]
                return min(12.0, self._market_price(state, animal_spec.product) / 20.0)
            return 0.0
        if at == ActionType.PLANT:
            from ..state import EmptyTile

            if isinstance(tile, EmptyTile):
                crop = action.farmer_action.crop  # type: ignore[attr-defined]
                crop_spec = self._config.crops[crop]
                product = crop.produce
                price = self._market_price(state, product)
                discount = self._eval.time_discount ** crop_spec.first_yield_day
                return min(
                    12.0,
                    crop_spec.max_yield * price * self._eval.crop_value_weight * discount / 50.0,
                )
            return 0.0
        if at == ActionType.BUY_SEED:
            crop = action.market_actions[0].crop  # type: ignore[attr-defined]
            crop_spec = self._config.crops[crop]
            product = crop.produce
            price = self._market_price(state, product)
            discount = self._eval.time_discount ** crop_spec.first_yield_day
            value = (
                crop_spec.max_yield * price * self._eval.crop_value_weight * discount
                - crop_spec.seed_cost
            )
            return min(10.0, max(0.0, value) / 50.0)
        if at == ActionType.WATER:
            from ..state import PlantTile

            if isinstance(tile, PlantTile) and not tile.plant.watered_today:
                return 4.0
            return 0.0
        if at == ActionType.FEED:
            return 3.0
        if at == ActionType.CARE:
            return 3.0
        return 0.0

    def _action_cost(self, state: SearchState, action: TurnAction) -> float:
        """Small penalty for money-spending actions so cheaper options win ties."""
        at = farmer_type(action)
        if at == ActionType.BUY_SEED:
            cost = self._config.crops[action.market_actions[0].crop].seed_cost  # type: ignore[attr-defined]
        elif at == ActionType.BUY_ANIMAL:
            cost = self._config.animals[action.market_actions[0].animal].cost  # type: ignore[attr-defined]
        elif at == ActionType.BUY_LAND:
            from ..simulator.game_config import LAND_ORDER, LAND_PRICES

            n_extra = len(state.game.players[state.game.current_player].farm.unlocked_quadrants) - 1
            cost = LAND_PRICES[min(n_extra, len(LAND_PRICES) - 1)]
        elif at == ActionType.HIRE:
            from .action_generator import _fib

            cost = self._config.farm_hand_cost_mult * _fib(
                state.game.players[state.game.current_player].farm.hires_today
            )
        elif at == ActionType.BUY_PRODUCT:
            cost = self._market_price(state, action.market_actions[0].item)  # type: ignore[attr-defined]
        else:
            return 0.0
        return min(6.0, cost / 200.0)


ActionFilter = Callable[[SearchState, list[TurnAction]], tuple[TurnAction, ...]]
