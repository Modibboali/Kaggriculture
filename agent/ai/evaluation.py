"""Classical + horizon-aware heuristic evaluation.

The official reward is the player's money at the end of the episode, so a
position is valued in money-equivalent terms. Two evaluators are provided so
they can be compared without changing MCTS:

* :class:`Evaluator` — the classic, *horizon-blind* estimator: cash plus the
  face value of assets (inventory at market price, crops at yield value,
  animals at replacement cost, flat credits for structures/workers/seeds).

* :class:`HorizonAwareEvaluator` — the same structure but with an explicit
  remaining-horizon term, ``V(s, h)``. Each asset class is valued by its
  *realizable* value: whether it can actually be converted to cash before the
  episode terminates.

Valuation convention (no double counting):

    V(s, h) = cash
            + liquidatable inventory (shed + carried) at current market price
            + realizable seeds at cost
            + realizable crops in the ground (yield x market price)
            + realizable animals (replacement cost scaled by production time)
            + structures (enabling value scaled by horizon)
            + workers (productive value scaled by horizon)

Every underlying asset is counted exactly once; land value is *not* added
separately because it flows through the tiles it unlocks (a locked tile can
produce nothing, and unlocked tiles are valued via crops/structures/workers).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..simulator import GameConfig
from ..state import (
    AnimalState,
    AnimalType,
    CoopTile,
    CropType,
    GameState,
    ItemType,
    PastureTile,
    PlantState,
    PlantTile,
    Tile,
)
from .search_state import SearchState


def horizon_remaining(state: SearchState, episode_steps: int) -> int:
    """Steps remaining until the episode terminates (clamped at 0)."""
    return max(0, episode_steps - state.game.step)


def horizon_days(state: SearchState, config: GameConfig) -> int:
    """Whole days remaining until the episode terminates."""
    return horizon_remaining(state, config.episode_steps) // config.turns_per_day


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Weights for the heuristic evaluation (all in money units).

    ``crop_realizability`` / ``animal_horizon_value`` / ``worker_horizon_value`` /
    ``structure_horizon_value`` are ablation switches: turning them off makes the
    corresponding asset class fall back to its classic flat valuation so the
    contribution of each horizon component can be measured in isolation.
    """

    inventory_value_weight: float = 1.0
    crop_value_weight: float = 0.6
    crop_maturity_discount: float = 0.5
    animal_value_weight: float = 0.5
    structure_value: float = 10.0
    worker_value: float = 15.0
    seed_value_weight: float = 0.5

    # Horizon-aware behaviour.
    horizon_window_days: int = 10  # full-value horizon for structures/workers
    crop_realizability: bool = True
    animal_horizon_value: bool = True
    worker_horizon_value: bool = True
    structure_horizon_value: bool = True
    # Time value of money / execution risk: a crop or seed that only becomes
    # cash after ``d`` more days is worth ``time_discount ** d`` of its face
    # value. This stops immature assets with "just enough" time from being
    # valued the same as cash-in-hand, which would otherwise over-encourage
    # investment that never converts before a short terminal.
    time_discount: float = 0.85


class Evaluator:
    """Estimates the money-equivalent value of a player's position."""

    def __init__(self, config: GameConfig, eval_config: EvaluationConfig | None = None) -> None:
        self._config = config
        self._eval = eval_config if eval_config is not None else EvaluationConfig()

    def immediate_reward(self, state: SearchState, player: int) -> float:
        """The actual reward signal: money (nonzero only at terminal)."""
        return float(state.game.players[player].farm.money)

    def evaluate(self, state: SearchState, player: int) -> float:
        """The estimated state value for ``player`` (money + asset value)."""
        game = state.game
        player_state = game.players[player]
        farm = player_state.farm
        value = float(farm.money)

        # Shed + carried inventories: products at market price, animals at cost.
        inventories = [player_state.inventory, *(worker.inventory for worker in player_state.workers)]
        for inventory in inventories:
            for item, count in inventory.items.items():
                value += count * self._item_value(game, item) * self._eval.inventory_value_weight

        # Seed stock at cost.
        for crop, count in player_state.seeds.counts.items():
            value += count * self._config.crops[crop].seed_cost * self._eval.seed_value_weight

        # Tiles: crops, structures, animals.
        for row in farm.tiles:
            for tile in row:
                value += self._tile_value(game, tile, farm, player_state)

        # Hired hands.
        value += len(farm.workers) * self._eval.worker_value

        return value

    # -- helpers ------------------------------------------------------------

    def _market_price(self, game: GameState, item: ItemType) -> float:
        return float(game.market.price(item))

    def _item_value(self, game: GameState, item: ItemType) -> float:
        """Money-equivalent value of one unit of ``item``.

        Products are valued at the current market price; animals (which have no
        market price) are valued at their purchase cost.
        """
        if item in game.market.prices:
            return float(game.market.prices[item])
        animal = AnimalType[item.name]
        return float(self._config.animals[animal].cost)

    def _tile_value(self, game: GameState, tile: Tile, farm: object, player_state: object) -> float:
        del farm, player_state
        if isinstance(tile, PlantTile):
            plant = tile.plant
            product = plant.crop.produce
            spec = self._config.crops[plant.crop]
            mature = game.day - plant.planted_day >= spec.first_yield_day
            maturity = 1.0 if (mature and plant.yield_units > 0) else self._eval.crop_maturity_discount
            return (
                plant.yield_units
                * self._market_price(game, product)
                * self._eval.crop_value_weight
                * maturity
            )
        if isinstance(tile, (CoopTile, PastureTile)):
            total = self._eval.structure_value
            if tile.animal is not None:
                total += self._config.animals[tile.animal.animal].cost * self._eval.animal_value_weight
            return total
        return 0.0


class HorizonAwareEvaluator(Evaluator):
    """Horizon-aware ``V(s, h)``: assets are valued by what is realizable
    before the episode terminates.

    Terminal states (``h <= 0``) are valued as pure cash — the game's actual
    reward — so no speculative future value survives past the end of the game.
    """

    def evaluate(self, state: SearchState, player: int) -> float:
        game = state.game
        h_steps = horizon_remaining(state, self._config.episode_steps)
        h_days = h_steps // self._config.turns_per_day
        if h_steps <= 0:
            # Terminal: only cash is realizable (matches the game reward).
            return float(game.players[player].farm.money)

        player_state = game.players[player]
        farm = player_state.farm
        value = float(farm.money)

        # Inventory (shed + carried): products at market price, scaled by
        # liquidity — how much can actually be sold before terminal (selling
        # is a market order, limited to ``max_market_orders_per_turn`` per
        # turn). Holding more than can be sold in the remaining steps is worth
        # less than face value, which pushes the agent to sell near the end.
        inventories = [player_state.inventory, *(worker.inventory for worker in player_state.workers)]
        for inventory in inventories:
            for item, count in inventory.items.items():
                value += (
                    count
                    * self._item_value(game, item)
                    * self._eval.inventory_value_weight
                    * self._inventory_liquidity(count, h_steps)
                )

        # Seeds: realizable only if they can be planted -> grown -> harvested
        # -> sold inside the remaining horizon.
        for crop, count in player_state.seeds.counts.items():
            value += (
                count
                * self._config.crops[crop].seed_cost
                * self._eval.seed_value_weight
                * self._seed_realizability(crop, h_days)
            )

        # Tiles: crops, structures, animals.
        for row in farm.tiles:
            for tile in row:
                value += self._tile_value_h(game, tile, h_days)

        # Hired hands.
        value += len(farm.workers) * self._eval.worker_value * self._worker_realizability(h_days)

        return value

    # -- horizon-aware helpers ---------------------------------------------

    def _clamp01(self, x: float) -> float:
        return min(1.0, max(0.0, x))

    def _inventory_liquidity(self, count: int, h_steps: int) -> float:
        """Fraction of ``count`` units of one product sellable before terminal.

        One SELL order moves one unit and at most ``max_market_orders_per_turn``
        orders may be submitted per turn, so the sellable quantity is
        ``orders_per_turn * h_steps``. Everything beyond that is unrealizable.
        """
        capacity = self._config.max_market_orders_per_turn * h_steps
        if capacity <= 0 or count <= 0:
            return 0.0
        return self._clamp01(capacity / count)

    def _seed_realizability(self, crop: CropType, h_days: int) -> float:
        """Fraction of a seed's cost that is realizable within ``h_days``.

        A seed needs ``first_yield_day`` days to grow plus roughly a day of
        plant/harvest/sell overhead before it becomes cash, and that future
        cash is time-discounted.
        """
        if not self._eval.crop_realizability:
            return 1.0  # classic flat credit
        spec = self._config.crops[crop]
        needed = spec.first_yield_day + 1
        feasibility = self._clamp01(h_days / needed if needed > 0 else 1.0)
        discount = self._eval.time_discount ** spec.first_yield_day
        return feasibility * discount

    def _crop_realizability(self, plant: PlantState, day: int, h_days: int) -> float:
        """Fraction of an in-ground crop's yield realizable before terminal.

        Mature crops can be harvested immediately (full value). Immature crops
        need ``time_to_mature`` more days to mature and a harvest day; if that
        does not fit in the horizon their value scales with the fraction of the
        growth that can still happen (partial, not deleted), and the future
        harvest is time-discounted so near-term cash is preferred over distant
        crop potential.
        """
        if not self._eval.crop_realizability:
            # Classic: mature -> full, otherwise the maturity discount.
            spec = self._config.crops[plant.crop]
            mature = day - plant.planted_day >= spec.first_yield_day
            if mature and plant.yield_units > 0:
                return 1.0
            return self._eval.crop_maturity_discount

        spec = self._config.crops[plant.crop]
        age = day - plant.planted_day
        if age >= spec.first_yield_day and plant.yield_units > 0:
            return 1.0  # harvestable now
        time_to_mature = max(0, spec.first_yield_day - age)
        needed = time_to_mature + 1  # +1 harvest day
        if needed <= 0:
            return 1.0
        feasibility = self._clamp01(h_days / needed)
        discount = self._eval.time_discount ** time_to_mature
        return feasibility * discount

    def _animal_realizability(self, animal: AnimalState, day: int, h_days: int) -> float:
        """Fraction of an animal's replacement value realizable in ``h_days``.

        An animal that cannot even reach its first production day before the
        terminal is worth little; one with enough time to produce is worth its
        replacement cost (scaled by ``animal_value_weight``).
        """
        if not self._eval.animal_horizon_value:
            return 1.0  # classic flat credit
        spec = self._config.animals[animal.animal]
        age = day - animal.placed_day
        time_to_first = max(0, spec.first_yield_day - age)
        needed = time_to_first + 1  # +1 harvest day
        if needed <= 0:
            return 1.0
        return self._clamp01(h_days / needed)

    def _worker_realizability(self, h_days: int) -> float:
        """Fraction of a hired hand's productive value realizable in ``h_days``."""
        if not self._eval.worker_horizon_value:
            return 1.0  # classic flat credit
        return self._clamp01(h_days / self._eval.horizon_window_days)

    def _structure_realizability(self, h_days: int) -> float:
        """Fraction of a structure's enabling value realizable in ``h_days``."""
        if not self._eval.structure_horizon_value:
            return 1.0  # classic flat credit
        return self._clamp01(h_days / self._eval.horizon_window_days)

    def _tile_value_h(self, game: GameState, tile: Tile, h_days: int) -> float:
        if isinstance(tile, PlantTile):
            plant = tile.plant
            product = plant.crop.produce
            return (
                plant.yield_units
                * self._market_price(game, product)
                * self._eval.crop_value_weight
                * self._crop_realizability(plant, game.day, h_days)
            )
        if isinstance(tile, (CoopTile, PastureTile)):
            total = self._eval.structure_value * self._structure_realizability(h_days)
            if tile.animal is not None:
                spec = self._config.animals[tile.animal.animal]
                total += (
                    spec.cost
                    * self._eval.animal_value_weight
                    * self._animal_realizability(tile.animal, game.day, h_days)
                )
            return total
        return 0.0


def evaluate(state: SearchState, player: int, config: GameConfig, eval_config: EvaluationConfig | None = None) -> float:
    """Module-level convenience wrapper (classic evaluator)."""
    return Evaluator(config, eval_config).evaluate(state, player)


def evaluate_horizon(
    state: SearchState,
    player: int,
    config: GameConfig,
    eval_config: EvaluationConfig | None = None,
) -> float:
    """Module-level convenience wrapper (horizon-aware evaluator)."""
    return HorizonAwareEvaluator(config, eval_config).evaluate(state, player)
