"""Classical heuristic evaluation.

The official reward is the player's money at the end of the episode, so the
evaluation values a position in money-equivalent terms: current cash plus the
realizable value of assets (inventory at market price, crop yield potential,
animals at replacement cost, plus modest credits for structures, workers and
seed stock). All weights live in :class:`EvaluationConfig` so they can be
tuned without touching MCTS.

``immediate_reward`` separates the actual reward signal (money at terminal)
from ``evaluate`` (the estimated state value used during search).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..simulator import GameConfig
from ..state import AnimalType, CoopTile, GameState, ItemType, PastureTile, PlantTile, Tile
from .search_state import SearchState


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Weights for the heuristic evaluation (all in money units)."""

    inventory_value_weight: float = 1.0
    crop_value_weight: float = 0.6
    crop_maturity_discount: float = 0.5
    animal_value_weight: float = 0.5
    structure_value: float = 10.0
    worker_value: float = 15.0
    seed_value_weight: float = 0.5


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


def evaluate(state: SearchState, player: int, config: GameConfig, eval_config: EvaluationConfig | None = None) -> float:
    """Module-level convenience wrapper."""
    return Evaluator(config, eval_config).evaluate(state, player)
