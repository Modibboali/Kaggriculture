"""Deterministic, decision-time feature encoders for Kaggriculture.

These are the *observation* / *action* encoders MuZero consumes. They are
pure functions of the current :class:`~agent.ai.search_state.SearchState` and
the current candidate action — no teacher action, no future state, no
terminal reward, and no opponent private state ever enters a feature vector
(each vector is built from the acting player's own public+private state plus
the shared market / town / time).

Two vectors:

* :class:`StateEncoder` — fixed-size 139-dim state observation
  (``state_feature_width() == 139``).
* :class:`ActionEncoder` — fixed-size 60-dim per-candidate action embedding
  (``action_feature_width() == 60``), so the policy / dynamics operate on
  *candidate-conditioned* action representations (never a global fixed action
  vocabulary).

The exact same encoders are used for training, self-play, MCTS and inference.
"""

from __future__ import annotations

import numpy as np

from ..actions import (
    BuyAnimalAction,
    BuyLandAction,
    BuyProductAction,
    BuySeedAction,
    HireAction,
    MovementAction,
    SellAction,
    TurnAction,
)
from ..ai.action_priority import ActionPriorityModel, farmer_type
from ..ai.search_state import SearchState
from ..simulator import GameConfig
from ..state import (
    AnimalType,
    CoopTile,
    CropType,
    Direction,
    EmptyTile,
    ItemType,
    LockedTile,
    PastureTile,
    PlantTile,
    Quadrant,
    ShopType,
    WeedTile,
)

ENCODER_VERSION = 1

# Canonical enum orderings (stable, independent of dict iteration order).
CROP_TYPES: tuple[CropType, ...] = tuple(CropType)
ANIMAL_TYPES: tuple[AnimalType, ...] = tuple(AnimalType)
ITEM_TYPES: tuple[ItemType, ...] = tuple(ItemType)
QUADRANTS: tuple[Quadrant, ...] = tuple(Quadrant)
SHOP_TYPES: tuple[ShopType, ...] = tuple(ShopType)
_DIRECTIONS: tuple[Direction, ...] = (Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST)

PHASE_NAMES = ("development", "production", "cash_conversion", "terminal")

_ACTION_INDEX: dict[object, int] | None = None


def _action_index(action_type: object) -> int:
    global _ACTION_INDEX
    if _ACTION_INDEX is None:
        from ..actions import ActionType

        _ACTION_INDEX = {t: i for i, t in enumerate(tuple(ActionType))}
    return _ACTION_INDEX[action_type]


def _phase_index(state: SearchState, config: GameConfig) -> int:
    from ..ai.phase import phase_of

    label = phase_of(state, config).label
    return PHASE_NAMES.index(label) if label in PHASE_NAMES else 0


# ---------------------------------------------------------------------------
# State encoder
# ---------------------------------------------------------------------------

STATE_FEATURE_NAMES: tuple[str, ...] = (
    # time / horizon (4)
    "day", "hour", "step", "remaining_steps",
    # economy (2)
    "money", "seed_total",
    # seeds per crop (5)
    *(f"seed_{crop.name}" for crop in CROP_TYPES),
    # shed inventory per item (12)
    *(f"shed_{item.name}" for item in ITEM_TYPES),
    # farmer carried inventory per item (12)
    *(f"farmer_{item.name}" for item in ITEM_TYPES),
    # workers (2)
    "worker_count", "worker_inventory_units",
    # unlocked quadrants (4)
    *(f"unlocked_{q.name}" for q in QUADRANTS),
    # tile composition (7)
    "tile_empty", "tile_weed", "tile_plant", "tile_coop", "tile_pasture",
    "tile_locked", "tile_unlocked_count",
    # crop lifecycle (5 crops x 6)
    *(f"{crop.name}_{attr}" for crop in CROP_TYPES
      for attr in ("planted", "mature", "yield_units", "age_days", "watered", "fertilized")),
    # animal state (3 x 5)
    *(f"{animal.name}_{attr}" for animal in ANIMAL_TYPES
      for attr in ("count", "yield_units", "fed", "cared", "age_days")),
    # market prices per item (12)
    *(f"price_{item.name}" for item in ITEM_TYPES),
    # market inventory per item (12)
    *(f"market_inv_{item.name}" for item in ITEM_TYPES),
    # town shops (8)
    *(f"shop_{shop.name}" for shop in SHOP_TYPES),
    # phase one-hot (4)
    *(f"phase_{name}" for name in PHASE_NAMES),
    # derived economic values (10)
    "cash", "liquid_inventory_value", "seed_value", "unrealized_crop_value",
    "unrealized_animal_value", "structure_value", "total_asset_value",
    "realizable_asset_value", "illiquid_asset_value", "time_to_next_yield_days",
)


def state_feature_width() -> int:
    return len(STATE_FEATURE_NAMES)


class StateEncoder:
    """Deterministic fixed-size state observation (decision-time only)."""

    VERSION = ENCODER_VERSION

    def __init__(self, config: GameConfig) -> None:
        self._config = config
        self._episode_steps = float(config.episode_steps)

    @property
    def width(self) -> int:
        return state_feature_width()

    def encode(self, state: SearchState) -> np.ndarray:
        """The 139-dim float32 observation for the acting player.

        ``state`` must have ``current_player`` set to the player whose
        observation this is. All features derive from the current state only —
        no future information.
        """
        game = state.game
        player = game.current_player
        ps = game.players[player]
        farm = ps.farm
        feats: list[float] = []
        remaining = max(0.0, self._episode_steps - float(game.step))

        feats += [float(game.day), float(game.hour), float(game.step), remaining]
        feats += [float(farm.money), float(ps.seeds.total())]
        feats += [float(ps.seeds.get(c)) for c in CROP_TYPES]
        feats += [float(ps.inventory.get(i)) for i in ITEM_TYPES]

        farmer_inv = farm.farmer.inventory
        feats += [float(farmer_inv.get(i)) for i in ITEM_TYPES]

        worker_inventory_units = 0
        for w in farm.workers:
            worker_inventory_units += sum(w.inventory.items.values())
        feats += [float(len(farm.workers)), float(worker_inventory_units)]

        unlocked = farm.unlocked_quadrants
        feats += [1.0 if q in unlocked else 0.0 for q in QUADRANTS]

        tile_counts = {EmptyTile: 0, WeedTile: 0, PlantTile: 0, CoopTile: 0, PastureTile: 0, LockedTile: 0}
        for row in farm.tiles:
            for tile in row:
                for cls in tile_counts:
                    if isinstance(tile, cls):
                        tile_counts[cls] += 1
                        break
        feats += [float(tile_counts[EmptyTile]), float(tile_counts[WeedTile]),
                  float(tile_counts[PlantTile]), float(tile_counts[CoopTile]),
                  float(tile_counts[PastureTile]), float(tile_counts[LockedTile])]
        unlocked_tiles = sum(
            1 for row in farm.tiles for t in row if not isinstance(t, LockedTile)
        )
        feats += [float(unlocked_tiles)]

        for crop in CROP_TYPES:
            planted = mature = 0
            yield_units = age_days = watered = fertilized = 0
            for row in farm.tiles:
                for tile in row:
                    if isinstance(tile, PlantTile) and tile.plant.crop == crop:
                        p = tile.plant
                        planted += 1
                        age_days += game.day - p.planted_day
                        yield_units += p.yield_units
                        watered += 1 if p.watered_today else 0
                        fertilized += 1 if game.day <= p.fertilized_until_day else 0
                        spec = self._config.crops[crop]
                        if p.yield_units > 0 and game.day - p.planted_day >= spec.first_yield_day:
                            mature += 1
            feats += [float(planted), float(mature), float(yield_units),
                      float(age_days), float(watered), float(fertilized)]

        for animal in ANIMAL_TYPES:
            count = yield_units = fed = cared = 0
            age_days = 0
            for row in farm.tiles:
                for tile in row:
                    if isinstance(tile, (CoopTile, PastureTile)) and tile.animal is not None:
                        a = tile.animal
                        if a.animal == animal:
                            count += 1
                            age_days += game.day - a.placed_day
                            yield_units += a.yield_units
                            fed += 1 if a.fed_today else 0
                            cared += 1 if a.cared_today else 0
            feats += [float(count), float(yield_units), float(fed), float(cared), float(age_days)]

        feats += [float(game.market.prices.get(i, 0)) for i in ITEM_TYPES]
        feats += [float(game.market.inventory.get(i)) for i in ITEM_TYPES]
        feats += [1.0 if s in game.town.unlocked_shops else 0.0 for s in SHOP_TYPES]
        phase = _phase_index(state, self._config)
        feats += [1.0 if i == phase else 0.0 for i in range(len(PHASE_NAMES))]
        feats += list(_derived_economic(state, self._config).values())

        arr = np.asarray(feats, dtype=np.float32)
        assert arr.shape == (state_feature_width(),), arr.shape
        return arr


# ---------------------------------------------------------------------------
# Action encoder
# ---------------------------------------------------------------------------

def _build_action_names() -> tuple[str, ...]:
    from ..actions import ActionType

    return (
        *(f"at_{t.value}" for t in tuple(ActionType)),
        *(f"crop_{c.name}" for c in CROP_TYPES),
        *(f"animal_{a.name}" for a in ANIMAL_TYPES),
        *(f"item_{i.name}" for i in ITEM_TYPES),
        *(f"quadrant_{q.name}" for q in QUADRANTS),
        *(f"dir_{d.name}" for d in _DIRECTIONS),
        "quantity", "cost", "market_price", "realizable",
        *(f"phase_{name}" for name in PHASE_NAMES),
    )


ACTION_FEATURE_NAMES: tuple[str, ...] = _build_action_names()


def action_feature_width() -> int:
    return len(ACTION_FEATURE_NAMES)


class ActionEncoder:
    """Deterministic fixed-size per-candidate action embedding."""

    VERSION = ENCODER_VERSION

    def __init__(self, config: GameConfig) -> None:
        self._config = config

    @property
    def width(self) -> int:
        return action_feature_width()

    def encode(self, state: SearchState, action: TurnAction) -> np.ndarray:
        """The 60-dim float32 embedding of ``action`` for the acting player."""
        from ..actions import ActionType

        feats: list[float] = [0.0] * 24
        feats[_action_index(farmer_type(action))] = 1.0

        crop_hot = [0.0] * 5
        animal_hot = [0.0] * 3
        item_hot = [0.0] * 12
        quad_hot = [0.0] * 4
        dir_hot = [0.0] * 4
        quantity = 0.0

        farmer = action.farmer_action
        ftype = farmer.action_type
        if isinstance(farmer, MovementAction):
            dir_hot[_DIRECTIONS.index(farmer.direction)] = 1.0
        elif ftype == ActionType.PLANT:
            crop = getattr(farmer, "crop", None)
            if crop is not None:
                crop_hot[CROP_TYPES.index(crop)] = 1.0
        elif ftype == ActionType.PLACE:
            animal = getattr(farmer, "animal", None)
            if animal is not None:
                animal_hot[ANIMAL_TYPES.index(animal)] = 1.0
        elif ftype in (ActionType.PICKUP, ActionType.DROP):
            item = getattr(farmer, "item", None)
            if item is not None:
                item_hot[ITEM_TYPES.index(item)] = 1.0
            quantity += float(getattr(farmer, "quantity", 0))

        if action.market_actions:
            ma = action.market_actions[0]
            mtype = ma.action_type
            if mtype == ActionType.BUY_SEED:
                crop = getattr(ma, "crop", None)
                if crop is not None:
                    crop_hot[CROP_TYPES.index(crop)] = 1.0
                quantity += float(getattr(ma, "quantity", 0))
            elif mtype == ActionType.BUY_ANIMAL:
                animal = getattr(ma, "animal", None)
                if animal is not None:
                    animal_hot[ANIMAL_TYPES.index(animal)] = 1.0
                quantity += float(getattr(ma, "quantity", 0))
            elif mtype == ActionType.BUY_PRODUCT:
                item = getattr(ma, "item", None)
                if item is not None:
                    item_hot[ITEM_TYPES.index(item)] = 1.0
                quantity += float(getattr(ma, "quantity", 0))
            elif mtype == ActionType.SELL:
                item = getattr(ma, "item", None)
                if item is not None:
                    item_hot[ITEM_TYPES.index(item)] = 1.0
                quantity += float(getattr(ma, "quantity", 0))
            elif mtype == ActionType.BUY_LAND:
                q = getattr(ma, "quadrant", None)
                if q is not None:
                    quad_hot[QUADRANTS.index(q)] = 1.0
            elif mtype == ActionType.HIRE:
                quantity += float(getattr(ma, "quantity", 0))

        cost = self._action_cost(state, action)
        market_price = self._action_market_price(state, action)
        realizable = 1.0 if self._realizable(state, action) else 0.0

        feats += crop_hot + animal_hot + item_hot + quad_hot + dir_hot
        feats += [quantity, cost, market_price, realizable]
        phase = _phase_index(state, self._config)
        feats += [1.0 if i == phase else 0.0 for i in range(len(PHASE_NAMES))]

        arr = np.asarray(feats, dtype=np.float32)
        assert arr.shape == (action_feature_width(),), arr.shape
        return arr

    # -- helpers ------------------------------------------------------------

    def _action_cost(self, state: SearchState, action: TurnAction) -> float:
        from ..simulator.game_config import LAND_PRICES

        game = state.game
        farm = game.players[game.current_player].farm
        money = 0.0
        if action.market_actions:
            ma = action.market_actions[0]
            if isinstance(ma, BuySeedAction):
                money += float(self._config.crops[ma.crop].seed_cost * ma.quantity)
            elif isinstance(ma, BuyAnimalAction):
                money += float(self._config.animals[ma.animal].cost * ma.quantity)
            elif isinstance(ma, BuyProductAction):
                price = game.market.prices.get(ma.item, 0)
                money += float(price * ma.quantity)
            elif isinstance(ma, BuyLandAction):
                n = len(farm.unlocked_quadrants) - 1
                money += float(LAND_PRICES[n]) if 0 <= n < len(LAND_PRICES) else 0.0
            elif isinstance(ma, HireAction):
                money += float(self._config.farm_hand_cost_mult * _fib(farm.hires_today) * ma.quantity)
        return money

    def _action_market_price(self, state: SearchState, action: TurnAction) -> float:
        game = state.game
        if action.market_actions:
            ma = action.market_actions[0]
            if isinstance(ma, (BuyProductAction, SellAction)):
                return float(game.market.prices.get(ma.item, 0))
        return 0.0

    def _realizable(self, state: SearchState, action: TurnAction) -> bool:
        return ActionPriorityModel(self._config).can_realize(action, state)


def _fib(n: int) -> int:
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# ---------------------------------------------------------------------------
# Derived economic values (decision-time only)
# ---------------------------------------------------------------------------

def _derived_economic(state: SearchState, config: GameConfig) -> dict[str, float]:
    """Decision-time asset breakdown (all from the current observation)."""
    game = state.game
    player = game.current_player
    ps = game.players[player]
    farm = ps.farm
    cash = float(farm.money)

    liquid = 0.0
    for item, count in ps.inventory.items.items():
        liquid += count * float(game.market.prices.get(item, 0))
    for item, count in farm.farmer.inventory.items.items():
        liquid += count * float(game.market.prices.get(item, 0))
    for w in farm.workers:
        for item, count in w.inventory.items.items():
            liquid += count * float(game.market.prices.get(item, 0))

    seed_value = 0.0
    for crop in CROP_TYPES:
        seed_value += ps.seeds.get(crop) * float(config.crops[crop].seed_cost)

    unrealized_crop = 0.0
    unrealized_animal = 0.0
    for row in farm.tiles:
        for tile in row:
            if isinstance(tile, PlantTile):
                p = tile.plant
                price = game.market.prices.get(p.crop.produce, 0)
                unrealized_crop += p.yield_units * float(price)
            elif isinstance(tile, (CoopTile, PastureTile)) and tile.animal is not None:
                a = tile.animal
                unrealized_animal += float(config.animals[a.animal].cost)

    structure_value = 10.0 * sum(
        1 for row in farm.tiles for t in row if isinstance(t, (CoopTile, PastureTile))
    )
    total = cash + liquid + seed_value + unrealized_crop + unrealized_animal + structure_value
    realizable = cash + liquid + min(unrealized_crop + unrealized_animal, total * 0.5)
    illiquid = total - cash - liquid
    next_yield_days = _time_to_next_yield(state, config)
    return {
        "cash": cash,
        "liquid_inventory_value": liquid,
        "seed_value": seed_value,
        "unrealized_crop_value": unrealized_crop,
        "unrealized_animal_value": unrealized_animal,
        "structure_value": structure_value,
        "total_asset_value": total,
        "realizable_asset_value": realizable,
        "illiquid_asset_value": illiquid,
        "time_to_next_yield_days": float(next_yield_days),
    }


def _time_to_next_yield(state: SearchState, config: GameConfig) -> int:
    game = state.game
    player = game.current_player
    farm = game.players[player].farm
    best = 10**6
    for row in farm.tiles:
        for tile in row:
            if isinstance(tile, PlantTile):
                p = tile.plant
                crop_spec = config.crops[p.crop]
                if p.yield_units > 0:
                    best = min(best, 0)
                elif game.day < crop_spec.first_yield_day:
                    best = min(best, crop_spec.first_yield_day - game.day)
            elif isinstance(tile, (CoopTile, PastureTile)) and tile.animal is not None:
                a = tile.animal
                animal_spec = config.animals[a.animal]
                best = min(best, max(0, animal_spec.first_yield_day - game.day))
    return best if best < 10**6 else 0
