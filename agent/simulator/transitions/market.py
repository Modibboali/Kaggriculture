"""Market transitions: pricing, town consumption, and order processing.

A faithful port of the official environment's market engine:

* ``market_price`` — the piecewise price model ``base +- amp * f(|inv - I0|)``
  with per-product shape functions, floored at ``PRICE_FLOOR``.
* ``refresh`` — ``_refresh_prices``: recompute every product's sell price.
* ``town_consume`` — ``_town_consume``: periodic shop + town-center demand and
  the unconditional price refresh that follows it each turn.
* ``process_orders`` — ``_process_market``: per-unit lockstep matching of both
  players' order queues (quote both, then commit both, per unit), with prices
  recomputed on the fly so later units see earlier commits.

HIRE and BUY_LAND (workers / land) are not yet implemented: they are handled as
silent no-ops exactly like the environment's handling of illegal orders, and
the first verified transition layer's scenarios never submit them.

Known, documented limitation: the domain ``Inventory`` value object drops
non-positive counts, so market inventories that the environment drives to 0 or
below compare as 0 here. The supported transition scenarios never do this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Union, cast

from ...actions import (
    Action,
    BuyAnimalAction,
    BuyLandAction,
    BuyProductAction,
    BuySeedAction,
    HireAction,
    SellAction,
)
from ...state import (
    AnimalType,
    CropType,
    GameState,
    Inventory,
    ItemType,
    Market,
    PlayerState,
    Seeds,
)
from ..game_config import (
    ANIMAL_COST,
    BUYABLE_PRODUCTS,
    GameConfig,
    PRODUCT_ITEMS,
    SHOPS,
    TOWN_CENTER_SCHEDULE,
)

# Town-center products: every product except FERTILIZER.
TOWN_CENTER_PRODUCTS = tuple(item for item in PRODUCT_ITEMS if item != ItemType.FERTILIZER)

_Item = Union[ItemType, CropType, AnimalType]


def _shape(func: str, x: float) -> float:
    """The environment's per-parameter shape function (``_shape``)."""
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    return x


@dataclass
class _OrderState:
    """One parsed market order, mirroring the environment's queue entry."""

    op: str
    item: _Item
    remaining: int


class MarketTransition:
    """Pricing, town demand, and lockstep order processing."""

    def __init__(self, config: GameConfig) -> None:
        self._config = config

    # -- pricing ------------------------------------------------------------

    def price(self, item: ItemType, inventory: int) -> int:
        """The market's unit price of ``item`` at ``inventory`` units in stock."""
        params = self._config.market_params[item]
        base, i0, t = params.base, params.i0, params.t
        if inventory < i0:
            func = params.below_func
            amp = params.below_target * base / _shape(func, float(t))
            raw = base + amp * _shape(func, float(i0 - inventory))
        else:
            func = params.above_func
            amp = params.above_target * base / _shape(func, float(t))
            raw = base - amp * _shape(func, float(inventory - i0))
        return max(self._config.price_floor, int(round(raw)))

    def refresh(self, market: Market) -> Market:
        """Return a market with every product's sell price recomputed."""
        prices = {
            item: self.price(item, market.inventory.get(item))
            for item in PRODUCT_ITEMS
        }
        return Market(market.inventory, prices)

    # -- town demand --------------------------------------------------------

    def town_consume(self, state: GameState) -> GameState:
        """Apply the periodic shop and town-center demand for the current step."""
        config = self._config
        step = state.step
        shop_interval = max(1, config.town_shop_sell_interval)
        center_interval = max(1, config.town_center_sell_interval)
        day = step // config.turns_per_day

        inventory = dict(state.market.inventory.items)

        if step % shop_interval == 0:
            for shop in state.town.unlocked_shops:
                products = SHOPS[shop]
                multiplier = 2 if len(products) == 1 else 1
                for item in products:
                    inventory[item] = inventory.get(item, 0) - multiplier

        if step % center_interval == 0:
            center_mult = next(
                multiplier for threshold, multiplier in TOWN_CENTER_SCHEDULE if day >= threshold
            )
            for item in TOWN_CENTER_PRODUCTS:
                inventory[item] = inventory.get(item, 0) - center_mult

        market = self.refresh(Market(Inventory(inventory), state.market.prices))
        return replace(state, market=market)

    # -- order processing ---------------------------------------------------

    def process_orders(
        self,
        state: GameState,
        actions0: tuple[Action, ...],
        actions1: tuple[Action, ...],
    ) -> GameState:
        """Match both players' market order queues per-unit, lockstep.

        Returns a new state with money / shed / seeds / market updated. Any
        orders that cannot be filled are silently dropped, as the environment
        does.
        """
        config = self._config
        max_orders = max(1, config.max_market_orders_per_turn)
        shed_capacity = config.shed_capacity

        queues = [
            self._to_order_states(actions0, max_orders),
            self._to_order_states(actions1, max_orders),
        ]
        max_len = max((len(q) for q in queues), default=0)
        if max_len == 0:
            return state

        # Working mutable copies of the private state we commit against.
        monies = [state.players[0].farm.money, state.players[1].farm.money]
        sheds = [
            dict(state.players[0].inventory.items),
            dict(state.players[1].inventory.items),
        ]
        seed_counts = [
            dict(state.players[0].seeds.counts),
            dict(state.players[1].seeds.counts),
        ]
        market_inventory = dict(state.market.inventory.items)

        for index in range(max_len):
            order_states = [
                queues[0][index] if index < len(queues[0]) else None,
                queues[1][index] if index < len(queues[1]) else None,
            ]

            # Atomic orders (HIRE, BUY_LAND) are not implemented: no-op them.
            for player_id, ostate in enumerate(order_states):
                if ostate is not None and ostate.op in ("HIRE", "BUY_LAND"):
                    order_states[player_id] = None

            # Per-unit lockstep: quote both players, then commit both.
            while True:
                quoted: list[tuple[_OrderState, int] | None] = [None, None]
                for player_id, ostate in enumerate(order_states):
                    if ostate is None or ostate.remaining <= 0:
                        continue
                    if ostate.op == "SELL" and isinstance(ostate.item, ItemType) and ostate.item in PRODUCT_ITEMS:
                        quoted[player_id] = (ostate, self.price(ostate.item, market_inventory.get(ostate.item, 0)))
                    elif ostate.op == "BUY_PRODUCT" and isinstance(ostate.item, ItemType) and ostate.item in BUYABLE_PRODUCTS:
                        quoted[player_id] = (
                            ostate,
                            self.price(ostate.item, market_inventory.get(ostate.item, 0) - 1),
                        )
                    elif ostate.op == "BUY_SEED" and isinstance(ostate.item, CropType):
                        quoted[player_id] = (ostate, self._config.crops[ostate.item].seed_cost)
                    elif ostate.op == "BUY_ANIMAL" and isinstance(ostate.item, AnimalType):
                        quoted[player_id] = (ostate, ANIMAL_COST[ostate.item])
                    else:
                        order_states[player_id] = None  # malformed / illegal: abort

                if all(q is None for q in quoted):
                    break

                committed_any = False
                for player_id, quote in enumerate(quoted):
                    if quote is None:
                        continue
                    ostate, price = quote
                    ok = self._commit_unit(
                        ostate, price, player_id, monies, sheds, seed_counts, market_inventory, shed_capacity
                    )
                    if ok:
                        ostate.remaining -= 1
                        committed_any = True
                    else:
                        order_states[player_id] = None

                if not committed_any:
                    break

        return self._rebuild(state, monies, sheds, seed_counts, market_inventory)

    def _to_order_states(self, actions: tuple[Action, ...], max_orders: int) -> list[_OrderState]:
        states: list[_OrderState] = []
        for action in actions[:max_orders]:
            ostate = self._order_state(action)
            if ostate is not None:
                states.append(ostate)
        return states

    def _order_state(self, action: Action) -> _OrderState | None:
        if isinstance(action, SellAction):
            return _OrderState("SELL", action.item, action.quantity)
        if isinstance(action, BuyProductAction):
            return _OrderState("BUY_PRODUCT", action.item, action.quantity)
        if isinstance(action, BuySeedAction):
            return _OrderState("BUY_SEED", action.crop, action.quantity)
        if isinstance(action, BuyAnimalAction):
            return _OrderState("BUY_ANIMAL", action.animal, action.quantity)
        if isinstance(action, HireAction):
            return _OrderState("HIRE", ItemType.WHEAT, action.quantity)
        if isinstance(action, BuyLandAction):
            return _OrderState("BUY_LAND", ItemType.WHEAT, 1)
        return None

    def _commit_unit(
        self,
        ostate: _OrderState,
        price: int,
        player_id: int,
        monies: list[int],
        sheds: list[dict[ItemType, int]],
        seed_counts: list[dict[CropType, int]],
        market_inventory: dict[ItemType, int],
        shed_capacity: int,
    ) -> bool:
        """Mirror the environment's ``_commit_unit``; returns whether filled."""
        if ostate.op == "SELL":
            assert isinstance(ostate.item, ItemType)
            if sheds[player_id].get(ostate.item, 0) <= 0:
                return False
            sheds[player_id][ostate.item] -= 1
            monies[player_id] += price
            if price > 1:
                market_inventory[ostate.item] = market_inventory.get(ostate.item, 0) + 1
            return True

        if ostate.op == "BUY_PRODUCT":
            assert isinstance(ostate.item, ItemType)
            if monies[player_id] < price:
                return False
            if sum(sheds[player_id].values()) >= shed_capacity:
                return False
            monies[player_id] -= price
            sheds[player_id][ostate.item] = sheds[player_id].get(ostate.item, 0) + 1
            market_inventory[ostate.item] = market_inventory.get(ostate.item, 0) - 1
            return True

        if ostate.op == "BUY_SEED":
            assert isinstance(ostate.item, CropType)
            if monies[player_id] < price:
                return False
            monies[player_id] -= price
            seed_counts[player_id][ostate.item] = seed_counts[player_id].get(ostate.item, 0) + 1
            return True

        if ostate.op == "BUY_ANIMAL":
            assert isinstance(ostate.item, AnimalType)
            if monies[player_id] < price:
                return False
            if sum(sheds[player_id].values()) >= shed_capacity:
                return False
            monies[player_id] -= price
            item = ostate.item.as_item
            sheds[player_id][item] = sheds[player_id].get(item, 0) + 1
            return True

        return False

    def _rebuild(
        self,
        state: GameState,
        monies: list[int],
        sheds: list[dict[ItemType, int]],
        seed_counts: list[dict[CropType, int]],
        market_inventory: dict[ItemType, int],
    ) -> GameState:
        players: list[PlayerState] = []
        for player_id in (0, 1):
            player_state = state.players[player_id]
            players.append(
                replace(
                    player_state,
                    farm=replace(player_state.farm, money=monies[player_id]),
                    inventory=Inventory(sheds[player_id]),
                    seeds=Seeds(seed_counts[player_id]),
                )
            )
        market = self.refresh(Market(Inventory(market_inventory), state.market.prices))
        return replace(state, players=cast("tuple[PlayerState, PlayerState]", tuple(players)), market=market)
