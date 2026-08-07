"""Top-level immutable game snapshot and observation adapter.

This module owns the single object that represents the whole game. All rules,
search, and evaluation live elsewhere; ``GameState`` only stores state and
offers the *stable* API surface (``clone``, ``apply``, ``legal_actions``,
``is_terminal``, ``winner``) that the future planner will implement on top of
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .enums import AnimalType, CropType, ItemType, Quadrant, ShopType
from .farm import Farm
from .inventory import Inventory, Seeds
from .market import Market
from .player import PlayerState
from .position import Position
from .tile import (
    EMPTY_TILE,
    LOCKED_TILE,
    WEED_TILE,
    AnimalState,
    CoopTile,
    PastureTile,
    PlantState,
    PlantTile,
    Tile,
)
from .town import Town
from .worker import Worker

# A Kaggle observation is an untyped dict. Only ``from_observation`` and its
# private parsing helpers may rely on these keys; nothing else in the domain
# model is allowed to know about the environment.
Observation = dict[str, Any]


@dataclass(frozen=True, slots=True)
class GameState:
    """An immutable snapshot of the whole game.

    ``players[current_player]`` is the acting player. Every component is
    frozen and hashable, so a ``GameState`` can be used directly as a key in
    a transposition table or cache owned by a search algorithm.
    """

    day: int
    hour: int
    market: Market
    town: Town
    players: tuple[PlayerState, PlayerState]
    current_player: int

    @property
    def current_player_state(self) -> PlayerState:
        """The state of the player whose turn it is."""
        return self.players[self.current_player]

    @property
    def opponent_state(self) -> PlayerState:
        """The state of the other player."""
        return self.players[1 - self.current_player]

    def clone(self) -> "GameState":
        """Return an independent copy of this state.

        The whole state graph is immutable, so the copy is just ``self``; the
        method exists so call sites written against a mutable API keep working
        if the model ever gains mutable internals.
        """
        return self

    def apply(self, action: Any) -> "GameState":
        """Advance the state by one ``action``.

        Intentionally unimplemented: game rules belong to a future environment
        / logic layer, not to the domain model.
        """
        raise NotImplementedError(
            "GameState.apply() is intentionally unimplemented: game rules are "
            "out of scope for the domain model."
        )

    def legal_actions(self) -> list[Any]:
        """The actions legal for the current player this turn."""
        raise NotImplementedError(
            "GameState.legal_actions() is intentionally unimplemented: action "
            "legality is game logic, out of scope for the domain model."
        )

    def is_terminal(self) -> bool:
        """Whether the season is over."""
        raise NotImplementedError(
            "GameState.is_terminal() is intentionally unimplemented."
        )

    def winner(self) -> int | None:
        """The winning player id (0 or 1), or ``None`` on a tie."""
        raise NotImplementedError(
            "GameState.winner() is intentionally unimplemented."
        )

    # -- Observation adapter -------------------------------------------------
    # Everything below understands the Kaggle observation format. This is the
    # ONLY boundary between the environment and the domain model.

    @classmethod
    def from_observation(cls, observation: Observation) -> "GameState":
        """Build a ``GameState`` from a Kaggle observation dict."""
        player_id = int(observation["player"])
        day = int(observation["day"])
        hour = int(observation["hour"])

        raw_farms: list[Any] = observation["farms"]
        market = _parse_market(observation["market"])
        town = _parse_town(observation["town"])

        player_states: list[PlayerState] = []
        for p in range(2):
            farm = _parse_farm(raw_farms[p])
            if p == player_id:
                private: Mapping[str, Any] = observation["private"]
                shed = _parse_inventory(private["shed"])
                seeds = _parse_seeds(private["seeds"])
                raw_inventories: list[Any] = private["inventories"]
                # `inventories` lists the main farmer first, then each hand,
                # matching the farm's public worker order.
                roster = (farm.farmer, *farm.workers)
                workers = tuple(
                    Worker(
                        worker.id,
                        worker.position,
                        _parse_inventory(inv),
                        worker.is_main_farmer,
                    )
                    for worker, inv in zip(roster, raw_inventories)
                )
                # Rebuild the farm so its worker entities carry the private
                # inventories as well.
                farm = Farm(
                    money=farm.money,
                    tiles=farm.tiles,
                    farmer=workers[0],
                    workers=workers[1:],
                    unlocked_quadrants=farm.unlocked_quadrants,
                    hires_today=farm.hires_today,
                )
                player_states.append(
                    PlayerState(
                        farm=farm, inventory=shed, seeds=seeds, workers=workers
                    )
                )
            else:
                # The opponent's private state (shed, seeds, inventories) is
                # not observable, so it is left empty. The search layer must
                # account for this information asymmetry.
                player_states.append(
                    PlayerState(
                        farm=farm,
                        inventory=Inventory.empty(),
                        seeds=Seeds.empty(),
                        workers=(farm.farmer, *farm.workers),
                    )
                )

        return cls(
            day=day,
            hour=hour,
            market=market,
            town=town,
            players=(player_states[0], player_states[1]),
            current_player=player_id,
        )


def _parse_inventory(raw: Mapping[str, Any]) -> Inventory:
    """Build an ``Inventory`` from a ``{item_label: count}`` observation dict."""
    return Inventory(
        {ItemType.from_label(label): int(count) for label, count in raw.items()}
    )


def _parse_seeds(raw: Mapping[str, Any]) -> Seeds:
    """Build a ``Seeds`` container from a ``{crop_label: count}`` dict."""
    return Seeds(
        {CropType.from_label(label): int(count) for label, count in raw.items()}
    )


def _parse_market(raw: Mapping[str, Any]) -> Market:
    """Build a ``Market`` from the shared market observation dict."""
    inventory = _parse_inventory(raw["inventory"])
    prices = {
        ItemType.from_label(label): int(price) for label, price in raw["prices"].items()
    }
    return Market(inventory=inventory, prices=prices)


def _parse_town(raw: Mapping[str, Any]) -> Town:
    """Build a ``Town`` from the shared town observation dict."""
    shops = frozenset(ShopType.from_label(label) for label in raw["unlocked_shops"])
    return Town(unlocked_shops=shops)


def _parse_farm(raw: Mapping[str, Any]) -> Farm:
    """Build a ``Farm`` from a per-player farm observation dict."""
    money = int(raw["money"])

    raw_tiles: list[list[Any]] = raw["tiles"]
    tiles = tuple(tuple(_parse_tile(tile) for tile in row) for row in raw_tiles)

    farmer_x, farmer_y = raw["farmer"]
    farmer = Worker(
        id=0,
        position=Position(int(farmer_x), int(farmer_y)),
        inventory=Inventory.empty(),
        is_main_farmer=True,
    )
    hands = tuple(
        Worker(
            id=index + 1,
            position=Position(int(hx), int(hy)),
            inventory=Inventory.empty(),
            is_main_farmer=False,
        )
        for index, (hx, hy) in enumerate(raw["hands"])
    )

    quadrants = frozenset(
        Quadrant.from_label(label) for label in raw["unlocked_quadrants"]
    )

    return Farm(
        money=money,
        tiles=tiles,
        farmer=farmer,
        workers=hands,
        unlocked_quadrants=quadrants,
        hires_today=int(raw["hires_today"]),
    )


def _parse_tile(raw: Any) -> Tile:
    """Convert one raw tile (``None`` | ``"LOCKED"`` | dict) into a ``Tile``."""
    if raw is None:
        return EMPTY_TILE
    if isinstance(raw, str):
        if raw == "LOCKED":
            return LOCKED_TILE
        raise ValueError(f"unexpected raw tile string: {raw!r}")

    kind = raw.get("kind")
    if kind == "WEED":
        return WEED_TILE
    if kind == "PLANT":
        return PlantTile(
            PlantState(
                crop=CropType.from_label(raw["crop"]),
                planted_day=int(raw["planted_day"]),
                watered_today=bool(raw["watered_today"]),
                consecutive_unwatered=int(raw["consecutive_unwatered"]),
                yield_units=int(raw["yield_units"]),
                fertilized_until_day=int(raw["fertilized_until_day"]),
                max_lifespan_step=int(raw["max_lifespan_step"]),
            )
        )
    if kind in ("COOP", "PASTURE"):
        animal_raw = raw.get("animal")
        animal = (
            AnimalState(
                animal=AnimalType.from_label(animal_raw),
                placed_day=int(raw["placed_day"]),
                yield_units=int(raw["yield_units"]),
                fed_today=bool(raw["fed_today"]),
                consecutive_unfed=int(raw["consecutive_unfed"]),
                cared_today=bool(raw["cared_today"]),
                fertilizer_available=bool(raw["fertilizer_available"]),
                pending_care_bonus=int(raw["pending_care_bonus"]),
            )
            if animal_raw is not None
            else None
        )
        if kind == "COOP":
            return CoopTile(animal)
        return PastureTile(animal)

    raise ValueError(f"unknown tile kind: {kind!r}")
