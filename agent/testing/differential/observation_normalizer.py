"""Canonical (semantic) state representation.

The differential framework never compares raw Kaggle dictionaries or Python
object identity. Instead, a :class:`~agent.state.game_state.GameState` (whether
produced by the observation adapter or by our future simulator) is normalized
into a plain, ordered, JSON-safe dictionary that captures every *relevant*
field. Two canonical dictionaries are then compared path-by-path by
:mod:`agent.testing.differential.state_diff`.

The canonical form mirrors the observation structure (``farms``, ``private``,
``market``, ``town``) so diff paths read naturally (e.g.
``farms[0].tiles[2][3].plant.watered_today``). It is built from the observing
player's perspective: the opponent's private state is unobservable and is
therefore not present.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...state import (
    CoopTile,
    EmptyTile,
    GameState,
    LockedTile,
    PastureTile,
    PlantTile,
    PlayerState,
    Tile,
    WeedTile,
)

# A canonical state is a plain, ordered, JSON-safe dictionary.
CanonicalState = dict[str, Any]


def normalize(state: GameState) -> CanonicalState:
    """Return the canonical form of ``state`` (deterministic and value-based)."""
    return {
        "day": state.day,
        "hour": state.hour,
        "step": state.step,
        "current_player": state.current_player,
        "farms": [_normalize_player(player) for player in state.players],
        "private": _normalize_private(state.players[state.current_player]),
        "market": {
            "inventory": _normalize_counts(state.market.inventory.items),
            "prices": {
                item.label: price
                for item, price in sorted(
                    state.market.prices.items(), key=lambda kv: kv[0].label
                )
            },
        },
        "town": {
            "unlocked_shops": sorted(shop.label for shop in state.town.unlocked_shops)
        },
    }


def _label(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def _normalize_counts(mapping: Mapping[Any, int]) -> dict[str, int]:
    """Normalize a count mapping into a sorted ``{label: count}`` dict."""
    return {
        _label(key): count
        for key, count in sorted(mapping.items(), key=lambda kv: _label(kv[0]))
    }


def _normalize_player(player: PlayerState) -> CanonicalState:
    farm = player.farm
    return {
        "money": farm.money,
        "farmer": [farm.farmer.position.x, farm.farmer.position.y],
        "hands": [[worker.position.x, worker.position.y] for worker in farm.workers],
        "unlocked_quadrants": sorted(
            quadrant.label for quadrant in farm.unlocked_quadrants
        ),
        "hires_today": farm.hires_today,
        "tiles": [[_normalize_tile(tile) for tile in row] for row in farm.tiles],
    }


def _normalize_private(player: PlayerState) -> CanonicalState:
    """The observing player's private state (shed, seeds, worker inventories)."""
    return {
        "shed": _normalize_counts(player.inventory.items),
        "seeds": _normalize_counts(player.seeds.counts),
        "inventories": [
            _normalize_counts(worker.inventory.items) for worker in player.workers
        ],
    }


def _normalize_tile(tile: Tile) -> CanonicalState:
    if isinstance(tile, EmptyTile):
        return {"kind": "EMPTY"}
    if isinstance(tile, LockedTile):
        return {"kind": "LOCKED"}
    if isinstance(tile, WeedTile):
        return {"kind": "WEED"}
    if isinstance(tile, PlantTile):
        plant = tile.plant
        return {
            "kind": "PLANT",
            "plant": {
                "crop": plant.crop.label,
                "planted_day": plant.planted_day,
                "watered_today": plant.watered_today,
                "consecutive_unwatered": plant.consecutive_unwatered,
                "yield_units": plant.yield_units,
                "fertilized_until_day": plant.fertilized_until_day,
                "max_lifespan_step": plant.max_lifespan_step,
            },
        }
    if isinstance(tile, (CoopTile, PastureTile)):
        kind = "COOP" if isinstance(tile, CoopTile) else "PASTURE"
        animal = None
        if tile.animal is not None:
            animal = {
                "animal": tile.animal.animal.label,
                "placed_day": tile.animal.placed_day,
                "yield_units": tile.animal.yield_units,
                "fed_today": tile.animal.fed_today,
                "consecutive_unfed": tile.animal.consecutive_unfed,
                "cared_today": tile.animal.cared_today,
                "fertilizer_available": tile.animal.fertilizer_available,
                "pending_care_bonus": tile.animal.pending_care_bonus,
            }
        return {"kind": kind, "animal": animal}
    raise TypeError(f"unhandled tile type: {type(tile).__name__}")
