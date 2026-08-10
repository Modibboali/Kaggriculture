"""Explicit validation for Kaggriculture observations.

Validation is strict and path-aware: every error identifies the offending
field, e.g. ``"Invalid observation: farms[0].farmer must be a [x, y] position"``.
The adapter calls :func:`validate_observation` before translating, so a
malformed observation can never silently produce a corrupt domain state.
"""

from __future__ import annotations

from typing import Any, Mapping, NoReturn, cast

from .observation_types import (
    ANIMAL_TILE_FIELDS,
    FARM_FIELDS,
    MARKET_FIELDS,
    PLANT_FIELDS,
    PRIVATE_FIELDS,
    TILE_KINDS,
    TOP_LEVEL_FIELDS,
    TOWN_FIELDS,
)


class InvalidObservationError(ValueError):
    """Raised when an observation does not match the Kaggriculture schema."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fail(path: str, message: str) -> NoReturn:
    raise InvalidObservationError(f"Invalid observation: {path} {message}")


def _require(
    container: Mapping[str, Any], key: str, expected: type, path: str
) -> Any:
    """Return ``container[key]`` after checking presence and type."""
    if key not in container:
        _fail(path, f"is required (missing key '{key}')")
    value = container[key]
    if not isinstance(value, expected):
        _fail(path, f"must be {expected.__name__}, got {type(value).__name__}")
    return value


def _require_int(container: Mapping[str, Any], key: str, path: str) -> int:
    value = _require(container, key, int, path)
    if not _is_int(value):
        _fail(path, f"must be an integer, got {type(value).__name__}")
    return cast(int, value)


def _require_number(container: Mapping[str, Any], key: str, path: str) -> int | float:
    value = container[key] if key in container else _require(container, key, object, path)
    if not _is_number(value):
        _fail(path, f"must be a number, got {type(value).__name__}")
    return cast(int | float, value)


def validate_observation(observation: Mapping[str, Any]) -> None:
    """Validate an observation against the Kaggriculture schema.

    Raises :class:`InvalidObservationError` on the first offending field. The
    input is only read, never modified.
    """
    if not isinstance(observation, Mapping):
        raise InvalidObservationError(
            "Invalid observation: top-level value must be an object/dict"
        )

    _validate_top_level(observation)

    farms = _require(observation, "farms", list, "farms")
    if len(farms) != 2:
        _fail("farms", f"must contain exactly 2 farms, got {len(farms)}")
    for index, farm in enumerate(farms):
        _validate_farm(farm, f"farms[{index}]")

    _validate_private(
        _require(observation, "private", dict, "private"), "private"
    )
    _validate_market(_require(observation, "market", dict, "market"), "market")
    _validate_town(_require(observation, "town", dict, "town"), "town")


def _validate_top_level(observation: Mapping[str, Any]) -> None:
    for key in ("player", "day", "hour", "step"):
        _require_int(observation, key, key)
    player = observation["player"]
    if player not in (0, 1):
        _fail("player", f"must be 0 or 1, got {player!r}")
    if "remainingOverageTime" in observation and not _is_number(
        observation["remainingOverageTime"]
    ):
        _fail("remainingOverageTime", "must be a number")


def _validate_farm(farm: Any, path: str) -> None:
    if not isinstance(farm, dict):
        _fail(path, "must be an object/dict")
    _require_number(farm, "money", f"{path}.money")
    tiles = _require(farm, "tiles", list, f"{path}.tiles")
    if not tiles:
        _fail(f"{path}.tiles", "must be a non-empty 2D grid")
    for y, row in enumerate(tiles):
        if not isinstance(row, list):
            _fail(f"{path}.tiles[{y}]", "must be a row (list of tiles)")
        for x, tile in enumerate(row):
            _validate_tile(tile, f"{path}.tiles[{y}][{x}]")

    _validate_position(
        _require(farm, "farmer", list, f"{path}.farmer"), f"{path}.farmer"
    )
    hands = _require(farm, "hands", list, f"{path}.hands")
    for index, hand in enumerate(hands):
        _validate_position(hand, f"{path}.hands[{index}]")

    unlocked = _require(farm, "unlocked_quadrants", list, f"{path}.unlocked_quadrants")
    for index, label in enumerate(unlocked):
        if not isinstance(label, str):
            _fail(f"{path}.unlocked_quadrants[{index}]", "must be a string label")

    _require_int(farm, "hires_today", f"{path}.hires_today")


def _validate_position(value: Any, path: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        _fail(path, "must be a [x, y] position")
    x, y = value
    if not (_is_int(x) and _is_int(y)):
        _fail(path, "must be a [x, y] position of integers")


def _validate_tile(tile: Any, path: str) -> None:
    if tile is None:
        return
    if isinstance(tile, str):
        if tile == "LOCKED":
            return
        _fail(path, f"unexpected tile string {tile!r}; expected 'LOCKED'")
    if not isinstance(tile, dict):
        _fail(path, "must be null, 'LOCKED', or a structure object")

    kind = tile.get("kind")
    if kind == "WEED":
        return
    if kind == "PLANT":
        _validate_plant(tile, path)
        return
    if kind in ("COOP", "PASTURE"):
        _validate_animal_structure(tile, path)
        return
    _fail(path, f"unknown tile kind {kind!r}")


def _validate_plant(tile: Mapping[str, Any], path: str) -> None:
    crop = tile.get("crop")
    if not isinstance(crop, str):
        _fail(f"{path}.crop", "must be a string crop name")
    for key in (
        "planted_day",
        "consecutive_unwatered",
        "yield_units",
        "max_lifespan_step",
        "fertilized_until_day",
    ):
        value = tile.get(key)
        if not _is_int(value):
            _fail(f"{path}.{key}", "must be an integer")
    if not isinstance(tile.get("watered_today"), bool):
        _fail(f"{path}.watered_today", "must be a boolean")


def _validate_animal_structure(tile: Mapping[str, Any], path: str) -> None:
    animal = tile.get("animal")
    if animal is not None and not isinstance(animal, str):
        _fail(f"{path}.animal", "must be a string animal name or null")
    # Empty structures (no animal) carry no animal fields.
    if animal is None:
        return
    for key in ("placed_day", "yield_units", "consecutive_unfed", "pending_care_bonus"):
        value = tile.get(key)
        if not _is_int(value):
            _fail(f"{path}.{key}", "must be an integer")
    for key in ("fed_today", "cared_today", "fertilizer_available"):
        if not isinstance(tile.get(key), bool):
            _fail(f"{path}.{key}", "must be a boolean")


def _validate_private(private: Mapping[str, Any], path: str) -> None:
    shed = _require(private, "shed", dict, f"{path}.shed")
    _validate_count_map(shed, f"{path}.shed")
    seeds = _require(private, "seeds", dict, f"{path}.seeds")
    _validate_count_map(seeds, f"{path}.seeds")
    inventories = _require(private, "inventories", list, f"{path}.inventories")
    for index, inventory in enumerate(inventories):
        if not isinstance(inventory, dict):
            _fail(f"{path}.inventories[{index}]", "must be an object")
        _validate_count_map(inventory, f"{path}.inventories[{index}]")


def _validate_count_map(mapping: Mapping[str, Any], path: str) -> None:
    for key, value in mapping.items():
        if not isinstance(key, str):
            _fail(path, "keys must be strings")
        if not _is_int(value):
            _fail(f"{path}.{key}", "must be an integer count")


def _validate_market(market: Mapping[str, Any], path: str) -> None:
    inventory = _require(market, "inventory", dict, f"{path}.inventory")
    _validate_count_map(inventory, f"{path}.inventory")
    prices = _require(market, "prices", dict, f"{path}.prices")
    for key, value in prices.items():
        if not _is_int(value):
            _fail(f"{path}.prices.{key}", "must be an integer price")


def _validate_town(town: Mapping[str, Any], path: str) -> None:
    shops = _require(town, "unlocked_shops", list, f"{path}.unlocked_shops")
    for index, label in enumerate(shops):
        if not isinstance(label, str):
            _fail(f"{path}.unlocked_shops[{index}]", "must be a string")
