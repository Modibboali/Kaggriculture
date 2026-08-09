"""Serialize domain actions into the Kaggle Kaggriculture action format.

The official environment expects, per step (verified against
``kaggle_environments/envs/kaggriculture/kaggriculture.py``):

    {"farmer": [<token>, ...], "hands": [[<token>, ...], ...],
     "market": [[<token>, <arg>, ...], ...]}

``farmer`` is the main farmer's action as a token list (e.g. ``["PLANT",
"WHEAT"]``), ``hands`` lists one token list per hired hand, and ``market``
lists ordered orders. This module is the reverse half of the observation
adapter: it converts our typed actions into the raw environment format so the
differential runner can feed the same action sequence to both the official
engine and our simulator.
"""

from __future__ import annotations

from typing import Any

from ..actions import (
    Action,
    BuyAnimalAction,
    BuyLandAction,
    BuyProductAction,
    BuySeedAction,
    HireAction,
    MovementAction,
    PickupAction,
    PlaceAction,
    PlantAction,
    SellAction,
    TurnAction,
)


def to_kaggle_action(turn: TurnAction) -> dict[str, Any]:
    """Convert a domain ``TurnAction`` into the Kaggle action dict."""
    return {
        "farmer": _farmer_tokens(turn.farmer_action),
        "hands": [_farmer_tokens(action) for action in turn.worker_actions],
        "market": [_market_order(action) for action in turn.market_actions],
    }


def _farmer_tokens(action: Action) -> list[Any]:
    """Token list for one unit action, e.g. ``["PLANT", "WHEAT"]``."""
    if isinstance(action, MovementAction):
        return [action.direction.label]
    if isinstance(action, PlantAction):
        return ["PLANT", action.crop.label]
    if isinstance(action, PickupAction):
        return ["PICKUP", action.item.label, action.quantity]
    if isinstance(action, PlaceAction):
        return ["PLACE", action.animal.label]
    # Parameterless actions (PASS, WATER, HARVEST, FERTILIZE, DIG,
    # BUILD_COOP, BUILD_PASTURE, FEED, CARE, COLLECT_FERTILIZER, DROP).
    return [action.action_type.label]


def _market_order(action: Action) -> list[Any]:
    command = action.action_type.label
    if isinstance(action, BuySeedAction):
        return [command, action.crop.label, action.quantity]
    if isinstance(action, BuyAnimalAction):
        return [command, action.animal.label, action.quantity]
    if isinstance(action, BuyProductAction):
        return [command, action.item.label, action.quantity]
    if isinstance(action, SellAction):
        return [command, action.item.label, action.quantity]
    if isinstance(action, HireAction):
        return [command, action.quantity]
    if isinstance(action, BuyLandAction):
        return [command, action.quadrant.label]
    raise ValueError(f"unsupported market action: {action!r}")
