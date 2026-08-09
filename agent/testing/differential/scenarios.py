"""Deterministic differential-test scenarios.

Every scenario is an explicit, ordered list of per-turn action pairs
``(player0, player1)``. Scenarios start from the official environment's default
initial state (farmer at the shed, 3000 coins, only the NW quadrant unlocked) and
include any setup actions they require (e.g. buying a seed before planting), so
the same sequence is valid on the Kaggle side and (eventually) the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...actions import (
    BuyProductAction,
    BuySeedAction,
    MovementAction,
    PlantAction,
    SellAction,
    TurnAction,
    WaterAction,
)
from ...state import CropType, Direction, ItemType

# One step's actions for both players.
TurnActions = tuple[TurnAction, TurnAction]


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named sequence of per-turn actions for both players."""

    name: str
    actions: tuple[TurnActions, ...]
    description: str = ""

    @classmethod
    def single_player(
        cls,
        name: str,
        actions: tuple[TurnAction, ...],
        description: str = "",
    ) -> "Scenario":
        """Build a scenario where only player 0 acts; player 1 always passes."""
        opponent = TurnAction()
        return cls(
            name=name,
            actions=tuple((action, opponent) for action in actions),
            description=description,
        )

    def __len__(self) -> int:
        return len(self.actions)


# --- Reusable per-player turns ---------------------------------------------
PASS_TURN = TurnAction()
BUY_WHEAT = TurnAction(
    market_actions=(BuySeedAction(crop=CropType.WHEAT, quantity=1),)
)
BUY_CARROT = TurnAction(
    market_actions=(BuySeedAction(crop=CropType.CARROT, quantity=1),)
)
BUY_WHEAT_PRODUCT = TurnAction(
    market_actions=(BuyProductAction(item=ItemType.WHEAT, quantity=1),)
)
SELL_WHEAT = TurnAction(
    market_actions=(SellAction(item=ItemType.WHEAT, quantity=1),)
)
PLANT_WHEAT = TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT))
WATER = TurnAction(farmer_action=WaterAction())
MOVE_NORTH = TurnAction(farmer_action=MovementAction(direction=Direction.NORTH))

# --- Built-in scenarios ----------------------------------------------------
SCENARIO_PASS = Scenario.single_player(
    "pass",
    (PASS_TURN, PASS_TURN, PASS_TURN),
    "Three consecutive PASS turns for player 0.",
)

SCENARIO_BUY_SEED = Scenario.single_player(
    "buy_seed",
    (BUY_WHEAT,),
    "Player 0 buys one WHEAT seed on the first turn.",
)

SCENARIO_MOVE_FARMER = Scenario.single_player(
    "move_farmer",
    (MOVE_NORTH, MOVE_NORTH),
    "Player 0's farmer moves NORTH twice from the starting shed tile.",
)

SCENARIO_PLANT_CROP = Scenario.single_player(
    "plant_crop",
    (BUY_WHEAT, PLANT_WHEAT),
    "Player 0 buys a WHEAT seed, then plants it on its tile.",
)

SCENARIO_WATER_CROP = Scenario.single_player(
    "water_crop",
    (BUY_WHEAT, PLANT_WHEAT, WATER),
    "Player 0 buys, plants, then waters WHEAT.",
)

SCENARIO_MULTIPLE_TURNS = Scenario.single_player(
    "multiple_turns",
    (BUY_WHEAT, PLANT_WHEAT, WATER, MOVE_NORTH, MOVE_NORTH, PASS_TURN),
    "A longer deterministic sequence exercising several action families.",
)

SCENARIO_BUY_PLANT_WATER = Scenario.single_player(
    "buy_plant_water",
    (BUY_WHEAT, PLANT_WHEAT, WATER, BUY_CARROT),
    "Buy + plant + water sequence with a follow-up purchase.",
)

SCENARIO_TWO_PLAYER = Scenario(
    name="two_player_interaction",
    actions=((BUY_WHEAT, BUY_CARROT), (PASS_TURN, PASS_TURN)),
    description=(
        "Player 0 buys WHEAT while player 1 buys CARROT simultaneously, "
        "then both pass."
    ),
)

SCENARIO_MARKET_TRANSACTION = Scenario.single_player(
    "market_transaction",
    (BUY_WHEAT_PRODUCT, SELL_WHEAT),
    "Player 0 buys WHEAT from the market, then sells it back.",
)

SCENARIOS: tuple[Scenario, ...] = (
    SCENARIO_PASS,
    SCENARIO_BUY_SEED,
    SCENARIO_MOVE_FARMER,
    SCENARIO_PLANT_CROP,
    SCENARIO_WATER_CROP,
    SCENARIO_MULTIPLE_TURNS,
    SCENARIO_BUY_PLANT_WATER,
    SCENARIO_TWO_PLAYER,
    SCENARIO_MARKET_TRANSACTION,
)


def all_scenarios() -> tuple[Scenario, ...]:
    """Every built-in scenario."""
    return SCENARIOS
