"""Unit tests for scenario construction and the Kaggle action serializer."""

from __future__ import annotations

from agent.actions import ActionType, BuySeedAction, TurnAction
from agent.environment import to_kaggle_action
from agent.state import CropType, Direction, ItemType
from agent.testing.differential import (
    SCENARIOS,
    Scenario,
    all_scenarios,
)


def test_all_scenarios_are_listed() -> None:
    scenarios = all_scenarios()
    assert len(scenarios) == len(SCENARIOS)
    names = {scenario.name for scenario in scenarios}
    assert names == {
        "pass",
        "buy_seed",
        "move_farmer",
        "plant_crop",
        "water_crop",
        "multiple_turns",
        "buy_plant_water",
        "two_player_interaction",
        "market_transaction",
    }


def test_scenario_actions_are_turn_pairs() -> None:
    for scenario in all_scenarios():
        assert scenario.actions
        for player0, player1 in scenario.actions:
            assert isinstance(player0, TurnAction)
            assert isinstance(player1, TurnAction)


def test_single_player_wraps_with_opponent_pass() -> None:
    move = TurnAction(farmer_action=__import__("agent.actions", fromlist=["MovementAction"]).MovementAction(direction=Direction.NORTH))
    scenario = Scenario.single_player("moves", (move, move))
    assert len(scenario) == 2
    for player0, player1 in scenario.actions:
        assert player0.farmer_action.action_type is ActionType.NORTH
        assert player1.farmer_action.action_type is ActionType.PASS


def test_two_player_scenario_has_distinct_actions() -> None:
    scenario = next(s for s in all_scenarios() if s.name == "two_player_interaction")
    p0, p1 = scenario.actions[0]
    p0_buy = p0.market_actions[0]
    p1_buy = p1.market_actions[0]
    assert isinstance(p0_buy, BuySeedAction)
    assert isinstance(p1_buy, BuySeedAction)
    assert p0_buy.crop is CropType.WHEAT
    assert p1_buy.crop is CropType.CARROT


def test_serialize_farmer_action() -> None:
    from agent.actions import PlantAction

    turn = TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT))
    assert to_kaggle_action(turn) == {
        "farmer": ["PLANT", "WHEAT"],
        "hands": [],
        "market": [],
    }


def test_serialize_market_order_keeps_int_quantity() -> None:
    turn = TurnAction(
        market_actions=(BuySeedAction(crop=CropType.WHEAT, quantity=1),)
    )
    serialized = to_kaggle_action(turn)
    assert serialized["farmer"] == ["PASS"]
    assert serialized["market"] == [["BUY_SEED", "WHEAT", 1]]
    assert isinstance(serialized["market"][0][2], int)


def test_serialize_multi_unit_and_pass() -> None:
    from agent.actions import MovementAction, WaterAction

    turn = TurnAction(
        farmer_action=MovementAction(direction=Direction.EAST),
        worker_actions=(WaterAction(),),
        market_actions=(BuySeedAction(crop=CropType.CARROT, quantity=5),),
    )
    assert to_kaggle_action(turn) == {
        "farmer": ["EAST"],
        "hands": [["WATER"]],
        "market": [["BUY_SEED", "CARROT", 5]],
    }


def test_serialize_sell_uses_item_label() -> None:
    from agent.actions import SellAction

    turn = TurnAction(market_actions=(SellAction(item=ItemType.WHEAT, quantity=4),))
    assert to_kaggle_action(turn) == {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["SELL", "WHEAT", 4]],
    }
