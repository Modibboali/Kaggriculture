"""ActionGenerator: valid actions, no crashes, determinism."""

from __future__ import annotations

from dataclasses import replace

from agent.ai import ActionGenerator
from agent.actions import TurnAction
from agent.state import CoopTile, CropType, ItemType, PlantState, PlantTile, Position

from ._build import make_components, make_search_state


def _generator(board_size: int = 4) -> ActionGenerator:
    config, *_ = make_components(board_size)
    return ActionGenerator(config)


def test_empty_start_state_candidates() -> None:
    gen = _generator()
    actions = gen.generate(make_search_state())
    # Always includes PASS plus the 4 in-bounds moves plus market orders.
    assert TurnAction() in actions
    assert len(actions) > 5
    # All are valid TurnActions.
    assert all(isinstance(a, TurnAction) for a in actions)


def test_generation_is_deterministic() -> None:
    gen = _generator()
    state = make_search_state()
    assert gen.generate(state) == gen.generate(state)


def test_crop_state_generates_water_and_harvest() -> None:
    gen = _generator()
    state = make_search_state(day=2, hour=0, step=48)
    plant = PlantState(CropType.WHEAT, 0, False, 0, 2, -1, 120)
    game = replace(state.game.players[0].farm.replace_tile(Position(1, 1), PlantTile(plant)))
    state = replace(state, game=replace(state.game, players=(replace(state.game.players[0], farm=game), state.game.players[1])))
    from agent.actions import HarvestAction, WaterAction

    action_types = {a.farmer_action.action_type for a in gen.generate(state)}
    assert HarvestAction().action_type in action_types
    assert WaterAction().action_type in action_types


def test_structure_state_generates_place() -> None:
    gen = _generator()
    state = make_search_state()
    from agent.state import AnimalState, AnimalType

    tile = CoopTile(AnimalState(AnimalType.GOOSE, 0, 0, False, 0, False, True, 0))
    game = replace(state.game.players[0].farm.replace_tile(Position(1, 1), tile))
    player = replace(state.game.players[0], farm=game)
    state = replace(state, game=replace(state.game, players=(player, state.game.players[1])))
    from agent.actions import CollectFertilizerAction, HarvestAction

    action_types = {a.farmer_action.action_type for a in gen.generate(state)}
    assert CollectFertilizerAction().action_type in action_types


def test_generated_actions_do_not_crash_simulator() -> None:
    config, adapter, generator, evaluator, terminal = make_components()
    state = make_search_state()
    # Plant a crop and run a few turns, applying every generated action.
    for _ in range(6):
        for action in generator.generate(state):
            nxt = adapter.transition(state, action)
            assert nxt.game.step >= state.game.step
        state = adapter.transition(state, generator.generate(state)[0])


def test_no_crash_on_various_states() -> None:
    gen = _generator()
    # Money-rich and poor states, late day, etc.
    for kw in ({"money": 0}, {"money": 5000, "day": 3, "step": 75}, {"day": 10, "step": 240}):
        actions = gen.generate(make_search_state(**kw))
        assert all(isinstance(a, TurnAction) for a in actions)
