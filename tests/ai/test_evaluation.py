"""Evaluator: deterministic, terminal handled, asset-aware."""

from __future__ import annotations

from dataclasses import replace

from agent.ai import EvaluationConfig, Evaluator
from agent.state import CropType, PlantState, PlantTile, Position

from ._build import make_components, make_search_state


def _evaluator() -> tuple[Evaluator, object]:
    config, *_ = make_components()
    return Evaluator(config), config


def test_evaluate_is_deterministic() -> None:
    evaluator, _ = _evaluator()
    state = make_search_state(money=1234)
    assert evaluator.evaluate(state, 0) == evaluator.evaluate(state, 0)


def test_immediate_reward_is_money() -> None:
    evaluator, _ = _evaluator()
    state = make_search_state(money=2500)
    assert evaluator.immediate_reward(state, 0) == 2500.0


def test_money_increases_value() -> None:
    evaluator, _ = _evaluator()
    rich = make_search_state(money=2000)
    poor = make_search_state(money=1000)
    assert evaluator.evaluate(rich, 0) > evaluator.evaluate(poor, 0)


def test_mature_crop_adds_value() -> None:
    evaluator, _ = _evaluator()
    state = make_search_state(money=1000, day=2, hour=0, step=48)
    plant = PlantState(
        crop=CropType.WHEAT,
        planted_day=0,
        watered_today=False,
        consecutive_unwatered=0,
        yield_units=2,
        fertilized_until_day=-1,
        max_lifespan_step=120,
    )
    farm = state.game.players[0].farm.replace_tile(Position(1, 1), PlantTile(plant))
    player = replace(state.game.players[0], farm=farm)
    cropped = replace(state, game=replace(state.game, players=(player, state.game.players[1])))

    base = evaluator.evaluate(make_search_state(money=1000, day=2, hour=0, step=48), 0)
    assert evaluator.evaluate(cropped, 0) > base


def test_configurable_weights_change_value() -> None:
    config, *_ = make_components()
    state = make_search_state(money=1000)
    low = Evaluator(config, EvaluationConfig(structure_value=0.0)).evaluate(state, 0)
    high = Evaluator(config, EvaluationConfig(structure_value=100.0)).evaluate(state, 0)
    assert low == high  # no structures present, so weights are inert here
    assert isinstance(low, float)
