"""Phase model, action priority, realizability, and CashConversion rollout."""

from __future__ import annotations

from dataclasses import replace

from agent.ai import (
    ActionPriorityModel,
    CashConversionRolloutPolicy,
    GamePhase,
    SearchState,
    phase_for,
)
from agent.ai.action_generator import ActionGenerator
from agent.actions import (
    BuyLandAction,
    BuySeedAction,
    BuildCoopAction,
    HarvestAction,
    SellAction,
    TurnAction,
)
from agent.state import (
    CropType,
    Inventory,
    ItemType,
    PlantState,
    PlantTile,
    Position,
    Quadrant,
)

from ._build import make_config, make_search_state

POS = Position(1, 1)


def _model() -> ActionPriorityModel:
    return ActionPriorityModel(make_config())


# -- phase detection --------------------------------------------------------


def test_phase_for_large_horizon() -> None:
    config = make_config()
    assert phase_for(720, 30, config) == GamePhase.DEVELOPMENT


def test_phase_for_medium_horizon() -> None:
    config = make_config()
    assert phase_for(720, 10, config) == GamePhase.PRODUCTION
    assert phase_for(720, 5, config) == GamePhase.PRODUCTION


def test_phase_for_short_horizon() -> None:
    config = make_config()
    assert phase_for(720, 3, config) == GamePhase.CASH_CONVERSION


def test_phase_for_terminal() -> None:
    config = make_config()
    assert phase_for(0, 0, config) == GamePhase.TERMINAL


def test_phase_of_state() -> None:
    model = _model()
    assert model.phase(make_search_state(step=0)).value == "development"
    assert model.phase(make_search_state(step=720 - 120)).value == "production"  # 5 days
    assert model.phase(make_search_state(step=720 - 24)).value == "cash_conversion"  # 1 day
    assert model.phase(make_search_state(step=720)).value == "terminal"


# -- realizability ----------------------------------------------------------


def test_buy_seed_realizability_enough_time() -> None:
    model = _model()
    state = make_search_state(step=720 - 120)  # 5 days remain
    action = TurnAction(market_actions=(BuySeedAction(crop=CropType.WHEAT, quantity=1),))
    assert model.can_realize(action, state)  # WHEAT needs 2+2 = 4 days


def test_buy_seed_realizability_exactly_enough() -> None:
    model = _model()
    state = make_search_state(step=720 - 24 * 4)  # exactly 4 days remain
    action = TurnAction(market_actions=(BuySeedAction(crop=CropType.WHEAT, quantity=1),))
    assert model.can_realize(action, state)


def test_buy_seed_realizability_insufficient_time() -> None:
    model = _model()
    state = make_search_state(step=720 - 24 * 3)  # 3 days remain
    action = TurnAction(market_actions=(BuySeedAction(crop=CropType.WHEAT, quantity=1),))
    assert not model.can_realize(action, state)


def test_buy_land_realizability_requires_long_horizon() -> None:
    model = _model()
    action = TurnAction(market_actions=(BuyLandAction(quadrant=Quadrant.NE),))
    assert model.can_realize(action, make_search_state(step=0))  # 30 days
    assert not model.can_realize(action, make_search_state(step=720 - 120))  # 5 days


# -- action priority --------------------------------------------------------


def _harvestable_crop_state(step: int) -> SearchState:
    state = make_search_state(money=2000, day=5, step=step)
    plant = PlantState(
        crop=CropType.WHEAT, planted_day=0, watered_today=False,
        consecutive_unwatered=0, yield_units=3, fertilized_until_day=-1, max_lifespan_step=720,
    )
    farm = state.game.players[0].farm.replace_tile(POS, PlantTile(plant))
    player = replace(state.game.players[0], farm=farm)
    return replace(state, game=replace(state.game, players=(player, state.game.players[1])))


def test_harvest_preferred_over_investment_near_terminal() -> None:
    model = _model()
    state = _harvestable_crop_state(step=720 - 24)  # 1 day left
    harvest = TurnAction(farmer_action=HarvestAction())
    build = TurnAction(farmer_action=BuildCoopAction())
    assert model.priority(state, harvest) > model.priority(state, build)


def test_sell_preferred_over_investment_near_terminal() -> None:
    model = _model()
    state = make_search_state(money=2000, day=5, step=720 - 24)  # 1 day left
    player = replace(state.game.players[0], inventory=Inventory({ItemType.WHEAT: 5}))
    state = replace(state, game=replace(state.game, players=(player, state.game.players[1])))
    sell = TurnAction(market_actions=(SellAction(item=ItemType.WHEAT, quantity=1),))
    buy_land = TurnAction(market_actions=(BuyLandAction(quadrant=Quadrant.NE),))
    assert model.priority(state, sell) > model.priority(state, buy_land)


def test_viable_seed_purchase_outranks_non_viable() -> None:
    model = _model()
    action = TurnAction(market_actions=(BuySeedAction(crop=CropType.WHEAT, quantity=1),))
    feasible = model.priority(make_search_state(step=720 - 120), action)  # 5 days
    infeasible = model.priority(make_search_state(step=720 - 24 * 3), action)  # 3 days
    assert feasible > infeasible


def test_rank_orders_best_last_for_mcts_pop() -> None:
    model = _model()
    state = make_search_state(money=2000, day=5, step=720 - 24)
    player = replace(state.game.players[0], inventory=Inventory({ItemType.WHEAT: 5}))
    state = replace(state, game=replace(state.game, players=(player, state.game.players[1])))
    actions = [
        TurnAction(market_actions=(BuyLandAction(quadrant=Quadrant.NE),)),
        TurnAction(market_actions=(SellAction(item=ItemType.WHEAT, quantity=1),)),
        TurnAction(),
    ]
    ranked = model.rank(state, actions)
    # The sell action must be ranked last (so MCTS pops it first).
    assert ranked[-1].market_actions and ranked[-1].market_actions[0].action_type.value == "SELL"


# -- CashConversion rollout -------------------------------------------------


def _rollout() -> CashConversionRolloutPolicy:
    config = make_config()
    model = ActionPriorityModel(config)
    generator = ActionGenerator(config, priority_model=model)
    return CashConversionRolloutPolicy(generator, model)


def test_rollout_is_deterministic() -> None:
    import random

    r1 = _rollout()
    r2 = _rollout()
    state = _harvestable_crop_state(step=720 - 120)
    a = r1.choose(state, random.Random(7))
    b = r2.choose(state, random.Random(7))
    assert a == b


def test_rollout_returns_legal_generated_action() -> None:
    import random

    rollout = _rollout()
    state = _harvestable_crop_state(step=720 - 120)
    action = rollout.choose(state, random.Random(1))
    generated = set(_rollout_generated(state))
    assert action in generated


def _rollout_generated(state: SearchState) -> list[TurnAction]:
    config = make_config()
    model = ActionPriorityModel(config)
    generator = ActionGenerator(config, priority_model=model)
    return list(generator.generate(state))


def test_rollout_does_not_mutate_state() -> None:
    import random

    rollout = _rollout()
    state = _harvestable_crop_state(step=720 - 120)
    before = state.state_key()
    rollout.choose(state, random.Random(2))
    assert state.state_key() == before


def test_rollout_follows_cash_conversion_chain() -> None:
    """With a harvestable crop the rollout chooses HARVEST (chain step)."""
    import random

    rollout = _rollout()
    state = _harvestable_crop_state(step=720 - 120)
    action = rollout.choose(state, random.Random(3))
    assert action.farmer_action.action_type.value == "HARVEST"


def test_rollout_follows_sell_chain() -> None:
    """With sellable inventory and no harvestable tile, the rollout sells."""
    import random

    rollout = _rollout()
    state = make_search_state(money=2000, day=5, step=720 - 24)
    player = replace(state.game.players[0], inventory=Inventory({ItemType.WHEAT: 5}))
    state = replace(state, game=replace(state.game, players=(player, state.game.players[1])))
    action = rollout.choose(state, random.Random(4))
    assert action.market_actions and action.market_actions[0].action_type.value == "SELL"
