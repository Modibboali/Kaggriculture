"""Horizon-aware evaluation: realizability, terminal, and double counting."""

from __future__ import annotations

from dataclasses import replace

from agent.ai import (
    EvaluationConfig,
    Evaluator,
    HorizonAwareEvaluator,
    SearchState,
    horizon_days,
    horizon_remaining,
)
from agent.simulator import GameConfig
from agent.state import (
    AnimalState,
    AnimalType,
    CoopTile,
    CropType,
    Inventory,
    ItemType,
    PlantState,
    PlantTile,
    Position,
    Worker,
)

from ._build import make_config, make_search_state

POS = Position(1, 1)


def _horizon(config: GameConfig) -> HorizonAwareEvaluator:
    return HorizonAwareEvaluator(config)


def _classic(config: GameConfig) -> Evaluator:
    return Evaluator(config)


def _config() -> GameConfig:
    return make_config()


def _crop_state(
    *,
    day: int,
    step: int,
    money: int = 1000,
    crop: CropType = CropType.WHEAT,
    planted_day: int = 0,
    yield_units: int = 2,
) -> SearchState:
    state = make_search_state(money=money, day=day, step=step)
    plant = PlantState(
        crop=crop,
        planted_day=planted_day,
        watered_today=False,
        consecutive_unwatered=0,
        yield_units=yield_units,
        fertilized_until_day=-1,
        max_lifespan_step=720,
    )
    farm = state.game.players[0].farm.replace_tile(POS, PlantTile(plant))
    player = replace(state.game.players[0], farm=farm)
    return replace(state, game=replace(state.game, players=(player, state.game.players[1])))


def _inventory_state(*, step: int, items: dict[ItemType, int], money: int = 1000, day: int = 5) -> SearchState:
    state = make_search_state(money=money, day=day, step=step)
    player = replace(state.game.players[0], inventory=Inventory(items))
    return replace(state, game=replace(state.game, players=(player, state.game.players[1])))


def _animal_state(
    *,
    day: int,
    step: int,
    animal: AnimalType = AnimalType.GOOSE,
    placed_day: int = 0,
    money: int = 1000,
) -> SearchState:
    state = make_search_state(money=money, day=day, step=step)
    animal_state = AnimalState(
        animal=animal,
        placed_day=placed_day,
        yield_units=0,
        fed_today=False,
        consecutive_unfed=0,
        cared_today=False,
        fertilizer_available=False,
        pending_care_bonus=0,
    )
    farm = state.game.players[0].farm.replace_tile(POS, CoopTile(animal_state))
    player = replace(state.game.players[0], farm=farm)
    return replace(state, game=replace(state.game, players=(player, state.game.players[1])))


# -- horizon abstraction ---------------------------------------------------


def test_horizon_remaining_boundaries() -> None:
    config = _config()  # episode_steps=720
    state = make_search_state(step=0)
    assert horizon_remaining(state, 720) == 720
    assert horizon_remaining(state, 720) == 720
    mid = make_search_state(step=120)
    assert horizon_remaining(mid, 720) == 600
    last = make_search_state(step=719)
    assert horizon_remaining(last, 720) == 1
    term = make_search_state(step=720)
    assert horizon_remaining(term, 720) == 0
    over = make_search_state(step=800)
    assert horizon_remaining(over, 720) == 0  # clamped


def test_horizon_days() -> None:
    config = _config()  # turns_per_day=24
    assert horizon_days(make_search_state(step=0), config) == 30
    assert horizon_days(make_search_state(step=719), config) == 0


# -- terminal valuation ----------------------------------------------------


def test_terminal_value_is_cash_only() -> None:
    config = _config()
    horizon = _horizon(config)
    classic = _classic(config)

    # Terminal state (step == episode_steps) holding a mature crop + inventory.
    state = _inventory_state(step=720, items={ItemType.WHEAT: 4}, day=5)
    plant = PlantState(
        crop=CropType.WHEAT, planted_day=0, watered_today=False,
        consecutive_unwatered=0, yield_units=2, fertilized_until_day=-1, max_lifespan_step=720,
    )
    farm = state.game.players[0].farm.replace_tile(POS, PlantTile(plant))
    player = replace(state.game.players[0], farm=farm)
    state = replace(state, game=replace(state.game, players=(player, state.game.players[1])))

    # Horizon-aware: at terminal only cash is realizable (matches the reward).
    assert horizon.evaluate(state, 0) == 1000.0
    # Classic: still credits the inventory and crop (horizon-blind).
    assert classic.evaluate(state, 0) > 1000.0


def test_terminal_with_structures_and_animals_is_cash_only() -> None:
    config = _config()
    horizon = _horizon(config)
    state = _animal_state(day=5, step=720)
    assert horizon.evaluate(state, 0) == 1000.0


# -- crop realizability ----------------------------------------------------


def test_mature_crop_large_horizon() -> None:
    config = _config()
    horizon = _horizon(config)
    state = _crop_state(day=5, step=0, crop=CropType.WHEAT, planted_day=0)  # mature
    base = make_search_state(money=1000, day=5, step=0)
    # WHEAT base price 25, yield 2, weight 0.6 -> 30 at full realizability.
    assert horizon.evaluate(state, 0) - horizon.evaluate(base, 0) == 30.0


def test_immature_crop_enough_time() -> None:
    import pytest

    config = _config()
    horizon = _horizon(config)
    classic = _classic(config)
    # TOMATO matures at day 8; planted day 0, age 2 at day 2, 30 days remain.
    state = _crop_state(day=2, step=0, crop=CropType.TOMATO, planted_day=0)
    base = make_search_state(money=1000, day=2, step=0)
    # Plenty of time -> full feasibility, discounted for 6 days to maturity.
    h_value = horizon.evaluate(state, 0) - horizon.evaluate(base, 0)
    c_value = classic.evaluate(state, 0) - classic.evaluate(base, 0)
    assert h_value == pytest.approx(2 * 60 * 0.6 * (0.85 ** 6))
    assert h_value > 0.0  # never deleted
    # Time discount: a crop 6 days from maturity is worth less than its face
    # value (and less than the classic flat 0.5 credit) because cash now is
    # preferred over distant crop potential.
    assert h_value < c_value


def test_immature_crop_insufficient_time() -> None:
    config = _config()
    horizon = _horizon(config)
    classic = _classic(config)
    # TOMATO needs 6 more days to mature; only 0 days remain.
    state = _crop_state(day=2, step=719, crop=CropType.TOMATO, planted_day=0)
    base = make_search_state(money=1000, day=2, step=719)
    h_value = horizon.evaluate(state, 0) - horizon.evaluate(base, 0)
    c_value = classic.evaluate(state, 0) - classic.evaluate(base, 0)
    assert h_value == 0.0  # cannot mature before terminal
    assert c_value > 0.0  # classic still overvalues it


def test_immature_crop_partial_credit() -> None:
    import pytest

    config = _config()
    horizon = _horizon(config)
    # TOMATO: age 2, needs 6 more days + 1 harvest = 7; only 3 days remain.
    state = _crop_state(day=2, step=720 - 24 * 3, crop=CropType.TOMATO, planted_day=0)
    base = make_search_state(money=1000, day=2, step=720 - 24 * 3)
    value = horizon.evaluate(state, 0) - horizon.evaluate(base, 0)
    # feasibility (3/7) x time discount over the 6 remaining growth days.
    assert value == pytest.approx(2 * 60 * 0.6 * (3 / 7) * (0.85 ** 6))


# -- seeds -----------------------------------------------------------------


def test_seed_realizability() -> None:
    import pytest

    from agent.state import Seeds

    config = _config()
    horizon = _horizon(config)

    def seed_value(step: int) -> float:
        state = make_search_state(money=1000, day=2, step=step)
        player = replace(state.game.players[0], seeds=Seeds({CropType.WHEAT: 10}))
        state = replace(state, game=replace(state.game, players=(player, state.game.players[1])))
        return horizon.evaluate(state, 0) - horizon.evaluate(make_search_state(money=1000, day=2, step=step), 0)

    # WHEAT seed cost 10, weight 0.5 -> 10 seeds worth 50 at full feasibility,
    # time-discounted for the 2 growth days: 50 * 0.85**2.
    assert seed_value(step=0) == pytest.approx(50.0 * (0.85 ** 2))
    # 1 day left: needs 3 (grow 2 + 1); 1/3 feasibility, same discount.
    assert seed_value(step=720 - 24) == pytest.approx(50.0 * (1 / 3) * (0.85 ** 2))
    # Terminal: 0.
    assert seed_value(step=720) == pytest.approx(0.0)


# -- animals ---------------------------------------------------------------


def test_animal_realizability() -> None:
    import pytest

    config = _config()
    horizon = _horizon(config)

    def animal_value(step: int, day: int = 0) -> float:
        # Base already contains an empty coop so the delta isolates the animal.
        state = _animal_state(day=day, step=step)
        base_state = make_search_state(money=1000, day=day, step=step)
        farm = base_state.game.players[0].farm.replace_tile(POS, CoopTile(None))
        player = replace(base_state.game.players[0], farm=farm)
        base = replace(base_state, game=replace(base_state.game, players=(player, base_state.game.players[1])))
        return horizon.evaluate(state, 0) - horizon.evaluate(base, 0)

    # GOOSE cost 300, weight 0.5 -> 150 at full realizability (needs 4+1 days).
    assert animal_value(step=0) == pytest.approx(150.0)
    # 1 day left: 1/5 realizable.
    assert animal_value(step=720 - 24) == pytest.approx(150.0 * (1 / 5))
    # Terminal: 0.
    assert animal_value(step=720) == pytest.approx(0.0)


# -- workers / structures --------------------------------------------------


def test_worker_value_scales_with_horizon() -> None:
    config = _config()
    horizon = _horizon(config)
    state = make_search_state(money=1000, day=0, step=0)
    hired = Worker(1, Position(1, 1), Inventory.empty(), False)
    farm = replace(state.game.players[0].farm, workers=(hired,))
    player = replace(state.game.players[0], farm=farm)
    with_worker = replace(state, game=replace(state.game, players=(player, state.game.players[1])))

    base = make_search_state(money=1000, day=0, step=0)
    # Long horizon: full worker value (15).
    assert horizon.evaluate(with_worker, 0) - horizon.evaluate(base, 0) == 15.0
    # 1 day left: 1/10 of the 10-day window.
    near = replace(with_worker, game=replace(with_worker.game, step=720 - 24))
    near_base = make_search_state(money=1000, day=0, step=720 - 24)
    assert horizon.evaluate(near, 0) - horizon.evaluate(near_base, 0) == 15.0 * 0.1
    # Terminal: 0.
    term = replace(with_worker, game=replace(with_worker.game, step=720))
    term_base = make_search_state(money=1000, day=0, step=720)
    assert horizon.evaluate(term, 0) - horizon.evaluate(term_base, 0) == 0.0


def test_structure_value_scales_with_horizon() -> None:
    config = _config()
    horizon = _horizon(config)
    state = make_search_state(money=1000, day=0, step=0)
    farm = state.game.players[0].farm.replace_tile(POS, CoopTile(None))
    player = replace(state.game.players[0], farm=farm)
    with_coop = replace(state, game=replace(state.game, players=(player, state.game.players[1])))

    base = make_search_state(money=1000, day=0, step=0)
    assert horizon.evaluate(with_coop, 0) - horizon.evaluate(base, 0) == 10.0  # full window
    near = replace(with_coop, game=replace(with_coop.game, step=720 - 24))
    near_base = make_search_state(money=1000, day=0, step=720 - 24)
    assert horizon.evaluate(near, 0) - horizon.evaluate(near_base, 0) == 1.0  # 1/10


# -- inventory -------------------------------------------------------------


def test_inventory_liquid_at_market_price() -> None:
    config = _config()
    horizon = _horizon(config)
    state = _inventory_state(step=100, items={ItemType.WHEAT: 4})
    base = make_search_state(money=1000, day=5, step=100)
    # WHEAT base price 25, weight 1.0 -> 4 units worth 100, realizable now.
    assert horizon.evaluate(state, 0) - horizon.evaluate(base, 0) == 100.0


def test_inventory_liquidity_discounts_large_holdings_near_terminal() -> None:
    import pytest

    config = _config()
    horizon = _horizon(config)
    # 50 wheat with only 1 step left: at most 10 units can be sold (10 orders/turn).
    state = _inventory_state(step=719, items={ItemType.WHEAT: 50})
    base = make_search_state(money=1000, day=5, step=719)
    value = horizon.evaluate(state, 0) - horizon.evaluate(base, 0)
    # 50 x 25 x (10/50) = 250, not the face value 1250.
    assert value == pytest.approx(50 * 25 * (10 / 50))
    # With plenty of steps, the same holding is fully liquid.
    far = _inventory_state(step=100, items={ItemType.WHEAT: 50})
    far_base = make_search_state(money=1000, day=5, step=100)
    assert horizon.evaluate(far, 0) - horizon.evaluate(far_base, 0) == pytest.approx(50 * 25)


# -- double counting -------------------------------------------------------


def test_no_double_counting_inventory() -> None:
    config = _config()
    horizon = _horizon(config)
    # 4 wheat in the shed: valued exactly once at market price (4 x 25 = 100).
    state = _inventory_state(step=100, items={ItemType.WHEAT: 4})
    base = make_search_state(money=1000, day=5, step=100)
    assert horizon.evaluate(state, 0) - horizon.evaluate(base, 0) == 100.0


def test_no_double_counting_crop_and_inventory() -> None:
    config = _config()
    horizon = _horizon(config)
    # A mature crop in the ground AND separate wheat in the shed are distinct
    # assets; each is counted once. Crop 2x25x0.6 = 30, inventory 4x25 = 100.
    state = _crop_state(day=5, step=100, crop=CropType.WHEAT, planted_day=0)
    base = make_search_state(money=1000, day=5, step=100)
    crop_only = horizon.evaluate(state, 0) - horizon.evaluate(base, 0)
    assert crop_only == 30.0
    both = _inventory_state(step=100, items={ItemType.WHEAT: 4})
    # The crop and the inventory do not overlap: adding both is additive.
    assert horizon.evaluate(both, 0) - horizon.evaluate(base, 0) == 100.0


def test_mature_crop_preferred_over_immature_crop_at_short_horizon() -> None:
    """Conceptual test: a harvestable crop is worth more than a crop that
    still needs days to mature, especially when the horizon is short."""
    config = _config()
    horizon = _horizon(config)

    def crop_value(step: int, day: int, crop: CropType, planted_day: int) -> float:
        state = _crop_state(day=day, step=step, crop=crop, planted_day=planted_day)
        base = make_search_state(money=1000, day=day, step=step)
        return horizon.evaluate(state, 0) - horizon.evaluate(base, 0)

    # Mature WHEAT (age 5 >= first_yield 2) vs a WHEAT planted today (needs 2 days).
    mature = crop_value(step=0, day=5, crop=CropType.WHEAT, planted_day=0)
    immature = crop_value(step=0, day=5, crop=CropType.WHEAT, planted_day=5)
    assert mature == 2 * 25 * 0.6
    assert mature > immature  # mature is strictly preferred
    # With only 1 day left the immature crop's value collapses (feasibility
    # 1/3 x discount^2) to a small fraction of the mature crop's value.
    immature_tight = crop_value(step=720 - 24, day=5, crop=CropType.WHEAT, planted_day=5)
    assert immature_tight < 0.5 * mature


# -- determinism + classic parity at long horizon --------------------------

def test_horizon_evaluator_is_deterministic() -> None:
    config = _config()
    horizon = _horizon(config)
    state = _crop_state(day=5, step=100)
    assert horizon.evaluate(state, 0) == horizon.evaluate(state, 0)


def test_horizon_matches_classic_at_full_horizon_for_assets() -> None:
    config = _config()
    horizon = _horizon(config)
    classic = _classic(config)
    # With a full 30-day horizon, mature assets realise at the same value.
    state = _crop_state(day=5, step=0, crop=CropType.WHEAT, planted_day=0)
    base = make_search_state(money=1000, day=5, step=0)
    assert horizon.evaluate(state, 0) - horizon.evaluate(base, 0) == (
        classic.evaluate(state, 0) - classic.evaluate(base, 0)
    )


def test_time_discount_zero_for_mature() -> None:
    """A mature, harvestable crop is not time-discounted."""
    import pytest

    config = _config()
    horizon = _horizon(config)
    state = _crop_state(day=5, step=100, crop=CropType.WHEAT, planted_day=0)
    base = make_search_state(money=1000, day=5, step=100)
    assert horizon.evaluate(state, 0) - horizon.evaluate(base, 0) == pytest.approx(2 * 25 * 0.6)


# -- ablation switches -----------------------------------------------------

def test_ablation_switches_fall_back_to_classic() -> None:
    config = _config()
    # Without crop realizability, an immature crop uses the classic discount.
    eval_config = EvaluationConfig(crop_realizability=False)
    horizon = HorizonAwareEvaluator(config, eval_config)
    classic = _classic(config)
    state = _crop_state(day=2, step=719, crop=CropType.TOMATO, planted_day=0)
    base = make_search_state(money=1000, day=2, step=719)
    assert horizon.evaluate(state, 0) - horizon.evaluate(base, 0) == (
        classic.evaluate(state, 0) - classic.evaluate(base, 0)
    )
