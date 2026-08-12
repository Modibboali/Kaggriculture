"""Tests for root-parallel MCTS (:mod:`agent.ai.parallel_mcts`).

Covers: budget distribution, worker seeding, canonical action identity, root
statistics aggregation, sequential fallback (workers == 1), parallel
determinism, exact budget accounting, worker-failure surfacing, and
spawn-safe process execution.
"""

from __future__ import annotations

import random

import pytest

from agent.ai import (
    ActionGenerator,
    Evaluator,
    HeuristicRolloutPolicy,
    MCTS,
    MCTSConfig,
    ParallelMCTS,
    SearchState,
    SimulatorAdapter,
    Terminal,
    WorkerResult,
    WorkerTask,
    aggregate_root_stats,
    canonical_action_key,
    run_mcts_worker,
    select_best_action_from_stats,
    select_best_action_key,
    split_budget,
    worker_seed,
)
from agent.actions import (
    BuyAnimalAction,
    PlantAction,
    TurnAction,
    WaterAction,
)
from agent.ai.parallel_mcts import RootStat
from agent.simulator import GameConfig
from agent.state import AnimalType, CropType

from ._build import make_search_state, make_components


def _components(
    board_size: int = 4,
) -> tuple[
    GameConfig,
    SimulatorAdapter,
    ActionGenerator,
    Evaluator,
    Terminal,
    HeuristicRolloutPolicy,
]:
    """Shared model objects (same ones sequential MCTS would use)."""
    config, adapter, generator, evaluator, terminal = make_components(board_size=board_size)
    rollout = HeuristicRolloutPolicy(generator)
    return config, adapter, generator, evaluator, terminal, rollout


def _sequential_mcts(
    iterations: int = 40,
    *,
    seed: int = 0,
    board_size: int = 4,
) -> tuple[MCTS[SearchState], object]:
    config, adapter, generator, evaluator, terminal, rollout = _components(board_size)
    mcts = MCTS(
        MCTSConfig(iterations=iterations, max_simulation_steps=15, seed=seed),
        transition=adapter.transition,
        generate=generator.generate,
        is_terminal=terminal.is_terminal,
        terminal_value=terminal.value,
        evaluate=evaluator.evaluate,
        rollout=rollout.choose,
        rng=random.Random(seed),
    )
    return mcts, adapter


def _parallel_mcts(
    iterations: int = 40,
    *,
    workers: int = 2,
    seed: int = 0,
    board_size: int = 4,
    process_start_method: str | None = None,
) -> ParallelMCTS[SearchState]:
    config, adapter, generator, evaluator, terminal, rollout = _components(board_size)
    return ParallelMCTS(
        MCTSConfig(
            iterations=iterations,
            max_simulation_steps=15,
            seed=seed,
            workers=workers,
        ),
        transition=adapter.transition,
        generate=generator.generate,
        is_terminal=terminal.is_terminal,
        terminal_value=terminal.value,
        evaluate=evaluator.evaluate,
        rollout=rollout.choose,
        rng=random.Random(seed),
        process_start_method=process_start_method,
        transition_counter=adapter,
    )


# ---------------------------------------------------------------------------
# Budget distribution (requirement #9)
# ---------------------------------------------------------------------------

def test_split_budget_exact_sum() -> None:
    for total in (1, 2, 3, 7, 100, 101):
        for workers in (1, 2, 3, 8):
            budget = split_budget(total, workers)
            assert sum(budget) == total
            assert len(budget) == min(workers, total)


def test_split_budget_even_distribution() -> None:
    assert split_budget(100, 3) == (34, 33, 33)
    assert split_budget(100, 4) == (25, 25, 25, 25)
    assert split_budget(100, 1) == (100,)


def test_split_budget_fewer_than_workers() -> None:
    assert split_budget(3, 8) == (1, 1, 1)
    assert split_budget(1, 8) == (1,)


def test_split_budget_zero() -> None:
    assert split_budget(0, 4) == ()
    assert split_budget(0, 0) == ()
    assert split_budget(5, 0) == ()


# ---------------------------------------------------------------------------
# Worker seeding (requirement #10)
# ---------------------------------------------------------------------------

def test_worker_seed_distinct_and_deterministic() -> None:
    seeds = [worker_seed(0, i) for i in range(8)]
    assert len(set(seeds)) == 8  # distinct
    assert worker_seed(5, 0) == worker_seed(5, 0)  # deterministic
    assert worker_seed(5, 0) != worker_seed(6, 0)  # base seed matters


# ---------------------------------------------------------------------------
# Canonical action identity (requirement #8)
# ---------------------------------------------------------------------------

def test_canonical_key_same_logical_action_same_key() -> None:
    a1 = TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT))
    a2 = TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT))
    assert a1 == a2
    assert canonical_action_key(a1) == canonical_action_key(a2)


def test_canonical_key_different_actions_different_keys() -> None:
    a = TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT))
    b = TurnAction(farmer_action=WaterAction())
    c = TurnAction(farmer_action=PlantAction(crop=CropType.TOMATO))
    d = TurnAction(market_actions=(BuyAnimalAction(animal=AnimalType.GOOSE, quantity=1),))
    keys = {canonical_action_key(x) for x in (a, b, c, d)}
    assert len(keys) == 4


def test_canonical_key_is_stable_under_round_trip_pickle() -> None:
    import pickle

    action = TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT))
    revived = pickle.loads(pickle.dumps(action))
    assert canonical_action_key(action) == canonical_action_key(revived)


# ---------------------------------------------------------------------------
# Root statistics aggregation (requirement #20)
# ---------------------------------------------------------------------------

def _keyed(action: TurnAction) -> int:
    return canonical_action_key(action)


def test_aggregate_root_stats_weighted_not_mean() -> None:
    action_a = TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT))
    action_b = TurnAction(farmer_action=WaterAction())
    key_a = _keyed(action_a)
    key_b = _keyed(action_b)

    worker1 = WorkerResult(
        worker_id=0,
        root_stats=((key_a, action_a, 10, 50.0), (key_b, action_b, 5, 30.0)),
        transitions=0,
        wall_time=0.0,
    )
    worker2 = WorkerResult(
        worker_id=1,
        root_stats=((key_a, action_a, 20, 80.0), (key_b, action_b, 15, 45.0)),
        transitions=0,
        wall_time=0.0,
    )
    agg = aggregate_root_stats([worker1, worker2])

    assert agg[key_a] == (30, 130.0)  # N=30, W=130 -> Q=4.333...
    assert agg[key_b] == (20, 75.0)  # N=20, W=75 -> Q=3.75
    q_a = agg[key_a][1] / agg[key_a][0]
    q_b = agg[key_b][1] / agg[key_b][0]
    assert abs(q_a - 130.0 / 30.0) < 1e-9
    assert abs(q_b - 75.0 / 20.0) < 1e-9
    # Weighted aggregation must differ from the naive mean of per-worker Q.
    naive_a = (50.0 / 10 + 80.0 / 20) / 2
    assert abs(q_a - naive_a) > 1e-6


def test_aggregate_root_stats_empty() -> None:
    assert aggregate_root_stats([]) == {}


def test_select_best_action_key_and_action() -> None:
    action_a = TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT))
    action_b = TurnAction(farmer_action=WaterAction())
    key_a, key_b = _keyed(action_a), _keyed(action_b)
    agg = {key_a: (30, 130.0), key_b: (20, 75.0)}
    assert select_best_action_key(agg) == key_a
    assert select_best_action_key({}) is None

    stats = {action_a: RootStat(30, 130.0), action_b: RootStat(20, 75.0)}
    assert select_best_action_from_stats(stats) == action_a
    assert select_best_action_from_stats({}) == TurnAction()


# ---------------------------------------------------------------------------
# Sequential fallback (requirements #2, #16)
# ---------------------------------------------------------------------------

def test_workers1_matches_sequential_reference() -> None:
    state = make_search_state()
    seq, _ = _sequential_mcts(iterations=50, seed=3)
    par = _parallel_mcts(iterations=50, workers=1, seed=3)
    assert par.search(state, 0) == seq.search(state, 0)


def test_workers1_does_not_spawn_process() -> None:
    # workers=1 goes through the sequential path: total sims = iterations and
    # no ProcessPoolExecutor is created (worker_count == 1).
    par = _parallel_mcts(iterations=30, workers=1, seed=1)
    result = par.search_root(make_search_state(), 0)
    assert result.worker_count == 1
    assert result.total_simulations == 30
    assert result.root_stats  # has children


def test_iterations1_uses_sequential() -> None:
    par = _parallel_mcts(iterations=1, workers=4, seed=1)
    result = par.search_root(make_search_state(), 0)
    assert result.worker_count == 1
    assert result.total_simulations == 1


# ---------------------------------------------------------------------------
# Determinism (requirements #11, #21)
# ---------------------------------------------------------------------------

def test_sequential_deterministic_fixed_seed() -> None:
    # Level A: sequential MCTS with a fixed seed is exactly deterministic.
    a, _ = _sequential_mcts(seed=3)
    b, _ = _sequential_mcts(seed=3)
    state = make_search_state()
    assert a.search(state, 0) == b.search(state, 0)


def test_parallel_deterministic_same_config() -> None:
    # Level B: parallel MCTS with fixed seed/workers/budget is deterministic
    # (same config twice -> same result). We do NOT require it to equal the
    # sequential result, which is architecturally different.
    state = make_search_state()
    p1 = _parallel_mcts(iterations=40, workers=2, seed=11)
    p2 = _parallel_mcts(iterations=40, workers=2, seed=11)
    assert p1.search(state, 0) == p2.search(state, 0)
    r1 = p1.last_result
    r2 = p2.last_result
    assert r1 is not None and r2 is not None
    assert r1.best_action == r2.best_action
    assert r1.total_simulations == r2.total_simulations


def test_parallel_deterministic_four_workers() -> None:
    state = make_search_state()
    p1 = _parallel_mcts(iterations=40, workers=4, seed=5)
    p2 = _parallel_mcts(iterations=40, workers=4, seed=5)
    assert p1.search(state, 0) == p2.search(state, 0)


def test_parallel_different_seeds_may_differ() -> None:
    # Different seeds produce (probabilistically) independent searches; this
    # just checks they run cleanly and stay within the legal action space.
    state = make_search_state()
    p1 = _parallel_mcts(iterations=30, workers=2, seed=1)
    p2 = _parallel_mcts(iterations=30, workers=2, seed=999)
    a1, a2 = p1.search(state, 0), p2.search(state, 0)
    assert isinstance(a1, TurnAction)
    assert isinstance(a2, TurnAction)


# ---------------------------------------------------------------------------
# Exact budget accounting (requirement #9)
# ---------------------------------------------------------------------------

def test_parallel_exact_total_budget() -> None:
    par = _parallel_mcts(iterations=100, workers=3, seed=2)
    result = par.search_root(make_search_state(), 0)
    assert result.total_simulations == 100
    assert result.worker_count == 3
    total_visits = sum(stat.visits for stat in result.root_stats.values())
    assert total_visits == 100


def test_parallel_workers_greater_than_budget() -> None:
    par = _parallel_mcts(iterations=3, workers=8, seed=2)
    result = par.search_root(make_search_state(), 0)
    assert result.worker_count == 3  # active_workers = min(8, 3)
    assert result.total_simulations == 3


def test_parallel_budget_zero() -> None:
    par = _parallel_mcts(iterations=0, workers=4, seed=2)
    result = par.search_root(make_search_state(), 0)
    assert result.total_simulations == 0
    assert result.best_action == TurnAction()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_parallel_terminal_state_returns_pass() -> None:
    config, adapter, generator, evaluator, terminal, rollout = _components()
    par = ParallelMCTS(
        MCTSConfig(iterations=20, workers=2, seed=3),
        transition=adapter.transition,
        generate=generator.generate,
        is_terminal=terminal.is_terminal,
        terminal_value=terminal.value,
        evaluate=evaluator.evaluate,
        rollout=rollout.choose,
    )
    # A state already at/after the episode length is terminal -> PASS.
    terminal_state = make_search_state(step=config.episode_steps)
    assert par.search(terminal_state, 0) == TurnAction()


def test_worker_function_runs_in_process() -> None:
    # Exercise the module-level worker directly (also proves it is picklable
    # and returns the compact result shape).
    config, adapter, generator, evaluator, terminal, rollout = _components()
    task = WorkerTask(
        root_state=make_search_state(),
        player=0,
        iterations=10,
        seed=worker_seed(7, 0),
        exploration_constant=1.41,
        max_simulation_steps=15,
        transition=adapter.transition,
        generate=generator.generate,
        is_terminal=terminal.is_terminal,
        terminal_value=terminal.value,
        evaluate=evaluator.evaluate,
        rollout=rollout.choose,
        worker_id=0,
        transition_counter=adapter,
    )
    result = run_mcts_worker(task)
    assert result.worker_id == 0
    assert result.root_stats
    assert result.transitions > 0
    for key, action, visits, value in result.root_stats:
        assert isinstance(key, int)
        assert isinstance(action, TurnAction)
        assert visits >= 1
        assert value >= 0.0


def test_worker_failure_surfaces_with_worker_id() -> None:
    # A worker whose model raises must surface the exception (with the worker
    # id), never silently produce a partial/random action.
    config, adapter, generator, evaluator, terminal, _rollout = _components()

    def broken_rollout(state: SearchState, rng: random.Random) -> TurnAction:
        del state, rng
        raise RuntimeError("boom in rollout")

    par = ParallelMCTS(
        MCTSConfig(iterations=20, workers=2, seed=3),
        transition=adapter.transition,
        generate=generator.generate,
        is_terminal=terminal.is_terminal,
        terminal_value=terminal.value,
        evaluate=evaluator.evaluate,
        rollout=broken_rollout,
    )
    with pytest.raises(RuntimeError, match="worker"):
        par.search(make_search_state(), 0)


# ---------------------------------------------------------------------------
# Windows / spawn safety (requirement #14)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_spawn_safe(workers: int) -> None:
    # On Windows this creates the pool with the spawn start method; the worker
    # function and payload must be importable/picklable. Using a small budget
    # keeps the test fast while still exercising real process execution.
    par = _parallel_mcts(iterations=24, workers=workers, seed=4)
    result = par.search_root(make_search_state(), 0)
    assert result.worker_count == workers
    assert result.total_simulations == 24
