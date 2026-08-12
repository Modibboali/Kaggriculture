"""Micro-benchmarks for the AI / MCTS layer.

Measures the per-operation costs that dominate search (simulator transition,
SearchState conversion, action generation, hashing, evaluation) and the
end-to-end MCTS throughput. Run with ``python -m agent.ai.benchmark``.

All measurements use a synthetic 10x10 state (no Kaggle dependency) so they
are stable and reproducible. ``MCTS simulations/sec`` counts one UCT iteration
(select + expand + rollout + backprop); ``environment transitions/sec during
MCTS`` counts the simulator transitions those simulations performed — these
are deliberately reported separately, since one simulation can contain many
transitions.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

from ..actions import TurnAction
from .action_generator import ActionGenerator
from .evaluation import EvaluationConfig, Evaluator, HorizonAwareEvaluator
from .mcts import MCTS, MCTSConfig
from .rollout import CashConversionRolloutPolicy, HeuristicRolloutPolicy
from .search_state import SearchState
from .simulator_adapter import SimulatorAdapter
from .terminal import Terminal


def _state() -> SearchState:
    from tests.ai._build import make_search_state

    return make_search_state(board_size=10)


def _rate(count: int, seconds: float) -> float:
    return count / seconds if seconds > 0 else 0.0


def _measure(fn: Callable[[], None], seconds: float = 0.4) -> tuple[int, float]:
    count = 0
    elapsed = 0.0
    while elapsed < seconds:
        start = time.perf_counter()
        fn()
        elapsed += time.perf_counter() - start
        count += 1
    return count, elapsed


def benchmark(
    *,
    seconds: float = 0.4,
    mcts_iterations: int = 200,
) -> dict[str, float]:
    from ..simulator import GameConfig
    from .action_priority import ActionPriorityModel

    state = _state()
    game_config = GameConfig(board_size=10)
    adapter = SimulatorAdapter(count_transitions=True)
    generator = ActionGenerator(game_config)
    evaluator = Evaluator(game_config)
    horizon_evaluator = HorizonAwareEvaluator(game_config)
    terminal = Terminal(game_config)
    rollout = HeuristicRolloutPolicy(generator)

    # Task-12 components (phase model / priority / realizability / cash rollout).
    eval_config = EvaluationConfig()
    priority_model = ActionPriorityModel(game_config, eval_config)
    cash_rollout = CashConversionRolloutPolicy(generator, priority_model)
    rng = random.Random(0)
    actions = generator.generate(state)

    results: dict[str, float] = {}

    # Simulator transitions/sec (one PASS turn).
    def trans() -> None:
        adapter.transition(state, TurnAction())

    count, elapsed = _measure(trans, seconds)
    results["simulator_transitions_per_sec"] = _rate(count, elapsed)

    # SearchState conversion/sec (GameState -> SearchState).
    def conv() -> None:
        SearchState(state.game)

    count, elapsed = _measure(conv, seconds)
    results["searchstate_conversions_per_sec"] = _rate(count, elapsed)

    # Action generation/sec.
    def gen() -> None:
        generator.generate(state)

    count, elapsed = _measure(gen, seconds)
    results["action_generations_per_sec"] = _rate(count, elapsed)

    # Phase detection/sec.
    def phase() -> None:
        priority_model.phase(state)

    count, elapsed = _measure(phase, seconds)
    results["phase_detections_per_sec"] = _rate(count, elapsed)

    # Priority ranking/sec (rank the generated candidate set).
    def rank() -> None:
        priority_model.rank(state, list(actions))

    count, elapsed = _measure(rank, seconds)
    results["action_priority_ranks_per_sec"] = _rate(count, elapsed)

    # Realizability filter/sec.
    def realize() -> None:
        priority_model.filter_realizable(state, list(actions))

    count, elapsed = _measure(realize, seconds)
    results["realizability_filters_per_sec"] = _rate(count, elapsed)

    # CashConversion rollout step/sec.
    def cash() -> None:
        cash_rollout.choose(state, rng)

    count, elapsed = _measure(cash, seconds)
    results["cashconversion_rollout_steps_per_sec"] = _rate(count, elapsed)

    # State hashing/sec.
    def hashing() -> None:
        state.state_key()

    count, elapsed = _measure(hashing, seconds)
    results["state_hashes_per_sec"] = _rate(count, elapsed)

    # Classic evaluation/sec.
    def eval_() -> None:
        evaluator.evaluate(state, 0)

    count, elapsed = _measure(eval_, seconds)
    results["evaluations_per_sec"] = _rate(count, elapsed)

    # Horizon-aware evaluation/sec (same state).
    def eval_h() -> None:
        horizon_evaluator.evaluate(state, 0)

    count, elapsed = _measure(eval_h, seconds)
    results["horizon_evaluations_per_sec"] = _rate(count, elapsed)

    # MCTS throughput with the horizon-aware evaluator (the production config).
    iterations = mcts_iterations
    mcts = MCTS(
        MCTSConfig(iterations=iterations, max_simulation_steps=12, seed=0),
        transition=adapter.transition,
        generate=generator.generate,
        is_terminal=terminal.is_terminal,
        terminal_value=terminal.value,
        evaluate=horizon_evaluator.evaluate,
        rollout=rollout.choose,
        rng=random.Random(0),
    )
    start = time.perf_counter()
    mcts.search(state, 0)
    search_time = time.perf_counter() - start
    transitions_during = adapter.transitions
    results["mcts_simulations_per_sec"] = _rate(iterations, search_time)
    results["mcts_env_transitions_per_sec"] = _rate(transitions_during, search_time)
    results["mcts_search_time_sec"] = search_time

    # MCTS throughput with the CashConversion rollout (recommended mode E).
    mcts_cash = MCTS(
        MCTSConfig(iterations=iterations, max_simulation_steps=12, seed=0),
        transition=adapter.transition,
        generate=generator.generate,
        is_terminal=terminal.is_terminal,
        terminal_value=terminal.value,
        evaluate=horizon_evaluator.evaluate,
        rollout=cash_rollout.choose,
        rng=rng,
    )
    start = time.perf_counter()
    mcts_cash.search(state, 0)
    search_time_cash = time.perf_counter() - start
    results["mcts_cash_simulations_per_sec"] = _rate(iterations, search_time_cash)
    results["mcts_cash_search_time_sec"] = search_time_cash

    return results


# ---------------------------------------------------------------------------
# Parallel MCTS benchmark (root-parallel throughput + serialization overhead)
# ---------------------------------------------------------------------------

def _pickle_task(
    task: object,
    *,
    reps: int = 3,
) -> tuple[float, int]:
    """Serialize ``task`` ``reps`` times; return (seconds, payload bytes)."""
    import pickle

    payload = pickle.dumps(task)
    start = time.perf_counter()
    for _ in range(reps):
        pickle.dumps(task)
    elapsed = (time.perf_counter() - start) / reps
    return elapsed, len(payload)


def parallel_benchmark(
    *,
    workers_list: tuple[int, ...] = (1, 2, 4, 8),
    iterations: int = 600,
    max_simulation_steps: int = 12,
) -> list[dict[str, float]]:
    """Measure root-parallel MCTS throughput across worker counts.

    Uses the exact same state / model components for every worker count so the
    only variable is the number of processes. Reports simulations/sec, speedup
    and efficiency relative to ``workers == 1`` (the sequential reference), plus
    state-serialization and process-startup overhead.
    """
    from ..simulator import GameConfig
    from .parallel_mcts import (
        ParallelMCTS,
        WorkerTask,
        split_budget,
        worker_seed,
    )

    state = _state()
    game_config = GameConfig(board_size=10)
    generator = ActionGenerator(game_config)
    horizon_evaluator = HorizonAwareEvaluator(game_config)
    terminal = Terminal(game_config)
    rollout = HeuristicRolloutPolicy(generator)

    rows: list[dict[str, float]] = []
    baseline = 0.0
    for workers in workers_list:
        adapter = SimulatorAdapter(count_transitions=True)
        seed = 0
        config = MCTSConfig(
            iterations=iterations,
            max_simulation_steps=max_simulation_steps,
            seed=seed,
            workers=workers,
        )
        parallel = ParallelMCTS(
            config,
            transition=adapter.transition,
            generate=generator.generate,
            is_terminal=terminal.is_terminal,
            terminal_value=terminal.value,
            evaluate=horizon_evaluator.evaluate,
            rollout=rollout.choose,
            rng=random.Random(seed),
            transition_counter=adapter,
        )

        # State serialization cost (one worker task payload) + process startup.
        serialize_sec = 0.0
        payload_bytes = 0
        startup_sec = 0.0
        if workers > 1:
            budgets = split_budget(iterations, workers)
            task = WorkerTask(
                root_state=state,
                player=0,
                iterations=budgets[0],
                seed=worker_seed(seed, 0),
                exploration_constant=config.exploration_constant,
                max_simulation_steps=config.max_simulation_steps,
                transition=adapter.transition,
                generate=generator.generate,
                is_terminal=terminal.is_terminal,
                terminal_value=terminal.value,
                evaluate=horizon_evaluator.evaluate,
                rollout=rollout.choose,
                worker_id=0,
                transition_counter=adapter,
            )
            serialize_sec, payload_bytes = _pickle_task(task)
            # Pool startup overhead: create + immediately shut down a pool.
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            from .parallel_mcts import run_mcts_worker

            ctx = multiprocessing.get_context()
            start = time.perf_counter()
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                pass
            startup_sec = time.perf_counter() - start

        start = time.perf_counter()
        result = parallel.search_root(state, 0)
        wall = time.perf_counter() - start
        sims_per_sec = _rate(iterations, wall)
        # For workers == 1 the sequential path runs in-process, so the
        # caller-side counting adapter sees the transitions; for workers > 1
        # they happen in the workers and are summed by ParallelMCTS.
        if workers == 1:
            transitions = adapter.transitions
            worker_peak_mb = 0.0
        else:
            transitions = parallel.transitions
            worker_peak_mb = sum(
                wr.peak_rss_mb for wr in result.worker_results
            )
        trans_per_sec = _rate(transitions, wall)
        if workers == 1:
            baseline = sims_per_sec
        speedup = sims_per_sec / baseline if baseline > 0 else 1.0
        rows.append(
            {
                "workers": float(workers),
                "simulations": float(result.total_simulations),
                "wall_time": wall,
                "sims_per_sec": sims_per_sec,
                "transitions": transitions,
                "trans_per_sec": trans_per_sec,
                "speedup": speedup,
                "efficiency": speedup / workers,
                "serialize_sec": serialize_sec,
                "payload_bytes": float(payload_bytes),
                "pool_startup_sec": startup_sec,
                "memory_mb": worker_peak_mb,
            }
        )
    return rows


def _peak_rss_mb() -> float:
    """Peak resident memory of the current process in MiB (0.0 if unavailable)."""
    try:
        import psutil
    except ImportError:
        return 0.0
    return float(psutil.Process().memory_info().rss) / (1024 * 1024)


def _print_parallel_table(rows: list[dict[str, float]]) -> None:
    print(f"\n{'workers':>7} {'sims':>6} {'wall_s':>8} {'sims/s':>9} "
          f"{'trans/s':>9} {'speedup':>8} {'efficiency':>10} {'mem_MB':>8}")
    for row in rows:
        print(
            f"{int(row['workers']):>7} {int(row['simulations']):>6} "
            f"{row['wall_time']:>8.2f} {row['sims_per_sec']:>9.1f} "
            f"{row['trans_per_sec']:>9.1f} {row['speedup']:>7.2f}x "
            f"{row['efficiency'] * 100:>9.1f}% {row['memory_mb']:>8.1f}"
        )
    print("\nSerialization / dispatch overhead (per search):")
    for row in rows:
        if row["workers"] > 1:
            print(
                f"  workers={int(row['workers'])}  task_pickle={row['serialize_sec'] * 1000:.3f} ms "
                f"payload={int(row['payload_bytes'])} B  pool_startup={row['pool_startup_sec']:.3f} s"
            )


def cmd_parallel(workers_arg: list[int], iterations: int) -> None:
    workers_list = tuple(workers_arg)
    print(
        f"Parallel MCTS benchmark: iterations={iterations}, workers={workers_list}, "
        f"max_simulation_steps=12"
    )
    rows = parallel_benchmark(workers_list=workers_list, iterations=iterations)
    _print_parallel_table(rows)


def main() -> None:
    import sys

    results = benchmark()
    for key, value in results.items():
        print(f"{key}: {value:.1f}")

    if "parallel" in sys.argv:
        cmd_parallel([1, 2, 4, 8, 16], iterations=600)


if __name__ == "__main__":
    main()
