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
from .evaluation import Evaluator, HorizonAwareEvaluator
from .mcts import MCTS, MCTSConfig
from .rollout import HeuristicRolloutPolicy
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

    state = _state()
    game_config = GameConfig(board_size=10)
    adapter = SimulatorAdapter(count_transitions=True)
    generator = ActionGenerator(game_config)
    evaluator = Evaluator(game_config)
    horizon_evaluator = HorizonAwareEvaluator(game_config)
    terminal = Terminal(game_config)
    rollout = HeuristicRolloutPolicy(generator)

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

    return results


def main() -> None:
    results = benchmark()
    for key, value in results.items():
        print(f"{key}: {value:.1f}")


if __name__ == "__main__":
    main()
