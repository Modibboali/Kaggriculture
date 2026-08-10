"""Quick performance sanity checks for the AI layer (fast variants)."""

from __future__ import annotations

from agent.ai.benchmark import benchmark


def test_benchmark_rates_are_positive() -> None:
    results = benchmark(seconds=0.05, mcts_iterations=20)

    # Per-operation throughput must be non-zero (and sane).
    assert results["simulator_transitions_per_sec"] > 0
    assert results["searchstate_conversions_per_sec"] > 0
    assert results["action_generations_per_sec"] > 0
    assert results["state_hashes_per_sec"] > 0
    assert results["evaluations_per_sec"] > 0

    # MCTS throughput: simulations/sec and env transitions/sec must be > 0.
    assert results["mcts_simulations_per_sec"] > 0
    assert results["mcts_env_transitions_per_sec"] > 0
    assert results["mcts_search_time_sec"] >= 0

    # A simulation applies many transitions through the search-state wrapper;
    # the per-simulation cost should be well below a raw simulator transition.
    assert results["mcts_simulations_per_sec"] < results["simulator_transitions_per_sec"]
