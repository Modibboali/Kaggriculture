"""Run one matchup chunk and append its aggregate to a CSV.

The environment suspends very long-running processes, so large matchups are
split across many short process invocations. This script runs ONE chunk of
games (a contiguous seed range) and appends one aggregate row to a results
CSV; a final ``summarize_matchups`` step aggregates the rows.

    python -m agent.ai.run_chunked --kind new --opponent starter --days 5 \
        --iters 12 --seed-start 1 --seed-end 11 --out output/matchups.csv
"""

from __future__ import annotations

import argparse
import csv

from ..simulator import GameConfig
from .run_horizon_experiments import _opponent, make_mcts
from .sim_experiment import run_sim_matchup


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one matchup chunk to a CSV")
    parser.add_argument("--kind", choices=("new", "old", "no-crop", "no-animal-worker"), default="new")
    parser.add_argument("--mode", choices=("A", "B", "C", "D", "E"), default="A")
    parser.add_argument("--opponent", choices=("random", "starter", "heuristic"))
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = GameConfig(episode_steps=args.days * 24)
    agent = make_mcts(args.kind, config, iterations=args.iters, mode=args.mode)
    result = run_sim_matchup(
        agent,
        _opponent(args.opponent, config),
        name=f"m{args.mode}_{args.kind}_vs_{args.opponent}",
        games=args.seed_end - args.seed_start,
        config=config,
        seed_start=args.seed_start,
    )
    with open(args.out, "a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                args.mode,
                args.kind,
                args.opponent,
                args.days,
                args.iters,
                args.seed_start,
                args.seed_end,
                f"{result.win_rate0:.4f}",
                f"{result.mean_reward0:.1f}",
                f"{result.median_reward0:.1f}",
                f"{result.std_reward0:.2f}",
                f"{result.mean_reward1:.1f}",
                f"{result.mean_steps:.1f}",
                f"{result.total_search_time / result.total_searches:.1f}" if result.total_searches else "0",
                str(result.total_transitions),
            ]
        )
    print(
        f"chunk {args.kind} vs {args.opponent} seeds[{args.seed_start},{args.seed_end}) "
        f"win={result.win_rate0 * 100:.0f}% r0={result.mean_reward0:.1f} "
        f"r1={result.mean_reward1:.1f} wall={result.wall_time:.1f}s"
    )


if __name__ == "__main__":
    main()
