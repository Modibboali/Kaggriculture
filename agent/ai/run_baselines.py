"""Head-to-head baseline experiments for the MCTS agent (task 10).

Plays matchups on the real Kaggle environment between the MCTS agent and
non-searching baselines (Random / Starter / Heuristic) and prints a compact
report: win rate, average reward, games, episode length, and MCTS search
statistics.

Run from the repo root:

    python -m agent.ai.run_baselines --episode-steps 120 --games 5 --iterations 50

The defaults are deliberately modest so the whole run finishes quickly; raise
``--episode-steps 720`` for a full-episode smoke test.
"""

from __future__ import annotations

import argparse

from .agent import (
    Agent,
    HeuristicAgent,
    MCTSAgent,
    RandomAgent,
    StarterAgent,
)
from .experiment import run_matchup
from .mcts import MCTSConfig


def _make_agents(iterations: int, seed: int, max_simulation_steps: int) -> dict[str, Agent]:
    return {
        "mcts": MCTSAgent(
            MCTSConfig(
                iterations=iterations,
                max_simulation_steps=max_simulation_steps,
                seed=seed,
            ),
            seed=seed,
        ),
        "random": RandomAgent(seed=seed),
        "starter": StarterAgent(),
        "heuristic": HeuristicAgent(seed=seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MCTS baseline experiments")
    parser.add_argument("--episode-steps", type=int, default=120)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--max-simulation-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="one full 720-step episode")
    args = parser.parse_args()

    agents = _make_agents(args.iterations, args.seed, args.max_simulation_steps)

    if args.smoke:
        from .experiment import play_episode

        print(
            f"Smoke: MCTS(it={args.iterations}, sim_steps={args.max_simulation_steps}) "
            f"vs Starter, 720 steps ..."
        )
        smoke = play_episode(
            agents["mcts"],
            agents["starter"],
            episode_steps=720,
            seed=args.seed,
        )
        print(
            f"  reward mcts={smoke.reward0:.1f} starter={smoke.reward1:.1f} "
            f"steps={smoke.steps} winner={smoke.winner}"
        )
        return

    print(
        f"Matchups: episode_steps={args.episode_steps} games={args.games} "
        f"mcts_iterations={args.iterations} sim_steps={args.max_simulation_steps} "
        f"seed={args.seed}\n"
    )
    header = (
        f"{'matchup':<28}{'win%':>6}{'reward0':>10}{'reward1':>10}"
        f"{'steps':>7}{'search_s':>10}{'trans':>10}{'searches':>10}{'wall_s':>9}"
    )
    print(header)
    print("-" * len(header))

    matchups: list[tuple[str, str, str]] = [
        ("mcts_vs_random", "mcts", "random"),
        ("mcts_vs_starter", "mcts", "starter"),
        ("mcts_vs_heuristic", "mcts", "heuristic"),
        ("random_vs_starter", "random", "starter"),
    ]
    for name, a0, a1 in matchups:
        # Fresh agents per matchup so per-game search stats are not cumulative.
        matchup_agents = _make_agents(args.iterations, args.seed, args.max_simulation_steps)
        result = run_matchup(
            matchup_agents[a0],
            matchup_agents[a1],
            name=name,
            games=args.games,
            episode_steps=args.episode_steps,
            seed=args.seed,
        )
        print(
            f"{name:<28}{result.win_rate0 * 100:>6.0f}"
            f"{result.avg_reward0:>10.1f}{result.avg_reward1:>10.1f}"
            f"{result.avg_steps:>7.0f}{result.total_search_time:>10.1f}"
            f"{result.total_transitions:>10}{result.total_searches:>10}"
            f"{result.wall_time:>9.1f}"
        )


if __name__ == "__main__":
    main()
