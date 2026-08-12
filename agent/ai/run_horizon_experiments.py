"""Horizon-aware MCTS experiments (task: horizon-aware milestone).

Subcommands (all simulator-based, Kaggle-free, fast enough for 100 games):

    regression   old vs new evaluator across horizons (1/3/5/10/30 days)
    matchups     100-game matchups: new evaluator vs random/starter/heuristic
    sweep        MCTS iteration-budget sweep (25..500) with latency
    ablation     A classic / B horizon / C no-crop / D no-animal+worker

Run from the repo root:

    python -m agent.ai.run_horizon_experiments regression --days 5 --games 10
    python -m agent.ai.run_horizon_experiments matchups --days 5 --games 100
    python -m agent.ai.run_horizon_experiments sweep --days 3 --games 5
    python -m agent.ai.run_horizon_experiments ablation --days 5 --games 10
"""

from __future__ import annotations

import argparse
from typing import Callable

from ..actions import TurnAction
from ..simulator import GameConfig
from .action_generator import ActionGenerator
from .agent import Agent, HeuristicAgent, MCTSAgent, RandomAgent, StarterAgent
from .evaluation import EvaluationConfig, Evaluator, HorizonAwareEvaluator
from .mcts import MCTSConfig
from .rollout import HeuristicRolloutPolicy, RolloutPolicy
from .sim_experiment import SimMatchupResult, run_sim_matchup

_BASE = 24  # turns per day


def make_mcts(
    kind: str,
    config: GameConfig,
    *,
    iterations: int,
    eval_config: EvaluationConfig | None = None,
    seed: int = 1,
    mode: str = "A",
    workers: int = 1,
) -> MCTSAgent:
    """Build an MCTS agent.

    ``kind`` selects the evaluator; ``mode`` selects the action-space
    configuration (task: action branching / rollout ablation):

        A  current  : plain generator, heuristic rollout
        B  farming  : generator filtered to crop/cash actions, heuristic rollout
        C  phase    : all actions, phase-prioritised ordering, heuristic rollout
        D  phase+realizability : prioritised AND unrealisable actions dropped
        E  cashconversion: D + CashConversionRolloutPolicy

    ``workers`` > 1 selects the root-parallel execution strategy (task:
    multi-vCPU parallel MCTS); workers == 1 keeps the canonical sequential
    MCTS. ``iterations`` is always the *total* simulation budget regardless of
    worker count.
    """
    from .action_priority import ActionPriorityModel
    from .rollout import CashConversionRolloutPolicy

    mcts_config = MCTSConfig(
        iterations=iterations,
        max_simulation_steps=12,
        seed=seed,
        workers=workers,
    )
    evaluator: Evaluator
    if kind == "new":
        evaluator = HorizonAwareEvaluator(config, eval_config)
    elif kind == "old":
        evaluator = Evaluator(config)
    elif kind == "no-crop":  # evaluator ablation C: no crop realizability
        evaluator = HorizonAwareEvaluator(config, EvaluationConfig(crop_realizability=False))
    elif kind == "no-animal-worker":  # evaluator ablation D
        evaluator = HorizonAwareEvaluator(
            config,
            EvaluationConfig(animal_horizon_value=False, worker_horizon_value=False),
        )
    else:
        raise ValueError(f"unknown evaluator kind: {kind}")

    model = ActionPriorityModel(config, eval_config)
    rollout: RolloutPolicy
    if mode == "A":
        generator = ActionGenerator(config)
        rollout = HeuristicRolloutPolicy(generator)
    elif mode == "B":
        generator = ActionGenerator(
            config, action_filter=ActionPriorityModel.farming_only_filter
        )
        rollout = HeuristicRolloutPolicy(generator)
    elif mode == "C":
        generator = ActionGenerator(config, priority_model=model)
        rollout = HeuristicRolloutPolicy(generator)
    elif mode == "D":
        generator = ActionGenerator(
            config, priority_model=model, action_filter=model.filter_realizable
        )
        rollout = HeuristicRolloutPolicy(generator)
    elif mode == "E":
        generator = ActionGenerator(
            config, priority_model=model, action_filter=model.filter_realizable
        )
        rollout = CashConversionRolloutPolicy(generator, model)
    else:
        raise ValueError(f"unknown mode: {mode}")

    return MCTSAgent(
        mcts_config,
        config=config,
        evaluator=evaluator,
        generator=generator,
        rollout=rollout,
        seed=seed,
    )


def _opponent(name: str, config: GameConfig) -> Agent:
    if name == "random":
        return RandomAgent(seed=1)
    if name == "starter":
        return StarterAgent(config=config)
    if name == "heuristic":
        return HeuristicAgent(seed=1)
    raise ValueError(f"unknown opponent: {name}")


def _print_result(r: SimMatchupResult) -> None:
    print(
        f"  {r.name:<26} win%={r.win_rate0 * 100:>5.0f} "
        f"r0(mean/med/sd)={r.mean_reward0:>8.1f}/{r.median_reward0:>8.1f}/{r.std_reward0:>7.1f} "
        f"r1={r.mean_reward1:>8.1f} steps={r.mean_steps:>4.0f} "
        f"wall={r.wall_time:>7.1f}s"
    )


def _run(
    kind: str,
    opponent: str,
    config: GameConfig,
    *,
    games: int,
    iterations: int,
    eval_config: EvaluationConfig | None = None,
) -> SimMatchupResult:
    agent = make_mcts(kind, config, iterations=iterations, eval_config=eval_config)
    return run_sim_matchup(
        agent, _opponent(opponent, config), name=f"{kind}_vs_{opponent}",
        games=games, config=config,
    )


def cmd_regression(args: argparse.Namespace) -> None:
    days_list = [int(d) for d in args.days.split(",")]
    print(f"Regression: old vs new evaluator, games={args.games}, iters={args.iterations}")
    for days in days_list:
        config = GameConfig(episode_steps=days * _BASE)
        print(f"\n--- horizon: {days} day(s) ({config.episode_steps} steps) ---")
        for opponent in ("starter", "heuristic"):
            for kind in ("old", "new"):
                result = _run(kind, opponent, config, games=args.games, iterations=args.iterations)
                _print_result(result)


def cmd_matchups(args: argparse.Namespace) -> None:
    config = GameConfig(episode_steps=args.days * _BASE)
    kinds = ("new", "old") if args.kind == "both" else (args.kind,)
    print(
        f"Matchups (100-game scale): horizon={args.days}d, games={args.games}, "
        f"iters={args.iterations}, kind={args.kind}"
    )
    for kind in kinds:
        print(f"  -- evaluator: {kind} --")
        for opponent in ("random", "starter", "heuristic"):
            result = _run(kind, opponent, config, games=args.games, iterations=args.iterations)
            _print_result(result)


def cmd_sweep(args: argparse.Namespace) -> None:
    config = GameConfig(episode_steps=args.days * _BASE)
    print(
        f"Budget sweep: horizon={args.days}d, games={args.games}, mode={args.mode}, "
        f"vs starter"
    )
    print(f"{'iters':>6} {'win%':>6} {'mean_r0':>9} {'sims/s':>8} {'trans/s':>9} {'latency_ms':>10}")
    for iterations in (25, 50, 100, 250, 500):
        agent = make_mcts("new", config, iterations=iterations, mode=args.mode)
        result = run_sim_matchup(
            agent, StarterAgent(config=config), name=f"sweep_{args.mode}_{iterations}",
            games=args.games, config=config,
        )
        sims = result.total_searches * iterations
        sims_s = sims / result.total_search_time if result.total_search_time > 0 else 0.0
        trans_s = (
            result.total_transitions / result.total_search_time
            if result.total_search_time > 0
            else 0.0
        )
        latency_ms = (
            1000.0 * result.total_search_time / result.total_searches
            if result.total_searches > 0
            else 0.0
        )
        print(
            f"{iterations:>6} {result.win_rate0 * 100:>6.0f} "
            f"{result.mean_reward0:>9.1f} {sims_s:>8.1f} {trans_s:>9.1f} {latency_ms:>10.1f}"
        )


def cmd_ablation(args: argparse.Namespace) -> None:
    config = GameConfig(episode_steps=args.days * _BASE)
    print(f"Ablation: vs starter at {args.days}d, games={args.games}, iters={args.iterations}")
    print(f"{'variant':<22}{'win%':>6}{'mean_r0':>10}{'mean_r1':>10}")
    for kind in ("old", "new", "no-crop", "no-animal-worker"):
        result = _run(kind, "starter", config, games=args.games, iterations=args.iterations)
        print(
            f"{kind:<22}{result.win_rate0 * 100:>6.0f}"
            f"{result.mean_reward0:>10.1f}{result.mean_reward1:>10.1f}"
        )


def _mode_run(
    mode: str,
    opponent: str,
    config: GameConfig,
    *,
    games: int,
    iterations: int,
) -> SimMatchupResult:
    agent = make_mcts("new", config, iterations=iterations, mode=mode)
    return run_sim_matchup(
        agent, _opponent(opponent, config), name=f"{mode}_vs_{opponent}",
        games=games, config=config,
    )


def cmd_modes(args: argparse.Namespace) -> None:
    """Action-space ablation (modes A-E) against a chosen opponent."""
    config = GameConfig(episode_steps=args.days * _BASE)
    print(
        f"Modes A-E: horizon={args.days}d, games={args.games}, "
        f"iters={args.iterations}, opponent={args.opponent}"
    )
    print(f"{'mode':<22}{'win%':>6}{'mean_r0':>9}{'med_r0':>9}{'sd_r0':>8}{'mean_r1':>9}{'steps':>6}")
    for mode in ("A", "B", "C", "D", "E"):
        result = _mode_run(mode, args.opponent, config, games=args.games, iterations=args.iterations)
        print(
            f"{mode:<22}{result.win_rate0 * 100:>6.0f}"
            f"{result.mean_reward0:>9.1f}{result.median_reward0:>9.1f}"
            f"{result.std_reward0:>8.1f}{result.mean_reward1:>9.1f}{result.mean_steps:>6.0f}"
        )


# Action-type categories for instrumentation.
_CATEGORY: dict[str, set[object]] = {
    "farming": {"PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG"},
    "movement": {"NORTH", "SOUTH", "EAST", "WEST"},
    "market": {"SELL", "BUY_SEED", "BUY_PRODUCT"},
    "land": {"BUY_LAND"},
    "building": {"BUILD_COOP", "BUILD_PASTURE"},
    "animal": {"PLACE", "FEED", "CARE", "COLLECT_FERTILIZER", "BUY_ANIMAL"},
    "worker": {"HIRE"},
    "inventory": {"PICKUP", "DROP"},
    "pass": {"PASS"},
}


def cmd_parallel(args: argparse.Namespace) -> None:
    """Sequential vs root-parallel MCTS quality comparison (task: parallel MCTS).

    Plays the SAME total simulation budget with workers=1 (canonical
    sequential) and workers=N (root-parallel) against the starter, so the only
    variable is the execution strategy. Root-parallel may pick different
    actions (independent trees) — the point is to confirm throughput scales and
    quality does not regress, not that the policies are identical.
    """
    config = GameConfig(episode_steps=args.days * _BASE)
    mode = args.mode
    print(
        f"Parallel quality: horizon={args.days}d, games={args.games}, "
        f"iters={args.iterations} (total, each), mode={mode}, vs starter"
    )
    print(f"{'config':<28}{'win%':>6}{'mean_r0':>9}{'med_r0':>9}{'sd_r0':>8}{'steps':>6}")
    for workers in args.workers:
        agent = make_mcts(
            "new", config, iterations=args.iterations, mode=mode, workers=workers
        )
        result = run_sim_matchup(
            agent,
            StarterAgent(config=config),
            name=f"w{workers}_vs_starter",
            games=args.games,
            config=config,
        )
        label = "sequential" if workers == 1 else f"parallel w={workers}"
        print(
            f"{label:<28}{result.win_rate0 * 100:>6.0f}"
            f"{result.mean_reward0:>9.1f}{result.median_reward0:>9.1f}"
            f"{result.std_reward0:>8.1f}{result.mean_steps:>6.0f}"
        )


def cmd_instrument(args: argparse.Namespace) -> None:
    """Profile the generated action space by category across horizons."""
    from ..actions import ActionType
    from ..state import GameState
    from .action_priority import farmer_type
    from .search_state import SearchState
    from .sim_experiment import initial_state
    from ..simulator import Simulator

    def profile(config: GameConfig, label: str, day: int) -> None:
        state: GameState = initial_state(config)
        sim = Simulator(config)
        for _ in range(day * config.turns_per_day):
            state = sim.apply(state, (TurnAction(), TurnAction()))
        gen = ActionGenerator(config)
        actions = gen.generate(SearchState(state))
        counts: dict[str, int] = {}
        for action in actions:
            at = farmer_type(action).label
            for cat, types in _CATEGORY.items():
                if at in types:
                    counts[cat] = counts.get(cat, 0) + 1
                    break
            else:
                counts[at] = counts.get(at, 0) + 1
        total = len(actions)
        cats = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  [{label}] day={day} total={total} :: {cats}")

    print(f"Action-space profile ({args.days}d horizon, {args.days * _BASE} steps):")
    config = GameConfig(episode_steps=args.days * _BASE)
    horizon_days = args.days
    for day in sorted(set([0, max(0, horizon_days // 2), max(0, horizon_days - 1)])):
        profile(config, "early" if day == 0 else ("mid" if day < horizon_days - 1 else "late"), day)


def main() -> None:
    parser = argparse.ArgumentParser(description="Horizon-aware MCTS experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("regression")
    p.add_argument("--days", default="1,3,5,10,30")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--iterations", type=int, default=12)
    p.set_defaults(func=cmd_regression)

    p = sub.add_parser("matchups")
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--kind", choices=("new", "old", "both"), default="both")
    p.set_defaults(func=cmd_matchups)

    p = sub.add_parser("sweep")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--mode", choices=("A", "B", "C", "D", "E"), default="A")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("ablation")
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--iterations", type=int, default=12)
    p.set_defaults(func=cmd_ablation)

    p = sub.add_parser("modes")
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--opponent", choices=("random", "starter", "heuristic"), default="starter")
    p.set_defaults(func=cmd_modes)

    p = sub.add_parser("instrument")
    p.add_argument("--days", type=int, default=5)
    p.set_defaults(func=cmd_instrument)

    p = sub.add_parser("parallel")
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--mode", choices=("A", "B", "C", "D", "E"), default="E")
    p.add_argument("--workers", nargs="+", type=int, default=[1, 4])
    p.set_defaults(func=cmd_parallel)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
