"""The AI / planning layer: classical MCTS baseline.

The simulator is the forward model; this package adds a compact immutable
``SearchState``, an ``ActionGenerator``, the MCTS <-> simulator adapter, a
classical heuristic evaluation, rollout policies, UCT MCTS, agents, and an
experiment harness. Nothing here depends on the Kaggle environment (except the
experiment harness, which plays real episodes).
"""

from __future__ import annotations

from .action_generator import ActionGenerator
from .action_priority import ActionPriorityModel, farmer_type
from .agent import HeuristicAgent, MCTSAgent, RandomAgent, StarterAgent
from .evaluation import (
    EvaluationConfig,
    Evaluator,
    HorizonAwareEvaluator,
    evaluate,
    evaluate_horizon,
    horizon_days,
    horizon_remaining,
)
from .experiment import EpisodeResult, MatchupResult, play_episode, run_matchup
from .mcts import MCTS, MCTSConfig, MCTSNode, NoTranspositionTable, TranspositionTable
from .parallel_mcts import (
    ParallelMCTS,
    ParallelSearchResult,
    RootStat,
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
from .phase import GamePhase, longest_crop_cash_days, phase_for, phase_of, shortest_crop_cash_days
from .rollout import (
    CashConversionRolloutPolicy,
    HeuristicRolloutPolicy,
    RandomRolloutPolicy,
    RolloutPolicy,
)
from .search_state import SearchState, state_key
from .sim_experiment import (
    SimEpisodeResult,
    SimMatchupResult,
    initial_state,
    play_sim_episode,
    run_sim_matchup,
)
from .simulator_adapter import SimulatorAdapter
from .terminal import Terminal

__all__ = [
    "ActionGenerator",
    "ActionPriorityModel",
    "CashConversionRolloutPolicy",
    "EpisodeResult",
    "EvaluationConfig",
    "Evaluator",
    "GamePhase",
    "HeuristicAgent",
    "HeuristicRolloutPolicy",
    "HorizonAwareEvaluator",
    "MCTS",
    "MCTSConfig",
    "MCTSNode",
    "MCTSAgent",
    "MatchupResult",
    "NoTranspositionTable",
    "ParallelMCTS",
    "ParallelSearchResult",
    "RandomAgent",
    "RandomRolloutPolicy",
    "RolloutPolicy",
    "RootStat",
    "SearchState",
    "SimEpisodeResult",
    "SimMatchupResult",
    "SimulatorAdapter",
    "StarterAgent",
    "Terminal",
    "TranspositionTable",
    "WorkerResult",
    "WorkerTask",
    "aggregate_root_stats",
    "canonical_action_key",
    "evaluate",
    "evaluate_horizon",
    "farmer_type",
    "horizon_days",
    "horizon_remaining",
    "initial_state",
    "longest_crop_cash_days",
    "phase_for",
    "phase_of",
    "play_episode",
    "play_sim_episode",
    "run_matchup",
    "run_mcts_worker",
    "run_sim_matchup",
    "select_best_action_from_stats",
    "select_best_action_key",
    "shortest_crop_cash_days",
    "split_budget",
    "state_key",
    "worker_seed",
]
