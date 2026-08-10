"""The AI / planning layer: classical MCTS baseline.

The simulator is the forward model; this package adds a compact immutable
``SearchState``, an ``ActionGenerator``, the MCTS <-> simulator adapter, a
classical heuristic evaluation, rollout policies, UCT MCTS, agents, and an
experiment harness. Nothing here depends on the Kaggle environment (except the
experiment harness, which plays real episodes).
"""

from __future__ import annotations

from .action_generator import ActionGenerator
from .agent import HeuristicAgent, MCTSAgent, RandomAgent, StarterAgent
from .evaluation import EvaluationConfig, Evaluator
from .experiment import EpisodeResult, MatchupResult, play_episode, run_matchup
from .mcts import MCTS, MCTSConfig, MCTSNode, NoTranspositionTable, TranspositionTable
from .rollout import HeuristicRolloutPolicy, RandomRolloutPolicy, RolloutPolicy
from .search_state import SearchState, state_key
from .simulator_adapter import SimulatorAdapter
from .terminal import Terminal

__all__ = [
    "ActionGenerator",
    "EpisodeResult",
    "EvaluationConfig",
    "Evaluator",
    "HeuristicAgent",
    "HeuristicRolloutPolicy",
    "MCTS",
    "MCTSConfig",
    "MCTSNode",
    "MCTSAgent",
    "MatchupResult",
    "NoTranspositionTable",
    "RandomAgent",
    "RandomRolloutPolicy",
    "RolloutPolicy",
    "SearchState",
    "SimulatorAdapter",
    "StarterAgent",
    "Terminal",
    "TranspositionTable",
    "play_episode",
    "run_matchup",
    "state_key",
]
