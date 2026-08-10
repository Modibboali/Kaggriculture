"""Head-to-head experiment harness.

Plays full Kaggriculture episodes between two agents on the official Kaggle
environment and aggregates win rate, rewards, episode length, and (for MCTS
agents) search statistics. Fixed seeds make the experiments reproducible.

This is the one place the experiment layer may depend on the Kaggle
environment; the MCTS / simulator core never does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping

from .agent import Agent, MCTSAgent


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One head-to-head episode."""

    reward0: float
    reward1: float
    winner: int  # 0, 1, or -1 for a tie
    steps: int
    stats0: Mapping[str, float] | None = None
    stats1: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class MatchupResult:
    """Aggregated results of ``n`` episodes between two agents."""

    name: str
    games: int
    win_rate0: float
    avg_reward0: float
    avg_reward1: float
    avg_steps: float
    total_search_time: float = 0.0
    total_transitions: int = 0
    total_searches: int = 0
    wall_time: float = 0.0


def _winner(r0: float, r1: float) -> int:
    if r0 > r1:
        return 0
    if r1 > r0:
        return 1
    return -1


def _delta_stats(
    agent: Agent,
    before: Mapping[str, float] | None,
) -> Mapping[str, float] | None:
    """Per-episode stats for an MCTS agent (``agent.stats`` minus ``before``)."""
    if before is None:
        return None
    after = agent.stats  # type: ignore[attr-defined]
    return {key: float(after[key] - before.get(key, 0.0)) for key in after}


def play_episode(
    agent0: Agent,
    agent1: Agent,
    *,
    episode_steps: int = 720,
    seed: int = 1,
) -> EpisodeResult:
    """Play one episode; returns the result. Deterministic for a fixed seed."""
    import kaggle_environments

    # MCTSAgent.stats is cumulative across calls, so snapshot before/after to
    # return per-episode deltas (agents may be reused across episodes).
    stats0_before = agent0.stats if isinstance(agent0, MCTSAgent) else None
    stats1_before = agent1.stats if isinstance(agent1, MCTSAgent) else None

    env = kaggle_environments.make(
        "kaggriculture",
        configuration={"episodeSteps": episode_steps, "seed": seed},
        debug=False,
    )
    steps = 0
    while not env.done and steps < episode_steps:
        # The framework only injects ``step`` into player 0's raw observation;
        # mirror it into player 1's so both agents see a valid observation.
        obs0 = dict(env.state[0]["observation"])
        obs1 = dict(env.state[1]["observation"])
        if "step" not in obs1:
            obs1["step"] = obs0.get("step", steps)
        action0 = agent0.choose(obs0)
        action1 = agent1.choose(obs1)
        env.step([action0, action1])
        steps += 1

    reward0 = float(env.state[0].get("reward", 0.0))
    reward1 = float(env.state[1].get("reward", 0.0))
    stats0 = _delta_stats(agent0, stats0_before)
    stats1 = _delta_stats(agent1, stats1_before)
    return EpisodeResult(
        reward0=reward0,
        reward1=reward1,
        winner=_winner(reward0, reward1),
        steps=steps,
        stats0=stats0,
        stats1=stats1,
    )


def run_matchup(
    agent0: Agent,
    agent1: Agent,
    *,
    name: str,
    games: int = 10,
    episode_steps: int = 720,
    seed: int = 1,
) -> MatchupResult:
    """Play ``games`` episodes and aggregate the results."""
    wall = time.perf_counter()
    wins0 = 0
    total0 = 0.0
    total1 = 0.0
    total_steps = 0
    total_search_time = 0.0
    total_transitions = 0
    total_searches = 0
    for game_index in range(games):
        result = play_episode(
            agent0, agent1, episode_steps=episode_steps, seed=seed + game_index
        )
        if result.winner == 0:
            wins0 += 1
        total0 += result.reward0
        total1 += result.reward1
        total_steps += result.steps
        if result.stats0 is not None:
            total_search_time += result.stats0.get("search_time", 0.0)
            total_transitions += int(result.stats0.get("simulator_transitions", 0))
            total_searches += int(result.stats0.get("searches", 0))
    return MatchupResult(
        name=name,
        games=games,
        win_rate0=wins0 / games,
        avg_reward0=total0 / games,
        avg_reward1=total1 / games,
        avg_steps=total_steps / games,
        total_search_time=total_search_time,
        total_transitions=total_transitions,
        total_searches=total_searches,
        wall_time=time.perf_counter() - wall,
    )
