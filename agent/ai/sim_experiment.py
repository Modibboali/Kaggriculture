"""Simulator-based head-to-head experiment harness (Kaggle-free).

The real Kaggle engine runs at only a few hundred steps/sec, which makes
100-game matchups impractical for a searching agent. Because the verified
simulator is the single source of truth for transitions (and is ~20x faster),
this harness plays complete head-to-head episodes *entirely in the simulator*:
each agent acts through ``select(game, player)`` (no observation parsing per
turn) and the simulator advances both players simultaneously.

The initial state is a faithful copy of the official environment's initial
observation (10x10 board, NW quadrant unlocked, farmer at the shed centre,
market at base prices), and the per-game RNG seed flows through ``GameConfig``
exactly as in the real engine. This harness never imports the Kaggle
environment.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, replace
from typing import Mapping

from ..simulator import GameConfig, Simulator
from ..simulator.game_config import DEFAULT_MARKET_PARAMS, MARKET_I0, PRODUCT_ITEMS
from ..state import (
    EmptyTile,
    Farm,
    GameState,
    Inventory,
    LockedTile,
    Market,
    PlayerState,
    Position,
    Quadrant,
    Seeds,
    Town,
    Worker,
)
from .agent import Agent, MCTSAgent
from .search_state import SearchState
from .terminal import Terminal


def initial_state(config: GameConfig) -> GameState:
    """A faithful copy of the official initial observation.

    On a 10x10 board the farmer starts at the centre of the unlocked NW
    quadrant (4, 4); the remaining quadrants are locked, the shed and seed
    stock are empty, and the market starts at its base prices.
    """
    size = config.board_size
    half = size // 2
    tiles: tuple[tuple[object, ...], ...] = tuple(
        tuple(EmptyTile() if (x < half and y < half) else LockedTile() for x in range(size))
        for y in range(size)
    )
    farmer = Worker(0, Position(half - 1, half - 1), Inventory.empty(), True)
    farm = Farm(
        money=config.starting_money,
        tiles=tiles,  # type: ignore[arg-type]
        farmer=farmer,
        workers=(),
        unlocked_quadrants=frozenset({Quadrant.NW}),
        hires_today=0,
    )
    market = Market(
        Inventory({item: MARKET_I0 for item in PRODUCT_ITEMS}),
        {item: params.base for item, params in DEFAULT_MARKET_PARAMS.items()},
    )
    player = PlayerState(farm, Inventory.empty(), Seeds.empty(), (farmer,))
    return GameState(
        day=0,
        hour=0,
        step=0,
        market=market,
        town=Town(frozenset()),
        players=(player, player),
        current_player=0,
    )


@dataclass(frozen=True, slots=True)
class SimEpisodeResult:
    """One head-to-head episode played entirely in the simulator."""

    reward0: float
    reward1: float
    winner: int  # 0, 1, or -1 for a tie
    steps: int
    stats0: Mapping[str, float] | None = None
    stats1: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class SimMatchupResult:
    """Aggregated statistics over ``games`` simulator episodes."""

    name: str
    games: int
    episode_steps: int
    win_rate0: float
    mean_reward0: float
    median_reward0: float
    std_reward0: float
    mean_reward1: float
    median_reward1: float
    std_reward1: float
    mean_steps: float
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
    """Per-episode MCTS stats (``agent.stats`` minus a pre-episode snapshot)."""
    if before is None:
        return None
    after = agent.stats  # type: ignore[attr-defined]
    return {key: float(after[key] - before.get(key, 0.0)) for key in after}


def play_sim_episode(
    agent0: Agent,
    agent1: Agent,
    *,
    config: GameConfig,
    seed: int = 1,
) -> SimEpisodeResult:
    """Play one episode in the simulator; deterministic for a fixed seed."""
    game_config = replace(config, seed=seed)
    simulator = Simulator(game_config)
    terminal = Terminal(game_config)
    state = initial_state(game_config)

    stats0_before = agent0.stats if isinstance(agent0, MCTSAgent) else None
    stats1_before = agent1.stats if isinstance(agent1, MCTSAgent) else None

    steps = 0
    while not terminal.is_terminal(SearchState(state)) and steps < game_config.episode_steps:
        action0 = agent0.select(state, 0)
        action1 = agent1.select(state, 1)
        state = simulator.apply(state, (action0, action1))
        steps += 1

    reward0 = float(state.players[0].farm.money)
    reward1 = float(state.players[1].farm.money)
    return SimEpisodeResult(
        reward0=reward0,
        reward1=reward1,
        winner=_winner(reward0, reward1),
        steps=steps,
        stats0=_delta_stats(agent0, stats0_before),
        stats1=_delta_stats(agent1, stats1_before),
    )


def run_sim_matchup(
    agent0: Agent,
    agent1: Agent,
    *,
    name: str,
    games: int = 100,
    config: GameConfig,
    seed_start: int = 1,
) -> SimMatchupResult:
    """Play ``games`` episodes with varied seeds and aggregate statistics."""
    wall = time.perf_counter()
    rewards0: list[float] = []
    rewards1: list[float] = []
    wins0 = 0
    total_steps = 0
    total_search_time = 0.0
    total_transitions = 0
    total_searches = 0
    for game_index in range(games):
        result = play_sim_episode(
            agent0, agent1, config=config, seed=seed_start + game_index
        )
        rewards0.append(result.reward0)
        rewards1.append(result.reward1)
        if result.winner == 0:
            wins0 += 1
        total_steps += result.steps
        if result.stats0 is not None:
            total_search_time += result.stats0.get("search_time", 0.0)
            total_transitions += int(result.stats0.get("simulator_transitions", 0))
            total_searches += int(result.stats0.get("searches", 0))
    return SimMatchupResult(
        name=name,
        games=games,
        episode_steps=config.episode_steps,
        win_rate0=wins0 / games,
        mean_reward0=statistics.fmean(rewards0),
        median_reward0=statistics.median(rewards0),
        std_reward0=statistics.pstdev(rewards0) if len(rewards0) > 1 else 0.0,
        mean_reward1=statistics.fmean(rewards1),
        median_reward1=statistics.median(rewards1),
        std_reward1=statistics.pstdev(rewards1) if len(rewards1) > 1 else 0.0,
        mean_steps=total_steps / games,
        total_search_time=total_search_time,
        total_transitions=total_transitions,
        total_searches=total_searches,
        wall_time=time.perf_counter() - wall,
    )
