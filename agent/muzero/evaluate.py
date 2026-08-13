"""Deterministic evaluation of the MuZero agent vs baselines.

Evaluation mode is strictly separate from training: no Dirichlet root noise,
temperature -> 0 (argmax visit counts), fixed search budget. Matches the MuZero
agent against:

* **Mode-E** — the classical MCTS benchmark (NOT a training dependency).
* **Starter** / **Random** — baselines.

Reports terminal cash, win rate, average reward and decision latency.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any

import numpy as np

from ..actions import TurnAction
from ..ai.agent import Agent
from ..ai.search_state import SearchState
from ..ai.sim_experiment import initial_state, run_sim_matchup
from ..ai.terminal import Terminal
from ..simulator import GameConfig, Simulator
from .config import MuZeroConfig
from .networks import MuZeroNetwork
from .puct import MuZeroAgentMCTS


class MuZeroAgent:
    """Deterministic evaluation agent wrapping the latent MCTS (Agent protocol)."""

    def __init__(
        self,
        net: MuZeroNetwork,
        config: MuZeroConfig,
        game_config: GameConfig | None = None,
        *,
        simulations: int | None = None,
    ) -> None:
        self._config = config
        self._game_config = game_config if game_config is not None else GameConfig(
            episode_steps=config.episode_steps, seed=config.seed
        )
        self._agent = MuZeroAgentMCTS(net, config, self._game_config, dirichlet=False, temperature=0.0)
        self._simulations = simulations if simulations is not None else config.simulations
        self._agent.set_temperature(0.0)
        self._search_time = 0.0
        self._searches = 0

    def select(self, game: Any, player: int) -> TurnAction:
        if game.current_player != player:
            game = replace(game, current_player=player)
        self._agent.set_simulations(self._simulations)
        t0 = time.perf_counter()
        action = self._agent.decide(SearchState(game)).action
        self._search_time += time.perf_counter() - t0
        self._searches += 1
        return action

    @property
    def stats(self) -> dict[str, float]:
        return {
            "searches": float(self._searches),
            "search_time": self._search_time,
            "simulator_transitions": 0.0,  # MCTS is latent; no sim in search
        }

    def choose(self, observation: dict[str, Any]) -> dict[str, Any]:
        from ..environment import KaggleObservationAdapter, to_kaggle_action

        game = KaggleObservationAdapter.from_observation(observation)
        return to_kaggle_action(self.select(game, game.current_player))


def _mode_e_agent(config: GameConfig, iterations: int = 12, seed: int = 0) -> Agent:
    from typing import cast

    from ..ai.action_generator import ActionGenerator
    from ..ai.action_priority import ActionPriorityModel
    from ..ai.agent import MCTSAgent
    from ..ai.evaluation import EvaluationConfig, HorizonAwareEvaluator
    from ..ai.mcts import MCTSConfig
    from ..ai.rollout import CashConversionRolloutPolicy

    eval_config = EvaluationConfig()
    model = ActionPriorityModel(config, eval_config)
    generator = ActionGenerator(config, priority_model=model, action_filter=model.filter_realizable)
    evaluator = HorizonAwareEvaluator(config, eval_config)
    rollout = CashConversionRolloutPolicy(generator, model)
    return cast(
        Agent,
        MCTSAgent(
            MCTSConfig(iterations=iterations, max_simulation_steps=12, seed=seed),
            config=config,
            rollout=rollout,
            evaluator=evaluator,
            generator=generator,
            seed=seed,
        ),
    )


def _starter_agent(config: GameConfig, seed: int = 0) -> Agent:
    from typing import cast

    from ..ai.agent import StarterAgent

    return cast(Agent, StarterAgent(config=config, seed=seed))


def _random_agent(config: GameConfig, seed: int = 0) -> Agent:
    from typing import cast

    from ..ai.agent import RandomAgent

    return cast(Agent, RandomAgent(config=config, seed=seed))


def compare(
    net: MuZeroNetwork,
    config: MuZeroConfig,
    *,
    games: int = 5,
    episode_steps: int = 720,
    simulations: int | None = None,
    iterations: int = 12,
    seed: int = 1,
) -> dict[str, dict[str, float]]:
    """MuZero vs Mode-E / Starter / Random over full episodes."""
    game_config = GameConfig(episode_steps=episode_steps, seed=seed)
    muzero = MuZeroAgent(net, config, game_config, simulations=simulations)
    opponents = {
        "mode_e": _mode_e_agent(game_config, iterations=iterations, seed=seed),
        "starter": _starter_agent(game_config, seed=seed),
        "random": _random_agent(game_config, seed=seed),
    }
    results: dict[str, dict[str, float]] = {}
    for name, opp in opponents.items():
        r = run_sim_matchup(muzero, opp, name=f"muzero_vs_{name}", games=games, config=game_config)
        latency_ms = 0.0
        if muzero.stats["searches"] > 0:
            latency_ms = muzero.stats["search_time"] / muzero.stats["searches"] * 1000.0
        results[name] = {
            "win_rate": r.win_rate0,
            "mean_reward": r.mean_reward0,
            "opponent_reward": r.mean_reward1,
            "median_reward": r.median_reward0,
            "std_reward": r.std_reward0,
            "decision_latency_ms": latency_ms,
            "games": float(r.games),
        }
        print(
            f"muzero vs {name}: win={r.win_rate0:.3f} reward={r.mean_reward0:.1f} "
            f"opp={r.mean_reward1:.1f} latency={latency_ms:.1f}ms"
        )
    return results


def run_episode_metrics(
    net: MuZeroNetwork,
    config: MuZeroConfig,
    *,
    episodes: int = 2,
    episode_steps: int = 720,
    seed: int = 1,
) -> dict[str, float]:
    """Self-play episodes under the *evaluation* agent (deterministic)."""
    game_config = GameConfig(episode_steps=episode_steps, seed=seed)
    muzero = MuZeroAgent(net, config, game_config, simulations=config.simulations)
    sim = Simulator(game_config)
    term = Terminal(game_config)
    cash_list: list[float] = []
    wins = 0
    for ep in range(episodes):
        state = initial_state(game_config)
        while not term.is_terminal(SearchState(state)):
            a0 = muzero.select(state, 0)
            a1 = muzero.select(state, 1)
            state = sim.apply(state, (a0, a1))
        c0 = float(state.players[0].farm.money)
        c1 = float(state.players[1].farm.money)
        cash_list.append(c0)
        if c0 > c1:
            wins += 1
    return {
        "mean_terminal_cash": float(np.mean(cash_list)) if cash_list else 0.0,
        "win_rate0": wins / max(1, episodes),
    }


def main() -> None:
    import argparse
    import os

    from .networks import build_network_from_checkpoint, load_checkpoint

    parser = argparse.ArgumentParser(description="Evaluate a MuZero model vs baselines")
    parser.add_argument("--model", required=True, help="checkpoint .pt or artifact .pt")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=12, help="Mode-E search budget")
    parser.add_argument("--out", default="output/muzero/eval.json")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.model)
    net = build_network_from_checkpoint(ckpt)
    extra = ckpt.get("extra", "")
    config_dict: dict[str, object] = {}
    if isinstance(extra, str) and extra:
        import json as _json

        try:
            parsed = _json.loads(extra)
            if isinstance(parsed, dict) and "config" in parsed:
                config_dict = parsed["config"]
        except Exception:
            config_dict = {}
    config = MuZeroConfig.from_dict(config_dict)
    if args.simulations is not None:
        from dataclasses import replace as _replace

        config = _replace(config, simulations=args.simulations)
    results = compare(net, config, games=args.games, episode_steps=args.episode_steps,
                      iterations=args.iterations)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
