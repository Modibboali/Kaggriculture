"""MuZero training CLI (CPU / cloud).

Usage::

    python -m agent.muzero.run --latent-dim 64 --hidden-dim 128 --unroll-steps 5 \\
        --simulations 25 --episodes-per-generation 2 --updates-per-generation 50 \\
        --num-updates 1000 --workers 2 --threads 1 --episode-steps 720
    python -m agent.muzero.run ... --resume

Prints host info (CPU / RAM / architecture / PyTorch / threads / workers),
avoids oversubscription (threads x workers bounded by CPU count), checkpoints
periodically, and exports a deployment-only artifact (model + encoders +
config) at the end.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys

import torch

from ..simulator import GameConfig
from .config import MuZeroConfig
from .learner import MuZeroLearner


def _host_info(config: MuZeroConfig) -> None:
    try:
        import psutil

        ram = f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
    except Exception:
        ram = "n/a"
    print(
        f"[host] python={sys.version.split()[0]} os={platform.system()} "
        f"arch={platform.machine()} cpu={os.cpu_count()} ram={ram} "
        f"torch={torch.__version__} threads={config.threads} workers={config.workers}"
    )


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-threshold", type=int, default=30)
    parser.add_argument("--random-openings", type=int, default=0)
    parser.add_argument("--reward-scale", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-updates", type=int, default=1000)
    parser.add_argument("--episodes-per-generation", type=int, default=1)
    parser.add_argument("--updates-per-generation", type=int, default=50)
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0, help="0 = auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--checkpoint-dir", default="output/muzero/checkpoints")
    parser.add_argument("--artifact-dir", default="output/muzero/artifacts")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")


def _config_from_args(args: argparse.Namespace) -> MuZeroConfig:
    return MuZeroConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        unroll_steps=args.unroll_steps,
        simulations=args.simulations,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        temperature=args.temperature,
        temperature_threshold=args.temperature_threshold,
        random_openings=args.random_openings,
        reward_scale=args.reward_scale,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_updates=args.num_updates,
        episodes_per_generation=args.episodes_per_generation,
        updates_per_generation=args.updates_per_generation,
        replay_capacity=args.replay_capacity,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        workers=args.workers,
        threads=args.threads,
        episode_steps=args.episode_steps,
        checkpoint_dir=args.checkpoint_dir,
        artifact_dir=args.artifact_dir,
        log_interval=args.log_interval,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MuZero training (CPU/cloud)")
    _add_config_args(parser)
    args = parser.parse_args()

    config = _config_from_args(args)
    # Avoid oversubscription on cloud CPU: bound (threads * workers) to CPUs.
    from dataclasses import replace

    cpus = os.cpu_count() or 2
    if config.workers * max(1, config.threads) > cpus:
        config = replace(config, workers=max(1, cpus // max(1, config.threads)))
    torch.set_num_threads(config.threads)
    _host_info(config)

    learner = MuZeroLearner(config, GameConfig(episode_steps=config.episode_steps, seed=config.seed))
    if args.resume:
        learner.resume()
    learner.train()
    artifact = learner.export_artifact()
    print(f"[artifact] {artifact}")


if __name__ == "__main__":
    main()
