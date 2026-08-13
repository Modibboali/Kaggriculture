"""MuZero agent for Kaggriculture (Task 15).

Canonical MuZero decomposition (representation / dynamics / reward /
prediction) with a *learned-model MCTS* that never calls the Kaggriculture
simulator to expand hypothetical search nodes. The real simulator is used
only to advance the real environment during self-play and to validate the
learned dynamics / evaluate the final agent.

Pipeline:

    run.py             CLI entry (config, workers, threads, checkpoint/resume)
    encoders.py        deterministic StateEncoder / ActionEncoder
    config.py          MuZeroConfig (all hyper-parameters, not hard-coded)
    networks.py        Representation / Dynamics / Prediction networks
    puct.py            PUCT search entirely in latent space
    replay.py          episode + replay buffer (append/sample/save/load)
    self_play.py       real-simulator self-play (latent MCTS -> real action)
    train.py           unrolled MuZero training step + loss
    learner.py         main loop: parallel self-play -> replay -> updates
    evaluate.py        deterministic evaluation vs Mode-E / Starter / Random
    synthetic_env.py   tiny synthetic env for the end-to-end smoke test
"""

from __future__ import annotations

from .config import MuZeroConfig
from .networks import DynamicsNetwork, MuZeroNetwork, PredictionNetwork, RepresentationNetwork

__all__ = [
    "DynamicsNetwork",
    "MuZeroConfig",
    "MuZeroNetwork",
    "PredictionNetwork",
    "RepresentationNetwork",
]
