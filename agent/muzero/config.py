"""MuZero configuration (all hyper-parameters configurable, not hard-coded).

Every knob required by the task (§18) lives here with a sensible CPU-sized
default. The defaults target *correctness* on a laptop (small latent, small
batch, few simulations); scale them up via the CLI for real training.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class MuZeroConfig:
    """All MuZero hyper-parameters."""

    # --- network ---
    latent_dim: int = 64
    hidden_dim: int = 128
    unroll_steps: int = 5

    # --- search ---
    simulations: int = 25
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0  # root temperature during self-play
    temperature_threshold: int = 30  # steps from end where temp -> 0.25
    random_openings: int = 0  # uniform-random opening actions per episode

    # --- reward / value ---
    reward_scale: float = 0.001  # cash delta -> scaled reward (invertible)
    gamma: float = 1.0  # 1.0 = undiscounted return for the finite episode
    bootstrap: bool = False  # use target-net bootstrap value (else exact MC returns)
    target_update_interval: int = 0  # 0 = no target network (exact returns)

    # --- training ---
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_updates: int = 1000
    update_target_every: int = 50
    gradient_clip: float = 10.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0

    # --- self-play / replay ---
    episodes_per_generation: int = 1
    updates_per_generation: int = 50
    replay_capacity: int = 10000
    max_unroll_dynamics: int = 50

    # --- runtime ---
    seed: int = 0
    workers: int = 0  # 0 = auto (self-play processes)
    threads: int = 1  # PyTorch CPU threads
    episode_steps: int = 720

    # --- paths ---
    checkpoint_dir: str = "output/muzero/checkpoints"
    artifact_dir: str = "output/muzero/artifacts"

    # --- misc ---
    log_interval: int = 10
    verbose: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "MuZeroConfig":
        allowed = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)  # type: ignore[arg-type]
