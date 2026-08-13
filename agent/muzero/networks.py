"""MuZero networks: representation / dynamics / prediction.

The decomposition is kept explicit (three separate networks composed into
:class:`MuZeroNetwork`) — never one giant model.

* :class:`RepresentationNetwork` — ``h_0 = f_theta(o_0)`` (obs -> latent).
* :class:`DynamicsNetwork` — ``h_{k+1}, r_{k+1} = g_theta(h_k, a_k)``
  (latent + action embedding -> next latent + scalar reward).
* :class:`PredictionNetwork` — ``p_k, v_k = h_theta(h_k)`` (latent -> value,
  plus per-candidate policy logits scored against the *current* candidate
  action embeddings — no fixed global action vocabulary).

All networks are small MLPs with configurable ``latent_dim`` / ``hidden_dim``.
"""

from __future__ import annotations

import json
import os
from typing import cast

import torch
from torch import nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class RepresentationNetwork(nn.Module):
    """``o_0 -> h_0``."""

    def __init__(self, obs_dim: int, hidden_dim: int, latent_dim: int, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = _mlp(obs_dim, hidden_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """``obs (B, D_obs) -> latent (B, D_latent)``."""
        return cast(torch.Tensor, self.net(obs))


class DynamicsNetwork(nn.Module):
    """``(h_k, a_k) -> (h_{k+1}, r_{k+1})``."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.latent_head = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor, action_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``latent (B, D) + action_emb (B, Da) -> (next (B, D), reward (B,))``."""
        x = torch.cat([latent, action_emb], dim=-1)
        h = cast(torch.Tensor, self.net(x))
        return cast(torch.Tensor, self.latent_head(h)), cast(torch.Tensor, self.reward_head(h).squeeze(-1))


class PredictionNetwork(nn.Module):
    """``h_k -> (p_k over candidates, v_k)``."""

    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def value(self, latent: torch.Tensor) -> torch.Tensor:
        """``latent (B, D) -> value (B,)``."""
        return cast(torch.Tensor, self.value_net(latent).squeeze(-1))

    def policy_logits(self, latent: torch.Tensor, action_embs: torch.Tensor) -> torch.Tensor:
        """Score the *current* candidate set.

        ``latent (B, D)`` and ``action_embs (B, N, Da) -> logits (B, N)``.
        Softmax is taken over the candidate axis by the caller. This is the
        same candidate-conditioned mechanism in training, self-play, MCTS and
        inference — there is no fixed global action softmax.
        """
        h = latent.unsqueeze(1).expand(-1, action_embs.size(1), -1)  # (B, N, D)
        x = torch.cat([h, action_embs], dim=-1)  # (B, N, D + Da)
        return cast(torch.Tensor, self.policy_net(x).squeeze(-1))  # (B, N)

    def forward(self, latent: torch.Tensor, action_embs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy_logits(latent, action_embs), self.value(latent)


class MuZeroNetwork(nn.Module):
    """Composes representation + dynamics + prediction into one checkpointable unit."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.representation = RepresentationNetwork(obs_dim, hidden_dim, latent_dim, seed=seed)
        self.dynamics = DynamicsNetwork(latent_dim, action_dim, hidden_dim, seed=seed)
        self.prediction = PredictionNetwork(latent_dim, hidden_dim, action_dim, seed=seed)

    def initial_inference(
        self, obs: torch.Tensor, action_embs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(o_0, candidate embeddings) -> (latent, policy logits, value)``."""
        latent = self.representation(obs)
        logits = self.prediction.policy_logits(latent, action_embs)
        value = self.prediction.value(latent)
        return cast(torch.Tensor, latent), logits, value

    def policy_at(self, latent: torch.Tensor, action_embs: torch.Tensor) -> torch.Tensor:
        return self.prediction.policy_logits(latent, action_embs)

    def value_at(self, latent: torch.Tensor) -> torch.Tensor:
        return self.prediction.value(latent)

    def recurrent_inference(
        self, latent: torch.Tensor, action_embs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(h_k, a_k) -> (h_{k+1}, r_{k+1})``."""
        return cast(tuple[torch.Tensor, torch.Tensor], self.dynamics(latent, action_embs))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def copy_target(self) -> "MuZeroNetwork":
        """A deep copy for the periodic target network."""
        import copy

        target = copy.deepcopy(self)
        for p in target.parameters():
            p.requires_grad_(False)
        return target

    def sync_target(self, target: "MuZeroNetwork") -> None:
        """Copy online weights into ``target`` (hard periodic update)."""
        with torch.no_grad():
            for tp, op in zip(target.parameters(), self.parameters()):
                tp.copy_(op)


def save_checkpoint(
    net: MuZeroNetwork,
    path: str,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    extra: dict[str, object] | None = None,
) -> None:
    """Save model (+ optional optimizer + step + metadata) to ``path``."""
    payload: dict[str, object] = {
        "model": net.state_dict(),
        "obs_dim": net.obs_dim,
        "action_dim": net.action_dim,
        "latent_dim": net.latent_dim,
        "hidden_dim": net.hidden_dim,
        "step": step,
        "extra": json.dumps(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict[str, object]:
    return cast(dict[str, object], torch.load(path, map_location=map_location))


def build_network_from_checkpoint(ckpt: dict[str, object]) -> MuZeroNetwork:
    from typing import cast

    net = MuZeroNetwork(
        cast(int, ckpt["obs_dim"]),
        cast(int, ckpt["action_dim"]),
        latent_dim=cast(int, ckpt["latent_dim"]),
        hidden_dim=cast(int, ckpt["hidden_dim"]),
        seed=0,
    )
    net.load_state_dict(cast(dict[str, object], ckpt["model"]))
    return net
