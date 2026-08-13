"""Unrolled MuZero training step.

Given a batch of sampled :class:`~agent.muzero.replay.TrainingPosition`
positions, the loss is the canonical MuZero decomposition:

    L = policy_loss + value_loss + reward_loss

* ``h_0 = representation(obs_t)``, then unroll ``k = 0..K-1``
  ``h_{k+1}, r_{k+1} = dynamics(h_k, a_{t+k})``.
* At every latent state the prediction head produces policy logits over that
  step's *current* candidate set and a scalar value.
* Policy targets are the stored self-play visit distributions; reward targets
  are the stored cash-delta rewards; value targets are the stored exact
  Monte-Carlo future returns (undiscounted by default).

The variable candidate set is handled by padding each step's candidate
embeddings to the batch max and masking the softmax — identical to the
candidate-conditioned mechanism used in MCTS and inference.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import MuZeroConfig
from .networks import MuZeroNetwork
from .replay import TrainingPosition


def _pad_candidates(
    cand_sets: list[np.ndarray], targets: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad ragged candidate sets -> (embs, mask, target), all (B, Nmax, ...)."""
    n_max = max(c.shape[0] for c in cand_sets)
    da = cand_sets[0].shape[1]
    b = len(cand_sets)
    embs = np.zeros((b, n_max, da), dtype=np.float32)
    mask = np.zeros((b, n_max), dtype=bool)
    tgt = np.zeros((b, n_max), dtype=np.float32)
    for i, (c, t) in enumerate(zip(cand_sets, targets)):
        n = c.shape[0]
        embs[i, :n] = c
        mask[i, :n] = True
        tgt[i, :n] = t
    return embs, mask, tgt


def _masked_policy_loss(logits: torch.Tensor, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Masked softmax CE over the candidate axis (0 * -inf slots avoided)."""
    logits_masked = logits.clone()
    logits_masked[~mask] = float("-inf")
    logp = torch.log_softmax(logits_masked, dim=-1)
    neg = torch.where(mask, target * logp, torch.zeros_like(target))
    per_example = -(neg.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1))
    return per_example.mean()


def _policy_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits_masked = logits.clone()
    logits_masked[~mask] = float("-inf")
    p = torch.softmax(logits_masked, dim=-1)
    eps = 1e-8
    ent = -(p * torch.log(p + eps)).sum(dim=-1)
    return (ent * mask.sum(dim=-1)).sum() / mask.sum(dim=-1).clamp(min=1).sum().clamp(min=1)


def compute_batch_loss(
    net: MuZeroNetwork,
    positions: list[TrainingPosition],
    config: MuZeroConfig,
) -> dict[str, torch.Tensor]:
    """The unrolled MuZero loss for one batch (all tensors CPU)."""
    b = len(positions)
    max_unroll = max(p.num_unroll for p in positions)

    # ---- step 0: initial inference ----
    obs = torch.from_numpy(np.stack([p.obs for p in positions]))  # (B, Do)
    embs0, mask0, tgt0 = _pad_candidates([p.cand_embs[0] for p in positions], [p.policy_targets[0] for p in positions])
    latent, logits0, value0 = net.initial_inference(
        obs, torch.from_numpy(embs0)
    )
    policy_loss = _masked_policy_loss(
        logits0, torch.from_numpy(mask0), torch.from_numpy(tgt0)
    )
    value_loss = _mse(
        value0, torch.tensor([p.value_targets[0] for p in positions], dtype=torch.float32)
    )
    reward_loss = torch.zeros((), dtype=torch.float32)
    entropy = _policy_entropy(logits0, torch.from_numpy(mask0))

    steps = 1  # number of prediction steps that contributed policy/value loss
    # ---- unrolled dynamics ----
    for k in range(max_unroll):
        active = [i for i in range(b) if k < positions[i].num_unroll]
        if not active:
            break
        a_embs = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(positions[i].cand_embs[k][positions[i].action_indices[k]], dtype=np.float32)
                )
                for i in active
            ]
        )  # (n_active, Da)
        lat_next, r_pred = net.recurrent_inference(latent[active], a_embs)
        r_tgt = torch.tensor([positions[i].reward_targets[k] for i in active], dtype=torch.float32)
        reward_loss = reward_loss + _mse(r_pred, r_tgt)

        embs_k1, mask_k1, tgt_k1 = _pad_candidates(
            [positions[i].cand_embs[k + 1] for i in active],
            [positions[i].policy_targets[k + 1] for i in active],
        )
        logits_k1 = net.policy_at(lat_next, torch.from_numpy(embs_k1))
        v_k1 = net.value_at(lat_next)
        policy_loss = policy_loss + _masked_policy_loss(
            logits_k1, torch.from_numpy(mask_k1), torch.from_numpy(tgt_k1)
        )
        v_tgt = torch.tensor([positions[i].value_targets[k + 1] for i in active], dtype=torch.float32)
        value_loss = value_loss + _mse(v_k1, v_tgt)
        entropy = entropy + _policy_entropy(logits_k1, torch.from_numpy(mask_k1))
        steps += 1

        # advance the full-batch latent so the next step's active slice is right
        latent = latent.clone()
        latent[active] = lat_next

    policy_loss = policy_loss / steps
    value_loss = value_loss / steps
    reward_loss = reward_loss / max(1, max_unroll)
    entropy = entropy / steps

    total = (
        config.policy_loss_weight * policy_loss
        + config.value_loss_weight * value_loss
        + config.reward_loss_weight * reward_loss
    )
    return {
        "total": total,
        "policy": policy_loss,
        "value": value_loss,
        "reward": reward_loss,
        "entropy": entropy,
    }


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)


def train_step(
    net: MuZeroNetwork,
    optimizer: torch.optim.Optimizer,
    positions: list[TrainingPosition],
    config: MuZeroConfig,
) -> dict[str, float]:
    """One optimizer step; returns float metrics."""
    net.train()
    optimizer.zero_grad()
    losses = compute_batch_loss(net, positions, config)
    losses["total"].backward()  # type: ignore[no-untyped-call]
    torch.nn.utils.clip_grad_norm_(net.parameters(), config.gradient_clip)
    optimizer.step()
    return {k: float(v.detach()) for k, v in losses.items()}
