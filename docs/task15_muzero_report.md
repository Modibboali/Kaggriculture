# Task 15 — Full MuZero Implementation

**Milestone:** the first complete MuZero-style agent for Kaggriculture —
learned representation, dynamics, reward, policy and value from self-play,
with PUCT search operating entirely inside the learned latent dynamics model.
**Status:** Complete (working closed loop; baseline benchmark comparison at
smoke scale).
**Date:** 2026-08-13

> The learned-policy-from-teacher stage (Task 14) was intentionally skipped.
> Mode-E classical MCTS is the **benchmark only** — it is NOT the training
> teacher. The simulator remains the authoritative real environment for
> self-play and evaluation, but **never** for expanding hypothetical search
> nodes.

---

## 1. Architecture

```
observation o_0 ──RepresentationNetwork f_θ──▶ h_0
h_k ──PredictionNetwork──▶ (policy over current candidates, value v_k)
(h_k, a_k) ──DynamicsNetwork g_θ──▶ (h_{k+1}, reward r_{k+1})
```

The decomposition is explicit — three separate networks composed into
`MuZeroNetwork` (never one giant model).

| Component | Module | Mapping |
|---|---|---|
| Representation | `RepresentationNetwork` | `o_0 -> h_0` (139-dim obs -> latent) |
| Dynamics | `DynamicsNetwork` | `(h_k, a_k) -> (h_{k+1}, r_{k+1})` |
| Prediction | `PredictionNetwork` | `h_k -> (p_k, v_k)` |

* **Input representation** — the deterministic `StateEncoder` (139 features,
  decision-time only: time/horizon, economy, seeds, shed+farmer inventory,
  land, tile composition, crop lifecycle, animal state, market prices +
  inventory, town shops, phase, derived economic values). No future state,
  no teacher action, no opponent private state. Raw `GameState` objects are
  never fed to the network.
* **Action representation** — the deterministic `ActionEncoder` (60 features)
  produces a per-candidate embedding. MuZero uses **candidate-conditioned**
  action representations — there is no massive fixed global action
  vocabulary.
* **Variable action set** — the policy softmax is over the *current* candidate
  set (masked softmax over candidate embeddings). The same mechanism is used
  in training, self-play, MCTS and inference. In the latent tree, the root's
  real candidate set is the per-search action vocabulary (the only faithful
  choice given the no-simulator invariant).
* **PUCT** — `PUCTNode` stores latent states + search statistics only.
  `score(a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`, with
  learned prior P, learned value, learned reward, learned latent dynamics.
* **Replay** — `ReplayBuffer` of complete episodes in numeric arrays (obs,
  candidate embeddings, action indices, policy targets, rewards, exact
  returns, players). Uniform sampling; no prioritized replay.
* **Self-play** — `MuZeroSelfPlay`: real simulator advances the real state;
  the latent MCTS picks the real action; reward = cash delta.
* **Training** — unrolled MuZero loss (policy + value + reward), exact
  Monte-Carlo value targets, Adam, gradient clipping, optional periodic-copy
  target network.

### Reward design (dense cash delta)

```
reward_t = (cash_{t+1} - cash_t) * reward_scale
```

Because the scale is a fixed invertible multiplier and `γ = 1` (undiscounted
for the finite episode), the sum over an episode telescopes:

```
Σ_t reward_t / reward_scale = cash_T - cash_0 = final_cash - initial_cash
```

`initial_cash` is fixed, so maximizing the dense reward sum is exactly
equivalent to maximizing the official terminal metric (**final cash**). The
official objective is untouched; the dense reward is only the learning/search
signal. This equivalence is verified by tests (`test_reward_equivalence`).
At terminal there is no future reward (value target = stored exact return).

### Value / policy targets

* **Value target** — exact Monte-Carlo return from the stored episode:
  `v_target(t) = Σ_{i>=t} γ^(i-t) r_i`. Because complete episodes are stored,
  no bootstrap / target network is required; a periodic-copy target net is
  available via config (`target_update_interval > 0`).
* **Policy target** — MuZero's *own* MCTS root visit distribution
  `π(a) = N(a)/ΣN(b)` at self-play time. **Mode-E is never a policy target.**

### Search stats (smoke, real game, untrained network)

| Quantity | value |
|---|---|
| simulations | 10 |
| c_puct | 1.5 |
| average depth | ~2.6 (10 sims, 19 candidates) |
| root policy entropy (untrained) | ~1.56 / 2.14 |
| decision latency | ~32 ms/decision (10 sims, latent 32) |

---

## 2. Synthetic end-to-end test (Task 22)

A tiny environment independent of Kaggriculture: state 0, actions A (reward
10) and B (reward 0), 4-step episodes. After training the **root policy
prefers A with P(A) = 1.0**, and the losses confirm the loop:

| metric | start | end |
|---|---|---|
| total loss | 2.187 | 0.671 |
| policy loss | 0.346 | 0.108 |
| value loss | 1.304 | 0.560 |
| reward loss | 0.537 | 0.003 |
| policy entropy | 0.693 | 0.205 |

The reward loss collapsing to ~0 proves the dynamics network learned the
reward structure; the entropy drop proves the policy sharpened toward A. This
demonstrates the full closed loop: model → self-play → PUCT → replay →
training → improved model.

---

## 3. Kaggriculture smoke training (Task 23)

Real self-play on 2-day (48-step) episodes, latent 32 / hidden 64, 10
simulations, 6 episodes, 45 updates:

| metric | start | end |
|---|---|---|
| total loss | 1.82 | 0.60 |
| value loss | 0.33 | 0.36 |
| reward loss | 1.26 | **0.002** |
| policy entropy | 2.14 | 2.12 |

* No NaNs; all rewards / values / dynamics outputs finite.
* Self-play completes; replay sampling works; checkpoint + resume verified
  (step 45 → 55, replay restored).
* **Reward prediction is learned** (reward loss → 0.002): the dynamics
  network predicts the per-step cash delta well.
* Policy / value remain early (only 6 tiny episodes), so the *evaluation*
  agent degenerates to **PASS** (all 120 steps, final cash unchanged at 3000)
  — the classic cold-start behavior: a uniform prior + index-0 tie-break in
  the argmax selects candidate 0 (PASS). It "wins" vs baselines only because
  the baselines spend money and end lower.

---

## 4. Model

| quantity | smoke | medium |
|---|---|---|
| latent dim | 32 | 64 |
| hidden dim | 64 | 128 |
| parameters | 43,971 | 141,187 |
| artifact size | < 1 MiB | < 1 MiB |

Deployment artifact contains **only** the MuZero model + encoders + config
(no optimizer / replay / training logs) — §31 artifact separation.

**Parallel self-play (§27) verified** with `--workers 2` (processes on
Windows; each worker owns its model copy + MCTS tree + simulator state; model
sent as a per-batch snapshot). Episodes aggregate into the parent replay.
Checkpoint / resume verified end-to-end (model, optimizer, step, replay,
config).

---

## 5. Comparison vs baselines (smoke scale, §21)

MuZero vs Mode-E / Starter / Random, 3 games, 120-step episodes (smoke-trained
model):

| opponent | win | reward (MuZero) | opponent reward |
|---|---|---|---|
| Mode-E | 1.000 | 3000.0 | 2996.3 |
| Starter | 1.000 | 3000.0 | 1996.0 |
| Random | 1.000 | 3000.0 | 0.7 |

The reward is *exactly* the starting cash (3000) — the untrained agent does
nothing (PASS). It does **not** beat Mode-E on strategy; it only preserves
its bankroll. This is reported honestly: at smoke scale the loop works but
the policy has not yet learned strategy.

### Medium comparison (16 episodes of 144 steps, 320 updates, latent 64)

MuZero vs Mode-E / Starter / Random, 3 games, 144-step episodes (the
training horizon):

| opponent | win | reward (MuZero) | opponent reward | latency |
|---|---|---|---|---|
| Mode-E | 0.000 | 2700.0 | 3009.3 | 46 ms |
| Starter | 1.000 | 2700.0 | 1846.0 | 46 ms |
| Random | 1.000 | 2700.0 | 1.0 | 45 ms |

After 16 short episodes the agent is no longer a pure pass: it spends ~300
(ends at 2700 < 3000 start) but does not yet convert assets back to cash, so
it still **loses to Mode-E** (2700 vs 3009) and only beats Starter / Random by
losing less. Value / policy heads are still near the cold start (policy
entropy ≈ 2.3, close to uniform over ~9 candidates) while the **reward head
is well learned** (reward loss ≈ 0.0004). This is the honest "the loop works,
strategy learning needs far more data" result.

---

## 6. Stability

* **NaNs** — none observed in synthetic, smoke or medium runs.
* **Exploding values** — gradient clipping (`--gradient-clip 10.0`); reward
  scaling keeps cash deltas O(1); values bounded by finite episodes.
* **Replay** — no issues; save/load round-trips verified by tests.
* **Checkpoint / resume** — verified end-to-end (model, optimizer, step,
  replay, config).
* **Representation consistency** — identical states encode identically
  (deterministic encoders + eval-mode network); different strategic states
  do not collapse (tested); geometry not yet optimized (by design).

---

## 7. Performance (local: 8-core / 15.9 GB / torch 2.13 CPU)

| quantity | value |
|---|---|
| simulator transitions/sec | ~2,500 |
| self-play generation | 2 × 144-step episodes @ 25 sims ≈ 55 s (≈ 26 s/episode) |
| decision latency | ~32 ms (10 sims, latent 32) / ~46 ms (25 sims, latent 64) |
| training updates | 40 updates (batch 32, latent 64) in ~2 s |
| medium run wall | 16 episodes + 320 updates in 389 s |

---

## 8. Recommendation (honest)

* **Has MuZero learned anything useful?** Yes at the mechanics level: the
  dynamics network learns to predict cash-delta rewards (reward loss ≈ 0),
  and the synthetic environment proves the full loop learns a correct action
  preference. No: at the *strategy* level on Kaggriculture with the tiny
  smoke budget, the policy is still near-uniform (cold-start PASS behavior).
* **Does it beat Mode-E?** Not yet. Mode-E's hand-crafted priors + realized
  cash-conversion search remain far stronger with only minutes of MuZero
  training. MuZero must train for many more episodes before it is a fair
  benchmark opponent.
* **Current bottleneck.** Data volume, confirmed by the medium run: after 16
  short episodes the reward head is converged but the policy/value heads are
  still near the uniform-prior cold start (policy entropy ≈ 2.3 ≈ uniform over
  ~9 candidates; agent spends but cannot convert assets back to cash). The
  reward head converges quickly; the policy/value need many more self-play
  episodes (and ideally longer, horizon-rich episodes) to escape the cold
  start.
* **Next step.** Before increasing model size or search budget, **train for
  more episodes** (the pipeline supports 720-step games, parallel workers,
  checkpoint/resume, and a cloud-CPU config). The first experimental grid
  (§34) is already wired: latent 64 / hidden 128 / unroll 5 / sims 25. Only
  after the value/policy meaningfully beat Starter should we consider larger
  latents, stronger MCTS, or representation improvements. Latent dynamics →
  PUCT → value-learning is all functioning; the path to beating Mode-E is
  data, not architecture.

---

## 9. Files

| Path | Purpose |
|---|---|
| `agent/muzero/encoders.py` | StateEncoder (139) / ActionEncoder (60) |
| `agent/muzero/config.py` | `MuZeroConfig` (all hyper-parameters) |
| `agent/muzero/networks.py` | Representation / Dynamics / Prediction / MuZeroNetwork |
| `agent/muzero/puct.py` | PUCT search in latent space (no simulator invariant) |
| `agent/muzero/replay.py` | Episode + ReplayBuffer (append/sample/save/load) |
| `agent/muzero/self_play.py` | real-simulator self-play engine |
| `agent/muzero/train.py` | unrolled MuZero loss + train step |
| `agent/muzero/learner.py` | parallel self-play → replay → updates → checkpoint |
| `agent/muzero/evaluate.py` | deterministic eval vs Mode-E/Starter/Random |
| `agent/muzero/synthetic_env.py` | synthetic end-to-end environment |
| `agent/muzero/metrics.py` | metrics accumulator |
| `agent/muzero/run.py` | CLI (config, workers, threads, resume) |
| `tests/muzero/*.py` | 36 tests (networks, PUCT, replay, train, self-play, synthetic) |

Full `mypy --strict` clean and full pytest green (see test run).
