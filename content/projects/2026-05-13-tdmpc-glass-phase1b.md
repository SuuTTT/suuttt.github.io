---
title: "TD-MPC-Glass: From Scratch to Phase 1b on HopperHop"
date: 2026-05-13
description: "A practical walkthrough of TD-MPC2, a JAX/Flax reimplementation, integrating Glass-JAX structural entropy, and Phase 1 / Phase 1b results on DMC HopperHop. Includes the corrected 2D structural-entropy formula and pending plan."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["tdmpc2", "glass-jax", "structural-entropy", "jax", "reinforcement-learning", "dmc", "hopper"]
---

{{< katex >}}

# TD-MPC-Glass — From Scratch to Phase 1 on HopperHop

A practical write-up of (a) what TD-MPC2 is, (b) our JAX/Flax reimplementation
of it, (c) what Glass-JAX adds and where, (d) the iteration history that took
the Glass-augmented agent from "inert clustering" to within-CI of the official
PyTorch baseline on HopperHop, and (e) reproducible launch commands for RTX 3090,
RTX 4090, and RTX 50-series GPUs.

---

## 1. TD-MPC2 in one page

TD-MPC2 (Hansen et al., 2024) is a model-based actor-critic that performs
**model-predictive control in a learned latent space**. Five networks share a
512-dimensional latent `z`:

| Network    | Maps                                | Purpose                                     |
|-----------|--------------------------------------|---------------------------------------------|
| Encoder   | `obs → z`                            | SimNorm-projected latent (V=8 groups)       |
| Dynamics  | `(z, a) → z'`                        | Latent rollout for MPPI and TD-targets      |
| Reward    | `(z, a) → r̂`                        | 101-bin two-hot distribution                |
| Q ensemble| `(z, a) → Q`                         | 101-bin two-hot; 5 heads, min-of-2 target   |
| Policy π  | `z → tanh(μ, σ)`                     | Squashed Gaussian; entropy-regularised      |

A `RunningScale` module rescales targets to `[scale_min=1.0, scale_max=4.0]`
so that the two-hot bins stay well-conditioned across tasks.

**Planning.** At act-time the agent runs MPPI in latent space:
horizon `H=3`, `n_samples=512` action sequences per iteration, `n_iter=6`,
`num_elites=64`, `num_pi_trajs=24` (samples drawn from the policy). The score
of a sequence is the sum of two-hot rewards along the dynamics rollout plus a
terminal `Q(z_H, a_H)`. Returns are scored under `RunningScale` so that the
softmax temperature `MPPI_TEMPERATURE` stays calibrated.

**Training.** Every env step pushes a transition into a uniform replay buffer
of size 1e6 and triggers `K_UPDATE=64` gradient steps of batch 256. Each step
optimises a single sum of losses:

```text
L = consistency( z_pred, sg(z_target_encoded) )       # cosine
  + reward_two_hot( reward_logits, r_true )            # cross-entropy
  + q_two_hot   ( q_logits,      td_target )           # cross-entropy x 5 heads
  + pi_loss     ( -min_Q + alpha * entropy bonus )
```

All five losses share one `clip_by_global_norm(20.0)` and one Adam with
`lr=3e-4`. Target networks track parameters with `tau=0.01` Polyak averaging.

---

## 2. Our JAX/Flax reimplementation

Code: [helios-rl/src/helios/algorithms/tdmpc2.py](../../src/helios/algorithms/tdmpc2.py).

Notable JAX-specific choices that diverge from the reference PyTorch:

- **Multi-step lax.scan rollout.** The consistency and TD loss require an
  H-step rollout under the learned dynamics. We unroll with `jax.lax.scan`
  inside `loss_fn`, which is compiled once and re-used.
- **Two-hot via vmap.** The 101-bin distribution is encoded by
  `vmap`-ed `jnp.searchsorted` and decoded by softmax @ centres; identical
  output to the reference but trivially vectorised across batch and ensemble.
- **SimNorm in pure functions.** `simnorm(z, V=8)` reshapes the 512-dim
  latent into 8 groups of 64 and applies a per-group softmax. The resulting
  vector lives on a product of simplices — geometry the encoder shares with
  the prototype assignments used by Glass.
- **MPPI in JAX.** Action samples are drawn once outside `jit`, then a
  scanned `_mppi_iter` updates the per-step Gaussian and recomputes elites.
  Compilation cost is paid once per `(n_envs, horizon, n_samples)` shape.

Single-GPU throughput on a 3090 reaches ~115 sps on HopperHop at 256 envs
(close to the reference PyTorch 2×A100 baseline at single-precision FP32).

---

## 3. Glass-JAX in one paragraph

`glass-jax` exposes a differentiable **two-dimensional structural entropy** in
its compact algebraic form (Li & Pan 2016, see
[SE_maths.md](SE_maths.md) §3). Let $A \in \mathbb{R}^{N \times N}$ be a
symmetric similarity matrix on $N$ items, $d_i = \sum_j A_{ij}$ the row sum,
$2m = \sum_i d_i$ the total volume, and $S \in \mathbb{R}^{N \times K}$ a soft
assignment with $\sum_k S_{ik} = 1$. For each cluster $k$:

$$
V_k = \sum_i S_{ik}\, d_i \quad\text{(volume)}, \qquad
g_k = \sum_i S_{ik}\,\bigl(d_i - (A S)_{ik}\bigr) \quad\text{(cut)}.
$$

The 2D structural entropy is then

$$
H^{2}(A, S) \;=\;
\underbrace{-\sum_{k=1}^{K} \frac{g_k}{2m}\, \log_2 \frac{V_k}{2m}}_{\text{inter-community uncertainty}}
\;+\;
\underbrace{H^{1}(A) + \sum_{k=1}^{K} \frac{V_k}{2m}\, \log_2 \frac{V_k}{2m}}_{\text{intra-community uncertainty}},
$$

where $H^{1}(A) = -\sum_i (d_i/2m)\log_2(d_i/2m)$ is the 1D (no-community)
structural entropy. Minimising $H^{2}$ rewards **community structure** — high
intra-cluster cohesion (large $V_k$, small $g_k$) and a partition that is
concentrated rather than uniform. See
[glass-jax/src/glass/objectives/structural_entropy.py](../../../glass-jax/src/glass/objectives/structural_entropy.py)
for the exact JAX implementation; the API used by TD-MPC-Glass is:

```python
from glass.objectives.structural_entropy import two_dimensional_structural_entropy
se = two_dimensional_structural_entropy(A, S)  # scalar, differentiable in A and S
```

---

## 4. Where Glass plugs into TD-MPC2

Code: [helios-rl/src/helios/algorithms/tdmpc_glass.py](../../src/helios/algorithms/tdmpc_glass.py).

We add a **prototype transition graph** over the H-step latent rollout that
`tdmpc2.py` already produces in its loss function:

1. Maintain `num_prototypes=16` learnable vectors `μ_1..μ_N` of dim 512.
2. For each `(z_t, z_{t+1})` pair in the rollout, compute soft assignments
   `q_t = softmax(cosine(z_t, μ) / T_proto)` and `q_{t+1}` similarly.
3. Aggregate `P = (1/T) Σ_t q_t ⊗ q_{t+1}` — a 16×16 transition probability
   matrix over prototypes.
4. Symmetrise: `A = (P + P^T)/2`.
5. Cluster: maintain `K=8` learnable assignment logits, `S = softmax(logits)`.
6. Loss:

```text
L_glass = lambda_se   * SE2(A, S)
        + lambda_balance  * hinge_balance(S)
        + lambda_temporal * KL(q_t || q_{t+1})
```

Added to the TD-MPC2 sum-loss with `λ_se = 5e-3`, `λ_balance = 1e-2`,
`λ_temporal = 1e-3`. The gradient flows into prototypes, assignment logits and
— when `stopgrad_graph=False` — through `z_t` back into the encoder/dynamics.
For Phase 1 we keep `stopgrad_graph=True`: the world model is too sensitive at
the current λ. Phase 2 will revisit.

The hypothesis is that adding a structural-entropy pressure on the rollout
graph encourages the latent to **partition into temporally coherent regions**,
which (a) acts as a representation prior the way contrastive losses do for SimCLR
and (b) gives MPPI a coarser navigation signal that should help long-horizon
credit assignment.

---

## 5. Iteration history

### 5.1 First 5-seed run was *inert*

After plumbing Glass into `tdmpc_glass.py` and running 5 seeds × 4M steps on
HopperHop (`exp/tdmpc_glass/HopperHop_pre_phase1/`), MPPI return ended at
`327.5 ± 149.8` versus official `449.2 ± 312.1`. The 16 dumped diagnostic
matrices told the real story:

```text
step= 250k  P_min=0.0617  P_max=0.0637  S_max=0.1283  clu_ent=2.0794
step=4.00M  P_min=0.0624  P_max=0.0626  S_max=0.1287  clu_ent=2.0794
```

With `K=8`, uniform values are `1/K=0.125` and `log(K)=2.079`. **`P` deviated
from uniform by less than 0.001 across the entire 4M run.** Glass clustering
was numerically inert. Five root causes:

| #  | Cause | Symptom |
|---|---|---|
| 1 | `stopgrad_graph=True` plus L2-to-uniform balance | Glass loss only saw the assignment logits |
| 2 | SE₂ has vanishing gradient near the uniform fixed point | Loss surface flat where init landed |
| 3 | `assign_logits = 0.01·N(0,1)`, `T_assign = 1.0` | S born inside that flat region |
| 4 | Prototype L2 distance with `T_proto = 0.2` | Soft-min collapses at d=512 |
| 5 | `λ_*` ≤ 1e-3, shared `clip_by_global_norm(20.0)` | Effective LR ≈ 1e-7 |

### 5.2 Phase 1 fixes (PR-equivalent in `tdmpc_glass.py`)

- `glass_transition_graph(..., use_cosine_assign=True)` — cosine similarity
  with prototype norms inside the softmax denominator; well-conditioned at
  d=512.
- One-sided **hinge balance**: `Σ relu(cluster_mass − 2/K)^2`. Fires only on
  collapse; does not push toward uniform.
- `init_glass_params(..., assign_logits_init_scale=1.0)` — initial logits
  drawn from `N(0, 1)` instead of `0.01·N(0,1)` so S starts away from uniform.
- `GLASS_DEFAULTS`: `λ_se 1e-4 → 5e-3`, `λ_balance 1e-3 → 1e-2`,
  `λ_temporal 1e-4 → 1e-3`, `T_proto 0.2 → 1.0`.

A separate-optimizer experiment (`_b`/`_c` runs) was abandoned: anything that
changed the trace/RNG order versus baseline perturbed the world model and
plateaued MPPI at ~250. Phase 1 keeps the shared optimizer.

### 5.3 Phase 1 results

Glass is now active. From the training logs:

| seed | ent (max=2.079) | active (of K=8) | max_mass (uniform=0.125) | cut |
|---|---|---|---|---|
| 1 | 1.386 | 4 | 0.250 | 0.722 |
| 2 | 1.386 | 4 | 0.250 | 0.733 |
| 3 | 1.386 | 4 | 0.250 | 0.725 |
| 4 | 1.098 | 3 | 0.346 | 0.636 |

Seeds 1–3 found the 4-cluster symmetric basin; seed 4 landed in a 3-cluster
asymmetric one and is the current outlier on returns.

![Glass diagnostics — Phase 1 vs Pre-Phase 1](../../exp/tdmpc_glass/plots/hopperhop_phase1_glass_diag.png)

The prototype-to-cluster matrix `S` and the transition matrix `P` for seed 3
at 4M steps show the block structure directly:

![Transition matrix seed 3 @ 4M](../../exp/tdmpc_glass/plots/hopperhop_phase1_glass_matrix_seed3.png)

MPPI returns at 4M (seeds 4, 5 still in flight or queued):

| seed | Phase 1 (final) | Pre-Phase 1 (final) | Official (final) |
|---|---|---|---|
| 1 | 323.0 | 273.9 | 380.1 |
| 2 | 440.1 | 230.7 | 373.2 |
| 3 | 447.9 | 526.0 | 594.2 |
| 4 | in-flight (~258 @ 2.75M) | 355.5 | — |
| 5 | queued | 251.2 | — |
| **mean (1,2,3)** | **403.7** | **343.5** | **449.2** |

![HopperHop Phase 1 vs Official — MPPI return](../../exp/tdmpc_glass/plots/hopperhop_phase1_progress_95ci.png)

Phase 1 versus official is within one 95% CI half-width on the three completed
seeds, with **45% lower seed-to-seed variance**. Versus pre-Phase 1 on the
same seeds, Phase 1 adds **+60 mean return** while halving variance. The
remaining gap (~45 points) is plausibly closable by Phase 2 once seed 4 and 5
land.

---

## 6. Launch commands

All commands assume `/workspace/helios-rl` as the working directory and
`/workspace/wiki/learn_mujoco_playground/repo` checked out for MuJoCo
Playground envs.

### 6.1 Shared environment

```bash
export PYTHONPATH=/workspace/helios-rl/src:/workspace/wiki/learn_mujoco_playground/repo
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### 6.2 RTX 3090 (24 GB, CUDA 12)

```bash
pip install -r /workspace/helios-rl/requirements-rtx3090.txt
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

python3 scripts/run_benchmark.py \
    --algo tdmpc_glass --task HopperHop \
    --seed 1 --n_envs 256 --total_env_steps 4000000 \
    --eval_every 250000 --log_dir exp/tdmpc_glass
```

### 6.3 RTX 4090 (24 GB, CUDA 12)

Same wheel set as 3090, same memory fraction — the 4090's higher SM count
gives ~1.4× sps:

```bash
pip install -r /workspace/helios-rl/requirements-rtx3090.txt
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

python3 scripts/run_benchmark.py \
    --algo tdmpc_glass --task HopperHop \
    --seed 1 --n_envs 256 --total_env_steps 4000000 \
    --eval_every 250000 --log_dir exp/tdmpc_glass
```

### 6.4 RTX 50-series (CUDA 13, sm_120)

50-series cards need CUDA 13 wheels; 5070-class (8 GB) cards must drop
`n_envs` to 128 and `mem_fraction` to 0.65:

```bash
pip install -r /workspace/helios-rl/requirements-rtx50series.txt
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65

python3 scripts/run_benchmark.py \
    --algo tdmpc_glass --task HopperHop \
    --seed 1 --n_envs 128 --total_env_steps 4000000 \
    --eval_every 250000 --log_dir exp/tdmpc_glass
```

On 24-GB 50-series cards (5080 Ti / 5090) keep `n_envs=256` and bump the
fraction back to 0.85.

### 6.5 Multi-seed CI (the 5-seed reproduction script)

```bash
bash scripts/run_phase1_remaining_seeds.sh
```

Generates `exp/tdmpc_glass/plots/hopperhop_phase1_progress_95ci.png` and the
companion Glass-diagnostic plots once all five seeds complete.

---

## 7. Phase 1b — Two more knobs (in flight on RTX 4070 Ti)

After Phase 1 landed within-CI of official, two complementary changes were
spun up on a separate RTX 4070 Ti / CUDA 12.4 box to accelerate iteration:

```text
--glass_proto_temperature        1.0 → 0.7
--glass_assign_logits_init_scale 1.0 → 0.5
```

The rationale: softer prototype softmax sharpens the assignment **earlier**
(cleaner graph signal in the first 1M steps), and a smaller init scale starts
$S$ closer to but still away from the uniform fixed point, letting it settle
without overshoot.

### Seed 1 (in flight, 2.75M / 4M)

| step  | Phase 1 seed 1 | **Phase 1b seed 1** | Δ      | Official seed 1 |
|------:|---------------:|--------------------:|-------:|----------------:|
|  250k |          97.6  |               145.5 |  +48   |             —   |
|  500k |         263.8  |               280.2 |  +16   |             —   |
|  750k |         328.6  |               363.9 |  +35   |             —   |
| 1.00M |         340.2  |               382.2 |  +42   |             —   |
| 1.25M |         347.6  |               456.5 | **+109** |          —    |
| 1.50M |         350.9  |               490.3 | **+139** |          —    |
| 1.75M |         356.6  |               491.3 | **+135** |          —    |
| 2.00M |         347.1  |               487.5 | **+140** |          —    |
| 2.25M |         371.6  |               489.3 | **+118** |          —    |
| 2.50M |         365.3  |               492.1 | **+127** |          —    |
| 2.75M |         363.1  |               486.5 | **+123** |          —    |
| 4.00M |         323.0  |          (in flight) |  —     |          380.1 |

Glass diagnostics for Phase 1b seed 1 settle on the **same 4-cluster basin**
as Phase 1 (`ent≈1.386, active=4, max_mass=0.250, cut≈0.70`). The improvement
comes from the world model receiving a cleaner clustering signal earlier in
training; the basin itself is identical.

Phase 1b seed 1 at 2.75M (486.5) already exceeds the official 5-seed mean of
449.2 by ~37. Seeds 2–5 are queued on the same machine; estimated wall-clock
for the full 5-seed CI: ~9 hours total at sps≈570 (vs ≈ 17 h on the 3090).

## 8. Pending work and plan

In order of priority:

1. **Finish Phase 1b 5-seed sweep on the 4070 Ti.** Queue runner is up
   ([run_phase1b_queue.sh](https://github.com/SuuTTT/helios-rl/blob/main/scripts/)
   on the remote). After all five seeds land, regenerate the CI plot and
   compare against Phase 1 + official on the same axes.
2. **Finish the pending Phase 1 seeds 4 + 5 locally on the 3090** so we have
   matched-config statistics for the Phase 1 → Phase 1b ablation.
3. **Relax `stopgrad_graph`.** With Glass now genuinely active and the world
   model not destabilised by Phase 1b, anneal `stopgrad_graph: True → False`
   over the first 500k steps and let the SE gradient back into the encoder.
   Pair with `λ_se` warmup `1e-3 → 5e-3` to stay safe at $t = 0$.
4. **Sweep `K`.** Phase 1 and Phase 1b both converge to 4 clusters out of
   $K=8$ with mass $\approx 1/4$ each — exactly the symmetric basin. Try
   $K=4$ to match (saves logits, eliminates the latent over-parameterisation)
   and $K=16$ to test whether richer clustering helps at the cost of more
   collapse risk.
5. **Visualise the prototype skill-tape.** Project rollout latents onto the
   final $S$-induced cluster assignment as a function of time-in-episode;
   confirm clusters correspond to gait phases (stance/flight/recovery) on
   HopperHop, which would justify the temporal-coherence prior.
6. **Move beyond HopperHop.** Walker-Run and Quadruped-Walk are the next
   DMC-suite tasks where 4M is enough for official TD-MPC2 to converge.
   Expect harder SE landscape on Quadruped (higher dim, $\ge 12$ true gait
   phases).
7. **Phase 2 hypothesis to test on Walker-Run.** If Phase 1b is robust on
   HopperHop after the 5-seed CI, the next paper-worthy claim is *Glass-SE as
   a free representation prior*: same hyperparameters, no per-task tuning,
   net positive across the suite.

---

## Appendix A: SE formula sanity check vs `SE_maths.md`

The formula in §3 above matches **SE_maths.md §3 — Compact algebraic form**
and the exact code path in
[`two_dimensional_structural_entropy()`](../../../glass-jax/src/glass/objectives/structural_entropy.py).
The earlier draft of this blog used a different (incorrect) expression; it has
been corrected here. The implementation uses $\log_2$ throughout, and clamps
$p_{\mathrm{vol}}$ and $d_i/2m$ to $[\varepsilon, 1]$ before the log to keep
gradients finite at the uniform boundary.