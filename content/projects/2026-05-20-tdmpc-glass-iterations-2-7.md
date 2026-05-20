---
title: "TD-MPC-Glass Iterations 2–7: Basin Lottery, Glass Internals, and the K_UPDATE Hypothesis"
date: 2026-05-20
description: "Seven iterations of TD-MPC-Glass on HopperHop: what the μ/S/c vectors actually do, what the video rollouts revealed about cluster→gait alignment, why 25 phases failed to beat vanilla TD-MPC2, and why K_UPDATE=64 may be the root cause we ignored for two weeks."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["tdmpc2", "glass-jax", "structural-entropy", "jax", "reinforcement-learning", "hopper", "mujoco-playground"]
---

{{< katex >}}

> This post is the sequel to [Phase 1b](https://suuttt.github.io/projects/2026-05-13-tdmpc-glass-phase1b/).
> It covers seven months of iteration — ~25 experimental phases, 8 GPUs, two
> goals that still aren't both solved — and ends with the current best hypothesis:
> **we've been training at 4× too low a gradient-update rate the entire time.**
> Live dashboard: [bus-brussels-fate-performed.trycloudflare.com](https://bus-brussels-fate-performed.trycloudflare.com/)

---

## 1. Two goals

We track every run against two success criteria:

| ID | Name | Criterion | Status |
|----|------|-----------|--------|
| **G1** | Consistency | *All 5* seeds reach best MPPI ≥ 500 | Never achieved. Best: 2/5 |
| **G2** | Ceiling | *At least one* seed reaches MPPI ≥ 600 | Phase-t (knee-penalty shaping) hit 612 — but benchmark-unfair |

G1 is the hard goal. It requires that the agent reliably escapes the gait basin
lottery on *every* seed — not just the lucky ones. G2 only needs one seed to
push the upper limit; it confirms the physical ceiling exists. The benchmark-fair
objective is G1 without reward shaping, behaviour cloning, or environment edits.

---

## 2. TD-MPC-Glass internals: what μ, S, and c actually do

Before the iteration history, here is the complete story of the three Glass
vectors. This is what the `K=` number on rendered rollout videos means.

### 2.1 The latent z

The TD-MPC2 encoder maps every observation \(o_t \in \mathbb{R}^{15}\) to a
512-dim latent \(z_t\) via SimNorm(V=8): the 512 dimensions are split into 8
groups of 64, and each group is independently softmax-normalised. So \(z\) is
not just an arbitrary vector — it looks like **8 concatenated soft histograms**,
one per group. This geometry matters: it means \(z \cdot \mu^\top\) is a sum of
group-wise inner products, and cosine similarity is well-defined even across
very different observations.

### 2.2 Prototypes μ — the anchor latents

\(\mu \in \mathbb{R}^{N \times d}\), \(N=16\), \(d=512\). Each row \(\mu_n\) is
a *learnable anchor state* in the same SimNorm-shaped latent space. Think of them
as learned "exemplars": after training, \(\mu_1\) might point toward the latent
direction of "leg fully extended mid-hop" and \(\mu_7\) toward "knee on ground
recovering from fall."

**Why learn N=16 anchors instead of just K=8 clusters directly?**
Two reasons:

1. *Computational.* Structural entropy requires building a transition graph over
   nodes. If nodes are individual data points (batch size B=256), the graph has
   256 nodes and \(O(B^2)\) edges per step. With N=16 prototypes the graph has
   16 nodes always — 256× cheaper.

2. *Stability.* The soft assignment matrix S (described next) maps prototypes to
   clusters and is a **parameter** — it doesn't depend on which batch we sampled.
   So the cluster partition is consistent across training steps and identifiable
   post-hoc, which is why you can watch a rendered video and see a stable K= label
   on each frame.

During training, each latent \(z_t\) is "assigned" to prototypes via a soft
cosine similarity:

\[
c_{t,n} = \frac{\exp\!\bigl(-\|z_t - \mu_n\|^2_{\text{cos}} / \tau_p\bigr)}{\sum_{n'} \exp\!\bigl(-\|z_{t} - \mu_{n'}\|^2_{\text{cos}} / \tau_p\bigr)}
\]

This is the **soft assignment vector** \(c_t \in \Delta^{N-1}\): an N-dimensional
probability distribution over which prototype the current latent "belongs to."
Temperature \(\tau_p = 0.2\) makes this fairly peaked — one prototype dominates.

### 2.3 The transition graph A

Over a replay batch of B=256 transition pairs \((z_t, z_{t+1})\), Glass builds
a soft transition count matrix:

\[
P_{\text{counts}} = \sum_{t=1}^{B} c_t \otimes c_{t+1} \in \mathbb{R}^{N \times N}
\]

Row-normalise and symmetrise:

\[
A = \tfrac{1}{2}(P + P^\top), \quad P = \text{row\_norm}(P_{\text{counts}} + \epsilon)
\]

**A is the prototype transition graph**: entry \(A_{mn}\) says "how often does
the agent transition from a state near prototype \(\mu_m\) to one near prototype
\(\mu_n\)?" A good hop policy should produce a cyclic graph over 4 nodes
(push-off → takeoff → flight → landing → push-off). A kneeling policy produces
a 2-node oscillator.

### 2.4 Cluster assignment S and the structural entropy loss

\[
S = \text{softmax}(\text{assign\_logits}, \,\text{axis}=1) \in \mathbb{R}^{N \times K}
\]

\(S_{nk}\) is the probability that prototype \(n\) belongs to cluster \(k\).
In well-trained Glass, each row is near one-hot — each prototype belongs to one
behavioural cluster. The K=8 clusters are a *coarsening* of the N=16 prototypes:
a two-level hierarchy where prototypes capture fine-grained state variation and
clusters capture coarse behavioural phases.

The auxiliary loss Glass adds to TD-MPC2 is the **2D structural entropy** of the
transition graph A under the partition S:

\[
H^2(A; S) = -\sum_{k=1}^{K} p_{\text{cut},k} \log p_{\text{vol},k}
           + H^1(A) + \sum_{k=1}^K p_{\text{vol},k} \log p_{\text{vol},k}
\]

where (in the differentiable soft form):

\[
d = A\mathbf{1}, \quad V = S^\top d \in \mathbb{R}^K, \quad
p_{\text{vol},k} = V_k / \text{vol}(A)
\]
\[
g_k = \sum_n S_{nk}\,(d_n - (AS)_{nk}), \quad
p_{\text{cut},k} = g_k / \text{vol}(A)
\]

Minimising \(H^2\) pushes the partition to have **small boundary cuts** (transitions
stay within clusters) and **balanced cluster volumes** (no dead clusters). This is
exactly the pressure to align clusters with gait phases: if push-off and landing
always co-occur in the same trajectory segment, the min-cut partition will group
their prototypes together.

The full Glass loss added on top of TD-MPC2:

\[
\mathcal{L} = \mathcal{L}_{\text{TD-MPC2}}
            + \lambda_{\text{SE}} \cdot H^2(A,S)
            + \lambda_{\text{bal}} \cdot (\mathcal{L}_{\text{cluster-bal}} + \mathcal{L}_{\text{proto-bal}})
            + \lambda_{\text{temp}} \cdot \mathcal{L}_{\text{temporal}}
\]

Current defaults: \(\lambda_{\text{SE}} = 10^{-4}\), \(\lambda_{\text{bal}} = 10^{-3}\), \(\lambda_{\text{temp}} = 10^{-4}\).

### 2.5 The K= label in rollout videos

When watching a rendered MP4, the overlay number comes from:

```
argmax(S[argmax(z · μᵀ)])
```

Step by step:
1. Compute cosine similarities \(\text{sim}_n = \hat{z} \cdot \hat{\mu}_n\) (unit-normalised, \(T=0.7\)).
2. `n_star = argmax(sim)` — the nearest prototype.
3. `cluster_id = argmax(S[n_star])` — which cluster that prototype belongs to.

This is the hard-assignment label of the prototype the agent is currently visiting.

---

## 3. What the rollout videos revealed

We rendered Phase-f seeds 1, 3, 4 (best_mppi.pkl) and overlaid the K= label.

### Seed 1 — winner (MPPI 571)

The K= label cycles **3 → 4 → 3 → 4 → ...** during steady hopping, with 6 and 7
appearing only during recovery from a fall. The mapping:

| Cluster | Gait phase |
|---------|-----------|
| K=3 | Hip leaves ground / push-off |
| K=4 | Leg extending, mid-takeoff |
| K=6 | Air-recovery (post-fall) |
| K=7 | Head/ground contact recovery |

This is exactly what a 4-state gait machine should look like. **Glass found a
meaningful partition aligned with the hop cycle.**

### Seed 3 — stuck (MPPI 262)

The K= label cycles **2 → 4 → 5 → 1 → 4 → 2**, 5 clusters active. Glass also found a
rich partition here — but the *behaviour* is a **knee-walk with nose dragging the
ground**. A latent partition that accurately describes kneeling is not useful for hopping.

**Conclusion: the bottleneck is downstream (policy/critic), not the representation.**
Seed 3 has healthy Glass geometry but converged to the wrong gait technique.

### Seed 4 — K=3 basin (MPPI 266)

Only **2 clusters active** (K=0 ↔ K=5) throughout. The Glass partition has 3
entries (K=3 basin) but the policy collapses two of them into "knee on ground."
This is the structural cap we labeled the K=3 basin: it lacks the fourth
behavioural primitive (foot-in-flight) needed for real hopping.

### The key observation about K stability

The best policies — both Phase-f seed-1 (571) and Phase-t seed-2 (612, knee-penalty)
— share a striking property: **K= does not change during the steady-state hop cycle**.
Seed-1 uses exactly K=3 during push-off and K=4 during takeoff, consistently,
for 7 consecutive hops. The cluster assignment is *locked* to the gait phase.

Stuck seeds show the opposite: K= oscillates even within a single gait phase.
Seed-3's K=2 → 4 → 5 → 1 within one "knee-walk" cycle means the encoder is
*not* finding stable behavioural anchors — the representation is changing faster
than the gait.

**This is the core failure mode.** Glass's structural entropy loss pushes prototypes
to form a good partition of the transition graph, but it does not enforce temporal
stability of the assignment. A prototype can be visited at a knee-walk phase and
also at a hop phase, breaking the gait alignment.

---

## 4. Iteration history — all 25 phases

### Iteration 1–2 (Phases 1b, 1c, d–h)

| Phase | Change | Best | Verdict |
|-------|--------|------|---------|
| 1b | TD-MPC-Glass default | 562 s5 | Baseline: 2/5 > 500. Seed lottery. |
| 1c | Act-noise anneal 0.30→0.10 | 412 | FALSIFIED: hurts winners |
| d_v1 | H=5 + noise=0.40 | 114 | FALSIFIED: Warp 901 crash |
| d_v2 | H=5 only | 199 | FALSIFIED: no help |
| e | Q-reset | 228 then collapse | FALSIFIED: impl bug, zeroed all opt state |
| **f** | latent_smooth=1e-3 | **571 s1** | First >500 ever; 1/5 survived |
| g | consistency_coef=1.0 | 482 s2 | Partial lift; caps below 500 |
| h | smooth+ccoef combined | 490 | No additive benefit |

**Key finding (iter 2):** latent action smoothing 1e-3 flipped seed-1's gait
from knee-walk to foot-hop, producing the first 571 winner. But 4/5 seeds didn't
make the same flip.

### Iteration 3 (Phases i–n)

| Phase | Change | Best | Verdict |
|-------|--------|------|---------|
| i | smooth=1e-4 (too weak) | 312 | FALSIFIED: below threshold |
| **j** | curriculum smoothing (off<250k) | **518 s2** | 1/5 > 500; curriculum helps |
| k | smooth + λ_temporal=0.05 | 292 | FALSIFIED: over-regularised |
| l_v1 | tdmpc2 + smooth, Glass OFF | 289 | FALSIFIED: Glass IS needed |
| m | basin perturbation (param noise) | 286 | Doesn't help |
| n | basin perturbation v2 | 56 | FALSIFIED |
| **o** | Glass OFF after 2M (hybrid) | **577 s3** | 1/3 > 500; Glass helps early |

**Key finding (iter 3):** K=3 basin is a structural cap (~310 average) — 4-phase
gait needs K=4. Curriculum smoothing (Phase-j) rescues some seeds by letting the
basin lock *before* smoothing engages.

### Iteration 4 (Phases p, t, and hierarchy)

| Phase | Change | Best | Verdict |
|-------|--------|------|---------|
| **p** | EXPL_UNTIL=500k (wider random phase) | **538 s4** | 1/3 > 500; slow-burn winner |
| **t** | Knee penalty reward shaping | **612 s2** | G2 hit! 2/3 > 500. Benchmark-unfair |
| y | Hierarchical Glass K_super=4 | 462 | 0/3 > 500; close but no cigar |

Phase-t confirmed the physical ceiling (612) but used reward shaping. Reward
shaping is excluded from the benchmark-fair objective.

### Iteration 5 (Paths 7, 9, 10 and intrinsic rewards)

| Phase | Change | Best | Verdict |
|-------|--------|------|---------|
| v | Cluster soft-dist as pi/q obs (Path 7) | 232 | FALSIFIED: representation drift |
| **x** | NS=2048 MPPI planner | **523 s3** | ~40% hit rate; planner scale helps |
| y | Hierarchical Glass K_super=4 (Path 10) | 462 | 0/3 > 500 |
| P | Cluster entropy intrinsic reward | 91 then 2.4 | FALSIFIED: non-stationary |
| Pa | Same with linear decay 0.1→0 | 25 | FALSIFIED: 3.6× worse |

**Key finding (iter 5):** Planner scale (NS=2048) raises good-seed ceiling but
doesn't rescue stuck seeds. Cluster entropy intrinsic reward is fundamentally
incompatible with the reward scale (max bonus ≈ 210, target ≈ 600).

### Iteration 6 — reward shaping audit (concluded)

We ran a systematic audit of benchmark-*unfair* reward shaping to understand the
ceiling, then studied whether stacking helps.

| Phase | Config | Seeds | G1 hit | Best |
|-------|--------|-------|--------|------|
| phasez | Vanilla tdmpc2 baseline (NS=512) | 5 | 1/5 | ~500 |
| phaseq_knee | Knee penalty + NS=2048 | 12 | 4/12 (33%) | 557 |
| phaser1_soft | Soft stand bonus 0.1→0@3M | 5 | 1/5 (20%) | 553 |
| phaser2_gait | Gait fall penalty + action smooth | 4 | 1/4 (25%) | 510 |
| **phaserstack** | **ALL shaping stacked** | 3 | **0/3 (0%)** | **5–16 COLLAPSED** |
| phaserstack_nosmooth | Stack − action_smooth | 2 | 0/2 | 10.2 |
| phaserstack_nosoft | Stack − soft_stand_bonus | 2 | 0/2 | 254 |

**Key finding (iter 6):** The stack collapse is **combinatorial**, not attributable
to any single component. Dropping either ablated component still mostly collapsed.
Individual shaping raises hit rate to 20–33% but no individual component gets 5/5.
Also: **Glass never clearly beat vanilla TD-MPC2** — phasez already produced one
seed above 500 with no Glass at all.

### Iteration 7 — K_UPDATE audit (in progress)

The strongest unresolved benchmark-fair lever: **gradient update rate.**

Current codebase: `K_UPDATE=64` → 64 gradient updates per 256-env-step collection
batch → ~0.25 updates per env step. Official TD-MPC2: ~1.0 updates per env step.
We've been training at **4× lower update rate for 7 iterations.**

Phase-aa sweep: vanilla tdmpc2, NS=2048, K_UPDATE ∈ {64, 128, 256}, seeds 1–3:

| K_UPDATE | Seed 1 | Seed 2 | Signal |
|----------|--------|--------|--------|
| 64 | 234 @ 3.25M | running | Weak |
| **128** | **538.6 G1** | 285 | **Strong** |
| **256** | **531.0 G1** | 324 | **Strong** |

k64 seed-1 is stuck at 234 — the same basin-lottery failure mode we've seen for
7 iterations. k128 and k256 both escaped it in seed-1. **This is the first time
any benchmark-fair change has rescued a seed that k64 gets stuck on.**

---

## 5. Why Glass didn't help (yet) and what to try next

### 5.1 What Glass does accomplish

Glass does find meaningful partitions. The video analysis confirms:
- Winner seeds: 4–5 active clusters, stable K= during hop phases, partition maps
  cleanly to push-off / takeoff / air / recovery.
- Stuck seeds: 4–5 active clusters too, but the partition describes the *kneeling*
  gait, not the hopping gait.

Glass is doing structurally correct representation learning. The problem is that
**a good representation of the wrong gait is useless.**

### 5.2 The stability problem

The key observation from the videos: the best-performing seed (571, seed-1) has
**K= stable within each gait phase**. Push-off is always K=3; takeoff is always K=4.
In contrast, stuck seeds show K= changing within a single gait phase — the
encoder is not finding stable temporal anchors.

Glass's structural entropy loss minimises cross-community boundary cuts on the
transition graph, which encourages *Markovian* cluster structure — but it does not
explicitly encourage cluster assignment to be **temporally stable within a single
gait phase**. A latent can visit prototype A during one push-off and prototype B
during the next push-off, and the SE loss will not penalise this if both A and B
belong to the same cluster.

### 5.3 Proposed Glass integration directions

**Direction 1: Temporal consistency on cluster assignments (not latents)**

Currently `λ_temporal` penalises \(\|z_t - z_{t+1}\|^2\) — smoothness in the
latent space. Instead, penalise entropy of the assignment distribution over a
short window:

\[
\mathcal{L}_{\text{assign-consistency}} = \mathbb{E}_t\!\left[H\!\left(\frac{1}{W}\sum_{\tau=t}^{t+W} c_\tau\right) - \frac{1}{W}\sum_{\tau=t}^{t+W} H(c_\tau)\right]
\]

This is the mutual information between time index and prototype assignment within
a window of length W. Minimising it encourages the same prototype to be active
throughout a gait phase, without constraining the *values* of the latent.

**Direction 2: Phase-locked prototypes via temporal contrastive loss**

Positive pairs: two frames within the same gait phase (e.g., consecutive frames
during steady-state hop). Negative pairs: frames from different gait phases. Use
NT-Xent on the soft assignment vector \(c_t\) rather than on z:

\[
\mathcal{L}_{\text{phase-contrastive}} = -\log \frac{\exp(c_t \cdot c_{t'}^+/\tau)}{\exp(c_t \cdot c_{t'}^+/\tau) + \sum_{j} \exp(c_t \cdot c_j^-/\tau)}
\]

This directly trains the prototype assignments to be consistent within a phase
and discriminative across phases — exactly the property the winning seed exhibits
but the stuck seeds don't.

**Direction 3: Coarser prototype count matched to known gait phases**

Reduce N=16 → N=4 and K=8 → K=4 to match the 4-phase hop cycle. The current
N=16/K=8 setup leaves 12+ unused capacity that the SE loss can fill with
arbitrary partitions. Constraining to N=4 forces the two argmaxes to compress
directly to gait phases, at the cost of losing within-phase granularity.

**Direction 4: Prototype-conditioned planning in MPPI**

Currently MPPI is cluster-unaware — it samples action sequences and evaluates
them with the same Q ensemble regardless of which cluster z is in. A lightweight
extension: learn cluster-specific Q offsets \(\Delta Q_k\) that capture
cluster-conditional value. Effectively this is a mixture-of-experts Q with K
components, using Glass to route.

**Direction 5: Fix K_UPDATE first, then re-evaluate Glass**

The most urgent question: does Glass add anything over vanilla TD-MPC2 *once the
training ratio is correct?* Phase-ac (K_UPDATE winner + 5-seed Glass vs. vanilla)
will answer this directly. If Glass is neutral at K_UPDATE=128, it means the
representation benefit was masked by under-training — and the architecture
changes above become much more interesting on a properly-trained base.

---

## 6. Infrastructure this week

- **8-GPU fleet**: Local 4070 Ti + ssh6 (4060 + 3080) + ssh17637 GPU0/1 (3060 Lap) + ssh1 (2080 Ti) + ssh3 (3070 + 3060 Ti)
- **Web dashboard** (port 5055): live learning curves, 95% CI bands per phase, multi-phase overlay with stable per-phase colors, canonical phase merging (e.g., phasex_local + phasex_4060 → phasex), comprehensive phase descriptions for all 40 phases
- **Auto-queue**: per-box queue files; idle boxes self-assign next experiment

---

## 7. What's running now

All 8 GPUs on Phase-aa K_UPDATE sweep. ETA: 2–3 days for seeds 2–3 to mature.
Next: pick K_UPDATE winner → Phase-ab (5-seed vanilla) → Phase-ac (Glass comparison).

If K_UPDATE fixes stuck-seed rate, the answer is: we were under-training for
7 iterations. The basin lottery may not be a lottery at all — just an artefact
of insufficient gradient updates in the first 500k steps.

---

## 8. Complete phase ledger

| Phase | Feature | Best MPPI | Hit rate |
|-------|---------|-----------|----------|
| 1b | TD-MPC-Glass default | 562 | 2/5 |
| 1c | Act-noise anneal | 412 | FALSIFIED |
| d_v2 | H=5 only | 199 | FALSIFIED |
| e | Q-reset (buggy) | 228 | FALSIFIED |
| f | smooth=1e-3 | **571** | 1/5 |
| g | consistency_coef=1.0 | 482 | 0/5 |
| h | smooth+ccoef | 490 | 0/2 |
| i | smooth=1e-4 | 312 | 0/1 |
| j | curriculum smooth | **518** | 1/5 |
| k | smooth+λ_temp=0.05 | 292 | 0/1 |
| l | tdmpc2+smooth (Glass OFF) | 289 | 0/1 |
| m | basin perturbation | 286 | 0/5 |
| n | basin perturb v2 | 56 | FALSIFIED |
| o | Glass OFF after 2M | **577** | 1/3 |
| p | EXPL_UNTIL=500k | **538** | 1/3 |
| t | Knee penalty (unfair) | **612** | 2/3 |
| v | Cluster obs (Path 7) | 232 | FALSIFIED |
| x | NS=2048 | **523** | ~40% |
| y | Hierarchical K_super=4 | 462 | 0/3 |
| P/Pa | Cluster entropy intrinsic | 91/25 | FALSIFIED |
| z | Vanilla tdmpc2 baseline | ~500 | 1/5 |
| phaseq_knee | Knee+NS=2048 | 557 | 4/12 |
| phaser1_soft | Soft stand bonus | 553 | 1/5 |
| phaser2_gait | Gait fall penalty | 510 | 1/4 |
| phaserstack | All shaping stacked | 16 | **COLLAPSED** |
| phaseaa k128 | K_UPDATE=128 (early) | **538** | promising |
| phaseaa k256 | K_UPDATE=256 (early) | **531** | promising |
