---
title: "Six Mirages: What 14 Iterations of Trying to Beat TD-MPC2 with Abstraction Taught Us"
date: 2026-06-07
description: "A pre-registered, compute-matched campaign to improve TD-MPC2 with latent abstraction ended in a complete null — after six separately-publishable effects each dissolved with sample size. Here is the full anatomy, a deep analysis of why abstraction can't help, what might still work, and what to expect from an ICLR submission."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["model-based-rl", "tdmpc2", "abstraction", "negative-results", "research-methodology", "bisimulation"]
---

{{< katex >}}

> **TL;DR.** We tried to beat vanilla TD-MPC2 at the architecture/algorithm level with three
> latent-abstraction mechanisms (geometric prototype clustering, reward-grounded behavioral
> clustering, pairwise bisimulation), under a strict fair protocol — identical hyperparameters,
> compute-matched, pre-registered gates, ~90 mature 1M-step runs on mujoco_playground (MJX) DMC.
> **Nothing beat vanilla.** Along the way, *six* distinct positive effects appeared — each one
> separately publishable from some interim snapshot — and every single one regressed to null
> (one inverted) as seeds accumulated. This post documents the null, dissects *why* TD-MPC2 is
> unimprovable by auxiliary abstraction in this regime, ranks what might still be promising, and
> gives an honest forecast for an ICLR submission.

## 1. The quest

TD-MPC2 [(Hansen et al., 2024)](https://arxiv.org/abs/2310.16828) is a strong latent world model:
encoder → SimNorm latent → latent dynamics + reward + twin-Q + policy heads, with MPPI planning
and a latent-consistency loss. The natural hypothesis: an *explicit abstraction* over its latent
space — grouping behaviorally-equivalent states, hierarchical clustering, behavioral metrics —
should improve it. The literature half-encourages this (DBC, BS-MPC, discrete-codebook world
models), and half-warns against it (Ni et al. show TD-MPC's objective already satisfies the
conditions of a sufficient self-predictive abstraction).

Our project ("Glass") ran 14 iterations over several weeks on a ~12-GPU vast.ai fleet:

| iterations | what we tried | outcome |
|---|---|---|
| 1–11 | geometric Glass: 32 SimNorm prototypes, soft cosine assignment, 2-level structural-entropy loss, on HopperHop | no robust gain; several falsified knobs (no-stopgrad graph, intrinsic cluster entropy, cluster-id obs) |
| 12 | restart-on-plateau procedure | "won" — but rescued a *vanilla* seed too: procedure trick, base-agnostic |
| 13 | population-based training (PBT) | "won" — inheritance is not Glass-specific: procedure trick |
| 14 | **fair protocol**: 4 arms × 5 tasks × 1M steps, identical everything, pre-registered gates | the subject of this post |
| 2a | synthetic-distractor falsification probe (8 runs) | falsified (ratio 1.23× < 1.5 gate) |
| C-probe | sparse-task structural-failure testbed search (6 runs) | no testbed exists in mujoco_playground; closed |

The iteration-12/13 realization is what forced the fair protocol: **basin-entry "wins" on
HopperHop were engineering tricks, not architecture wins.** A luck-dominated best-of-N metric on
a single exploration-bound task rewards exactly the interventions we wanted to exclude.

## 2. The arms

All arms share every hyperparameter, network size, planner budget, env-step count, and eval
schedule. The abstraction term is the *only* change.

- **vanilla** — TD-MPC2 as-is.
- **geometric Glass** — prototypes + structural-entropy auxiliary, \(\lambda_{SE}=5\cdot10^{-3}\).
- **behavioral Glass** — geometric Glass plus a learned per-prototype reward vector \(r_p\) and loss

$$\mathcal{L}_b = \lambda_b \left( c(z)^\top r_p - r \right)^2, \quad \lambda_b = 0.5$$

  pushing prototypes to group *reward-equivalent* states — a soft hierarchical analogue of bisimulation.
- **bisimulation aux** (BS-MPC-style reference) — permuted-pair encoder loss

$$\mathcal{L}_{bisim} = \left( \lVert z_i - z_j \rVert_1 - |r_i - r_j| - \gamma \lVert \mathrm{sg}(z'_i) - \mathrm{sg}(z'_j) \rVert_2 \right)^2$$

  at coefficient 0.01 (0.1 collapses training outright — the L1 pairwise scale is ~O(10) against an O(1) target).

Tasks: CheetahRun, WalkerRun, FingerSpin (primary), AcrobotSwingup, WalkerWalk (breadth);
metric = per-task-normalized mean of the final two MPPI evals; aggregate = IQM with
stratified-bootstrap 95% CI [(rliable)](https://arxiv.org/abs/2108.13264).

## 3. The final result: a complete null

**Aggregate over mature runs (≥950k steps):**

| arm | n | IQM | 95% CI |
|---|---|---|---|
| behavioral Glass | 28 | 0.726 | [0.713, 0.740] |
| geometric Glass | 16 | 0.748 | [0.695, 0.815] |
| **vanilla TD-MPC2** | **37** | **0.738** | **[0.715, 0.771]** |
| bisimulation aux | 6 | 0.549 | [0.527, 0.599] |

The three non-bisimulation arms are statistically indistinguishable. The bisimulation arm is
separated — *below*. The behavioral arm's point estimate ends **below the baseline it twice
"significantly beat."**

**The lower tail (3 primary tasks):**

| arm | n | mean | std | min | weak seeds (<0.65) |
|---|---|---|---|---|---|
| behavioral Glass | 24 | 0.747 | 0.176 | **0.127** | 2/24 |
| geometric Glass | 12 | 0.781 | 0.166 | 0.505 | 2/12 |
| vanilla | 33 | 0.764 | 0.137 | 0.419 | 3/33 |

Weak-seed rates 8–17%, all pairwise Fisher \(p \approx 1.0\). The single worst seed in the
entire study belongs to the behavioral arm (a late-training collapse from 0.643 at 800k steps
to 0.127 — verified from raw eval CSVs).

## 4. The six mirages

Each of these effects was real in our data at some point. Each would have made a plausible paper.
Each dissolved under a fixed protocol with growing n:

1. **Procedure wins (iters 12–13).** Restart-on-plateau and PBT "beat vanilla" — until we noticed
   they rescue vanilla seeds equally. Confound, not effect.
2. **The IQM advantage.** The behavioral-vs-vanilla gap crossed in and out of CI separation
   **three times**: +0.051 at n=17/23 (separated), +0.046 at n=19/32 (separated), +0.015 at
   n=20/37 (overlapping), +0.008 at n=22/37, **−0.012 at n=28/37 final**.
3. **Sample efficiency.** A "+30% reward@300k" edge swung to −22% before settling at parity.
4. **Distractor robustness.** Round 1 of the pre-registered probe showed behavioral Glass ahead
   under 64 OU nuisance dims (233 vs 153 on CheetahRun). Round 2 *reversed it* (67 vs 90 on
   WalkerRun). Combined ratio 1.23× < the 1.5× gate → falsified. Both encoders tolerate ~16
   nuisance dims and collapse identically between 16 and 32.
5. **Sparse-task rescue.** Three mujoco_playground sparse tasks show literal 0-vs-solved seed
   bimodality under flat TD-MPC2. Behavioral grounding rescues nothing (CartpoleSwingupSparse
   1/3 vs 0/3; BallInCup 1/3 vs 2/3; direction flips across tasks).
6. **The floor effect.** Zero weak seeds through n=16 (min 0.700) vs vanilla's 3/33 — we had a
   full "floor-raising mechanism" story drafted. Then weak seed #1 arrived (0.619 at n=18), then
   the study's worst seed (0.127). The effect didn't dissolve; it **inverted**.

The full estimate trajectory of the behavioral arm's IQM, recomputed at every maturity snapshot:

```
n=3: 0.818   n=4: 0.736   n=5: 0.829   n=6: 0.801   n=7: 0.793
n=8: 0.749   n=9: 0.743   n=17: 0.785  n=20: 0.753  n=22: 0.746
n=25: 0.747  n=28 (final): 0.726 [0.713, 0.740]
```

Several snapshots support a "significant win." Others support a "confirmed null." The final
estimate lands below baseline. **Every intermediate certainty was wrong.**

## 5. Deep analysis: why abstraction cannot improve TD-MPC2 here

This is the part we wish someone had written before we started. Five reasons, ordered from
theory to data.

### 5.1 The objective is already a sufficient abstraction

[Ni et al. (2024)](https://arxiv.org/abs/2401.08898) unify DeepMDP, DBC, SPR, and TD-MPC under
one condition: a latent is a *self-predictive abstraction* if it can predict its own next value
and the reward. TD-MPC2's consistency loss (latent dynamics predicts the next encoded latent,
stop-gradient EMA target) + reward head + value head train exactly this. The latent provably
retains what is needed for control — so any further auxiliary that *adds* information is
redundant, and any auxiliary that *removes* information can only break ties or destroy signal.
There is no free directional improvement left at the objective level. Our null is the empirical
shadow of their theorem.

### 5.2 Each mechanism we added duplicates (or fights) an existing component

This is the precise mechanism-by-mechanism account of the null:

- **Geometric clustering duplicates SimNorm.** TD-MPC2's SimNorm projects the latent onto a
  product of softmax simplices — the latent is *already* a soft categorical code. Prototype
  clustering with a structural-entropy objective re-imposes discrete-ish cluster structure on a
  space that has it by construction. Eleven iterations of geometric Glass on HopperHop and the
  fair-protocol arm (0.748 vs 0.738, overlapping) both say the same thing: the machinery is
  already in the box.
- **Behavioral grounding duplicates the reward head.** The encoder already receives reward
  gradient through TD-MPC2's reward head. Routing the *same scalar signal* through a prototype
  bottleneck \(c(z)^\top r_p\) adds zero new information — it adds a noisier, lower-capacity
  path to information the representation already encodes. Our diagnostics showed the behavioral
  loss decreasing smoothly while returns didn't move: the latent could always linearly decode
  reward; we just taught it to do so a second way.
- **Pairwise bisimulation *fights* the existing geometry.** A bisimulation loss is a *metric*
  constraint (latent distances must equal behavioral distances) imposed on a space that only
  needs to be *predictively sufficient*. On SimNorm simplices the L1 pairwise scale is an order
  of magnitude off the reward-difference target; at coefficient 0.1 it collapses training, at
  0.01 it still finishes last (0.549). This replicates, on a modern strong baseline, the
  large-scale finding of [arXiv:2506.00563](https://arxiv.org/abs/2506.00563) that
  behavioral-metric losses add nothing beyond self-prediction — and the fact that BS-MPC's
  published gains were against TD-MPC **v1**, before SimNorm and the v2 improvements absorbed them.

### 5.3 Proprioceptive observations leave nothing to abstract away

Abstraction earns its keep by *discarding task-irrelevant information*. A 17–24-dim
proprioceptive observation on a dense-reward locomotion task is already nearly minimal state —
there is no nuisance subspace to collapse. Our dose–response probe makes this quantitative:
inject \(k\) temporally-correlated OU nuisance dims and watch returns (CheetahRun, last-2 mean):

| nuisance dims | vanilla | behavioral Glass |
|---|---|---|
| 0 | ≈551 | ≈539 |
| 16 | 508 | ≈430 |
| 32 | 179 | ≈175 |
| 64 | 153 | 233 / 67 (two rounds) |

Below ~1× native dimensionality of nuisance, the plain encoder filters it unaided — no
abstraction needed. Above it, *both* encoders collapse identically — the abstraction doesn't
change what gradient descent can find. The window where an explicit invariance mechanism could
matter is, in this regime, empty.

### 5.4 The actual bottleneck is exploration, and representation can't move it

The sparse-task probe is the cleanest evidence: CartpoleSwingupSparse / BallInCup /
AcrobotSwingupSparse produce literal 0-vs-solved seed bimodality, determined by whether the
random action sequence ever touches reward at all. That's an *exploration* failure. Weak seeds
on dense tasks (our floor mirage) are *optimization* luck. Neither is a representation problem —
so a representation-level intervention has nothing to grab. The MPPI planner compounds this:
replanning every step against the learned model absorbs small representation-quality differences
into the search. On these benchmarks, seed outcome variance — the thing we kept reading as
signal — is generated almost entirely downstream of the representation.

### 5.5 The literature's abstraction wins live exactly where we weren't

Read the prior art with the baseline's strength in mind and a pattern emerges:

- **DBC** wins on *pixels with natural-video distractors* against reconstruction baselines.
- **BS-MPC** wins against *TD-MPC v1* (no SimNorm, smaller nets) and under distractors.
- **DC-MPC** wins on *high-dimensional Dog/Humanoid*, and its controlled "codebook inside
  TD-MPC2" ablation shows only "some improvement."
- **Hierarchical world models** (2024 study): no final-return gains; abstract-level model
  exploitation is a core failure mode.

The abstraction gap closes as the base objective approaches sufficiency and as the observation
space approaches minimal state. TD-MPC2 on proprioceptive DMC is the limit point of both trends.
Our null is one careful data point confirming the extrapolation.

## 6. What might still be promising

Ranked by our posterior after all of the above — each with the reason it survives the null:

1. **Attack exploration, not representation.** Our own data says the residual variance on these
   benchmarks is exploration/optimization luck. Intrinsic-reward mechanisms (BYOL-Explore-style
   novelty, count-free bonuses) target the actual bottleneck. This is *not* abstraction — which
   is the point.
2. **Use the abstraction for control, not just as a regularizer.** A confession: in every Glass
   variant, prototypes were *trained* but the planner still searched the full latent space. The
   abstraction never *did* anything at decision time. Planning in prototype space (coarser
   actions, longer effective horizons, cheaper MPPI) is a genuinely untested axis — and the one
   place where "abstraction" could pay in compute even at equal returns.
3. **Replace components instead of adding losses.** DC-MPC's codebook *replaces* SimNorm rather
   than regularizing alongside it. §5.1 dooms auxiliaries on a sufficient objective; changing the
   latent *parameterization* at least changes the optimization landscape. Modest wins on
   Dog/Humanoid suggest this is where the remaining (small) headroom lives.
4. **Pixels + natural distractors.** The literature's actual home turf for behavioral
   abstraction. Our synthetic-OU falsification (1.23× < 1.5 gate) lowers but does not eliminate
   this prior — structured natural video is not white-ish OU noise. Requires a GPU-native
   renderer (Madrona-MJX) to keep training fast; gate it with a 2-arm × 2-seed probe first.
5. **Multi-task / transfer.** Abstraction's value proposition is *reuse*; single-task evaluation
   cannot reveal it by design. Does prototype sharing accelerate TD-MPC2's massively-multitask
   (80-task) setting or few-shot transfer? Untested by us, and structurally immune to our null.
6. **Temporal abstraction — only on benchmarks that require it.** mujoco_playground contains no
   task with the long-horizon structural failure that hierarchy fixes (our Family-C probe:
   flat TD-MPC2 never structurally fails there; it's always seed-bimodal exploration). Crafter /
   AntMaze-class tasks are the only honest venue, and the hierarchical-world-model literature's
   negative evidence demands a pre-registered kill criterion.

The meta-lesson sharpens all six: **pick benchmarks where the mechanism's failure mode is the
bottleneck**, and kill-probe with the minimum experiment (our Stage-2a falsification cost 8
half-length runs; iteration 14 cost ~90 full ones).

## 7. If we submit this to ICLR, what should we expect?

Honest forecast, calibrated against what this paper is and isn't.

**What it is:** a rigorous multi-mechanism null with pre-registered gates on a modern SOTA
baseline, plus a six-instance anatomy of small-sample mirages with full estimate trajectories.
**What it isn't:** a new method that wins, a large-scale study (one suite, one base agent,
proprio only), or a theory paper.

**ICLR main track: expect rejection, ~10–20% acceptance probability.** Likely score profile
around 3–4 / 5 / 6 (avg ≈ 4.5–5, below the ~6 bar). Predictable reviews:

- *R1 (representation learning):* "Rigorous, but scope is narrow — one suite, proprioceptive
  only, one base agent. The literature claims gains on pixels+distractors, which is exactly the
  untested setting." (Our synthetic-distractor probe partially answers this; a reviewer will say
  OU noise ≠ natural video, and they'd be right.)
- *R2 (methods):* "A negative result without a mechanistic diagnosis at scale. Why should I
  believe n=28 vs 37 settles it when the paper's own thesis is that effects dissolve with n?"
  (Fair hit — our CIs answer it formally, but it stings rhetorically.)
- *R3 (empirical rigor):* champion review — "the field needs this; the trajectory-reporting
  recommendation should be standard."

Null-result papers historically clear ICLR only when they are *sweeping* (dozens of
methods/envs, e.g. the 2506.00563 genre) or carry a theory payload. Ours is deep but narrow.

**Better-fit venues, in order:**

1. **TMLR** — reviews for correctness, not impact. This paper is exactly TMLR-shaped:
   high acceptance odds, citable, fast. *Our recommendation.*
2. **ICLR Blogposts Track** — peer-reviewed, proceedings-indexed, explicitly created for
   analysis and negative results. The "six mirages" anatomy is an ideal fit; could run
   *alongside* a TMLR submission of the full study.
3. **Workshop track** (e.g. an RL or "negative results / reproducibility" workshop) — fast
   feedback, zero downside.

**If the user insists on ICLR main track,** the upgrade path is: (a) add a pixel+natural-video
distractor arm (Madrona), (b) add a second base agent (DreamerV3) to show the null generalizes,
(c) add representation diagnostics (probing/CKA showing the latents are already
reward-predictive — direct evidence for §5.2). That repositions it from "we failed to improve
TD-MPC2" to "an empirical study of when abstraction helps world models" — a known, acceptable
genre. Estimated cost: 4–6 weeks and a real pixel-training budget; estimated probability
post-upgrade: 35–45%.

## 8. Protocol lessons (the part worth stealing)

1. **Pre-register the gate before the data exists** — n-bar, threshold, and what "falsified"
   means. Our distractor probe died cleanly in 8 runs because the 1.5× gate was written first.
2. **Tail claims need ~3× the n of mean claims.** Our floor effect was built on 16 runs' worth
   of lower-tail; it needed 50+.
3. **Report estimate trajectories, not final tables.** A final table at any single n in our
   trajectory would have been misleading — including, we must note, possibly the final one.
4. **Distinguish procedure from architecture.** Any intervention that helps a vanilla seed too
   is a trick, not a mechanism.
5. **Minimum-falsification staging.** Cheapest probe that can kill the idea first; scale only
   on surviving signal. Iteration 14 cost ~90 mature 1M-step runs to learn what staged probes
   could have killed for a fraction.

---

*All numbers verified from raw eval CSVs (mean of final two MPPI evals, per-task normalized,
IQM with stratified-bootstrap 95% CIs, 20k resamples). Compute: ~12 vast.ai GPUs
(A4000/2080Ti/TitanV/1660S), JAX/MJX via mujoco_playground, pinned code SHA. A HopperHop 5M-step
replication pair is still running at post time; it is narrative color and cannot change the
aggregate conclusion.*

**References:** TD-MPC2 [arXiv:2310.16828](https://arxiv.org/abs/2310.16828) · Ni et al.
[arXiv:2401.08898](https://arxiv.org/abs/2401.08898) · DBC
[arXiv:2006.10742](https://arxiv.org/abs/2006.10742) · BS-MPC
[arXiv:2410.04553](https://arxiv.org/abs/2410.04553) · DC-MPC
[arXiv:2503.00653](https://arxiv.org/abs/2503.00653) · behavioral-metric study
[arXiv:2506.00563](https://arxiv.org/abs/2506.00563) · rliable
[arXiv:2108.13264](https://arxiv.org/abs/2108.13264) · SPR
[arXiv:2007.05929](https://arxiv.org/abs/2007.05929)
