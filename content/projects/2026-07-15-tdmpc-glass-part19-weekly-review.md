---
title: "TD-MPC-Glass, Part 19: A Week in Review — Turning Three Dissections Into Instruments"
date: 2026-07-15
description: "The week of July 8–15, reviewed end to end. Last week (Part 10) left three papers as dissections and a pile of open 'why' questions. This week turned each dissection into a measured instrument: a value-sufficiency bottleneck that fingerprints every task in three shapes; a collection-mode dissociation that inverts on Cheetah (removing the world model HELPS) with one variance-inflation mechanism behind both regimes; and a first data point on the long-deferred JEPA/SE line — uniformity vs VICReg on a collapse-prone task, a clean null. The recurring law from Part 10 held again: explicit structure is redundant exactly where the value objective already supplies it."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["world-models", "TD-MPC2", "PPO", "abstraction", "JEPA", "VICReg", "mujoco-playground", "reproducibility", "weekly-review", "vastai"]
---

{{< katex >}}

> A review of one week — July 8 through July 15 — picking up exactly where
> [Part 10](https://suuttt.github.io/projects/2026-07-08-tdmpc-glass-weekly-review-part10/) left off.
> Part 10 closed with three papers framed as *dissections* (a wall, a mechanism, a sufficiency law) and a
> recurring one-line result: **abstraction and reweighting buy nothing the value pathway doesn't already use.**
> The obvious risk with a "dissection" is that it stays anecdotal. This week was about converting each one into a
> **measured instrument** — a knob you can turn on any task/model and read a number off. Every figure below is read
> from disk; the nulls are reported as loudly as the positives.

## Where last week left us (the motivation)

Part 10 had three-act shape: five bets to beat PPO with abstraction, almost all nulled, then a pivot to dissecting
*why* the planner already wins. It produced three papers and a set of unanswered "why" questions I flagged for this
week:

- **Q5 — why does TD-MPC2 clear HopperHop when SAC can't?** (the categorical exploration wall)
- **Q6 — why did abstraction only ever help HopperHop and navigation, nowhere else?**
- **Q8 — the tension:** HopperHop is simultaneously the *sharpest* PPO wall *and* the only cell where the world
  model turned out **removable**. How can the hardest task be the one where the model matters least?
- **Q2 / Q7 — the JEPA/SE debt:** the pure-JEPA and structural-entropy bets (Bets D, E) reversed or hurt back in
  Part 7, and the whole "anti-collapse structure" line had been parked ever since.

Three of these got answered from data this week. The fourth (the JEPA debt) finally got a number instead of a shrug.

## Thread 1 — a value-sufficiency instrument for Paper A (answers Q6)

Paper A ("when is explicit abstraction redundant") had a hole: its probe was **decode-\(R^2\)** — how well the
latent reconstructs the observation. That probe *saturates* for both a strong policy and a collapsed one, so it
cannot tell you whether an abstraction objective has room to help. It was measuring the wrong thing.

The replacement is a **value-sufficiency bottleneck (VBN)**: insert a width-\(D\) bottleneck between the latent and
the value head, sweep \(D \in \{16,32,64,128\}\) against the unmodified agent, and read the *shape* of the resulting
curve. That shape is a per-task fingerprint of **how much of the latent the value function actually needs** — which
is the quantity that decides whether added structure can reorganize anything.

Three tasks, three qualitatively distinct fingerprints (final MPPI return, 5M steps):

| Task | D=16 | D=32 | D=64 | D=128 | vanilla | fingerprint |
|---|---|---|---|---|---|---|
| CheetahRun (n=5) | 548 (64%) | 589 (69%) | 627 (73%) | 726 (85%) | 855 | **strictly monotone** — no width suffices |
| WalkerRun (n=5) | 625 (86%) | 643 (88%) | 666 (92%) | 694 (95%) | 727 | **flat-high** — most compressible; D=16 already 86% |
| AcrobotSwingup (n=6, medians) | 211 (41%) | 251 (49%) | 311 (61%) | 304 (59%) | 511 | **ramp-to-D64** — least compressible; saturates at D=64, D=128 adds nothing |

And the instrument agrees with the intervention. The stripped-vs-full sufficiency ablation (delete the consistency
loss, keep everything else) ranks the tasks the *same way* the VBN fingerprint does:

$$\text{HopperHop } 0\% \;<\; \text{WalkerRun } {-}7.5\% \;<\; \text{CheetahRun } {-}23.8\% \;<\; \text{AcrobotSwingup } {-}44\%.$$

That co-ranking is the answer to **Q6**: abstraction helps exactly where the value function needs *little* of the
latent (the flat-high, highly-compressible regime), because that is where an added structural objective has slack to
reorganize a nearly-sufficient code. The criterion is **value-information compressibility, not horizon length.** The
VBN curve is checkpoint-cheap and predictive — it is the valid replacement for decode-\(R^2\).

> A note on rigor: the Acrobot fingerprint changed *during* the week and I want to flag why, because it is the kind
> of thing that quietly corrupts a table. An early read called it "step-at-128." On re-harvest, two of the seeds had
> only reached ~2.7M steps and were dragging the means; on the six complete-at-5M seeds the curve is a smooth ramp
> saturating at D=64. Same qualitative story (least-compressible, needs a mid-width, tops out well below vanilla),
> corrected location. Report medians — Acrobot has two total-collapse seed×width cells and the mean is not robust.

## Thread 2 — the collection-mode dissociation and an inversion for Paper 3 (answers Q5, Q8)

Our JAX variant collects data with the *policy*; the canonical TD-MPC2 collects with the *planner* (MPPI rolls the
world model to **act**, not only to score). If the world model's value comes from planning, then collection mode
should *modulate* how load-bearing the model is. So we re-ran the stripped-vs-full contrast under
**planner-collection** on three tasks. The result is a double dissociation with a genuine inversion (finals @2.5M,
now at **n=9**):

| Task | full (median) | stripped (median) | Δ | reading |
|---|---|---|---|---|
| HopperHop (n=5) | ~455 | ~468 | ≈0 | removable — both arms stable |
| WalkerRun (n=9) | **739** | **606** | **−18.0%** | load-bearing — higher but volatile |
| CheetahRun (n=9) | **327** | **475** | **+45% (stripped > full)** | **inversion** — the model is *actively harmful* |

On CheetahRun under planner-collection, removing the world model's accuracy doesn't merely fail to hurt — it
**helps by 45%**. We had *pre-registered* the opposite (stripped would degrade ≥15%); the falsification is the
spine of the section. And this is not a fluke of one metric: the inversion magnitude settled from +104% (n=5) to a
stable **+45% at n=9**, direction never wavering.

The mechanism turned out to be readable straight off the eval traces, no new experiment required. The full model
**inflates eval-return variance ~3× on both tasks** under planner-collection (the poisoned-planner-target signature
— MPPI selecting actions on a model that periodically hallucinates value). The two tasks differ only in the *scale
of that variance relative to the mean*:

- On **Walker**, the model lifts the mean to ~740, so the ~3× swings stay net-positive → higher-but-volatile.
- On **Cheetah**, the swings are large enough that final returns dip to ~117, dragging the mean *below* the stable
  stripped model (~475) → variance tips into net harm.

**One mechanism (the world model inflates the planner's target variance ~3×), two regimes, set by the mean/variance
ratio.** That also dissolves **Q8**: HopperHop can be both the sharpest PPO wall and the only removable cell because
stripping the consistency loss removes the model's *accuracy* but not the *planner*. TD-MPC2 clears Hopper via
planning-over-a-rough-model plus off-policy data (**Q5**: deterministic actor, no entropy term — the entropy grid
has SAC failing Hopper 0/9 while the planner-free TD core is 8/8), and the model is removable *because the planner
carries it*. The wall itself is the conjunctive reward, independent of the model.

## Thread 3 — paying down the JEPA/SE debt (a first number for Q2/Q7)

The anti-collapse line (Bets D and E) had been parked since Part 7. Mid-week I pulled it back in and gave it the same
treatment as the others: pick the task the VBN instrument flags as **least compressible / most collapse-prone**
(CheetahRun — the strictly-monotone fingerprint) and run the two canonical anti-collapse levers head to head against
a matched vanilla baseline:

| arm | CheetahRun (last-6 median) | vs vanilla (~818) |
|---|---|---|
| **uniformity** (hypersphere) | 725.8 | −11.3% |
| **VICReg** (variance-covariance) | 751.3 | −8.2% |
| vanilla | ~818 | — |

Uniformity ≈ VICReg (within seed noise), and **both sit slightly *below* vanilla** (n=2; a refill to n=4 is running).
A null — and, importantly, the *expected* null. It is the JEPA-line instance of Paper A's central claim: TD-MPC2's
latent is already shaped by the value/TD objective, which prevents the degenerate collapse these regularizers exist
to defend against. Adding an explicit anti-collapse term is therefore redundant (and mildly harmful, via an extra
loss competing with the value signal). This matches the earlier H-JEPA and SE-structured-JEPA nulls on Panda — the
same law, now with a clean DMControl data point instead of a hand-wave.

## The one-line law, again

Three independent threads, one recurring result:

> **Explicit structure — a reconstruction bottleneck, a consistency loss, an anti-collapse penalty — is redundant
> exactly where the value objective already supplies what it would add; and where the value pathway is *not*
> nearly-sufficient, adding structure buys volatility, not a ceiling.**

The novelty this week is that each of those is now an *instrument*, not an anecdote: a VBN curve you can read per
task, a collection-mode × variance mechanism you can compute from eval traces, and an anti-collapse comparison with
a baseline. Both paper cores (Paper A: sufficiency; Paper 3: the mechanism/inversion) are compute-complete at their
final \(n\); what remains is writing.

## Next

- **Freeze compute, write.** Papers A and 3 are data-complete; the deadline work is prose and figures, not GPUs.
- **JEPA #59 to n=4** finishes the one open confirmatory run.
- The **fleet is over-provisioned for what's left** — eight RTX 3060s, half of them now idle because their sweeps
  are done. The honest bottleneck this week was never total GPU count; it was that small DMControl models
  underutilize a 3060 and we serialize two widths per card. If we open a *new* experimental front next week
  (resuming the JEPA/SE program properly, or a confirmatory planner-target probe for Paper 3), the right move is a
  small number of *faster* cards run one-job-per-GPU, not more 3060s tightening error bars on finished science.
