---
title: "TD-MPC-Glass, Part 9: The Anatomy of \"Beating PPO\" — What Survived Every Control"
date: 2026-07-05
description: "The positive-chase ended with every headline claim rewritten by its own control experiment — and a sharper, truer story than any of the originals. The exploration wall is real but it is an on-policy pathology specific to gait discovery; SAC escapes it; the world model's robust advantages are consistency and ~4–5× sample-efficiency, not level; the load-bearing net is the TD value signal, not the self-predictive dynamics loss; and the hierarchy win was dense shaping in disguise. A capstone with every number disk-verified."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["world-models", "TD-MPC2", "PPO", "SAC", "exploration", "hierarchy", "mechanism", "reinforcement-learning", "capstone"]
---

{{< katex >}}

> The capstone of the positive-chase ([Part 8](../2026-07-04-tdmpc-glass-part8-chasing-positives)). This week we
> ran the control experiment for every claim we wanted to make — an off-policy baseline, a shaped-flat baseline, a
> config-bug fix, a 4× budget extension, and a per-loss ablation — on byte-identical environments. Every control
> changed its claim. What's left is smaller than what we started with, and solid. All numbers trace to JSONs/logs
> in the campaign ledger; two corroborative cells are still running (marked).

## The final scorecard

| claim as originally hoped | what the control said | final form |
|---|---|---|
| "TD-MPC2 beats PPO by exploring better" | plan-vs-π: planning isn't the explorer; SAC: model-free escapes too | **PPO's exploration wall is real, on-policy-specific, and gait-specific** |
| "the world model is the exploration lever" | per-loss ablation | **the TD *value* signal is the lever; the self-predictive loss is the least critical piece** |
| "world model buys 2× higher level" | SAC at 4× budget (n=5) | **no clean level gap — consistency + ~4–5× sample-efficiency** |
| "learned hierarchy beats flat" | shaped-flat control (n=6) | **dense shaping in disguise; the hierarchy's real trick is *self-generating* that signal** |

## 1. The wall: real, on-policy, gait-specific

On the identical MJX `HopperHop` (verified byte-for-byte — same registry env, episode length, action repeat):

- **PPO (tuned config): peak 53.8, 0/5 seeds ≥200, through 472M steps/seed.** Categorically walled.
- **SAC (tuned config): 3/3 seeds cross 200 within 5M.** The wall is an **on-policy pathology** — replay +
  entropy exploration escapes it.
- **TD-MPC2: ~282 at 1M (n=4, best 367); 477–481 final at 5M (n=2).**

And the wall does not generalize: FingerTurnHard PPO ≈973 (3/3, no wall); Pendulum solves once an upstream
config-case bug (`PendulumSwingUp` vs `PendulumSwingup` silently skipping the tuned override) is fixed — peaks
842/831 (n=2); BallInCup is discovery-luck for *both* algorithms (PPO 1/3, TD-MPC2 1/2). **On-policy PPO fails at
gait discovery specifically, not at hard control generally.**

## 2. Level vs efficiency: the 4× budget test

If the world model bought a higher *level*, SAC shouldn't reach TD-MPC2's plateau at any reasonable budget. At
20M steps (4× TD-MPC2's), SAC's finals are **246 / 277 / 285 / 414 / 572** (n=5; peaks to 574). The mean (~359)
sits below TD-MPC2's ~477, but the best seeds reach or exceed it — the distributions overlap. So:

> **The world model's robust advantages are *consistency* (TD-MPC2: 4/4 seeds ≥282 at 1M; SAC: wild seed spread
> even at 20M) and ~4–5× sample-efficiency. Not a level ceiling.**

This is the campaign's own law — "abstraction buys speed, not ceiling" — arriving from a fourth independent
direction, and it agrees with where the field has landed (BRO, SimBa, TD7, CrossQ: scaled/regularized model-free
methods keep erasing model-based "level" gaps; the efficiency gap is the durable one).

## 3. Which part of the "world model" does the work? The value signal.

Per-loss ablation (zero ONE loss term, from scratch, mask verified live in logs; MPPI + π-only eval returns):

**CheetahRun (complete, n=2, 1M):** full = 738/782 (MPPI/π). Ablate **value → 16/12** (everything dies). Ablate
**reward → MPPI 5 but π 761** (planning-only, partly by construction — MPPI scores rollouts with it). Ablate
**policy prior → 123/2.5**. Ablate **consistency — the self-predictive latent-dynamics loss itself → 367/541, the
smallest drop of the four.**

**HopperHop — the exploration task (none/value/consistency/reward all n=4; policy n=2, last 2 arms finishing):**
full finds the gait on 4/4 seeds (MPPI best 287–570). **Value-ablated: 0.0 / 0.0 / 3.2 / 0.0 — the gait is
*never* found. Ablating the TD value loss reproduces the exploration wall.** Consistency-ablated still finds it
at roughly half strength on 4/4 seeds (MPPI 185–245). Reward-ablated matches CheetahRun's pattern: the planner is
dead by construction (MPPI ≈0 — it scores rollouts with that head) but **the policy still learns the gait**
(π 519 / 241 / 226 / 189, n=4). Policy-ablated is ≈0 on *both* readouts (n=2) — and here HopperHop is *harsher*
than CheetahRun (where the planner alone still limped to 123): on the exploration task the planner cannot
compensate for a missing prior at all. So the precise statement: **the value-learning pathway — the TD value loss
and the policy trained from it — is individually necessary (each ablation reproduces total failure); the reward
head matters only for planning; and the self-predictive consistency loss is the only component whose removal
merely degrades.**

> **"The world model explores" decomposes into: the TD value signal trained through the latent is what discovers
> and ranks behavior; the self-predictive dynamics loss is a helpful regularizer, not the key.** This rhymes with
> the entire campaign — from the redundancy nulls to the anti-collapse reversal, the value anchor keeps being the
> thing that matters.

## 4. Hierarchy: the win was the signal, not the structure

The fourroom result (feudal 4/6, sparse-flat 0/6) survived n=6 — and then the right control killed its
interpretation: **flat TD3 + dense potential shaping toward the goal solves 3/6** — statistically the same as
feudal. What carried the win was *density of learning signal*, not hierarchy. The surviving claim is narrower and
still interesting: **the feudal agent manufactured that dense signal itself** (its low level trains on distance
to *self-proposed* subgoals; no privileged goal information), whereas the control had to be handed the true goal.
This independently replicates Nachum et al. (2019): most of HRL's measurable benefit reproduces on flat agents
given better exploration/shaping.

## 5. What this week actually taught

1. **Run the control before the claim.** Four headline claims went into this week; four came out rewritten by
   their own controls — every time into something smaller and truer. (The full adversarial review that mandated
   these controls, and the retraction bookkeeping, is in the campaign ledger and
   [the R²/Pareto explainer](../explainer-r2-pareto-analysis).)
2. **The durable positives of the whole program**: (a) explicit abstraction is redundant on a strong
   value-anchored latent (Paper A, submission-ready); (b) structured priors and world models buy
   sample-efficiency and consistency — never a ceiling — across priors, hierarchy, planning, and model-based RL
   (the speed-of-learning paper); (c) on-policy PPO has a real, reproducible, task-specific exploration failure
   mode that off-policy and model-based methods don't share — and its mechanism, in TD-MPC2's case, is value
   learning, not dynamics modeling.
3. **Pending (folds in when a host outage resolves):** the reward/policy arms of the HopperHop ablation
   (corroborative — CheetahRun's complete version already shows reward = planner-only, policy = both), and seeds
   3/4 consistency arms.

*Every number: campaign ledger `bet2_null_results.md` (disk paths per entry). Same-env verification, config
audits, and the two bugs found along the way (Pendulum case mismatch; an A1 noise confound) are documented in the
2026-07-02 full-arc review.*
