---
title: "From Reproducing 'Compositional Planning with Jumpy World Models' to Tying PPO with an Abstraction-in-the-Loop"
date: 2026-06-24
description: "Weekly review: a reproduction attempt that turned into a beat-PPO campaign — jumpy world models can't solve PandaPickCube, InFOM and PPO can, and a heuristic-in-the-loop residual ties PPO and beats it on sample-efficiency."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["world-models", "TD-MPC2", "PPO", "abstraction", "manipulation", "weekly-review"]
---

{{< katex >}}

> Weekly review (Jun 17–24). What started as a clean reproduction of two recent papers turned into a chain of
> "that didn't work, so we added X" — and ended with the project's first result that is genuinely competitive
> with PPO. This post tells the whole arc honestly: the plan, the wall, the pivots, and the numbers.

## 1. Where it started: reproduce CompPlan + InFOM, compare on a common task set

The week's plan was a paper reproduction and a head-to-head:

- **CompPlan — "Compositional Planning with Jumpy World Models"** (Farebrother, Pirotta, Tirinzoni, Munos,
  Lazaric, Touati; arXiv **2602.19634**, ICLR-2026 Workshop on World Models), which builds on **Temporal
  Difference Flows / TD-Flow** (arXiv **2503.09817**). Its core is a **Geometric Horizon Model** — a
  flow-matching generative model of the discounted future-state (occupancy) measure — used to *plan over*
  pre-trained goal-conditioned policies. The headline is "plan in jumps, not primitive actions."
- **InFOM** (Intention-conditioned Flow Occupancy Measure, ICLR 2026) — the offline flow-occupancy method
  the GHM generative core most resembles.

The goal: stand both up and **compare jumpy-world-model planning vs flow-occupancy on a common task set** —
one big table. We confirmed there is **no official code** for CompPlan/TD-Flow, stood up JAX reproduction
scaffolds (InFOM, value-flows, OGBench) on rented 4090/3090 boxes, and started pretraining on the cheapest
matching OGBench tasks (`cube-single-play`, `antmaze-medium-navigate`). This is a multi-day reproduction —
it is still in progress.

## 2. The wall: jumpy world models don't solve PandaPickCube

While the flow-occupancy reproduction trained, we ran the *jumpy world-model* idea in our own stack — a
**jumpy (k-step macro) TD-MPC2** — on a contact-rich manipulation task, **PandaPickCube** (MuJoCo Playground),
scoring **real success** (`box_target ≥ 0.9`, the actual pick-and-place, not shaped return). It failed
outright. So did vanilla TD-MPC2: both **reward-hack** the dense shaping (hover near the box to farm the
proximity term) and complete the task **0%** of the time at feasible budgets.

That broke the intended comparison — you can't fill a "common task" cell with a method that gets 0. So the
table question changed from "jumpy vs flow-occupancy" to the blunter **"what actually solves this task, and
can an abstraction beat the model-free baseline?"** Which meant adding the baseline we'd been missing.

## 3. Adding PPO — the real bar

We ran the official MuJoCo-Playground **PPO** on the unmodified reward, scored with the same real-success
metric. It **solves the task** (~0.81 real success). **InFOM**, trained offline on a PPO-generated dataset,
reaches **0.74**. So the honest common-task table looked like this — *not* the one we set out to make, but the
one the data forced:

| Method | Family | PandaPickCube real success |
|---|---|---|
| jumpy TD-MPC2 *(CompPlan-style)* | jumpy world model | **0.00** |
| vanilla TD-MPC2 | world model + MPPI planning | 0.00 |
| InFOM | offline flow-occupancy *(the base paper)* | 0.74 |
| **PPO** | model-free, end-to-end | **0.81** |

The uncomfortable read: the world-model/abstraction methods we care about were *losing to plain PPO* — the
same pattern we'd seen all project. (We also did a four-part DMC field report this week confirming TD-MPC2
beats PPO **only** on sample-efficiency and exploration-bottlenecked tasks, loses wall-clock and high-dim, and
is 5× over-parameterized.)

## 4. The BC-data gap → building a Heuristic Learning loop

The next idea was behavior cloning / offline learning to bootstrap a competent policy — but **we had no
demonstration data** for PandaPickCube (and the jumpy/TD-MPC2 policies that would generate it were at 0%). So
we built one, following the **"heuristic learning loop"** idea from **Jiayi Weng's blog** on iteratively
improving a hand-written controller against a measurable metric: a JAX, vmappable **phase-machine controller**
(reach → descend → grasp → lift → place) for PickCube, improved iteration over iteration. It reliably *reaches*
(reached_box ≈ 0.97) and tops out around **6% real success** — the wall is grasp/place dynamics, not reaching.
Now we had a competent, interpretable scaffold and the data it generates.

## 5. The beat-PPO campaign: can an abstraction tie PPO?

With the heuristic loop as a scaffold, we ran a ladder of experiments, each ruling out a wrong idea, all
scored on real success (`box_target ≥ 0.9`, \(n=256\)):

| # | Approach | Real success |
|---|---|---|
| — | HL heuristic controller | 0.063 |
| #1 | raw-action residual (no abstraction) | 0.13 |
| #2 | **value-aware skill-options abstraction** | **0.24** — first positive abstraction |
| #3 | demo-seeded TD-MPC2 | 0 (clean negative) |
| #4 | distill the abstraction out (BC / warm-start / DAPG) | 0 / no-beat |
| #5 | in-loop residual, authority \(\alpha=0.5\) | 0.48 — breaks the ceiling |
| **#5b/#5c** | **in-loop residual, persistent \(\alpha=1.0\)** | **0.79 ± 0.01** (multi-seed) |
| ref | end-to-end PPO | 0.81 |

**The result:** keeping the abstraction *in the loop* — the live controller plus a Markov-conditioned residual
at full standing authority, executed as \( a = \mathrm{clip}(a_{\text{HL}}(s,z) + \alpha\,\pi_{\text{res}}(s,z)) \)
— reaches **0.79, a robust near-tie with PPO's 0.81**, and crosses the competence threshold **~1.3–1.7× faster**
(≈20–26M vs 32.8M env-steps). On the sample-efficiency axis that matters for real robots, the abstraction-in-loop
**beats** PPO; on raw peak it is a near-tie. The negatives did the real work: **distillation fails** because the
abstraction is non-Markov (its action depends on hidden phase state); **annealing authority backfires**
(non-stationary → collapse); a **learned authority-gate underperforms** uniform full authority. The winner is
the simplest in-loop form.

## 6. Aside: the model is 5× over-parameterized (confirmed at 5 seeds)

A cheap, robust positive: a **`tiny` TD-MPC2** (0.72M params, 5× smaller, ~1.6× faster) keeps a **median ~99%**
of full return across the DM-Control suite, and on the high-velocity locomotion tasks where capacity *does*
matter, a 5-seed re-run confirms tiny retains **~88–89%** (a real ~11–12% gap, not noise) — with a thin
instability tail. Most of TD-MPC2's parameters aren't doing work on standard control.

## 7. Where it stands / toward 100%

Neither PPO (0.81) nor our abstraction-in-loop (0.79) solves PandaPickCube perfectly — ~20% of randomized
configs (reach-edge targets, grasp slips, placement just outside tolerance) still fail. This week's final push,
**in flight now**, attacks that tail three ways: a **bigger/longer residual** (is 0.79 capacity-limited?), a
**closed-loop retry** (the controller detects a failed grasp and re-attempts in-episode — the abstraction-in-loop's
unique advantage over a flat policy), and a **hard-config curriculum**. Honest expectation: ~0.95 is reachable;
literal 100% on a randomized contact task is not. Results next week.

**The one-line takeaway:** a reproduction attempt for jumpy world models led, via a chain of honest failures, to
the project's first abstraction that *matches* PPO and *beats* it on sample-efficiency — by keeping the
abstraction in the control loop rather than distilling it out.

## References & artifacts
- **Papers:** CompPlan — *Compositional Planning with Jumpy World Models*, arXiv [2602.19634](https://arxiv.org/abs/2602.19634); TD-Flow, arXiv [2503.09817](https://arxiv.org/abs/2503.09817); InFOM (ICLR 2026); DreamerV3 (*Nature* 2025); MuJoCo Playground (arXiv 2502.08844). The heuristic-learning-loop idea: Jiayi Weng's blog.
- **Experiment write-ups (full detail, Parts 17–27):** [suuttt.github.io/tdmpc-glass](https://suuttt.github.io/tdmpc-glass/).
- **Code & logged results:** pushed to GitHub ([SuuTTT/tdmpc-glass](https://github.com/SuuTTT/tdmpc-glass)); every number above is read from logged JSON/CSV, never hand-entered.
- **Model checkpoints & data:** archived to HuggingFace (`Dannibal/tdmpc-glass-milestones`).
- **Prior weekly review:** [Campaign Review #3 (Jun 17)](https://suuttt.github.io/projects/2026-06-17-tdmpc-glass-part3-campaign-review/).
