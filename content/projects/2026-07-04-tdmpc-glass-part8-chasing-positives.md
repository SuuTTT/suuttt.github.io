---
title: "TD-MPC-Glass, Part 8: Chasing Positives — What the Re-Run Found"
date: 2026-07-04
description: "After Part 7's mostly-null five bets, we re-ran each with stronger methods and one sharp new question: is TD-MPC2's edge on exploration-hard tasks sample-efficiency or genuine exploration? The chase produced one reframe-level positive — the world model, not the planner, is the exploration lever (PPO can't crack HopperHop at 94× the budget) — plus one localized positive (learned hierarchy on multi-room nav). The 'add novelty to planning' ideas nulled. Honest scorecard."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["world-models", "TD-MPC2", "PPO", "exploration", "planning", "hierarchy", "structural-entropy", "reinforcement-learning"]
---

{{< katex >}}

> A positive-chase. After [Part 7](../2026-07-03-tdmpc-glass-part7-five-bets-resolved) closed the five bets
> mostly null, we re-ran each with a stronger method and added the question that turned out to matter most:
> *is TD-MPC2's win on exploration-hard tasks sample-efficiency, or real exploration?* Everything below is
> disk-verified with matched controls; two confirmations are still running (flagged).

## Scorecard

| bet / test | verdict |
|---|---|
| **Planning-as-exploration → the real question** | **POSITIVE (reframe): the *world model* is the exploration lever** |
| **Learned hierarchy (feudal) vs flat** | **POSITIVE, localized** — wins on multi-room, not open rooms |
| A2 — novelty-seeking MPPI | null / harmful |
| D — SE as *structure* (not penalty) | null |
| E — SE-subgoal option discovery | null |
| MiniGrid — PPO+RND on hard-exploration | null |

## The win: the world model, not the planner, is what explores

Part 7 showed planning (MPPI) is *not* a directed-exploration operator — a policy-only ablation of TD-MPC2
explored/discovered just as well. That left the real question open: **why does TD-MPC2 crush PPO on `HopperHop`
(367 vs 33)?** Sample-efficiency, or exploration?

We settled it by running **PPO at a huge budget**: brax PPO, 5 seeds, up to **472M env-steps ≈ 94× the ~5M
TD-MPC2 needs to reach 367**.

| | peak (mean / best) | seeds crossing 200 / 367 |
|---|---|---|
| **PPO @ 472M** | **40.0 / 53.8** | **0 / 5** |
| TD-MPC2 @ ~5M | ~367 | — |

PPO at 94× the samples **never gets past ~54** — a genuine **exploration wall**, not slow learning. Combined with
the planning-vs-policy null, this pins the mechanism:

> **Planning (MPPI) is a *pruning / exploitation* operator; the *world model* (learned latent dynamics + value
> from imagined rollouts) is the *exploration* lever.** It lets TD-MPC2 discover a gait that model-free PPO cannot
> find at any practical budget.

This reframes the whole flagship honestly: the exploration advantage is real, but it lives in the **model**, not
the **planner**. *(Two confirmations running: a **generalization** sweep — does the wall hold on Pendulum /
Finger / BallInCup, where TD-MPC2 also succeeds? early signal: PPO ~15–55 vs TD-MPC2 ~911, leaning yes — and a
**per-head ablation** to say which of the ~5 world-model nets carries it. Results fold in when done.)*

## The localized positive: hierarchy helps where it should

A fully-**learned** 2-level feudal hierarchy (learned high-level subgoal emitter + goal-conditioned low level)
vs a matched-budget flat baseline, on sparse navigation. The honest, verification-checked result:

- **Multi-room maze (`fourroom`, doorway bottlenecks), n=6:** **feudal solves 4/6 seeds, flat 0/6.** Flat
  literally cannot cross the connected rooms; the hierarchy's committed subgoals do.
- **Open rooms (n=3):** feudal ≥ flat but **within seed variance** (bigroom is a tie) — flat is a competent
  baseline there. *(An earlier n=1 "feudal 1.0 vs flat 0.0" was seed luck; firming to n=3 corrected it.)*

So the positive is **localized and mechanistic**: learned hierarchy earns its keep on **bottleneck / multi-room**
structure, not on open long-horizon rooms. That's a more defensible claim than "hierarchy beats flat."

## The nulls (honest, and consistent)

- **A2 — novelty-seeking MPPI** (RND / disagreement bonus in the MPPI objective): null-to-harmful. One n=2 bump on
  `CartpoleSwingupSparse` is within that task's huge seed variance; on dense tasks novelty **hurts**
  (`PendulumSwingup` 766 → 135). Adding novelty to the planner over-widens the search and wrecks exploitation.
- **D — SE as *structure*** (community-contrastive / encoding-tree, not a penalty): doesn't beat plain JEPA
  `none` on any DMControl readout.
- **E — SE-subgoal options**: SE-community goal-shaping ties or hurts flat; worst on the sparse task it was meant
  to help.
- **MiniGrid** (`MultiRoom-N6`, `KeyCorridorS3R3`): PPO **and** PPO+RND both stall at 0 success. RND explores more
  (coverage/discovery up) but **coverage ≠ task success** — exactly why RIDE/NovelD/AMIGo exist.

A consistent thread runs through the nulls: **novelty widens coverage but does not, by itself, solve
hard-credit tasks** — and bolting novelty onto a planner mostly costs exploitation.

## Takeaway

Chasing positives on purpose turned one Part-7 null into a sharper, *truer* positive: not "planning explores,"
but **"the world model explores; the planner exploits."** Plus a clean, localized hierarchy win. The
novelty-injection ideas (A2/E/MiniGrid) and SE-as-structure (D) are honest nulls that tighten the story rather
than pad it. The two running confirmations (wall generalization + per-head ablation) will finish the picture.
