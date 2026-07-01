---
title: "Thread A — Planning as a Directed-Exploration Operator (flagship, living log)"
date: 2026-07-01
description: "The flagship bet: on exploration-bottlenecked tasks, model-based planning explores the state space more effectively than model-free RL, and that exploration — not a higher ceiling — is the win. Isolate it (coverage), amplify it (novelty-seeking MPPI), direct it (SE-discovered subgoals). Living log, updated as arms land."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["world-models", "TD-MPC2", "PPO", "exploration", "planning", "structural-entropy", "living-log"]
---

{{< katex >}}

> **Living log.** One post per research thread; updated whenever an experiment finishes, a finding
> lands, or something breaks. Newest progress on top. Context: [Part 5 method map](../2026-07-01-tdmpc-glass-part5-beat-ppo-reality-check),
> [Part 6 five bets](../2026-07-02-tdmpc-glass-part6-five-bets-next-phase).

## The bet

**Hypothesis.** On exploration-bottlenecked tasks, model-based planning (a learned world model + lookahead)
reaches states — and therefore policies — that on-policy model-free RL never finds. Our sharpest datapoint is
TD-MPC2 beating PPO on `HopperHop` **367 vs 33**: that is *not* a ceiling story and not even mainly a "learning
speed" story. It's that **planning is a directed-exploration operator**. On dense/explorable tasks PPO's raw
throughput wins; on exploration-bottlenecked ones, planning explores its way to a solution PPO can't reach at any
budget.

**Why us / white space.** Plan2Explore / LEXA / MAX / Go-Explore explore via curiosity or model-disagreement, but
nobody uses a **learned abstraction (structural-entropy communities / bottlenecks) to define the exploration
targets a planner pursues**. We already own SE, a working planner, and a controlled exploration-difficulty axis
(the actuation-weakened "escape frontier"). That intersection is the paper.

**Three arms.**
- **A1 — Isolate.** Measure state-space **coverage** (occupancy entropy, distinct grid cells, #SE-communities
  visited) for planning vs a no-plan \(\pi\)-only ablation vs PPO. Show planning → coverage → success.
- **A2 — Amplify.** Add an **intrinsic-novelty bonus to the MPPI objective** (model-disagreement / latent count) —
  Plan2Explore *inside* TD-MPC2. Does it push the escape frontier to even weaker actuation?
- **A3 — Direct (SE).** Run min-2D structural entropy on the replay-buffer latent graph → communities are abstract
  states, boundaries are bottleneck subgoals → a high level plans toward *under-visited* communities.

**Metrics.** Steps-to-competence (the learning-speed axis), coverage, escape-frontier shift, final success.

## Status board

| arm | state | verdict |
|---|---|---|
| A1 mechanism (same-weights ablation, **trained** ckpt) | ✅ done | **GO (n=1 task)** — planning covers 2.2× more cells |
| A1-core (planning vs π-only, **from scratch**, w/ coverage) | ✅ done | **NULL (mildly reversed)** — no coverage gain at 80k |
| A1-full (learnable task, coverage across curve) | ✅ done | **NULL** — no coverage *or* return gain (WalkerRun) |
| A1-decisive (exploration-hard **and** learnable) | ⏳ queued | the test that actually isolates the claim |
| A2 (novelty-seeking MPPI) | ⏳ queued | — |
| A3 (SE-subgoal discovery) | ⏳ queued | — |

**Honest status of the flagship (this is trending NULL).** Across three tests, planning-as-a-directed-exploration
operator is **so far unsupported**: (a) A1-core — from scratch on HopperHop, no coverage gain (but nothing
learned); (b) A1-full — on WalkerRun (which *does* learn), no coverage gain *and* no return gain (π-only is even
non-significantly ahead); (c) the one positive, the mechanism GO's 2.2× coverage, is an **inference-time
same-weights ablation** (turn MPPI on/off at a fixed trained model) — it does *not* show that an agent *trained*
with planning explores more during learning. And the famous HopperHop **367 vs 33** is TD-MPC2-vs-**PPO**
(model/algorithm), not planning-vs-π-only within one agent — that edge was only **+1.7**. The one test that could
still rescue the thesis is the **decisive** one: an *exploration-hard AND learnable* task (CartpoleSwingupSparse
escape, or HopperHop at a longer budget), where the thesis actually predicts an effect. Until then, no GO.

## Progress log

### 2026-07-01 — A1-full: second NULL, now on a task that *learns*
Reran the de-risk on **WalkerRun** (which reaches real returns, unlike HopperHop@80k): 140k steps, PLAN vs
PI-ONLY, n=3, coverage logged across the curve. Returns went 15.6 → 603.6 (mean 320; warmup ~30k), so both
links are finally testable.

- **Planning → coverage: NULL, mildly reversed — even after warmup.** Late-training PLAN vs PI-ONLY: distinct
  bins 464 vs **501** (t=−2.75), projected entropy 4.81 vs 4.86, per-dim entropy 1.84 vs 1.87 (t=−3.11).
  Planning does not become a coverage-increasing operator once the model is competent.
- **Coverage → performance:** correlated (Pearson 0.72 overall, 0.48 post-warmup) — but that's the shared
  training-progress trend (both arms rise together), not a planning effect.
- **Per-arm final return: PLAN 479.6 ± 66.9 vs PI-ONLY 541.8 ± 65.0** — π-only is *non-significantly ahead*.
  Planning improved **neither** coverage **nor** return.

**Caveat that keeps the thesis alive (barely):** WalkerRun is a *dense* locomotion task, not
exploration-bottlenecked — so the thesis doesn't even predict a planning advantage here. This mainly confirms the
expected: no planning benefit when exploration isn't the bottleneck. But combined with A1-core, that's **two
from-scratch nulls**, and the only positive evidence is an inference-time artifact. The flagship now hinges
entirely on the **decisive** test — an *exploration-hard and learnable* task — which is the next thing to run.

### 2026-07-01 — A1-core: NULL from scratch, and it *reframes* the flagship
The clean 2-arm de-risk finished (n=3, HopperHop, 80k steps, online coverage logging wired into the training
loop with frozen bins fit on the shared random warmup so the arms are comparable).

**Link 1 — planning → coverage: NULL, mildly reversed.** Final-step PLAN vs PI-ONLY:

| coverage metric | PLAN | PI-ONLY | Δ |
|---|---|---|---|
| distinct occupancy bins | 614±123 | **685±151** | −71 (n.s.) |
| projected occupancy entropy | 4.87±0.20 | **5.24±0.20** | −0.37 (t=−2.28, *wrong direction*) |
| mean per-dim entropy | 1.65±0.11 | **1.72±0.05** | −0.07 (n.s.) |

The sign is consistent at *every* checkpoint from 20k on — PI-ONLY covers as much or *more* state than PLAN.
**Link 2 — coverage → performance: untestable.** Returns are floor-heavy (mean 2.64, mostly 0); neither arm
reaches a learning regime in 80k steps, so there's nothing to correlate (Pearson r = 0.105, n.s.).

**Why (from the logs, not fabricated):** early in training the world model is untrained (`mpc=0.0`,
`mppi_return ≈ 0`), so MPPI actions are conservative model-noise while the policy prior + exploration noise
disperses at least as widely. This is not a refutation of the thesis — it's a **scope correction**: planning
can only *be* a directed-exploration operator once the model is good enough to plan through. The mechanism GO
(2.2× coverage) used a trained 750k checkpoint; this null used a cold-start model. Both are true.

**Next (A1-full, reframed):** pick a task/budget where TD-MPC2 actually reaches separable nonzero returns, run
PLAN vs PI-ONLY (and PPO) to competence, and measure coverage **across the training curve** — the prediction is
that PLAN pulls ahead on coverage (and return) *after* the model warms up, and that coverage then predicts the
gap. Single-task, single-budget caveats on the null are explicit.

### 2026-07-01 — A1-core running; a process fix
Launched the clean 2-arm de-risk on b3060b (b3060b's TD-MPC2 can run DMControl): **PLAN** (full MPPI) vs
**PI-ONLY** (identical model/budget, act with the policy prior, no planning), from scratch, with online
coverage logging. The PLAN wave (n=3 seeds) finished cleanly. **Process hiccup, fixed honestly:** the first
agent tried to self-resume via a shell watcher to run the PI-ONLY wave — that doesn't work (background agents
only resume via the parent), so it stalled after PLAN. Relaunched a completion agent that blocks to the end,
audits whether coverage was actually logged online (no checkpoints exist to recover it post-hoc), runs PI-ONLY,
and writes the verdict. **No numbers claimed until that lands.**

### 2026-07-02 — A1 mechanism: PARTIAL GO (n=1 task, clean)
Isolated the mechanism with a **same-weights** ablation on a `PandaPickCubeOrientation` 750k checkpoint —
MPPI planning ON vs OFF at *identical* weights (so any difference is planning, not training stage):

| metric (plan ON vs OFF) | ON | OFF | ratio |
|---|---|---|---|
| distinct grid cells | 222 | 100 | **2.2×** |
| occupancy entropy | 4.74 | 3.61 | — |
| 10-NN dispersion | 3.84 | 1.37 | **2.8×** |
| return | 405 | 349 | — |

**Planning explores far more state, and coverage tracks return.** The agent caught and avoided a real confound:
comparing `best_pi@200k` vs `best_mppi@750k` conflates training-stage with planning and produces a spurious
near-parity — only the same-weights ablation isolates the planning effect.

**What is *not* yet de-risked (the headline):** `HopperHop`/`CartpoleSwingupSparse` have no usable checkpoints
(we never `--save_full_state`, leaving only CSV logs) and no saved PPO models, so the 3-arm
{TD-MPC2 / π-only / PPO} coverage test needs a **fresh instrumented training run**. And the famous 367-vs-33 is
TD-MPC2-vs-PPO; the planning-vs-π-only *return* edge on HopperHop is only +1.7 (n=31, 77% of seeds) — so the
claim must be **carried by coverage, not return**. That is exactly what A1-core (now) and A1-full (next) test.

**Next:** A1-full — train TD-MPC2(plan) / π-only / PPO from scratch on HopperHop + the escape frontier *with* the
coverage logger; show coverage(plan) ≫ coverage(PPO) during the exploration phase and that it predicts the final
gap. Reusable logger (`scripts/coverage_rollout.py`) is ready; queued for the next free b3060 slot.
