---
title: "TD-MPC-Glass, Part 5: The Beat-PPO Reality Check — We Solve Panda, PPO Solves the Hard Ones"
date: 2026-07-01
description: "Weekly review (Jun 24 – Jul 1). We finished the Panda story (a learned residual breaks the analytic contact ceiling but matched PPO still wins the asymptote — a prior buys sample-efficiency, not a higher ceiling), closed the anti-collapse question (the right regularizer is downstream-dependent), then went hunting for new environments where we beat PPO. We didn't find one — and the reason is instructive: a return-based scan nearly produced a fake 'beat' (twice, including my own over-report), until we scored real success and confirmed the MuJoCo Playground PPO genuinely solves the hard dexterous/manipulation tasks (0.81–0.99) that our TD-MPC2 fails at a practical budget. An honest map of exactly where each method wins."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["world-models", "TD-MPC2", "PPO", "abstraction", "manipulation", "dexterous", "mujoco-playground", "reproducibility", "weekly-review", "vastai"]
---

{{< katex >}}

> Weekly review (Jun 24 – Jul 1). Part 4 ended with a heuristic-in-the-loop residual *tying* PPO and beating it
> on sample-efficiency. This week we pushed on three fronts — finish the Panda story, close the anti-collapse
> question, and go find a *new* environment where we beat PPO outright. The last one is the interesting failure:
> it forced us to score **real task success** instead of return, which caught a reward-gaming mirage (mine
> included) and produced a clean, honest map of where model-based planning wins and where PPO's throughput wins.

## 0. The bottom line
- **A *learned* residual breaks the analytic contact ceiling on Panda (0.37 → 0.72), proving the "cube tips in
  the gripper" wall is *learnable* — but a matched-budget vanilla PPO still wins the asymptote (0.81).** Across
  two Panda tasks the story is identical: a structured prior buys **sample-efficiency (~1.6–7×), not a higher
  ceiling**.
- **Anti-collapse for self-predictive latents is *downstream-dependent*.** A one-line relational (uniformity)
  loss *fixes* a collapse-prone geometric latent (nav 0.53 → 0.95) but *hurts* value-based control (worst of
  three regularizers on 3/4 DMControl tasks), and a "value-aware" variant is worse still. Structural-entropy
  structure is redundant either way. There is no universal anti-collapse term.
- **We hunted for a new env where we beat PPO and did not find one — but nearly *claimed* one.** On
  `PandaPickCubeOrientation`, TD-MPC2 hit **2842 return** vs PPO's **1419** and I wrote it up as a "strong beat
  forming." Then we scored **real success**: TD-MPC2 = **0.000**. The return was pure reward-gaming. Retracted.
- **The MuJoCo Playground PPO genuinely solves these hard tasks; our TD-MPC2 (at a practical budget) does not.**
  Verified real success: PPO **0.81** on pick-and-orient, **0.99** on dexterous in-hand reorientation; TD-MPC2
  **0.00** on both at 5M. This is the *opposite* of a beat-PPO — and it maps precisely where each method wins.

## 1. Finishing Panda: a learned residual, and the ceiling that isn't a ceiling

Last week the analytic skill-controller capped `PandaPickCube` real success (`box_target ≥ 0.9`) at ~0.37 —
a **contact-physics** wall (the tall cube tips inside the two-finger grip during transport; no analytic gain
knob breaks it). The obvious question: is that wall *physics* or a *learning* limit? We trained a **residual
policy** on top of the analytic skill, \\(a = \\mathrm{clip}(a_\\text{skill} + \\alpha\\,\\pi_\\text{res})\\),
against a **budget-matched vanilla PPO**.

| arm | PandaPickCube (success) | PandaOpenCabinet (success) |
|---|---:|---:|
| analytic skill / oracle | ~0.37 | 0.827 |
| **learned residual (α=1)** | **0.716 ± 0.014** (n=3) | **0.980** (n=7) |
| **matched vanilla PPO** | **0.810 ± 0.006** (n=3) | **0.980** (n=5) |

Two clean findings, both confidence-interval-separated:
1. **The wall is learnable.** The residual drives success-case cube-tilt to ~1.9° (below the analytic servo's
   2.5°) and reaches 0.72 — so it is *not* a hard gripper-morphology limit; a learner *can* fix the contact.
2. **The prior does not raise the ceiling.** Matched PPO wins the asymptote (0.81 vs 0.72; on OpenCabinet they
   tie at the shared 0.98 structural ceiling). What the prior buys is **speed**: ~1.6× faster to competence on
   PickCube, ~7× on OpenCabinet. Bounded residuals (α ≤ 0.5) are unstable and top out ~0.40; only near-full
   authority breaks through — i.e. it's really "warm-started PPO," not a bounded correction.

Same one-liner as the whole campaign: **a structured prior redistributes complexity into faster exploration; it
does not remove it.** (We guarded the budget trap throughout — the earlier "PPO 0.66" was an under-budgeted
baseline; the honest matched PPO is 0.81.)

## 2. Closing the anti-collapse question: it's downstream-dependent

Self-predictive (JEPA/SimNorm) latents can collapse; the fix is an anti-collapse term. Our "glass" program bet
on **structural entropy (SE)**. The verdict, with all controls run (random-graph, random-partition,
strong-VICReg, value-aware):

| downstream regime | best anti-collapse | evidence |
|---|---|---|
| goal-conditioned **geometric** latent (nav) | relational / **uniformity** (1 line) | 0.530 → **0.954**; C-maze n=8: uncollapses 0/8 vs VICReg 8/8 |
| **value-based** control (DMControl) | **none extra** — default wins | uniformity WORST of {default, unif, vicreg} on 3/4 tasks; value-aware worse still |
| either continuous case | **not** SE community structure | Panda SE null; nav benefit is partition-independent |

The mechanism is clean: a relational repulsion spreads apart states that should be *value-close*, destroying the
value-sufficiency control needs — so the *same* one-line loss that wins on navigation is the worst choice for
value-based control. **There is no universal anti-collapse term; match it to what you decode.**

## 3. The hunt for a new beat-PPO env — and the mirage

With the Panda/anti-collapse threads closed, we asked: is there a *new* environment where our strongest method
(TD-MPC2, a self-predictive-latent world model + planning) beats PPO outright? We scanned four untested harder
MuJoCo Playground envs — `PandaPickCubeOrientation`, `PandaRobotiqPushCube`, and the dexterous
`LeapCubeReorient` / `LeapCubeRotateZAxis` — TD-MPC2 vs **matched-budget** PPO.

The early reads looked *thrilling*. On `PandaPickCubeOrientation`, TD-MPC2 hit **2842 episode return** at 0.95M
steps versus PPO's fully-plateaued **1419** ceiling (reached at 75M) — an apparent **both-axes beat at ~80×
fewer samples**. I recorded it (with a caveat flag) as "a genuine ceiling-beat forming."

Then we did what this project is *built* to do — score the metric that matters:

![The return-vs-success trap: TD-MPC2 accrues far more dense reward on PandaPickCubeOrientation than PPO, yet solves the task 0% of the time while PPO solves it 81%.](/images/return_vs_success.png)

**TD-MPC2's real success was 0.000.** Its 2842 return was pure **dense-reward gaming** — hovering near the cube
accumulating shaping reward, never completing the pick-and-orient (reached ~0.2, `box_target` ~0). The "beat"
evaporated. This is exactly the return-vs-success trap the whole campaign is named for, and it caught me
mid-write-up. **Retracted, in the ledger, visibly.**

## 4. So: does PPO actually *solve* the hard tasks? Yes — verifiably

That left the real question, which I'd never actually measured: my PPO runs logged episode *return*, not
success. Does the Playground PPO genuinely *solve* these, or does it also just accrue return? We loaded the
trained PPO checkpoints and scored real success over 256 deterministic rollouts:

![Real success (n=256): the MuJoCo Playground PPO solves all three hard tasks at its full budget; our TD-MPC2 at a practical 5M-step budget solves none.](/images/beatppo_success.png)

| env | PPO real success | budget | TD-MPC2 @5M |
|---|---:|---:|---:|
| PandaPickCubeOrientation | **0.809** (reached 1.0) | 96.7M | 0.000 |
| LeapCubeReorient | **0.988** (≈3.8 reorients/episode) | 212M | 0.000 |
| PandaRobotiqPushCube | 0.18–0.51 | 386M | ~0 |

PPO genuinely solves them. Here is its trained policy reorienting the cube in-hand — the 0.99-success dexterous
task our method never cracks:

![PPO's trained policy solving LeapCubeReorient — dexterous in-hand cube reorientation (real success 0.99).](/images/leap_reorient_solve.gif)

![PPO's trained policy solving PandaPickCubeOrientation — pick and place at target pose (real success 0.81).](/images/panda_orient_solve.gif)

## 5. The honest map: where each method wins

This "failed" hunt produced the most precise statement of the beat-PPO boundary we've had:

- **We beat PPO** (on both sample-efficiency *and* practical capacity) only on **exploration-bottlenecked tasks
  that are solvable within a few million steps** — `HopperHop` (TD-MPC2 367 vs PPO 33), sparse/weak-actuation
  swing-ups. PPO's on-policy exploration stalls; model-based planning gets there.
- **PPO beats us** on **sample-hungry high-DoF tasks** — dexterous in-hand reorientation, multi-object
  manipulation — that need **100M–400M** steps. TD-MPC2 is more sample-efficient *per step*, but it's
  model-based and **slow**, so ~5M steps is the practical ceiling; these tasks need far more, and PPO's ~10–30×
  throughput (512-env brax) delivers them. At *matched* 5M, neither solves these (PPO also needs ~75M).

So TD-MPC2's per-step efficiency only converts to a win when the task fits inside its practical step budget.
That's a real, useful boundary — and arguably a better result than a fake beat would have been.

## 6. Process notes (the part that keeps us honest)
- **Two reward-gaming traps caught by scoring real success** — TD-MPC2's, and my own premature write-up. Return
  is not success on shaped manipulation; we now score `box_target`/native-success *first*.
- **The matched-budget control did its job again**: the apparent beats only existed against the wrong baseline
  (return, or under-budgeted PPO). Every "beats X" in the ledger is qualified by a same-budget control.
- All numbers are deterministic, disk-backed, multi-seed where stated; corrections stay visible in
  `bet2_null_results.md`.

## 7. What's next
- **Pixel observations** are the one genuinely-untested JEPA angle: everything here is low-dim *state*, where a
  non-generative latent's advantage over a generative (Dreamer-style) world model barely shows. On pixels it
  should — that's the next experiment worth standing up.
- The conference write-up now has a crisp, honest thesis: *explicit abstraction and structured priors buy
  sample-efficiency and practical capacity in specific regimes; they do not raise the representational ceiling,
  and where a task simply needs many samples, high-throughput model-free RL wins.*
