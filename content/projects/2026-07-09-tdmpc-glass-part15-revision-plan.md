---
title: "TD-MPC-Glass, Part 15: Revision Plan & Live Log — Integrating the New Findings into Papers A & 3"
date: 2026-07-09T15:00:00
description: "A working plan (and running log) for the next two days: fold the value-sufficiency curve, the bisim/reweighting/bottleneck nulls, the Hopper conjunctive-reward result, and the planning-vs-world-model dissection into our two existing papers — the redundancy-criterion paper (A) and the beating-PPO anatomy paper (3) — rather than spinning a fourth. It records the narrative upgrade for each paper, resolves the sharpest open question (if the world model is removable, what makes TD-MPC2 near-SOTA on HopperHop?), commits to a positive artifact per paper so these are not pure-null results, and lays out the exact experiments, GPU schedule, and time estimate. Updated in place as runs land."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["world-models", "TD-MPC2", "PPO", "SAC", "hopper", "value-sufficiency", "paper-revision", "planning", "reward-design", "living-log"]
---

{{< katex >}}

> **📋 Living plan + log** (started 2026-07-09). We already have three drafted papers; the new findings from Parts
> 9–14 mostly *strengthen* two of them rather than justify a fourth. This post records the integration plan, the
> narrative upgrade, the one open mechanism question we still owe, the positive artifacts that keep these from being
> pure-null papers, and the experiment schedule. The **Live log** at the bottom is updated in place as runs complete.

## The decision: integrate, don't spawn

The findings we've accumulated — the value-sufficiency curve, the bisimulation / reweighting / bottleneck nulls, the
Hopper conjunctive-reward result, the planning-vs-world-model dissection — overlap almost entirely with two papers we
have already drafted:

- **Paper A — *"When Is Explicit Abstraction Redundant for a World Model? A Negative Criterion."*** The redundancy /
  value-sufficiency story.
- **Paper 3 — *"The Anatomy of 'Beating PPO': Exploration Walls, the Value Pathway, and the World Model."*** The
  dissection / attribution story.

(A third, *"Abstraction is a Speed-of-Learning Lever, not a Capacity Lever,"* is untouched by the new work.)

So the plan is to **upgrade A and 3** — and, importantly, to make each *more convincing and less purely-negative*.

## Narrative upgrade

### Paper A — from "our criterion failed" to "we built the right instrument"

Paper A already contains a strong, honest negative: the tempting way to certify value-sufficiency — a linear
decode-\\(R^2\\) — is **invalid** (it saturates at ≈0.98 for a strong *and* a collapsed policy; the return-to-go
\\(R^2\\) even *anti*-correlates with performance). But the arc ends deflating: "a valid probe is still open."

The **value-sufficient bottleneck (VSB)** closes exactly that gap. Instead of decoding value off the latent, force
the value and policy heads to read only the first \\(D\\) of the 512 latent dimensions and measure return\\((D)\\).
That is a *behavioral* measurement of how much of the latent the value pathway actually needs. New three-act arc:
1. **The trap** — why decode-\\(R^2\\) can't certify sufficiency (existing result).
2. **The instrument** — VSB: return\\((D)\\) is smooth, monotone, and distributed; *no small sufficient subspace*,
   and *no imposed structure beats the vanilla distributed latent*.
3. **The verdict on three legs** — redundancy holds because the latent is *behaviorally* value-sufficient: **theory**
   (the basis-change/data-processing proposition) + **instrument** (VSB) + **breadth** (the null campaign, now four
   taxa: added SE/graph, reweighted VAC/URC, metric bisimulation, architectural bottleneck).

### Paper 3 — from "a wall exists" to "here's why, and what the world model actually does"

Paper 3 characterizes the HopperHop PPO wall but does not explain it, and leaves the world model as merely "the
mildest cut." Two upgrades:
- **Explain the wall.** Reading the environment, HopperHop has **no early termination** and a **conjunctive,
  multiplicative reward** (`standing × hopping`). Env-gating the reward shows PPO **escapes under an additive reward**
  but **stays walled under the product even with an easier hop threshold** — so the wall is the *conjunction*, a
  benchmark-design property, not a capability limit (the Voelcker caveat, confronted head-on).
- **Resolve the world model's role.** Not "mildest cut" but a **task-conditional rollout regularizer** — *removable
  on the very task it is celebrated for* (HopperHop, n=8) and load-bearing only where precise multi-step rollouts set
  the return level (Walker/Cheetah/Acrobot).

## The sharpest open question (Point 1): if the world model is removable, why is TD-MPC2 near-SOTA on HopperHop?

Removing the *consistency loss* removes the objective that makes the latent an accurate **predictor** — it does not
remove the **planner**. TD-MPC2 differs from SAC in two things: MPPI planning, and the value architecture. Our
working hypothesis:

> TD-MPC2's HopperHop edge is **off-policy TD value learning + MPPI acting as a *structured explorer* during data
> collection** — not the predictive accuracy of the world model. On a conjunctive-reward wall the hard part is
> *finding* the joint standing+hopping behavior; MPPI's population-based, value-greedy action search finds it faster
> than SAC's local Gaussian noise (TD-MPC2 ~1M ≫ SAC ~8M ≫ PPO walls). The consistency objective is redundant because
> MPPI only needs the model to *rank actions by value*, not to roll out accurately.

This is **not yet proven** — it is the paper's key remaining experiment (**P2** below): TD-MPC2 with **policy-only
data collection** (MPPI disabled during rollouts). If it collapses toward SAC's slow curve, MPPI-as-explorer is the
edge; if it still beats SAC, the edge is the value architecture. Either way, this converts Paper 3's thesis from a
negative ("no level advantage") into a positive mechanism.

## Positive artifacts (so these are not pure-null papers)

- **Paper A → the VSB diagnostic.** A valid, checkpoint-cheap probe for value-sufficiency that predicts whether an
  abstraction objective can help on a given task/model — the correct replacement for the invalid decode-\\(R^2\\).
- **Paper 3 → the reward-conjunctivity benchmark knob + exploration-wall law.** The env-gated reward-structure
  variants (conjunctive ↔ additive) packaged as a controlled axis for probing exploration under conjunctive
  sparsity, plus the predictive statement *reward-conjunctivity → on-policy wall depth*.
- **Stretch (either) → a world-model-necessity predictor:** a checkpoint signal (k-step rollout-error / removability)
  that flags where the consistency loss is redundant — a compute-saving gate.

## Experiments, GPU schedule, estimate

A 5M TD-MPC2 run ≈ 5–6 h on a 3060 (4 concurrent/box); PPO (brax) is minutes. **b3060b cannot build HopperHop**
(mjx-warp dependency); **b3060 is the Hopper box**.

| # | Experiment | Paper | Box | Wall est. |
|---|---|---|---|---|
| A1 | VBN value-sufficiency curve → **n=5** (Cheetah/Walker/Acrobot) | A | b3060b | ~2 d |
| P4 | Sufficiency grid → **n=5** (pin removable/load-bearing %) | 3(+A) | both | ~1.5 d |
| P2 | **MPPI-as-explorer ablation** (policy-only collection) — *build + runs* | 3 | b3060 | ~1.5 d |
| P1 | **SAC-core isolation** (SAC actor-critic + WM/MPPI) — *build + runs* | 3 | b3060 | ~2 d |
| P3 | H3 seeds + 2nd conjunctive-reward task (n=5) | 3 | b3060 | ~1.5 d |
| A2 | *(stretch)* OOD/compositional value-decodability probe | A | b3060b | ~1 d |

**Two-day scheduling target:** Days 1–2 — builds (SAC-core, MPPI-disable flag) + launch A1 (b3060b) and P4 (b3060);
begin drafting both reframed abstracts + section skeletons. Then P2/P1/P3 roll behind on b3060, A2 behind A1 on
b3060b. **Honest total to n≥5 with the positive artifacts: ~1.5–2 weeks** including builds, reruns, and slack for
box flakiness. Critical path is **P2** (it answers Point 1).

## Live log

- **2026-07-09 15:00** — Plan posted. Launched **A1** (Cheetah VBN width sweep, seed 51 → toward n=5) on b3060b and
  **P4** (WalkerRun full-vs-stripped, seeds 70/71 → pins the world-model removable/load-bearing margin) on b3060.
  Both 4 arms up, 0 nan at launch. Next: build the **P2** MPPI-disable flag; extend A1 to further seeds/tasks;
  begin the Paper A VSB section and the Paper 3 wall-mechanism section. *(This log is updated in place as runs land.)*
- **2026-07-09 15:30** — GPU packing: b3060b doubled to **8 jobs** (added Walker VBN s51 alongside Cheetah s51,
  2/GPU, mem-fraction 0.35), 100% util both boxes.
- **2026-07-09 16:20 — CODE/EXP AUDIT (post-weekly review).** Four faults found and fixed:
  **(F1, data-integrity)** the `run_arm.sh` jsonl tag lacked the *task* name (`wmabl_<ablate>_s<seed>`), so the
  Walker two-axis (s62/63) appended into the *Hopper* s62/63 jsonl files — the ledgered Hop numbers were harvested
  *before* contamination (benchmark CSVs, which are task-named, remain the durable source), but this explains the
  garbage Walker π-vs-MPPI read, and a planned Cheetah batch on s70/71 would have corrupted the live Walker P4 data.
  **Fixed:** tag now `wmabl_<TASK>_<ablate>_s<seed>` (backup `.bak_tagfix`).
  **(F2, ops)** heredoc-through-compound-ssh silently failed twice — the Cheetah sufficiency batch *never launched*,
  and the earlier status report mis-attributed the 4-proc count to "compute saturation" instead of a failed launch.
  **Fixed:** printf-style drivers; Cheetah full-vs-stripped s70/71 now actually running (b3060 packed to 8 jobs,
  task-qualified tags).
  **(F3, claim precision)** Part 14's "PPO escapes the wall" overstated: the additive-reward curve climbs steadily
  to 135 at 20M (not plateaued) but is well below even the standing component's ~500 ceiling — the finding is
  *learnable signal vs flat zero*, not task-solved. Part 14 edited; `HOP_SPEED=1.0` margin-coupling footnoted
  (margin = speed/2, so that variant is threshold+margin, not threshold-only).
  **(F4, paper caveat)** VBN arms compare against van2 baselines run under earlier patch stacks; all env-gates
  (VBN/VAC/URC) verified present and default-off (byte-identical when unset) — to be stated in the paper's setup.
  Current fleet: b3060b 8 jobs (Cheetah+Walker VBN s51), b3060 8 jobs (Walker s70/71 + Cheetah s70/71
  full-vs-stripped), 0 nan.
