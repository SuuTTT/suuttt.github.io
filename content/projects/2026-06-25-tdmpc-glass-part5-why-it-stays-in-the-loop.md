---
title: "Why the Abstraction Has to Stay in the Loop — Negative Results, a Physical Ceiling, and What an Abstraction Is Actually For"
date: 2026-06-25
description: "Follow-up to the beat-PPO update: the in-loop residual on a hand-coded controller ties PPO (0.79 vs 0.81) and beats it 1.7x on sample-efficiency. This post details the loop and the analytic controller, explains the negative results (distillation, authority-annealing, and a learned gate all fail) as a necessity proof that the controller must stay in the loop, shows that 0.83 is a physical kinematic ceiling (proven both directions), and reframes the contribution: comparable success with better sample-efficiency, stability, reusability, and interpretability — conditional on a task-matched, parametrized prior."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["world-models", "TD-MPC2", "PPO", "abstraction", "residual-RL", "manipulation", "negative-results"]
---

{{< katex >}}

> Continuing from the [last update](https://suuttt.github.io/projects/2026-06-24-tdmpc-glass-part4-jumpy-to-beat-ppo/):
> the headline was a heuristic-in-the-loop residual that **ties PPO (0.79 ± 0.01 vs 0.81) and reaches competence
> ~1.7× faster**. This post answers the three questions that came back: *what exactly are the negative results,
> why were we pushing PPO's success at all, and what is the actual claim* — with the loop, the controller, and
> the latest numbers spelled out.

## 1. The loop and the analytic controller (what "in the loop" means)

The method has two parts that run **together at every timestep**.

**(a) The analytic controller** — a hand-coded, closed-loop phase machine for PandaPickCube. It reads the box
and target poses directly and steps through phases, solving a small analytic level-IK each step to produce a
Cartesian action:

```
phase ∈ {REACH, DESCEND, GRASP, LIFT, PLACE}
loop each step:
    if   REACH:   goal ← above(box);          if EE over box      → DESCEND
    elif DESCEND: goal ← box;                 if EE at box        → GRASP
    elif GRASP:   close gripper;              if grasped          → LIFT
    elif LIFT:    goal ← box + z_clear;       if lifted           → PLACE
    elif PLACE:   goal ← target; open near target
    a_ctrl ← levelIK(goal, current_pose)      # closed-loop, per step
```

On its own this controller solves the task only ~9% of the time (open-loop grasp dynamics on a small cube are
the limit) — it is a *competent prior*, not a solution.

**(b) The in-loop residual.** We keep the controller **live** and learn a small policy that corrects it. The
key fix is making the problem Markov: the controller's hidden phase index \(z\) is appended to the observation,
so the learned residual sees \((s, z)\). The executed action is

$$ a = \mathrm{clip}\big(a_{\text{ctrl}}(s) + \alpha\,\pi_{\text{res}}(s, z)\big),\qquad \alpha = 1.0\ \text{(persistent)}. $$

\(\pi_{\text{res}}\) is trained with ordinary PPO inside an environment wrapper that adds \(a_{\text{ctrl}}\) and
exposes \(z\). Crucially \(\alpha\) is a **constant**, not annealed (Section 3 explains why). This persistent
in-loop residual reaches **0.79 ± 0.01** real success (box_target ≥ 0.9, n=256), a near-tie with end-to-end
PPO's **0.81**, and crosses the 0.66 competence bar at **19.7M** steps vs PPO's **32.8M** — the ~1.7× sample
-efficiency win.

## 2. The negative results, explained — a *necessity proof*, not failures

The supervisor's question — "I didn't understand the negative-result part" — is fair, because the negatives are
only meaningful together. They are **three independent attempts to take the controller *out* of the runtime
loop, all of which are worse than keeping it in.** That is how you prove a component is necessary: remove it,
show it breaks.

| route to *remove* the controller | what it does | result |
|---|---|---|
| **Distillation** | BC/DAPG the controller+residual into a standalone raw-action net | **0.0** (BC); DAPG only matches plain PPO 0.82 with worse sample-efficiency |
| **Authority-annealing** | anneal \(\alpha: 0\to 1\) so the learned policy takes over | **collapses to 0.008** |
| **Learned gate** | learn a per-state gate that switches the controller off when "not needed" | **0.613** — strictly below persistent \(\alpha\)=1.0's **0.781** |

Three independent eliminations all degrade or collapse ⇒ **the analytic controller must stay in the loop at
runtime.** The mechanism is identical in all three: the abstraction is **non-Markov** — its action depends on
the hidden phase state — so the controller is a *runtime provider* of the phase structure the residual corrects,
**not a discardable training scaffold**. Distill it and the reactive net can't reconstruct the phase; anneal it
and a half-trained residual inherits a non-stationary target; gate it off and the phase context is lost
mid-episode. The residual has nothing to stand on without the controller beneath it.

That is the real content of "the abstraction has to stay in the loop": not a slogan, but a claim falsified three
ways and surviving.

## 3. Why we pushed PPO's success — and why that was the wrong scoreboard

We pushed success because *beating strong model-free RL* is the field's bar for "this structure earns its
keep." So we threw seven levers at PandaPickCube to exceed PPO and approach 100%: bigger nets, a capacity
sweep, closed-loop retry, two curricula, deploy-time best-of-k, an orientation-aware controller, longer budget,
and end-to-end learned grasping. **None beats ~0.83.**

The reason is not a learning shortfall — it is **physics**, and we proved it both directions:

- **Forward:** at far horizontal reach (>0.84 m) the Panda **cannot hold a grasped cube within ~8° of upright**,
  which the `box_target ≥ 0.9` metric requires — an upright-grasp IK feasibility test finds **99.9% of the
  far-reach configs infeasible** (while 100% are *position*-reachable). Position-reachable ≠ upright-graspable.
- **Backward (the clean falsification):** simply **remove** the far-reach configs from the spawn distribution
  (17.6% of spawns exceed 0.84 m, matching the ~17% failure rate) and PPO success goes to **1.000**.

So "push success higher" is impossible on this task/metric for *any* method — which is exactly why the project
stopped optimizing success and moved the contribution elsewhere.

## 4. The corrected claim — and the reusability arc

To be precise, because it is easy to over-state: **the in-loop residual does *not* beat PPO on success.** It
*ties* where the controller's prior fits and *trails* where it doesn't:

| task | in-loop residual | PPO |
|---|---|---|
| PandaPickCube (prior fits) | 0.79 (tie; 1.7× faster) | 0.81 |
| PickCubeOrientation, **fixed** prior | 0.33 (loses) | 0.82 |
| PickCubeOrientation, **parametrized** prior | 0.67 (recovers ~70%) | 0.82 |
| PickCubeOrientation, *fuller* parametrization | 0.68 (class ceiling) | 0.82 |

The orientation task is the clean test of *reusability*: the fixed top-down/fixed-yaw grasp is the wrong prior
for a random-yaw target, so the abstraction **loses** (0.33, worse than learning from scratch). Making the
controller **orientation-aware** — grasp/place at the target yaw, with the target pose fed to the residual —
recovers it to **0.67** (~70% of the gap), but fuller parametrization plateaus at **0.68**: a residual
correcting an analytic grasp has a ceiling below end-to-end RL on hard tasks. The design lesson: **abstractions
are reusable only when their priors are *parametrized* to the task's degrees of freedom, not hard-coded.**

So the honest, defensible thesis is **not** "abstraction beats PPO." It is:

> **A small residual on a task-matched, parametrized abstraction gives comparable success with better
> sample-efficiency, stability, and interpretability — and transfers across tasks when the prior is
> parametrized.** The value is a *more trustworthy, transferable, debuggable* controller at matched success,
> not a higher score; and the three negatives above explain why the structure that delivers it must stay in the
> loop.

(Stability shows up concretely: the matched/parametrized residual barely moves from peak to final, whereas an
ill-matched residual exhibits a fragile optimum that collapses under continued training. Interpretability is
structural — a legible phase machine with phase-attributable failures, which is *how* we localized the ceiling
to the grasp phase and then proved it kinematic.)

## 5. The original mission, closed honestly (InFOM)

The week began as a reproduction of CompPlan + InFOM. Result: **InFOM-base reproduces on its actual
(manipulation) benchmark** — cube-single ~0.96/0.98 on 2 of 3 seeds (with one seed collapsing to 0.16, real
seed-variance). AntMaze, which an initial draft scored as a "failure," turned out to be **outside InFOM's
benchmark entirely** (InFOM's finetuning datasets are generated locally for manipulation envs only; antmaze
appears nowhere in the repo or paper). So the antmaze number is an out-of-benchmark extension, not a
reproduction — a correction we made rather than leave a misleading comparison standing.

## 6. Ongoing

- **When does planning help vs hurt?** TD-MPC2's planner helps HopperHop (+8% over the reactive policy) but
  *hurts* PandaPickCube (it plans over a reward-hacked value). We're running a controlled correlation across
  several DMC tasks — planning advantage vs learned-model accuracy — to turn that into a "plan-when-the-model-is
  -accurate" rule, and a confidence-gated planner.
- **Does the parametrize-the-prior recipe generalize to a third task?** Open — the clean fixed-vs-parametrized
  contrast is harder to construct than it looks, and we're choosing the right task.

Full per-experiment write-ups (Parts 25–37), with learning curves, the benchmark table, rollout videos, the
upright-IK feasibility proof, and the workspace-constraint falsification, are in the
[TD-MPC-Glass series](https://suuttt.github.io/tdmpc-glass/).
