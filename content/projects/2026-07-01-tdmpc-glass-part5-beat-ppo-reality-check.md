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

> Weekly review (Jun 24 – Jul 1). Part 4 ended on a cliffhanger: an abstraction-in-the-loop residual *tied* PPO
> and beat it on sample-efficiency. So this week opened with one driving question — **can we actually *beat* PPO,
> not just tie it?** — and the whole week is the story of chasing that answer honestly, including the moment it
> almost fooled us.

### The road this week (the through-line)
Everything below is one arc; here's the map so the pieces connect:

1. **Unfinished business (→ §1).** Our Panda pick controller was stuck at 0.37 real success — a *contact-physics*
   ceiling. Before chasing new wins we had to answer: is 0.37 the *task's* limit, or just our hand-written
   control's? We hand the wall to a **learned residual** — and it breaks (0.37 → 0.72). *But matched PPO still
   wins the ceiling (0.81).* First data point: a prior buys **speed, not ceiling**.
2. **A side door we had to close (→ §2).** If behavioral abstraction only buys speed, maybe *representation*
   abstraction (our "glass"/structural-entropy latent) is where the win hides. We settle the **anti-collapse**
   question with every control — and it's redundant (with one genuinely-useful, downstream-dependent nuance).
   Two doors closed; the prior isn't a ceiling-lever on either.
3. **So go hunt a real beat (→ §3–4).** Tie on the tasks we know ⇒ maybe there's a *new* environment where our
   model-based planner beats PPO outright. We scan harder MuJoCo Playground envs (dexterous hands, orientation
   picks)… and the early numbers look *thrilling* — until they don't. **A return-based mirage nearly became a
   published "beat" (mine).** Scoring *real success* killed it.
4. **The honest verdict + the map (→ §5).** We verify what PPO actually does on those hard tasks (it *solves*
   them, 0.81–0.99; we don't at a practical budget), which finally pins down **exactly where each method wins** —
   and drops the whole hunt onto the broader **DMControl benchmark** (§5b) it's part of.

The motivation never changed — *beat PPO honestly* — and the answer this week is a precise "here's where you can,
here's where you can't," bought partly by catching ourselves red-handed. On to the details.

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

### 1.1 The task and the exact bar
`PandaPickCube` (MuJoCo Playground) is a 7-DoF Franka that must **reach** a small cube, **grasp** it, **lift**
it, and **place** it upright at a target. We score *real* success, not shaped return: an episode counts only if

$$\text{box\_target} \;=\; 1 - \tanh\!\big(5\,(0.9\,\lVert p_\text{err}\rVert + 0.1\,\theta_\text{err})\big) \;\ge\; 0.9,$$

i.e. the cube must end **both** at the target *position* and near-upright *orientation* — concretely
\(0.9\,\lVert p_\text{err}\rVert + 0.1\,\theta_\text{err} \lesssim 0.02\). That `0.1·θ` term is the whole story below.

### 1.2 Where the 0.37 comes from — the tall cube tips in the grip
Our analytic skill (a hand-written reach/grasp/lift/transport/place controller) reaches, grasps, and lifts the
cube essentially every time — grasp success **1.00**, lift **~0.85**. But it caps at **~0.37** real success,
and the reason is a *contact-physics* failure that a still image can't convey:

![The failure mechanism: the tall cube rotates inside the two-finger parallel-jaw grip during lift/transport (tilt grows from ~5° at grasp to ~20° at place), so it arrives off-upright and misses the box_target≥0.9 bar even though position is fine.](/images/cube_tipping_mechanism.gif)

The cube is **tall (4×4×6 cm)** and the gripper is a **two-finger parallel jaw** — a *single* compliant line of
contact with no wrist torque about the grasp axis to resist rotation. During lift and transport the cube slowly
**rotates away from upright inside the fingers**: we measured tilt growing from **~5° at grasp to ~20° at place**.
Position is fine, but that residual tilt blows the `0.1·θ_err` budget, so `box_target` lands just under 0.9. It
isn't a grasp *detection* problem and it isn't reach precision — it's that the parallel-jaw contact *cannot hold
orientation* through the carry. Instrumented on the failures: cube-tilt-at-place ~15–20° on misses vs ~2–3° on
the ~19% that succeed. (The clip above is a real miss — seed 26, tilt **4.5° → 18.3°**, box_target 0.36. For
contrast, a success keeps the cube upright the whole carry — tilt **4.5° → 2.8°**, box_target 0.96:)

![Contrast: a successful episode keeps the cube upright through transport (tilt stays ~3–4°, box_target 0.96) — the ~19% the analytic skill gets right.](/images/cube_upright_success.gif)

### 1.3 Can analytic tuning fix it? (rounds 4–5) — a little, then a wall
- **Round 4 (yaw-aware place):** null — we were correcting the wrong axis (it's mid-flight *tipping*, not grasp
  yaw). Oracle barely moved (0.305 → 0.320).
- **Round 5 (closed-loop upright servo):** recompute the gripper orientation every step to actively *right* the
  live cube during transport. This helped — final tilt **20.6° → 16.4°**, and the analytic **oracle** (brute-force
  best parameters searched directly in sim) rose **0.305 → 0.352**, with the full n=512 solve landing at **0.367**.

But that's it. No analytic gain knob crosses ~0.37, because the servo *itself* perturbs placement *position* while
righting orientation — through the single compliant contact you can't satisfy both halves of the
\(0.9\,p + 0.1\,\theta \le 0.02\) budget at once. So we had a genuine ceiling **for the analytic-skill family**.
The question that names this section: is ~0.37 a ceiling of the *task/hardware*, or just of *hand-written control*?

### 1.4 Hand the wall to a learner: a residual policy
We put a **learned residual** on top of the analytic skill and trained it by RL against the true reward,
\(a = \mathrm{clip}\big(a_\text{skill} + \alpha\,\pi_\text{res},\,-1,1\big)\), sweeping the authority
\(\alpha\), and — the load-bearing control — a **budget-matched vanilla PPO** (same env, same step budget, same
eval).

| arm | PandaPickCube (success) | PandaOpenCabinet (success) |
|---|---:|---:|
| analytic skill / oracle | ~0.37 | 0.827 |
| residual α=0.25 / 0.5 (bounded) | 0.19 / 0.40 (unstable) | — |
| **learned residual α=1 (full)** | **0.716 ± 0.014** (n=3) | **0.980** (n=7) |
| **matched vanilla PPO** | **0.810 ± 0.006** (n=3) | **0.980** (n=5) |

Two clean findings, both confidence-interval-separated:

1. **The wall is *learnable* — it was never task/hardware physics.** The full-authority residual drives
   success-case cube-tilt down to **~1.9°** (below even the analytic servo's 2.5°) and reaches **0.716**, nearly
   doubling the analytic 0.37. A learner *can* hold the cube upright through transport where hand-tuned control
   can't. So "the ceiling that isn't a ceiling": ~0.37 is a hard limit of the *analytic-skill family*, not of the
   task. (Note the authority sweep: strictly *bounded* residuals — α ≤ 0.5 — are unstable and top out ~0.40; only
   near-full authority breaks through, i.e. it's effectively a *warm-started* policy, not a small correction.)
2. **But the prior does not raise the *RL* ceiling.** Matched vanilla PPO wins the asymptote outright — **0.810
   vs 0.716** on PickCube; on OpenCabinet both hit the same **0.98** structural ceiling. What the analytic prior
   actually buys is **speed**: ~1.6× faster to competence on PickCube, ~7× on OpenCabinet.

Same one-liner as the whole campaign: **a structured prior redistributes complexity into faster exploration; it
does not remove it, and it does not lift the ceiling.** (Budget-trap guarded throughout — the earlier "PPO 0.66"
was an under-budgeted baseline; the honest matched PPO is 0.81.)

## 2. Closing the anti-collapse question: it's downstream-dependent

*Why this, now?* §1 said a **behavioral** prior buys speed, not ceiling. The last place a real win could hide is
the **representation** itself — the "glass"/structural-entropy latent this whole project is named for. If a better
latent doesn't lift control, then abstraction is redundant on *both* axes and the only honest path to beating PPO
is finding the right *task* (§3). So we close it properly, with every control.

A self-predictive (JEPA/SimNorm) latent trained only to predict its own future can **collapse** — the encoder maps
everything to (nearly) one point, prediction becomes trivially perfect, and the representation is useless. You need
an **anti-collapse term**. Our "glass" program bet on **structural entropy (SE)**; the honest answer is that SE
*per se* is the wrong bet, and the right one is *downstream-dependent*. Here are the exact terms we compared and
how each is implemented:

- **VICReg** — *per-dimension* anti-collapse: a **variance hinge** (push each latent coordinate's std ≥ 1) plus a
  **covariance penalty** (decorrelate coordinates). Fights collapse one axis at a time.
- **Uniformity** (Wang & Isola) — the **one-line relational** term
  \(\mathcal{L}_\text{unif}=\log\,\mathbb{E}_{i\ne j}\,e^{-2\lVert z_i-z_j\rVert^2}\), i.e.
  `(-2*pdist(z).pow(2)).exp().mean().log()` — pushes *all pairs* of embeddings apart on the hypersphere. No graph,
  no partition, no tree.
- **SE (structural entropy)** — build a kNN graph on the latents, compute its minimal-2D-SE **community partition**
  (an encoding tree, via `selib`), and regularize the latent toward low coding-cost of that partition. Structure,
  not pairwise.
- **knnrep** — pairwise repulsion over the kNN graph only (relational, graph but no partition).

And the four controls that make the conclusion airtight:

- **random-graph** (Panda) — run the *whole SE machinery on a shuffled/random* kNN graph. Isolates: is SE's effect
  the community *structure*, or just "some regularizer"?
- **random-partition** (nav) — SE with *randomized* community assignments. Isolates: the specific SE tree, or any
  relational grouping?
- **strong-VICReg** (nav) — VICReg with a large coefficient that *fully* restores effective rank. Isolates: does
  fixing latent *health* fix *control*?
- **value-aware** (DMControl) — uniformity re-weighted by *value* similarity (down-weight repulsion between
  value-close states). Isolates: does matching the term to value structure rescue it?

### The nav collapse testbed (n=4, deterministic n=256 eval)
This is a 2-D point-maze where SimNorm+VICReg *provably* collapses, so it cleanly separates the regularizers. It is
**not** a beat-PPO task — it's a controlled representation-learning probe. The metric is success *relative to the
raw-observation ceiling* (an identity encoder that trivially solves the maze):

| arm | success | collapsed? | reads as |
|---|---:|---|---|
| raw-obs ceiling | **0.979** | — | the achievable max |
| VICReg (default) | 0.530 | 4/4 | collapses → fails |
| VICReg (strong) | 0.442 ± 0.37 | 0/4 | **un-collapses but still fails** — health ≠ control |
| SE (se_w=5) | 0.906 | 0/4 | works |
| SE (**random partition**) | 0.823 | — | ≈ real SE (overlaps) → **partition-independent** |
| knnrep (graph, no partition) | 0.887 | 0/4 | works |
| **uniformity (1 line)** | **0.954 ± 0.05** | 0/4 | works best, simplest |

Two things fall out. **(a)** SE *fixes a collapse VICReg cannot* — but the random-partition control shows the win
is the *relational/pairwise* structure, not the SE community tree; a one-line uniformity loss matches it. **(b)** On
Panda, the same SE machinery is a **null**: a frozen-encoder value/geometry probe gives VICReg \(R^2=0.987\) for
end-effector→cube vs SE's 0.736 — and **SE-on-random-graph recovers to 0.986**, proving SE's *community bucketing*
(effective rank 31→13) is what destroys the continuous manipulation geometry.

### But it flips on value-based control
Run those same arms on collapse-prone **DMControl** (return-AUC, n=3), and the relational term that *won* on nav is
now the **worst**:

| task | default (no extra) | uniformity | value-aware |
|---|---:|---:|---:|
| CheetahRun | **58.9** | 36.6 | 20.6 |
| WalkerWalk | **293.8** | 89.9 | 49.5 |
| FingerSpin | **249.4** | 171.8 | 1.6 |

Uniformity is worst on 3/4 tasks, and the **value-aware** variant is *worse still* — matching the term to value
structure didn't rescue it, it hurt more. The mechanism is clean: a relational repulsion spreads apart states that
should be *value-close*, destroying the value-sufficiency control needs. So the taxonomy:

> **There is no universal anti-collapse term.** Relational/uniformity for goal-conditioned **geometric** latents;
> **nothing extra** for **value-based control** (SimNorm's built-in pressure + the value gradient suffice);
> **never SE community structure** for either continuous case. Match the term to what you decode.

**Does the 0.954 "beat PPO"? No — and it shouldn't be read that way.** It's a *recovery* number: ~97% of the
raw-obs ceiling (0.979), un-doing the collapse that cost VICReg 0.53→0.95, on an easy nav task where PPO was never
run and would also succeed. The anti-collapse result is a **representation-learning** finding (which regularizer
fixes JEPA collapse), not a control-benchmark win. The only place this touches "beat PPO" is negatively: even the
*best* fixed latent doesn't change the §1/§3 story that a structured latent doesn't raise the RL ceiling.

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

## 5b. The DMControl benchmark, consolidated (the dashboard table)

The manipulation results above sit on top of a broader **DMControl suite benchmark** we've been running on a
live dashboard (16 tasks × 3 seeds, glass / TD-MPC2 / PPO). Two headlines it settles:

**(i) Representation abstraction is redundant.** Across **16 DMControl tasks (n=3–4)**, our structural-entropy
"glass" latent shows **no systematic return difference from vanilla TD-MPC2** — ties within 95% CI on ~12/16,
the few separations tracking single collapsed seeds in *both* directions. The only robust effect is that glass
costs **~1.35× wall-clock**. On a value-sufficient self-predictive (SimNorm) latent — where a linear value-probe
already gets \(R^2 \approx 0.999\) — an explicit representation abstraction has nothing left to add.

**(ii) TD-MPC2 vs PPO on DMControl is a clean split by exploration difficulty** (peak return; note the wildly
different budgets):

| task | TD-MPC2 | PPO | who wins the ceiling |
|---|---:|---:|---|
| **HopperHop** | **367** @2M | 33 @285M | **TD-MPC2 ~11×** — PPO never learns to hop |
| **AcrobotSwingup** | **473** | 268 @285M | **TD-MPC2** — exploration-limited for PPO |
| CheetahRun | 639 @1M | **928** @285M | PPO — dense/explorable, but at **~285× the samples** |
| WalkerWalk | ~970 (tie region) | 970 @79M | tie — both solve |
| FingerTurnHard | — | 968 @285M | PPO solves (dense) |

The pattern is the same one the whole post is about: **on exploration-bottlenecked tasks TD-MPC2's planning wins
outright and at 1–2 orders of magnitude fewer samples; on dense/explorable tasks PPO reaches a higher (or tied)
ceiling given its enormous throughput.** No world-model variant Pareto-dominates PPO — the sample-efficiency ↔
wall-clock trade is fundamental, and the manipulation scan in §3–4 is just the high-DoF, sample-hungry end of
this same axis (where PPO's throughput wins because the task needs 100M+ steps and TD-MPC2 can't practically get
there). *(Live per-task CSVs + learning curves are on the dashboard; PPO was run on a subset at 50–285M steps —
budgets labelled, never fabricated.)*

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
