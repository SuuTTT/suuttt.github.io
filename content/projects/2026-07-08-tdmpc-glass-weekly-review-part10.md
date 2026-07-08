---
title: "TD-MPC-Glass, Part 10: A Week in Review — From Five Failed Bets to a Dissection (and a First SOTA Swing)"
date: 2026-07-08
description: "The week of July 1–8, reviewed end to end. It began with a method map and five concrete bets to finally beat PPO with abstraction or planning-as-exploration (Parts 5–6). Almost all of them nulled (Part 7). That failure was the useful part: it forced a pivot away from 'add structure to win' toward rigorously dissecting WHY the planner beats PPO in the first place — the categorical exploration wall, the loss-by-loss mechanism, and a five-task sufficiency grid (Parts 8–9, three papers). We close with the first constructive SOTA attempt built on those lessons: value-aware consistency (a clean no-go) and an uncertainty-aware variant now running. The recurring law held all week: abstraction and reweighting buy nothing the value pathway doesn't already use."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["world-models", "TD-MPC2", "PPO", "SAC", "abstraction", "JEPA", "mujoco-playground", "reproducibility", "weekly-review", "vastai"]
---

{{< katex >}}

> This is a review of one week — July 1 through July 8 — across [Part 5](https://suuttt.github.io/projects/2026-07-01-tdmpc-glass-part5-beat-ppo-reality-check/)
> through Part 9 and the work since. The week has a clean three-act shape. **Act I:** we laid out a map of four
> method families and made *five concrete bets* to beat PPO with abstraction or planning-as-exploration.
> **Act II:** the data killed almost all of them — and that failure redirected the whole program from "add
> structure to win" to "dissect *why* the planner already wins." **Act III:** that dissection produced three
> papers (a wall, a mechanism, a sufficiency law) and, finally, a first constructive attempt at a new SOTA world-model
> objective. Every number below is read from disk; the nulls are reported as loudly as the positives.

## Where the week started: the method map (Part 5)

[Part 5](https://suuttt.github.io/projects/2026-07-01-tdmpc-glass-part5-beat-ppo-reality-check/) framed the entire program not as
"beat PPO once" but as a **map**: across Panda manipulation and DMControl, *when does each of four families win?*

- **State abstraction** — structural-entropy ("glass") / relational latents.
- **Behavioral abstraction** — analytic controllers, TAMP, skill options.
- **Planning** — TD-MPC2 (a learned latent world model + MPPI).
- **Plain PPO.**

Scored LeCun-style by *learning speed*, not just final return, one law kept recurring: **abstraction and priors buy
sample-efficiency where the prior fits, not a higher ceiling.** Part 5 also set up the JEPA pivot — LeCun's argument
that a non-generative predictor needs an information (anti-collapse) term, and that *hierarchy* is where structure
should pay off — and a beat-PPO reality check that, when the returns were audited for reward-gaming, collapsed a
"TD-MPC2 beats PPO on PandaPickCubeOrientation" claim into a verified **PPO solves, TD-MPC2 fails** (real success
0.000 under return-gaming).

## Act I — five bets (Part 6)

[Part 6](https://suuttt.github.io/projects/2026-07-02-tdmpc-glass-part6-five-bets-next-phase/) turned the open
questions into five falsifiable bets:

- **A. Planning-as-exploration** — does using the MPPI planner to *seek* novel/uncertain states break tasks a
  reactive policy can't explore?
- **B. Behavioral-prior taxonomy** — where exactly does an injected controller help vs hurt?
- **C. Glass-as-variance-reduction** — even if the mean return ties, does structure reduce seed variance?
- **D. JEPA done right** — pure (decoder-free) latent prediction + a real anti-collapse term.
- **E. SE-structured latent discovery** — structural entropy as the anti-collapse signal.

## Act II — what survived contact with the data (Part 7)

[Part 7](https://suuttt.github.io/projects/2026-07-03-tdmpc-glass-part7-five-bets-resolved/) is where the week
turned. Five bets, all multi-seed and matched-budget. Verdicts first, then the experiments and the data behind each.

| Bet | What it claimed | Verdict |
|---|---|---|
| A — planning-as-exploration | the planner *seeks* novel states a policy can't | **Refuted** |
| C — glass as variance-reduction | structure lowers seed variance even at tied mean | **Null** |
| D — pure-JEPA anti-collapse | a decoder-free predictor collapses without it | **Reversed** |
| E — SE-structured latent | structural entropy is the right anti-collapse | **Hurts** |
| B — behavioral-prior taxonomy | injected controllers raise the ceiling | **Sample-eff only** |

#### Bet A — is planning a directed-exploration operator?

The claim: TD-MPC2 beats PPO because the MPPI planner actively *explores* — it seeks novel states a reactive policy
never reaches. We tested it with a same-weights plan-vs-policy ablation, escalating to a decisive sparse task. On a
learnable dense task (WalkerRun, n=3) planning leads through the competence phase then **converges to the same
ceiling** — a speed effect, and it actually *narrows* late-training state coverage (directed exploitation onto the
gait manifold, not exploration):

| step | PLAN return | PI-only return |
|---|---|---|
| 50k | 335 | 186 |
| 90k | 408 | 247 |
| 140k | 480 | 446 (converged) |

The make-or-break test was a sparse, exploration-hard, learnable task — CartpoleSwingupSparse (n=3, 300k). Does
planning discover the sparse reward where policy-only stalls? **No — policy-only discovers it *earlier*:**

| metric | PLAN | PI-only |
|---|---|---|
| reward-discovery rate | 3/3 | 3/3 |
| mean discovery step | 156,672 | **140,032** |
| final return | 454 | 239 |

Planning's real value is *post-discovery exploitation*, not exploration. The other half of Bet A — novelty bonuses
(disagreement/RND) inside the planner on HopperHop 1M, with a **matched-seed** vanilla control — was worse on 4/4:

| seed | vanilla | + novelty |
|---|---|---|
| 61 | 442.3 | 5.6 (broke) |
| 62 | 508.5 | 284.6 |
| 63 | 321.4 | 258.7 |
| 64 | 359.4 | 273.8 |

**Refuted:** planning is a sample-efficiency + exploitation operator, not a directed-exploration one.
(`b3060b:exp/proposal_A1_coverage/`.)

#### Bet C — does glass reduce seed variance even when the mean ties?

Out-of-sample across all 16 tasks: mean seed-sd **glass 123.2 vs TD-MPC2 114.5** — glass is *higher* variance;
lower-sd on only 8/16, better worst-seed 9/16 (coin flips), and glass has its own collapses TD-MPC2 avoids
(HopperHop 0 vs 179). **Null** — no variance-reduction effect. (`b3060:exp/proposal_C_variance/`.)

#### Bet D — does a "proper" JEPA collapse without an anti-collapse term?

A pure decoder-free self-predictive latent (encoder + jumpy predictor + EMA target, no reward/value/policy),
frozen-encoder ridge probes, three tasks (n=3) plus a pixel version (30 runs). The premise was **falsified** — a
pure JEPA does *not* collapse; the predictor+EMA (BYOL) asymmetry prevents it with no anti-collapse term at all, and
adding one *hurts*:

| arm (WalkerWalk) | geom R² | value R² | eff-rank |
|---|---|---|---|
| none (no anti-collapse) | **0.795** | **0.304** | 5 |
| + uniformity | 0.583 | 0.121 | 40 |

Uniformity *maximizes* eff-rank yet *destroys* the readouts — the same inverse eff-rank↔decodability pattern held
on pixels. The Part-5 "anti-collapse taxonomy" **reversed**: anti-collapse is load-bearing only in the narrow
closed-loop nav regime where the latent actually collapses; on broad DMControl data it's neutral-to-harmful.
(`exp/proposal_D2_pure_jepa/`, `exp/proposal_D3_pixel_jepa/`.)

#### Bet E — structural-entropy latent on TD-MPC2

SE anti-collapse added to the value-anchored TD-MPC2 latent (fixed-λ, return-AUC, n=3):

| task | default | SE | uniformity | SE+unif |
|---|---|---|---|---|
| CheetahRun | 58.9 | 23.4 | 36.6 | 29.0 |
| WalkerWalk | 293.8 | 110.3 | 89.9 | 62.5 |
| FingerSpin | 249.4 | 214.4 | 171.8 | 63.8 |

**default > SE on every task**; SE+uniformity is worse than either alone (no synergy). On a value-sufficient latent
the right anti-collapse is *none-extra*. **Hurts.**

#### Bet B — behavioral-prior taxonomy

A matched-budget-controlled sweep confirmed the campaign law: injected controllers buy **sample-efficiency where the
actuated DOF match the goal DOF** (ReacherHard OSC ~3× faster), and are dead weight or an anchor on unfit tasks. No
ceiling gain survives a same-budget PPO control.

**The tally: four of five bets came back null-to-harmful**, and the one preliminary positive (a learned 2-level
feudal hierarchy beating flat) was attributable to signal density, not abstraction. This is the pivot point of the
week: every constructive attempt to *add* structure to win reproduced the program's oldest finding — the TD value
pathway is the engine, and structure the value head doesn't consume is redundant.

So instead of asking "what can we add to beat PPO," we asked the sharper question: **why does the planner beat PPO at
all, and what inside it is doing the work?**

## Act III, scene 1 — the wall (Part 8–9, Paper 3 Claim 1)

The first dissection result is that PPO doesn't merely lose to the planner on some tasks — it hits a **categorical
exploration wall** on contact-critical hopper tasks.

- **HopperHop:** tuned PPO reaches **0/5 seeds ≥ 200 return at 472M steps/seed** (peak 53.8). SAC crosses 200 in
  **6/12 seeds by 5M and ~6/9 by 8M**; TD-MPC2 does it **6/6 by ~1M**.
- The wall **survives the standard exploration knob**: entropy-cost ×3 (peaks 4.2 / 73.9) and ×10 (48.1 / 64.1) at
  150M leave PPO in the same ≤74 class — under-exploration hyperparameters are *not* the explanation.
- **HopperStand** is a graded, near-categorical barrier: PPO escapes **2/16** (all four 285M runs walled at
  105–195); entropy ×3 put 1/4 over (a rate consistent with the baseline lottery, not a repair), ×10 0/4. Final
  three-method gradient: **TD-MPC2 5/5 (≤0.9M) ≫ SAC 7/10 (5M) ≫ PPO 2/16 (120–285M).**

And the wall is **causally scoped**: on **AcrobotSwingup** — unstable but *contact-free* — PPO learns fine (267–344,
n=4), no wall. The wall needs *contact-criticality*, not mere instability.

## Act III, scene 2 — the mechanism (Paper 3 Claim 3)

If the planner is the thing that clears the wall, *which of its losses* is the engine? The five-loss ablation
(mask one loss at a time), **4 tasks × n≥4 per arm**, MPPI-best per seed:

| Arm removed | CheetahRun | HopperHop | WalkerRun | HopperStand |
|---|---|---|---|---|
| none (full) | 721–795 | 287–570 | 680–731 | 911–946 |
| **value** | 16–58 dead | ~0 dead | 28–56 dead | 6–13 dead |
| **policy** | 123–192 dead | ~0 dead | 53–83 dead | 9–34 dead |
| reward | 5–31 (π 681–805) | ~0 (π full) | 44 (π full) | 265–542 (π 943–944) |
| consistency | 367–558 mild | 185–245 mild | 483–674 mild | 816–898 near-full |

The reading, confirmed on a fourth task the full model solves in 0.3M steps: **without the TD value loss the agent
cannot even learn to stand.** Value and policy are individually fatal everywhere; the reward head only feeds the
planner (the policy still reaches full strength without it); and the **consistency loss — the actual "world model"
objective — is the mildest cut on all four tasks.** This contradicts TD-MPC2's own ablation story and is the most
checkable claim in the program. Rounding out the matrix: **humanoid (21-DoF) is where only SAC survives** (Walk
625–909 in 4/5, Stand 918/922) — PPO nan under every config, and TD-MPC2 diverges to loss=nan in 6/6 default runs
(the one stable knob variant never learns, 21.8). The efficiency anchor at publication n: **TD-MPC2 HopperHop 5M =
420 ± 113 over 12 seeds.**

## Act III, scene 3 — the sufficiency law (Paper 4, the 2×5 grid)

The ablation proves *necessity* one loss at a time. The flip side is *sufficiency*: train the stripped agent
(consistency loss OFF from scratch) at full 5M budget — does it match the full model? Five tasks in:

| Task | stripped (consistency-OFF) | full baseline | gap | verdict |
|---|---|---|---|---|
| **HopperHop** | 306/449/443/477 + 165/475/481/511 (n=8) | 420 ± 113 (n=12) | ≈0 | **removable** (7/8 in band) |
| WalkerRun | 537/574/554/594 (565) | 709/705/753/782 (737) | **−23%** | load-bearing |
| CheetahRun | 527/528/516/524 (524) | 903/904/782/806 (849) | **−38%** | load-bearing |
| AcrobotSwingup | 297/233/352/256 (284) | 533/511/513/488 (511) | **−44%** | load-bearing |
| CartpoleSwingupSparse | 0/0/0/0 | 0/0/1.3/0 | n/a | both-fail (uninformative) |

I got the first cut of this wrong and corrected it in the open: with only Hop and Walker in, "removable when
exploration-bound, load-bearing when dense" looked clean — until **Acrobot** (exploration-*flavored*) showed the
*largest* gap. The reading that survives all four informative tasks is sharper: **the consistency loss underwrites
the planner's MPPI rollout quality wherever the planner carries learning** (Walker, Cheetah, Acrobot are all
planner-led). **HopperHop is the sole removable case** — there the policy head learns the behavior directly and the
planner only amplifies it, so the self-predictive loss is redundant (confirmed at n=8, 7/8 stripped seeds inside the
full band). Cartpole-sparse is a degenerate both-fail cell (the full model can't solve it either), so the grid's
evidentiary core is the four solved tasks.

## Act III, scene 4 — the first SOTA swing

Those findings hand us a constructive lever: if the consistency loss is a *rollout-quality regularizer for the
planner*, can we make it better and beat vanilla TD-MPC2 on exactly the planner-led tasks?

- **Bet 1 — Value-Aware Consistency (VAC):** weight each latent dimension's consistency error by its value
  sensitivity \\(|\partial Q/\partial z|\\), spending model capacity where the planner's value-ranking looks.
  Result, paired same-seed vs matched vanilla: **no-go.** Walker −8.9%, Cheetah −4.4% at 5M (worse at every
  checkpoint). The interpretation *reinforces* the sufficiency thesis: the planner's rollouts need **faithful
  dynamics on every dimension they might explore**, so concentrating capacity on currently-high-value dims
  *starves* the rest. Uniform consistency isn't just load-bearing — it's near-optimal in form.
- **Bet 2 — Uncertainty/Rollout-reliability-weighted Consistency (URC):** the better-motivated idea — weight
  consistency by the model's *own* rollout drift (open-loop vs teacher-forced one-step prediction), i.e. fix the
  dynamics exactly where multi-step rollouts compound error. This is running now (paired vs matched vanilla on
  Walker and Cheetah); early signal is mildly positive but far too young to trust. Verdict pending at 5M.

## The through-line

Read top to bottom, the week is one argument delivered four ways. Planning-as-exploration, JEPA anti-collapse,
SE-structured latents, glass variance-reduction, and value-aware consistency **all failed to beat a plain baseline**,
while the value/policy pathway proved individually fatal to remove and the uniform consistency loss proved both
load-bearing (on planner-led tasks) and near-optimal in form. The consistent moral: **in a value-based planner,
structure and reweighting buy nothing the TD value pathway doesn't already consume.** That is a negative result with
teeth — it's precisely why the three papers this week are a dissection, not a new method.

## The JEPA line, reopened — a plan

A fair objection: did we drop JEPA too early? The uniformity result looks like a flip — it *helped* the nav H-JEPA
early, then *hurt* everywhere later. It isn't a flip; it's **regime-dependence**. Anti-collapse helps if and only
if the latent actually collapses, which happened in exactly one place — closed-loop online goal-conditioned nav
(eff-rank → ~0; anti-collapse restored it, point-maze 0.53 → 0.95). On broad DMControl a pure JEPA doesn't collapse
(the predictor+EMA/BYOL asymmetry prevents it), so anti-collapse is redundant-to-harmful; on value-anchored TD-MPC2
the TD loss already keeps the latent value-sufficient. So we closed **anti-collapse-as-a-penalty** — but we never
tested **SE as *structure***: an SE-community partition that *defines* an abstraction, rather than an SE gradient
that competes with the predictor.

That is the open line, and it has a home — not dense value-based control (structure is provably redundant there),
but the three niches where structure showed a pulse or where there's no value signal to hide behind:

- **J1 — SE-community anti-collapse vs uniformity/VICReg, *in the collapse regime*** (cheap; finishes an open cell).
  Does a compact SE community structure preserve goal-decodability better than uniformity where the latent really
  collapses?
- **J2 — SE *as* the H-JEPA abstraction** (the real novelty). Use min-2D-SE community detection to partition the
  latent trajectory into **temporal subgoals**; the high-level planner plans over community transitions. This is
  "SE as structure," which we never built — targeted at long-horizon tasks where flat TD-MPC2 is weak.
- **J3 — SE-structured JEPA for offline/transfer** (paper-friendly). Frozen-encoder multi-task probes, no dense
  value; does SE's community geometry transfer better than plain/VICReg JEPA?

Decision rule: run J1 first (one box-day), gate the J2 build on its result, run J3 in parallel. No "JEPA+SE SOTA"
claim without a matched multi-seed win in a niche where structure isn't already redundant. Full design in
`PLAN_jepa_se_sota.md`.

## What's next

Two tracks, run without competing for the same claim. **(1) Value pathway:** the value-conditioned abstraction bet
(bisimulation, running now) tests structure inside the value head on control tasks — the redundancy wall.
**(2) Representation pathway:** the JEPA+SE plan above tests structure in the encoder on goal-conditioned /
hierarchical / transfer tasks — where structure already showed a pulse. If both null, the program's thesis is
airtight; if the SE-hierarchy (J2) wins, it is the abstraction-SOTA, correctly located *outside* the
value-redundant regime. Either way it is an honest result, which after a week like this one is the point.
