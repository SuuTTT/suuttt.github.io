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
turned. The scoreboard, all multi-seed and matched-budget:

| Bet | Result |
|---|---|
| A — planning-as-exploration | **Refuted.** A1 from scratch on HopperHop (n=3, 80k) mildly reversed; on WalkerRun (n=3, 140k) null; **decisive** on CartpoleSwingupSparse (n=3, 300k) — steering the planner toward novelty did *not* crack the sparse task a reactive policy fails. |
| A (novelty-MPPI half) | **Closed null.** Disagreement/RND bonuses inside the planner, β∈{0.3,1.0}, HopperHop 1M, then a *matched-seed* vanilla control: novelty **worse on 4/4 same-seed pairs** (442/509/321/359 → 5.6/285/259/274), one catastrophic break. |
| C — variance reduction | **Null** (n=4–5). Structure didn't tighten the seed spread either. |
| D — pure-JEPA on DMControl | **Reversal, firmed** (WalkerWalk/CheetahRun/ReacherEasy, n=3): the anti-collapse term that *helped* geometric goal-conditioned nav **hurts** value-based control. Pixel-JEPA (D3): null (30 runs). |
| E / D1 — SE-structured latent | **Hurts**, like uniformity; SE+uniformity shows no synergy on TD-MPC2 (n=3, L16). |

Four of five bets came back null-to-harmful, and the one "positive" (a learned 2-level feudal hierarchy beating flat)
was preliminary and, on inspection, attributable to signal density rather than abstraction. **This is the pivot
point of the week.** Every constructive attempt to *add* structure to win reproduced the program's oldest finding:
the TD value pathway is the engine, and structure the value head doesn't consume is redundant.

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

## What's next

The consistency lever (URC) finishes its head-to-head first. In parallel we're queuing the one direction that would
genuinely test *abstraction* against our own nulls: a **value-conditioned** structure — forcing the abstraction into
the value pathway, the single way our redundancy work said structure *could* matter but never did. If even that ties
vanilla, the program's contribution crystallizes as "we tried, rigorously, to beat TD-MPC2's world-model design with
abstraction and reweighting, and it is already near-optimal" — which, after a week like this one, would be an honest
and useful thing to have proven.
