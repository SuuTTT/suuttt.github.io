---
title: "TD-MPC-Glass, Part 19: A Week in Review — Turning Three Dissections Into Instruments"
date: 2026-07-15
description: "The week of July 8–15. Last week (Part 10) left a pile of open 'why' questions. This week turned each into a measured instrument and framed the program around four named hypotheses: a value-sufficiency bottleneck that fingerprints every task; a collection-mode dissociation that inverts on Cheetah (removing the world model HELPS) with one variance mechanism behind both regimes; a first data point on the anti-collapse (JEPA/SE) line; and a sharper question — is the world model itself just a learned abstraction, and if so which knob is doing the work? Tables, plots, and the experiments now queued to close the gaps."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["world-models", "TD-MPC2", "PPO", "abstraction", "JEPA", "VICReg", "mujoco-playground", "reproducibility", "weekly-review", "vastai"]
---

{{< katex >}}

> A review of one week — July 8 through July 15 — picking up where
> [Part 10](https://suuttt.github.io/projects/2026-07-08-tdmpc-glass-weekly-review-part10/) left off.
> Part 10 closed with a recurring result — *abstraction and reweighting buy nothing the value pathway
> doesn't already use* — but stated it as an anecdote across a few "dissections." This week was about
> converting each dissection into a **measured instrument** and stating the program as a small set of
> **named, falsifiable hypotheses** so the rest of this notebook (and the papers) can refer to them
> instead of to "paper 1 / paper 3." Every figure is read from disk; nulls are reported as loudly as
> positives.

## The four hypotheses this program is testing

To keep the rest of this post unambiguous, here are the hypotheses by name. Later sections say which
each one supports or dents.

- **H-COMPRESS (value-information compressibility).** Explicit structure helps a value-based agent only
  where the *value function needs little of the latent*. The quantity that predicts "is there room to
  help" is how compressible the value-relevant information is — not horizon length, not task family.
- **H-WM-ABSTRACT (the world model *is* a learned abstraction).** The consistency loss that trains the
  latent dynamics builds a task-relevant compression of the observation — i.e. a *learned* abstraction.
  So "abstraction" and "world model" are not two separate things; the world model is one point on the
  abstraction axis, and both should be ablated on the same footing.
- **H-COLLECT (collection mode gates the world model's value).** How load-bearing the world model is
  depends on whether data is collected by the *policy* or by the *planner* (MPPI rolling the model to
  act). Planner-collection can make the model matter *more* — or, if it is inaccurate, actively *harm*.
- **H-VARIANCE (one mechanism, two regimes).** Under planner-collection the world model's net effect is
  a **variance inflation** of the planner's targets; whether that variance is net-positive or net-harmful
  is set by the mean/variance ratio of the resulting returns.

Three of these got sharp evidence this week; H-WM-ABSTRACT is the one I'm now building experiments
around (see *What's queued*).

## Instrument 1 — the value-sufficiency bottleneck (evidence for H-COMPRESS)

Part 10's probe was **decode-\(R^2\)** — how well the latent reconstructs the observation. It *saturates*
for both a strong policy and a collapsed one, so it can't tell you whether structure has room to help.
The replacement is a **value-sufficiency bottleneck (VBN)**: insert a width-\(D\) bottleneck **on the
value head's input**, sweep \(D\in\{16,32,64,128\}\) against the unmodified agent, and read the *shape*
of the curve. That shape is a per-task fingerprint of how much of the latent the value function needs.

![VBN fingerprints](/images/part19-vbn-fingerprints.png)

| Task | D=16 | D=32 | D=64 | D=128 | vanilla | fingerprint |
|---|---|---|---|---|---|---|
| WalkerRun (n=5) | 625 (86%) | 643 (88%) | 666 (92%) | 694 (95%) | 727 | **flat-high** — most compressible; D=16 already 86% |
| CheetahRun (n=5) | 548 (64%) | 589 (69%) | 627 (73%) | 726 (85%) | 855 | **strictly monotone** — no width suffices |
| AcrobotSwingup (n=6, med) | 211 (41%) | 251 (49%) | 311 (61%) | 304 (59%) | 511 | **ramp-to-D64** — least compressible; saturates at D=64 |
| **HopperHop** | — | — | — | — | — | **not yet swept (queued)** — the removable cell; H-COMPRESS predicts *flat-high* |

The instrument agrees with the intervention. The stripped-vs-full ablation (delete the consistency loss)
ranks the tasks the *same way* the fingerprint does:

$$\text{HopperHop } 0\% \;<\; \text{WalkerRun } {-}7.5\% \;<\; \text{CheetahRun } {-}23.8\% \;<\; \text{AcrobotSwingup } {-}44\%.$$

That co-ranking is the evidence for **H-COMPRESS**: structure helps where the value function needs little
of the latent (the flat-high regime), because that is where an added objective has slack to reorganize a
nearly-sufficient code. This *answers Part 10's Q6* ("why did abstraction only ever help HopperHop and
nav?").

**Two honest limits of the instrument** (both now queued as experiments, not swept under the rug):

1. **It probes the *value* head only.** That is deliberate — the planner selects actions by rolling the
   model and scoring with the value/reward heads, so value-sufficiency is the pathway that gates
   planning. But it means the fingerprint does not yet speak to the *reward* or *dynamics* heads; a
   head-by-head bottleneck sweep is queued.
2. **HopperHop has no row.** We only swept the three DMControl tasks above; the Hopper VBN sweep is
   queued (H-COMPRESS predicts flat-high, matching its removable status).

> A rigor note: the Acrobot fingerprint was called "step-at-128" mid-week. On re-harvest, two seeds had
> only reached ~2.7M steps and were dragging the means; on the six complete-at-5M seeds it is a smooth
> ramp saturating at D=64. Same story, corrected location. Report medians (two collapse cells).

## Instrument 2 — collection mode and the world model (evidence for H-COLLECT + H-VARIANCE)

Our JAX variant collects with the *policy*; canonical TD-MPC2 collects with the *planner*. Under
**planner-collection**, re-running stripped-vs-full on three tasks gives a double dissociation with a
genuine inversion (finals @2.5M, now **n=9**):

![Collection-mode dissociation](/images/part19-dissociation.png)

| Task | full (median) | stripped (median) | Δ | reading |
|---|---|---|---|---|
| HopperHop (n=5) | ~455 | ~468 | ≈0 | removable — both arms stable |
| WalkerRun (n=9) | **739** | **606** | **−18.0%** | load-bearing — higher but volatile |
| CheetahRun (n=9) | **327** | **475** | **+45% (stripped > full)** | **inversion** — the model is *actively harmful* |

On Cheetah, removing the model's accuracy **helps by 45%** — we had *pre-registered* the opposite
(≥15% degrade). That falsification is the spine of the section, and it is **evidence for H-COLLECT**:
the world model's value is not a constant, it is gated by collection mode and task.

The *why* is **H-VARIANCE**, and it is readable straight off the eval traces. The full model inflates
eval-return variance **~3× on both tasks**. The tasks differ only in scale relative to the mean: on
Walker the model lifts the mean to ~740 so the swings stay net-positive; on Cheetah the swings drag
finals to ~117, below the stable stripped model (~475), so variance tips into net harm. **One mechanism,
two regimes, set by the mean/variance ratio.**

This also dissolves Part 10's **Q8** (Hopper is both the sharpest PPO wall *and* the only removable cell):
stripping consistency removes the model's *accuracy* but not the *planner*. TD-MPC2 clears Hopper via
planning + off-policy data (**Q5**: deterministic actor, no entropy — SAC fails Hopper 0/9 while the
planner-free TD core is 8/8), and the model is removable *because the planner carries it*.

So the two headline conclusions — **(a)** the world model's value is task-dependent, **(b)** planning-in-
collection is (mostly) a sample-efficiency operator — now have a shared mechanism underneath (a), not
just an observation: it is the variance-inflation of H-VARIANCE, with the sign flip explained.

## Instrument 3 — the anti-collapse line, and a sharper question (H-WM-ABSTRACT)

The anti-collapse (JEPA/SE) bets were parked since Part 7. This week I ran the two canonical levers on
the task the VBN instrument flags as *least compressible* (CheetahRun): **uniformity** vs **VICReg**
against a matched vanilla baseline.

![JEPA #59 null on Cheetah](/images/part19-jepa59-null.png)

| arm | CheetahRun (last-6 median, n=2) | vs vanilla (~818) |
|---|---|---|
| uniformity (urc) | 725.8 | −11.3% |
| VICReg (vac) | 751.3 | −8.2% |
| vanilla | ~818 | — |

Uniformity ≈ VICReg, both *below* vanilla — a null, and the *expected* null under H-COMPRESS: TD-MPC2's
latent is already shaped by the value/TD objective, which prevents the collapse these regularizers exist
to fight.

> **Update (07-15, live).** A third Cheetah seed finished, taking this to **n=3**: uniformity **739**
> (−10%), VICReg **779** (−5%), still both below vanilla (~818) — the null holds. The **WalkerRun**
> boundary test (the *most*-compressible regime, where anti-collapse has the most room to help) is
> running now across a rented 4×4070 plus the freed 3060 box; at the interim ~2M/5M checkpoint all three
> arms are clustered ~695–724 with **no separation yet** — early-consistent with the null. If it stays
> null through 5M, H-COMPRESS is airtight across both VBN-fingerprint extremes. Final numbers next update.

**But there is a caveat I want to state plainly, because it is the most interesting gap.** "Anti-collapse
regularizer" and "consistency loss" are only two points on the abstraction axis. They are *not* the
richer abstraction machinery — **SimNorm**, the **structural-entropy / glass** latents, **clustering /
discretization**, or **behavioral** abstractions like **TAMP / skill-options**. The redundancy claim, so
far, is proven for anti-collapse + consistency; whether those *other* mechanisms are also redundant is an
open, and separately testable, question. It is queued.

### The world model as an abstraction (H-WM-ABSTRACT)

Here is the reframing this review is really driving at, and the reader's question that prompted it:
*the consistency loss trains the world model, so is the world model a kind of abstraction?* I think the
promising hypothesis is **yes** — the latent-dynamics model is a *learned, task-relevant compression*,
which is exactly what an abstraction is. Under that view:

- "Stripped vs full" is *already* an abstraction ablation — it ablates the **learned** abstraction (the
  world model) while leaving the planner.
- The SE/glass/SimNorm experiments are ablations of an **imposed** abstraction.
- The clean experiment is to **cross them**: ablate the imposed abstraction *and* the world model on the
  same grid, per task, so we can see which piece — imposed structure, learned dynamics, or the planner —
  is the *working part*. My current bet: on the flat-high tasks the working part is the value pathway and
  both abstractions are redundant; on the ramp/monotone tasks the learned world model carries real weight
  (that's what the −24%/−44% strip costs say) while imposed structure still adds nothing.

That cross-ablation is the headline experiment queued below.

## The one-line law, restated by hypothesis

> **H-COMPRESS:** explicit structure is redundant exactly where the value objective already supplies what
> it would add. **H-WM-ABSTRACT:** the world model is itself a learned abstraction, and *it* is the
> abstraction that carries weight where the value code is not nearly-sufficient. Where neither holds,
> structure buys variance (**H-VARIANCE**), not a ceiling.

## What's queued (issues filed to the repo)

The review surfaced concrete gaps; each is now a tracked issue on `SuuTTT/tdmpc-glass`:

1. **Official-parity / dual-implementation test** — run the same tasks on official TD-MPC2 vs our JAX
   reimplementation, matched-seed, to certify the reimplementation before the papers lean on it.
2. **VBN on HopperHop + head-by-head bottleneck** — fill the missing fingerprint row and extend the
   bottleneck from the value head to the reward and dynamics heads.
3. **Do the *other* abstractions help? (SimNorm / SE-glass / clustering / TAMP)** — the redundancy claim
   is currently only tested on anti-collapse + consistency; test the richer abstraction menu, guided by
   the VBN fingerprint (predict where each should help).
4. **World-model-as-abstraction cross-ablation (H-WM-ABSTRACT)** — jointly ablate imposed abstraction and
   the learned world model per task, to isolate the working part.
5. **Deeper theory for task-conditional load-bearingness** — formalize *why* the mean/variance ratio (and
   HopperHop's conjunctive reward + exploration wall) makes some tasks removable-cell and others not.

## Next

- **Freeze compute on Papers A & 3, write.** Both are data-complete at their final \(n\).
- The **WalkerRun anti-collapse boundary sweep** is running now across a rented 4×4070 plus the freed
  3060 box; its verdict (null or not) closes the H-COMPRESS boundary.
- The **H-WM-ABSTRACT cross-ablation** is the first experiment of the new front.
