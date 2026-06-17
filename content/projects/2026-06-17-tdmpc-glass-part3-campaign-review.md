---
title: "TD-MPC-Glass, Part 3: A Full Campaign Review — What Beats TD-MPC2, What Doesn't, and One Live Lead"
date: 2026-06-17
description: "A consolidated review of the whole campaign to beat TD-MPC2 at the architecture/abstraction level under a strict fair protocol. The bottom line: no explicit abstraction beats it, and we now have a criterion that explains why. We reproduced the one real (prior-art) win — jumpy/temporal abstraction on contact manipulation — from scratch, and learned a sharp peak-vs-final lesson. Every backbone, frozen-dynamics, and motion-phase variant is null. The one genuinely new, non-abstraction lead: TD-MPC2's value-scale cap under-normalizes advantages on high-return tasks, causing a late-training collapse that a one-line fix partly repairs."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["tdmpc2", "glass-jax", "world-models", "jumpy-models", "structural-entropy", "reinforcement-learning", "dmc", "maniskill", "reproducibility", "rliable", "vastai", "mechanism-check"]
---

{{< katex >}}

> A standing-meeting-ready field report. Parts 1–2 put our abstraction ideas through a fair protocol
> and most dissolved to null. This post consolidates everything since: a *criterion* for why abstraction
> is redundant, a clean reproduction of the one real win (which isn't ours), a comprehensive null sweep,
> and one live non-abstraction lead worth pursuing.

## 0. The bottom line
- **No explicit abstraction beats TD-MPC2** — across state clustering, temporal, relational, compositional,
  network backbone, and motion-phase variants. This is not bad luck; we have a **redundancy criterion**
  that predicts it.
- **The one real win is prior art, not ours:** a *jumpy* (k-step macro) world model genuinely beats vanilla
  on **contact manipulation** — reproduced from scratch — but it's Farebrother et al. (2026), and it's a
  pure *planning-time* effect, not a new representation.
- **One genuinely new, non-abstraction lead:** TD-MPC2's value RunningScale is **capped at 4.0**, but the
  true value-IQR on manipulation is **≥16** → policy advantages are under-normalized ~4× → **late-training
  collapse**. Raising the cap gives a **CI-separated win on Open-Cabinet (+640)** with no locomotion harm.

## 1. The redundancy criterion (the headline result)
TD-MPC2's SimNorm latent is already a **value-sufficient** abstraction: a *linear* probe decodes the value
function from the latent at \\(R^2 = 0.9994\\). Once that holds, any explicit abstraction you bolt on is
**redundant** — the information is already there in a form the value head and planner can use.

This criterion correctly predicted **null** for, in order:
geometric prototype clustering, behavioral/reward clustering, bisimulation (actively hurts),
community-detection skills, Laplacian/eigenpurpose exploration, SI2E/VCSE and world-model-latent SE
exploration, value-equivalence losses, distractor-robustness, and sparse-task rescue — **16+ levers, all null.**

## 2. The one real win, reproduced — and a sharp lesson
A jumpy k-step world model (predict \\(z_{t+k}\\) directly, plan with macro-MPPI) genuinely beats vanilla
on contact manipulation. We reproduced it from scratch at **n=5 seeds × 500k steps**:

![Learning curves: jumpy vs vanilla TD-MPC2 across six tasks, mean ± SEM over seeds](/images/d2_learning_curves.png)

![Δ(jumpy − vanilla), peak and final, 95% bootstrap CIs](/images/d2_summary_bars.png)

| Task | type | Δpeak | Δfinal | verdict |
|---|---|---|---|---|
| PandaPickCube | contact | **+1017** | **+1114** | genuine win (both metrics) |
| PandaPickCubeOrientation | contact | **+697** | **+1625** | genuine win (both metrics) |
| PandaOpenCabinet | contact | −372 | +929 | *stability-only* (peak ≤0) |
| CheetahRun | locomotion | −62 | −96 | neutral/worse |
| HopperHop | locomotion | **−121** | **−91** | jumpy **hurts** |
| CartpoleSwingupSparse | sparse | +168 | +110 | null |

**The lesson (peak vs final):** a *final-only* read made manipulation look like a 3/3 sweep. On the fair
**peak** metric only Pick/Ori are genuine; Cabinet's "win" is **vanilla collapsing late in training**, not
jumpy capability. We now always report **both**.

**Where the win comes from (ablation):** running jumpy's *loss* without its *planning* (`jumpy_k=8`, no
macro-MPPI) is **neutral everywhere** — the gain *and* the locomotion harm both come from the **macro-MPPI
planning**, not the representation. Jumpy is a pure planning-time intervention.

## 3. The comprehensive null sweep (what does NOT beat TD-MPC2)
- **Network backbone** (n=5, clean): a gated-residual MLP (`resmlp`) and group-attention (`attn`) do **not**
  beat TD-MPC2's plain MLP; an earlier "+40%" was a small-n (2–4) mirage that did not replicate. `resmlp`
  also **does not stack** with jumpy — it *hurts* it (Pick −600, Ori −718, CI-separated).
- **Frozen-random dynamics:** a random, untrained dynamics net **collapses** planning (Pick −1764 peak) →
  the **learned dynamics is essential**. So the *latent* is value-sufficient, but the *dynamics* must be
  learned for planning. (Refutes "value-sufficient latent ⇒ dynamics redundant.")
- **Motion-phase abstraction — closed in every framing:** subgoals, jump-length gating, switching-dynamics,
  geometric clustering, kNN-graph structural-entropy, and phase-balanced replay are **all null**. Motion
  phases are real (gait/contact cycles) but are either redundant with the value-sufficient latent
  (model-structure uses) or unrelated to the failure mode (data uses). A kNN-SE clustering predicts model
  error no better than raw motion magnitude, and worse than plain k-means.

## 4. The live lead: an under-normalization bug, not an abstraction
Diagnosing the **late-training collapse** (vanilla drops 59–80% from peak on Cab/Ori): the value
**RunningScale is pinned at its cap (4.0)** the entire run, while the true value-IQR is **≥16**. TD-MPC2
normalizes the policy advantage by this scale; capped 4× too low, advantages are over-scaled → policy
gradients too large → late-training **oscillation/collapse**. (The critic already uses LayerNorm, so the
textbook fix is a no-op — this is a *scale-cap* issue.)

**One-line fix probe (`SCALE_MAX` 4→16), n=3:**

| Task | Δfinal vs vanilla | peak-final gap | read |
|---|---|---|---|
| Open-Cabinet | **+640 [254, 1034]** ✅ | 2159 → 1452 | **CI-separated win** |
| Orientation | −51 (null) | 1367 → 779 | stabilized, but 16 over-normalizes peak |
| Cheetah | −59 (null) | — | no locomotion harm |

The fix **provably stabilizes** (peak-final gap shrinks on both manipulation tasks) and **wins on Cabinet**
with no locomotion harm. `=16` over-corrects Orientation's peak, so we're tuning the cap (sweep in flight).
This is the first **non-abstraction**, **non-prior-art** positive signal toward beating TD-MPC2, and it's a
*mechanistically diagnosed bug class* (under-normalized advantages on high-return tasks).

## 5. Methodology contributions (defensible regardless of the headline)
- **Mechanism-check before fan-out:** every bet gets a cheap offline gate first; this killed multiple ideas
  (phase-switching, kNN-SE, phase-replay) before any GPU training.
- **Peak *and* final, paired-bootstrap CIs, pre-registered gates, single-variable, read-from-JSON** — the
  fair protocol that repeatedly caught inflated "wins."
- **Reproducibility infra:** a public benchmark/task-queue repo
  ([`wm-bench`](https://github.com/SuuTTT/wm-bench)), a multi-GPU fleet harness, and checkpoint streaming to
  HuggingFace so no result is ever lost.

## 6. Status & decisions
- **In flight:** `SCALE_MAX` cap sweep (find the stabilization sweet-spot that keeps Cabinet's win without
  Orientation's peak loss) → if it threads the needle, a clean universal stabilization that beats TD-MPC2.
- **Paper A (the redundancy criterion + null campaign) is self-contained and shippable now.**
- **Open question for discussion:** is the right paper-B story (a) the *value-scale fix* (a concrete TD-MPC2
  improvement), (b) a *task-adaptive planner* (macro-plan on contact, vanilla on locomotion → beat fixed
  TD-MPC2 on a mixed suite), or (c) consolidate everything into one "why world models are hard to beat,
  and the one place they're not" paper.
