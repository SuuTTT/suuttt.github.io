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

### 2b. How this relates to Farebrother et al. 2026 (and why it's not the same method)
The cited prior art is **"Compositional Planning with Jumpy World Models"** (Farebrother, Pirotta,
Tirinzoni, Bellemare, Lazaric, Touati, 2026). On a close read, it is a *different* method that shares the
word "jumpy":

| | Farebrother 2026 | This work |
|---|---|---|
| Jumpy model predicts | state-occupancies of **pre-trained policies** across timescales (off-policy) | **k-step latent dynamics from primitive actions** (online, inside TD-MPC2) |
| Macro-planning | yes — over **sequences of policies/options** (compositional) | yes — macro-MPPI over **k-step primitive-action chunks** (horizon k·n_macro=24) |
| Requires | a library of **pre-trained policies** | nothing beyond TD-MPC2 + a k-step head |
| Per-task-type regime | not reported | **mapped here** |

Both have a macro-planning module, but they compose *different units* (policies vs primitive-action chunks).
Our result **reproduces their headline regime** (a large win on long-horizon manipulation, ~consistent with
their +200% on long-horizon tasks) on a simpler, online instantiation — and **adds what they do not report:
a per-task-type breakdown, including a failure regime.**

### 2c. Why does it help on manipulation and *hurt* on Hopper? (mechanism)
It is **not** model accuracy: the macro model's k-step error is *uniformly low* on every task (jumpy_err <
iter1_err everywhere, including Hopper). The regime split is governed by **(task credit-horizon) ×
(dynamics forgiveness):**

- **Manipulation (win):** long-horizon, goal-reaching *and* forgiving — a slightly-wrong multi-step plan
  just means re-approach. A 24-step effective horizon improves credit assignment toward the grasp.
- **Hopper (hurts → 0):** an unstable, fall-prone limit cycle needing **reactive per-step balance**. A long
  macro-plan is brittle — even with an accurate model, committing multiple steps can't react to *this*
  step's balance error → it falls and the episode dies. Longer horizon = less reactive = catastrophic.
- **Cheetah (null):** stable runner, dense reward, reactive 1-step already near-optimal → no long-horizon
  credit to gain, no instability to trigger → wash.
- **Cartpole-sparse (null):** should benefit, but is exploration-limited/high-variance → horizon isn't the
  bottleneck.

**Implication:** temporal abstraction in MBRL pays off as a function of *task credit-horizon × dynamics
forgiveness* — not representational abstraction and not raw model accuracy. It **wins on long-horizon
forgiving tasks and actively harms unstable reactive-control tasks.** This failure regime appears to be a
**new finding** relative to Farebrother (who test long-horizon manipulation/navigation, not fast unstable
locomotion).

**What it enables:** a **task-adaptive macro-planner** — long horizon only where it helps (long-horizon +
forgiving), reactive otherwise — selected from cheap signals (reward sparsity + an episode-fall/instability
proxy, *not* the uniformly-good model error). This would beat both fixed-jumpy and fixed-vanilla on a mixed
suite, sitting on top of the Farebrother line rather than duplicating it.

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

**Fix probe — raising the cap (`SCALE_MAX`), n=3, Δfinal vs vanilla (=4):**

| Task | cap = 8 | cap = 16 | peak-final gap (cap16) |
|---|---|---|---|
| Open-Cabinet | −263 (null) | **+640 [254, 1034]** ✅ | 2159 → 1452 |
| Orientation | +386 [−1, 800] (borderline) | −51 (null) | 1367 → 779 |
| Cheetah | null (no harm) | null (no harm) | — |

The fix **provably stabilizes** (peak-final gap shrinks on both manipulation tasks) and gives a
**CI-separated win on Cabinet (+640)** with no locomotion harm — the first **non-abstraction, non-prior-art**
positive signal toward beating TD-MPC2, on a *mechanistically diagnosed bug class*.

**The nuance (important):** there is **no single fixed cap that wins both tasks** — Cabinet's value-IQR is
large (needs ≥16), Orientation's is smaller (16 over-normalizes its peak; ~8 is better). The optimum is
**task-dependent because the value-IQR is task-dependent.** So the principled fix is not a bigger constant
but to **remove the cap entirely** and let the RunningScale self-normalize to each task's true IQR. That
**uncapped/adaptive** variant is running now; if it wins Cabinet *without* costing Orientation's peak, it's a
clean, universal, one-line improvement to TD-MPC2.

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
