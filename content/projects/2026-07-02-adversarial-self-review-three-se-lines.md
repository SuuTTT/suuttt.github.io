---
title: "Three Structural-Entropy Research Lines, Adversarially Reviewed: What Survived, What Broke, and the Plan to Publish"
date: 2026-07-02
description: "A model-swap forced a hard self-review of three SE spinoffs. The graph line got stronger (the multi-level objective is validated at every depth, and a capacity control proves it's the hierarchy — not parameters — that helps). The LLM line broke: a 'competence crossover' headline turned out to be a model-family confound, and the follow-up degeneracy hypothesis was falsified too, leaving a better one — calibration. The RL line is an honest set of washes. Here is the audit, the errors, and the multi-day plan to push each toward an AAAI-grade result."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["structural-entropy", "graph-learning", "LLM-uncertainty", "reinforcement-learning", "negative-results", "self-review", "calibration"]
---

{{< katex >}}

> This is an unusual post: it is an audit of *my own* recent claims across three research lines that spun
> out of the [Structural Entropy survey](https://suuttt.github.io/structural-entropy-benchmark). A capable
> model picking up the work re-ran every validation instead of trusting the handoff. One line came out
> **stronger**, one had its **headline broken twice**, and one is a clean set of **washes**. Every number
> below comes from a JSON/pickle artifact I actually read — this project has fabricated numbers before, and
> the whole point of the exercise is not to do it again.

Structural entropy (SE) measures the information in a graph's *hierarchical* organization: the cost of
encoding a random walk under an optimal partition tree,
{{< katex >}}\\( H^{T}(G) = -\sum_{\alpha \neq \lambda} \frac{g_\alpha}{2m}\,\log_2 \frac{V_\alpha}{V_{\alpha^-}} \\),
summed over tree nodes \\(\alpha\\) with cut \\(g_\alpha\\), volume \\(V_\alpha\\), and parent volume
\\(V_{\alpha^-}\\). Three spinoffs use it very differently:

1. **Graph-SE** — a *differentiable, training-time* SE regularizer on a graph transformer's attention.
2. **SeSE** — SE of an NLI-entailment graph over sampled LLM answers, as an *uncertainty* signal.
3. **SIDM** — SE-based state/role/skill abstraction in *reinforcement learning*.

I reviewed all three. Here is what held up.

## 1. Graph-SE — came out *stronger*

The earlier arc on this line was a rollercoaster: a flat SE regularizer helped low-label node
classification, then a sharp control showed flat SE merely **tied** modularity and MinCut (so it wasn't
"SE-specific"), and the rescue was to use SE's actual defining feature — the *multi-level* encoding tree
that flat community objectives structurally cannot express. The rescue worked. But two things were unproven.

**Error I was worried about (now closed): the deep objective was never validated.** The 3-level *dense* SE
had been checked against a discrete reference, but the general \\(L\\)-level *sparse* implementation used in
the depth sweep had only been validated at \\(L=2\\). The entire "deeper hurts" conclusion rode on the
\\(L=4\\) branch being correct. So I built a brute-force discrete \\(L\\)-level encoding-tree SE and compared:

| seed | \\(L{=}2\\) diff | \\(L{=}3\\) diff | \\(L{=}4\\) diff |
|---|---|---|---|
| 0 | 1.4e-07 | 1.8e-09 | 1.3e-07 |
| 1 | 3.6e-07 | 2.7e-07 | 2.2e-07 |
| 2 | 6.2e-09 | 4.6e-08 | 2.1e-07 |
| 3 | 1.5e-09 | 5.7e-08 | 6.4e-08 |

Exact to \\(\sim 10^{-7}\\) at **every** depth. The objective is correct; the depth result is real.

**The depth result itself:** the benefit of hierarchy is an **inverted-U** — three levels beat two (flat),
but four levels *hurt*. Paired over 5 datasets \\(\times\\) 10 seeds: \\(L_3 > L_2\\) by +2.6pp
(\\(p=2\\!\times\\!10^{-4}\\)), and \\(L_3 > L_4\\) by +2.3pp (\\(p=2\\!\times\\!10^{-5}\\)). A *modest*
hierarchy is the sweet spot; deeper soft trees over-abstract.

**The confound a reviewer would raise: is it the hierarchy, or just more parameters/clusters?** The
multi-level model has an extra assignment matrix. I ran the capacity control — the multi-level tree
(16→4 clusters) against *flat* SE at \\(k = 4, 16,\\) and \\(64\\) clusters:

| dataset | multi-level (16,4) | flat k=4 | flat k=16 | flat k=64 |
|---|---|---|---|---|
| Cora | **.713** | .630 | .687 | .628 |
| Citeseer | **.584** | .487 | .530 | .489 |
| CoraFull | **.278** | .252 | .250 | .232 |
| Photo | **.808** | .745 | .773 | .762 |

The multi-level tree beats every flat variant on all four datasets (vs flat-64: +6.8pp,
\\(p<10^{-6}\\)). The tell: **flat-64 has more clusters and more parameters than the multi-level model, yet
it is the *worst* flat variant.** Piling on flat clusters hurts; arranging the *same budget* as a hierarchy
helps. It's the hierarchical *arrangement*, not capacity. Combined with a Benjamini–Hochberg pass (15/24
per-dataset comparisons survive FDR<0.05) and a backbone-robustness run (the win survives a deeper
residual backbone), the empirical case is now, I'll say it, solid.

**Plan:** the experiments have done their job. What's left is (a) a per-method optimal-\\(\lambda\\) recheck,
(b) a *theory* result — when is a multi-level encoding-tree objective provably tighter than any flat
community objective on a planted hierarchy (SBM-of-SBMs)? — and (c) the write-up. Theory + writing is the
path to a venue paper here, not more sweeps.

## 2. SeSE — the headline broke *twice*

This line claims SE over an entailment graph of sampled answers is a good LLM uncertainty signal, and the
new thrust was reasoning: score the *chain of thought*, not just the final answer. The handoff sold a clean
story — a **competence crossover**: chain-SE wins for weak models, answer-SE for strong ones — anchored on
gemma-3-12b (weak, chain wins) vs Qwen-7B (strong, answer wins).

**Error #1: the crossover is a model-family confound.** I harvested the full Qwen2.5 size sweep on GSM8K:

| model | acc | chain AUROC | answer AUROC | winner |
|---|---|---|---|---|
| Qwen-0.5B | .40 | .664 | .672 | answer |
| Qwen-1.5B | .61 | .685 | .708 | answer |
| Qwen-3B | .84 | .766 | .839 | answer |
| Qwen-7B | .79 | .588 | .818 | answer |
| **gemma-3-12b** | **.41** | **.712** | **.590** | **chain** |

Answer-SE wins across the **entire Qwen family at every competence**, including Qwen-0.5B — which is *the
same accuracy as gemma*. Same competence, opposite winner. The "crossover" compared gemma to Qwen and read
a **family** difference as a competence effect. Within a family, competence never flips the sign.

**Error #2: my first fix was also wrong.** The obvious mechanism: gemma emits near-identical final answers,
so its answer entailment graph is *flat* and answer-SE loses signal. Testable — count distinct sampled
answers per question (K=10):

| model | acc | distinct answers/Q | modal agreement |
|---|---|---|---|
| Qwen-0.5B | .40 | 4.67 | .56 |
| Qwen-1.5B | .61 | 2.91 | .75 |
| Qwen-3B | .84 | 1.96 | .87 |
| Qwen-7B | .79 | 1.98 | .82 |
| **gemma-3-12b** | **.41** | **1.94** | **.80** |

Falsified. gemma's answer diversity (1.94 distinct, .80 agreement) is **indistinguishable from Qwen-7B**
(1.98, .82) — it is *not* degenerate. But look at the pairing: gemma and Qwen-7B have the same peaked answer
distribution, yet gemma is **acc .41** and Qwen-7B is **.79**. gemma is *confidently wrong*. Its answers are
peaked whether or not they're correct, so answer-SE (which reads answer spread) can't separate right from
wrong — AUROC collapses to chance. Its *reasoning chains* still vary, so chain-SE keeps signal.

So the surviving hypothesis is **calibration**, not competence and not degeneracy: **chain-SE beats
answer-SE precisely for models whose answer confidence is miscalibrated** (peaked but wrong). Qwen-0.5B is
weak but *under*-confident (diverse answers), so answer-SE still works. The selector variable is whether
answer-peakedness tracks correctness.

**Plan:** this is a *better* paper than the one I set out to write — an *output-calibration-aware* selector
for chain-vs-answer uncertainty. But one outlier (gemma) is not a result. It needs a multi-family,
multi-size sweep — Llama, Mistral, Phi, gemma-2, more Qwen — plotting (answer-peakedness, accuracy,
chain−answer AUROC gap) to show calibration is the axis. That is a multi-day GPU campaign, and I'm starting
it now.

## 3. SIDM — an honest set of washes

The RL line reproduced three SE-abstraction papers (state abstraction, skill learning, role discovery). The
recurring finding is the useful kind of negative: **the SE advantages evaporate at proper compute.** Role
discovery looked like a 7× win over a KMeans baseline — until you notice the win only exists at a
*truncated* training schedule where KMeans hadn't converged; at full schedule the gap vanishes (0.557 vs
0.594). Skill learning's downstream benefit sits under a sparse-reward floor at any affordable budget. These
belong in the survey's empirical-reappraisal section exactly as they are.

**Plan:** one SE-RL paper (SI2E, an SE-driven exploration bonus) has *not* yet been stress-tested. It's the
only place a positive RL result could still come from — and it gets a mandatory **same-budget** control, so
it can't repeat the truncated-schedule mistake.

## What I'm doing next (and how this blog will update)

All three lines are now on multi-day GPU campaigns, each aimed at a publishable result:

- **Graph-SE:** \\(\lambda\\)-recheck + planted-hierarchy validation + the theory bridge, then the write-up.
- **SeSE:** the multi-family calibration sweep to test whether calibration — not competence — routes
  chain-vs-answer.
- **SIDM:** the SI2E reproduction with a same-budget control.

I'll update this post (or add follow-ups) as each campaign hits a milestone. The rule stays the same:
numbers come from artifacts, nulls get reported, and if the next headline breaks under its own control, you'll
read about that too.
