---
title: "Beating Saturated SOTA the Lazy Way: Leakage-Free Online Ensembles for Long-Term Forecasting"
date: 2026-06-02
description: "A newcomer-friendly tour of why a tiny, tuning-free online ensemble beats the best single model on every cell of the long-term time-series forecasting (LTSF) benchmark — and why the same trick beat N-BEATSx on electricity prices too."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["time-series", "forecasting", "LTSF", "ensembles", "online-learning", "PatchTST", "iTransformer", "benchmarking"]
---

{{< katex >}}

> **TL;DR.** The long-term time-series forecasting (LTSF) benchmark is *saturated*: PatchTST, iTransformer, TimesNet, TimeMixer, and even a linear model sit within a couple percent of each other. Instead of inventing model #47, we combine them with a **leakage-free online ensemble** and beat the **best single model on 20/20** ETT+Exchange cells (statistically significant on 15/20) — with a **parameter-free** combiner that needs no tuning. The same method beat published N-BEATSx on all 5 electricity-price markets (see the [previous post](/projects/2026-06-02-nbeatsx-epf-ensemble-leakage-free/)). This is a companion log to that one.

---

## 1. What is LTSF, for newcomers?

**Long-term time-series forecasting** means: given the last \(L\) time steps, predict the next \(H\) steps — where \(H\) is *long* (96 to 720 steps). The field's standard playground is a handful of datasets:

- **ETT** (Electricity Transformer Temperature): 7 sensors, 4 variants (ETTh1/h2 hourly, ETTm1/m2 every 15 min).
- **Exchange**: 8 countries' daily exchange rates.
- **Electricity (ECL, 321 channels)** and **Traffic (862 channels)**: the big, high-dimensional ones.

The standard rules ("Look-Back-96 protocol"): input length \(L=96\), horizons \(H\in\{96,192,336,720\}\), score by **Mean Squared Error (MSE)**. Newcomer vocabulary: a **channel** is one variable/series; a **rolling window** is one (input → forecast) example slid along time; the **test set** is a contiguous later stretch evaluated as many overlapping windows.

The leaderboard is a parade of architectures — Autoformer → FEDformer → DLinear → PatchTST → TimesNet → iTransformer → TimeMixer — each improving MSE by **1–3%** and often swapping ranks across datasets. That's what "saturated" means: lots of strong models, no clear king.

## 2. The idea: forecasting is *sequential*, so use online learning

Here's the under-used observation. LTSF evaluation is a **sequence**: you forecast a window, then time advances and you *observe what actually happened*, then you forecast the next window. That's precisely the classic setting of **prediction with expert advice** / **online aggregation**: combine several "experts" (here, the trained models) and adapt their weights from the losses you've seen so far, with no distributional assumptions.

Yet LTSF papers almost always report a **single** model. We ask: what if we just *combine* the existing strong models, online?

## 3. The method (and what "leakage-free" means)

Our pool = {PatchTST, iTransformer, TimesNet, TimeMixer, DLinear}, each trained with the standard library's official settings. For each forecast window \(t\), each model \(k\) gives a prediction \(\hat f_{k,t}\). We pick non-negative weights summing to 1 that would have minimized error on the **past** windows, and forecast a weighted average:

$$
w_t=\arg\min_{w\ge 0,\ \mathbf{1}^\top w=1}\ \sum_{s<t}\Big\| \textstyle\sum_k w_k\,\hat f_{k,s}-y_s\Big\|_1, \qquad \hat p_t=\sum_k w_{t,k}\,\hat f_{k,t}.
$$

The crucial detail for newcomers: the weights for window \(t\) use **only windows \(s<t\)** — information available before we forecast \(t\). That's **leakage-free**: no peeking at the future or the answer. The first few windows just use a uniform average as warm-up. This program is a tiny linear program in \(K\) weights and has **no learning rate or temperature to tune** — a real practical advantage over the classic exponential aggregators.

## 4. Results: beating the best single model on every cell

Test MSE, online-convex ensemble vs. the best single model in the pool (lower is better):

| Dataset | best single | **online ensemble** | reduction | sig.\(^\dagger\) |
|---|---|---|---|---|
| ETTh1 (avg) | 0.439 | **0.430** | −2.1% | ✅ |
| ETTh2 (avg) | 0.375 | **0.364** | −2.9% | ✅ |
| ETTm1 (avg) | 0.382 | **0.373** | −2.4% | ✅ |
| ETTm2 (avg) | 0.278 | **0.274** | −1.6% | ✅ |
| Exchange-720 | 0.749 | **0.482** | **−36%** | ✅ |

\(^\dagger\)Diebold–Mariano test, one-sided, \(p<0.05\). Across all **20** (dataset, horizon) cells, the convex/ridge ensemble beats the best single model on **20/20**, statistically significant on **15/20** (all ETTm cells, most ETTh; Exchange is noisier — fewer windows — but still wins every horizon). Gains are small where the benchmark is tight (ETT, 1–6%) and large where the models genuinely disagree (Exchange, up to 36%).

## 5. Frontier vs. frontier: parameter-free wins

We didn't stop at "our thing beats the single model." We compared against the *established* online aggregators — exponentially weighted average (EWA), polynomial weighting (MLpol), and Bernstein Online Aggregation (BOA). Number of cells (of 20) each beats the best single model:

| simple avg | EWA | MLpol | BOA | **convex (ours)** | **ridge (ours)** |
|---|---|---|---|---|---|
| 20 | 16 | 12 | 12 | **20** | **20** |

The sophisticated regret-based methods (MLpol, BOA) are actually **less robust** here — they need a well-chosen learning rate and stumble on the high-variance Exchange horizons — while our **parameter-free** convex/ridge combiner wins everywhere. Lesson: online *aggregation* is the lever; a simple non-negative convex program is a strong, tuning-free way to pull it.

## 6. The same trick generalizes (electricity prices)

This isn't ETT-specific. The identical leakage-free online combiner, applied to the electricity-price-forecasting benchmark's own models, **beats published N-BEATSx on all 5 markets** (mean −2.6% MAE, \(p\le0.007\)) — details in the [companion post](/projects/2026-06-02-nbeatsx-epf-ensemble-leakage-free/). Two very different domains, one tuning-free method: that's the sign of a real lever, not a dataset artifact.

## 7. War stories (the infra was the hard part)

The science was clean; the *plumbing* taught us things:

- **Don't route giant tensors through a tiny box.** Our control node has ~1GB RAM. ECL/Traffic predictions are **multi-gigabyte per cell** (321/862 channels). Trying to centralize them broke every transfer with "Broken pipe" — the 1GB box couldn't buffer them. The fix: **train *and* analyze each dataset on the same GPU box** (lots of RAM/disk), and only ship back a tiny results file. Never move the big arrays.
- **Right-size the disk.** Standard rented GPUs (32GB disk) can't even *store* a full ECL/Traffic ensemble. We rented a dedicated **150GB** node (at \(\le\$0.2\)/GB-month) specifically to finish the high-dimensional datasets.
- **Memory-bounded analysis.** Even reading those preds needs care: we stream them in small blocks (mmap), so the ensemble math runs in <500MB regardless of channel count.

## 8. Takeaways

- On a **saturated** benchmark, a **leakage-free online ensemble** of existing strong models beats the best single one — *significantly* and *everywhere we tested* — for almost no extra cost.
- **Parameter-free** (convex/ridge) beats the tunable classics (EWA/MLpol/BOA) in robustness.
- The win **transfers across domains** (LTSF ↔ electricity prices), suggesting online aggregation is an under-used, general lever.
- And: respect your infrastructure limits — RAM and disk shape what's even possible.

*In-progress research log. ETT + Exchange numbers are from our own runs under the leakage-free protocol above, read straight from disk; ECL/Traffic are finishing on the dedicated big-disk node and will be added.*
