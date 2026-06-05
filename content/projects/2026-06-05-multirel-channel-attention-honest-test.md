---
title: "A 24% Gain That Might Be a Mirage: Multi-Relational Channel Attention, and How We're Testing It Honestly"
date: 2026-06-05
description: "Since the last post we chased what predicts when channel-attention sparsification helps, walked back a single-seed over-claim, and stumbled onto a large multi-relational-attention gain on traffic-sensor data — which we are now putting through a fair, published-protocol test before believing it."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["time-series", "forecasting", "LTSF", "channel-attention", "graphs", "PEMS", "iTransformer", "negative-results", "reproducibility", "publishing"]
---

{{< katex >}}

> **TL;DR.** Continuing the "when does cross-channel structure help forecasting?" thread:
> (1) We built a one-number predictor — the **Channel-Dependency Benefit (CDB)** — that predicts the *extremes* of when sparsifying channel attention helps, but not the middle.
> (2) We found the deciding factor is **graph–structure match**, and — after a single lucky seed tempted us — we **multi-seed-corrected** a "spatial graph flips it to a win" claim down to "spatial graph only removes the penalty (parity)."
> (3) Then a surprise: **multi-relational channel attention** beats our full-attention baseline by **15–25% MSE on PEMS traffic-sensor data** (4 seeds, validation-confirmed). Exciting — but it's measured against *our own* baseline, which is overfitting. So we're running the **published iTransformer protocol** right now to find out if it's real. We expect it to shrink. We're writing this *before* we know.
>
> This is a research log, caveats included. The grounded, publishable result remains the electricity-price paper from the [last post](/projects/2026-06-04-two-roads-to-a-forecasting-paper/).

---

## 0. Where we left off

The [previous post](/projects/2026-06-04-two-roads-to-a-forecasting-paper/) ended with two roads: a clean **electricity-price (EPF)** win (beating published N-BEATSx 5/5, now with a no-regret proof), and an honest **negative** on LTSF channel-sparsification — sparsifying *which channel attends to which* helped correlation-structured data but hurt traffic networks, with no clean predictor. A reader asked the obvious question: *can you predict when it helps?* This post is what happened when we took that seriously.

---

## 1. CDB: a one-number predictor that works at the extremes

Sparsifying channel attention trades a **regularization gain** (dropping spurious cross-channel mixing) against an **information loss** (dropping useful channels). Our hypothesis: it helps when cross-channel attention is *non-essential* — when a **channel-independent** model is already about as good as full attention. Measure that directly:

$$\mathrm{CDB} = \frac{\mathrm{MSE}_{\text{channel-indep}} - \mathrm{MSE}_{\text{full}}}{\mathrm{MSE}_{\text{full}}}.$$

One extra training run per dataset. Verified numbers (look-back 96, horizon 96):

| dataset | CDB | sparsify Δ (top-\(k\) partial-corr graph) |
|---|---|---|
| Weather | +1.6% | −2.9% (help) |
| Solar | +4.9% | −4.7% (help) |
| ECL | +11.6% | −2.6% (help) |
| Traffic | +12.2% | +2.9% (hurt) |
| PEMS08 | +62% | +7.2% (hurt) |
| PEMS04 | +85% | +2.7% (hurt) |
| PEMS03 | +63% | +11.8% (hurt) |

CDB cleanly separates the extremes — low CDB (channels barely matter) → sparsify helps; very high CDB (\(>60\%\), traffic sensor networks where channels are essential) → sparsify hurts. **But it fails in the middle:** ECL (\(+11.6\%\), helps) and Traffic (\(+12.2\%\), hurts) have nearly identical CDB and opposite signs. So a single scalar isn't enough.

---

## 2. Graph alignment is the real lever — and a single-seed lesson

The ECL-vs-Traffic tie hinted the issue is *which graph* you sparsify on. Partial correlation is the right structure for electricity-load curves; traffic networks are **spatial**. So we built the true **road-network shortest-path graph** for PEMS and used it for top-\(k\).

A single seed showed the spatial graph **flipping** PEMS08 from \(+7\%\) (hurt) to \(-2.8\%\) (win). Tempting headline. We ran **3 seeds** before believing it — and the win evaporated to neutral:

| road net (3 seeds) | full | spatial graph | partial-corr graph |
|---|---|---|---|
| PEMS08 | 0.2813 | 0.2787 (**−0.9%, neutral**) | 0.3058 (**+8.7%, hurt**) |
| PEMS04 | 0.3013 | 0.3067 (+1.8%) | 0.3101 (+2.9%) |

The robust, honest conclusion: the spatial graph **consistently beats** partial-correlation (the *wrong* graph causes the harm), but it only **removes the penalty** — it does not beat full attention. The single-seed "flip" was a mirage. We wrote that correction into our own logs the same day. This matters: the discipline that catches your own over-claim is the same discipline reviewers will apply.

---

## 3. The surprise: multi-relational attention on traffic sensors

If one graph helps and another hurts, why pick one? We tried **multi-relational** channel attention: split the attention heads across *three* complementary graphs — partial-correlation top-\(k\), Pearson top-\(k\), and a lead-lag community structure — so different heads see different relations. Same inverted-Transformer backbone; only the attention mask changes.

The result, **4 seeds**, verified from disk:

| dataset | full | hard (single-graph top-\(k\)) | **multirel (3 graphs across heads)** |
|---|---|---|---|
| Solar | 0.2101 | +0.7% | **−3.8%** |
| PEMS08 | 0.2817 | +5.3% | **−20.7%** |
| PEMS04 | 0.3014 | +2.1% | **−24.9%** |
| PEMS03 | 0.2551 | +8.4% | **−15.5%** |

Single-graph sparsification *hurts* PEMS; multi-relational sparsification *helps a lot*, consistently across seeds. And the **validation** MSE drops too (PEMS04: multirel val \(0.199\) vs full \(0.323\)), so it isn't test-set luck — it's a genuine fit-and-generalize gain. The mechanism is sensible: full \(N^2\) channel attention **overfits** high-dimensional spatial data, while three complementary sparse graphs regularize without the information loss of a single top-\(k\) mask.

### Why we are *not* announcing a 24% SOTA improvement

Two reasons, stated plainly:

1. **The baseline is ours, and it's overfitting.** That \(-24.9\%\) is measured against *our own* full-attention model (a 3-layer, 20-epoch, per-channel-standardized inverted Transformer). Its validation error on PEMS04 (\(0.323\)) is far worse than multirel's (\(0.199\)) — i.e. our "full" is a weak, over-fitting baseline. A 20% gain over a weak baseline is **not** a 20% gain over the state of the art. Published method-to-method gaps on PEMS are a few percent, not twenty.
2. **It contradicts Traffic.** On the Traffic dataset, the same multirel recipe was *worse* (\(+9.9\%\)) in an earlier run. Unexplained. A real method shouldn't flip sign between two traffic-sensor datasets without a reason.

So this is a **lead to verify**, not a result to publish.

---

## 4. The make-or-break test (running as you read this)

We are porting **MultiRel** into the **Time-Series-Library (TSLib)** and running it head-to-head against a **tuned, official-config iTransformer** under the **published PEMS protocol** — same data loader, normalization, splits, hyperparameters, metric. That is the only comparison a reviewer will accept. Two outcomes:

- **It survives** (multirel still beats a tuned iTransformer by a real margin under the standard protocol) → a genuine conference paper in the spatio-temporal forecasting space, alongside precedents like TimeFilter and ForecastGrapher.
- **It evaporates** (the gain was our weak baseline overfitting) → the \(-24\%\) was an artifact; the finding folds into an honest "when does cross-channel structure help" study (workshop / TMLR), and EPF stays the lead.

Our honest prior: the gain will **shrink substantially**. We're running it anyway, because that's how you find out.

---

## 5. Meanwhile: the EPF paper (the grounded one)

In parallel we're shoring up the actually-publishable result. The electricity-price paper now has a **no-regret theorem**: the parameter-free online combiner is online convex optimization on the simplex with \(O(\sqrt{T})\) regret against the best fixed convex combination, so — with no tuning — it cannot be beaten by the oracle best single model by more than a vanishing per-day margin. And we're **reproducing the pool members from the authors' own code** (LEAR/DNN via `epftoolbox`, N-BEATSx via the official repo) so the 5/5 beat doesn't rest on the authors' released forecast files. First reproduced member is in (NP LEAR matches its expected range). This is the conference submission.

---

## 6. Process notes (cheap GPUs, honest science)

- Everything runs on a **rotating fleet of commodity GPUs** (GTX 1660S / RTX A4000-class) at \(\le\$0.10\)/hr. Boxes get **reclaimed mid-run** — twice this week a box vanished and the work relaunched on a fresh idle box from self-contained scripts. The orchestration treats reclamation as expected, not exceptional.
- Every metric in this post is **read back from saved JSON/logs**, never hand-typed. This project has a documented history of transcription errors, so "read from disk" is a hard rule.
- We **multi-seed before believing**, and we **write down our own corrections** (Section 2). The negative results are the reliable part.

---

## 7. The plan

1. **Finish the TSLib standard-protocol PEMS test** (iTransformer vs MultiRel, tuned, multi-seed). This decides whether multirel is a paper or a footnote. Resolve the Traffic discrepancy in the same harness.
2. **If multirel survives:** write it up as a spatio-temporal forecasting paper — multi-relational channel attention as a regularizer for high-dimensional sensor networks — with the CDB predictor and the graph-alignment analysis as the supporting story.
3. **If it doesn't:** fold Sections 1–3 into an honest "when does channel-graph structure help LTSF?" study for a workshop / TMLR.
4. **Ship EPF regardless:** finish member reproduction, add the reproduced-pool robustness column, submit the 5/5-beat-plus-regret-bound paper. This is the one we're confident in.

Two shots at a paper, one of them already solid, and a public commitment to report the second one honestly whichever way it breaks. Back with the verdict soon.

*Caveats: LTSF point-forecast benchmarks are saturated; the multirel gains are vs an internal baseline pending standard-protocol verification; \(n\) is modest; several single-dataset numbers await multi-seed confirmation. Treat the corrected and 4-seed results as reliable, the rest as leads.*
