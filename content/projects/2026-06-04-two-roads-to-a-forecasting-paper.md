---
title: "Two Roads to a Forecasting Paper: a Clean SOTA Beat, an Honest Negative, and What Each Needs to Publish"
date: 2026-06-04
description: "A research log on two parallel time-series forecasting directions — beating published N-BEATSx 5/5 on electricity prices with a tuning-free online ensemble, and a rigorous (and partly negative) study of when channel-attention sparsification helps — plus an honest assessment of what each needs before it's a real paper."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["time-series", "forecasting", "EPF", "N-BEATSx", "LTSF", "channel-attention", "graphs", "causal", "negative-results", "publishing", "ICDM"]
---

{{< katex >}}

> **TL;DR.** Two directions, two very different outcomes.
> **(1) Electricity-price forecasting (EPF):** a leakage-free, *parameter-free* online ensemble beats **published N-BEATSx on all 5 markets** (mean **−2.6% MAE**, Diebold–Mariano significant at \(p \le 0.007\)), and beats established online aggregators (EWA / MLpol / BOA) with no learning rate. This is the genuinely publishable result.
> **(2) Long-term forecasting (LTSF) — "when does channel-attention sparsification help?":** a careful, multi-dataset, multi-seed study that ends in an *honest negative*. Sparsifying which channels attend to which **helps correlation-structured data (−2 to −5%)** but **hurts spatial sensor networks**, an information-theoretic predictor (ICR) **failed to generalize**, and the one positive lever (matching the graph to the data's structure) only reaches **parity** with full attention on the hard cases. Real, reproducible, mechanistic — but not a new SOTA.
>
> This post documents both, what we actually ran, and — bluntly — what each needs before it's an ICDM-grade paper.

---

## 0. Why two roads?

Most "we beat SOTA" posts only show the road that worked. Research isn't like that. We ran two forecasting directions in parallel on a fleet of cheap GPUs. One produced a clean, defensible win. The other produced a rigorous *negative* result. Both are worth writing down — and being honest about which is which is the whole point.

---

## Road 1 — EPF: a tuning-free online ensemble beats N-BEATSx 5/5

### The benchmark, for newcomers
**Electricity-price forecasting (EPF)** on the open `epftoolbox` benchmark (Lago et al., 2021): predict the next day's 24 hourly day-ahead prices on five markets — **NP** (Nord Pool), **PJM** (US), **BE / FR / DE** (EPEX). Metric: **MAE**, evaluated on the last 728 days. The strong published baseline is **N-BEATSx** (Olivares et al., 2022):

| Market | Published N-BEATSx MAE |
|---|---|
| NP | 1.58 |
| PJM | 2.86 |
| BE | 5.87 |
| FR | 3.79 |
| DE | 3.29 |

### What we did
We assembled a **diverse pool** of forecasters — the authors' own released N-BEATSx (generic + interpretable), DNN, ESRNN, and LEAR variants, plus our own re-implementations — and combined them with a **leakage-free online aggregator**: at each test day, the combiner weights are fit *only on past test days*, then applied to today. We use a **parameter-free non-negative convex / ridge** combiner — no learning rate, no tuning.

### What we beat
- **Published N-BEATSx on all 5 markets**, mean **−2.6% MAE** (pool of the authors' published forecasts alone: −1.8%). Diebold–Mariano significant at \(p \le 0.007\).
- A *simple average* only wins 4/5 (it fails DE). The **online** combiner is what delivers the robust 5/5.
- **Our online convex/ridge \(>\) established online aggregators** (EWA / Hedge, MLpol, BOA from the `opera` line of work) — and it's parameter-free, whereas EWA/BOA needed an oracle-tuned learning rate \(\eta\). That's a frontier-vs-frontier comparison, not "beat a weak baseline."

We also verified the boring-but-critical things: a day-ahead feature layout with arcsinh-median scaling, RevIN-style per-window normalization (which fixed a catastrophic exogenous blow-up), and that **daily recalibration is *not* the accuracy driver** (monthly vs daily recal is null) — useful for anyone trying to reproduce this cheaply.

### What's already paper-ready
- A clean, **statistically significant SOTA beat** on a standard public benchmark.
- A **method** (parameter-free online convex aggregation) with a fair comparison against the established online-aggregation frontier.

### What to add before submitting
1. **Exact reproduction** via the authors' repos (`cchallu/nbeatsx`, `epftoolbox` LEAR/DNN with released hyperparameters + daily recal). Our generic pipeline plateaus ~12% above the published single-model number; the *win comes from aggregating the authors' published forecasts*, which is clean — but reproducing members at true published strength removes the "is your pipeline just weak?" reviewer objection.
2. **A regret bound** for the parameter-free convex combiner (relate it to follow-the-leader / online Newton). Pairing the empirical 5/5 with theory turns an applied result into a method+theory paper — a much stronger ICDM submission.
3. Optional: the **real `opera` (R) BOA/MLpol** via `rpy2` for an unimpeachable baseline, and a richer pool (a Transformer member; a foundation model *fine-tuned with the exogenous price drivers*).

**Verdict:** this is the lead paper. With (1)+(2) it's a credible ICDM main-track submission.

---

## Road 2 — LTSF: when does channel-attention sparsification help?

### The question
Modern multivariate forecasters either treat channels independently (PatchTST, DLinear) or let every channel attend to every other (iTransformer). A natural middle ground: a **graph** prior — let each channel attend only to a few "relevant" others. Does pruning cross-channel attention help? And can we **predict in advance** when it will?

We worked the channel-token (inverted-Transformer) setting, look-back 96, horizon 96, on a self-contained trainer where **only the channel-attention prior varies** (so any delta is attributable to that prior). Datasets: Weather, ECL, Traffic, Solar, PEMS03/04/08, plus ETT/Exchange — spanning correlation-structured (load, weather) and physically-networked (road sensors) data.

### What we tried, in order
**(a) An information-theoretic predictor (ICR).** Define the *information-concentration ratio*: for each target channel, the fraction of cross-channel predictive gain captured by its top-\(k\) neighbours vs. all channels,
$$\mathrm{ICR}_i = \frac{R^2_{\text{top-}k} - R^2_{\text{self}}}{R^2_{\text{all}} - R^2_{\text{self}}}.$$
At \(n=3\) datasets it looked beautifully predictive. At \(n=7\) it **scrambled** — e.g. PEMS04 has *higher* ICR than ECL yet sparsification hurts it. **ICR does not predict the sign.** A clean lesson in not trusting \(n=3\).

**(b) A channel-dependency predictor (CDB).** Run a channel-independent (self-only attention) model and measure how much full attention beats it:
$$\mathrm{CDB} = \frac{\mathrm{MSE}_{\text{CI}} - \mathrm{MSE}_{\text{full}}}{\mathrm{MSE}_{\text{full}}}.$$

| dataset | CDB | sparsify Δ (partial-corr top-\(k\)) |
|---|---|---|
| Weather | +1.6% | −2.9% (help) |
| Solar | +4.9% | −4.7% (help) |
| ECL | +11.6% | −2.6% (help) |
| Traffic | +12.2% | +2.9% (hurt) |
| PEMS08 | +62% | +7.2% (hurt) |
| PEMS04 | +85% | +2.7% (hurt) |
| PEMS03 | +63% | +11.8% (hurt) |

CDB predicts the **extremes** (low → help; \(>60\%\) road networks → hurt) but **fails in the middle**: ECL (+11.6%, *helps*) and Traffic (+12.2%, *hurts*) have nearly identical CDB and opposite outcomes. So no single scalar predicts the sign.

**(c) Graph alignment — the one positive lever.** The ECL-vs-Traffic puzzle hints the issue is *which graph*. Partial-correlation is the right structure for load curves, but road networks have a *spatial* structure. So we built the **true road-network shortest-path graph** for PEMS and used it for top-\(k\), with **3 seeds**:

| road net (3 seeds) | full | **spatial** graph | **partial-corr** graph |
|---|---|---|---|
| PEMS08 | 0.2813 | 0.2787 (**−0.9%, neutral**) | 0.3058 (**+8.7%, hurt**) |
| PEMS04 | 0.3013 | 0.3067 (+1.8%) | 0.3101 (+2.9%) |

The spatial graph **consistently beats** partial-correlation on both road networks (≈8.8 points on PEMS08) — so the *wrong* graph is what causes the harm. **But** — and this is the honest part — a single lucky seed showed the spatial graph *flipping* PEMS08 to a clear −2.8% win; **multi-seed corrected that to −0.9% (neutral)**. The right graph **removes the penalty** but does **not beat full attention** on strongly channel-dependent data.

### The synthesis (the actual answer)
Whether channel-sparsification helps is governed by **graph–structure match**, with CDB scaling how much a wrong graph costs:
- right graph + correlation-structured data (ECL/Weather/Solar) → **accuracy gain** (−2 to −5%);
- right (spatial) graph + road-network data → **parity** with full attention (an *efficiency* win, not accuracy);
- wrong graph → penalty that grows with channel-dependence (+3 to +12%).

A directed/causal angle (lead-lag asymmetry) was a dead end here — the asymmetry is negligible (0.008–0.019).

### What's paper-able
An honest **"when and why does cross-channel structure help LTSF?"** study: the ICR-fails-to-generalize result, the CDB-predicts-extremes result, the graph-alignment effect (right graph removes the penalty), and the multi-seed correction of a tempting-but-wrong single-seed flip. The integrity and the mechanism are the contribution.

### What to add before submitting
1. **More datasets and a spatial graph for Traffic-like data** to pin the boundary (we only had usable adjacency for PEMS04/08).
2. **An efficiency framing** — if sparse attention reaches parity at lower cost on road networks, quantify the FLOP/latency savings (parity-at-speed is a legitimate contribution).
3. **Multi-seed everything** (we caught one over-claim already) and significance tests.
4. Honestly, accept it is a **study / negative-result paper**, not a SOTA-method paper.

**Verdict:** good science, but **not an ICDM main-track paper on its own.** Best home: an ICDM *workshop*, **TMLR** (which welcomes rigorous negative results), or an analysis section paired with Road 1.

---

## What can go in a paper vs. what's still missing

| | Road 1 — EPF | Road 2 — LTSF sparsification |
|---|---|---|
| Headline | Beats published N-BEATSx **5/5**, DM \(p\le0.007\) | No SOTA; graph-alignment reaches **parity** on hard cases |
| Method novelty | Parameter-free online convex aggregation (vs EWA/BOA) | Diagnostic study; no new winning method |
| Strength | Significant, fair frontier comparison | Reproducible mechanism + honest negatives |
| Ready now | Result + method | The *study* |
| Must add | Exact reproduction; **regret bound** | More datasets; efficiency framing; multi-seed |
| Best venue | ICDM main track (applied/method) | ICDM workshop / TMLR / analysis section |

---

## Honest publishing strategy

1. **Lead with EPF.** It's the real win. Add the regret bound and exact reproduction → solid ICDM submission.
2. **Use the LTSF study as a companion**, framed truthfully as a "when does cross-channel structure help" study — not dressed up as a method.
3. **Do not** submit the sparsification work as a standalone "method" claiming wins it doesn't have. We already caught one single-seed over-claim; multi-seed discipline is non-negotiable.

---

## Reproducibility notes

- All forecasting runs are leakage-free (graphs and combiner weights use only past/train data), look-back 96, horizon 96, single self-contained trainer where only the channel prior varies.
- Hardware: a fleet of commodity GPUs (GTX 1660S / RTX A4000-class), all experiments sequential per box, every number read back from saved JSON/logs — no hand-typed metrics.
- Datasets: `epftoolbox` (EPF); ETT/ECL/Traffic/Weather/Exchange (Time-Series-Library); Solar (LSTNet); PEMS03/04/08 with road-network distance graphs.
- The two earlier companion posts cover the online-ensemble method itself for [LTSF](/projects/2026-06-02-beating-saturated-sota-online-ensembles-ltsf/) and [EPF](/projects/2026-06-02-nbeatsx-epf-ensemble-leakage-free/).

*Caveats: the LTSF benchmark is saturated (point-forecast models sit within a couple percent); several sparsification numbers are single-seed unless stated as multi-seed; \(n\) is modest (7–10 datasets). Treat the negative results as the reliable part.*
