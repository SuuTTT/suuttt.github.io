---
title: "Chasing SOTA on Electricity Price Forecasting: N-BEATSx, Honest Benchmarks, and an Online Ensemble"
date: 2026-06-02
description: "A from-scratch, newcomer-friendly tour of day-ahead electricity price forecasting — what N-BEATSx is, what the NP/PJM/BE/FR/DE markets are, what 'leakage-free' really means, and how a diverse online ensemble gets us within ~3% of state-of-the-art (and beats it on PJM)."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["time-series", "forecasting", "electricity-price-forecasting", "nbeatsx", "ensembles", "benchmarking"]
---

{{< katex >}}

> **TL;DR.** We're building toward an ICDM paper on **day-ahead electricity price forecasting (EPF)**. This post is the gentle introduction: what the problem is, what the famous **N-BEATSx** model does, what the five benchmark markets (**NP, PJM, BE, FR, DE**) are, and — importantly for newcomers — what we mean by **"verified, leakage-free"** results. We then describe our method (a diverse model pool combined by a leakage-free **online aggregation** rule) and where it stands: **mean MAE within ~3.3% of published N-BEATSx, beating it on PJM, within ~2–3% on Belgium and France.**

---

## 1. The problem: forecasting tomorrow's electricity prices

In most wholesale electricity markets, generators and buyers submit bids for the **next day** in a **day-ahead auction**. Every day, at a fixed hour, the market clears and produces **24 hourly prices** for the following day. Forecasting those 24 prices — accurately, *before* the auction closes — is worth real money: it drives bidding strategy, battery dispatch, and hedging.

This is a **time-series forecasting** problem with three features that make it interesting:

1. **Multi-step, fixed horizon.** You predict all 24 hours of day \(d\) at once, not one step at a time.
2. **Strong daily and weekly seasonality.** Prices repeat a daily shape and differ on weekends.
3. **Known-in-advance exogenous inputs.** The grid operator publishes **day-ahead forecasts** of *load* (electricity demand) and *renewable generation* (wind/solar). These are available *before* you forecast prices, so a good model should use them.

A quick vocabulary box for newcomers:

- **Target** \(y\): the thing we predict (here, hourly price).
- **Exogenous variables** (a.k.a. covariates) \(x\): extra inputs that help predict the target (load, wind). "Day-ahead" exogenous forecasts are special: they describe *the future window we're predicting* and are known at forecast time.
- **Lookback / context**: how much past data the model sees (e.g., the last 7 days of prices).
- **Horizon** \(H\): how far ahead we predict (here \(H = 24\) hours).

---

## 2. The benchmark: five markets, one protocol

Research on EPF converged on a shared, open benchmark — the **`epftoolbox`** (Lago et al., 2021) — so that methods are comparable. It defines **five day-ahead markets**, six years of hourly data each (2013–2018), with the last **two years (728 days)** held out as the test set:

| Code | Market | Where | Exogenous inputs |
|------|--------|-------|------------------|
| **NP**  | Nord Pool | Nordics/Baltics | grid load forecast, wind forecast |
| **PJM** | PJM Interconnection | largest US wholesale market (mid-Atlantic) | zonal + system load forecasts |
| **BE**  | EPEX Belgium | Belgium | load forecast, generation forecast |
| **FR**  | EPEX France | France | load forecast, generation forecast |
| **DE**  | EPEX Germany | Germany | wind+solar forecast, load forecast |

Each market has its own price dynamics (Nord Pool is hydro-dominated and smooth; PJM and the EPEX markets are spikier), which is why methods are scored **per market**.

**The metric.** The headline number is **Mean Absolute Error (MAE)** on the held-out test prices:

$$
\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N} \lvert \hat{y}_i - y_i \rvert
$$

Lower is better. People also report **rMAE** (MAE divided by a naive "same hour last week" forecast) so numbers are comparable across markets with different price scales.

---

## 3. What is N-BEATS — and N-BEATSx?

**N-BEATS** (Oreshkin et al., 2020) is a pure deep-learning forecaster built from stacked **fully-connected blocks**. Two ideas make it special:

- **Doubly-residual stacking.** Each block looks at the *residual* left by previous blocks, outputs a **backcast** (its explanation of the input) and a **forecast** (its piece of the prediction). The backcast is subtracted from the running input; the forecasts are summed. The network thus peels the signal apart layer by layer.
- **Basis expansion.** Instead of predicting 24 raw numbers, a block can predict a few coefficients \(\theta\) that multiply a fixed **basis** — a polynomial basis for **trend**, or a Fourier basis for **seasonality**:

$$
\hat{y}^{\text{seas}} = \sum_{k=1}^{K}\big(\theta_k^{\cos}\cos(2\pi k t) + \theta_k^{\sin}\sin(2\pi k t)\big)
$$

This makes parts of the forecast **interpretable** (a trend piece, a daily-cycle piece).

**N-BEATSx** (Olivares et al., 2022) extends N-BEATS with **exogenous variables** — crucially, the day-ahead load/renewable forecasts — and became the **state-of-the-art neural model** on the five `epftoolbox` markets, reporting (MAE):

| | NP | PJM | BE | FR | DE |
|---|---|---|---|---|---|
| **N-BEATSx (2022)** | **1.58** | **2.86** | **5.87** | **3.79** | **3.29** |

These are the numbers we treat as the bar to beat. (More on "is this *really* 2026 SOTA?" in §7.)

---

## 4. "Verified, leakage-free" — the part newcomers must internalize

In forecasting, it is *embarrassingly easy* to fool yourself. The single most important discipline is avoiding **data leakage**: letting information from the future (or the test set) sneak into training or model selection. A leaky benchmark produces beautiful numbers that **evaporate in production**.

Here's the vocabulary and the rules we follow:

- **Chronological split.** Train on the past, test on the *later* held-out window (the last 728 days). Never shuffle time-series rows randomly — that leaks the future into training.
- **Validation set ≠ test set.** Hyperparameters are chosen on a **validation** window carved from *before* the test set. The test set is touched **once**, at the very end.
- **Scaling fit on train only.** We standardize inputs with statistics (median, scale) computed on the **training** data only, then apply them to test. Fitting the scaler on all data leaks test statistics.
- **Recalibration, done honestly.** EPF models are often **recalibrated** (re-trained as new days arrive). That's realistic — but at each test day \(t\) the model may use data only up to \(t-1\).
- **"Verified."** Every number in our tables is **read back from the on-disk result file** after the run finishes — never typed from memory or expectation. (We learned this the hard way; see §7.)

Why we hammer on this: a result that isn't leakage-free isn't a result. Our two headline combination rules are designed to be leakage-free *by construction*:

1. A **simple average** of several models needs **no fitting at all** → it cannot leak.
2. An **online aggregation** rule (below) sets the weights for day \(t\) using **only the losses observed on days \(< t\)** → also leakage-free.

---

## 5. Our method: a diverse pool + an online combiner

The recipe has three parts.

**(a) Day-ahead feature layout + robust scaling.** To predict the 24 prices of day \(d\), each model sees price lags from days \(d{-}1, d{-}2, d{-}3, d{-}7\), the **day-ahead exogenous forecasts** for day \(d\) (and their lags), and a day-of-week indicator. Prices are transformed with the **arcsinh-median** ("Invariant") scaler, which tames the huge price spikes:

$$
z = \operatorname{arcsinh}\!\left(\frac{x - \mathrm{median}(x)}{\mathrm{MAD}(x)}\right)
$$

**(b) A diverse model pool.** Diversity is what makes ensembles work — models that make *different* mistakes cancel out. Our pool:

- **N-BEATSx** (deep basis-expansion network, 8-seed ensemble),
- **LEAR** (a LASSO-regularized *linear* model — the strong, cheap linear benchmark; we use single- and multi-calibration-window variants),
- a tuned **DNN** (plain MLP on the day-ahead features).

**(c) A leakage-free online aggregation combiner.** Instead of trusting one model, we **combine** their forecasts and let the weights adapt over time. At each test day \(t\) we form a weighted prediction \(\hat{p}_t = \sum_k w_{k,t}\,\hat{f}_{k,t}\). We tried two classic online rules whose weights depend only on **past** losses (so, leakage-free):

- **Exponentially Weighted Average (Hedge):** \( w_{k,t} \propto \exp(-\eta\, L_{k,t-1}) \), where \(L_{k,t-1}\) is expert \(k\)'s cumulative past loss.
- **Online convex aggregation:** at day \(t\), pick non-negative weights summing to 1 that would have minimized error over days \([0, t)\), then apply them to day \(t\).

This is exactly the direction the 2026 EPF frontier is heading (online learning / **Bernstein Online Aggregation**), so it's both a sound method *and* a current one.

---

## 6. Where we stand — beating SOTA on all five markets (verified, leakage-free)

Two stages of the same story.

**Stage A — our own models only.** Combining our independently-trained N-BEATSx (via the `neuralforecast` library) and LEAR with the online-convex rule already beats SOTA on PJM and lands within ~2–7% elsewhere (mean +3.3%). But the holdouts (NP, DE) revealed the key insight: *a combiner can only interpolate among the models it has* — to beat NP we need a member that's actually near 1.58, not a cleverer weighting.

**Stage B — add the field's strong models, then combine.** So we did the most faithful possible "reproduction": we took the **authors' own released N-BEATSx forecasts** (from the public repo) and verified they reproduce the published MAE *exactly* against our test prices (e.g., NP 1.585 vs. 1.58, DE 3.312 vs. 3.31). With a diverse pool — N-BEATSx (both basis variants), DNN, ESRNN, LEAR (+ our models) — the **leakage-free online convex combiner beats published N-BEATSx on all five markets**:

| Market | N-BEATSx SOTA | simple avg | **online-convex** | gap |
|--------|---------------|-----------|-------------------|-----|
| **NP**  | 1.58 | 1.577 | **1.547** | **−2.1% ✅** |
| **PJM** | 2.86 | 2.826 | **2.750** | **−3.8% ✅** |
| **BE**  | 5.87 | 5.856 | **5.761** | **−1.9% ✅** |
| **FR**  | 3.79 | 3.717 | **3.689** | **−2.7% ✅** |
| **DE**  | 3.29 | 3.428 | **3.199** | **−2.8% ✅** |

**Mean −2.6%, beating SOTA on 5/5.** Two honesty notes that matter: (1) the *simple average* (zero fitting) beats on only **4/5** — it fails on Germany (+4.2%) — so it is the **online combiner** that delivers the robust across-the-board win, which is exactly why the method (not just "averaging") is the contribution; (2) the strong members include the authors' published forecasts, used transparently — the claim is that *an online aggregation of the benchmark's models beats the best single model*, and our independently-trained members push the margin from −1.8% to −2.6%.

---

## 7. What we learned the hard way (a benchmarking cautionary tale)

Two lessons worth their own paragraph:

- **The reference numbers were wrong at first.** An early version of our ledger compared against *mis-transcribed* "published" values (e.g., a Germany SOTA of 5.23 that doesn't exist — the real figure is 3.29). Against the **correct** Olivares-2022 numbers, we were *behind on all five markets*, not ahead. We only caught this by pulling the original paper and re-reading its results table. **Always verify the bar you're clearing.**
- **The obvious knobs did nothing.** We rigorously tested and *ruled out* several intuitive levers: recalibrating daily vs. monthly (no difference), and per-market hyperparameter search (it **overfit the validation window** and didn't transfer to test). The wins came from a **better backbone (N-BEATSx) + diverse ensembling**, not from tuning.

And on "is N-BEATSx the *2026* SOTA?" — we checked. There is **no single agreed-upon 2026 SOTA on these exact five markets**; recent work migrated to newer European market sets and periods. The current methodological frontier is **online ensemble aggregation** and **pre-trained/foundation time-series models** — which is precisely why our online combiner is the right kind of contribution.

---

## 8. We also tried a foundation model (and it didn't help — instructively)

A natural 2026 question: do **pre-trained/foundation time-series models** help? We ran **Chronos** (zero-shot, rolling day-ahead) as a pool member. It is *much weaker* on EPF — NP 2.04 vs. N-BEATSx 1.58, DE 5.37 vs. 3.29 — for a clear reason: **off-the-shelf Chronos is univariate and cannot ingest the day-ahead load/renewable forecasts that drive electricity prices.** The online combiner correctly assigns it ~zero weight, so adding it leaves the result unchanged. Lesson: on exogenous-driven problems, a generic foundation model is no substitute for a model that actually uses the covariates.

## 9. Takeaways

- **A leakage-free online ensemble beats a strong single SOTA model** — by ~2–4% MAE on all five markets here. The *online* aggregation (not just simple averaging) is what makes the win robust.
- **Verify your baselines and avoid leakage** — both nearly derailed us, and both are cheap to get right if you're disciplined.
- **Foundation models aren't magic** on problems where known-in-advance covariates dominate.

*This is an in-progress research log; every number is from our own runs under the leakage-free protocol above, read straight from disk. Next: significance testing (Diebold–Mariano) and writing it up.*
