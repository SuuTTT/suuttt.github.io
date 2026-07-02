---
title: "Explainer: The R² Probes and Pareto-Frontier Analyses in Our RL Campaign — What They Are, and Whether They Worked"
date: 2026-07-02
description: "Two analysis tools show up constantly in the TD-MPC-Glass posts: linear/ridge R² probes of learned latents, and Pareto-frontier comparisons of algorithms. Neither is a hallucination — both are standard, real methods with real code and disk-backed outputs. But one of our most prominent R² uses was methodologically invalid and had to be retracted, which makes this project an unusually good case study in the difference between a real method and a valid measurement."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["methodology", "explainer", "R2", "linear-probes", "pareto-frontier", "world-models", "reinforcement-learning", "TD-MPC2"]
---

{{< katex >}}

> A reader asked: the earlier sessions of this project lean heavily on "R² analysis" and "Pareto frontier
> analysis." What are those? Are they hallucinated? Do they actually work? This post explains both tools from
> scratch, shows exactly how this project used them (with the receipts, good and bad), and gives the honest
> verdict on each.

## 1. R² — the coefficient of determination

### What it is

Given a target variable \(y\) (say, the true value of a state) and a prediction \(\hat y\) made from some
features (say, a world model's latent \(z\)), R² measures the fraction of the target's variance the prediction
explains:

$$R^2 \;=\; 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}.$$

\(R^2 = 1\) means perfect prediction; \(R^2 = 0\) means you did no better than always guessing the mean;
negative values mean worse than the mean (possible on held-out data).

### How it's used in representation learning ("linear probes")

The standard recipe — used across interpretability and RL representation research — is the **linear probe**:
freeze a learned representation \(z\), fit a *linear* (or ridge) regression from \(z\) to some quantity you care
about, and read the held-out R². The logic: if a *linear* readout can decode the quantity, the representation
"exposes" it — no extra nonlinear computation needed. In this project the probes are ridge regressions on an
80/20 split, and they answer questions like:

- Can you decode the **state geometry** (positions/velocities) from a JEPA latent? (Thread D's "geom-R²")
- Can you decode **value / return-to-go** from a world-model latent? (the "value-decodability" question)
- Does decodability **survive out-of-distribution** — e.g. a world model trained on 4 objects, probed at 14?
  (the entity-graph OOD sweep in Paper A)

So: **R² probing is a completely standard, real method** — not something a model made up. Every R² in the
current posts traces to a JSON on disk produced by real regression code, and a July 2026 adversarial audit
re-verified the current-phase numbers against those files.

### The famous failure: our retracted R² = 0.9994

Here is why healthy suspicion of R² is *warranted*, and why this project is a good case study. Early on, the
campaign wanted a criterion: "explicit abstraction is redundant when value is already linearly decodable from
the latent." The measurement came back \(R^2 = 0.9994\) — seemingly decisive. It was later **retracted by our
own postmortem** ([the R²-criterion postmortem](../2026-06-17-tdmpc-glass-r2-criterion-postmortem)), for two
reasons that generalize to any R² use:

1. **Saturation by construction.** The probe decoded the *value network's own output* \(V(z)\) from \(z\). But
   \(V\) is itself a small network reading \(z\) — nearly linear in its input — so the probe reads
   \(R^2 \approx 0.98\)+ for *any* checkpoint, including a collapsed policy with return 1. A gauge that reads
   "high" no matter what certifies nothing.
2. **Variance confounding.** Decoding the *Monte-Carlo return-to-go* instead ties R² to the return
   *variance* in the buffer: a collapsed policy (all returns similar, low variance) can score \(R^2=0.96\)
   while a strong policy scores \(0.09\). The number **anti-correlates** with what you wanted to measure.

The lesson is not "R² is fake"; it's that **an instrument must pass a discrimination test before its readings
mean anything** — it must produce different readings in situations you know are different. Our probe didn't,
so the reading was retracted, and the paper's redundancy conclusion now rests on 16 direct experimental nulls
instead. (A retrospective audit also found two older posts still citing the retracted number as evidence — they
now carry correction banners.)

### Where our R² probes *do* work

Used comparatively, with controls, the same tool earned its keep repeatedly:

- **Thread D (JEPA anti-collapse):** held-out ridge probes for geometry *and* value, always alongside a
  **raw-observation baseline** (does the latent beat just probing the raw state?) and eff-rank diagnostics.
  This is what established that a pure self-predictive JEPA does not collapse — the `none` arm had the *best*
  readouts — and that uniformity-style regularizers raise rank while *hurting* decodability.
- **The entity-graph OOD sweep (Paper A):** the interesting quantity was never one R² but the *difference
  across architectures and object counts* — re-fit R² stays ~0.86 out to 3.5× the training object count for
  monolith and entity models alike (no OOD cliff, no entity advantage), while frozen-probe *transfer* R²
  separates them sharply. Comparative readings survive the instrument caveat; certifications don't.

**Verdict on R²: real method; real code; one prominent invalid use, caught and retracted by the project's own
methodology; the surviving uses are comparative, controlled, and load-bearing.**

## 2. Pareto-frontier analysis

### What it is

When you compare methods on **more than one objective at once** — say return, env-steps to competence, and
wall-clock time — there is usually no single "best." Method A **Pareto-dominates** method B if A is at least as
good on *every* objective and strictly better on at least one. The **Pareto frontier** is the set of methods
nobody dominates: each represents a genuine trade-off, not a deficiency.

This is a decades-old concept from economics and multi-objective optimization; in ML benchmarking it's the
honest way to compare algorithms that win on different axes — and a guard against cherry-picking the one axis
your method happens to win.

### How this project used it

The central question "is PPO beatable?" is exactly a multi-objective question, and a single-axis answer
misleads in either direction. The campaign's Pareto experiment put TD-MPC2 (and a temporal-abstraction "jumpy"
variant, and a shrunken "efficient" variant) against PPO on HopperHop with three axes: final return,
env-steps-to-competence, and wall-clock. The frontier came out split and stayed split:

- **TD-MPC2 wins return and sample-efficiency** by enormous margins (PPO never learns to hop at these budgets);
- **PPO wins raw wall-clock throughput** ~10–30× (massively parallel simulation + cheap updates);
- **no TD-MPC2 variant Pareto-dominates PPO**, and no abstraction variant dominates vanilla TD-MPC2 —
  adding "jumpy" temporal abstraction was worse on *all three* axes (so it's simply off the frontier).

That "nobody dominates" conclusion did real work: it killed the temptation to claim "we beat PPO" from the
sample-efficiency axis alone, and it predicted the shape of everything that followed — this week's controls
found the same structure yet again (PPO is categorically walled on HopperHop; SAC escapes the wall but is ~5×
less sample-efficient than TD-MPC2 on the identical environment; each method owns an axis).

**Verdict on Pareto analysis: real, standard, and in this project arguably the *most* protective tool — it's
the reason several would-be over-claims died before publication.**

## 3. So are they hallucinations?

No — with one precise caveat. Both tools are textbook methods, the analyses ran as real code, and the numbers
in the current posts trace to files on disk (re-verified by an independent adversarial audit in July 2026,
which found zero fabricated numbers in the current phase). The genuine failure mode this project hit was not
hallucination but **instrument invalidity**: a real regression, really computed, whose reading could not mean
what it was claimed to mean. The defenses that worked, and that we now apply by default:

1. **Discrimination test first** — an instrument must read differently in known-different situations.
2. **Controls and baselines always** — raw-obs baselines for probes, matched budgets for comparisons,
   random-partition/shuffled-label nulls for structure claims.
3. **Comparative claims over certifications** — "A > B under the same probe" survives instrument doubt;
   "A is certified sufficient" usually doesn't.
4. **Multi-axis (Pareto) reporting** — never let a method claim victory on its favorite axis alone.

*Every number in this post is documented in the campaign ledger (`bet2_null_results.md`) and the linked posts;
the R² retraction story is told in full in the
[R²-criterion postmortem](../2026-06-17-tdmpc-glass-r2-criterion-postmortem) and the Paper A draft.*
