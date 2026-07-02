---
title: "Evaluation Infrastructure That Catches Its Own Lies: Lessons from a Mahjong AI Campaign"
date: 2026-07-02
description: "Our IJCAI-2026 Mahjong campaign killed ~10 of its own fake wins — and then discovered that one of its cleanest published-quality NULLS was fake too: a silent no-op that ran for weeks. The evaluation stack that caught both, the one metric that would have caught the no-op on day one, and nine portable best practices for competitive game-AI evaluation."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["mahjong", "game-ai", "evaluation", "reproducibility", "infrastructure", "best-practices", "botzone"]
---

{{< katex >}}

## The problem: your evaluation will lie to you

For the IJCAI-2026 Chinese Standard Mahjong competition we ran a months-long
campaign: imitation policies, claim-suppression knobs, offline and online RL,
value-guided 1-ply search, PIMC, architecture sweeps (GNN, transformer,
temporal/GRU), data augmentation, ensembles. Along the way the project produced
roughly **ten apparent "wins" that were not real** — small-sample flukes,
uncalibrated comparisons, one memorable "3.06 vs 2.5 baseline" that evaporated
under a proper gate.

The interesting part is not that noise happens. It is that the effect sizes that
matter in a mature competitive setting are **tiny** — ±0.01 placement points on
a 2.500 baseline — while a single Mahjong deal carries ±1.5 points of pure wall
luck. Any evaluation loop that is even slightly sloppy will hand you a
publishable-looking result every week. Our infra evolved, under fire, into
something that reliably catches its own lies. This post documents it.

And then — the week we wrote this — the stack caught its biggest catch yet: one
of our cleanest *negative* results turned out to be a silent no-op. More on that
below, because the failure mode is general and the fix is one metric.

## The measurement core

### 1. A calibration trap in every gate

Every evaluation gate is built so that a known input **must** produce a known
output *by construction*:

- Play the **same wall** four times, rotating the candidate through every seat,
  with the reference policy in the other three seats.
- Rank the four seats' scores per game; the candidate earns placement points
  \\(5 - \text{rank}\\) (4/3/2/1).
- With candidate = reference, the four rotations are the identical game, ranks
  sum to 10, and the gate reads **exactly 2.500** — not approximately, exactly,
  with standard deviation 0.0.

Any drift from 2.500 on the self-test is a harness bug, detected before it can
contaminate a verdict. A claimed win is then a CI-separated deviation from a
self-verifying constant. This one design choice retired an entire class of
"the baseline moved" bugs.

### 2. Pair everything

The 4-seat duplicate rotation cancels seat and wall luck within each game. This
is the only reason ±0.01 effects are resolvable at all: unpaired, the same power
would need ~100× the games.

The same trick later rescued our online-RL fine-tuning. Vanilla PPO on raw
per-deal placement was drowning: signal +0.01, noise ±1.5. Playing every rollout
wall **twice** — once with the learner in seat \\(s\\), once all-reference —
and using the difference as the advantage cancels the wall+seat luck term in
the gradient. Same idea, applied to the optimizer instead of the evaluator.

### 3. Block-level confidence intervals, and a hard bar

One gate run (2,000 games) is a point estimate. A claim requires:

- **N seed-disjoint blocks** (12–24 blocks × 2,000 games),
- a block-level Student-t 95% CI,
- **CI lower bound > 2.500** — the mean being above the line is not a win.

This bar killed the "3.06" fluke, killed a "+2.53 vs distill" fluke, and this
week correctly killed a promising ensemble whose first blocks read +0.015: by
block 24 the mean had collapsed to +0.003 and the CI straddled the line.
Regression to the mean is not a hypothetical.

### 4. No train/serve skew, by construction

The self-play simulator consumes the **same feature encoder** the deployed bot
uses, and validates wins/fans with the **official rule library** (the same one
the competition judge runs). Deployment parity is closed with a bit-exactness
check: our pure-NumPy deployment engine reproduces the PyTorch model with max
logit difference **0.00000** over held-out states. A gate win therefore
transfers to the platform by construction, not by hope.

## The catch of the campaign: a null that never ran

We trained a placement value model that was verifiably excellent at
*prediction* (last-place AUC 0.955). Could it *control*? We built a 1-ply
overlay: rerank the policy's top-K actions by
\\(\mathrm{logit}(a) - \lambda \cdot V(\text{state after } a)\\). The sweep came
back **null** across all \\(\lambda\\) — "good value prediction ≠ useful value
control." A tidy, publishable negative. It sat in the results table for weeks.

Then a fresh-eyes mechanism audit asked a boring question: *how often does the
overlay actually change anything?* Tracing the code:

- The feature encoder's `request2obs("Player p Play X")` returns **`None`**
  when the acting seat replays **its own** discard (there is no decision to
  make, so there is no observation to return — a perfectly reasonable
  contract that nobody had written down).
- The overlay's `try/except` fallback then scored every discard candidate with
  the **same current-state value**.
- A constant added to every candidate changes nothing: the "value-guided"
  reranking was the raw policy argmax on the discard axis — **~80% of all
  decisions**. Only chi/peng claims were genuinely reranked, and we had already
  shown claims are a near-zero-EV surface.

The experiment never tested its hypothesis. The null was a no-op wearing a
lab coat. A second confound compounded it: the value net had been trained only
on decision-point states, while the overlay queried post-action states it had
never seen — so even the engaged fraction was evaluated out-of-distribution.

**The fix is one metric.** The corrected harness renders true post-discard
observations from the encoder's internal state, retrains the value net on
decision + post-discard states, and reports a **mechanism-engagement rate**:
the fraction of reranked candidates that received a genuine 1-ply evaluation.
Old harness: ~0 on discards. Fixed harness: 0.94. If that number had existed on
day one, the no-op would have died on day one.

> **The lesson, stated once:** an intervention experiment must instrument
> *whether its mechanism fired*, not merely whether the code ran. Silent
> fallbacks turn no-ops into confident nulls — and a fake null is more
> dangerous than a fake win, because nothing ever re-tests it.

## The layer around the core (where our failures actually lived)

The measurement core above was right from mid-campaign. Every failure since
lived in the *glue*:

**Fail-open orchestration.** A sub-sweep crashed on a wrong checkpoint path;
the orchestrator kept going and wrote a results file *without that variant* — a
partial result indistinguishable from a completed one. Aggregators now assert
expected block/variant counts and write an explicit `integrity` field; a
crashed sweep exits loudly.

**No provenance.** Nothing stopped a stale checkpoint path from gating the
wrong model (that is exactly what the crash above almost did, silently).
Result JSONs now embed content hashes of the model files and the gate script
that produced them. Every number is mechanically traceable to the binary that
generated it.

**No preflight.** The invariants existed (calibration = 2.500, augmentation
fan-invariance, numpy/torch parity) but ran ad-hoc. They are now a bundled
`preflight.py` — including a regression test that `request2obs` on a self-Play
returns `None`, so the E8 trap is documented in executable form — run before
any gate campaign.

**Statistics hygiene at the campaign level.** With ~25 gated hypotheses, a 95%
CI hands you about one false separation for free. Deployed claims need multiple
independent legs (for us: two seeds separating independently, plus a two-step
CI-separated chain). And a few hundred live-platform A/B matches cannot resolve
±0.01 — treat real-field A/Bs as smoke tests for time-limit violations and
gross regressions, never as confirmation.

**Split leakage.** Validation accuracy computed on a random split over
*decision points* leaks: positions from one deal land on both sides. Ours were
mutually comparable but optimistic — and demonstrably did not order models by
strength (our highest-val-acc net merely tied on the placement gate). Split by
game, or disclose.

## The nine practices

1. **Calibration trap in every gate** — known input must produce a known
   constant, exactly, by construction.
2. **Pair everything** — duplicate rotation in eval; antithetic paired-wall
   baselines in RL.
3. **Block-level t-CIs; the win bar is the CI lower bound**, not the mean.
4. **Instrument mechanism engagement** — every overlay reports how often its
   interesting path fired.
5. **Ban silent fallbacks** — count them into the result artifact or fail loud.
6. **Fail loudly on partial results** — aggregators assert completeness.
7. **Embed provenance** — hash the models and the harness into every artifact.
8. **Kill train/serve skew at the source** — same encoder, official rules,
   bit-exact deployment parity.
9. **Account for the lever table** — multiple-comparison honesty; small live
   A/Bs are smoke tests.

None of these are novel individually. What we can report from the trenches is
their *joint* effect: a small solo-maintained campaign ran ~25 hypothesis gates
over months, deployed only two model upgrades, and — as far as the strongest
audit we can construct can tell — **both are real and nothing fake shipped.**
The stack caught ten fake wins and one fake null. The fake null is the one that
still keeps us honest: it survived every statistical defense because statistics
cannot detect an experiment that never ran. Only instrumentation can.

*The evaluation harnesses, preflight invariants, and gate results discussed
here are in the public repo
([SuuTTT/IJCAI-mahjong](https://github.com/SuuTTT/IJCAI-mahjong)); a companion
paper with the full lever table and protocols is in preparation.*
