---
title: "DM-SISA: A Live Dashboard — the RL×Structural-Entropy Periodic Table and a Systematic Hunt for a Beat on SISA"
date: 2026-07-04
description: "We built a differentiable, multi-level structural-entropy state abstraction for pixel RL that matches SISA at ~16,000× less abstraction cost. Matching isn't beating, so this is the living dashboard where we map what RL+SE has ever tried (the periodic table), enumerate every lever that could turn parity into a significant win, and post results as the ~150-run campaign completes. Updated as experiments land — including the honest negatives."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["reinforcement-learning", "structural-entropy", "SISA", "DM-SISA", "dashboard", "negative-results"]
---

{{< katex >}}

> A living dashboard, updated as runs complete. Code: [dm-sisa](https://github.com/SuuTTT/dm-sisa).
> Paper: [dmsisa-paper](https://github.com/SuuTTT/dmsisa-paper). Every number is from a results artifact;
> parity and losses are reported as carefully as wins.

## The setup in one paragraph

**SISA** (Xianghua Zeng et al.) abstracts an RL agent's states by minimizing structural entropy over a
state-transition graph and using the community structure to shape a pixel encoder. In its implementation
the abstraction is **flat** (a hard-coded height-2 encoding tree), **discrete** (a greedy CPU tree), and
so expensive it fires on only **1-in-100** updates. **DM-SISA** replaces it at the same seam with a
**differentiable, GPU-native, multi-level** soft-SE module that fires **every** update at
**~16,000× less cost** (1.3 ms vs 21.4 s per abstraction call), validated exact against the discrete
objective (\\(|\Delta|=1.15\times10^{-7}\\)).

**Core result (n=5): DM-SISA matches SISA on return** — walker \\(676{\pm}39\\) vs \\(725{\pm}107\\)
(Mann–Whitney \\(p{=}0.55\\)), cheetah \\(349{\pm}101\\) vs \\(393{\pm}115\\) (\\(p{=}0.73\\)). Parity,
decisive efficiency win. **Matching isn't beating** — so the rest of this page is the systematic search
for a lever that turns parity into a significant return win.

## 1. The RL × Structural-Entropy periodic table

What has ever been tried at the RL+SE intersection (the SIDM family), and where DM-SISA moves:

| method | RL problem | tree | computation | injection | fires |
|---|---|---|---|---|---|
| SI2E | exploration | flat | discrete | reward shaping | per-update |
| **SISA** | state abstraction | **flat** | **discrete** | **encoder aux-loss** | **1/100** |
| SISL | skill learning | flat | discrete | skill feature | per-batch |
| SIRD | role discovery (MARL) | flat | discrete | role clusters | per-update |
| **DM-SISA (ours)** | state abstraction | **multi-level** | **differentiable** | aux-loss **or policy-input** | **every** |

Every prior RL+SE method is flat, discrete, and injects only into an auxiliary objective. DM-SISA is the
first to move the *tree*, *computation*, and *injection* axes — the last of which (feeding the abstraction
into the policy) is **impossible for a discrete tree**. Open cells: differentiable versions of
SI2E/SISL/SIRD; policy-injection anywhere; SE depth in RL.

## 2. The levers — systematic sweep for a beat

Baseline to beat: SISA (walker 725±107, cheetah 393±115). Current best (λ=0): 676 / 349 — parity.
Target: **mean > SISA, \\(p<0.05\\)** at n=5.

| # | lever | idea | status | result (walker / cheetah, n) |
|---|---|---|---|---|
| L1 | λ=0 (no SE penalty) | remove over-regularization | ✅ | 676±39 / 349±101 (n5) — parity |
| L2 | λ=0.01 (default) | shows over-regularization | ✅ | 562±41 (n2) — **−22%** |
| L3 | λ=0.003 | small penalty may help | 🔄 | 735.8 (n1) — promising, need n5 |
| L4 | λ∈{0.001,0.002,0.005} | bracket the optimum | 🔄 | pending |
| L5 | **policy-input injection** | give the policy the abstraction (SISA can't) | 🔄 running | built+validated, trains clean; n accumulating |
| L6 | multi-level vs flat | depth helps (graph result) | 🔄 | pending |
| L7 | deeper (3-level) | more levels | ⏳ | — |
| L8 | community count k1/k2 | richer partition | ⏳ | — |
| L9 | aux-task coefficients | tune head weights | ⏳ | — |
| L10 | distractor / hard regime | abstraction matters most in noise | ⏳ | — |

**Priority:** L3/L4 (cheapest; the only above-SISA hint) → **L5 policy-injection** (strongest *novel*
lever) → L6/L7 depth (does the graph-learning "multi-level is the lever" transfer to RL return?) → tuning.

## 3. The mechanism we already have (the honest strength)

Even at return-parity, the **λ_SE over-regularization curve** is a real result: the every-update SE
auxiliary loss, at \\(\lambda{=}0.01\\), *costs* \\(\sim22\%\\) return (562 vs 676); removing it recovers
parity. SISA avoids this only *by accident* — its slow discrete tree fires 1-in-100, which acts as an
implicit regularizer. This is the paper's load-bearing scientific finding regardless of whether a return
beat materializes.

## 4. Honest stop condition

If none of L3–L10 gives \\(p<0.05,\ \text{mean}>\text{SISA}\\) at n=5, DM-SISA is a *return-parity,
16,000×-cheaper* method → an efficiency + mechanism paper, not a "beats SISA" paper. We will say so. This
dashboard updates as the ~150-run campaign (3× 4-GPU boxes) completes — wins **and** the negatives.

*Last updated: 2026-07-04. Live status in the [levers doc](https://github.com/SuuTTT/dm-sisa/blob/main/docs/LEVERS.md).*
