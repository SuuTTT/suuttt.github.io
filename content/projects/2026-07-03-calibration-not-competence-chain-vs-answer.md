---
title: "Calibration, Not Competence: When Chain-of-Thought Uncertainty Beats Answer Uncertainty"
date: 2026-07-03
description: "The LLM-uncertainty thread from the three-line review resolves. I set out to show a 'competence crossover' — score the reasoning chain for weak models, the final answer for strong ones. That was wrong twice: the crossover was a model-family confound, and the obvious mechanism (a degenerate answer graph) was falsified too. What survives is calibration: chain-level structural entropy beats answer-level SE precisely when a model is overconfident — peaked answers that don't track correctness. Across 13 validity-filtered cells the correlation with overconfidence is 0.83, and it's task-robust within the one severely-overconfident model (gemma-3, on both GSM8K and SVAMP). Honest scope: it's anchored by a single model family, and one apparent second case was an answer-parsing artifact I had to exclude."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["LLM-uncertainty", "structural-entropy", "chain-of-thought", "calibration", "negative-results", "SeSE"]
---

{{< katex >}}

> This closes the LLM-uncertainty thread from the
> [three-line self-review](https://suuttt.github.io/projects/2026-07-02-adversarial-self-review-three-se-lines/).
> It's a story of two falsified hypotheses and one that survived — and of catching a would-be-fabrication
> mid-stream. Every number is read from a results artifact.

**SeSE** scores an LLM's uncertainty by sampling responses, building a semantic (entailment) graph, and taking
the structural entropy of its hierarchical abstraction. For chain-of-thought reasoning there are two graphs:
over the final **answers**, or over the full **chains**. Which is the better error detector?

## Hypothesis 1 — competence. Falsified.

The intuitive story: weak models emit uninformative bare answers, so score the chain; strong models produce
clean answers, so score the answer. Anchored on gemma-3-12b (weak, chain wins) vs Qwen-7B (strong, answer
wins), it looked right. Then I ran the full Qwen2.5 size sweep:

| model | acc | winner |
|---|---|---|
| Qwen-0.5B | .40 | answer |
| Qwen-1.5B | .61 | answer |
| Qwen-3B | .84 | answer |
| Qwen-7B | .79 | answer |
| **gemma-3-12b** | **.41** | **chain** |

Answer-SE wins across the *entire* Qwen family — including Qwen-0.5B, which is *the same accuracy as gemma*.
Same competence, opposite winner. The "crossover" had compared gemma to Qwen and read a **family** difference
as a competence effect.

## Hypothesis 2 — degenerate answer graph. Also falsified.

Next guess: gemma emits near-identical final answers, so its answer graph is flat and answer-SE loses signal.
Testable — count distinct sampled answers per question. Falsified: gemma's answer diversity is
*indistinguishable from a strong, well-calibrated model's*. Its answer graph isn't degenerate.

## What survives — calibration.

The tell is in the pairing: gemma and the strong models have similar answer peakedness, but gemma is **acc .41**
and confidently wrong. Its answers are peaked whether or not they're correct, so answer-spread carries no error
signal — while its *reasoning* still varies, so the chain does. Define **overconfidence** = peakedness $-$
accuracy. Across 13 validity-filtered model$\times$dataset cells:

$$\mathrm{corr}\big(\text{overconfidence},\ \text{chain}-\text{answer AUROC gap}\big) = 0.83.$$

Chain-level SE beats answer-level SE *exactly* when a model is overconfident. And it's **task-robust within
the overconfident model**: gemma-3 stays severely overconfident and chain-winning on **both** GSM8K
(overconf. $\approx +0.40$, gap $+0.12$ to $+0.18$) and SVAMP (overconf. $+0.46$, gap $+0.08$), while every
calibrated model (Qwen, Phi) favors the answer on every dataset.

## The catch — a parsing artifact I had to exclude.

When I widened the harvest, a *second* overconfident, chain-winning model appeared: Mistral-7B, apparent
accuracy 0.05, overconfidence $+0.69$ — and the correlation jumped. That's exactly where it's tempting to
declare "gemma isn't a singleton." It isn't real: **73% of Mistral's extracted answers were empty** — an
answer-format mismatch, not incompetence — inflating both peakedness and apparent miscalibration. I added a
validity filter (drop any cell with >25% empty answers) and excluded it. Keeping it would have fabricated
support through a broken metric.

## Honest scope.

The practical takeaway is a **calibration-aware selector**: route to chain-level SE when a model's answer
confidence is miscalibrated, to answer-level SE otherwise. The limitation is real and I'll state it plainly:
among the accessible models, **only one family (gemma-3) reaches the severe-overconfidence regime** where the
chain wins. The other natural candidates for that regime (gemma-2, Llama-3.2) are gated and couldn't be
evaluated, and the one non-gated candidate that appeared to qualify was the Mistral parsing artifact. So the
correlation, while strong and now task-robust *within* gemma, is anchored by a single model family;
cross-family confirmation is the key open question, not a footnote.

That's the honest shape of it: a clean mechanism, a strong correlation, a genuine within-model task-robustness
result — and a scope boundary I'm not going to paper over. Draft write-up:
[sese-calibration-paper](https://github.com/SuuTTT/sese-calibration-paper).
