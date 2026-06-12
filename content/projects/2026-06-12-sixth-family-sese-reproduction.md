---
title: "The Sixth Family: Reproducing an LLM Uncertainty Method, and What the Release Actually Computes"
date: 2026-06-12
description: "DRAFT — the SE survey's LLM-track reproduction: the pipeline runs end-to-end after 12 patches, the baselines behave, and the headline measure tells a release-vs-paper story worth its own post."
layout: "post"
showTableOfContents: true
draft: true
tags: ["research", "structural-entropy", "survey", "llm", "reproducibility", "uncertainty"]
---

<!-- DRAFT: held for author review before publishing — the subject paper is
     ours (SeSE, UAI'26), and the finding concerns its public code release.
     All numbers below trace to JSONs in the artifact repo
     (benchmark/suite/results/llm/). -->

## The last empty row

Our reproduction scorecard had one family we'd never been able to run: LLM
uncertainty quantification, gated on a 24 GB GPU. A rented RTX 3090 closed
that gap, and we ran the official SeSE release (UAI'26 Oral) under its
published protocol: Llama-3.1-8B-Instruct, 400 TriviaQA questions, 10 sampled
+ 1 greedy answers each, DeBERTa-MNLI semantic graphs, GPT-5-mini as the
correctness judge.

## What reproduced

The pipeline itself, after 12 documented patches (a syntax error in a shipped
file, a missing function, several transformers-5.x API breaks). And the
baselines computed by the same pass behaved exactly as the literature says
they should: p_true AUROC 0.856, length-normalized entropy 0.747, semantic
entropy 0.669.

## What didn't

The release's own structural-entropy measure scored **0.445** — chance level
(0.555 even after correcting the sign convention; the released routine
returns the negative of the entropy). Its correlation with the number of
semantic clusters is 0.23, where semantic entropy's is 0.96. The released
measure is nearly signal-free on this run.

We checked our own hands first: the NLI patch is exact on canonical probes
(entailment 0.999 / contradiction 0.999), the GPT enhancement step ran on all
400 questions, the labels are sane (greedy accuracy 0.79).

## The diagnosis

The paper specifies a *directed* semantic graph with a PageRank stationary
distribution and a height-K encoding tree built by merge/combine operators.
The released `compute_se` evaluates an *undirected depth-2* tree over a
cluster-blocked entailment graph and returns its negative. These are
different algorithms. Our measurement characterizes the release, not the
published method — the same class of finding as the "computationally
infeasible" method that ran 10k nodes in 74 seconds, just pointed at a much
newer paper.

The cross-check settles it: re-scoring the *same* generations with selib's
independently validated optimizer (min 2D SE of the plain pairwise entailment
graph, one fixed orientation derived from the construction: low minimized SE =
fragmented semantic space = uncertain) **beats semantic entropy on both
datasets** — TriviaQA 0.694 vs 0.669, SVAMP 0.774 vs 0.657 — and is sign-stable
where the release flips orientation between datasets (0.445 vs 0.660 as-emitted).
The idea works. The released computation is what loses the signal.

## The meta-point

Code decay in this field is usually framed as a years-scale problem
(PyTorch 0.4 repos, dead dataset mirrors). This case suggests it can be a
*months*-scale problem: a 2026 oral-track release whose shipped measure does
not implement the published algorithm. Reproduction-first surveying exists
precisely to catch this while it is still cheap to fix — the repo is one
commit away from matching its paper.
