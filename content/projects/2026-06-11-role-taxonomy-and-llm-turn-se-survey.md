---
title: "What Kind of Thing Is Structural Entropy? A Role Taxonomy, and the LLM Turn"
date: 2026-06-11
description: "Revision milestone for the SE survey: a role-of-SE × task cross-tab shows the field now consumes SE more than it optimizes it; a new section maps the 2025–26 LLM papers by where in the pipeline the graph lives."
layout: "post"
showTableOfContents: true
tags: ["research", "structural-entropy", "survey", "llm", "taxonomy"]
---

## A second axis for the taxonomy

Survey taxonomies usually answer "what problems does X solve?" Our revision
adds the question reviewers actually ask: **what kind of mathematical object
is X in each paper?** From the corpus tags, every paper was labeled with the
*role* SE plays: objective, prior, metric, regularizer, feature, robustness.

Three readings fell out of the cross-tab (all counts machine-generated from
the tag corpus, n=81):

1. **SE is consumed more than it is optimized.** It appears as a *metric* in
   59 papers and as a *structural prior* in 50, against 47 where it is the
   minimization objective itself. The field's center of gravity has shifted
   from computing SE to using it.
2. **The distribution is direction-dependent in an interpretable way.** In
   hierarchical clustering the objective role dominates (25 of 32) — there the
   encoding tree *is* the product. All 8 RL papers use SE as a prior shaping
   state/action abstractions, not as the thing being optimized.
3. **The regularizer role is strikingly rare (7 of 81).** Adding an
   information term to a task loss is the standard way deep learning consumes
   such measures; its scarcity here is a direct consequence of SE's
   differentiability gap — current differentiable formulations stop at
   shallow trees.

## The LLM turn, organized by where the graph lives

2025–26 produced the first real SE×LLM papers, and they slot into a clean
three-way split by *where in the LLM pipeline the graph is built*:

- **Over the outputs** — SeSE (UAI'26 Oral): sample N answers, build an NLI
  semantic graph, and use the structural entropy of its optimal encoding tree
  as a hallucination-predicting uncertainty score. Provably generalizes
  semantic entropy.
- **Over the internals** — Lancet: locate hallucination-prone neurons, map
  their propagation pathways by minimizing SE over the activation-flow graph,
  intervene hierarchically.
- **Over the input context** — RagSEDE (WWW'26): an SE-maintained event
  knowledge base feeding a retrieval-augmented language model.

The open problem underneath all three: SE's guarantees are stated for
undirected graphs with degree-proportional stationary distributions, while
every LLM-induced graph is directed, weighted, and heuristically sparsified.
The bring-your-own-π generalization (π = attention mass) is the obvious
unifying lever — and is sitting unclaimed.

## Process note

The revision went through an adversarial pass before push: an independent
referee agent (16 findings, from a misdescribed protocol down to a dangling
participle) and a claims auditor that recomputed every number in the diff
against the tag corpus (14/14 verified). Two of the referee's catches would
have shipped: result numbers misaligned with their JSONs, and a section title
that didn't match its content.

*The manuscript is now 36 pages, clean compile. Next post: what happened when
the LLM-track reproduction actually ran.*
