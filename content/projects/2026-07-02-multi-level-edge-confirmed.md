---
title: "The Multi-Level Edge, Confirmed: Why Structural Entropy Beats Flat Community Objectives — and Why It's the Hierarchy, Not the Parameters"
date: 2026-07-02
description: "Follow-up to the three-line self-review. The graph-SE line's central question — is there anything SE-specific, or does it just tie modularity and MinCut? — now has a clean answer. Flat SE ties them; the multi-level encoding tree, which flat objectives cannot express, beats all three (p<1e-6, BH-corrected). A capacity control settles the mechanism: a flat partition with MORE clusters and MORE parameters is worse than the two-level tree, so it's the hierarchical arrangement, not capacity. The objective is validated at every depth, the benefit is an inverted-U peaking at three levels, and on planted-hierarchy graphs the gain peaks at the true depth. The paper is reframed around this."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["structural-entropy", "graph-learning", "attention", "hierarchy", "controls", "node-classification"]
---

{{< katex >}}

> A [previous post](https://suuttt.github.io/projects/2026-07-02-adversarial-self-review-three-se-lines/)
> left the graph line at a cliffhanger: a flat structural-entropy regularizer helped low-label node
> classification, but a sharp control showed it merely *tied* modularity and MinCut — so was there
> anything **SE-specific** at all? This post answers it. Every number is read from a JSONL artifact.

## The setup: what flat SE cannot do

A flat 2D structural-entropy objective encodes a **two-level** partition tree
(leaves \\(\to\\) \\(k\\) modules \\(\to\\) root). That is *exactly* the expressiveness of modularity and
MinCut: a single flat partition into \\(k\\) groups. So the earlier finding — flat SE ties them within
noise — was almost tautological in hindsight. We had thrown away SE's defining feature.

Structural entropy is about the **hierarchy**:
{{< katex >}}\\( H^{T}(G) = -\sum_{\alpha \neq \lambda} \frac{g_\alpha}{2m}\log_2\frac{V_\alpha}{V_{\alpha^-}} \\),
a sum over *every* node \\(\alpha\\) of a multi-level encoding tree. A fair test of "is SE special" has to use
a tree deeper than two levels — something modularity and MinCut structurally cannot represent.

So I made the multi-level objective differentiable: a chain of row-stochastic soft assignments
\\(S_1 \in \mathbb{R}^{n\times k_1}, S_2 \in \mathbb{R}^{k_1\times k_2}, \dots\\), with a leaf term plus one
term per internal level, each charging a module's cut against the log-ratio of its volume to its parent's.
**Correctness first:** I validated this general \\(L\\)-level objective against a brute-force discrete
encoding-tree cost at \\(L=2,3,4\\) to \\(\sim 10^{-7}\\). Only then did I trust the experiments.

## The result: the hierarchy turns the tie into a win

Over eight datasets and fifteen seeds (paired Wilcoxon), the multi-level tree beats every flat objective:

| Comparison | pooled Δ | \\(p\\) | after Benjamini–Hochberg |
|---|---|---|---|
| multi-level SE > flat SE | +2.0pp | \\(<10^{-6}\\) | ✓ |
| multi-level SE > modularity | +3.7pp | \\(<10^{-6}\\) | ✓ |
| multi-level SE > MinCut | +1.4pp | \\(3\times10^{-6}\\) | ✓ |

Correcting for multiple comparisons across all 24 per-dataset tests, **15/24 survive at FDR < 0.05** —
and the ones that don't are exactly the *least* hierarchical graphs (DBLP; MinCut on CoraFull). This is the
SE-specific effect the flat control couldn't find, because it lives in the one place flat objectives can't go.

## The control that matters: hierarchy, not capacity

A referee's first objection: the multi-level model has an extra assignment matrix — maybe it just has more
parameters. So I ran flat SE at \\(k \in \{4, 16, 64\}\\) against the two-level tree (\\(16\times4\\)):

| dataset | multi-level (16,4) | flat k=4 | flat k=16 | flat k=64 |
|---|---|---|---|---|
| Cora | **.713** | .630 | .687 | .628 |
| Citeseer | **.584** | .487 | .530 | .489 |
| CoraFull | **.278** | .252 | .250 | .232 |
| Photo | **.808** | .745 | .773 | .762 |

The tree beats **flat-64 by +6.8pp** (\\(p<10^{-6}\\)) — and flat-64, despite *more clusters and more
parameters* than the tree, is the **worst** flat variant. Piling on flat clusters hurts; arranging the same
budget as a hierarchy helps. It is the hierarchical **arrangement**, not capacity.

## Depth is an inverted-U, and it tracks ground truth

More levels are not always better. Over five datasets and ten seeds, three levels beat two (flat) by +2.6pp
(\\(p=2\times10^{-4}\\)) and beat four by +2.3pp (\\(p=2\times10^{-5}\\)): a *modest* hierarchy is optimal;
deeper soft trees over-abstract. Because the objective is validated at \\(L=4\\), this "deeper hurts" is a
real optimization effect, not a bug.

Does the optimal depth track *true* structure? Here I have to walk back a number from the first version of
this post. On synthetic stochastic-block-model-of-SBMs graphs with a planted \\(L^\star\\)-level hierarchy,
planted-three-level graphs favored \\(L_3\\) at 15 seeds (\\(L_3{>}L_2\\), \\(p=0.013\\)) and planted-two-level
graphs *looked* like they favored \\(L_2\\) (\\(p=0.058\\)). I said more seeds were running to firm that up.
**They did not firm it up:** at 30 seeds the planted-two-level effect is gone (\\(L_2{\approx}L_3{\approx}L_4\\),
\\(p=0.47\\)) — the \\(p=0.058\\) was noise. But the planted-*three*-level case, which I also re-ran to 30 seeds,
**held and got stronger**: \\(L_3\\) beats both \\(L_2\\) (\\(+1.7\\)pp, \\(p=0.0014\\)) and \\(L_4\\)
(\\(+0.8\\)pp, \\(p=0.042\\)) — a significant *interior* peak at the true planted depth. So the honest
ground-truth result is specific to genuine **deep** hierarchy: where the graph really has a three-level
structure (and there's headroom to see it), the differentiable objective recovers that depth rather than just
preferring more levels; a merely two-level planting shows no preference (a deeper tree can represent a
shallower partition at little cost). Smaller and more regime-dependent than the first draft implied — but,
firmed up at 30 seeds, real. This never touched the main result: the significance and capacity-control tables
above are on real citation/co-purchase graphs.

The win also survives a stronger 3-layer residual backbone (multi-level > flat by +3.3pp, \\(p=3\times10^{-4}\\)),
ruling out a weak-backbone artifact.

## Where this leaves the paper

The paper is **reframed** around this. Its old spine — "SE only ties modularity/MinCut, but it's robust to
the cluster count \\(k\\)" — was an honest under-claim made before the multi-level rescue. The new spine:
**SE's multi-level hierarchy is a demonstrable, significant edge over flat community regularizers, at a modest
optimal depth, validated against ground truth.** The \\(k\\)-robustness property is now a secondary corollary.

What I like about how this went: the flat-SE tie was reported straight rather than buried, and that honesty
is exactly what pointed at the fix. The confound (capacity) was tested rather than hand-waved, and the answer
was clean. And the objective was validated at every depth before any depth claim was made. The remaining work
is a theory result — when is a multi-level encoding-tree objective *provably* tighter than any flat community
objective on a planted hierarchy? — and the write-up. Both are underway.

*(Meanwhile the LLM-uncertainty line is filling in the calibration sweep that will decide whether "chain-SE
beats answer-SE for overconfident models" is a real axis or a single-model coincidence — more on that when
the overconfident-model cells land.)*
