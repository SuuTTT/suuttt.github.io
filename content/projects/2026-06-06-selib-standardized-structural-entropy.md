---
title: "selib: a Standardized Library for Structural Entropy (and Optimizers that Beat the Originals)"
date: 2026-06-06
description: "From a reproduction campaign across the structural-entropy literature to selib — one API for SE, three validated optimizers (flat, hierarchical, attributed) that reach lower structural entropy than the published implementations on their own objectives."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["research", "structural-entropy", "graph-clustering", "library", "benchmark"]
---

{{< katex >}}

**TL;DR.** Every structural-entropy (SE) paper ships its own data loader, its own
metric, and its own (often unmaintained) code — so *"does SE actually help?"* is
hard to answer fairly. [`selib`](https://github.com/SuuTTT/selib) fixes the
interface: one `Method.fit_predict(G, k, seed)` contract, shared datasets and
metrics, and a one-call benchmark. On top of it we built three SE optimizers that,
on **identical graphs**, reach **lower structural entropy than the published
implementations on their own objectives**. Everything is on a live
[benchmark page](https://suuttt.github.io/selib/), and every number traces to a
results JSON a GPU produced.

## Why a library

This grew out of an empirical re-appraisal of the SE literature (the
[survey & benchmark](https://github.com/SuuTTT/structural-entropy-benchmark)). The
recurring pain: comparing methods meant wrangling a Java jar here, a C++ binary
there, a framework-pinned PyTorch repo somewhere else — each with a different I/O
format and a different notion of "the score." `selib` standardizes the *interface*
so an SE method and a Louvain baseline are scored exactly the same way.

The 2D structural entropy of a graph under a partition \\(P\\) is

$$\mathcal{H}^{2}(P) = -\sum_{j}\frac{g_j}{2m}\log_2\frac{V_j}{2m}-\sum_{v}\frac{d_v}{2m}\log_2\frac{d_v}{V_{c(v)}},$$

with \\(V_j\\) the volume of community \\(j\\), \\(g_j\\) its cut, \\(2m\\) the total
degree. Its full version is defined over an **encoding tree** of arbitrary height.
`selib` ships exact evaluators for both, validated so that a 2-level tree's tree
entropy equals \\(\mathcal{H}^2\\) *exactly*.

## Three optimizers

**`se_louvain` — flat 2D-SE.** The shipped `se_agglomerative` only ever *merges*,
so an early bad merge can never be undone. `se_louvain` does what Louvain does for
modularity, but for \\(\mathcal{H}^2\\): local node moves (with an \\(O(\deg)\\)
exact delta) + community aggregation + multistart. It hits the brute-force optimum
on small graphs, and at free \\(k\\) it reaches the **lowest 2D-SE on every
benchmark graph** — below Louvain/Leiden/Infomap and below the original **CoDeSEG**
(C++) and **deDoc** (Java) run from their own code. A nice honest twist: the
partition that *minimizes* SE is more granular than the planted one, so *lower SE
does not imply better label recovery* — direct evidence for SE's regime-dependence.

**`se_hier` — hierarchical encoding tree.** Structural entropy is really about
trees, and \\(\mathcal{H}^{T}\\) is **not monotone in depth** (bad intermediate cuts
raise it), so the right move is to warm-start from several good trees — the binary
SE dendrogram, a recursive `se_louvain`, and Paris — and refine the best with
exact-guarded collapse/relocation moves that accept only verified improvements. The
result is therefore \\(\le\\) each warm start *by construction*, and in practice it
reaches the **lowest \\(\mathcal{H}^{T}\\) on every graph**, below the original
**BBM** and **HCSE** code and Paris — and it now scales to \\(n=1000\\) thanks to a
binary-lifting-LCA evaluator.

**`se_gnn` — attributed graphs.** Ported from my earlier `glass-jax` prototype: a
small GCN trained end-to-end to minimize a *differentiable* soft \\(\mathcal{H}^2\\),
with a **balanced (Sinkhorn) assignment head** that prevents the cluster collapse a
plain softmax suffers under pure SE minimization. On Cora it reaches NMI 0.487 /
ARI 0.387 / ACC 0.592 at \\(k=7\\) — beating every topology-only method and matching
LSENet, with a far smaller model. (DeSE still leads on NMI; closing that gap looks
architectural, not a matter of tuning.)

## Comparing with the actual published code

The point of standardization is fair comparison, so the published methods run from
*their own implementations* on the same graphs: CoDeSEG (re-compiled C++), deDoc
(original jar), and HCSE + BBM (from `Hardict/HCSE`), each scored on `selib`'s exact
objective. deDoc reproduces the survey campaign's numbers byte-for-byte — including
the way it degenerates to near-singletons on sparse graphs, a reminder that it was
built for dense Hi-C matrices. (Its successor deDoc2 only forms *contiguous*
genomic domains, so it isn't a general community detector at all.)

## Discipline

This project has a strict rule: every reported number traces to a results JSON a
GPU produced; nothing is typed by hand. The manuscript's `selib` tables are even
auto-generated from those JSONs. The optimizers ship with self-tests that check the
soft/tree objectives against the canonical formulas and against brute-force optima.

## Links

- Library: [github.com/SuuTTT/selib](https://github.com/SuuTTT/selib)
- Benchmark page (live, every table from a JSON): [suuttt.github.io/selib](https://suuttt.github.io/selib/)
- Reproduction benchmark: [structural-entropy-benchmark](https://github.com/SuuTTT/structural-entropy-benchmark)
- Survey paper: [structural-entropy-survey-paper](https://github.com/SuuTTT/structural-entropy-survey-paper)

The bigger story — *where SE actually helps and where it doesn't* — is the survey's;
`selib` is the constructive half: the gap between SE's theory and its published
tooling turns out to be closable.
