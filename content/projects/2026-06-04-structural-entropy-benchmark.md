---
title: "Does Structural Entropy Actually Work? A Reproducible Benchmark Across Five Task Families"
date: 2026-06-04
description: "We re-ran the original code behind a decade of structural-entropy papers — community detection, graph pooling, structure learning, RL, hierarchy quality, and Hi-C — and measured, honestly, where SE helps and where it doesn't."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["structural-entropy", "benchmark", "graph-learning", "reinforcement-learning", "reproducibility"]
---

{{< katex >}}

> A turn from *surveying* structural entropy to *measuring* it: original code, a shared harness, and numbers that only count if a GPU produced them.

---

## TL;DR

Structural entropy (SE) extends Shannon entropy to the **hierarchical** structure of a graph: the cost of describing a random walk localized by an encoding tree. A decade of papers claim it helps everything from community detection to LLMs. We built a **reproducible benchmark** that runs the *original authors' code* across five task families and asks one question: **does SE actually work, and when?**

The honest answer is **regime-dependent**:

- ✅ **Supervised graph learning** — SE-guided pooling (SEP) reproduces on **7/7** graph-classification datasets; SE graph-structure-learning (SE-GSL) reproduces on node classification.
- ✅ **Hierarchy quality** — when you score the *dendrogram* (Dasgupta cost) instead of a flat cut, SE produces the **best hierarchy on 4/5 graphs**, beating Paris/Ward/average-linkage. This is the dimension flat ARI/NMI hides.
- ✅ **Bioinformatics** — the SE TAD caller (SuperTAD) reproduces on real Hi-C, SE's oldest real-world win.
- ⚠️ **Unsupervised community detection** — on LFR benchmarks SE methods **lose** to plain Louvain/spectral; deDoc collapses to singletons, CoDeSEG over-segments.
- ⚠️ **Reinforcement learning** — SE intrinsic motivation (SI2E) is **difficulty-graded**: it reliably solves easy sparse-reward tasks, is bimodal on medium ones, and fails on hard ones — *not* the uniform "100%" the paper reports.

Every number traces to a results JSON a GPU produced. Code, results, and an awesome-paper page are open-sourced.

---

## Why redo this?

The conference version of our SE survey did what most surveys do: it listed ~60 papers. Reviewers, fairly, called it *redundant* and said it "does not establish the **why** of structural entropy." So we did the un-survey-like thing: **cloned the papers' repos and ran them.**

SE of a graph \(G\) under a partition \(P\), the workhorse 2-D form:

$$\mathcal{H}^{2}(G;P) = -\sum_{c\in P}\sum_{v\in c}\frac{d_v}{2m}\log_2\frac{d_v}{V_c}\;-\;\sum_{c\in P}\frac{g_c}{2m}\log_2\frac{V_c}{2m}$$

where \(d_v\) is degree, \(V_c\) the volume (sum of degrees) of community \(c\), \(g_c\) its cut, and \(2m\) the total degree. Minimizing it trades off *within-community localization* against *cross-community description cost* — a hierarchical, flow-based prior, not a flat edge-count one.

## Method: run the original code, modernize minimally, trace every number

Three rules:

1. **Original code.** We run each paper's published repository, pinned by commit — not a reimplementation.
2. **Minimal modernization.** Every SE repo is framework bit-rotted (PyTorch 1.8 / PyG 2.0 / gym 0.21 / dgl 0.9). We patch only what's needed to run on current CUDA, documenting each change. SEP needed **2** PyG-API patches; SI2E needed **5** gym→gymnasium patches; SE-GSL needed **~8** dgl-2.x fixes. *The bit-rot itself is a finding.*
3. **No number without a JSON.** A GPU produced it, or it doesn't exist. (We even caught and corrected seven of our *own* artifacts mid-campaign — a signal-inverting SBM, a false "Infomap bug," a mis-tuned config — before they became "results.")

---

## The findings, family by family

### Community detection: SE loses on LFR

On the LFR mixing-parameter sweep (the canonical difficulty axis), classical baselines win:

| \(\mu\) | Louvain | Leiden | Infomap | Spectral | CoDeSEG (SE) | deDoc (SE) |
|---|---|---|---|---|---|---|
| 0.3 | 0.96 | 0.96 | 1.00 | 1.00 | 0.66 | ~0 |
| 0.5 | 0.49 | 0.41 | 0.00 | **0.61** | 0.10 | ~0 |

deDoc collapses to near-singletons on heterogeneous high-degree graphs; CoDeSEG over-segments (178 communities where ~24 exist). On attributed graphs (Cora/Citeseer/Photo) even feature-augmented SE clustering (DeSE, LSENet) **does not reliably beat topology-only Louvain** — and the two SE methods disagree sharply (LSENet *fails* on Citeseer, NMI 0.03, where DeSE gets 0.40).

We also corrected two factual errors in the literature: deDoc is **not** "infeasible for N>50" (the real Java jar does 10k nodes in 74 s, ≈ \(O(N^{1.4})\)), and DeSE's "no public code" claim is false — the repo exists and reproduces.

### Graph pooling: SEP reproduces cleanly

SE-guided hierarchical pooling reproduces the paper across **all seven** TU datasets (validation accuracy):

| PROTEINS | DD | NCI1 | MUTAG | IMDB-B | IMDB-M | COLLAB |
|---|---|---|---|---|---|---|
| 0.761 | 0.771 | 0.780 | 0.850 | 0.735 | 0.515 | 0.803 |

All within noise of the published numbers. SE-GSL likewise reproduces on node classification (Cora GCN 0.869 / GAT 0.880).

### Hierarchy quality: the result flat metrics hide

Here is the payoff for SE. Most SE evaluations report only a **flat cut** (ARI/NMI), which structurally undersells a method whose *output is a tree*. Score the full dendrogram with **Dasgupta cost** (lower is better) and SE shines:

| Graph | SE-agglom | average | Ward | Paris |
|---|---|---|---|---|
| SBM-Clean | **81,858** | 87,547 | 92,640 | 85,409 |
| SBM-6blk | **330,268** | 414,353 | 360,222 | 393,511 |
| LFR-μ0.1 | **113,694** | 158,299 | 176,201 | 126,919 |
| LFR-μ0.4 | **339,468** | 348,932 | 374,617 | 364,662 |

SE-agglomerative has the **lowest Dasgupta cost on 4 of 5 graphs**. The lesson: *SE may lose the flat-cut contest while winning the hierarchy contest* — and the hierarchy is what SE is actually for.

### Reinforcement learning: difficulty-graded, not uniform

SI2E uses value-conditional SE over a state-action graph as an intrinsic reward. Reproduced on four MiniGrid environments (success rate, multiple seeds):

| Environment | Difficulty | SI2E success |
|---|---|---|
| KeyCorridorS3R1 | easy | 0.91 / 0.90 / 0.91 — **reliable** |
| DoorKey-8×8 | medium | 0.00 / 0.94 / 0.25 / 0.07 / 0.94 — **bimodal** |
| RedBlueDoors-6×6 | medium | 0.89 / 0.06 / 0.76 — **bimodal** |
| KeyCorridorS3R2 | hard | 0.00 / 0.02 / 0.00 — **fails** |

The SE exploration bonus genuinely helps on easy/medium sparse-reward tasks, but is seed-unstable on medium and breaks down on hard ones. This is more useful — and more honest — than the paper's reported uniform 100%±0. (Caveat: we run a modernized gym/minigrid; the authors' exact frozen env was unavailable, which may contribute to the instability.)

### Bioinformatics: SE's oldest real win

SuperTAD (SE-minimization) detects 20 hierarchical TADs on real Hi-C (GM12878 & IMR90, chr19, 25 kb) — the original real-world application reproduces.

### One we skipped, honestly

SE-for-LLM uncertainty (SeSE) needs ≥24 GB GPUs plus paid API access — out of our budget. We document it as the highest-value *future* track rather than fake it.

---

## So, does structural entropy work?

**It depends — and that nuance is the contribution.** SE is a hierarchical, flow-based prior. It reliably helps when you're doing *supervised* graph learning or when you actually want a *hierarchy*; it's fragile or beaten by simple baselines in *unsupervised* community detection and in *hard* RL exploration. A faithful survey should say exactly that, with numbers — not praise the method uniformly.

## What's open-sourced

- **Benchmark code + results** (shared harness, per-method repro cards, all result JSONs).
- **The survey paper** (Overleaf-ready).
- An **awesome-structural-entropy** reading list.

If you work on SE, we'd love contributions — especially the LLM track and head-to-head TAD baselines.
