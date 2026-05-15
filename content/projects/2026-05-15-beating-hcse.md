---
title: "Beating HCSE: Next-Generation Structural Entropy Minimization"
date: 2026-05-15
description: "A deep dive into the recent UAI 2025 paper on Hierarchical Clustering via Structural Entropy, and our roadmap to beating it."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["research", "clustering", "sota"]
---

{{< katex >}}

## Introduction

Hierarchical clustering is foundational for understanding the multi-granularity structure of complex networks. A recent and highly impactful paper, *"An Information-theoretic Perspective of Hierarchical Clustering on Graphs"* (Pan et al., UAI 2025), introduces a novel perspective using Structural Entropy (SE) instead of traditional combinatorial cost functions like Dasgupta's.

In this post, we'll review their theory, method, and results, and introduce our new theoretical findings and proposed improvements to push the state-of-the-art even further.

## Their Method and Theory

The authors formulate a new cost function based on the structural information theory of graphs. Structural entropy measures the complexity of hierarchical networks by determining the minimum average code length needed to encode a random walk on the graph. 

### Theory
The theoretical backbone bridges combinatorial objectives with information-theoretic ones. For a coding tree \(T\), the high-dimensional structural entropy (HD-SE) formulation naturally acts as a balance regularization. The authors proved that for unweighted cliques, the tree achieving minimum structural entropy is strictly a balanced binary tree, meaning the cost function implicitly encourages balance, unlike prior admissible functions. They also provide \(O(1)\)-approximation guarantees for expander-like and well-clustered graphs.

### Method
They proposed two main algorithms:
1. **BBM (Balanced Binary Merge):** A binary hierarchical clustering algorithm with rigorous approximation guarantees.
2. **HCSE:** A hyperparameter-free framework for non-binary clustering. It recursively identifies and stratifies the *sparsest level* of a tree using a "stretch-and-compress" scheme. It builds a local binary tree ("stretch") and then shrinks overlong paths ("compress") to naturally determine the optimal number of hierarchies.

## Their Results

Extensive experiments on Stochastic Block Models (SBM) and Hierarchical SBMs demonstrated HCSE's superiority. 
- **Binary Clustering:** BBM achieves competitive or superior combinatorial costs while drastically improving balance metrics (size, volume, and depth balance) compared to linkage-based baselines (like Linkage++).
- **Non-binary Clustering:** HCSE reliably identifies the true number of hierarchical levels without hyperparameters, substantially outperforming traditional methods like LOUVAIN and HLP in Normalized Mutual Information (NMI) on HSBM benchmarks.

## Our Idea and Improvement to Beat It

While HCSE represents a massive leap forward, it fundamentally relies on a greedy agglomerative heuristic during its "stretch" phase. This makes it susceptible to local minima, particularly on highly complex or noisy graphs.

We are developing a new algorithm to consistently outperform HCSE by treating HD-SE minimization as a global optimization problem.

### Theoretical Foundation: NP-Hardness of HD-SE
To establish the necessity of a better optimizer, we have formally proven that Unconstrained HD-SE Minimization is strictly **NP-hard**. Our proof uses a reduction from the strongly NP-complete 3-PARTITION problem. We demonstrate that for specific graph gadgets with dense element cliques and a uniform weakly-connected background, the optimal coding tree structure collapses to exactly depth 2. This maps the HD-SE problem directly to 2D-SE, where minimizing the entropy naturally forces a perfect volumetric balance equivalent to solving 3-PARTITION.

### The Path Forward
Armed with the knowledge that greedy approaches will inevitably hit local optima, our strategy aims to bridge the gap between flat (2D) and hierarchical (HD) structural entropy. We are building an advanced optimizer based on **multilevel refinement, simulated annealing, or a differentiable surrogate**. 

Our next-generation framework is designed to escape local minima that trap HCSE, theoretically and empirically achieving lower HD-SE scores across standard benchmarks like Cora, Citeseer, and synthetic SBMs. Stay tuned for our upcoming release!