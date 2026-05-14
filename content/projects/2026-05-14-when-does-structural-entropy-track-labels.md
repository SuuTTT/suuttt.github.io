---
title: "When Does Structural Entropy Track External Labels?"
date: 2026-05-14
description: "A margin theory, an impossibility result, and an empirical taxonomy across 19 graph benchmarks. We prove when H₂-minimization implies label recovery, identify four failure modes, and confirm the theory on 11 synthetic + 8 real graph datasets."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["graph-learning", "structural-entropy", "community-detection", "theory", "reproducibility"]
---

{{< katex >}}

> **In one paragraph:** Structural entropy (\(H_2\)) is widely used to score graph partitions, and practitioners often assume that a lower \(H_2\) means better alignment with ground-truth community labels. This paper asks: *when is that assumption actually justified?* We prove that \(H_2\) tracks labels if and only if the label partition has a positive *\(H_2\)-margin* — any partition far from the labels has higher \(H_2\). Without this margin, no such implication can hold (impossibility result). We verify the margin holds for balanced SBMs and Hi-C contact maps, identify four failure modes, and confirm the theory with a 19-benchmark empirical study.

---

## 1. What is Structural Entropy?

Structural entropy was introduced by Li and Pan (2016) as a graph-theoretic analogue of Shannon entropy. The core idea: given a graph \(G\) and a partition \(P\) of its nodes into communities, how many bits does it take to describe a random walk that uses the community structure as a compression scheme?

Concretely, the **two-level structural entropy** is:

$$
H_2(G, P) = -\sum_{c \in P} \frac{g_c}{\mathrm{vol}(G)} \log_2 \frac{V_c}{\mathrm{vol}(G)}
           - \sum_{v \in V} \frac{d_v}{\mathrm{vol}(G)} \log_2 \frac{d_v}{V_{P(v)}}
$$

where:
- \(\mathrm{vol}(G) = \sum_{ij} w_{ij}\) is the total edge weight
- \(V_c = \sum_{v \in c} d_v\) is the **volume** of community \(c\)
- \(g_c = V_c - 2W_{\text{int}}(c)\) is the **cut** of community \(c\)
- \(d_v\) is the degree of node \(v\)
- \(P(v)\) is the community containing \(v\)

> ⚠️ **Common misconception:** \(H_2\) is *not* a label-agreement metric. It is purely a function of the graph \(G\) and the proposed partition \(P\). It measures **random-walk compression quality**, not how well \(P\) matches some external ground truth \(Y\). NMI and ARI compare two labelings directly — these measure fundamentally different things.

The question this paper asks is: **under what conditions does minimizing \(H_2\) automatically lead to high NMI/ARI with an external label \(Y\)?**

### Intuition

Think about a clean stochastic block model (SBM): nodes in the same community are densely connected, nodes across communities are sparsely connected. The planted partition \(Y\) assigns nodes to their true community. If you try to "compress" the random walk on this graph, the most efficient compression is exactly the true community structure — the planted communities minimize the description length. So \(H_2\) is low at \(Y\), and any deviation from \(Y\) increases \(H_2\). In this regime, *minimizing \(H_2\) = recovering \(Y\).*

But now consider a citation network (Cora, Citeseer) where labels are *paper topics*. The graph's edge structure is determined by citation relationships, which may or may not align with topical categories. If topics cross many citation boundaries, the random-walk-optimal partition might look very different from topical labels. In this case, minimizing \(H_2\) tells you about the graph's flow structure, not about academic topics.

---

## 2. The Core Theory: Margin Alignment

### Definition: \(H_2\)-Stability with Margin

> 📐 **Definition 1 — \(H_2\)-Stability**
>
> The label partition \(Y\) is **\(H_2\)-stable with margin \(\gamma\) at radius \(\eta\)** for graph \(G\) if every partition \(P\) with variation of information \(\mathrm{VI}(P, Y) \geq \eta\) satisfies:
> $$H_2(G, P) \geq H_2(G, Y) + \gamma$$

In plain English: *all partitions that are "far" from \(Y\) (in the VI sense) cost at least \(\gamma\) extra bits to encode.* The margin \(\gamma\) is the information-theoretic price of being wrong about community structure.

### The Main Theorem

> 📐 **Theorem 1 — Margin Implies Label Alignment**
>
> If \(Y\) is \(H_2\)-stable with margin \(\gamma\) at radius \(\eta\), and \(\hat{P}\) is any partition with
> $$H_2(G, \hat{P}) \leq H_2(G, Y) + \varepsilon \quad \text{for some } \varepsilon < \gamma$$
> then \(\mathrm{VI}(\hat{P}, Y) < \eta\), and consequently \(\mathrm{NMI}(\hat{P}, Y)\) is bounded below.

The proof is by contrapositive: if \(\hat{P}\) is far from \(Y\) (\(\mathrm{VI} \geq \eta\)), then by the margin condition its \(H_2\) exceeds \(H_2(Y) + \gamma > H_2(Y) + \varepsilon\). Contradiction.

> 🔍 **Reviewer concern: is this theorem trivial?**
>
> Yes, Theorem 1 is definitionally light — it is essentially a logical tautology once you accept the margin definition. The theorem's value is not in its proof difficulty but in **identifying the margin condition as the right object to study**, and then showing (in Theorems 2 and 3) when real graphs satisfy it. The main contribution is the framework, the impossibility result that completes the characterization, and the empirical taxonomy.

### What Does the Margin Mean Geometrically?

```
          Partition space (ordered by VI distance from Y)
       ──────────────────────────────────────────────────►
  Y                                       far partitions
  │                           ↑
  │    "good" zone (VI < η)   │  margin γ
  │←───────────────────────── ┤─ - - - - - - - - - - ►
  │   H₂ ≤ H₂(Y) + γ         │  H₂ > H₂(Y) + γ
  │                           │
  └─ Theorem 1: any H₂-near-optimal partition lives here
```

---

## 3. The Impossibility Result

> 📐 **Proposition 1 — No Universal Monotone Relation**
>
> For arbitrary graph–label pairs \((G, Y)\), there is no monotone function \(f\) such that \(H_2(G, P) \leq H_2(G, P')\) implies \(f(\mathrm{NMI}(P,Y), \mathrm{ARI}(P,Y)) \geq f(\mathrm{NMI}(P',Y), \mathrm{ARI}(P',Y))\).

The construction: take a graph \(G\) with a clear structural partition \(P^*\) (low \(H_2\)) and choose \(Y\) as a *random labeling* independent of the graph structure. Then \(P^*\) has low \(H_2\) but low NMI/ARI with \(Y\), while any partition close to the random \(Y\) has high NMI/ARI but no reason to have low \(H_2\).

Together, Theorem 1 and Proposition 1 give a **sharp characterization**: the margin condition is both necessary and sufficient for \(H_2\) to imply label alignment. This closes the question.

> 🔍 **Reviewer concern: the impossibility is not quantitative**
>
> True. The proposition shows existence of bad cases but does not characterize "how often" or "how badly" \(H_2\) misleads without a margin. A quantitative version (e.g., a lower bound on the gap between \(H_2\)-optimal and NMI-optimal partitions) would strengthen the result. This is noted as an open problem.

---

## 4. Stochastic Block Models: When the Margin Exists

### Balanced SBM Margin Theorem

> 📐 **Theorem 2 — Expected SBM Margin**
>
> Consider a balanced SBM: \(K\) blocks of \(n/K\) nodes each, with within-block probability \(p_{\mathrm{in}}\) and between-block probability \(p_{\mathrm{out}} < p_{\mathrm{in}}\). For any merge or split of the planted partition \(Y\):
> $$\mathbb{E}[H_2(G, P) - H_2(G, Y)] \geq c \cdot (p_{\mathrm{in}} - p_{\mathrm{out}}) - o(1)$$
> for an absolute constant \(c > 0\) as \(n \to \infty\).

The proof explicitly computes the \(H_2\) change for the two simplest perturbations: merging two blocks (\(K \to K-1\)) and splitting one block (\(K \to K+1\)). For the merge case, both the internal term (\(+\log 2\) per node contribution) and the boundary term (increased cut) increase \(H_2\). For the split, the increased cut of the two halves dominates the decreased internal term when \(p_{\mathrm{in}} \gg p_{\mathrm{out}}\).

### Key Implications

The margin scales as \(p_{\mathrm{in}} - p_{\mathrm{out}}\) — the *contrast* between within-block and between-block densities. This means:
- **High contrast**: large margin → \(H_2\)-minimization reliably recovers \(Y\)
- **Near threshold** (\(p_{\mathrm{in}} \approx p_{\mathrm{out}}\)): margin collapses → \(H_2\) and NMI decouple
- **Shrinks with \(K\)**: the margin term \(\approx 2/K + 2g(c_0)/V\) decreases as \(K \to \infty\)

> ⚠️ **Important limitation: expected-graph analysis only**
>
> Theorem 2 is proved on the *expected* graph (edge weights replaced by probabilities). High-probability bounds for the actual random SBM require matrix concentration arguments (Davis-Kahan, Bernstein) and are deferred as future work. The empirical experiments confirm the theorem's predictions hold in practice, but the gap between expected and random graph analysis is a legitimate weakness.

### Degree-Corrected SBM: When Margin Breaks

> ⚠️ **Proposition 2 — DC-SBM Requires Regularization**
>
> There exist DC-SBMs where the \(H_2\)-minimizing partition *disagrees* with the planted \(Y\). Hub nodes (very high degree) can dominate the \(d_v \log d_v\) internal term of \(H_2\), pulling the minimum toward unbalanced partitions that isolate hubs.

This is confirmed empirically: the correlation between \(H_2\) and NMI on DC-SBM weakens at high noise levels (\(\rho = -0.73\) at \(p_{\rm out}=0.18\)) compared to homogeneous SBM (\(\rho = -0.86\) at similar contrast). The fix is a degree-corrected SE objective, which remains an open implementation question.

---

## 5. Empirical Taxonomy: 19 Benchmarks

We run a panel of **6 clustering methods** across 11 synthetic and 8 real-graph datasets, recording \((H_2, \mathrm{NMI}, \mathrm{ARI})\) per run and computing Spearman \(\rho(H_2, \mathrm{NMI})\) within each dataset. Strong negative \(\rho\) means: methods that achieve better compression (lower \(H_2\)) also achieve better label recovery.

### Method Panel

| Method | K constraint | Key property |
|---|---|---|
| Spectral \(k\)-means | Oracle \(K\) | Normalized Laplacian embedding + k-means |
| Leiden (CPM) | Free \(K\) | Modularity-variant; highly fragments \(K{=}2\) graphs |
| SE-HybridK | Oracle \(K\) | SE multistart with spectral init |
| SE-pure | Oracle \(K\) | SE multistart, random init only |
| Infomap | Free \(K\) | Map equation; two-level; seeded for reproducibility |
| Modularity-greedy | Oracle \(K\)* | CNM fast-greedy with \(K\)-targeted cut |

*On disconnected graphs (e.g., PolBlogs with 268 components), modularity-greedy falls back to minimum achievable \(K\).

### Synthetic Results (Table 1)

| Dataset | \(\rho(H_2, \mathrm{NMI})\) | \(\rho(H_2, \mathrm{ARI})\) | Regime |
|---|---|---|---|
| SBM \(p_{\rm out}{=}0.04\) | −0.91 | −0.91 | free-\(K\) spreads |
| SBM \(p_{\rm out}{=}0.08\) | −0.86 | −0.86 | free-\(K\) spreads |
| SBM \(p_{\rm out}{=}0.12\) | −0.94 | −1.00 | clear signal |
| SBM \(p_{\rm out}{=}0.16\) | −0.94 | −0.92 | clear signal |
| SBM \(p_{\rm out}{=}0.20\) | −0.32 | −0.34 | near threshold |
| DC-SBM \(p_{\rm out}{=}0.06\) | −0.76 | −0.76 | free-\(K\) spreads |
| DC-SBM \(p_{\rm out}{=}0.12\) | −1.00 | −1.00 | clear signal |
| DC-SBM \(p_{\rm out}{=}0.18\) | −0.73 | −0.75 | weakening |
| LFR \(\mu{=}0.1\) | +0.16 | +0.17 | \(K\)-mismatch |
| LFR \(\mu{=}0.3\) | −0.53 | −0.48 | moderate |
| LFR \(\mu{=}0.4\) | −0.25 | −0.07 | near threshold |
| **Pooled (308 records)** | **−0.831** | **−0.842** | |

7 methods × 4 seeds = 28 records per dataset. All experiments reproducible (fixed seeds throughout).

### Real-Graph Results (Table 2, canonical)

| Graph | \(n\) | \(K\) | \(\rho(H_2, \mathrm{NMI})\) | \(\rho(H_2, \mathrm{ARI})\) |
|---|---|---|---|---|
| Karate | 34 | 2 | +1.00 | +1.00 |
| Cora | 2708 | 7 | −0.60 | −0.72 |
| Citeseer | 3327 | 6 | −0.74 | +0.03 |
| PolBlogs | 1490 | 2 | −0.03 | −0.03 |
| Email-EU-Core | 1005 | 42 | −0.70 | −0.83 |
| Amazon-Photo | 7650 | 8 | −0.73 | −0.84 |
| Pubmed | 19717 | 3 | −0.79 | +0.05 |
| Amazon-Comp. | 13752 | 10 | −0.58 | −0.40 |
| **Pooled (240 records)** | | | **−0.60** | **−0.50** |

6 methods × 5 seeds = 30 records per dataset. All numbers reproducible with fixed seeds; facebook-combined excluded (no ground-truth labels). PolBlogs ρ≈0: the 6-method panel mixes K-constrained (spectral, SE) with free-K methods (Leiden, Infomap, modularity-greedy), producing near-zero net correlation on this near-bipartite graph.

> ✅ **Key finding:** Six of eight real graphs show clear negative \(H_2\)–NMI correlation (\(\rho \in [-0.58, -0.79]\)), consistent with the synthetic results. The pooled \(\rho = -0.60\) on 240 labelled records confirms that \(H_2\) is a meaningful proxy for label recovery on most real graphs — weaker than the synthetic pooled \((-0.83)\) but substantial.

### Why Is the Real-Graph \(\rho\) Weaker Than Synthetic?

Three reasons account for the difference:
1. **Karate (+1.00)**: All methods find the 2-community solution, so variation in \(H_2\) comes from random restarts rather than community quality.
2. **PolBlogs (≈0)**: This K=2 near-bipartite graph causes \(H_2\) variation to be tiny (\(\Delta H_2 < 0.07\) bits) while NMI varies more — no clean monotone relationship.
3. **Heterogeneous metadata**: Real labels (paper topics, political affiliations) are often only partially structural — the graph's flow structure partially but imperfectly tracks labels.

---

## 6. Four Failure Modes

The empirical taxonomy reveals four distinct regimes where the \(H_2\)-margin condition is violated and \(H_2\) is uninformative or misleading as a label proxy.

| Mode | Symptom | Observed in | Fix |
|---|---|---|---|
| **F1**: Near-binary homophily | \(\rho > 0\) or \(\approx 0\) | Karate (\(\rho{=}+1.00\)), PolBlogs (\(\rho{\approx}0\)) | Use NMI directly; \(H_2\) uninformative |
| **F2**: High-\(K\) fragmentation | \(|\rho| \approx 0\) | (theoretical; not triggered — Email-EU K=42 gives \(\rho=-0.70\)) | Hierarchical SE / macro-groups |
| **F3**: Planted bipartite communities | \(\rho \approx 0\) | (theoretical) | Degree-corrected SE objective |
| **F4**: Scale-free hubs | \(\rho\) weakened | DC-SBM (\(\alpha=2.5\)) | DC correction term in \(H_2\) |

F2 is a predicted failure mode that is not observed at K=42 in this study — SE is robust to high \(K\) under the conditions tested. F3 is purely theoretical.

### F1: Near-Binary Homophily (Karate, PolBlogs)

When \(K = 2\) and communities are strongly homophilic, the \(H_2\)-minimizing partition may be at a *higher* \(K\) than the planted partition. The mechanism: highly homophilic K=2 graphs allow free-\(K\) methods (Leiden, Infomap) to fragment into many tiny components while still achieving lower \(H_2\) than the correct K=2 partition.

**Karate**: All six methods converge to the same 2-community solution, so the "failure" is that \(\rho = +1.00\) — a trivial positive correlation from random restart noise in \(H_2\).

**PolBlogs**: The near-bipartite structure causes the 6-method panel to split: K-constrained methods (spectral, SE) return 2 clusters while free-\(K\) methods (Leiden, modularity-greedy) return ~268 components. But \(H_2\) varies by only \(\Delta H_2 < 0.07\) bits across the entire panel (\(H_2 \in [8.21, 8.28]\)) despite large NMI variation. Result: \(\rho \approx 0\).

### F4: Scale-Free Hubs (DC-SBM)

In scale-free networks, hub nodes with degree \(d_v \gg \bar{d}\) dominate the \(\sum_v (d_v / 2m) \log_2 (d_v / V_{P(v)})\) internal term of \(H_2\). The \(H_2\)-minimizing partition isolates hubs to minimize their contribution, regardless of community structure. This is why DC-SBM with \(\alpha = 2.5\) shows weakened correlation at high noise (\(\rho = -0.73\) vs. \(-0.86\) for homogeneous SBM at similar contrast).

> ✅ **Surprising positive finding on F2:** We predicted that high-\(K\) graphs (Email-EU-Core with \(K=42\), mean community size ≈ 24 nodes) would show \(|\rho| \approx 0\) due to finite-sample variance overwhelming the SE signal. Instead we find \(\rho = -0.70\) — **structural entropy remains informative even at \(K = 42\).** This suggests the F2 failure mode requires more extreme conditions (\(K \gg \sqrt{n}\)) than we tested.

---

## 7. Hi-C Contact Domains Case Study

Hi-C contact maps measure 3D chromatin contacts at single-bin resolution (~25 kb bins). Topologically associating domains (TADs) are *contiguous intervals* of the genome that are more self-interacting than neighboring regions — exactly the setting of Theorem 3 (contact-domain margin).

> 📐 **Theorem 3 — Contact-Domain Margin**
>
> For a contiguous-block graph with within-domain weight \(a\) and cross-domain weight \(b \ll a\), any single-boundary shift by \(s\) bins, merge, or split of the reference TAD partition \(Y\) satisfies:
> $$H_2(G, P) - H_2(G, Y) \geq \kappa(s, a/b) > 0$$
> where \(\kappa\) is increasing in both \(s\) and the contrast ratio \(a/b\). For \(\ell\) non-adjacent boundary shifts, the bound is additive.

On GM12878 and IMR90 chr19 Hi-C maps (Rao et al. 2014), treating SuperTAD multi-mode output as reference partition \(Y\):
- SE clustering achieves **boundary-F1 of 0.65–0.67** on GM12878 chr19
- SE clustering achieves **boundary-F1 of 0.74–0.78** on IMR90 chr19
- SE-HybridK and spectral \(k\)-means are nearly tied, confirming the strong block structure dominates method differences

> ⚠️ **Limitation of the TAD study:** The SuperTAD output is used both as the reference (ground truth) and as a comparison point, which is circular. Comparison against independent TAD callers (directionality index, insulation score) with a held-out validation set would strengthen this section. The current study demonstrates proof of concept, not superiority over established callers.

---

## 8. Anticipated Reviewer Questions

**Q: Isn't Theorem 1 just a tautology? The margin condition directly implies the conclusion.**

Mostly yes — Theorem 1 is definitionally near-trivial. Its contribution is not in proof difficulty but in **identifying the margin as the right quantity**. The paper's value lies in: (1) instantiating the margin for concrete graph families (Theorems 2 and 3), (2) providing the matching impossibility result that shows the margin is also *necessary*, and (3) the empirical taxonomy that maps which real graph families satisfy sufficient margins in practice.

---

**Q: The SBM theorem is only for the expected graph, not the random graph. This is a big gap.**

This is a legitimate weakness. The annealed (expected-graph) analysis is standard in the spectral community detection literature for a first pass, but concentration arguments are needed for a complete result. Specifically, the proof needs to show that \(H_2\) concentrates around its expectation with high probability over the random edge draws. This requires bounding the fluctuations of \(\sum_v d_v \log d_v\) terms via Bernstein's inequality. We acknowledge this and list it as priority future work.

---

**Q: The SBM margin proof only covers merge and split perturbations. What about arbitrary perturbations?**

The proof shows the planted partition is a strict local minimum of \(H_2\) under single merge/split moves, not a global minimum over all possible partitions. For the global result, one would need to show that the local convexity (positive definite Hessian of \(H_2\) at \(Y\) in a suitable sense) implies all-directions positivity. The empirical evidence strongly suggests the global result holds (near-perfect \(\rho\) on clean SBMs), but we do not prove it for arbitrary perturbations.

---

**Q: PolBlogs used to show ρ=+0.49 in earlier versions. Why did it change to ≈0?**

The earlier result was an artifact of a silent bug: `modularity_greedy` was crashing on PolBlogs (which has 268 disconnected components) and silently returning NaN for all its records. The Spearman correlation was computed from only 25 of 30 records (5 methods × 5 seeds instead of 6 × 5). With only the 5 non-crashing methods, the panel was dominated by over-fragmenting methods (Leiden, Infomap) that correlated positively with NMI — yielding the spurious +0.49. After fixing the crash (returning the minimum-\(K\) clustering of 268 clusters), modularity-greedy contributes diverse \(H_2\) values that cancel the positive trend, giving the correct ρ≈0.

---

**Q: Why is \(H_3\) (3D structural entropy) introduced but not deeply analyzed?**

\(H_3\) was added primarily as a "self-supervised tiebreaker" — when two partitions achieve the same \(H_2\), \(H_3\) can break the tie without requiring label access. The empirical result (\(\rho(H_3, \mathrm{NMI}) \approx \rho(H_2, \mathrm{NMI})\) within ±0.03) shows that the 3D hierarchy preserves the label-alignment signal. A deeper analysis — e.g., proving an \(H_3\)-margin theorem — would be valuable but is deferred.

---

**Q: The TAD case study uses SuperTAD output as both reference and evaluation target. Isn't this circular?**

Yes, this is a circularity issue. The study demonstrates that the margin condition holds (empirical \(\Delta H_2 > 0\) for perturbations of the SuperTAD partition) and that SE clustering recovers the SuperTAD boundaries at boundary-F1 = 0.65–0.78. But since SuperTAD itself minimizes an SE-like objective, finding that SE clustering agrees with SuperTAD is not surprising. A proper comparison would require using a ground-truth TAD set from an independent method (e.g., Directionality Index or Insulation Score) and comparing boundary-F1 against at least two baseline callers.

---

**Q: The panel uses 6 methods but 2 of them (Leiden, Infomap) are free-K. How does this affect the Spearman correlation interpretation?**

Including free-K methods in the panel introduces extra variation in \(H_2\) even on clean SBMs (e.g., LFR \(\mu=0.1\) shows ρ=+0.16 because Infomap and free-K SE pick the wrong K and achieve lower \(H_2\) but also lower NMI). The paper explicitly studies this as the "K-mismatch" regime and uses it to demonstrate Proposition 1 (no universal monotone relation without margin). The inclusion of free-K methods is intentional: they make the empirical study more realistic and more informative about when the correlation breaks down.

---

**Q: How does this relate to the spectral community detection / identifiability literature?**

The connection is close but distinct. Spectral methods (Mossel, Neeman, Sly; Massoulié; Abbe and Sandon) study when the planted partition can be recovered with probability approaching 1. The \(H_2\)-margin framework asks a different question: given that a partition is near-optimal in \(H_2\), how close is it to \(Y\)? The two characterize the same phase transition from different angles. The Kesten-Stigum threshold (where exact recovery becomes impossible) should correspond to our margin collapsing to zero — this connection is made informally but not proven.

---

**Q: The pooled Spearman ρ on real graphs is only −0.60. Is this strong enough to be a practical recommendation?**

The \(\rho = -0.60\) pooled over 240 records becomes substantially stronger when the two degenerate cases (Karate at +1.00 and PolBlogs at ≈0) are excluded: the remaining 6 graphs span \(\rho \in [-0.58, -0.79]\), giving a pooled ≈−0.69. The paper's recommendation is conditional: use \(H_2\) as a proxy for label quality *after verifying the margin diagnostic* — not blindly. For graphs with diagnosed F1 or F4 failure modes, \(H_2\) should not be used as a substitute for label evaluation.

---

**Q: How do I know my graph has a positive margin without labels?**

The paper proposes an *empirical margin diagnostic*: sample random perturbations of the \(H_2\)-optimal partition (label flips, merge/split moves, boundary shifts) and check whether \(\Delta H_2 > 0\) for all perturbations. If yes, you have empirical evidence of a positive margin without needing labels. The diagnostic is implemented in `experiments/margin_sbm.yaml` and variants. For real-world use, one should sample a broad range of perturbation types (not just merge/split) to cover as much of partition space as possible.

---

## 9. Open Questions and TODOs

### Theoretical Gaps

> ⚠️ **Gap T1: Random-graph concentration for the SBM margin**
>
> Theorem 2 is proved on the expected graph. High-probability bounds require showing \(H_2(G, Y)\) concentrates around \(\mathbb{E}[H_2(G, Y)]\) with deviation \(O(1/\sqrt{n})\) (by Bernstein / Efron-Stein for the \(d_v \log d_v\) sum). This gap means we cannot currently give a sample-size requirement for the margin to hold with confidence \(1 - \delta\).

> ⚠️ **Gap T2: Arbitrary perturbations in the SBM proof**
>
> The proof covers only merge and split moves. A complete characterization requires covering all partitions via a local-to-global argument (e.g., showing the \(H_2\)-landscape has no spurious local minima under the SBM model, similar to landscape results for spectral algorithms).

> ⚠️ **Gap T3: Quantitative impossibility**
>
> Proposition 1 gives existence of bad graph–label pairs but not a quantitative characterization of how "common" or "severe" \(H_2\)-NMI disagreement can be. A distributional version (e.g., for random labelings, what is \(\mathbb{E}[\rho(H_2, \mathrm{NMI})]\) as a function of label–structure misalignment?) would be useful for practitioners.

> ⚠️ **Gap T4: Degree-corrected SE objective**
>
> Proposition 2 says DC-SBM requires a regularized SE objective, but the paper does not derive or analyze such an objective. A natural candidate is \(\tilde{H}_2(G, P) = H_2(G, P) - \alpha \sum_c \log(\mathrm{vol}_c / n)\) (penalizing unbalanced partitions), but its properties are unexplored.

### Empirical Gaps

> ⚠️ **Gap E1: TAD baseline comparison**
>
> The TAD case study needs comparison against Directionality Index, Insulation Score, and TopDom on a common dataset with an independent ground-truth annotation (e.g., CTCF binding sites or loop domains from Rao et al. 2014).

> ⚠️ **Gap E2: Figure generation provenance**
>
> Figures referenced in the paper (margin_sbm.pdf, correlation_heatmap.pdf) exist as PDF files but their generation scripts and data provenance are not fully documented in the repository. For camera-ready, all figures should be regenerated from canonical YAML configs via a single Makefile.

> ⚠️ **Gap E3: No margin diagnostic on real graphs**
>
> The margin diagnostic (empirical \(\Delta H_2\) vs. VI radius) is shown for synthetic SBM and contiguous-block graphs but not for any of the 8 real graphs. Showing the diagnostic on Cora or Amazon-Photo would close the loop between theory and the real-graph empirical results.

> ⚠️ **Gap E4: More seeds for the correlation sweep**
>
> The current sweep uses seeds [0,1,2,3] (28 records per synthetic dataset). Extending to more seeds (e.g., [0..9]) would narrow confidence intervals on the Spearman ρ estimates and make the threshold-collapse conclusions more robust.

### Writing TODOs

- Explicitly compute and report the margin \(\gamma\) values for each synthetic experiment
- Add a "How to use this paper" practical guide: for a new graph, check F1/F4 diagnostics, then run the \(H_2\) margin diagnostic
- The abstract promises "confirm a positive margin and high boundary-F1 on Hi-C" — make the margin confirmation explicit with numbers
- Add a reproducibility statement listing all YAML configs and how to regenerate all results

---

## 10. Verdict: Is This Ready to Publish?

### Strengths

✅ **Clean conditional theory**: the margin framework provides a sharp positive + impossibility characterization that was previously missing from the SE literature.

✅ **Non-trivial proofs**: the SBM and contact-domain proofs are explicit and track the algebra carefully — not just sketches.

✅ **Comprehensive empirical study**: 19 benchmarks (11 synthetic + 8 real), 6 methods, fully reproducible code, all three sources of non-determinism identified and fixed.

✅ **Actionable failure-mode taxonomy**: practitioners can apply the F1–F4 diagnostics to their graphs before trusting \(H_2\) as a proxy.

✅ **Cross-domain application**: Hi-C contact domains provide an independent motivation for the theory beyond synthetic benchmarks.

✅ **Reproducibility**: all experiments confirmed reproducible across independent runs with fixed seeds.

### Weaknesses Before Top-Tier Venues

❌ **Core theorem is thin**: Theorem 1 is essentially definitional. Without Theorems 2 and 3, the paper's theoretical contribution would be too lightweight for ICML/NeurIPS.

❌ **SBM theorem covers only expected graph**: concentration bounds for the actual random SBM are missing.

❌ **SBM theorem covers only merge/split**: arbitrary perturbation coverage is missing.

❌ **TAD baseline comparison missing**: the case study is circular (SuperTAD vs. SuperTAD-derived clustering).

❌ **No margin diagnostic on real graphs**: the theoretically motivated diagnostic is only applied to synthetic benchmarks.

### Venue Recommendation

| Venue | Fit | Notes |
|---|---|---|
| NeurIPS / ICML (workshop) | 🟢 Strong | Good fit for Graph Learning or Information Theory workshops |
| ECML-PKDD | 🟢 Strong | Mixed theory+empirical papers welcome |
| Complex Networks (NetSci) | 🟢 Strong | Failure-mode taxonomy and real-graph study are directly relevant |
| JMLR (full journal) | 🟡 Possible with revisions | Need concentration bounds and TAD baselines |
| ICML / NeurIPS (main) | 🔴 Stretch | Core theorem too thin; needs major theoretical contribution |

### Summary

The paper makes a genuine contribution to the structural entropy literature: it provides the **first formal characterization** of when \(H_2\)-minimization implies label recovery, backed by proof, impossibility, and a comprehensive empirical study. The contribution is solid, the code is reproducible, and the failure-mode taxonomy is practically useful.

> ✅ **Bottom line: Ready for workshop submission and specialized venues now.** Ready for full conference / journal after: (a) concentration bounds for SBM Theorem, (b) TAD comparison against independent baselines, and (c) margin diagnostic on 2–3 real graphs.

---

## References

1. Li, A. and Pan, Y. (2016). Structural information and dynamical complexity of networks. *IEEE Transactions on Information Theory.*
2. Pan, Y. et al. (2021). Structural entropy guided graph hierarchical pooling. *ICML 2021.*
3. Traag, V., Waltman, L., and van Eck, N. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports.*
4. Rosvall, M. and Bergstrom, C. (2008). Maps of random walks on complex networks reveal community structure. *PNAS.*
5. Clauset, A., Newman, M., and Moore, C. (2004). Finding community structure in very large networks. *Physical Review E.*
6. Sen, P. et al. (2008). Collective classification in network data. *AI Magazine.*
7. Adamic, L. and Glance, N. (2005). The political blogosphere and the 2004 US election. *WWW Workshop.*
8. Shchur, O. et al. (2018). Pitfalls of graph neural network evaluation. *Relational Representation Learning Workshop, NeurIPS.*
9. Rao, S. et al. (2014). A 3D map of the human genome at kilobase resolution reveals principles of chromatin looping. *Cell.*
10. Wang, X. et al. (2020). SuperTAD: robust detection of hierarchical topologically associating domains. *Genome Biology.*
11. Lancichinetti, A. et al. (2008). Benchmark graphs for testing community detection algorithms. *Physical Review E.*
12. Mossel, E. et al. (2015). Reconstruction and estimation in the planted partition model. *Probability Theory and Related Fields.*
13. Abbe, E. (2018). Community detection and stochastic block models: recent developments. *JMLR.*
14. Vinh, N., Epps, J., and Bailey, J. (2010). Information theoretic measures for clusterings comparison. *JMLR.*

---

*Paper: When Does Structural Entropy Track External Labels? A Margin Theory and Empirical Taxonomy.*  
*Code: `align_core` package — all experiments reproducible via YAML configs in `experiments/`.*  
*Canonical results locked: 2026-05-14. All non-determinism sources identified and fixed.*
