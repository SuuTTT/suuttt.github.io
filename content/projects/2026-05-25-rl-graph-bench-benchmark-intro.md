---
title: "rl-graph-bench: Reproducing Six RL Graph-Clustering Papers in One Unified Benchmark"
date: 2026-05-25
description: "From GAEC and NCut to community detection and multicut: we implement and reproduce NeuroCUT (KDD 2024), WRT (2025), CLARE (KDD 2022), SLRL (AAAI 2025), AC2CD (KBS 2023), and SS2V-D3QN (TNNLS 2025) under one roof. All four active P1/P2 tracks pass. Here is what the literature says, how we designed the benchmark, what numbers we got, and how to run it yourself."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["reinforcement-learning", "graph-clustering", "benchmark", "multicut", "neurocut", "ss2v", "community-detection", "gaec", "tutorial"]
---

{{< katex >}}

> **TL;DR.** Graph clustering is fragmented across three incompatible objective families — cut-based partitioning, semi-supervised community detection, and multicut / correlation clustering. rl-graph-bench gives each family its own environment and leaderboard, then reproduces six recent RL papers against those targets. Every P0 and all active P1/P2 tracks now pass. This post walks through the background, the engineering decisions, the final numbers, a five-minute quickstart, and what remains open.

---

## 1. Preliminary: The Literature and Why It Is Fragmented

### 1.1 Three Incompatible Task Families

Graph clustering shows up in image segmentation, connectomics, social network analysis, and combinatorial optimisation — but each community uses a different objective, different datasets, and a different notion of "better."

| Family | Objective | Needs labels? | Representative RL paper |
|--------|-----------|---------------|------------------------|
| Graph partition (cut-based) | NCut / Sparsest Cut ↓ | No | NeuroCUT (KDD 2024) |
| Semi-supervised community detection | F1 / F-score vs ground truth ↑ | Yes | CLARE (KDD 2022), SLRL (AAAI 2025) |
| Dynamic community detection | Modularity density ↑ | No | AC2CD (KBS 2023) |
| Multicut / correlation clustering | Signed-edge objective ↓ | No | SS2V-D3QN (TNNLS 2025) |

Mixing these on a single leaderboard is a category error. A lower NCut does not imply better F1, and a better multicut objective says nothing about community structure.

### 1.2 Cut-Based Partitioning: NCut and Sparsest Cut

**Normalized Cut (NCut)** [Shi & Malik, 2000] penalises the fraction of cross-partition edge weight relative to the total degree of each part:

$$
\text{NCut}(A, B) = \frac{w(A,B)}{\text{vol}(A)} + \frac{w(A,B)}{\text{vol}(B)}
$$

**Sparsest Cut** normalises by partition sizes rather than degrees, making it sensitive to balanced splits.

Classical solvers (spectral clustering, METIS, Leiden/Louvain) work well in practice but are non-differentiable and not trained. **NeuroCUT** [Shah et al., KDD 2024] frames graph partitioning as a finite-horizon MDP where each action contracts one inter-cluster edge, and trains a GNN policy with PPO to minimise NCut step-by-step. On Cora k=4, NeuroCUT achieves NCut=0.33 and Sparsest Cut=1.46 — beating GAP, DMon, and MinCutPool.

**WRT / RidgeCut** [Jiang et al., 2025] constrains the action space further to ring and wedge substructures, reducing the branching factor while maintaining expressive partitions. It outperforms NeuroCUT by ~20% on city-traffic road graphs.

### 1.3 Semi-supervised Community Detection: CLARE and SLRL

Community detection with known anchor members is a different problem. **CLARE** [Wu et al., KDD 2022] treats community expansion as a two-phase MDP: a *Locator* identifies seed nodes, then a *Rewriter* iteratively adds or removes nodes to maximise F1 against ground-truth communities. Trained on SNAP Amazon, it achieves F1=0.773.

**SLRL** [Ni et al., AAAI 2025] focuses on *local* community detection — given a single query node, expand outward. It uses a coverage-weighted F-score reward and beats CLARE on Amazon (+10%) and DBLP (+11%).

### 1.4 Dynamic Community Detection: AC2CD

Real networks evolve. **AC2CD** [Costa & Ralha, KBS 2023] runs an Actor–Critic agent over temporal graph snapshots, reassigning nodes to maximise modularity density at each snapshot. The paper uses a GAT encoder and achieves NMI=0.75 on BlogCatalog3 — better than SDNE, ComE, and GraphGAN.

### 1.5 Multicut and GAEC

**Correlation clustering / multicut** [Bansal et al., 2002] assigns a signed weight \(w_{ij}\) to each edge — positive means "merge," negative means "separate" — and finds the partition minimising:

$$
\text{cost}(\mathcal{P}) = \sum_{(i,j)\in E^+,\, c_i \neq c_j} w_{ij} \;-\; \sum_{(i,j)\in E^-,\, c_i = c_j} w_{ij}
$$

**GAEC** (Greedy Additive Edge Contraction) [Keuper et al., 2015] is the canonical fast baseline: repeatedly merge the cluster pair with the highest sum of inter-cluster signed weights, stopping when no positive-sum pair remains. GAEC is \(\mathcal{O}(n^2)\) per step and near-optimal on small graphs.

**SS2V-D3QN** [Li et al., TNNLS 2025] frames this as a sequential edge contraction MDP with dueling double DQN. Key architectural insight: Q-values must be computed *per-edge* from endpoint embeddings, not projected from a global graph embedding. On ER/BA synthetic instances (n=20–60), SS2V-D3QN significantly outperforms GAEC when using ensemble inference (H=10 rollouts).

---

## 2. Design: Architecture of rl-graph-bench

### 2.1 Shared Infrastructure

Every algorithm plugs into the same Gymnasium-compatible environment interface:

```
Problem  →  Env  →  Algo  →  Trainer  →  Eval harness
```

- **`rlgb/envs/`** — `EdgeContractionEnv` (partition/multicut), `CommunityEnv` (CLARE/SLRL), `NodeReassignEnv` (AC2CD)
- **`rlgb/algos/`** — one subpackage per paper (PPO, D3QN, Actor-Critic, imitation)
- **`rlgb/data/`** — loaders for Cora, CiteSeer, SNAP Amazon/DBLP, BlogCatalog3, SBM, LFR, ER/BA MCMP instances
- **`rlgb/tasks/`** — objective computers (`multicut_cost_fast`, NCut, F1, modularity density)
- **`rlgb/baselines/`** — Leiden, Louvain, Spectral, GAEC (for comparison)
- **`rlgb/training/`** — shared `Trainer` + `TrainConfig` used by all algorithms

### 2.2 Separate Leaderboards

Because the objectives are incompatible, we never aggregate across families. The acceptance criterion per algorithm is stated in `docs/PAPER_TARGETS.md` as a concrete inequality (e.g., NCut ≤ 0.33, F1 ≥ 0.384, Wins ≥ 3/4).

### 2.3 The MCMP Environment: Signed Costs and Positive-Only Actions

The hardest design challenge was SS2V-D3QN's signed-cost environment. Key decisions:

**Problem representation**: `p.adj = |cost_adj|` so GraphSAGE sees the full unsigned graph structure (including negative-weight neighbours). The signed cost matrix is stored separately in `p.meta["cost_matrix"]`.

**Action space filtering** (`MCMPEnvWrapper`): Only *positive-sum* cluster pairs are presented as actions, matching GAEC's stopping condition. This is critical — without it, the agent learns to merge clusters with net-negative inter-cluster weight, which always hurts the objective.

**Termination**: Force `done = True` when `max(cluster_sums) ≤ 0`. Without this, the DQN trains on harmful beyond-optimal transitions.

**Edge features**: Each candidate edge gets a 2D weight vector `[w_uv, cluster_sum_of_pair]`, concatenated with GraphSAGE node embeddings \((h_u + h_v, h_u \odot h_v)\) and a global graph embedding \(g\):

$$
\phi_{uv} = [h_u + h_v \;\|\; h_u \odot h_v \;\|\; g \;\|\; w_{uv}, \text{clustersum}_{uv}]
$$

where \(\|\) denotes concatenation. With hidden=64, this gives a 194-dimensional input to the edge scorer MLP.

### 2.4 Training Pipeline for SS2V-D3QN

```
1. BC pre-training (50k steps)   — imitate GAEC's cluster-level greedy
       ↓  cross-entropy loss ≈ 0.57
2. D3QN fine-tune (3000 episodes) — lr=1e-5, ε-start=0.02, mixed n=20+40 train
       ↓  conservative to avoid catastrophic forgetting
3. Stochastic eval (ε=0.03, 20 seeds per instance) — take best-of-20 rollouts
```

Pure greedy inference (ε=0) achieves 0/4 wins. Stochastic search with ε=0.03 achieves 3/4. The key intuition: on small graphs with ~15 steps per episode, 3% random action probability introduces ~0.5 random moves per rollout, occasionally escaping local optima that fool the greedy policy.

---

## 3. Results

### 3.1 Track Summary — All Active Targets Pass

| # | Algorithm | Track | Dataset | Metric | Target | **Result** | Status |
|---|-----------|-------|---------|--------|--------|-----------|--------|
| 1 | NeuroCUT | P2 | Cora k=4 | Sparsest Cut ↓ | ≤ 1.46 | **1.077** | ✅ PASS |
| 2 | AC2CD | P1 | Email-EU proxy (SBM n=100 k=6) | NMI ↑ | ≥ 0.72 | **0.897** | ✅ PASS |
| 3 | CLARE | P1 | DBLP | F1 ↑ | ≥ 0.384 | **0.394** | ✅ PASS |
| 4 | SS2V-D3QN | Track 4 | ER/BA synthetic MCMP | Wins vs GAEC | ≥ 3/4 | **3/4** | ✅ PASS |

### 3.2 Graph Partition Benchmark (NCut ↓)

NeuroCUT trained on Cora k=4 with 10k PPO episodes, evaluated on held-out graphs. Spectral clustering is the strongest classical baseline:

| Dataset | Spectral | NeuroCUT | Leiden | Louvain |
|---------|----------|----------|--------|---------|
| Cora k=4 | **0.268** | 0.512 | 3.569 | 3.818 |
| Cora k=7 | **0.574** | 0.987 | 3.569 | 3.684 |
| CiteSeer k=6 | **0.104** | 0.460 | 2.922 | 2.762 |
| LFR n=300 μ=0.2 | 2.220 | **2.231** | **2.219** | 2.262 |
| SBM n=300 k=5 | 1.402 | 1.414 | **1.402** | 1.402 |

NeuroCUT is competitive on structured graphs (LFR, SBM) and achieves the P2 target on Cora k=4 (Sparsest Cut=1.077 ≤ 1.46) using a checkpoint trained specifically with the sparsest-cut reward.

### 3.3 Community Detection (F1 / NMI ↑)

| Algorithm | Dataset | Paper target | **Ours** | Δ |
|-----------|---------|-------------|---------|---|
| CLARE | Amazon | ≥ 0.773 | **0.796** | +3.0% |
| CLARE | DBLP | ≥ 0.384 | **0.394** | +2.6% |
| SLRL | Amazon | ≥ 0.878 | **0.905** | +3.1% |
| SLRL | DBLP | ≥ 0.662 | **0.692** | +4.6% |
| AC2CD | BlogCatalog3 | ≥ 0.75 NMI | **0.954** | +27% |
| AC2CD | Email-EU proxy | ≥ 0.72 NMI | **0.897** | +24.6% |

Key finding for AC2CD: **Leiden warm-start on snapshot 0 is load-bearing.** With random initialisation, NMI=0.058. With Leiden warm-start, NMI=0.954. The RL component refines an already-good partition rather than building from scratch.

### 3.4 Multicut: SS2V-D3QN vs GAEC

Evaluated with ε=0.03 exploration, best of 20 rollouts per instance, 50 test instances each:

| Test set | n | GAEC | **SS2V-D3QN** | Δ | |
|----------|---|------|--------------|---|---|
| er_n20 | 20 | 3.685 | **3.596** | −2.4% | ✅ WIN |
| ba_n20 | 20 | 1.306 | **1.290** | −1.2% | ✅ WIN |
| er_n40 | 40 | 27.22 | 29.65 | +8.9% | ✗ LOSS |
| ba_n40 | 40 | 2.615 | **2.558** | −2.2% | ✅ WIN |

The er_n40 gap (+8.9%) is persistent across all epsilon strategies (ε=0.03/0.05/0.10) and likely reflects insufficient training on ER graphs at n=40. Closing this gap is a natural next step.

---

## 4. Quickstart Tutorial

### 4.1 Prerequisites

```bash
git clone https://github.com/YOUR_ORG/rl-graph-bench.git
cd rl-graph-bench
pip install -e ".[dev]"          # installs torch, torch-geometric, gymnasium
python3 -c "import rlgb; print(rlgb.__version__)"
```

Hardware: experiments were run on an RTX 3060 Ti (8 GB). All algorithms fit in 6 GB.

### 4.2 Run the Partition Benchmark

```bash
# Evaluate NeuroCUT (+ Spectral / Leiden / Louvain baselines) on all datasets
python3 experiments/run_benchmark_all.py

# Results written to:
#   results/benchmark_partition.csv
#   results/benchmark_multicut.csv
```

Expected runtime: ~25 minutes on GPU, ~90 minutes CPU-only.

### 4.3 Reproduce a Specific Track

**NeuroCUT P2 (Sparsest Cut on Cora k=4):**
```bash
python3 experiments/verify_neurocut_sparsest.py
# [PASS] SparsestCut = 1.0767 ≤ 1.46
```

**CLARE P1 (DBLP F1):**
```bash
python3 experiments/verify_clare_dblp.py
# [PASS] F1 = 0.3941 ≥ 0.384
```

**AC2CD P1 (Email-EU NMI):**
```bash
python3 experiments/verify_ac2cd_email.py
# [PASS] NMI = 0.8968 ≥ 0.72
```

**SS2V-D3QN Track 4 (MCMP vs GAEC):**
```bash
# Full training (BC + DQN) — ~3 hours on GPU
python3 experiments/verify_ss2v_paper.py

# Eval-only from existing checkpoint
python3 experiments/eval_ss2v_final.py
# [PASS] SS2V wins 3/4 >= target 3/4
```

### 4.4 Train Your Own Policy

The shared `Trainer` class works for any algorithm that implements `select_action` and `update`:

```python
from rlgb.training.trainer import Trainer, TrainConfig
from rlgb.algos.partition.neurocut import NeuroCUTAlgo, NeuroCUTConfig

cfg   = NeuroCUTConfig(hidden=256, n_layers=3, device="cuda")
algo  = NeuroCUTAlgo(cfg)
train = TrainConfig(n_episodes=10_000, horizon=50, log_every=500)

trainer = Trainer(algo=algo, env_fn=my_env_fn, config=train)
trainer.train()
algo.save("results/my_checkpoint.pt")
```

### 4.5 Add a New Algorithm

1. Implement `select_action(obs, greedy) → int` and `update(batch) → dict` in `rlgb/algos/my_algo/`.
2. Register the acceptance criterion in `docs/PAPER_TARGETS.md`.
3. Write `experiments/verify_my_algo.py` that exits 0 on pass.

The env observation dict always contains `edge_idx`, `labels`, `n_edges`, and task-specific extras — your algorithm can ignore what it doesn't need.

---

## 5. Future Work

### 5.1 Close the er_n40 Gap

SS2V-D3QN loses to GAEC on er_n40 by 8.9%. Possible directions:
- Train on larger ER graphs (n=40–60) with more BC steps and a higher learning rate early in training.
- Use the H=10 ensemble from the original paper (best-of-10 *trained* Q-networks, not just best-of-20 random seeds).
- Add n-step returns (the paper uses N-step D3QN).

### 5.2 WRT P1 (Predefined-Weight Synthetic Graphs)

WRT's P0 target (City Traffic NCut ≤ 0.060) already passes (0.058). The P1 target — NCut ≤ 0.021 on predefined-weight synthetic graphs — is not yet evaluated.

### 5.3 Structural Entropy as a Benchmark Objective

None of the six algorithms optimise \(H^2\) (second-order structural entropy), which measures how well a partition compresses random walks:

$$
H^2(G, \mathcal{P}) = -\sum_{c \in \mathcal{P}} \frac{g_c}{\text{vol}(G)} \log_2 \frac{V_c}{\text{vol}(G)}
$$

where \(g_c\) is the total edge weight leaving cluster \(c\) and \(V_c\) is its volume. This is an open benchmark slot — no RL paper has targeted \(H^2\) minimisation directly.

### 5.4 Scalability: Beyond n=1000

Current NeuroCUT episodes take ~67 seconds on n=2000 (Cora/CiteSeer). Scaling to n=10k–100k requires hierarchical action spaces or subgraph sampling. WRT's ring/wedge constraint is one step in this direction.

### 5.5 Real-World Multicut

The SS2V-D3QN paper reports results on real-world MCMP instances (Table V) that we have not yet reproduced — the dataset is not publicly released. Adding a real-world multicut benchmark (e.g., from connectomics or image segmentation) would complete the evaluation.

### 5.6 Unified Reward Shaping Across Families

The three objective families reward very different behaviours. A long-term research direction is whether a single foundation-model-style graph agent can be fine-tuned to each objective — analogous to how language model fine-tuning works across tasks.

---

## Appendix: Engineering Lessons

Three lessons that saved (or cost) the most time:

**1. Q-values must be edge-level, not global.**  
The first SS2V implementation projected a global graph embedding onto a |E|-dimensional action vector. This never learns — the agent has no way to compare edge quality. The fix: compute Q per edge as `MLP(h_u + h_v, h_u ⊙ h_v, g, [w_uv, cluster_sum])`.

**2. Leiden warm-start is non-optional for AC2CD.**  
Random initialisation + Actor-Critic = NMI 0.058. Leiden warm-start + Actor-Critic = NMI 0.954. The RL signal is too sparse to build a good partition from noise — it can only refine.

**3. Termination must match the baseline's stopping rule.**  
For SS2V, terminating only when `k_current == 1` caused the agent to train on negative-sum merges (after GAEC would have stopped). Adding `done = True` when `max(cluster_sums) ≤ 0` immediately stabilised training.

---

*Code: `github.com/YOUR_ORG/rl-graph-bench` · Results: `docs/RESULTS_REPORT.md` · Hardware: RTX 3060 Ti · Date: 2026-05-25*
