---
title: "RL for Structural Entropy Graph Clustering: MergeEnv, Behavior Cloning, and Why Greedy Is Hard to Beat"
date: 2026-05-14
description: "We build a Gymnasium MDP for H²-minimization merge sequences, train a GNN policy via behavior cloning from a lookahead planner, and fine-tune with PPO. Best result: 4/17 benchmark wins vs greedy — still far below Leiden's 45%. Here's what worked, what failed, and the surprising lesson from a critical bug."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["reinforcement-learning", "graph-clustering", "structural-entropy", "behavior-cloning", "ppo", "gnn"]
---

{{< katex >}}

> **In one paragraph:** Structural entropy \(H^2\) is a principled graph partition objective, but minimizing it over discrete merge sequences is hard — greedy works well, yet occasionally misses globally better solutions requiring non-greedy moves. We formulate this as a finite-horizon MDP ("MergeEnv"), train a 3-layer GAT policy via behavior cloning from a Monte Carlo lookahead expert, and fine-tune with PPO. On a 17-graph benchmark, the best policy wins 4/17 head-to-heads vs greedy. On 38 held-out instances, win rate is only 21% vs Leiden's 45%. The key lesson: the hardest part is not training — it's finding training data that actually generalizes.

---

## 1. The Problem: When Does Greedy Fail at Structural Entropy Minimization?

**Structural entropy** [Li & Pan, 2016] measures how well a partition \(\mathcal{P}\) compresses a random walk on graph \(G\). The second-order variant is:

$$H^2(G, \mathcal{P}) = -\sum_{j=1}^k \frac{g_j}{2m}\log_2\frac{V_j}{2m} - \sum_{j=1}^k\sum_{i \in C_j}\frac{d_i}{2m}\log_2\frac{d_i}{V_j}$$

where \(g_j = \text{cut}(C_j)\) is the edge cut, \(V_j = \text{vol}(C_j)\) is the total degree in cluster \(j\), and \(2m = \sum_i d_i\) is twice the edge count. Lower \(H^2\) means the partition better captures the graph's community structure.

The standard approach is **greedy descent**: starting from each node in its own cluster, repeatedly merge the pair of clusters that most reduces \(H^2\), until \(k\) clusters remain. This is \(O(n^2)\) but fast in practice and often finds excellent solutions.

**Where greedy struggles:** On graphs with a 3-level hierarchical structure (e.g., a planted SBM with fine, medium, and coarse community levels), the greedy merge considers all candidate pairs at each step and selects the one with the largest immediate \(\Delta H^2\). When the target \(k\) corresponds to the *coarsest* level, greedy may first merge all fine-to-medium boundaries (large immediate gain) before tackling medium-to-coarse merges — but those medium-level clusters are now incorrect from the coarse perspective, and the subsequent merges carry higher cost. A lookahead planner can recognize this trap.

**The question:** Can a neural policy *amortize* lookahead planning across a training distribution of hierarchical graphs, producing a cheap inference-time alternative to expensive Monte Carlo planning?

---

## 2. The MergeEnv: Structural Entropy as a Markov Decision Process

We implement a Gymnasium-compatible environment that frames \(H^2\)-minimization as a finite-horizon MDP.

### State and Action Space

At each step \(t\), the state is the current partition \(\mathcal{P}_t\) (encoded as a supernode graph). The action space \(\mathcal{A}_t\) is the top-\(M\) candidate merge pairs by \(|\Delta H^2|\) (default \(M=64\)). The episode terminates when the number of clusters reaches \(k_\text{target}\).

### Incremental Entropy Tracking

A naive implementation would recompute \(H^2\) from scratch after each merge — \(O(n)\) per step, \(O(n^2)\) per episode. Instead, we maintain per-cluster sufficient statistics:

$$(\text{vol}_C,\quad g_C,\quad S_C = \textstyle\sum_{v \in C} d_v \log_2 d_v)$$

Merging \((C_a, C_b)\) updates only these three scalars for the merged cluster and adjusts \(g\) for the new boundary. This enables **\(O(1)\) entropy delta computation** regardless of graph size.

### Reward Modes

Two reward modes were explored:

- **Dense**: \(r_t = \Delta H^2_t / H^2_0\) at each step. *Problem*: the greedy action always has the highest immediate reward, so any gradient-based method converges back to greedy. PPO collapses to greedy under dense reward within a few thousand steps.
- **Terminal**: \(r_t = 0\) for \(t < T\); at termination, \(r_T = (H^2_\text{greedy} - H^2_T) / H^2_0\). This rewards improvement over greedy and avoids the reward collapse. We use terminal reward for all final experiments.

---

## 3. The Policy: Graph Attention Network on the Supernode Graph

The policy \(\pi_\theta\) is a 3-layer GAT operating on the **supernode graph**: each current cluster is a supernode with 7-dimensional features:

$$\mathbf{x}_C = \bigl[\,g_C/2m,\; V_C/2m,\; |C|/n,\; H_\text{within}(C),\; d_C^\text{norm},\; \Delta_{\max},\; \Delta_{\min}\,\bigr]$$

The policy scores candidate pairs \((C_a, C_b)\) by concatenating supernode embeddings and passing through an MLP head. Action: \(a_t = \arg\max \text{score}(C_a, C_b)\).

**Architecture**: hidden dimension \(h=64\), 3 GAT layers, 2 attention heads, ~25k parameters. We also tested \(h=128\) (~100k parameters) — it performed *worse* (1/17 vs 3/17 wins on the fixed suite), suggesting the bottleneck is not model capacity but training data coverage.

---

## 4. Expert Trajectories: A Tale of Two Planners

### mpc-td Lookahead Planner

The expert is `seclust_target_k_lookahead` from the `glass-jax` library: Monte Carlo planning with multiple random restarts and local-search passes. Given a target partition \(\mathcal{P}^*\), the planner greedily moves toward it at each merge step.

The planner genuinely beats greedy only on **3-level hierarchical SBMs**:

| Graph family | mpc-td beats greedy? |
|---|---|
| Ring-of-cliques | No (greedy is optimal) |
| 2-level SBM | No (greedy is optimal) |
| 3-level hierarchical SBM | ✅ Yes — 24/43 trajectories improve |
| Karate club | No (greedy is optimal) |

This asymmetry shapes everything downstream.

### Oracle Trajectories: The Bug That Became a Feature

To expand the training distribution, we generated trajectories on **oracle graphs** — graphs where the ground-truth partition is known (ring-of-cliques, dense SBM, planted 3-level SBM, 2-level SBM, small exact). We called `trajectory_from_labels`, which uses the environment's candidate list to find merges consistent with the oracle partition.

**The critical bug:** Inside `trajectory_from_labels`, we compared candidate cluster IDs against labels from `env.current_labels()` — which returns **canonically renumbered** labels (0, 1, 2, ..., k-1 after each merge). But the environment's candidates use **raw internal IDs** (original node indices as supernodes). These IDs don't match. Result: `_valid_merge_mask` always returned all-False, and the trajectory generator silently fell back to `env.sample_greedy_action()` every step.

```python
# BUGGY: canonical labels ≠ raw candidate IDs
current_labels = env.current_labels()  

# FIXED: raw internal labels match candidate IDs  
current_labels = env._state.labels.copy()
```

**After the fix:** 97% of trajectory steps are multi-target (many valid within-oracle-community merges at each step). Before the fix: 0% multi-target — the system was silently generating pure greedy trajectories on oracle graphs.

**The irony:** `bc_oracle_v1` was trained on the *buggy* trajectories — greedy paths on diverse oracle graph families. It achieves **4/17 wins on the fixed suite** and **8/38 (21%) on 38 held-out instances**, beating all subsequent post-fix variants. Why? Because the policy is evaluated with greedy-style rollout at test time. Greedy training paths minimize distribution shift between training and evaluation. Oracle-within-community paths (from the fixed version) create a train/test mismatch when the policy ever deviates from the oracle path.

> **Key insight:** For imitation learning on discrete combinatorial problems, the *training trajectory distribution* matters as much as the *oracle trajectory distribution*. Training on expert paths that the policy can't reliably reproduce leads to compounding errors at test time.

---

## 5. Multi-Target Loss for Behavior Cloning

When the oracle partition is known, there may be *multiple valid merge actions* at each step (any merge within the correct community is fine — all produce valid progress toward the oracle). Standard cross-entropy BC would arbitrarily pick one.

We implement a **multi-target logsumexp loss**:

$$\mathcal{L} = -\log\sum_{a \in \mathcal{V}} \exp(\log\text{softmax}(\text{logits})_a)$$

where \(\mathcal{V}\) is the set of valid (within-oracle-community) candidate indices. This is equivalent to maximizing the probability of the *union* of valid actions, with a smooth interpolation between targets.

When \(|\mathcal{V}| = 1\) (unique valid action), this reduces to standard cross-entropy. When \(|\mathcal{V}|\) is large (many valid merges), the loss approaches zero — trivially easy to satisfy. This caused the "trivial loss" problem: with 97% multi-target steps and large valid sets, the loss collapsed to 0.034 within a few epochs without learning anything useful.

**Fix:** A `top_k_valid` parameter limits the valid mask to the top-\(k\) valid candidates by greedy preference. With `top_k_valid=1`, we recover standard BC on the greedy-best-within-oracle action — full multi-target is counterproductive here.

---

## 6. Results

### 6.1 Fixed 17-Graph Evaluation Suite

The suite covers: karate, ring-of-cliques (2 variants), SBM (2 variants), 3-level hierarchical SBM (family a/b/c × 2 seeds each), and 2-level hierarchical SBM (2 variants).

| Model | Description | 17-graph wins |
|---|---|---|
| greedy | Baseline | 0/17 (by definition) |
| Leiden (CPM) | Resolution-tuned | 3/17 |
| mpc-td (teacher) | Monte Carlo planner | **7/17** |
| bc7 | BC from mpc-td, h=64 | 2/17 |
| bc8 | BC from mpc-td, mpc-td-only data | 2/17 |
| bc8\_fixed | BC + ring/karate augment | 3/17 |
| bc\_oracle\_v1 | BC on greedy paths / oracle graph families | **4/17** |
| ppo\_terminal\_v2 | PPO fine-tune from bc7, terminal reward | **4/17** |

Selected per-graph results for `bc_oracle_v1` (4/17 wins):
```
hier3_s5:  +0.056 ✓   hier3_s7:  +0.003 ✓   hier3_s8:  +0.010 ✓
hier3_s10: +0.002 ✓   hier3_c_s8: −0.300 ✗   hier2_n90: −0.138 ✗
```

### 6.2 Multiseed Statistical Evaluation (38 instances)

Held-out seeds 8–12 for 9 graph families × 5 seeds each + karate (1) + ring (2):

| Family | N | Leiden | bc\_oracle\_v1 | bc8\_fixed |
|---|---|---|---|---|
| hier2\_n80 | 5 | **+0.009 (1/5)** | −0.067 (1/5) | −0.020 (0/5) |
| hier2\_n90 | 5 | **+0.019 (4/5)** | −0.118 (1/5) | −0.077 (0/5) |
| hier3\_a | 5 | −0.055 (2/5) | **−0.001 (3/5)** | +0.000 (2/5) |
| hier3\_b | 5 | 0.000 (0/5) | −0.058 (0/5) | −0.015 (0/5) |
| hier3\_c | 5 | −0.001 (2/5) | −0.070 (2/5) | −0.017 (2/5) |
| karate | 1 | −0.004 (0/1) | −0.026 (0/1) | −0.026 (0/1) |
| ring | 2 | 0.000 (0/2) | 0.000 (0/2) | 0.000 (0/2) |
| sbm\_n100 | 5 | **+0.017 (5/5)** | −0.011 (1/5) | +0.001 (2/5) |
| sbm\_n200 | 5 | +0.007 (3/5) | −0.012 (0/5) | −0.008 (0/5) |
| **OVERALL** | **38** | **17/38 (45%)** | **8/38 (21%)** | **6/38 (16%)** |

`bc_oracle_v1` beats `bc8_fixed` on hier3\_a (3/5 vs 2/5) but both policies fail hard on SBM — the training oracle SBMs used dense parameters (\(p_\text{in}=0.65\)–0.80) while eval uses weak SBMs (\(p_\text{in}=0.40\), \(p_\text{out}=0.03\)). Distribution mismatch again.

**Leiden at 45%** outperforms all learned policies by 2× on the multiseed suite, driven by SBM dominance (5/5 sbm\_n100, 4/5 hier2\_n90).

### 6.3 Composite Checkpoint Selection

Standard loss-based checkpoint selection would pick the epoch with lowest validation loss — but minimizing cross-entropy does not directly optimize the metric we care about (improvement over greedy on benchmark graphs). We instead evaluate the policy's rollout every 5 epochs and select the checkpoint maximizing a **composite score**:

$$\text{composite} = \Delta_{\text{hier3}} + \min(0,\; \Delta_\text{karate} + 0.1) + \min(0,\; \Delta_\text{ring} + 0.1)$$

where \(\Delta\) denotes improvement over greedy. The \(\min(0, \cdot + 0.1)\) terms penalize only *catastrophic* regressions (worse than −0.1 bits), acting as guards rather than targets. This prevented ring-of-cliques catastrophic failures (some seed-0 runs show ring Δ=−0.47 bits) from contaminating the saved checkpoint.

---

## 7. Failure Mode Analysis

### F1: Greedy Is Already Optimal on Ring and SBM

Ring-of-cliques: within-clique merges reduce \(H^2\) more than cross-clique merges by construction (density ratio). Greedy always selects within-ring merges first. mpc-td exhaustive search agrees. **Any policy that deviates gets penalized.** The BC policy must learn to reproduce greedy behavior exactly here.

2-level SBM with \(p_\text{in} \gg p_\text{out}\): the degree-weighted structure makes within-community merges unambiguously better at every step. mpc-td never beats greedy on SBM. Training on SBM data adds noise without signal.

### F2: Distribution Shift on hier2\_n90

The 6-community 2-level SBM (n=90) is the hardest benchmark graph: all BC variants show strongly negative Δ (−0.077 to −0.186). Leiden wins 4/5 seeds here. The 6-community structure requires accurate ordering of ~84 merge steps — any early error cascades into a wrong coarse partition.

The oracle dataset contains hier2 graphs but they're different random seeds than the eval graphs. The policy learns to navigate 6-community merges for training seeds but doesn't transfer to eval seeds.

### F3: PPO Entropy Collapse (Terminal Reward)

PPO from bc\_oracle\_v1 with terminal reward and a hier3-only pool completely collapses (0/17 wins), despite starting from the 4/17 checkpoint. Policy entropy grows to 3.75 (close to uniform over 64 actions) within 200k steps — the policy is exploring randomly rather than exploiting its BC-learned policy.

The issue: with a hier3-only training pool, the policy specializes to hier3 structure and loses its SBM/ring knowledge (catastrophic forgetting). The karate graph jumps from Δ=−0.026 to Δ=−0.421 and ring\_6x5 goes from Δ=0.000 to Δ=−0.250.

Successful PPO fine-tuning (ppo\_terminal\_v2, 4/17 wins) used the full graph pool including SBM, ring, and karate — diversity prevents forgetting.

### F4: The Composite Metric Ring Guard Gap

The composite checkpoint uses ring\_4x5 for the ring guard but not ring\_6x5. Some seeds learn to "sacrifice" ring\_6x5 performance in exchange for composite score. Seed=1 achieves composite=0.032 (best observed) but ring\_6x5 Δ=−0.467 (catastrophic) → only 1/17 final wins. Adding ring\_6x5 to the guard is a necessary fix.

---

## 8. Why Is This Hard: A Framework

Three properties of the merge MDP make it resistant to standard RL/BC:

### 8.1 Greedy Near-Optimality

On the majority of graph families (ring, SBM, karate — which constitute the bulk of the benchmark), greedy is globally optimal. The policy must learn to perfectly reproduce greedy behavior on these graphs while *deviating intelligently* on hier3 graphs. Distinguishing "greedy is optimal here" from "greedy will regret this later" requires fine-grained structural understanding that 25k parameters may not capture.

### 8.2 Compounding Errors over Long Horizons

A ring with 6 cliques of 5 nodes requires ~25 merge steps to reach \(k=5\). A hier3 graph with \(5 \times 4 \times 4 = 80\) nodes requires ~76 merge steps. Early errors in step 3 or 4 may not manifest as bad \(H^2\) until step 70. Both BC (open-loop replay) and PPO (sparse terminal reward) struggle with this horizon.

### 8.3 Training-Evaluation Distribution Shift

The core problem: the policy is evaluated on graphs from the same *family* as training graphs but with different random seeds. Same generative parameters, different random realizations. For most graph families, greedy's merge sequence on seed A looks completely different from seed B at the supernode feature level — the intermediate cluster shapes are seed-specific. A policy trained on 10 seeds of hier3 has seen 10 specific intermediate cluster trajectories, none of which appears in a new seed.

**Contrast with Leiden**: Leiden doesn't "learn" at all — it runs CPM resolution search on each graph independently. It always sees the actual graph, not a training distribution proxy. This is why a parameter-free classical algorithm at 0 parameters outperforms a 25k-parameter neural network by 2× on held-out instances.

---

## 9. What Does Work

Despite the limitations, several components produced genuine, reproducible improvements:

### Diverse Oracle Graph Families

`bc_oracle_v1` (4/17, 21% multiseed) beats `bc8_fixed` (3/17, 16% multiseed) primarily because of **graph family diversity**, not trajectory quality. Training on ring-of-cliques, dense SBM, 3-level hier SBM, 2-level hier SBM, and small exact graphs gives the policy a broader structural vocabulary. The accidental greedy paths (due to the label-ID bug) were actually the right training distribution.

### Terminal Reward PPO Fine-Tuning

Starting from a BC-initialized policy, PPO with terminal reward (4/17) adds 2 wins over the BC baseline (2/17). The key: terminal reward doesn't collapse to greedy, and the BC initialization provides a warm start that keeps policy entropy in a useful range (~3.4, not random) throughout training.

### Composite Checkpoint Guards

Without the karate/ring guard terms, some checkpoints show 4/17 wins but catastrophic regressions on ring graphs (Δ=−0.47). The composite metric successfully filters these unstable checkpoints.

---

## 10. Comparison to DeSE

DeSE [Zhang et al., 2025] is a concurrent deep-learning approach to structural entropy minimization. Key contrasts:

| Dimension | DeSE | Ours |
|---|---|---|
| SE formulation | Soft (continuous relaxation) | Hard (discrete partitions, exact \(H^2\)) |
| Input | Node attributes + adjacency | Adjacency only |
| Optimization | Gradient descent on cluster assignments | RL over discrete merge sequence |
| Cluster count \(k\) | Learned/fixed | Explicitly targeted by MDP |
| Inference | Single forward pass | Sequential merges (\(O(n)\) steps) |
| Generalization | Transductive (re-train per graph) | Inductive (fixed policy) |

DeSE's transductive, feature-rich setting is complementary — it targets attributed graphs where node features carry rich cluster signal. Our MDP framework targets structure-only clustering and explicitly learns *when* merges should deviate from greedy. A future integration could use DeSE's soft-assignment entropy as a reward-shaping signal within MergeEnv.

---

## 11. Open Questions

### Can DAgger Close the Teacher–Student Gap?

The teacher (mpc-td) achieves 7/17 wins; the best student (BC + PPO) achieves 4/17. The standard imitation learning explanation for this gap is **distribution shift**: BC trains on the teacher's state distribution, but at test time the policy visits different states. DAgger [Ross et al., 2011] directly addresses this by iteratively querying the teacher at states visited by the *current policy* and augmenting the training set. Applied here: run the current BC policy, collect all states it visits, query mpc-td for the best action at each, and re-train. Repeat.

### Is 25k Parameters Enough?

The architecture ablation (h=64 vs h=128) shows that simply scaling up does not help. But the right question may not be "more parameters" — it may be "better inductive biases." A policy that explicitly represents hierarchical community structure (e.g., a hierarchical graph pooling architecture) might generalize better than flat GAT layers.

### What Reward Shape Helps Without Collapsing?

Dense reward collapses to greedy. Terminal reward is too sparse. A **potential-based shaping** reward \(r_t = \Phi(\mathcal{P}_{t+1}) - \Phi(\mathcal{P}_t)\) where \(\Phi\) is the mpc-td value estimate would provide per-step signal without reward collapse (by the potential shaping theorem). But computing \(\Phi\) requires a value function trained on mpc-td rollouts, adding a critic training loop.

### Leiden as the Target, Not Greedy?

All BC training targets outperforming *greedy*, but Leiden already beats greedy at 45% win rate. A policy trained to imitate Leiden's actions (instead of mpc-td's actions) might transfer better, since Leiden is faster to compute at training time and more diverse.

---

## 12. Reproducibility Notes

All experiments use:
- **JAX_PLATFORMS=cpu** for deterministic JAX computation
- Fixed `torch.manual_seed` and `np.random.seed` in BC trainer (added in session 7)
- Composite checkpoint selection (not loss-based) to guard against ring regressions
- Eval always uses `compare_methods.py` with the 17-graph fixed suite for comparable numbers

Key files:
- `src/env/merge_env.py` — MergeEnv implementation
- `src/policy/gnn_policy.py` — ClusterPolicy (3-layer GAT, h=64)
- `src/trainer/bc.py` — BC trainer with multi-target loss and `--init_ckpt` warm start
- `src/trainer/ppo.py` — PPO fine-tuning with terminal reward
- `experiments/generate_expert_trajectories.py` — mpc-td trajectory generation (with canonical/raw label bug fixed)
- `experiments/generate_oracle_dataset.py` — oracle graph family trajectory generator
- `experiments/compare_methods.py` — 17-graph fixed eval suite
- `experiments/eval_multiseed.py` — 38-instance multiseed statistical eval

---

## 13. Conclusion

We present a complete pipeline for reinforcement-learning-based structural entropy minimization: a Gymnasium MDP (MergeEnv), expert trajectory collection from a Monte Carlo planner (mpc-td), behavior cloning with multi-target loss, and PPO fine-tuning with terminal reward. The best policy (bc\_oracle\_v1 / ppo\_terminal\_v2) achieves **4/17 wins on a fixed 17-graph suite** vs greedy, and **21% on 38 held-out instances** vs Leiden's 45%.

The most important finding is not about model architecture or reward design — it's about **training data generalization**. A policy trained on greedy-path trajectories from *diverse graph families* (oracle graphs) outperforms one trained on near-optimal trajectories from a narrower distribution. The accidentally-introduced bug that made all oracle trajectories fall back to greedy turned out to be the right training regime.

The gap between teacher (mpc-td: 7/17) and best student (4/17) suggests that standard BC + PPO is insufficient for this problem, and that DAgger-style iterative dataset aggregation or hierarchical architectural inductive biases are the most promising next steps.

---

## References

1. Li, A. and Pan, Y. (2016). Structural information and dynamical complexity of networks. *IEEE Transactions on Information Theory.*
2. Pan, Y. et al. (2021). Structural entropy guided graph hierarchical pooling. *ICML 2021.*
3. Traag, V., Waltman, L., and van Eck, N. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports.*
4. Ross, S., Gordon, G., and Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. *AISTATS.*
5. Schulman, J. et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
6. Zhang, J. et al. (2025). Unsupervised graph clustering with deep structural entropy. *KDD 2025*, arXiv:2505.14040.
7. Velickovic, P. et al. (2018). Graph attention networks. *ICLR 2018.*
8. Ho, J., Ermon, S. (2016). Generative adversarial imitation learning. *NeurIPS 2016.*

---

*Code: `src/` — MergeEnv, ClusterPolicy, BC trainer, PPO trainer.*
*Eval: `experiments/compare_methods.py` (17-graph suite), `experiments/eval_multiseed.py` (38-instance multiseed).*
*Best checkpoint: `checkpoints/bc_oracle_v1.pt` (4/17, 21% multiseed). Reproducible with fixed seeds throughout.*
