---
title: "SIDM in Practice: A Distill-Style Guide from Paper to Code"
date: 2026-04-22
description: "A reader-friendly, implementation-grounded introduction to the SIDM framework from the original paper to the public code."
layout: "post"
showTableOfContents: true
---

> A reader-friendly, implementation-grounded introduction to the SIDM framework from the original paper to the public code.

---

## TL;DR

**SIDM (Structural Information-based Decision Making)** is proposed in the paper **“Hierarchical Decision Making Based on Structural Information Principles”** ([paper link](https://arxiv.org/html/2404.09760v2)).

The paper presents a unified abstraction idea; the code implements this in separate tracks:
- **SISA** (state abstraction) built on **RAD/CURL + SAC** ([RAD](https://github.com/MishaLaskin/rad), [CURL](https://github.com/MishaLaskin/curl)).
- **SISL** (skill learning) built on a **ReSkill-style hierarchical pipeline** ([ReSkill reference](https://github.com/krishanrana/reskill)).
- **SIRD** (role side, not the deep focus of this blog).

Core takeaway:
- **Paper**: one conceptual SI framework.
- **Code**: baseline-integrated engineering variants.

---

## 1) What the original paper proposes

The SIDM paper ([arXiv HTML](https://arxiv.org/html/2404.09760v2)) proposes using **Structural Information (SI)** / structural entropy as a general principle to:
1. extract compact, meaningful abstraction structures,
2. model transition structure between abstractions,
3. improve hierarchical decision learning.

In plain terms, SIDM tries to answer:
> “Can we make RL easier by learning better structure first, then learning control on top of that structure?”

### 1.1 Method-section description (paper view, simplified)

To keep this blog implementation-focused, we summarize the paper’s method section at the workflow level and treat the structural-entropy math blocks as a **graph clustering method**.

From that lens, the method section can be read as:

1. **Build a graph from trajectories**
   - nodes represent states (or abstract units),
   - edges represent transition relationships (with empirical weights/probabilities).

2. **Run graph clustering / hierarchy extraction**
   - use the paper’s SI machinery to partition/merge nodes,
   - form higher-level abstract states (communities) and their transition structure.

3. **Use abstract structure to define higher-level decisions**
   - in state abstraction settings: regularize representation learning with abstract transition/action/reward structure,
   - in skill settings: use clustered transition structure to support skill-level decision variables and context.

4. **Train policy/policies on top of abstracted structure**
   - low-level control still optimizes RL objectives,
   - high-level abstractions guide what latent features, contexts, or options the controller uses.

5. **Iterate**
   - new data updates graph structure estimates,
   - policy learning and abstraction learning co-evolve.

This “graph clustering -> abstraction -> policy optimization” view is the most practical bridge between paper method and repository implementation.

### 1.2 Key theory & formulas (paper ideas, intuitive version)

Below are the main theory ingredients you need to read the paper quickly, with simple interpretations.

#### (a) RL objective (control goal)

\[
J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T}\gamma^t r_t\right]
\]

Meaning:
- maximize expected discounted return,
- all SIDM abstractions are in service of improving this base control objective.

#### (b) Empirical transition graph from trajectories

Let \(N_{ij}\) be how many times we observe transition \(i \to j\) in data.

\[
\hat{P}(j \mid i) = \frac{N_{ij}}{\sum_k N_{ik}}
\]

Meaning:
- build a directed graph from trajectory counts,
- normalize outgoing counts to get transition probabilities.

This is the practical bridge from raw trajectories to graph-structured abstraction.

#### (c) Graph clustering / abstraction mapping

Define an assignment \(z_i \in \{1,\dots,K\}\) that maps original state node \(i\) to abstract cluster/community \(z_i\).

\[
\phi(i) = z_i
\]

Meaning:
- \(\phi\) is the abstraction map,
- SIDM methods differ mainly in how they estimate/refine this map from graph structure.

#### (d) Aggregate transition between abstract states

For abstract groups \(a,b\), aggregate transition mass from members:

\[
\hat{P}_{ab} \propto \sum_{i:\phi(i)=a}\sum_{j:\phi(j)=b} \hat{P}(j\mid i)
\]

Meaning:
- collapse a large fine-grained graph into a smaller abstract graph,
- this gives a higher-level dynamics view used for skills/contexts/losses.

#### (e) Objective decomposition intuition

Paper-level intuition can be summarized as:

\[
\text{Total objective} \approx \text{RL control loss} + \lambda \cdot \text{structure/abstraction regularization}
\]

Meaning:
- keep solving RL,
- but bias representation/policy learning with graph-structure signals.
- In code terms: this appears as extra SI losses (SISA) or SI-conditioned context channels (SISL).

#### (f) Why this helps (intuitive)

If two states have similar roles in transition structure, graph clustering maps them to similar abstractions.
Then policy learning sees a simpler, more stable decision space:
- less sensitivity to noisy local variation,
- better reuse across similar situations,
- easier hierarchical control.

---

## 2) One framework, two concrete code paths

Repository: [SIDM codebase](https://github.com/SELGroup/SIDM).

In practice, the implementation splits by baseline/task ecosystem:

- **SISA**: SI as a representation-shaping branch inside a RAD/CURL-style SAC training loop.
- **SISL**: SI as a context/goal abstraction module injected into a ReSkill-style hierarchy (SkillVAE + prior + PPO).

---

## 3) Method pipeline figure (high-level)

```text
                +---------------------------------------------+
                |         SIDM Conceptual Pipeline            |
                +---------------------------------------------+
                    data/trajectories/states/actions
                                |
                                v
                  +------------------------------+
                  |   [SI] Structure Extraction  |
                  |   (graphs, partitions, trees)|
                  +------------------------------+
                                |
                                v
                  +------------------------------+
                  | [SI] Abstract Representation |
                  |   (goal/context/relations)   |
                  +------------------------------+
                                |
                                v
                  +------------------------------+
                  |  Baseline RL Backbone        |
                  |  (SAC or Hierarchical PPO)   |
                  +------------------------------+
                                |
                                v
                            policy update
```

`[SI]` marks components proposed/added by SIDM-style integration.

---

## 4) SISA track (RAD/CURL + SI) in depth

## 4.1 What SISA is essentially doing

**Essence:** SISA is **model-free SAC control + SI-driven latent shaping**.
It does not replace the control algorithm; it regularizes the representation with structure-aware losses.

## 4.2 SISA pipeline figure

```text
obs -> encoder -> actor/critic (SAC) ---------------------> action
        |
        +--> [SI] pretrain loss (inverse/contrastive/smoothness)
        +--> [SI] finetune loss (clustering KL)
        +--> [SI] abstract loss (transition/action/reward graphs)
```

## 4.3 SISA phase schedule

- frequent base SI-pretrain updates,
- periodic SI updates switch from `finetune` (early) to `abstract` (later).

This is a staged multi-objective curriculum over shared encoder parameters.

## 4.4 Graph building in SISA (with concrete mini-example)

Each `abstract_sisa` call builds partition-level graphs from the current minibatch pairings.

Suppose a minibatch yields 4 transition pairs after partition mapping:
- Pair1: `P0 -> P1`, action=0.2, reward=1.0
- Pair2: `P0 -> P2`, action=0.4, reward=0.0
- Pair3: `P1 -> P2`, action=0.1, reward=0.5
- Pair4: `P0 -> P1`, action=0.3, reward=-0.2

Then:
- **Transition graph counts**
  - `P0->P1`: 2, `P0->P2`: 1, `P1->P2`: 1
  - source-normalized: from `P0`, weights become `2/3` and `1/3`.
- **Action graph weights**
  - edge weights from action values per pair (implementation-specific projection).
- **Reward graph weights**
  - only positive rewards retained (`1.0`, `0.5`), negative/zero ignored.

These graphs are rebuilt every call (batch-local, online estimates).

---

## 5) SISL track (ReSkill-style + SI) in depth

## 5.1 What SISL is essentially doing

**Essence:** SISL is **hierarchical skill RL with SI-augmented context features**.
It mostly keeps the ReSkill control architecture, but changes what information the modules consume.

## 5.2 SISL pipeline figure

```text
Stage A: demos ----------------------------------------------+
                                                              |
Stage B: Skill training                                       v
obs/actions -> SkillVAE + skill prior <- [SI] goal/context (from graphs)
                                                              |
Stage C: Hierarchical RL                                      v
high-level policy ----> latent skill ----> decoder action ----+--> env
                           ^                    |
                           |                    +--> residual policy refinement
                           +------ [SI] goal/context conditioning
```

`[SI]` marks SIDM-added abstraction path.

## 5.3 Graph building in SISL (with concrete mini-example)

SISL graph construction has two layers:

1. **Undirected similarity graph** over flattened observation vectors.
2. **Directed community transition graph** over adjacent timesteps.

Mini-example:
- Trajectory communities across time: `[C0, C1, C1, C2]`
- Adjacent transitions counted:
  - `C0->C1` (+1)
  - `C1->C1` (+1)
  - `C1->C2` (+1)
- With many trajectories, counts aggregate and become weighted directed edges.
- If SCCs are disconnected, small bridging edges are added to enforce strong connectivity.

Output is a structural community representation used as `goal_state`/context feature, not a planner rollout model.

---

## 6) Is SIDM code model-based RL?

Short answer: **not in the standard planning/imagined-rollout sense.**

Why people ask this:
- SIDM builds explicit structural graphs from transitions, which resembles “world understanding”.

Why it is usually still considered model-free + model-informed:
- no explicit MPC/tree-search/planning over learned dynamics,
- no imagined trajectory rollout for policy improvement loop,
- RL backbones still update from real sampled transitions/rollouts.

So a better label is:
- **structure-regularized model-free RL** or
- **model-informed representation learning for RL**.

---

## 7) Reproducible specification (appendix)

## A. SISA reproducible spec

Inputs:
- replay transitions `(obs, action, reward, next_obs, done)`.

Modules:
- SAC actor/critic/encoder,
- SI branch with pretrain/finetune/abstract losses.

Per update:
1. sample replay minibatch,
2. update SAC critic (and actor/alpha on schedule),
3. run SI branch by phase schedule,
4. in abstract phase, rebuild partition-level transition/action/reward graphs from minibatch pair mapping,
5. backprop combined loss to encoder/SI head.

Inference:
- actor path for action selection; SI affects policy through trained representation.

## B. SISL reproducible spec

Inputs:
- demo trajectories + online env rollouts.

Modules:
- SkillVAE, skill prior (flow), high-level PPO, residual PPO,
- SI modules (undirected abstraction + directed structural model).

Stage B (skill training):
1. compute SI graph abstractions from observation batches,
2. derive `goal_state`,
3. train SkillVAE/prior on `[obs, goal_state]`-conditioned inputs.

Stage C (hierarchical RL):
1. periodically recompute SI `goal_state`,
2. condition high-level and residual decisions on SI context,
3. execute environment steps and PPO updates.

---

## 8) Concise conclusions: what each method is *really* doing

- **SISA (one-line view):**
  > “SAC with SI-based latent-space shaping and graph-regularized feature engineering.”

- **SISL (one-line view):**
  > “ReSkill-style hierarchy with SI-derived context features injected into skill learning and control.”

- **Framework-level view:**
  > “SIDM is less a single runtime algorithm and more a transferable structural-abstraction principle applied to different RL backbones.”

---

## References

- SIDM paper: https://arxiv.org/html/2404.09760v2
- SIDM code: https://github.com/SELGroup/SIDM
- RAD: https://github.com/MishaLaskin/rad
- CURL: https://github.com/MishaLaskin/curl
- ReSkill reference: https://github.com/krishanrana/reskill
