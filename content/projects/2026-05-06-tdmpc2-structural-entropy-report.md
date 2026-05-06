---
title: "TD-MPC2 + Structural Entropy: Experimental Report"
date: 2026-05-06
description: "Results and analysis of integrating a structural-entropy regularizer into TD-MPC2. One acrobot run shows +217.8 over baseline at 300k steps; broader evidence is mixed."
layout: "post"
showTableOfContents: true
---

## Executive Summary

This report uses only local artifacts under `logs/` and `tdmpc2/logs/`.

- TD-MPC2 was extended with a structural-entropy (SE) regularizer in the main training path and with a separate experimental 2D-SE branch.
- The strongest pure local SE signal is `logs/acrobot-swingup/1/vastai_iter8_acrobot_se_m1_steps400000/eval.csv`, which reaches `564.7` at `300k` steps. Relative to the local official TD-MPC2 mean at `300k` (`346.9`), that is `+217.8`. However, this local run is incomplete because the local `eval.csv` stops at `300k`, not `400k`.
- High-coefficient and hierarchical SE ablations on local acrobot logs are weak: `acrobot_se0.1_eval10k` ends at `30.5` and `acrobot_2dse_m2_eval10k` ends at `104.6` at `100k`, both below the local official mean `179.1` at `100k`.
- On `hopper-hop`, the longer SE runs do not beat the local official TD-MPC2 mean at `4M` steps: `hopper_1dse_4m_final=335.4` and `hopper_2dse_4m_final=340.1` vs official mean `449.2`.
- The mixed `SE+IB` sweeps on acrobot are promising, but they do not isolate SE because `ib_coef` is also nonzero.

Bottom line: the SE integration is real and functional in code, and one local pure-SE acrobot run looks promising, but the broader local evidence does not yet justify a strong claim that SE robustly improves TD-MPC2 across tasks.

## Results Artifacts

The repository contains both raw training logs and exported result CSVs.

Relevant result exports:

- `results/tdmpc2/`
  - local official TD-MPC2 benchmark curves
- `results/tdmpc2-codex/acrobot-swingup_abstraction_compile.csv`
  - exported curve for the iteration-6 acrobot compile-abstraction run
- `results/tdmpc2-codex/acrobot-swingup_abstraction_compile_compare.csv`
  - exported comparison against the official acrobot baseline
- `results/tdmpc2-codex/acrobot-swingup_se_abstraction_vastai_partial.csv`
  - local partial export of the pure SE acrobot run through `300k`
- `results/tdmpc2-codex/acrobot-swingup_se_abstraction_vastai_partial_compare.csv`
  - comparison file for the local partial SE export
- `results/tdmpc2-codex/acrobot-swingup_se_abstraction_vastai.csv`
  - later finalized `400k` export, but this was produced from synced remote data and is excluded from the main evidence base in this report

Important note:

- The old compare files in `results/tdmpc2-codex/` are not always consistent with the official means recomputed directly from `results/tdmpc2/*.csv`.
- For this report, the authoritative baseline values are the means recomputed directly from `results/tdmpc2/`, because that matches the current local benchmark tables and avoids mixing in stale exported comparisons.

## What TD-MPC2 Does

TD-MPC2 is a model-based reinforcement learning algorithm. It learns a latent world model and then plans actions in latent space with model predictive control.

At a high level:

1. The encoder maps observations to latent states.
2. The dynamics model predicts the next latent state from the current latent state and action.
3. Reward and value heads score imagined rollouts.
4. At action time, TD-MPC2 samples candidate action sequences, rolls them forward in the learned latent model, scores them, and executes the first action from a high-value sequence.
5. During training, it jointly optimizes consistency, reward, value, termination, and policy losses.

Relevant code:

- `tdmpc2/train.py`: standard training entrypoint.
- `tdmpc2/tdmpc2.py`: main agent, planning loop, and training update.
- `tdmpc2/train_2dse.py`: alternate entrypoint for the 2D-SE branch.
- `tdmpc2/tdmpc2_2dse.py`: alternate agent with optional 2D structural entropy.

## What Was Implemented

### Main SE integration

The main integration adds SE as an optional regularizer to the world-model loss.

Implemented pieces:

- `tdmpc2/config.yaml`
  - Adds `ib_coef: 0.0`
  - Adds `se_coef: 0.0`
- `tdmpc2/tdmpc2.py`
  - Adds `_latent_ib_loss(z)`
  - Adds `_structural_entropy_loss(zs)`
  - Stores latent rollout states `zs`
  - Computes `se_loss`
  - Adds `self.cfg.se_coef * se_loss` into `total_loss`
  - Logs `se_loss`

Loss structure:

```text
total_loss =
  consistency_coef * consistency_loss +
  reward_coef * reward_loss +
  termination_coef * termination_loss +
  value_coef * value_loss +
  ib_coef * ib_loss +
  se_coef * se_loss
```

The SE loss is computed on rollout latents already produced during TD-MPC2 training. It uses SimNorm symbol groups as soft modules, builds a soft transition-flow matrix across consecutive latent states, and minimizes a one-level structural entropy proxy over that flow.

### IB integration

The same code path also includes an information bottleneck (IB) regularizer.

Implemented pieces:

- `tdmpc2/config.yaml`
  - adds `ib_coef`
- `tdmpc2/tdmpc2.py`
  - adds `_latent_ib_loss(z)`
  - accumulates IB loss over rollout states
  - adds `self.cfg.ib_coef * ib_loss` into `total_loss`

So the repository is really exploring a family of latent regularizers:

- no regularizer
- IB only
- SE only
- IB + SE

### 2D-SE branch

There is also a separate experimental branch:

- `tdmpc2/train_2dse.py`
  - Uses `tdmpc2_2dse.TDMPC2` instead of the standard agent.
- `tdmpc2/tdmpc2_2dse.py`
  - Keeps the 1D SE path
  - Adds `_structural_entropy_2d_loss(zs)`
  - Switches between 1D and 2D SE using `se_2d`
  - Uses `num_super_modules` for a fixed hard grouping of modules into super-modules

This 2D branch is not just a config toggle in the main file; it is a separate experimental code path.

## How SE Is Integrated

The SE regularizer is integrated into the model-learning stage, not directly into the planner.

Mechanically:

1. During `_update`, TD-MPC2 already rolls latent states forward across the planning horizon.
2. Those latent states are saved in `zs`.
3. Each latent state is reshaped into SimNorm groups and averaged into a soft module assignment.
4. Consecutive assignments define a soft transition-flow matrix between modules.
5. Structural entropy is computed from module volumes and cuts.
6. The weighted SE penalty is added to the same `total_loss` that trains the world model.
7. The policy still trains from detached latent rollouts, so SE influences policy indirectly through the learned representation and dynamics.

Interpretation:

- Small `se_coef` encourages latent transitions to cluster into reusable symbolic modules.
- Large `se_coef` can over-regularize and collapse useful distinctions.
- The current implementation is a lightweight differentiable proxy, not a full replay-wide graph partitioner or hierarchical abstraction learner.

## Loss Details

### IB loss

Implementation:

```text
ib_loss(z) = mean over groups and batch of
             sum_i p_i (log p_i + log K)
```

where:

- `p_i` is the SimNorm probability of symbol `i` within a group
- `K = simnorm_dim`

This is the KL divergence from the current symbol distribution to a uniform distribution over `K` symbols:

```text
L_IB = KL(p || Uniform(K))
     = sum_i p_i log(p_i / (1/K))
     = sum_i p_i (log p_i + log K)
```

What it optimizes in simple words:

- If one symbol becomes too dominant, the loss grows.
- If the probabilities are spread more evenly, the loss shrinks.
- So IB pushes the latent code away from overly sharp, overconfident, symbol assignments.

Easy-world intuition:

- Imagine each latent group has `8` buckets.
- IB says: “don’t always dump everything into the same bucket.”
- It encourages the representation to use its symbol capacity more evenly.

What that can help with:

- avoids latent collapse into a few symbols
- encourages broader symbol usage
- can make the representation less brittle

What it can hurt:

- if too strong, it may force the model to stay too diffuse and not commit to useful distinctions

### SE loss

Implementation:

```text
flow = source^T target / N
volume[a] = incoming_flow[a] + outgoing_flow[a]
internal[a] = flow[a, a]
cut[a] = max(volume[a] - 2 * internal[a], 0)
L_SE = - sum_a (cut[a] / total_volume) * log(volume[a] / total_volume)
```

where:

- `source` and `target` are soft module assignments from consecutive latent states
- `flow[i,j]` is soft transition mass from module `i` to module `j`
- `volume[a]` measures how much transition traffic touches module `a`
- `internal[a]` is how much of that traffic stays inside the same module
- `cut[a]` is the boundary-crossing traffic for module `a`

What it optimizes in simple words:

- The loss is smaller when the latent transition graph can be described by a few high-traffic modules with relatively low boundary leakage.
- It encourages transitions to stay coherent within modules rather than constantly scattering across many modules.

Easy-world intuition:

- Imagine the latent world is a map of cities and roads.
- `volume` is how much traffic passes through each city.
- `internal` is traffic that stays inside the city.
- `cut` is traffic constantly leaving for other cities.
- SE prefers a map where traffic forms a few meaningful regions instead of messy everywhere-to-everywhere flow.

What that can help with:

- cleaner transition structure
- more reusable macro-states
- potentially easier planning in a smaller model

What it can hurt:

- if too strong, it may merge states that should stay distinct
- early in training, the rollout graph is noisy, so SE may push on unstable structure

### 2D SE loss

The 2D branch adds one more level:

- first, modules are grouped into a small number of super-modules
- then the loss includes both:
  - structure between super-modules
  - structure of modules inside each super-module

Simple intuition:

- 1D SE tries to organize “cities”
- 2D SE tries to organize both “districts” and “cities inside districts”

In this repository, the super-module grouping is fixed and hard-coded by index ranges rather than learned dynamically.

## Method For This Report

This report uses only:

- `logs/**/eval.csv`
- `tdmpc2/logs/**/eval.csv`
- local scripts and metadata files

It explicitly ignores synced remote result files under `remote_results/`.

For official TD-MPC2 comparisons, I used the local CSVs in `results/tdmpc2/` and averaged duplicate rows by `step` to recover the local official mean curve. The most relevant means are:

- `acrobot-swingup`: `100k=179.1`, `300k=346.9`, `4M=662.9`
- `hopper-hop`: `100k=13.6`, `4M=449.2`
- `dog-run`: `50k` is not present in the official CSV, so the dog-run smoke experiments are compared only against local project baselines, not the official benchmark curve.

More official baseline points used in the interpretations:

- `acrobot-swingup`
  - `0=5.2`, `100k=179.1`, `200k=296.3`, `300k=346.9`, `400k=322.0`
  - later rise: `700k=466.1`, `800k=506.7`, `1M=517.9`, `4M=662.9`
- `hopper-hop`
  - `0=0.0`, `100k=13.6`, `200k=166.6`, `300k=259.7`, `400k=285.4`
  - later rise: `600k=357.8`, `1M=337.7`, `2M=399.0`, `3.6M=453.2`, `4M=449.2`

## Local Experiment Inventory

### A. Dog-run project experiments in `logs/`

These are the earlier non-SE project experiments that led into the acrobot work.

| Experiment | Local file | Setup | Final result | Interpretation |
| --- | --- | --- | --- | --- |
| `default` | `logs/dog-run/1/default/eval.csv` | `model_size=5`, very short smoke baseline | `7.4 @ 0` | Startup artifact / smoke only |
| `iter1_steps2000` | `logs/dog-run/1/iter1_steps2000/eval.csv` | `model_size=5`, `2000` steps | `7.4 @ 0` | No improvement over smoke |
| `iter2_abstraction_m1_steps50000` | `logs/dog-run/1/iter2_abstraction_m1_steps50000/eval.csv` | `model_size=1`, `50k` steps | `58.0 @ 50k` | Stronger than the matched original comparator |
| `iter3_original_m5_steps50000` | `logs/dog-run/1/iter3_original_m5_steps50000/eval.csv` | `model_size=5`, `50k` steps | `35.4 @ 50k` | Worse than the abstraction probe |
| `iter5_fastplanner_m1_steps50000` | `logs/dog-run/1/iter5_fastplanner_m1_steps50000/eval.csv` | `model_size=1`, faster planner settings | `44.8 @ 50k` | Competitive with the abstraction path but below `iter2` |
| `debug_m1_steps100` | `logs/dog-run/1/debug_m1_steps100/eval.csv` | debug / smoke | `5.3 @ 0` | Debug only |

Dog-run takeaway:

- The local project work before SE already showed that small-model abstraction-style variants could be competitive.
- These runs provide context, but they do not test structural entropy directly.

### B. Acrobot and IB/SE project experiments in `logs/`

| Experiment | Local file | Setup | Final result | Comparison / note |
| --- | --- | --- | --- | --- |
| `compile_smoke_acrobot` | `logs/acrobot-swingup/1/compile_smoke_acrobot/eval.csv` | acrobot smoke | `15.8 @ 0` | Smoke only |
| `iter6_acrobot_compile_abstraction_m1_steps4000000` | `logs/acrobot-swingup/1/iter6_acrobot_compile_abstraction_m1_steps4000000/eval.csv` | `model_size=1`, `4M` steps, compile-abstraction path | `643.4 @ 4M`, best `730.9 @ 3M` | Below local official mean `662.9` at `4M`, but reasonably close |
| `iter7_acrobot_ib_m1_steps400000` | `logs/acrobot-swingup/1/iter7_acrobot_ib_m1_steps400000/eval.csv` | `ib_coef=0.01`, `model_size=1`, target `400k` | `483.4 @ 200k` | Partial local run only; no local `300k` or `400k` point |
| `vastai_iter8_acrobot_se_m1_steps400000` | `logs/acrobot-swingup/1/vastai_iter8_acrobot_se_m1_steps400000/eval.csv` | `se_coef=0.01`, `model_size=1`, target `400k` | `564.7 @ 300k` | Strong pure local SE signal, but local artifact is incomplete |

Acrobot takeaway from `logs/`:

- The pure local SE run is the strongest local SE result in the repository.
- The local file only contains steps `0, 100k, 200k, 300k`, so any claim about `400k` must not use this file alone.
- The exported partial comparison file in `results/tdmpc2-codex/acrobot-swingup_se_abstraction_vastai_partial_compare.csv` agrees with this interpretation: the local run is clearly ahead through `300k`.

### C. SE-focused experiments in `tdmpc2/logs/acrobot-swingup/1/`

These are the densest local SE ablations.

| Experiment | Setup | Final result | Best result | Versus local official mean |
| --- | --- | --- | --- | --- |
| `acrobot_se0.1_eval10k` | `se_coef=0.1`, `ib_coef=0.01`, eval every `10k` | `30.5 @ 100k` | `79.4 @ 80k` | Well below official mean `179.1 @ 100k` |
| `acrobot_2dse_m2_eval10k` | `se_coef=0.1`, `ib_coef=0.01`, `se_2d=true`, `num_super_modules=2` | `104.6 @ 100k` | `161.8 @ 80k` | Still below official mean `179.1 @ 100k` |
| `acrobot_se0.01_ib0.01` | `se_coef=0.01`, `ib_coef=0.01` | `8.1 @ 0` | `8.1 @ 0` | Aborted / not informative |
| `acrobot_se0.01_ib0.01_sweep_final` | `se_coef=0.01`, `ib_coef=0.01` | `337.9 @ 100k` | `357.8 @ 80k` | Above official mean `179.1 @ 100k` |
| `acrobot_se0.01_ib0.05` | `se_coef=0.01`, `ib_coef=0.05` | `8.1 @ 0` | `8.1 @ 0` | Aborted / not informative |
| `acrobot_se0.01_ib0.05_sweep_final` | `se_coef=0.01`, `ib_coef=0.05` | `255.5 @ 100k` | `398.1 @ 90k` | Above official mean `179.1 @ 100k` |
| `acrobot_se0.01_ib0.1` | `se_coef=0.01`, `ib_coef=0.1` | `8.1 @ 0` | `8.1 @ 0` | Aborted / not informative |
| `acrobot_se0.01_ib0.1_sweep_final` | `se_coef=0.01`, `ib_coef=0.1` | `293.3 @ 100k` | `340.7 @ 80k` | Above official mean `179.1 @ 100k` |
| `default` | local artifact with only `step 0` | `9.1 @ 0` | `9.1 @ 0` | Not useful for comparison |

Acrobot SE takeaway from `tdmpc2/logs/`:

- Pure high-coefficient SE is bad locally.
- Early 2D SE is better than high-coefficient 1D SE, but still not good enough.
- Mixed `SE+IB` runs can be quite strong, especially `acrobot_se0.01_ib0.01_sweep_final`, but those runs are confounded and should not be presented as SE-only evidence.

### D. Hopper SE experiments in `tdmpc2/logs/hopper-hop/1/`

| Experiment | Setup | Final result | Best result | Versus local official mean |
| --- | --- | --- | --- | --- |
| `hopper_1dse_4m_final` | `se_coef=0.1`, `ib_coef=0.1`, 1D SE | `335.4 @ 4M` | `357.5 @ 3.75M` | Below official mean `449.2 @ 4M` |
| `hopper_2dse_4m_final` | `se_coef=0.1`, `ib_coef=0.1`, `se_2d=true`, `num_super_modules=2` | `340.1 @ 4M` | `352.3 @ 3.6M` | Below official mean `449.2 @ 4M` |

Hopper takeaway:

- The 2D branch finishes slightly above the 1D branch at the final step.
- Neither branch beats the local official mean curve.
- Hopper does not currently support a strong “SE helps” claim.

## Task-by-Task Baseline Interpretation

### Acrobot

The local official TD-MPC2 mean for `acrobot-swingup` grows quickly by `300k`, dips a bit at `400k`, and then keeps improving substantially after that, reaching `662.9` by `4M`.

What the local SE evidence says:

- The pure local SE run is very strong through `300k`.
- The longer compile-abstraction run is competitive at `4M`, but not clearly better than the official mean.
- So on acrobot, SE looks most convincing as an early-learning or mid-training representation aid, not yet as a proven better final asymptotic solution.

### Hopper

The local official TD-MPC2 mean for `hopper-hop` is different from acrobot:

- it rises steadily early
- it keeps finding additional gains much later
- the strongest official improvements happen well after `400k`

The local 1D and 2D SE hopper runs behave differently:

- they improve early from near zero to the low/mid-300s
- then they hover in that range for a long time
- they do not get the later-stage lift that the official baseline gets between roughly `300k-400k` and the final multi-million-step regime

Why might original TD-MPC2 keep rising while SE plateaus?

Most likely explanation:

- original TD-MPC2 preserves a more flexible latent representation, so later in training it can keep refining fine-grained distinctions in contact dynamics, hopping rhythm, and recovery behavior
- SE encourages compressible transition structure, which may help organize the latent space early but may also over-merge states that need to stay distinct for late-stage control improvement
- on hopper, the later gains may depend on subtle distinctions between “similar but not identical” states; SE may smooth those away

In easy words:

- baseline TD-MPC2 keeps a more detailed map, so it can discover extra late tricks
- SE makes the map cleaner and simpler, but maybe too simple, so progress flattens earlier

Why 1D and 2D SE both stay near the same range:

- both runs use fairly strong regularization: `se_coef=0.1` and `ib_coef=0.1`
- the 2D hierarchy changes organization, but it does not remove the basic pressure toward compression
- if the main issue is over-compression, both 1D and 2D variants can plateau similarly

So the current hopper evidence suggests:

- SE is not helping late-stage asymptotic performance here
- the coefficient is probably too large for this task
- the latent abstraction pressure may be arriving too early or too strongly

## SE-Specific Interpretation

### What looks positive locally

- `logs/acrobot-swingup/1/vastai_iter8_acrobot_se_m1_steps400000/eval.csv`
  - Pure SE, `se_coef=0.01`
  - Reaches `374.9 @ 100k`, `390.6 @ 200k`, `564.7 @ 300k`
  - Beats the local official acrobot mean at all available local points
  - Best evidence that the SE idea can help

### What looks negative locally

- `tdmpc2/logs/acrobot-swingup/1/acrobot_se0.1_eval10k/eval.csv`
  - Strong sign that `se_coef=0.1` is too aggressive on acrobot
- `tdmpc2/logs/acrobot-swingup/1/acrobot_2dse_m2_eval10k/eval.csv`
  - 2D SE did not rescue the weak high-coefficient setting
- `tdmpc2/logs/hopper-hop/1/hopper_1dse_4m_final/eval.csv`
- `tdmpc2/logs/hopper-hop/1/hopper_2dse_4m_final/eval.csv`
  - Longer-horizon hopper evidence is negative relative to the local official mean

### What is promising but confounded

- `acrobot_se0.01_ib0.01_sweep_final`
- `acrobot_se0.01_ib0.05_sweep_final`
- `acrobot_se0.01_ib0.1_sweep_final`

These all mix SE and IB, so they show that the combined regularization family can work, but they do not tell us how much of the gain comes from SE alone.

## Main Conclusions

### Code conclusion

SE is genuinely integrated into this repository.

- The main TD-MPC2 code path supports 1D SE through `se_coef`.
- There is a separate experimental 2D-SE path with `se_2d` and `num_super_modules`.
- The implementation is lightweight and differentiable, and it regularizes the latent world model rather than the planner directly.

### Experiment conclusion

Using only local logs:

- There is one strong pure-SE acrobot result, but the local artifact is incomplete and stops at `300k`.
- High-coefficient SE is locally bad on acrobot.
- 2D SE does not clearly improve on the simpler setup.
- Hopper local results are negative relative to the official mean baseline.
- The strongest acrobot sweep results come from mixed `SE+IB`, not SE alone.

### Practical conclusion

The fairest current statement is:

> Structural entropy has been successfully implemented in TD-MPC2 and shows a promising local pure-SE signal on acrobot at small coefficient, but local evidence does not yet show a stable, cross-task improvement over baseline TD-MPC2.

## Recommended Next Claims

Claims that are supported by the local repo:

- “We implemented an SE regularizer for TD-MPC2 latent transition flow.”
- “A local pure-SE acrobot run with `se_coef=0.01` shows strong intermediate performance through `300k` steps.”
- “SE behavior is coefficient-sensitive.”
- “2D SE and hopper results do not currently demonstrate a robust advantage.”
- “IB and SE regularize the latent symbols in different ways: IB encourages balanced symbol usage, while SE encourages coherent transition structure.”

Claims that are not yet supported by local repo evidence:

- “SE improves TD-MPC2 overall.”
- “2D SE outperforms 1D SE.”
- “SE consistently beats the official TD-MPC2 benchmark across tasks.”
