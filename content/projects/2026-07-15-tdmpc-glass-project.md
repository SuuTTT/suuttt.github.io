---
title: "TD-MPC-Glass — When Is a World Model's Abstraction Worth It? (project log)"
date: 2026-07-15
description: "A multi-month field report on abstraction and planning in model-based RL (TD-MPC2). Detailed lab-notebook lives on the project page; this is the milestone index."
layout: "post"
tags: ["world-models", "TD-MPC2", "abstraction", "JEPA", "reproducibility"]
---

**The full, detailed lab notebook (Parts 1–19+) now lives on the dedicated project page:
[suuttt.github.io/tdmpc-glass](https://suuttt.github.io/tdmpc-glass/).** This page keeps only the milestones.

## What the project found

- **Abstraction is a speed lever, not a capacity lever.** Across DMControl + Panda, explicit structure
  (consistency, anti-collapse, SE/glass, analytic controllers) buys *sample-efficiency where the prior fits*,
  not a higher ceiling versus a budget-matched baseline.
- **A value-sufficiency instrument (VBN).** Bottlenecking the value head fingerprints each task by how much
  of the latent the value function needs — three shapes (monotone / flat-high / ramp) that predict where
  structure has room to help.
- **A world model can be *harmful*.** Under planner-collection, removing the world model *helps by 45%* on
  CheetahRun (an inversion), while it's load-bearing on Walker — one variance-inflation mechanism, two regimes.
- **Anti-collapse regularizers are redundant** for a value-based world model (uniformity ≈ VICReg ≈ vanilla).

## Papers (drafts)

Three write-ups in [github.com/SuuTTT/wm-redundancy-paper](https://github.com/SuuTTT/wm-redundancy-paper):
*When Is Explicit Abstraction Redundant for a World Model?*, *The Anatomy of "Beating PPO"*, and
*Abstraction is a Speed-of-Learning Lever, not a Capacity Lever*.

→ **Read the week-by-week detail at [the project page](https://suuttt.github.io/tdmpc-glass/).**
