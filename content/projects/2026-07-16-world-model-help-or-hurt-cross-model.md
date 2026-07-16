---
title: "When Does a Learned World Model Help — or Hurt? (cross-model result)"
date: 2026-07-16
description: "The value of a learned world model is task-dependent, and it's the same tasks across TD-MPC2 and Dreamer. A milestone from the world-model-redundancy campaign."
layout: "post"
tags: ["world-models", "TD-MPC2", "Dreamer", "model-based-RL", "diagnostics"]
---

**Short version:** whether a learned world model *helps* a planner or *hurts* it is a
property of the **task**, not of the world-model architecture. We now see the *same*
tasks flip load-bearing vs redundant across two independent WM families — TD-MPC2
(latent-consistency) and Dreamer (reconstruction RSSM).

## The result

We ablate the world model (strip its forward-dynamics learning) and measure the change
in return, per task, in each framework:

| task | TD-MPC2 | Dreamer |
|---|---|---|
| **cheetah-run** | stripping the WM under planner-collection *helps* +45% (the "inversion") | vanilla **beats** stripped by **+9.7%** (WM helps) |
| **walker-run** | **null** — imposed structure is redundant | vanilla ≈ stripped (**null**) |

Cheetah is the task where the value function needs a large fraction of the latent
(low compressibility); walker is value-sufficient (a tiny bottleneck already recovers
most of the return). The world model matters on the former and is redundant on the
latter — **and this ordering is identical in both frameworks.**

## Why it matters

Most "does a world model help" debates are argued at the level of *architectures*. This
says the more useful question is at the level of *tasks*: a cheap, checkpoint-time
probe (a value-sufficiency bottleneck) tells you, per task, whether the learned WM has
room to help — before you spend the compute. And on the tasks where the value head is
already sufficient, the WM is not just neutral: under planner-collection it can inflate
the planner's own target variance (~3×) and actively destabilize learning.

## What's next

The diagnostic is prescriptive, not just descriptive: it points at a **gated world
model** — one that measures its own value-sufficiency and rollout variance and
*down-weights itself* on the tasks (and states) where it would otherwise hurt.

The detailed lab notebook lives on the [project page](https://suuttt.github.io/tdmpc-glass/);
the underlying numbers and commit hashes are in the campaign's public issues and ledger.
