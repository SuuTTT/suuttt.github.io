---
title: "Reaching for Rank 1: The Parity Trap, Distilling the Champion, and What the Mahjong Winners Actually Do"
date: 2026-06-04
description: "After discovering that a CNN — not our MLP — was the real unlock, we pushed for rank-1-level play. We settled the architecture (a 40-block normalized ResNet), hit the classic RL 'parity trap' (self-play can't beat a strong supervised base), tried distilling the actual #1 Botzone bot, and then did deep research into how SOTA mahjong AI (Suphx) and the Botzone winners actually train. The research validated our hardest-won fixes (BatchNorm fusion for the deploy crash; KL-to-SL for RL) and handed us the conversion lever we'd been missing. This is the honest middle of the climb."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["research", "mahjong", "reinforcement-learning", "self-play", "distillation", "cnn", "game-ai", "botzone", "deployment"]
---

{{< katex >}}

## TL;DR

In the [last post]({{< relref "2026-06-03-the-wall-was-the-architecture-mahjong-cnn-breakthrough.md" >}}) a reproduced **CNN** crushed our heavily-tuned MLP and became the new base. Since then we tried to reach rank-1-level play and learned a lot — much of it humbling:

- **Architecture is settled.** A **40-block ResNet with BatchNorm** (`resbn40`) beats the 16-block CNN champion **+973 head-to-head**. Wider, deeper-still, CNN+attention, and a tile-graph GNN all *tie or lose*. Depth + normalization is the sweet spot; more architecture search has diminishing returns.
- **We hit the RL "parity trap."** Self-play PPO (learner vs a frozen copy of the strong base) converges to **parity** — it can't surpass an already-good supervised policy. This is a known failure mode, not a bug.
- **We can see the #1 bot but not beat it cheaply.** We reproduced and benchmarked against the rank-1 Botzone bot (chunjiandu, which is **SL + RL**); our policy already **agrees with ~68% of its discards**. Distilling its self-play games nudged us a marginal **+121** (within noise) — promising but data-starved.
- **Deep research closed the loop.** Studying Suphx and the Botzone/PKU winners *validated our two hardest fixes* — **BatchNorm fusion** (the exact cause + cure of our deploy crash) and **KL-to-SL regularization** (the cure for the parity trap) — and pointed at the conversion lever the champions use: **8-fan look-ahead masking**. We built it and tested it — and it came back *null* (the expert CNN already converts), which is its own useful lesson.

This post is the honest middle of the climb: what worked, what plateaued, and the research-driven plan to break parity.

---

## Recap: the CNN was the unlock

Our entire MLP lineage was strong only *relative to itself*; a standard supervised **CNN** (spatial `(38,4,9)` tile grid, ResNet backbone) beat our most-tuned model 0-of-60 and *converted* (built 8-fan hands) where the MLP only drew. The lesson — **question the representation before the recipe** — set up everything that followed.

## Part 1 — Settling the architecture

With a real cross-architecture benchmark in hand, we ran a fleet search over deeper/wider/hybrid variants, training each on the official data and ranking by head-to-head through the official judge:

| Variant | vs the 16-block CNN champ |
|---|---|
| **resbn40** (40 blocks + BatchNorm) | **+973 (52–25)** — new best |
| resbn56 (56 blocks) | +128 (tie) — deeper doesn't help |
| resbn24 (24 blocks) | +90 (tie) — lighter, same strength |
| wide (256ch) / CNN+attention | tie-or-below |
| tile-graph GNN | clearly worse (loses spatial info) |

**Verdict:** the *un-normalized* deep net that diverged earlier trains fine once you add BatchNorm, and **40×128 is the sweet spot.** Architecture is no longer the bottleneck.

## Part 2 — The RL parity trap

The obvious next step is RL fine-tuning (the #1 bot does it). We built a self-play simulator on the CNN's features (validated against the official judge — its self-play draw rate matched to within a percent), added a value head, and ran PPO with the learner playing two seats against a **frozen copy of the supervised base**.

It converged to **parity.** Across checkpoints, the learner hovered around even with the base — a slight positive lean buried in the noise. This is the textbook outcome: against one fixed opponent, the policy just learns that opponent's specific weaknesses and stops improving. Mahjong strategies are **non-transitive** (rock-paper-scissors-like), so a narrow opponent distribution yields a narrow, non-improving policy.

## Part 3 — Distilling the champion

If we can't *out-self-play* the base, can we *imitate* the actual #1 bot? We benchmarked against the rank-1 Botzone bot directly: our policy already **matches ~68% of its discards**. The remaining ~32% is largely the gap between supervised imitation and the champion's RL fine-tuning.

So we distilled: blend the champion's self-play decisions (upweighted) into the official data and fine-tune. With ~72 games (≈2,600 champion decisions) the result was **+121 over our base — within noise.** Distilling the #1 policy *is* a real shortcut (it clones its RL-improved play), but it's data-hungry; a few dozen games isn't enough.

## Part 4 — What the winners actually do

At this point we did deep research into SOTA mahjong AI (Microsoft **Suphx**) and the Botzone/PKU winners. Four findings mattered:

1. **The parity trap is real and named** — the cure is a **population league** (not a single frozen opponent): SL base + historical RL checkpoints + dedicated **exploiters**, sampled by Prioritized Fictitious Self-Play, plus a **KL-to-SL leash** (MPPO/BPPO) so the policy refines without drifting into exploitable aggression.
2. **Dense reward.** Mahjong's round score is high-variance; Suphx trains a **global reward predictor** Φ(s) and shapes the reward as r̃ₜ = Φ(sₜ) − Φ(sₜ₋₁).
3. **The 8-fan conversion lever.** Champions integrate the fan/shanten library into the policy loop and **mask any action that makes a ≥8-fan win unreachable** — directly attacking the "fast sub-8-fan tenpai scores zero → draw" ceiling.
4. **Oracle guiding & pMCPA** — train with perfect information then wean it off; adapt the policy to the dealt hand at run time.

Crucially, the research **named our exact deploy crash** ("exit code 120" from a BatchNorm model on the legacy torch-1.4 runtime) and its cure.

## Part 5 — Applying the research

- **Deploy fix (validated, shipped).** We **fold every Conv+BatchNorm into a single Conv** (mathematically exact in eval mode) and save with legacy serialization. The result is a BatchNorm-free model, numerically identical to resbn40, that loads on torch 1.4 — unblocking the **+973-stronger** model for the live bot.
- **RL, done right.** We replaced the single frozen opponent with a **20-model pool** (SL + learner snapshots) and added a **KL-to-SL penalty** (decaying), and made the rollout a **parallel actor-learner** (many CPU actors generating games into one GPU learner — about **60× faster**, and the design extends across machines). This directly targets the parity trap.
- **Conversion (tested, null).** We built the **8-fan look-ahead filter**: for the policy's top discard candidates, reject any that lock the hand into a dead-end tenpai (verified — it tells a 6-fan dead-end from a 45-fan flush). Head-to-head it was a **tie (+74 over 60 games, 29–29 wins, 3% draws)**. The reason is instructive: our supervised CNN *already* converts — its self-play draw rate is only ~3%, so it almost never walks into a dead-end the filter would catch. The conversion ceiling we feared is largely handled by a strong enough policy, not a separate rule. (Kept behind a flag, off by default — it also costs inference time.)

## Where we are, honestly

We have a **strong, deployable base** (resbn40, BatchNorm-fused for the live runtime) and a **research-grounded plan** to push past it. The parity-trap fixes (pool + KL, then league/exploiters + dense reward) are running or queued; the conversion filter tested null (the policy already converts); distillation is one good data-collection away from a real boost. Nothing here is a victory lap — it's the honest middle, with the difference that we now know *which* levers the winners actually pull.

The full prioritized plan lives in the repo's `doc/TODO.md`. Next post: whether pool+KL and 8-fan masking finally move us off parity — toward the rank-1 line.

## Lessons so far

- **Self-play alone won't beat a strong supervised base** — you need a *population* and a leash to the base.
- **A rival's open code + the literature beat blind compute** — they validated our two hardest fixes and pointed straight at the missing lever.
- **The deployment runtime is part of the model** — a BatchNorm layer and a serialization flag stood between a +973 model and a crash.
- **Distilling the champion is a real shortcut** — but it's only as good as the data you can collect.
