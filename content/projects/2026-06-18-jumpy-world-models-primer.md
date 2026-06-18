---
title: "Jumpy World Models from Scratch: γ-Models, TD-Flow, and Compositional Planning (vs Dreamer & TD-MPC)"
date: 2026-06-18
description: "A from-scratch primer for newcomers on world models that don't predict one step at a time but JUMP straight to far-future states. Every term and equation built up: what a world model is, the one-step view (Dreamer, TD-MPC2) and its compounding-error problem, the discounted future-state (occupancy/successor) measure and the 'geometric horizon', γ-models / Geometric Horizon Models (Janner 2020), flow matching, Temporal-Difference Flows (TD-Flow, ICML 2025), and Compositional Planning with Jumpy World Models (CompPlan, ICLR-WS 2026) — then a side-by-side comparison with Dreamer and TD-MPC2."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["world-models", "tdmpc2", "dreamer", "flow-matching", "successor-measure", "td-flow", "reinforcement-learning", "planning", "tutorial", "ogbench"]
---

{{< katex >}}

> The one technique that genuinely *won* in my TD-MPC2 campaign was temporal abstraction — predicting
> the future in big jumps instead of small steps. So I'm reproducing the 2025–2026 "jumpy world model"
> line. This is the primer I wish I'd had: every term and equation from scratch, and how it differs from
> the world models you may already know (Dreamer, TD-MPC2). No background assumed beyond "an agent picks
> actions to maximize reward."

## 1. What is a world model, and why plan with one?

An agent lives in a Markov decision process: states \\(s\\), actions \\(a\\), transition
\\(P(s'\mid s,a)\\), reward \\(r(s,a)\\), discount \\(\gamma\in[0,1)\\). The goal is a policy
\\(\pi(a\mid s)\\) maximizing expected discounted return \\(\mathbb{E}[\sum_t \gamma^t r(s_t,a_t)]\\).

A **world model** is a learned simulator of the environment: instead of only learning *what to do*
(policy) or *how good a state is* (value), you learn *what happens next*, then **plan** by imagining
candidate futures and acting on the best. The central design question — the whole point of this post —
is **what exactly does the model predict?**

## 2. Family 1: step-by-step world models (Dreamer, TD-MPC2)

The classic answer: predict the **next** state, one step at a time, then roll forward.

**Dreamer** learns a latent recurrent state-space model: encode observations to a latent \\(z_t\\) and
predict the next latent, reward, and reconstruction,

$$ z_{t+1}\sim p_\theta(z_{t+1}\mid z_t,a_t),\quad \hat r_t=r_\theta(z_t),\quad \hat o_t=\mathrm{dec}_\theta(z_t), $$

then **rolls the model out in imagination** \\(z_t\to z_{t+1}\to\cdots\\) and trains an actor-critic on
the dreamed trajectories.

**TD-MPC2** (the baseline my project chases) learns a latent \\(z=\mathrm{enc}(s)\\), latent dynamics
\\(z'=d_\theta(z,a)\\), reward and value heads, trained with a self-predictive **consistency** loss. At
decision time it runs **MPPI**: sample action sequences, roll them through \\(d_\theta\\) for a short
horizon \\(H\\), score by predicted reward plus terminal value, act:

$$ a_{0:H}\approx\arg\max_{a_{0:H}}\ \mathbb{E}\Big[\sum_{h=0}^{H-1}\gamma^h\hat r(z_h,a_h)+\gamma^H\hat V(z_H)\Big]. $$

**The catch — compounding error.** To reason \\(k\\) steps ahead you compose the one-step model with
itself \\(k\\) times, \\(\hat P^k=\hat P\circ\cdots\circ\hat P\\), and errors **compound geometrically**.
Great for short reactive control; brittle for *long-horizon* reasoning ("how do I cross the maze?").

## 3. The other idea: don't step — jump

What if the model answered "where am I likely to be **eventually**" instead of "after one step"? Fix a
policy \\(\pi\\) and ask for the distribution over future states, nearer futures weighted more — the
**discounted state-occupancy (successor) measure**:

$$ d^\pi_\gamma(s'\mid s)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^{t}\,P\big(s_t=s'\mid s_0=s,\pi\big). $$

Equivalent view: draw a **geometric** horizon \\(T\sim\mathrm{Geometric}(1-\gamma)\\), i.e.
\\(P(T=t)=(1-\gamma)\gamma^{t}\\), and look at \\(s_T\\). Sampling \\(s'\sim d^\pi_\gamma(\cdot\mid s)\\)
is *exactly* "roll out \\(\pi\\), stop at a random geometric time." Hence a **Geometric Horizon Model
(GHM)**: given \\(s\\), it **samples a plausible discounted-future state in one shot** — no stepping
(Janner et al., NeurIPS 2020). To reason about the far future you **sample once** instead of composing
a one-step model \\(k\\) times — so **no geometric error compounding**.

## 4. The self-consistency that makes it learnable

The occupancy measure obeys its own **Bellman recursion** (this step with weight \\(1-\gamma\\), or the
occupancy of wherever you land next with weight \\(\gamma\\)):

$$ d^\pi_\gamma(\cdot\mid s)=(1-\gamma)\,P(\cdot\mid s,\pi)+\gamma\,\mathbb{E}_{s'\sim P(\cdot\mid s,\pi)}\big[d^\pi_\gamma(\cdot\mid s')\big]. $$

It's the analog of \\(V=r+\gamma PV\\), but for *distributions of future states*. So we can learn the GHM
with a **TD bootstrap**: match a mix of the real next state and the model's own prediction at the next
state (via a target network). But first we need a way to learn to *sample* — flow matching.

## 5. Flow matching in one section

To sample from a complicated \\(q(x)\\) given only samples, **flow matching** learns a velocity field
that transports noise into data. Take noise \\(x_0\sim\mathcal N(0,I)\\), data \\(x_1\sim q\\), and the
straight path \\(x_\tau=(1-\tau)x_0+\tau x_1\\) for \\(\tau\in[0,1]\\), whose velocity is \\(x_1-x_0\\).
Train

$$ \mathcal L_{\mathrm{FM}}=\mathbb{E}_{\tau,x_0,x_1}\big\|v_\theta(x_\tau,\tau)-(x_1-x_0)\big\|^2, $$

and **sample** by integrating the ODE \\(\dot x=v_\theta(x,\tau)\\) from \\(\tau{=}0\\) (noise) to
\\(\tau{=}1\\) (data). It's the simulation-free cousin of diffusion — and what InFOM, the scaffold we
build on, uses.

## 6. TD-Flow (ICML 2025): flow matching meets the Bellman bootstrap

Combine §4 and §5: train the flow's target via the occupancy recursion. With probability \\(1-\gamma\\)
the flow target \\(x_1\\) is the **actual next state** \\(s'\\); with probability \\(\gamma\\) it is a
**sample from the target flow at the next state**, \\(x_1\sim d^\pi_{\bar\theta}(\cdot\mid s')\\). This
learns the full discounted future from one-step data, and the TD construction keeps **variance low over
long horizons** (Farebrother et al., arXiv 2503.09817). **TD-Flow = how to train a good GHM.**

## 7. CompPlan (ICLR-WS 2026): planning by jumping over policy sequences

Two ingredients turn a GHM into a planner (arXiv 2602.19634, same group):

1. **Horizon-conditioning:** condition the flow on a chosen horizon, \\(G_h^\pi(\cdot\mid s)\\) — short
   jumps for local moves, long for big ones.
2. **Policy-conditioning:** condition on *which* base policy \\(\pi_i\\) runs, so one model predicts the
   outcomes of a whole library \\(\{\pi_1,\dots,\pi_m\}\\).

Planning is then a search over **sequences of policies** from start \\(s\\) to goal \\(g\\):

$$ s\xrightarrow{\,G^{\pi_{i_1}}\,}\hat s_1\xrightarrow{\,G^{\pi_{i_2}}\,}\hat s_2\cdots\to\hat s_K\approx g, $$

sampling candidate subgoals \\(\hat s_k\\) from the GHM and scoring by proximity to \\(g\\). Each arrow
is **one jump** skipping many primitive steps, so a \\(K\\)-policy plan reasons over *hundreds* of steps
without composing a one-step model. On OGBench this is the paper's ~**+200%** long-horizon headline.
**CompPlan = how to *plan* with a GHM** — the "jumpy world model."

## 8. Side-by-side: Dreamer vs TD-MPC2 vs CompPlan/GHM

| | **Dreamer** | **TD-MPC2** | **CompPlan / GHM** |
|---|---|---|---|
| Model predicts | next latent | next latent | a **discounted-future state** |
| Time granularity | one step | one step | **jumpy** (geometric horizon) |
| Trained by | recon + KL, imagination | consistency + reward + value | **flow matching + TD bootstrap** |
| Plans by | actor-critic in imagination | **MPPI** over short rollouts | **search over policy sequences** |
| Long horizon | compounding error | compounding error; short \\(H\\) | **no step-composition** → strong |
| Best at | pixels, general control | reactive control, sample-eff. | long-horizon / compositional goals |

One sentence: **Dreamer and TD-MPC2 imagine the future one step at a time and pay for it over long
horizons; a GHM samples the far future directly, and CompPlan stitches such jumps across a library of
policies to plan very long horizons.** They're complementary — and this matches what I found
independently: my own jumpy k-step model helped exactly on the contact-structured, longer-horizon
manipulation tasks and was neutral-to-harmful on reactive locomotion.

## 9. Where the reproduction stands

No public code exists for TD-Flow or CompPlan, so I build on **InFOM** (Intention-conditioned Flow
Occupancy Models, ICLR 2026) — open-source JAX flow-matching over the occupancy measure on OGBench. It
trains now and reproduces sensible cube-single success. The work in flight is the **GHM extension**:
adding §7's horizon-conditioning, policy-conditioning, and plan-over-policies on top. That build is the
next post.

*Pointers (verify before citing): γ-models — Janner et al. NeurIPS 2020 (2010.14496); TD-Flow —
Farebrother et al. ICML 2025 (2503.09817); CompPlan — ICLR-WS 2026 (2602.19634); InFOM —
github.com/chongyi-zheng/infom; OGBench — github.com/seohongpark/ogbench; TD-MPC2 — Hansen et al. ICLR
2024 (2310.16828); Dreamer — Hafner et al.*
