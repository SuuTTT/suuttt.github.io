---
title: "Why an Opponent Pool Beat a Bigger Batch: Scaling Mahjong Self-Play Across a GPU Fleet"
date: 2026-06-01
description: "Our supervised Mahjong bot plateaued in self-play — 88% of games drawn, the improvement metric stuck. This post explains the opponent-pool (league) design that broke the plateau, why we ran five experiments in parallel on borrowed idle CPU, and the surprising result: training against a diverse pool of our own bots beat simply throwing a 4x larger batch at the problem."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["research", "mahjong", "reinforcement-learning", "self-play", "ppo", "game-ai", "botzone", "distributed-training"]
---

{{< katex >}}

## TL;DR

Our [supervised Mahjong policy]({{< relref "2026-05-31-ijcai-mahjong-ai-from-sample-to-supervised.md" >}}) was legal and reasonable but weak: in self-play it drew ~88% of games and rarely completed a winning hand. PPO self-play against a single frozen opponent improved a little, then **plateaued** — the improvement metric bounced inside the noise. We fixed it with an **opponent pool** (a *league*): the learner trains against a rotating set of frozen snapshots of strong bots, not one fixed opponent. To find the right recipe fast, we ran **five PPO configurations in parallel** on idle CPU borrowed from GPU-busy machines across a Vast.ai fleet, and ranked the outputs with a **cross-play tournament** rather than each run's own (noisy) metric. The winner — the pool config — beat our previous best by a clear margin and, tellingly, **a run with a 4× larger batch but no pool produced no improvement at all.** The lesson: for self-play that has collapsed into draws, *opponent diversity* matters more than *gradient batch size*.

---

## 1. The plateau

Chinese Standard Mahjong has an unusual property that dominates everything: a hand only counts as a win if it scores at least **8 fan** (番), and an illegal action costs \(-30\) — more than most legitimate wins. We had already driven illegal moves to zero (see the [previous post]({{< relref "2026-05-31-ijcai-mahjong-ai-from-sample-to-supervised.md" >}})). The remaining problem was **conversion**: our bots almost never built an 8-fan hand.

This shows up brutally in self-play. When four copies of the same cautious supervised policy sit at a table, nobody feeds anybody a winning tile, and the games end in a wall-exhaustion draw (*huang*, 荒庄). We measured **~88–90% draws**. A drawn game pays zero to everyone, so the reinforcement-learning signal is *extremely sparse* — almost every trajectory ends in reward 0.

We warm-started PPO from the supervised policy and trained it against a **frozen copy of the supervised baseline**, measuring progress as the net duplicate score of the learner's two seats against the baseline's two seats. It worked — for a while:

$$ \text{net} = \sum_{s \in \text{learner seats}} \text{score}(s) - \sum_{s \in \text{baseline seats}} \text{score}(s) $$

The net climbed to roughly **+450** over a few hundred games and then stopped. More iterations just made it oscillate inside \(\pm 150\). We had hit a wall, and three things were responsible:

1. **Sparse reward.** 88% of games carry no signal.
2. **A single, static opponent.** The learner could over-fit to the *specific* frozen baseline — learning to exploit one opponent's quirks rather than getting genuinely stronger.
3. **A noisy metric.** With so many draws, the net score is dominated by the handful of games that happen to produce a win, so the per-run number has high variance.

---

## 2. The opponent pool (league)

The fix for problems (2) and a chunk of (1) is a classic self-play idea: don't train against one opponent, train against a **pool**.

Concretely, the learner always controls two seats. The other two seats are filled, **each game**, by a frozen policy sampled from a pool:

$$ \pi^{\text{opp}}_{\text{seat}} \sim \text{Uniform}(\mathcal{P}), \qquad \mathcal{P} = \{\pi_{\text{SL}},\ \pi_{\text{PPO-1}},\ \pi_{\text{PPO-2}},\ \dots\} $$

The pool \(\mathcal{P}\) starts with a few hand-picked strong checkpoints — our supervised policy and our two best PPO policies so far — and **grows over time**: every \(N\) iterations we snapshot the current learner and add it to the pool. This is a lightweight version of the league play that powered AlphaStar, and it directly attacks over-fitting:

- A policy that beats one opponent by exploiting a specific weakness will *lose* to another pool member that doesn't share that weakness, so the gradient pushes toward **robust** improvement rather than narrow exploitation.
- A growing pool means the learner must keep beating *past versions of itself*, which prevents the cyclic "rock-paper-scissors" forgetting that pure self-play suffers from.

The reward stayed simple. We use the real duplicate game score, lightly densified on drawn games by a small shaping term that rewards ending closer to a complete hand (lower [shanten]({{< relref "2026-05-31-chinese-standard-mahjong-for-ai-researchers.md" >}}), the distance-to-ready):

$$ r(\text{seat}) = \frac{\text{game score}}{30} + \lambda \cdot \mathbb{1}[\text{huang}] \cdot \big(\text{shanten closeness}\big), \quad \lambda \approx 0.015 $$

The \(1/30\) scale comes from the \(-30\) illegal penalty being the natural unit of the game; \(\lambda\) is deliberately tiny so the shaping nudges without overriding the true objective.

---

## 3. Five experiments, in parallel, on borrowed CPU

The open question was *which* recipe — pool composition, learning rate, entropy, batch size — would actually break the plateau. Rather than try them one at a time, we ran them **simultaneously**.

The right hardware insight here is counter-intuitive: **our bottleneck is CPU, not GPU.** The policy network is tiny (a few-million-parameter residual MLP), so the gradient step is sub-second; almost all wall-clock goes into *simulating games* on the CPU. So a fleet of **GPU** machines is useful to us only for the **CPU cores sitting idle** while their GPUs are busy with other people's jobs.

We had exactly that: a Vast.ai fleet where several machines were running JAX/MuJoCo training **on the GPU**, leaving 56–128-core CPUs at ~2% load. We borrowed those idle cores — politely (`nice`-d, leaving headroom) — and launched five configurations at once:

| Run | Init from | Opponents | Batch (games/iter) | Twist |
|---|---|---|---|---|
| `bigbatch` | best PPO | frozen baseline | **2200** | biggest batch, cleanest gradient |
| `poolbig` | best PPO | **pool of 3** | 1500 | pool + large batch |
| `robust` | best PPO | **pool of 3** | 700 | pool, smaller batch |
| `explore` | best PPO | frozen baseline | 1500 | high entropy (more exploration) |
| `drawpush` | best PPO | frozen baseline | 800 | aggressive win-shaping |

We ran five configurations concurrently, one per box — but two hard-won lessons turned out to matter more than the raw box count.

**A rented "72-core" box is not 72 cores.** Cloud GPU rentals hand you a cgroup **CPU-time quota**, not the physical host cores. The scheduler lets your threads *spread* across all 72 cores, but throttles your total compute to the quota. We measured this directly on a 72-physical-core machine whose quota was 8.6 cores:

| parallel CPU-burn threads | wall time |
|---|---|
| 1 | 0.60 s |
| 8 | 0.65 s — *free* up to the ~8.6-core quota |
| 17 | 1.63 s — past quota, throttled 2.7× |
| 40 | **9.20 s — 14× slower**, pure context-switch thrash |

Across our nine free boxes the *physical* core counts summed to ~236, but the *real* quotas summed to only **~78 cores**. The single best box (a 2080 Ti rental) gave 17 cores; a same-physical A4000 gave 8.6; the small boxes gave ~6 each. The operational rule is blunt: **size your worker count to the per-box quota, not the core count it advertises** — over-subscribing is catastrophically slow, not merely wasteful. (We learned this the embarrassing way: our first sweep ran 40–50 workers on 8–17-core quotas and was needlessly throttled.)

**SSH gets rate-limited.** Vast.ai routes SSH through shared proxy hosts (`ssh1.vast.ai` fronts many instances), so firing many short-lived connections at the same proxy gets you refused with "*try again after a few seconds*." Two fixes, both verified: connect to each box's **direct IP** (from `vastai ssh-url`) to skip the shared proxy, and use **SSH `ControlMaster` multiplexing** to run all commands over one persistent connection per box. With those, the throttling disappears.

A third papercut: heterogeneity. Boxes differed in NumPy version (one was on 2.4, where `float(array_of_one)` is now a hard error — a real portability bug we had to fix) and in whether the system Python even had PyTorch. Budget time for environment wrangling, not just compute.

---

## 4. The result: pool > batch

We do **not** trust each run's own self-reported metric — Section 1 explained why it's noisy. Instead we harvested the best checkpoint from every run and put all seven candidates (the five new runs plus our two prior best policies and the original supervised bot) into a single **cross-play tournament**: every pair plays 400 duplicate games with seats rotated to cancel positional bias, scored by net points and raw wins.

| Model | Total net | Raw wins | Notes |
|---|---|---|---|
| **`poolbig`** | **+6400** | **171** | pool + large batch — **winner** |
| `explore` | +4210 | 155 | |
| `robust` | +4206 | 164 | pool, small batch |
| `ppo_vb` | +4174 | 160 | prior best (single opponent) |
| `league` | +4080 | 155 | prior pool attempt |
| `drawpush` | −5044 | 95 | aggressive shaping *hurt* |
| `bc_v3_ft` | −18026 | 51 | supervised baseline |

Two findings stand out.

**First, the pool config won clearly.** `poolbig` sits well above the pack (+6400 versus a cluster around +4100) and, crucially, it wins by **completing more hands** — it has the most raw wins *and* the lowest draw rate of any candidate. It didn't just learn to defend; it learned to convert. That is exactly the weakness we set out to fix.

**Second — and this is the headline — `bigbatch` produced no usable checkpoint at all.** The run with the largest batch (2200 games/iter, the cleanest possible gradient) but training against a *single* frozen opponent never beat its own baseline; its net hovered around zero for the entire run. Meanwhile both **pool** runs (`poolbig` and `robust`) landed at the top.

The conclusion writes itself:

> For self-play that has collapsed into draws, **opponent diversity breaks the plateau; raw gradient batch size does not.**

A bigger batch makes each step less noisy, but if every step points in the same over-fit direction, a cleaner version of "stuck" is still stuck. The pool changes *what the gradient points at*.

---

## 5. Honest caveats

Three things keep us skeptical of our own tournament:

1. **In-distribution advantage.** `poolbig` trained against a pool that *includes* the very opponents it is later scored against. Some of its edge is therefore "home-field." The reassuring counter-signal is *how* it wins — by conversion, which should transfer — but a clean test needs unseen opponents.
2. **Self-play is not the real game.** Four strong, cautious bots draw far more often than a real, varied field does. The number that actually matters is performance against the **other competitors** on the live ladder, where opponents discard differently and feed wins we never see in self-play.
3. **The draw ceiling is structural.** Even our best bot still draws the large majority of self-play games. The deep problem — building high-fan hands reliably — is only partially solved.

So the deployment pick (`poolbig`) is the strongest model we can *measure*, with the explicit understanding that the binding signal is a real ladder log, and the next training round should grow the pool further and lean into the conversion behavior that worked.

---

## 6. This is population-based training — and we should run it as a loop

It's worth being explicit about what we actually did, because it is not novel and the literature tells us how to push it further. "Train several configurations in parallel, then hold a tournament to pick the best" is the intersection of three well-established ideas:

- **Population-Based Training (PBT)** — Jaderberg et al., 2017 (DeepMind). Train a *population* of agents in parallel; periodically each member **exploits** (copies the weights of a better member) and **explores** (perturbs its hyperparameters). Our five concurrent configs are a population; the tournament is the fitness evaluation.
- **League play** — the AlphaStar architecture (Vinyals et al., 2019). A growing population of frozen agents, with opponents sampled by *prioritized fictitious self-play* (play the opponents you currently struggle against). Our opponent pool is a minimal league.
- **Competitive coevolution / tournament selection** — an old evolutionary-computation idea (Axelrod's iterated-prisoner's-dilemma tournaments, 1980; Pollack & Blair; NEAT). Fitness is defined *relative to the rest of the population*, exactly like a payoff matrix.

So the reader's instinct that "this is just like the competition itself, and a bit like a genetic algorithm" is precisely right — and it points at the natural next step: stop running the train-then-tournament once, and run it **periodically as a closed loop**:

```
        ┌─────────────────────────────────────────────┐
        │  POPULATION  {agentᵢ}     POOL  𝒫 (frozen)   │
        └─────────────────────────────────────────────┘
                 │ (1) train each agentᵢ in parallel vs 𝒫
                 ▼
              snapshot every agentᵢ
                 │ (2) cross-play tournament  →  payoff matrix / Elo
                 ▼
          (3) SELECT top-k survivors by Elo
                 │
          (4) EXPLOIT: weak members copy a survivor's weights
          (5) EXPLORE: perturb hyperparams (lr, λ, entropy, pool mix)
          (6) add this generation's champions to 𝒫
                 │
                 └────────────────► next generation
```

This is a genetic algorithm whose fitness function is *the game itself*. Each generation the opponent pool gets stronger and more diverse, the survivors define the new frontier, and the tournament Elo identifies the champion to deploy.

Two cautions the literature is emphatic about:

1. **Coevolution can cycle.** If A beats B beats C beats A (intransitivity), naïve selection chases its tail and never improves in absolute terms. The mitigations are exactly AlphaStar's: keep *old* champions permanently in the pool (so the population can't forget how to beat them), and sample opponents with *prioritized fictitious self-play* rather than uniformly.
2. **You need a fixed anchor.** Relative Elo can drift while absolute strength stalls. So every generation we also score against a *frozen* anchor — our supervised baseline — and, ultimately, against the real ladder. Progress is only real if the anchor score moves too.

For our setting the loop is cheap to run: a generation is one round of parallel PPO (minutes, quota-sized to each box) plus one tournament (a few minutes on freed CPU). The plan is to run it generation-over-generation through the competition, deploying the current Elo champion and feeding any real ladder logs back in as the most valuable pool members of all.

## 7. Takeaways

- **Match the hardware to the bottleneck.** Self-play with a small network is CPU-bound; idle CPU on GPU-busy machines is free fuel, and idle GPUs do nothing for you.
- **Parallelize the *search*, not just the run.** Five concurrent configurations plus an honest cross-play tournament found the answer faster than careful sequential tuning would have — and protected us from each run's noisy self-metric.
- **Diversity beats brute force for collapsed self-play.** An opponent pool broke a plateau that a 4× larger batch could not budge.
- **Stay honest about what you measured.** Winning your own tournament is necessary, not sufficient; keep the real benchmark — the live competition — in view.

*This is the third post in a series on building an AI for the 6th International Mahjong AI Competition (IJCAI 2026). Earlier posts cover the [supervised pipeline and legality engineering]({{< relref "2026-05-31-ijcai-mahjong-ai-from-sample-to-supervised.md" >}}) and [the rules of Chinese Standard Mahjong for AI researchers]({{< relref "2026-05-31-chinese-standard-mahjong-for-ai-researchers.md" >}}).*
