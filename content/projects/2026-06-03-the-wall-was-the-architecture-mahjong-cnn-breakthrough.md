---
title: "The Wall Was the Architecture: How Reproducing a Rival's Bot Broke Our Mahjong Plateau"
date: 2026-06-03
description: "For a week our Mahjong bot improved against itself and never against anything else — every in-house lever (bigger batches, leagues, PBT, planning, defense, fan-shaping) converged to the same passive, 89%-draw equilibrium. Then a competitor shared their bot's source. We reproduced it, ran the first real cross-architecture benchmark, and a barely-trained CNN crushed our heavily-tuned MLP 0-of-60. The wall was never the algorithm or the compute. It was the model class. This is the milestone post: the honest negative results, the breakthrough, and the surprisingly painful Botzone deployment that followed."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["research", "mahjong", "reinforcement-learning", "supervised-learning", "cnn", "game-ai", "botzone", "deployment", "negative-results"]
---

{{< katex >}}

## TL;DR

After [an opponent-pool league broke our self-play plateau]({{< relref "2026-06-01-opponent-pool-beat-bigger-batch-mahjong-self-play.md" >}}), we spent days pushing every in-house lever we could think of — drift-proof leagues, AlphaStar-style exploiters, test-time planning, defense/deal-in avoidance, fan-potential reward shaping, behavior-cloning on high-fan expert hands. **Every single one converged to the same passive equilibrium: ~89% of self-play games drawn, the bot near-optimal *against itself* and unmeasurable against anything else.** We even discovered our headline "+4008" improvement had been a measurement-harness artifact; with one consistent harness, all our models were statistically indistinguishable.

Then a competitor shared their bot's source code. We reproduced it from the public dataset and ran the first **cross-architecture** head-to-head through the official judge. A **CNN trained for a single epoch beat our most-tuned MLP +1039 over 60 games — our bot won zero** — and the draw rate collapsed from 89% to 40%. The CNN *converts*; our MLP never could. **The wall was the architecture all along.** We were tuning a fundamentally weaker model class.

This post is the honest arc: the negative results, the clue hiding in real game logs, the breakthrough, and the three-layer Botzone deployment bug-hunt that stood between a working model and a working *submission*.

---

## Where we left off

The [previous post]({{< relref "2026-06-01-opponent-pool-beat-bigger-batch-mahjong-self-play.md" >}}) ended on a real but small win: an opponent **pool** (a league of our own frozen snapshots) beat a single-opponent baseline, and — notably — beat a 4× larger batch that had no pool. The deep problem was unchanged, though: in self-play, four equally-cautious bots **draw ~88–90% of games**. Chinese Standard Mahjong has an **8-fan floor** (a hand must score at least 8 fan to win), so a fast, cheap tenpai scores nothing. Our bot raced to tenpai and then sat there. The *conversion* problem — actually building 8-fan hands — was untouched.

So we attacked it from every angle we had.

## Part 1 — The wall: a week of honest negative results

**Drift-proof leagues.** We rebuilt the league with a permanent held-out anchor (never in any training pool), a cheap ranking tournament followed by a high-N confirmation tournament, and a promotion gate requiring a candidate to beat *both* the anchor and the current champion by a margin. This fixed real bugs (earlier "promotions" had been noise, and a coevolution drift had once silently shipped a *regressed* model). But the ceiling held: candidates hovered around the champion ± noise.

**A measurement lesson that stung.** We had recorded a champion beating the anchor by "+4008." When we later re-measured every model through one *consistent* in-process harness, the real margin was **+142 over 1200 games at 90% draws — i.e. noise.** The "+4008" had been an artifact of mixing two harnesses with different score scales. Quantified bluntly: **in-house self-play margins under ~±400 per 1200 games are noise.** Several of our "improvements" had been measuring the ruler, not the bot.

**Defense is a non-lever.** Surely avoiding deal-ins (放铳) helps? We instrumented it. The champion's **deal-in rate was 1.8% in self-play and 0.7% against a looser opponent.** The bot was already near-optimally defensive; the maximum recoverable points sat far below the noise floor. The problem was never defense — it was conversion.

**Conversion is unmeasurable in-house.** We tried test-time planning (Monte-Carlo fan-potential rollouts to use our unused time budget), a value-network search, fan-potential reward shaping, and finally a **behavior-clone fine-tune on 562k samples from high-fan (≥12) expert wins**. Each one *hurt* or did nothing. The high-fan BC was the clearest: it made the bot **greedier and slower** — self-play draws went *up* to 96%, and against weaker bots it dealt in *more* while winning less. Chasing big hands, stripped of the expert's timing and defense, just held dangerous tiles longer.

The pattern was unmistakable. **Every lever converged to the same passive, draw-heavy, near-identical equilibrium — because that equilibrium is optimal *against itself*.** Whether more aggression helps *against the real field* was something none of our in-house metrics could see. We wrote it down as the binding constraint and went looking for outside signal.

## Part 2 — A clue from real games

A few real ladder replays among top competitors arrived. We parsed them. The numbers were a splash of cold water:

- **Strong play draws ~0–3% of games. Ours drew 89%.**
- All the wins we saw were real big hands — 碰碰和 (18 fan), 组合龙 (14), 一色三步高 (16) — and they were won by **farming a weak/loose seat that fed tiles**.

The 89% draw ceiling was a **self-play artifact** of four identical cautious bots starving each other. The actual contest converts. Our in-house tournaments had been measuring a world that doesn't exist on the ladder. This didn't tell us *how* to fix conversion — but it told us, unambiguously, that the problem was real and that our measurements had been lying to us.

## Part 3 — The breakthrough: reproduce the rival, then benchmark

Then a competitor shared their **bot's source code** — a PKU-lineage supervised baseline. It was a **CNN**: the tile state encoded as a `(38, 4, 9)` stack of feature planes, run through a 3-conv stem, **16 residual bottleneck blocks**, and a fully-connected head over the 235-action space (~10M parameters). Our entire lineage, by contrast, was a flat **240-dimensional MLP** (~3.4M parameters).

It shipped its training pipeline and expected the *same public dataset we already had*. So we reproduced it: preprocessed 98k games into ~5.9M samples, trained the CNN, and — for the first time — ran a **cross-architecture** match through the official judge. Not our-model-vs-our-model. A genuinely different bot.

The result reframed the entire project:

| Matchup (official judge, 60 games) | Draw rate | Result |
|---|---|---|
| our MLP (r18) vs our MLP (bc baseline) | 78% | r18 +156 — *as in-house tournaments always showed* |
| our MLP (r18) vs **CNN, epoch 0** | **40%** | **CNN +1039, our bot won 0 of 60** |
| our MLP (r18) vs **CNN, epoch 3** | **10%** | **CNN +1623, our bot won 2 of 60** |

A CNN trained for **one epoch** beat our most-tuned model and our bot **won zero games**. By epoch 3 the draw rate had fallen to 10% — the CNN *converts*, building and completing the 8-fan hands our MLP never could. We validated the harness (our MLP beats the weak baseline exactly as in-house, so it loads and plays fine); the gap is real.

**The wall was never the algorithm, the reward shaping, or the compute. It was the model class.** A flat MLP discards the spatial structure of the tile grid — adjacency, runs, partial melds — that a convolution captures natively. We had spent a week running PPO, PBT, leagues, and planning *on top of a representation that couldn't see the game*. Every plateau was a symptom of that.

The honest, slightly humbling lesson: **we were tuning hyperparameters when we should have been questioning the architecture — and the thing that finally told us was a rival's code, not any metric we owned.**

## Part 4 — Working model ≠ working submission

Adopting the CNN turned out to be the easy part. Getting it to *run on Botzone* was a three-layer bug-hunt worth documenting, because each layer produced a different cryptic verdict.

1. **`RE` (runtime error).** The reference bot read input with `input()`, which raises `EOFError` when the judge closes stdin between turns. Our proven MLP bot reads with `sys.stdin.readline()` and **breaks on EOF**. Fix: mirror that. (Botzone does provide PyTorch 1.4 under Python 3.6, so the CNN deploys natively — but a checkpoint saved by a modern PyTorch must be re-serialized with `_use_new_zipfile_serialization=False`, or 1.4 can't load it.)

2. **`NJ` (output is not JSON).** Botzone's simple interaction sends a JSON blob and parses the program's **entire stdout** as one JSON object. We were emitting plain `PASS`. Fix: emit `{"response": ...}` and rebuild the full game state by **replaying the request/response history each turn** (applying our own past actions, since they aren't echoed back).

3. **`NJ` again.** We added the JSON — *and* the keep-running marker line after it. But that trailing `>>>BOTZONE_REQUEST_KEEP_RUNNING<<<` line makes stdout invalid JSON in simple mode. Fix: in JSON mode emit **only** the JSON, no marker. (The marker belongs to the long-running protocol, a different mode.)

The meta-lesson: **the deployment protocol is part of the model.** A bot that plays perfectly in your simulator but mis-handles EOF, output encoding, or the interaction mode scores `-30` just as surely as an illegal move. We now keep a single I/O layer validated against the real judge and treat it as load-bearing.

## Part 5 — The milestone

The fixed CNN is **live on Botzone**, and the verdicts tell the story:

- **344 judge verdicts in one match, all `OK`** — no `RE`, no `NJ`, no illegal moves.
- The game **ended in a win**: a self-drawn, fully-concealed **8-fan hand** (不求人 + 平和 + 喜相逢 + 缺一门), score `[-16, +48, -16, -16]`.
- The bot **claims actively** (Peng, Chi) and **completes hands** — exactly the conversion behavior our MLP could never produce.

In parallel we ran an 8-hour **architecture search** — depth, width, head, and capacity variants — to confirm we'd landed on the right model rather than a lucky one. The verdict: the **16-block, 128-channel** configuration is the sweet spot. Going deeper *hurts* (a 32-block net diverged outright without normalization), wider (256 channels) didn't help, and — crucially — **full convergence did**: trained to 14 epochs, that config reaches **+2826 vs our old r18 (99 of 120 games won, validation accuracy 0.863)**, nearly double its 3-epoch margin. That converged model is what's now deployed.

## Lessons

- **Question the representation before the algorithm.** A week of RL on the wrong model class produced beautifully-tuned mediocrity. The single highest-leverage change was switching MLP → CNN, and we should have tried it far sooner.
- **Self-play metrics can describe a world that doesn't exist.** Our 89%-draw equilibrium was real and stable and *irrelevant* — real games draw ~3%. Validate against *out-of-distribution* opponents (a different architecture counts) before trusting any in-house number.
- **A rival's open code is worth more than another week of compute.** It gave us a calibrated, reproducible benchmark *and* a better architecture in one afternoon.
- **Negative results compound into a direction.** Defense, planning, fan-shaping, high-fan BC all failed — and the *pattern* of their failure ("optimal against itself, blind to the field") is what sent us looking outside, which is where the answer was.
- **The submission protocol is part of the model.** `RE` → `NJ` → `NJ` taught us that EOF handling, JSON encoding, and interaction mode each silently cost full games.

With the contest deadline days away, we finally have a bot that does the one thing the old one never could: it *wins*. And now that we have a strong base **and** a trustworthy cross-architecture benchmark, we're pushing two fronts in parallel on borrowed fleet GPUs:

- **Architecture exploration** — training transformer (tile-token self-attention), CNN+attention hybrid, a graph network over the tile-adjacency structure, and a *normalized* deep ResNet (to test whether depth helps once it can train) — each benchmarked against the converged CNN through the real judge.
- **RL fine-tuning on the CNN** — the thing we spent a week attempting on a model that couldn't support it. With a value head and a league guard (to avoid the self-play passivity trap), seeded from the converged supervised base.

The throughline of this whole episode: **measure against something you didn't build, and question the representation before the recipe.** A rival's open source did more for us in an afternoon than a week of our own compute — and it turned a plateau into a bot that finally converts.
