---
title: "Building a Chinese Standard Mahjong AI: From Official Sample to Supervised Policy"
date: 2026-05-31
description: "How we built a legality-first Mahjong bot for the IJCAI 2026 competition on Botzone — the system design, the 98k-game supervised pipeline, the five-bug hunt that drove illegal moves to zero, and an honest read on what local evaluation can and cannot tell us."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["research", "mahjong", "reinforcement-learning", "imitation-learning", "game-ai", "botzone"]
---

{{< katex >}}

## TL;DR

We are building an agent for the **6th International Mahjong AI Competition (IJCAI 2026)**, played on the Botzone platform under **Chinese Standard Mahjong (国标麻将)** rules. This first post documents the engineering: a legality-first bot architecture, a supervised-learning pipeline trained on the official **98,209-game** strong-AI dataset, an evaluation harness built around the real competition judge, and a debugging story in which five distinct state-tracking bugs were driving a catastrophic illegal-move rate from **10% down to 0%**. We also explain why our encouraging local numbers should be read with heavy skepticism — and how the contest's own scoring rules tell us so.

---

## 1. The problem, and why it is harder than it looks

From an AI standpoint Chinese Standard Mahjong is a **four-player, imperfect-information, stochastic, sequential-decision game with a complex terminal payoff**. Each player sees only their own 13–14 tiles plus public discards and melds. A hand only counts as a win if it scores at least **8 fan** (番) — the official threshold — where fan are awarded by an intricate table of 81 patterns.

The scoring asymmetry is the whole game. A self-drawn win pays

$$ \text{self-draw: } +3(8+f) \text{ to winner}, \quad -(8+f) \text{ to each opponent} $$

while a win on a discard ("rong") pays

$$ \text{rong: } +(24+f) \text{ to winner}, \quad -(8+f) \text{ to the discarder}, \quad -8 \text{ to the other two} $$

where \(f\) is the fan count above the 8-fan floor. But the most important number in the whole rulebook is the **penalty for an illegal action**: the offender gets \(-30\) and the other three each get \(+10\). A single illegal move — an out-of-turn claim, a discard of a tile you do not hold, or declaring a win that scores under 8 fan — is worth more than most legitimate wins. **Legality is not a detail; it is the dominant term in the objective.**

### The competition's real metric

The formal rounds do not rank by ladder Elo. They use **duplicate scoring (复式赛)**: the same tile walls are replayed across all seat permutations, so luck largely cancels and what remains is strategy. A bot that occasionally explodes for a huge score but is high-variance can still lose the duplicate ranking. This shapes everything downstream: we must evaluate on duplicate-style packets, not on single-game win rate.

---

## 2. A reality check that reframed the whole project

Before optimizing anything, we analyzed the match records of our first deployed bot ("Official Sample 2") across **512 games** of Simulation-6. The raw positive-score rate looked like a respectable **30.9%**. But filtering revealed the truth:

- **139 of 158** positive games were exactly \(+10\) — the failure-compensation payout from *another* bot's \(-30\), not a win of ours.
- After removing failure-compensation games and draws, the **corrected true-win rate was only 5.49%**, with an average filtered score of \(-12.24\) per game.

Two lessons fell out of this, and they steered every subsequent decision:

1. **Beating the random "official sample" locally proves nothing.** Against three random opponents our bots win the duplicate packet easily, but ~70–78% of those games end in a draw (荒牌) because nobody reaches 8 fan. The signal is almost entirely noise.
2. **Defense matters as much as offense.** The deployed bot's problem was not only that it rarely won — it lost frequently and heavily (a 94.5% negative-score rate). That observation later told us *which* training data to use.

---

## 3. System design

The project is organized around a strict separation between **what is legal** and **what is good**.

```
official judge + rules      official 98k-game dataset
        │                            │
        ▼                            ▼
 legality / state tracker      feature extraction (240-d)
        │                            │
        ▼                            ▼
   action masking  ◄──────────  supervised policy (MLP)
        │
        ▼
  bot response  →  Botzone judge  →  replay / error analysis
```

### 3.1 The feature space

We ported the official reference feature agent into Python. Each decision is encoded as a **240-dimensional observation** and the action space has **235 entries**:

- Observation: prevalent wind, seat wind, an "unshown" count for each of the 34 tile types (how many copies remain unseen), the 14 hand slots, a private wall view, and per-player histories and melds.
- Actions: \(\{\text{Pass}, \text{Hu}\}\) plus Play (34), Chi (63), Peng (34), Gang (34), AnGang (34), BuGang (34).

Crucially, the feature agent also produces, every turn, the exact set of **legal** actions. This `valid` set is the action mask: the policy never even considers an illegal move at the logit level.

### 3.2 The policy network

A compact residual MLP, chosen because the 240-d feature already encodes tile structure (a CNN buys little here):

- Stem: \(\text{Linear}(240 \to 512) \to \text{LayerNorm} \to \text{ReLU} \to \text{Dropout}\)
- Six residual blocks of width 512 with dropout 0.15
- A policy head over the 235 actions and a value head with \(\tanh\) output
- ~3.4M parameters

At inference time the weights are exported to a numpy `.npz` and the forward pass is plain matrix multiplies — about **1.3 ms per move on a single CPU core**, well inside Botzone's per-turn budget. (We force single-threaded BLAS: the matmuls are tiny, and thread spawning on a single-core judge environment is pure overhead.)

### 3.3 Training objective

Behavior cloning with a legal-action mask. For observation \(o\), legal set \(\mathcal{A}(o)\), and expert action \(a^\star\), the masked policy is

$$ \pi(a \mid o) = \frac{\exp\bigl(z_a(o)\bigr)\,\mathbb{1}[a \in \mathcal{A}(o)]}{\sum_{a' \in \mathcal{A}(o)} \exp\bigl(z_{a'}(o)\bigr)} $$

and we minimize the cross-entropy \(-\log \pi(a^\star \mid o)\) with AdamW and a cosine schedule, training on an RTX 3060 with CUDA.

### 3.4 Two datasets, two skills

We parsed the official dataset two ways:

- **Winner-only** (1.3M samples): learn from the eventual winner's decisions — pure offense.
- **All-players** (5.1M samples): learn from every seat — this is where **defense** comes from, since most players in a strong game are *not* winning and are instead discarding safely.

Because the Simulation-6 analysis flagged defense as the deployed bot's weakness, our deployment model is the **all-players** policy.

---

## 4. The bug hunt: driving illegal moves to zero

This was the heart of the work, and it is worth telling honestly because the bug was invisible to our first test harness.

Our initial 200-game "stress test" reported zero illegal moves — but it used a *simplified* self-play harness, **not the real judge**. When we wired up the actual competition judge (`judge/main.cpp`) and ran games through it, the illegal-move rate was **~10%**. Five distinct root causes, found and fixed one at a time:

1. **Win tile inside the hand.** The fan calculator expects the concealed hand *without* the winning tile; we were passing all 14 tiles, so every genuine win returned "not a win." Fixed by separating the win tile.

2. **A second, drifting state tracker.** We had a hand-rolled game-state object *and* the feature agent, and they disagreed on chi-heavy hands — the hand-rolled one accumulated phantom melds (we once saw six melds in a four-meld game). Fixed by making the bot **feature-agent-driven, single source of truth.**

3. **Own kongs never applied.** When the bot declared a concealed/added kong, the echo notification re-fed only `PLAY` actions, so the kong never updated state — and the bot would try to kong the same tile twice. Fixed by applying *all* of my own actions at decision time and skipping the echo.

4. **The concealed-meld fan inflation.** This one was subtle. A chi/peng recorded with meld-offer \(0\) is read by the fan calculator as a **concealed** meld, which awards the 门前清 ("fully concealed") bonus of \(+2\) fan. A hand worth a real \(6\) fan was being scored as \(8\) — just enough to trigger an illegal win declaration. Fixed by forcing every meld to **exposed** in the win check, which yields a conservative *lower bound* on fan:

$$ f_{\text{safe}} = \text{fan}(\text{hand} \mid \text{all melds exposed}) \le f_{\text{true}} $$

If \(f_{\text{safe}} \ge 8\) the win is guaranteed legal, because the judge — using the true offers — can only score it the same or higher.

5. **The off-by-thirteen wall.** In duplicate play each player draws from their own 34-tile wall; after the 13-tile deal, \(34 - 13 = 21\) tiles remain. We had initialized the wall counter to 34, so our "wall nearly empty" guard fired **13 draws too late** — and the bot would try to chi at the true end of the wall, which the judge forbids (no replacement draw remains). Fixed by initializing the counter to 21.

On top of these we added a **universal emit-time verifier**: before any action leaves the bot, it is re-checked physically against the hand (do I hold these tiles? is this the last discard? is the wall exhausted?). If anything fails, the bot falls back to a guaranteed-legal pass or discard.

One diagnostic insight saved us a lot of confusion: **a \(-30\) score is not always an illegal move.** A score line like \([-30, 46, -8, -8]\) is a legitimate deal-in — someone won a 22-fan hand off our discard, and the discarder pays \(-(8+22) = -30\). The *illegal* signature is specifically \([-30, +10, +10, +10]\). Our evaluator now distinguishes the two.

**Result:** 0 illegal moves across 200 games, and 0 across a further 60 mixed-seat games including bot-vs-itself self-play.

---

## 5. The evaluation harness

Two pieces of infrastructure made the bug hunt tractable:

- **A persistent-process match runner.** Naively, each turn spawns a fresh bot process that reloads the 14 MB model (~0.6 s); a 150-turn game becomes minutes. By keeping each bot alive for a whole game via the "keep running" protocol, evaluation got **~15× faster** and matched how the bot actually runs in production.
- **A legality-plus-strength classifier** that, for each game, labels the bot's outcome as a true win (score > \(+10\)), failure-compensation (\(+10\) next to someone's \(-30\)), a legitimate big deal-in, a draw, or an illegal move — exactly the taxonomy the Simulation-6 analysis taught us to use.

---

## 6. Where the bot stands

Against a mixed field (our heuristic plus random samples), over 200 judged games:

| Metric | Value |
|---|---|
| Illegal moves | **0** |
| True wins | 3% |
| Draws | 74% |
| Average score / game | \(-1.0\) |
| (Deployed heuristic baseline) | \(-12.24\) |

The legality result is solid and meaningful. **Everything else here is preliminary and should not be trusted as a measure of competitive strength** — the 74% draw rate is the tell-tale sign of a weak opponent field, exactly the trap the Simulation-6 report warned about. The only number that will matter is the corrected true-win rate measured on Botzone against real competitors in Simulation-7.

---

## 7. Deployment

Botzone bots upload as **source code** plus a **256 MB Storage folder** mounted at `./data/` at runtime. Our bot ships as a small Python package (`__main__.py` at the root) with the 14 MB model placed in Storage as `data/...weights.npz`. The package **degrades gracefully**: if the official fan library is unavailable in the server's Python, the bot still plays every move legally — it simply cannot declare a win — so an upload can never crash or commit an illegal move. A one-line probe bot reports which dependencies the server actually provides.

---

## 8. What's next

The published literature on this competition is consistent: hand-crafted heuristics lose to supervised learning, and supervised learning loses to **self-play reinforcement learning warm-started from a supervised policy**. Our roadmap follows that arc:

1. Deploy the supervised bot to Simulation-7 and measure the *corrected* true-win rate against real opponents.
2. Feed losses — especially deal-ins — back into the defensive training set.
3. Warm-start **PPO self-play** from the supervised policy, optimizing the duplicate ranking reward rather than single-game win rate, with an opponent pool to keep the policy honest.

The supervised bot is the floor, not the ceiling. But getting the floor *legal and measurable* — and being honest about what our own numbers do and do not prove — was the necessary first step.

---

*This is preliminary competition engineering, not a benchmarked research result. Dataset: the official IJCAI 2026 strong-AI game logs (98,209 games). Hardware: a single RTX 3060 for training, single-core CPU inference for play. All evaluation caveats above are load-bearing.*
