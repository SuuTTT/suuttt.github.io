---
title: "Chinese Standard Mahjong, Explained for AI Researchers"
date: 2026-05-31
description: "A from-zero primer on Chinese Standard Mahjong as a reinforcement-learning problem: the tiles, the sets, shanten, and the 8-fan economy that makes 'getting ready' and 'being able to win' two different goals — the single idea that determines whether a bot wins or draws every game."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["mahjong", "game-ai", "reinforcement-learning", "imitation-learning", "tutorial", "botzone"]
---

{{< katex >}}

## Why a Mahjong primer?

If you work in game AI you know Go, chess, poker, and maybe Hanabi. **Chinese Standard
Mahjong (国标麻将)** is less familiar to Western researchers but is a wonderful testbed: it is
a **four-player, imperfect-information, stochastic, sequential game with a sparse and oddly
shaped reward.** This post teaches just enough of the game to understand *why it is hard for
RL*, with no prior Mahjong knowledge assumed. It is the companion primer to our build notes
for the IJCAI 2026 Mahjong AI Competition.

---

## 1. Tiles and the goal

There are **34 tile types**, **4 copies each** → 136 tiles. Three number "suits" run 1–9:

- **W** (characters / 万), **B** (dots / 饼), **T** (bamboo / 条)

plus **honor** tiles (4 winds, 3 dragons) that have no numeric order.

Each player holds **13 tiles**. On your turn you **draw** a 14th and **discard** one, going
around the table. A **winning hand** is **4 sets + 1 pair**, where a *set* is either:

- a **triplet** — three identical tiles, e.g. \(\text{W2 W2 W2}\); or
- a **sequence** — three consecutive in one suit, e.g. \(\text{T4 T5 T6}\)
- (a **kong** — four identical — is a special set that grants an extra draw)

So a complete hand has \(4 \times 3 + 2 = 14\) tiles. Example:

$$ \underbrace{\text{W2 W3 W4}}_{\text{seq}}\;\; \underbrace{\text{B5 B6 B7}}_{\text{seq}}\;\; \underbrace{\text{T1 T1 T1}}_{\text{triplet}}\;\; \underbrace{\text{T7 T8 T9}}_{\text{seq}}\;\; \underbrace{\text{J1 J1}}_{\text{pair}} $$

## 2. Claiming other players' discards

You don't only draw from the wall — you can **claim a tile someone discards** to finish a set:

- **Peng (碰)**: you hold two of a tile; claim a third anyone discards → exposed triplet.
- **Chi (吃)**: claim a discard to complete a **sequence** — but only from the player to your **left**.
- **Gang (杠)**: make a four-of-a-kind (kong).

Claiming reveals that set face-up and jumps you straight to discarding. Claims are how you
**speed up** — and, as we'll see, how you **commit** to a plan.

---

## 3. Shanten: distance to "ready"

A natural progress metric is **shanten** (上听数): the minimum number of tile swaps until your
hand is **ready** (one tile away from completion). Shanten 0 is *tenpai* (ready); higher is
further away. A rough lower bound used in practice is

$$ \text{shanten} \approx 8 - 2 m - \max(1,\, t + p) $$

where \(m\) is completed sets, \(t\) partial sets, \(p\) pairs. Minimizing shanten greedily —
"always take the move that gets me closest to ready" — is the obvious baseline policy.

**And it is exactly the trap.** Here is why.

---

## 4. The twist that defines the game: the 8-fan floor

In Chinese Standard Mahjong, **a complete 4-sets-+-pair hand is *not* automatically a win.**
It must also score at least **8 fan (番)**. Fan are points awarded for *patterns*, from a
table of 81. If you declare a win worth fewer than 8 fan, it is an **illegal move** with a
heavy penalty.

So there are really **two objectives, and they conflict**:

1. **Be ready** (low shanten).
2. **Be worth ≥ 8 fan** when you get there.

A plain hand of mixed sequences scores ~0–4 fan — *complete but unwinnable*. The points live
in **structured** hands you must commit to early:

| Pattern | Description | Fan |
|---|---|---|
| 清一色 (full flush) | all tiles in **one suit** | 24 |
| 组合龙 (knitted straight) | 1-4-7 / 2-5-8 / 3-6-9 across the three suits | 12 |
| 混一色 (half flush) | **one suit + honors** only | 6 |
| 碰碰和 (all triplets) | every set is a triplet | 6 |

Reaching the 8-fan floor almost always means **sacrificing speed for value** — passing a
tempting claim that would make you "ready" cheaply, in order to chase a hand that is actually
worth winning.

### The reward, stated cleanly

Let \(f\) be fan above the floor. A self-drawn win pays the winner \(3(8+f)\) and each
opponent \(-(8+f)\); a win on a discard pays the winner \(24+f\), the discarder \(-(8+f)\),
and the other two \(-8\). An illegal action pays \(-30\) to the offender and \(+10\) to each
of the others. So the reward is **sparse** (most hands end in a draw worth 0), **highly
asymmetric**, and **dominated by a legality cliff** — getting the 8-fan rule wrong costs more
than most wins earn.

---

## 5. Why a naive bot draws every single game

Picture a shanten-minimizing bot. You hold \(\text{W2 W2}\) and an opponent discards
\(\text{W2}\). Peng-ing instantly forms a triplet and lowers shanten, so the greedy bot takes
it. But if your other tiles are a three-suit jumble, you have now rushed to *ready* on a hand
worth maybe **2 fan**. You are one tile from a "win" you can **never legally declare**. You
sit there discarding until the wall is exhausted, and the game ends **0–0–0–0**.

This is not hypothetical — it is exactly what our first bots did: clean, legal, fast play, and
a draw in ~75% of games, with the best hand anyone reached worth only 4 fan. The bug was not
in the code; it was in the **objective**. A shanten-only rule is **fan-blind**: it optimizes
*distance* and ignores *value*, so it never deliberately builds 清一色 or 碰碰和 — the only
hands that clear the 8-fan bar.

The fix mirrors what a human expert does: **don't peng that W2.** Keep collecting one suit,
or peng only tiles that build all-triplets. Trade speed for value, because a fast sub-8-fan
hand is worthless.

---

## 6. Why this makes Mahjong a good RL problem

The 8-fan floor turns Mahjong into a genuinely strategic planning problem rather than a
greedy-completion race:

- **Long-horizon credit assignment**: the decision that wins the game (committing to a suit)
  happens dozens of moves before the payoff.
- **Value vs. speed trade-off**: the optimal policy routinely *declines* locally-greedy moves.
- **Imperfect information + opponents**: you infer others' hands from discards, and a careless
  discard can hand an opponent a big win (放铳 / dealing in).
- **A legality cliff**: the action space must be masked and win-declarations verified, or a
  single mistake erases a match.

It is precisely because "getting ready" and "being able to win" are different goals that
hand-written heuristics plateau and **learning** — imitation from strong players, then
self-play — becomes necessary. A policy trained on millions of expert decisions implicitly
encodes the fan economy: it claims the tiles experts claim and passes the ones they pass,
because it learned *what good play looks like*, not just *what is closest to ready*.

---

## 7. Takeaways

- A Mahjong win = **4 sets + 1 pair** *and* **≥ 8 fan**. The second condition is the whole game.
- **Shanten** measures speed-to-ready; **fan** measures value. Optimizing only the first is the
  classic beginner (and beginner-bot) mistake — it draws every game.
- High value comes from **committing early** to structured hands (清一色, 碰碰和, 混一色, 组合龙).
- This speed-vs-value tension, plus imperfect information and a legality cliff, is what makes
  Chinese Standard Mahjong a rich reinforcement-learning benchmark — and why data-driven
  policies beat hand-crafted rules.

*Next in this series: how we turned those ideas into a bot — feature design, supervised
learning on 98k expert games, and the move from a fan-blind heuristic to a model-driven policy.*
