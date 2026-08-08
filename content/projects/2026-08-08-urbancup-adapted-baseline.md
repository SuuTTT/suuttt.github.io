---
title: "The Adapted Baseline: how five GPU-hours won a video-QA competition, and what that says about leaderboards"
date: 2026-08-08
description: "First place in UrbanCup 2026 Track 1 with a five-GPU-hour LoRA adapter. The interesting part is not the win — it is that the benchmark's own paper had already fine-tuned, reported 31, and nobody re-ran it."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["video-QA", "LoRA", "benchmarks", "evaluation", "competition", "vision-language-models"]
---

{{< katex >}}

**Short version:** we won Track 1 of UrbanCup 2026 with a LoRA adapter trained in
about five GPU-hours on one consumer card — 69.4 average accuracy against a best
published baseline of 45.5. The win is not the interesting part. The interesting
part is that the benchmark's own paper *had already fine-tuned*, reported around
31, and that number became the field's working belief about what adaptation is
worth. We have since replicated the effect on three further benchmarks.

We also took second in Track 2 (navigation) with **Avg. SR 30.0** — and that
result raises a question about our own work that I will come to at the end.

## 1. The setup

[UrbanVideo-Bench](https://arxiv.org/abs/2503.06157) (Zhao et al., ACL 2025) asks
a video-language model to reason about first-person city footage — drone and
street level, real cities and a digital twin. Roughly 1,500 clips, 5,355
multiple-choice questions, sixteen task types spanning recall, perception,
reasoning and navigation.

It is a hard benchmark. The strongest baseline in the paper is Qwen-VL-Max at
**45.5%**. Our own zero-shot Qwen2.5-VL-7B managed **29.2%** on held-out
questions — near chance on several categories.

## 2. The recipe, which is deliberately boring

| | |
|---|---|
| Base | Qwen2.5-VL-7B-Instruct |
| Adapter | LoRA rank 32 / α 64, all linear layers, vision tower frozen |
| Objective | **answer-only** — predict the option letter, no rationale |
| Input | 12 frames @ 448 px |
| Decoding | greedy |
| Split | **by video**, 4,452 train / 903 dev, zero clip overlap |
| Compute | 2 epochs ≈ **5 GPU-hours** on one RTX 4080 Super |

The ladder, all on that leak-free split:

| | accuracy |
|---|--:|
| Qwen2.5-VL-7B, zero-shot | 29.2 |
| **The benchmark paper's own fine-tune** | **31.2** |
| Best published baseline (Qwen-VL-Max) | 45.5 |
| + LoRA SFT, **one epoch** | 69.1 |
| + second epoch → champion 7B | 70.9 |
| + 32B log-prob fusion → submitted | 72.9 |
| **Official hidden test set** | **69.4** |

One epoch of supervised fine-tuning more than doubled zero-shot and cleared the
best published baseline by 24 points.

## 3. The part I got wrong, and the correction

My first draft of this story said *"every reference entry on the leaderboard was
zero-shot."* **That was false**, and I am glad it was caught before the talk
rather than after.

UrbanVideo-Bench's own Table 3 carries sections headed *"Fine-Tuning: Training
set"* and *"Fine-Tuning: Test set"*, with InternVL2-4B and 8B adapted by LoRA at
**rank 128 / α 256** on **70% of the questions** and 39.56 GB of video, vision
tower frozen (their Appendix D). That is a **larger** adapter on **more** data
than we used.

It reported **31.4 / 31.2** on the training set and **31.5 / 31.7** on test —
*below* the 45.5 zero-shot best in the same table.

So an adapted baseline existed. And it read as evidence that adaptation does not
help on this benchmark.

This is a better story than "nobody tried", and it is the actual thesis:

> A single under-specified adapted number, published once, becomes a field's
> working belief about what adaptation is worth — and nobody re-runs it.

I want to be careful about blame here, because there is none to assign. The
authors ran the experiment and reported it honestly. Their stated purpose was
validating **sim-to-real transfer**, not establishing a fine-tuning baseline. And
we cannot say which of our six differing choices caused the gap — we ran no
ablation isolating them.

| # | Choice | Their fine-tune | Ours |
|---|---|---|---|
| 1 | Base model | InternVL2-4B / 8B | Qwen2.5-VL-7B |
| 2 | Where LoRA attaches | language model only | all linear layers |
| 3 | Adapter size | rank 128 / α 256 | rank 32 / α 64 |
| 4 | What is graded | not stated | the answer letter only |
| 5 | How data is split | 70% of questions | **by video** — leak-free |
| 6 | Frames per clip | not stated | 12 @ 448 px, greedy |

Two rows say "not stated" because their paper does not say. I have left those
visible rather than filling them in.

## 4. Where the gain actually goes

The 41-point jump is not spread evenly. It concentrates almost entirely in the
categories that were near chance zero-shot:

| Task | zero-shot | adapted |
|---|--:|--:|
| Action Generation | 16.8 | 69.5 |
| Landmark Position | 20.0 | 68.3 |
| Cognitive Map | 24.4 | 88.9 |

Fine-tuning did not polish an existing skill. It **taught the task**. A model
never shown what "the answer" looks like for urban spatial recall cannot fairly
be said to lack the capacity for it.

## 5. Scale did not help, and that was informative

We trained a 32B QLoRA on two A40s. It reached **70.2%** — a statistical tie with
the 7B's 70.9. Four and a half times the parameters bought nothing, which told us
the ceiling is **data, not capacity**, and redirected the rest of the campaign.

What did help was fusing the two by summed per-option log-probability: **72.87%**.
We then built a five-member pool and searched all 26 multi-member subsets
exhaustively:

| | solo |
|---|--:|
| M1 — 7B, 2 epochs (champion) | 70.9 |
| M2 — 32B QLoRA | 70.2 |
| M3 — 7B + preference learning | 69.7 |
| M4 — 7B, different data mix | 70.0 |
| M5 — 7B, different data mix | 70.5 |

All five score 69–71 alone. **Only M1 + M2 helped.** Every subset containing M3,
M4 or M5 scored *lower* — they are error-correlated, so averaging them adds noise.
Ensemble diversity has to come from a real axis of difference; ours was scale.
The oracle bound of the pool — always picking whichever member happened to be
right — was 78.1.

## 6. Five things that did not work

Competition write-ups usually omit this section. It is the part with the most
hours in it.

| Experiment | dev acc. | verdict |
|---|--:|---|
| Chain-of-thought self-distillation | 69.1 | −1.8 |
| Real-world oversampling | 69.1 | net −1.8 (+1.5 real, −sim) |
| Self-consistency voting | 67.0 | −3.9 |
| 24 evaluation frames instead of 12 | 71.0 | +0.1, noise |
| Preference learning on own errors | regressed | — |

Every one of these is a decoding-side or objective-side intervention, and every
one failed. The pattern is consistent enough to state as a rule: **after
in-domain fine-tuning, the residual errors are data-limited, not
decoding-limited.**

That phrase deserves unpacking, since it decides where to spend the next GPU-day.
*Decoding-limited* means the right answer is already sitting in the model's
distribution and a better extraction procedure recovers it. *Data-limited* means
the distribution itself is wrong and no extraction helps. Self-consistency voting
is the direct test — it lost 3.9 points, which means greedy was already returning
the argmax of a very peaked distribution.

## 7. The finding worth more than the win

We split the adapted model's accuracy by where the video came from:

| | accuracy |
|---|--:|
| Simulator footage | **82.6** |
| Real drone footage | **62.7** |

**Twenty points.** Same model, same questions, same evaluation run. On real-world
*Progress Evaluation* — judging whether you are getting closer to your goal — it
scored **zero**.

This also explains the one awkward number in the ladder. Our dev 72.9 became
**69.4** on the hidden set, and the likeliest reason is simply that the hidden set
leans more real-world than our dev split did. We were partly trained on a world
easier than the one we were tested in.

So the honest conclusion from a first-place run is not "fine-tuning solves urban
video QA." It is that **the remaining headroom is in real footage, not bigger
models** — and that any benchmark mixing rendered and real video should report the
two separately, because one averaged number hides a twenty-point failure.

## 8. Does it generalise? Three more benchmarks say yes

The obvious objection to all of the above is that it is one benchmark. So we ran
the same fixed recipe and the same 7B model on three further video benchmarks,
with a group-disjoint split each time, three random seeds per adapted arm:

| Benchmark | n | zero-shot | adapted (3 seeds) | gap | s.d. |
|---|--:|--:|---|--:|--:|
| UrbanVideo-Bench (anchor) | 903 | 29.2 | 70.9 | **+41.7** | 1 seed |
| STI-Bench | 1,003 | 31.9 | 53.04 · 53.34 · 52.34 | **+21.0** | 0.51 |
| MM-UAVBench | 2,441 | 42.2 | 50.23 · 49.08 · 50.84 | **+7.9** | 0.89 |
| CG-Bench | 4,623 | 27.2 | 60.35 · 59.57 · 59.03 | **+32.4** | 0.66 |

Four benchmarks, four positive gaps, and the smallest effect is about **nine seed
standard deviations** from zero. That is the opposite of the decay pattern I have
watched kill three previous campaigns, where single-seed findings shrank toward
nothing as seeds accrued. These did not move.

The per-category decomposition explains the spread, and it is the same mechanism
as the anchor. On MM-UAVBench:

| Task | zero | adapted | gap |
|---|--:|--:|--:|
| Swarm Collaborative Planning | 27.6 | 68.3 | **+40.7** |
| Scene Damage Assessment | 32.3 | 61.4 | +29.1 |
| Orientation Classification | 26.8 | 32.9 | +6.1 |
| Target Backtracking | 61.5 | 61.5 | 0.0 |
| Urban OCR | 57.8 | 55.3 | **−2.5** |

**Adaptation gain is inversely proportional to zero-shot competence.** Where the
model already knows the task, it gains nothing or slightly loses. MM-UAVBench's
modest +7.9 overall is therefore not a weak result — it is a benchmark full of
categories the base model already handles.

## 9. What this does not show

Three limits, stated plainly because a reviewer will find them anyway.

**Every number here is our protocol on our split.** None can sit beside a
published leaderboard row, because for evaluation-only benchmarks the test labels
are not public. That is partly the point — anyone computing an adapted baseline
hits the same wall — but it does mean we criticise a leaderboard with a number
that cannot go on it. Closing this properly needs a benchmark with a real
train/test split; we are working on it.

**One model family.** All of the above is Qwen2.5-VL. A cross-family replication
with LLaVA-NeXT-Video is running as I write this; its zero-shot on STI-Bench came
in at 20.4% against Qwen's 31.9, with roughly 17% of outputs not even being option
letters — which is itself the answer-format half of the thesis showing up in a
second architecture.

**A starved configuration.** Two frames at 128 px for training, because the
available 24 GB card would not take more. The gaps are therefore probably *lower*
bounds.

## 10. Track 2, and an uncomfortable question about our own work

We also placed second in Track 2 — real-time interactive goal-oriented navigation
— with **Avg. SR 30.0**, the organizers running our policy closed-loop in their
simulator. For scale, the benchmark paper reports human performance at 92% SR and
the best published model at 34%; zero-shot VLA baselines score 1–3%.

That 30.0 is the only closed-loop success rate we have. And it points at something
worth being honest about.

Our navigation research since the competition has produced a careful result — an
exposure-bias gap cut from 9.9 to 2.0 points across six seeds, with a
hierarchical cluster bootstrap. But every one of those numbers is an **offline
next-action match rate**, not a success rate. The two are not interchangeable, and
the relationship between them is not something we have measured.

It is tempting to convert. Our best self-history match rate is about 58.6%. If
steps were independent over a ~19-step episode, that implies \\( 0.586^{19} \approx 0.003\% \\) success — obviously absurd, since the real system scores 30.
The absurdity is the point: step accuracy and success rate are related by *error
recoverability*, which we have never measured. A policy 58% right per step could
land anywhere from 5% to 40% SR.

So: we have one closed-loop number, it came from a submission frozen in July, and
none of the research we have done since has been closed-loop evaluated at all.
That is the next thing to fix, and it is why I would rather publish this paragraph
than a converted estimate.

## Takeaways

1. **Fine-tune the in-domain data before anything else.** Zero-shot leaderboards
   are a lower bound on the field, not a measurement of it.
2. **Diversify ensembles by scale or architecture, never by seed.** Four
   same-size members added exactly zero.
3. **Split your score by subgroup before you trust the average.** A twenty-point
   simulated-versus-real failure was invisible in ours.
4. **Publish the negative ledger.** Five refusals cost us GPU-days; sharing them
   is the cheapest contribution a competition entry can make.
5. **Never compare across metric axes.** A next-action match rate is not a
   success rate, and no arithmetic converts one to the other.

---

*Team moyu, UrbanCup 2026 (城市具身智能与世界模型挑战赛): Track 1 first place,
Track 2 second place. Presented at the 4th International Conference on Urban
Science and Intelligence, HKUST(GZ), 9 August 2026. Benchmark:
[UrbanVideo-Bench](https://arxiv.org/abs/2503.06157). Base model: Qwen2.5-VL-7B.
Training: ms-swift. Serving: vLLM.*

*The multi-benchmark results in §8 are unpublished work in preparation and have
not been peer-reviewed.*
