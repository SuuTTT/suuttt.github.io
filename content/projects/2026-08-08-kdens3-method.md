---
title: "CNN + Augmented BC + Distill-Then-Ensemble: The Method Behind kdens3"
date: 2026-08-08
description: "A concise writeup of the three techniques that shipped in IJCAI 2026's Chinese Standard Mahjong competition (2nd of 16) — architecture, data augmentation, distill-then-ensemble — and the full deployed workflow."
layout: "post"
tags: ["mahjong", "imitation-learning", "distillation", "ensemble-learning", "IJCAI-2026"]
---

**Result:** kdens3 finished 2nd of 16 in the IJCAI 2026 Chinese Standard Mahjong (MCR) competition. It is a behavior-cloning policy — no search, no reinforcement learning at inference time. Three techniques make up the entire shipped method.

## 1. Architecture

A residual CNN with batch normalization: **128 channels × 40 residual blocks** (~14.3M parameters). Input is ~38 feature planes encoding the 34 tile types plus auxiliary planes for hand, melds, discards, and public game state. Output is a 235-dimensional discrete action space (discards plus Peng/Chi/Gang/Hu), masked at inference so the policy only ever proposes legal moves.

## 2. Data and augmentation

Base corpus: **5.87M decisions**, cleaned from 98,209 real ranked matches. This is behavior cloning — the model imitates recorded human/expert decisions, not self-play. Each decision stores the full board observation, the legal-action mask, and the action actually taken.

Training uses heavy on-GPU augmentation composed from three label-preserving symmetry groups — **suit permutation (6) × rank reflection (2) × dragon-tile permutation (6)** — which multiply the effective training data without touching the underlying corpus. Combined with label smoothing, EMA weights, and a cosine learning-rate schedule:

```
python3 e11_train.py --channels 128 --blocks 40 --steps 130000 \
    --seed 0 --bs 1024 --lr 3e-4 --wd 1.5e-4 --lsm 0.05 --ema 0.999 \
    --p_suit 0.8 --p_ref 0.5 --p_drag 0.5 --out ckpt/aug/aug_128x40_s0.pkl
```

Six of these are trained independently — same recipe, same data, only the seed differs. That pool of six feeds the next stage.

## 3. Distillation, then ensemble

Every training batch, all six teachers score the batch; their softmax outputs (masked to legal moves) are averaged into a single soft target. Three fresh students are trained against that target, blended with the original hard labels:

```
tsoft = mean_i( softmax(teacher_i_logits, legal_mask) )        # i = 1..6
loss  = α · cross_entropy(student_logits, tsoft)                # match the teacher pool
      + (1-α) · label_smoothed_ce(student_logits, human_action) # α = 0.7
```

At inference, the three students' legal-move softmax outputs are averaged and the policy acts on the argmax — the same mean-softmax mechanism used to build the teacher target, one level up. This is the deployed model: **kdens3**, a mean-softmax ensemble of 3 distilled students.

## The complete workflow

1. **Data.** Unzip the official 98,209-game dataset → `preprocess_single.py` → `cooked_single.npz` (5.87M decisions: `obs` / `mask` / `act` arrays).
2. **Judge.** Build the official C++ Chinese Standard Mahjong judge for local self-play evaluation.
3. **Six teachers.** Run `e11_train.py` six times, seeds 0–5, identical hyperparameters, on the full corpus.
4. **Three students.** Run `e13_kd_train.py` three times, seeds 0–2, `--teachers` set to all six teacher checkpoints, `--alpha 0.7`, 90k steps each.
5. **Fuse for deployment.** Fold BatchNorm into the convolution weights; load through a numpy-only runtime so the competition platform's torch-free environment can run it.
6. **Verify.** Confirm argmax parity between the fused numpy weights and the original torch checkpoints on a held-out batch; store as fp16 (halves file size, compute stays fp32).
7. **Deploy.** Lock the three fused, verified student checkpoints as the entry. Inference is the 3-way mean-softmax ensemble above.

One architecture, one augmentation recipe, one distill-then-ensemble step — that's the entire method.
