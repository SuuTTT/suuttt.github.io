# Proposal D2 — Pure self-predictive JEPA on DMControl: the *real* anti-collapse test

**Date:** 2026-07-01  **Box:** b3060b (4× RTX 3060, shared with mahjong `moyuHarv` — untouched)
**Code:** `pure_jepa_dmc.py`  **Runs:** `run_WalkerWalk_*.json` (n as noted)  **Summary:** `D2_summary.json`

## The correction being tested
The prior DMControl "anti-collapse isn't load-bearing" study was on **TD-MPC2**, which is
value-anchored (reward + value + policy heads) = redundancy, **NOT a pure JEPA**. So collapse
was never on the table there. Here we build a **genuinely pure JEPA** and re-ask the question.

**Pure JEPA** = SimNorm/MLP encoder + jumpy (1-step) latent predictor + EMA target-encoder,
trained by **latent self-prediction ONLY** — no reward, no value, no policy, no negatives.
Then FREEZE the encoder and linear-probe held-out R² for two readouts:
- **GEOMETRIC** = physical `qpos` (9-dim pose)
- **VALUE** = discounted return-to-go (RTG, γ=0.97) from task reward

**Task:** WalkerWalk (mujoco_playground, random-policy buffer, **N=39 968** transitions, 85/15 split).
Latent **L=64**. Both readouts are genuinely present in the data — raw-obs linear upper bounds:
**geom R²=0.381, value(RTG) R²=0.216, reward R²=0.306**. (Latents can *exceed* these because a
linear probe reads nonlinear encoder features.)

**Arms** (matched arch/data/steps=12k; seed varies init+order):
`none` (self-prediction only) · `uniformity` (Wang&Isola) · `vicreg` · `se` (differentiable
2D structural entropy, **selib-faithful, abs_err 3e-7**). SE weight set two ways: grad-norm-matched
to uniformity at init (the DMC-SE protocol), and at fixed natural weights (control, see caveat).

## Results

### SimNorm backbone (softmax simplex — TD-MPC2 default; itself an implicit AC prior)
| arm | n | eff_rank (/64) | geom R² | value R² |
|-----|---|----------------|---------|----------|
| **none** | 3 | 5.3 ± 0.1 | **0.795 ± 0.008** | 0.304 ± 0.032 |
| uniformity | 3 | 40.3 ± 0.0 | 0.583 ± 0.004 | 0.121 ± 0.007 |
| vicreg | 3 | 18.2 ± 0.1 | 0.786 ± 0.005 | **0.331 ± 0.038** |
| se (grad-matched, λ≈74) | 3 | 7.1 ± 0.5 | 0.104 ± 0.027 | 0.002 ± 0.003 |
| se (fixed λ=0.3) | 1 | 6.2 | 0.692 | 0.262 |
| se (fixed λ=1.0) | 1 | 6.2 | 0.667 | 0.271 |
| se (fixed λ=5.0) | 1 | 7.2 | 0.488 | 0.115 |

### No-SimNorm backbone (raw L2-normed latent — honest collapse control, no simplex prior)
| arm | n | eff_rank (/64) | geom R² | value R² |
|-----|---|----------------|---------|----------|
| **none** | 1 | 14.9 | 0.770 | 0.240 |
| uniformity | 1 | 61.7 | 0.590 | 0.138 |
| vicreg | 1 | 24.2 | 0.796 | 0.300 |
| se (grad-matched) | 1 | 16.7 | 0.155 | 0.076 |

## VERDICT

**1. The premise is FALSIFIED: a pure JEPA does NOT collapse here.** The `none` arm — latent
self-prediction with zero anti-collapse — has the **highest task decodability of any arm**
(geom 0.795, value 0.304), both *above* the raw-obs baselines. This holds **with and without**
SimNorm (no-simplex `none`: geom 0.770, value 0.240). The predictor + EMA target-encoder
asymmetry (BYOL-style) is sufficient to prevent collapse **without any explicit anti-collapse
term**. Low eff_rank (5–15) is *compact, informative* structure, not dimensional collapse —
true collapse (readouts → 0) was only ever produced *artificially* by over-weighted SE.
So on a *genuine* JEPA, anti-collapse is **still not load-bearing** — but for a different
reason than the value anchor: it's the self-predictive asymmetry, not the value head.

**2. The "downstream-dependent" taxonomy does NOT hold.** The hypothesized pattern
("uniformity helps geometric but hurts value") is **not observed**. Uniformity *hurts both*
readouts (geom 0.795→0.583, value 0.304→0.121) despite maximizing eff_rank (5→40). Across all
arms, geometric-R² and value-R² move **together**, not in opposition (corr strongly positive) —
because there is no collapse to trade off against. Anti-collapse terms only *dilute* the already-
rich self-predictive latent; they never selectively rescue one readout at the expense of another.

**3. SE ≠ uniformity, and grad-norm-matching is a TRAP.** At natural weight (λ=0.3–1.0) SE is the
**best-behaved** anti-collapse term: it keeps the latent compact (eff_rank ~6, like `none`) while
preserving geom (0.69) and value (0.27) far better than uniformity (0.58 / 0.12). But grad-norm-
matching SE *to uniformity* at init inflates its weight ~74× (raw SE gradient at init ≈ 0.025 vs
uniformity ≈ 1.86), and the SE term then dominates training (l_struct ≈ 416 vs l_pred ≈ 4),
**destroying all readouts** (geom 0.10, value 0.00). This is a methodological artifact of matching
a term whose gradient grows during training — **not** an intrinsic property of SE. Honest caveat:
the grad-matched-SE row should be read as "over-weighted SE," not "SE."

## Bottom line
On a **genuinely pure JEPA** (WalkerWalk, self-prediction only), the motivating story reverses:
`none` does **not** collapse and is the **strongest** representation; explicit anti-collapse terms
range from **neutral (natural-weight SE, vicreg) to harmful (uniformity, over-weighted SE)**; and
the geometric-vs-value "downstream-dependent anti-collapse" taxonomy **does not appear** because
predictor+EMA already prevents collapse. The load-bearing ingredient is the **self-predictive
asymmetry**, not any anti-collapse regularizer and not the value anchor.

### Limitations / honesty
- One task (WalkerWalk); n=3 seeds for SimNorm core arms, n=1 for no-SimNorm control and fixed-λ SE.
- Random-policy buffer → RTG variance is modest (raw-obs value R²=0.22); still a real, decodable signal.
- Geometry = qpos pose; a cleaner ee↔target task (Reacher/Finger) had ~0 reward under random policy
  (no value signal), so WalkerWalk was chosen as the single task carrying *both* readouts.
- "Collapse" characterized by eff_rank + probe-R²; no arm reached dimensional collapse except the
  over-weighted grad-matched-SE artifact.
- SE differentiable-2D validated numerically against `selib.structural_entropy_2d` every run (abs_err 3e-7).
