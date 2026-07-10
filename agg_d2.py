"""Aggregate pure-JEPA DMControl runs -> per-arm table + taxonomy verdict."""
import json, glob, numpy as np
from pathlib import Path
ROOT = Path("/root/tdmpc_glass/exp/proposal_D2_pure_jepa")
runs = [json.loads(Path(f).read_text()) for f in glob.glob(str(ROOT/"run_*.json"))]

# raw-obs baselines (computed once, hardcoded from probe)
BASE = {"geom": 0.381, "value": 0.216, "reward": 0.306}

def agg(sel):
    g = [r["geom_r2"] for r in sel]; v = [r["value_r2"] for r in sel]; e = [r["eff_rank"] for r in sel]
    n = len(sel)
    m = lambda x: (float(np.mean(x)), float(np.std(x)))
    return n, m(g), m(v), m(e)

order = ["none","uniformity","vicreg","se"]
lines = []
lines.append("# Proposal D2 — Pure self-predictive JEPA on DMControl: anti-collapse taxonomy\n")
lines.append("Pure JEPA = encoder + jumpy(1-step) latent predictor + EMA target-encoder, trained by")
lines.append("latent self-prediction ONLY (NO reward/value/policy). Task: WalkerWalk (mujoco_playground,")
lines.append("random-policy buffer N=39968). Latent L=64. Freeze encoder -> held-out ridge probe R^2.")
lines.append("Geometric readout = physical qpos (9-dim pose). Value readout = discounted return-to-go (RTG, gamma=0.97).")
lines.append("SE = differentiable 2D structural-entropy (selib-faithful, abs_err 3e-7), GRAD-NORM-MATCHED to uniformity at init.\n")
lines.append(f"Raw-obs upper-bound probe (linear): geom R2={BASE['geom']}, value(RTG) R2={BASE['value']}, reward R2={BASE['reward']}\n")

for sn, label in [(1,"SimNorm backbone (softmax simplex — TD-MPC2 default; implicit AC prior)"),
                  (0,"NO-SimNorm backbone (raw L2-normed latent — honest collapse control)")]:
    lines.append(f"\n## {label}\n")
    lines.append(f"| arm | n | eff_rank(/64) | geom R2 | value R2 |")
    lines.append(f"|-----|---|--------------|---------|----------|")
    for cond in order:
        sel = [r for r in runs if r["cond"]==cond and r["simnorm"]==sn]
        if not sel: continue
        n,(gm,gs),(vm,vs),(em,es) = agg(sel)
        lines.append(f"| {cond} | {n} | {em:.1f}±{es:.1f} | {gm:.3f}±{gs:.3f} | {vm:.3f}±{vs:.3f} |")

# machine-readable summary
summ = {}
for sn in (1,0):
    for cond in order:
        sel = [r for r in runs if r["cond"]==cond and r["simnorm"]==sn]
        if not sel: continue
        n,(gm,gs),(vm,vs),(em,es) = agg(sel)
        summ[f"sn{sn}_{cond}"] = {"n":n,"eff_rank":round(em,2),"eff_rank_sd":round(es,2),
                                  "geom_r2":round(gm,4),"geom_r2_sd":round(gs,4),
                                  "value_r2":round(vm,4),"value_r2_sd":round(vs,4)}
(ROOT/"D2_summary.json").write_text(json.dumps({"baselines":BASE,"arms":summ},indent=2))
print("\n".join(lines))
print("\n[wrote D2_summary.json]")
