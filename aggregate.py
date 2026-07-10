"""Aggregate per-run JSONs into summary.json + VERDICT.md. All numbers from disk."""
import argparse, json, glob, os, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--outdir", required=True)
args = p.parse_args()

files = sorted(glob.glob(os.path.join(args.outdir, "*.json")))
runs = []
for f in files:
    if os.path.basename(f) in ("summary.json",):
        continue
    runs.append(json.load(open(f)))

# group by (env, arm)
groups = {}
for r in runs:
    groups.setdefault((r["env"], r["arm"]), []).append(r)

summary = {"budget_env_steps_per_run": None, "cells": {}}
rows = []
for (env, arm), rs in sorted(groups.items()):
    steps = rs[0]["total_steps"]
    summary["budget_env_steps_per_run"] = steps
    sr = [x["success_rate_last100"] for x in rs]
    sro = [x["success_rate_overall"] for x in rs]
    solved = [x["solved"] for x in rs]
    fss = [x["first_success_step"] for x in rs if x["first_success_step"] is not None]
    cov = [x["cum_unique_cells_all_envs"] for x in rs]
    epcov = [x["mean_ep_coverage"] for x in rs]
    entry = {
        "env": env, "arm": arm, "n_seeds": len(rs),
        "seeds": [x["seed"] for x in rs],
        "budget_steps": steps,
        "n_solved_seeds": int(sum(solved)),
        "success_rate_last100_mean": round(float(np.mean(sr)), 4),
        "success_rate_last100_per_seed": [round(x, 4) for x in sr],
        "success_rate_overall_mean": round(float(np.mean(sro)), 4),
        "first_success_step_mean": (round(float(np.mean(fss))) if fss else None),
        "first_success_step_per_seed": [x["first_success_step"] for x in rs],
        "cum_unique_cells_mean": round(float(np.mean(cov)), 1),
        "cum_unique_cells_per_seed": cov,
        "mean_ep_coverage_mean": round(float(np.mean(epcov)), 2),
    }
    summary["cells"][f"{env}|{arm}"] = entry
    rows.append(entry)

with open(os.path.join(args.outdir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# VERDICT.md
lines = ["# MiniGrid Hard-Exploration: PPO vs PPO+RND (novelty)\n"]
lines.append(f"Budget: {summary['budget_env_steps_per_run']:,} env-steps/run, "
             f"n={rows[0]['n_seeds'] if rows else 0} seeds/cell, CPU-only.\n")
lines.append("Maps: MiniGrid-MultiRoom-N6-v0 (6 rooms, sparse), "
             "MiniGrid-KeyCorridorS3R3-v0 (key+locked door, deceptive).\n")
lines.append("Obs: FullyObs->flatten. Reward: sparse extrinsic ONLY (no shaping). "
             "Success = terminated at goal (REAL). Coverage = unique (x,y) cells "
             "across all 8 vec-envs. Discovery = env-step of first success.\n")
lines.append("\n| Env | Arm | solved seeds | SR(last100) mean | SR per-seed | "
             "first-success step | cum unique cells | mean ep cov |")
lines.append("|---|---|---|---|---|---|---|---|")
for r in rows:
    env_short = r["env"].replace("MiniGrid-", "").replace("-v0", "")
    fss = r["first_success_step_per_seed"]
    lines.append(
        f"| {env_short} | {r['arm']} | {r['n_solved_seeds']}/{r['n_seeds']} | "
        f"{r['success_rate_last100_mean']} | {r['success_rate_last100_per_seed']} | "
        f"{fss} | {r['cum_unique_cells_mean']} | {r['mean_ep_coverage_mean']} |"
    )

# auto verdicts per env
lines.append("\n## Verdict\n")
by_env = {}
for r in rows:
    by_env.setdefault(r["env"], {})[r["arm"]] = r
for env, arms in by_env.items():
    es = env.replace("MiniGrid-", "").replace("-v0", "")
    ppo = arms.get("ppo"); rnd = arms.get("rnd")
    if ppo and rnd:
        pp = ppo["success_rate_last100_mean"]; rr = rnd["success_rate_last100_mean"]
        pcov = ppo["cum_unique_cells_mean"]; rcov = rnd["cum_unique_cells_mean"]
        verdict = ("RND SOLVES where PPO stalls" if rr > 0.05 and pp <= 0.05
                   else "both stall (0 success)" if rr <= 0.05 and pp <= 0.05
                   else "PPO already solves" if pp > 0.05 and rr <= 0.05
                   else "both make progress")
        cov_v = ("RND explores MORE" if rcov > pcov * 1.1
                 else "PPO explores MORE" if pcov > rcov * 1.1
                 else "coverage ~equal")
        lines.append(f"- **{es}**: success -> {verdict} (PPO {pp} vs RND {rr}); "
                     f"coverage -> {cov_v} (PPO {pcov} vs RND {rcov} cells).")

with open(os.path.join(args.outdir, "VERDICT.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("Aggregated", len(runs), "runs ->", args.outdir)
print("\n".join(lines))
