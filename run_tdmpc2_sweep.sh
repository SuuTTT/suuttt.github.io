#!/bin/bash
# TD-MPC2 arm for PPO-wall generalization, b3060 GPU0/1 (idle post-MiniGrid).
# Fresh paired 2-seed runs on the 3 tasks TD-MPC2 is known to solve.
# Uses helios-rl run_benchmark (vanilla tdmpc2). Per-seed tag-aware CSVs.
# NO --save_full_state. Writes fresh-summary JSON + DONE marker. Only OUR pids.
set -u
REPO=/root/helios-rl
PY=$REPO/.venv/bin/python
BASE=$REPO/exp/ppo_wall_generalization
mkdir -p "$BASE"
TASKS=${TASKS:-"PendulumSwingup FingerTurnHard BallInCup"}
STEPS=${STEPS:-1000000}
SEEDS=${SEEDS:-"1 2"}
TAG=ppowall

export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.28
export MUJOCO_GL=egl
export MJPG_IMPL=jax
export TDMPC_GLASS_OUTPUT_TAG=$TAG

cd "$REPO" || exit 1

launch () { # task seed gpu
  local task=$1 seed=$2 gpu=$3
  CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/run_benchmark.py \
    --algos tdmpc2 --tasks "$task" --total_steps "$STEPS" --seed "$seed" --no_plot \
    > "$BASE/tdmpc2_${task}_s${seed}.log" 2>&1
}

pids=()
for task in $TASKS; do
  # seed1 -> GPU0, seed2 -> GPU1 (TD-MPC2-jax ~0.7GB each, plenty of room)
  g=0
  for s in $SEEDS; do
    launch "$task" "$s" "$g" &
    pids+=($!)
    g=$((g+1)); [ "$g" -gt 1 ] && g=0
    sleep 20
  done
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

# Harvest fresh peaks into tdmpc2_fresh.json (real, read from per-seed CSVs).
$PY - "$BASE" $TASKS <<'PYEOF'
import csv, glob, json, os, sys
base = sys.argv[1]; tasks = sys.argv[2:]
gl = "/root/helios-rl/exp/tdmpc_glass"
out = {}
for t in tasks:
    peaks=[]; finals=[]; steps=[]; ns=0
    for csvf in sorted(glob.glob(os.path.join(gl, f"{t}_ppowall", "seed_*.csv"))):
        try:
            rows=list(csv.DictReader(open(csvf)))
        except Exception:
            continue
        r=[float(x["reward"]) for x in rows if x.get("reward") not in (None,"")]
        s=[int(x["step"]) for x in rows if x.get("step") not in (None,"")]
        if r:
            peaks.append(max(r)); finals.append(r[-1]); steps.append(max(s) if s else 0); ns+=1
    if peaks:
        out[t]={"peak":round(sum(peaks)/len(peaks),1),"peak_best":round(max(peaks),1),
                "final":round(sum(finals)/len(finals),1),"max_step":max(steps),
                "n":ns,"source":"fresh 2-seed b3060 run_benchmark tdmpc2"}
json.dump(out, open(os.path.join(base,"tdmpc2_fresh.json"),"w"), indent=2)
print("fresh summary:", json.dumps(out))
PYEOF

echo "tdmpc2_sweep_finished tasks=[$TASKS] steps=$STEPS seeds=[$SEEDS] fail=$fail $(date -u +%FT%TZ)" > "$BASE/TDMPC2_SWEEP_DONE"
