#!/bin/bash
# PPO exploration-wall GENERALIZATION sweep on b3060b GPU2/3 (idle).
# 3 exploration-relevant DMControl tasks where TD-MPC2 succeeds, large PPO budget.
# Per task: N seeds concurrent, split across GPU2 and GPU3. Real return-vs-steps JSON.
# NO checkpoints, NO videos, NO save_full_state. Writes DONE markers.
set -u
BASE=/root/tdmpc_glass/exp/ppo_wall_generalization
DRIVER=$BASE/ppo_driver.py
OUT=$BASE/runs
TASKS=${TASKS:-"PendulumSwingup FingerTurnHard BallInCup"}
BUDGET=${BUDGET:-60000000}
NEVALS=${NEVALS:-31}
SEEDS=${SEEDS:-"1 2 3"}
mkdir -p "$OUT"
cd /root/tdmpc_glass || exit 1
source venv/bin/activate

launch () { # env seed gpu
  local env=$1 seed=$2 gpu=$3
  mkdir -p "$OUT/$env"
  CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_MEM_FRACTION=.3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python "$DRIVER" --env "$env" --seed "$seed" \
      --num_timesteps "$BUDGET" --num_evals "$NEVALS" --outdir "$OUT/$env" \
      > "$OUT/$env/seed${seed}.log" 2>&1
}

for env in $TASKS; do
  echo "=== $env $(date -u +%FT%TZ) ===" >> "$BASE/sweep_master.log"
  pids=(); i=0
  for s in $SEEDS; do
    if [ $((i % 2)) -eq 0 ]; then g=2; else g=3; fi
    launch "$env" "$s" "$g" &
    pids+=($!); i=$((i+1))
    sleep 25   # stagger XLA compile to avoid simultaneous VRAM spikes
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  echo "task_done env=$env fail=$fail $(date -u +%FT%TZ)" >> "$BASE/sweep_master.log"
done

# Finalize: compute PPO peaks + write VERDICT/summary (TD-MPC2 anchors from disk).
python "$BASE/analyze_ppowall.py" >> "$BASE/sweep_master.log" 2>&1
echo "ppo_sweep_finished tasks=[$TASKS] budget=$BUDGET seeds=[$SEEDS] $(date -u +%FT%TZ)" > "$BASE/PPO_SWEEP_DONE"
