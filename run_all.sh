#!/bin/bash
# Robust nohup driver: 2 maps x 2 arms x 2 seeds = 8 concurrent CPU runs.
# Writes per-run JSON, then aggregate (summary.json + VERDICT.md), then DONE marker.
set -u
cd /root/helios-rl/exp/minigrid_hardexp
source .venv/bin/activate
OUT=runs
mkdir -p "$OUT" logs
ENVS="MiniGrid-MultiRoom-N6-v0 MiniGrid-KeyCorridorS3R3-v0"
ARMS="ppo rnd"
SEEDS="0 1"
STEPS=3000000
echo "START $(date)"
pids=""
for env in $ENVS; do
  for arm in $ARMS; do
    for s in $SEEDS; do
      tag="${env}__${arm}__seed${s}"
      nohup python minigrid_ppo_rnd.py --env "$env" --arm "$arm" --seed "$s" \
        --total-steps $STEPS --num-envs 8 --torch-threads 1 --outdir "$OUT" \
        > "logs/${tag}.log" 2>&1 &
      pids="$pids $!"
      echo "launched $tag pid $!"
    done
  done
done
echo "waiting on:$pids"
wait
echo "all runs finished $(date)"
python aggregate.py --outdir "$OUT" > logs/aggregate.log 2>&1
touch "$OUT/DONE"
echo "DONE $(date)"
