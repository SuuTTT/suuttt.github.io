#!/bin/bash
# Pure-JEPA DMControl grid. 4 GPUs, 1 job/GPU at a time. ~20s/run.
cd /root/tdmpc_glass/exp/proposal_D2_pure_jepa
PY=/root/tdmpc_glass/venv/bin/python
mkdir -p logs
JOBS=()
# core: simnorm=1, 4 arms x 3 seeds (n=3 for taxonomy)
for cond in none uniformity vicreg se; do
  for s in 0 1 2; do JOBS+=("WalkerWalk $cond $s 1"); done
done
# collapse control: simnorm=0 (no simplex prior), 4 arms x 1 seed
for cond in none uniformity vicreg se; do JOBS+=("WalkerWalk $cond 0 0"); done

i=0
for job in "${JOBS[@]}"; do
  set -- $job; task=$1; cond=$2; seed=$3; sn=$4
  gpu=$((i % 4))
  tag="${task}_${cond}_sn${sn}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_MEM_FRACTION=0.22 \
    $PY pure_jepa_dmc.py train --task $task --cond $cond --seed $seed --simnorm $sn --gpu 0 \
    > logs/$tag.log 2>&1 &
  i=$((i+1))
  # throttle: 4 concurrent
  if (( i % 4 == 0 )); then wait; fi
done
wait
echo "ALL_DONE"
