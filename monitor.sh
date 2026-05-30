#!/bin/bash

echo "=== ACTIVE WORKERS ==="
pgrep -af run_hyperparameter_tuning.py
echo

echo "=== WORKER STATUS (LOGS) ==="
for f in logs/hpt_parallel_*w*.log; do
    printf "%-75s" "$f"
    tail -n 1 "$f"
done

echo
echo "=== GLOBAL TRIAL COUNTS ==="
python scripts/summarize_hpt_arm_d.py | grep -E "ML-10M|ML-20M"
