#!/bin/bash
# Arm D: Joint Positive-Negative SVD  run ONLY the new Arm D phases.
#
# Assumes Phases 1-9 (baseline, post-hoc, HPT, Arm A, Arm B, Arm C, known-neg)
# are already complete on the server.  This script adds the new joint
# positive-negative training experiment WITHOUT rerunning completed work.
#
# Disconnect-safe (run on server):
#   HPT_TRIALS=200 HPT_MAX_USERS=5000 nohup ./run_arm_d.sh > arm_d_run.log 2>&1 &
#   echo $! > arm_d_run.pid
#   tail -f arm_d_run.log
#   ps -p $(cat arm_d_run.pid)   # check still running
#
# Quick local test first:
#   HPT_TRIALS=5 MAX_USERS=500 ./run_arm_d.sh > arm_d_test.log 2>&1
#
# All phases are resume-safe — re-run if interrupted.
# Skips configs that already have best_params.json / grid_summary_arm_d.json.

set -e
export PYTHONUNBUFFERED=1

HPT_TRIALS=${HPT_TRIALS:-200}
MAX_USERS=${MAX_USERS:-""}
HPT_MAX_USERS=${HPT_MAX_USERS:-5000}
N_PARALLEL=${N_PARALLEL:-12}

CONFIGS=(
    configs/movielens_1m.yaml
    configs/movielens_10m.yaml
    configs/movielens_20m.yaml
    configs/spotify.yaml
)

# Arm D grid: (pos_threshold, neg_threshold) pairs
# Format: "pos neg"
ARM_D_PAIRS=("4 2" "5 1" "4 1" "3 2")

LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)_arm_d"
mkdir -p "$LOG_DIR"
echo "Per-job logs: $LOG_DIR"

run_bg() {
    local label="$1"; shift
    local log="$LOG_DIR/${label}.log"
    "$@" > "$log" 2>&1 &
    echo "  [bg] $label  (log: $log)"
}

max_users_flag() {
    [ -n "$MAX_USERS" ] && echo "--max_users $MAX_USERS" || echo ""
}

hpt_max_users_flag() {
    [ -n "$HPT_MAX_USERS" ] && echo "--max_users $HPT_MAX_USERS" || echo ""
}

cfg_label() {
    basename "$1" .yaml
}


echo "=========================================================="
echo " Arm D: Joint Positive-Negative SVD"
echo " Phases 1-9 assumed complete — only running new Arm D work"
echo " HPT_TRIALS=$HPT_TRIALS  HPT_MAX_USERS=$HPT_MAX_USERS"
echo "=========================================================="


echo ""
echo "=========================================================="
echo " Phase 10: Arm D HPT"
echo "           ${#ARM_D_PAIRS[@]} configs x ${#CONFIGS[@]} datasets = $((${#ARM_D_PAIRS[@]} * ${#CONFIGS[@]})) jobs"
echo "           Trials: $HPT_TRIALS per job  HPT_MAX_USERS: $HPT_MAX_USERS"
echo "           Skips configs with existing best_params.json"
echo "=========================================================="
for pair in "${ARM_D_PAIRS[@]}"; do
    pos="${pair% *}"
    neg="${pair#* }"
    for cfg in "${CONFIGS[@]}"; do
        run_bg "hpt_d_p${pos}_n${neg}_$(cfg_label $cfg)" \
            python scripts/run_hyperparameter_tuning.py \
                --config "$cfg" \
                --arm d \
                --threshold "$pos" \
                --neg_threshold "$neg" \
                --n_trials "$HPT_TRIALS" \
                $(hpt_max_users_flag)
    done
done

set +e
wait
HPT_D_EXIT=$?
set -e
if [ $HPT_D_EXIT -ne 0 ]; then
    echo "WARNING: one or more Arm D HPT jobs exited non-zero (exit $HPT_D_EXIT) — continuing"
fi
echo " Phase 10 complete"


echo ""
echo "=========================================================="
echo " Phase 11: Arm D joint SVD grid 4 datasets in parallel"
echo "           Trains SVD on binary targets (pos=1.0, neg=0.0)"
echo "           Skips configs already in grid_summary_arm_d.json"
echo "=========================================================="
for cfg in "${CONFIGS[@]}"; do
    run_bg "arm_d_$(cfg_label $cfg)" \
        python scripts/run_arm_d_joint_svd_grid.py \
            --config "$cfg" $(max_users_flag)
done
wait
echo " Phase 11 complete"


echo ""
echo "=========================================================="
echo " Phase 12: Regenerate all figures and tables"
echo "           Includes Arm A vs Arm C vs Arm D comparison"
echo "=========================================================="
python scripts/generate_all_figures.py
echo " Phase 12 complete"


echo "=========================================================="
echo " Arm D DONE"
echo ""
echo " Logs:    $LOG_DIR/"
echo " Results: outputs/*/grid_summary_arm_d.json"
echo " Models:  outputs/*/models/arm_d/*/model.pkl"
echo " Tuning:  outputs/*/tuning/arm_d/*/best_params.json"
echo " Figures: reports/figures/"
echo " Tables:  reports/tables/"
echo ""
echo " Review results, then commit:"
echo "   git add outputs/ reports/"
echo "   git commit -m 'add Arm D joint pos-neg SVD results'"
echo "=========================================================="