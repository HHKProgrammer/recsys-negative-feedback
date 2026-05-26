#!/bin/bash
# Full experiment pipeline (server: 48 cores, 128 GB RAM, Debian 13)
# Runs all four arms: A (pos-only SVD), B (neg-only SVD), C (hybrid), D (joint pos-neg SVD)
#
# Parallelism strategy:
#    All 4 datasets run simultaneously for each phase
#    HPT: all threshold x dataset combos launched at once (48 jobs: 12 A + 20 B + 16 D)
#    Arm C: --n_parallel 12 (48 cores / 4 datasets = 12 workers per dataset)
#
# To run ONLY Arm D (when Phases 1-9 are already done on the server):
#    HPT_TRIALS=200 HPT_MAX_USERS=5000 nohup ./run_arm_d.sh > arm_d_run.log 2>&1 &
#
# Disconnect-safe:
#   HPT_TRIALS=200 nohup ./run_all.sh > full_run.log 2>&1 &
#   echo $! > run_all.pid
#   tail -f full_run.log          # monitor
#   ps -p $(cat run_all.pid)      # check still running
#
# Quick test:
#   HPT_TRIALS=5 MAX_USERS=500 N_PARALLEL=2 nohup ./run_all.sh > test_run.log 2>&1 &
#
# Stop:
#   kill $(cat run_all.pid)                          # stop the main process
#   pkill -f run_hyperparameter_tuning.py            # stop any HPT jobs still running
#   pkill -f run_arm_a_positive_svd_grid.py          # stop Arm A jobs
#   pkill -f run_arm_b_negative_svd_grid.py          # stop Arm B jobs
#   pkill -f run_arm_c_hybrid_grid.py                # stop Arm C jobs
#   pkill -f run_arm_d_joint_svd_grid.py             # stop Arm D jobs
#
# Restart (resume-safe — all phases skip completed work):
#   HPT_TRIALS=200 nohup ./run_all.sh > full_run.log 2>&1 &
#   echo $! > run_all.pid
#
# All phases are resume-safe — re-run if interrupted.

set -e
export PYTHONUNBUFFERED=1

HPT_TRIALS=${HPT_TRIALS:-200}
MAX_USERS=${MAX_USERS:-""}
# HPT_MAX_USERS: separate limit for HPT evaluation only (keeps each trial fast)
# Grid/Arm C use MAX_USERS (default: all users)
HPT_MAX_USERS=${HPT_MAX_USERS:-5000}
# Arm C workers per dataset: 48 cores / 4 datasets = 12
# Reduce if memory is tight (each worker loads the full dataset + models)
N_PARALLEL=${N_PARALLEL:-12}

CONFIGS=(
    configs/movielens_1m.yaml
    configs/movielens_10m.yaml
    configs/movielens_20m.yaml
    configs/spotify.yaml
)
ARM_A_THRESHOLDS=(3 4 5)
ARM_B_THRESHOLDS=(1 2 3 median modus)
# Arm D grid: (pos_threshold, neg_threshold) pairs format "pos neg"
ARM_D_PAIRS=("4 2" "5 1" "4 1" "3 2")

# Per-job log directory so output doesn't interleave
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Per-job logs: $LOG_DIR"

# Helper: run a command in background, redirect to its own log file
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
    # configs/movielens_1m.yaml -> movielens_1m
    basename "$1" .yaml
}


echo "=========================================================="
echo " Phase 1: Prepare datasets (sequential )"
echo "=========================================================="
python main.py prepare --dataset 1m
python main.py prepare --dataset 10m
python main.py prepare --dataset 20m
python main.py prepare --dataset spotify


echo "=========================================================="
echo " Phase 2: Snapshot v1 results"
echo "=========================================================="
python scripts/snapshot_results.py --label v1_post_hoc


echo "=========================================================="
echo " Phase 3: Post-hoc grid4 datasets in parallel"
echo "          36 experiments per dataset (cached runs skipped)"
echo "=========================================================="
for cfg in "${CONFIGS[@]}"; do
    run_bg "phase3_$(cfg_label $cfg)" python main.py grid --config "$cfg"
done
wait
echo " Phase 3 complete"


echo "=========================================================="
echo " Phase 4: Hyperparameter tuning — all jobs in parallel"
echo "          Arm A: ${#ARM_A_THRESHOLDS[@]} thresholds x ${#CONFIGS[@]} datasets = $((${#ARM_A_THRESHOLDS[@]} * ${#CONFIGS[@]})) jobs"
echo "          Arm B: ${#ARM_B_THRESHOLDS[@]} thresholds x ${#CONFIGS[@]} datasets = $((${#ARM_B_THRESHOLDS[@]} * ${#CONFIGS[@]})) jobs"
echo "          Arm D: ${#ARM_D_PAIRS[@]} pairs x ${#CONFIGS[@]} datasets = $((${#ARM_D_PAIRS[@]} * ${#CONFIGS[@]})) jobs"
echo "          Trials: $HPT_TRIALS per job  HPT_MAX_USERS: ${HPT_MAX_USERS:-all}"
echo "=========================================================="

for t in "${ARM_A_THRESHOLDS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
        run_bg "hpt_a_t${t}_$(cfg_label $cfg)" \
            python scripts/run_hyperparameter_tuning.py \
                --config "$cfg" --arm a --threshold "$t" \
                --n_trials "$HPT_TRIALS" $(hpt_max_users_flag)
    done
done

for t in "${ARM_B_THRESHOLDS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
        run_bg "hpt_b_t${t}_$(cfg_label $cfg)" \
            python scripts/run_hyperparameter_tuning.py \
                --config "$cfg" --arm b --threshold "$t" \
                --n_trials "$HPT_TRIALS" $(hpt_max_users_flag)
    done
done

for pair in "${ARM_D_PAIRS[@]}"; do
    pos="${pair% *}"
    neg="${pair#* }"
    for cfg in "${CONFIGS[@]}"; do
        run_bg "hpt_d_p${pos}_n${neg}_$(cfg_label $cfg)" \
            python scripts/run_hyperparameter_tuning.py \
                --config "$cfg" --arm d \
                --threshold "$pos" --neg_threshold "$neg" \
                --n_trials "$HPT_TRIALS" $(hpt_max_users_flag)
    done
done

set +e
wait
HPT_EXIT=$?
set -e
if [ $HPT_EXIT -ne 0 ]; then
    echo "WARNING: one or more HPT jobs exited non-zero (exit $HPT_EXIT) — continuing pipeline"
fi
echo " Phase 4 complete"


echo "=========================================================="
echo " Phase 5: Arm A grid ${#ARM_A_THRESHOLDS[@]} thresholds x ${#CONFIGS[@]} datasets = $((${#ARM_A_THRESHOLDS[@]} * ${#CONFIGS[@]})) jobs in parallel"
echo "          Trains SVD on rating >= pos_threshold"
echo "          Saves models for Arm C"
echo "=========================================================="
for pt in "${ARM_A_THRESHOLDS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
        run_bg "arm_a_t${pt}_$(cfg_label $cfg)" \
            python scripts/run_arm_a_positive_svd_grid.py \
                --config "$cfg" --threshold "$pt" $(max_users_flag)
    done
done
wait
echo " Phase 5 complete"


echo "=========================================================="
echo " Phase 6: Arm B grid ${#ARM_B_THRESHOLDS[@]} thresholds x ${#CONFIGS[@]} datasets = $((${#ARM_B_THRESHOLDS[@]} * ${#CONFIGS[@]})) jobs in parallel"
echo "          Trains SVD on rating <= neg_threshold"
echo "          Saves models for Arm C"
echo "=========================================================="
for t in "${ARM_B_THRESHOLDS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
        run_bg "arm_b_t${t}_$(cfg_label $cfg)" \
            python scripts/run_arm_b_negative_svd_grid.py \
                --config "$cfg" --threshold "$t" $(max_users_flag)
    done
done
wait
echo " Phase 6 complete"


echo "=========================================================="
echo " Phase 7: Arm C hybrid grid 4 datasets in parallel"
echo "          90 combinations per dataset"
echo "          $N_PARALLEL workers per dataset (multiprocessing)"
echo "=========================================================="
for cfg in "${CONFIGS[@]}"; do
    run_bg "arm_c_$(cfg_label $cfg)" \
        python scripts/run_arm_c_hybrid_grid.py \
            --config "$cfg" --n_parallel "$N_PARALLEL" $(max_users_flag)
done
wait
echo " Phase 7 complete"


echo "=========================================================="
echo " Phase 8: Known-negative injection eval4 datasets in parallel"
echo "          (Krichene & Rendle 2020)"
echo "          SVD trained once per dataset, $N_PARALLEL eval threads"
echo "=========================================================="
for cfg in "${CONFIGS[@]}"; do
    run_bg "known_neg_$(cfg_label $cfg)" \
        python scripts/run_known_negative_eval.py \
            --config "$cfg" --n_workers "$N_PARALLEL" $(max_users_flag)
done
wait
echo " Phase 8 complete"


echo "=========================================================="
echo " Phase 9: Arm D joint SVD grid  4 datasets in parallel"
echo "          Trains SVD on binary targets (pos=1.0, neg=0.0)"
echo "          Skips configs already in grid_summary_arm_d.json"
echo "=========================================================="
for cfg in "${CONFIGS[@]}"; do
    run_bg "arm_d_$(cfg_label $cfg)" \
        python scripts/run_arm_d_joint_svd_grid.py \
            --config "$cfg" $(max_users_flag)
done
wait
echo " Phase 9 complete"


echo "=========================================================="
echo " Phase 10: Generate all figures and tables"
echo "=========================================================="
python scripts/generate_all_figures.py


echo "=========================================================="
echo " DONE"
echo ""
echo " Logs:       $LOG_DIR/"
echo " Figures:    reports/figures/"
echo " Tables:     reports/tables/"
echo " Results:"
echo "   outputs/*/grid_summary.json"
echo "   outputs/*/grid_summary_arm_a.json"
echo "   outputs/*/grid_summary_arm_b.json"
echo "   outputs/*/grid_summary_arm_c.json"
echo "   outputs/*/grid_summary_arm_d.json"
echo "   outputs/*/grid_summary_known_neg_eval.json"
echo ""
echo " To push:"
echo "   git add outputs/ reports/ scripts/ run_all.sh run_arm_d.sh"
echo "   git commit -m 'four-arm results'"
echo "   git push"
echo "=========================================================="
