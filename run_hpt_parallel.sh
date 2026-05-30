#!/bin/bash
# Parallel Arm D HPTresume-safe, no --overwrite, correct paths
#
# USAGE
#   ./run_hpt_parallel.sh [DATASET] [PAIR] [N_WORKERS] [N_TRIALS_TARGET]
#   ./run_hpt_parallel.sh movielens_20m p3_n2 3 200    # 3 workers, target 200 total
#   ./run_hpt_parallel.sh movielens_10m p4_n2 2 200    # 2 workers, finish ML-10M p4_n2
#   ./run_hpt_parallel.sh all_10m 4 200                # all ML-10M configs, 4 workers each
#
# Resume logic:
#   - Never uses --overwrite
#   - run_hyperparameter_tuning.py now skips ONLY when trials.csv >= n_trials
#   - Workers sharing the same study.db via Optuna SQLite naturally cooperate
#
# MONITOR
#   tail -f logs/hpt_parallel_latest.log
#   watch -n 60 'python scripts/summarize_hpt_arm_d.py'
#   for f in logs/hpt_parallel_*_w*.log; do echo "== $(basename $f) =="; tail -1 "$f"; done

set -e
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

_TS=$(date +%Y%m%d_%H%M%S)
_MASTER="logs/hpt_parallel_${_TS}.log"
mkdir -p logs
exec > >(tee -a "$_MASTER") 2>&1
ln -sf "$_MASTER" logs/hpt_parallel_latest.log
echo "Master log: $_MASTER"

DATASET=${1:-all_10m}
PAIR=${2:-p4_n1}
N_WORKERS=${3:-4}
N_TRIALS=${4:-200}
MAX_USERS=5000

cfg_to_thresholds() {
    case "$1" in
        p4_n1) echo "4 1" ;;
        p4_n2) echo "4 2" ;;
        p5_n1) echo "5 1" ;;
        p3_n2) echo "3 2" ;;
        *) echo "ERROR: unknown pair $1" >&2; exit 1 ;;
    esac
}

ds_to_cfg() {
    case "$1" in
        movielens_1m)  echo "configs/movielens_1m.yaml" ;;
        movielens_10m) echo "configs/movielens_10m.yaml" ;;
        movielens_20m) echo "configs/movielens_20m.yaml" ;;
        spotify)       echo "configs/spotify.yaml" ;;
        *) echo "ERROR: unknown dataset $1" >&2; exit 1 ;;
    esac
}

ds_to_output_path() {
    case "$1" in
        movielens_1m)  echo "movielens/ml-1m" ;;
        movielens_10m) echo "movielens/ml-10m" ;;
        movielens_20m) echo "movielens/ml-20m" ;;
        spotify)       echo "spotify" ;;
        *) echo "$1" ;;
    esac
}

count_trials_csv() {
    local csv_path="$1"
    if [ -f "$csv_path" ]; then
            local total header count
        total=$(wc -l < "$csv_path")
        header=$(grep -c '^trial,' "$csv_path" 2>/dev/null || echo 1)
        count=$(( total - header ))
        [ "$count" -lt 0 ] && count=0
        echo "$count"
    else
        echo 0
    fi
}

launch_workers() {
    local ds="$1"
    local pair="$2"
    local n_workers="$3"
    local n_trials="$4"

    local thresholds
    thresholds=$(cfg_to_thresholds "$pair")
    local pos="${thresholds% *}"
    local neg="${thresholds#* }"
    local cfg
    cfg=$(ds_to_cfg "$ds")
    local label="${ds}_${pair}"
    local out_path
    out_path=$(ds_to_output_path "$ds")

    local study_dir="outputs/${out_path}/tuning/arm_d/pos_ge_${pos}_neg_le_${neg}"
    local trials_csv="${study_dir}/trials.csv"
    local already
    already=$(count_trials_csv "$trials_csv")
    local remaining=$(( n_trials - already ))
    [ "$remaining" -lt 0 ] && remaining=0

    # Per-worker ach worker runs ceil(remaining/n_workers) additional trials.

    local per_worker=$(( (remaining + n_workers - 1) / n_workers ))
    local worker_target=$(( already + per_worker ))

    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  Dataset/pair: $ds / $pair  (pos≥$pos  neg≤$neg)"
    echo "  Config file:  $cfg"
    echo "  Study dir:    $study_dir"
    echo "  Trials:       $already done / $n_trials target  →  $remaining remaining"
    echo "  Workers:      $n_workers  (≈$per_worker trials each, target $worker_target)"
    echo "  MAX_USERS:    $MAX_USERS"
    echo "══════════════════════════════════════════════════════════════"

    if [ "$remaining" -le 0 ]; then
        echo "  ALREADY COMPLETE ($already >= $n_trials) — skipping."
        return 0
    fi

    local pids=()
    for w in $(seq 1 "$n_workers"); do
        local wlog="logs/hpt_parallel_${_TS}_${label}_w${w}.log"

        nohup python scripts/run_hyperparameter_tuning.py \
            --config "$cfg" \
            --arm d \
            --threshold "$pos" \
            --neg_threshold "$neg" \
            --n_trials "$worker_target" \
            --max_users "$MAX_USERS" \
            > "$wlog" 2>&1 &
        local pid=$!
        pids+=("$pid")
        echo "  [w$w] PID $pid  →  $wlog"
    done

    echo "  Waiting for all $n_workers workers..."
    local all_ok=0
    for pid in "${pids[@]}"; do
        wait "$pid" || all_ok=1
    done
    if [ $all_ok -ne 0 ]; then
        echo "  WARNING: one or more workers exited non-zero — check logs above"
    else
        echo "  All workers done."
    fi

    # Final result summary
    local final
    final=$(count_trials_csv "$trials_csv")
    echo "  Trials now: $final / $n_trials"
    for w in $(seq 1 "$n_workers"); do
        local wlog="logs/hpt_parallel_${_TS}_${label}_w${w}.log"
        local best
        best=$(grep "Best nDCG" "$wlog" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
        [ -n "$best" ] && echo "    w$w: $best"
    done
}



echo "N_TRIALS=$N_TRIALS  N_WORKERS=$N_WORKERS  MAX_USERS=$MAX_USERS"

if [ "$DATASET" = "all_10m" ]; then
    echo "=== Running ALL pending ML-10M configs sequentially ==="
    # Priority: nearly-done first, then larger jobs
    launch_workers movielens_10m p4_n2 "$N_WORKERS" "$N_TRIALS"
    launch_workers movielens_10m p4_n1 "$N_WORKERS" "$N_TRIALS"
    launch_workers movielens_10m p5_n1 "$N_WORKERS" "$N_TRIALS"
    launch_workers movielens_10m p3_n2 "$N_WORKERS" "$N_TRIALS"

    echo ""
    echo "All ML-10M done. Re-run grid:"
    echo "  python scripts/run_arm_d_joint_svd_grid.py --config configs/movielens_10m.yaml"
    echo "  python scripts/generate_all_figures.py"

elif [ "$DATASET" = "all_20m_remaining" ]; then
    echo "=== Running remaining ML-20M configs (all 4, skip if already complete) ==="

    launch_workers movielens_20m p4_n2 2 "$N_TRIALS"
    launch_workers movielens_20m p4_n1 2 "$N_TRIALS"
    launch_workers movielens_20m p5_n1 2 "$N_TRIALS"
    launch_workers movielens_20m p3_n2 3 "$N_TRIALS" 

    echo ""
    echo "ML-20M done. Re-run grid:"
    echo "  python scripts/run_arm_d_joint_svd_grid.py --config configs/movielens_20m.yaml"
    echo "  python scripts/generate_all_figures.py"

else
    # Single config
    launch_workers "$DATASET" "$PAIR" "$N_WORKERS" "$N_TRIALS"
    echo ""
    cfg=$(ds_to_cfg "$DATASET")
    echo "Done. Run grid + figures:"
    echo "  python scripts/run_arm_d_joint_svd_grid.py --config $cfg"
    echo "  python scripts/generate_all_figures.py"
fi


echo ""
echo "══════════════════════════════════════════════════════════════"
echo " HPT PARALLEL DONE"
echo " Summary: python scripts/summarize_hpt_arm_d.py"
echo " Logs:    logs/hpt_parallel_${_TS}_*.log"
echo "══════════════════════════════════════════════════════════════"
