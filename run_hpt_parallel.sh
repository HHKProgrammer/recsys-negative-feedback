#!/bin/bash
# Parallel Arm D HPT using Optuna shared SQLite storage for some last comparisons 
#

#
# SCIENTIFIC JUSTIFICATION
#   ML-20M is DONE all 4 configs have  best_params.json.
#   ML-10M is 
#   ML-1M and Spotify are DONE (200 trials)
#

#   1. ML-10M p4_n1   50/200 done, best 0.2422, fastest config
#   2. ML-10M p4_n2 50/200 done, best 0.2165
#   3. ML-10M p3_n2  30/200 done, best 0.2575 (promising)
#   4. ML-10M p5_n1   170/200 done, best 0.1919, 30 more to converge
#
# USAGE 
#   ./run_hpt_parallel.sh [DATASET] [ARM_D_PAIR] [N_WORKERS] [N_TRIALS]
#   ./run_hpt_parallel.sh movielens_10m p4_n1 4 150     # 4 workers, 150 total trials
#
# Or run all pending ML-10M configs:
#   nohup ./run_hpt_parallel.sh all_10m 2>&1 &
#   tail -f logs/hpt_parallel_latest.log
#
# MONITOR
#   watch -n 60 'python scripts/summarize_hpt_arm_d.py'
#   for f in logs/hpt_parallel_*w*.log; do echo "== $f =="; tail -2 "$f"; done

set -e
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

_TS=$(date +%Y%m%d_%H%M%S)
_MASTER="logs/hpt_parallel_${_TS}.log"
mkdir -p logs
exec > >(tee -a "$_MASTER") 2>&1
ln -sf "$_MASTER" logs/hpt_parallel_latest.log
echo "Master log: $_MASTER"

DATASET=${1:-all_10m}      
PAIR=${2:-p4_n1}           # p4_n1 | p4_n2 | p3_n2 | p5_n1
N_WORKERS=${3:-4}          # parallel workers (4-6 = safe )
N_TRIALS=${4:-150}         # TOTAL desired trials per config 
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

# Launch  one (dataset, pair, n_trials) 
# Workers share same Optuna study.db 
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

    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  Launching $n_workers workers: $ds / $pair"
    echo "  Config: $cfg  |  pos≥$pos neg≤$neg"
    echo "  Total target trials: $n_trials  |  MAX_USERS: $MAX_USERS"
    echo "  Workers share: outputs/$(echo $ds | tr _ /)/tuning/arm_d/pos_ge_${pos}_neg_le_${neg}/study.db"
    echo "══════════════════════════════════════════════════════════════"

    local pids=()
    for w in $(seq 1 "$n_workers"); do
        local wlog="logs/hpt_parallel_${_TS}_${label}_w${w}.log"
        nohup python scripts/run_hyperparameter_tuning.py \
            --config "$cfg" \
            --arm d \
            --threshold "$pos" \
            --neg_threshold "$neg" \
            --n_trials "$n_trials" \
            --max_users "$MAX_USERS" \
            --overwrite \
            > "$wlog" 2>&1 &
        local pid=$!
        pids+=("$pid")
        echo "  [w$w] PID $pid  →  $wlog"
    done

    echo "  Waiting for all $n_workers workers to finish..."
    local all_ok=0
    for pid in "${pids[@]}"; do
        wait "$pid" || all_ok=1
    done
    if [ $all_ok -ne 0 ]; then
        echo "  WARNING: one or more workers exited non-zero — check logs"
    else
        echo "  All workers done."
    fi

    # Print final best from each worker log
    echo "  Results:"
    for w in $(seq 1 "$n_workers"); do
        local wlog="logs/hpt_parallel_${_TS}_${label}_w${w}.log"
        local best
        best=$(grep "Best nDCG" "$wlog" 2>/dev/null | tail -1 || echo "  no result yet")
        echo "    w$w: $best"
    done
}


#  Main dispatch 

if [ "$DATASET" = "all_10m" ]; then
    echo "Running ALL pending ML-10M configs with $N_WORKERS workers each"
    echo "N_TRIALS=$N_TRIALS  MAX_USERS=$MAX_USERS"
    echo ""
t
    launch_workers movielens_10m p5_n1 "$N_WORKERS" "$N_TRIALS"

    launch_workers movielens_10m p4_n1 "$N_WORKERS" "$N_TRIALS"

    launch_workers movielens_10m p4_n2 "$N_WORKERS" "$N_TRIALS"

    launch_workers movielens_10m p3_n2 "$N_WORKERS" "$N_TRIALS"

    echo ""
    echo "All ML-10M configs done. Run grid + figures:"
    echo "  python scripts/run_arm_d_joint_svd_grid.py --config configs/movielens_10m.yaml"
    echo "  python scripts/generate_all_figures.py"

elif [ "$DATASET" = "all_10m_parallel" ]; then
    echo "Launching ALL ML-10M configs in parallel ($N_WORKERS workers × 4 configs = $((N_WORKERS*4)) processes)"
    echo "Only do this if the server can handle $((N_WORKERS*4)) parallel training jobs."

    for pair in p5_n1 p4_n1 p4_n2 p3_n2; do
        thresholds=$(cfg_to_thresholds "$pair")
        pos="${thresholds% *}"; neg="${thresholds#* }"
        for w in $(seq 1 "$N_WORKERS"); do
            wlog="logs/hpt_parallel_${_TS}_10m_${pair}_w${w}.log"
            nohup python scripts/run_hyperparameter_tuning.py \
                --config configs/movielens_10m.yaml \
                --arm d --threshold "$pos" --neg_threshold "$neg" \
                --n_trials "$N_TRIALS" --max_users "$MAX_USERS" --overwrite \
                > "$wlog" 2>&1 &
            echo "  [${pair} w${w}] PID $!  →  $wlog"
        done
    done
    echo "All launched. Monitor with:"
    echo "  watch -n 60 'for f in logs/hpt_parallel_${_TS}_10m_*.log; do echo \"\$f:\"; tail -1 \"\$f\"; done'"
    wait
    echo "All workers complete."

else
    launch_workers "$DATASET" "$PAIR" "$N_WORKERS" "$N_TRIALS"
    echo ""
    echo "Done. Run grid + figures to use the new best params:"
    cfg=$(ds_to_cfg "$DATASET")
    echo "  python scripts/run_arm_d_joint_svd_grid.py --config $cfg"
    echo "  python scripts/generate_all_figures.py"
fi


echo ""
echo "══════════════════════════════════════════════════════════════"
echo " HPT PARALLEL DONE"
echo " Summary: python scripts/summarize_hpt_arm_d.py"
echo " Logs:    logs/hpt_parallel_${_TS}_*.log"
echo "══════════════════════════════════════════════════════════════"