#!/bin/bash
# Full experiment pipeline  the server
#
# Resume-safe: every grid runner and HPT script skips already-completed work.
# No automatic git pushreview results first, then push manually.
#
# Phases:
#   1  Prepare all datasets
#   2  Snapshot pre-run results (v1)
#   3  Post-hoc inference-time grid  (filter / rerank / weighted 36 runs x dataset)
#   4  Hyperparameter tuning         (Arm A x 3 thresholds, Arm B x 5 thresholds) x 4 datasets
#   5  Arm A  Positive-Preference SVD grid   (3 runs x dataset, saves models)
#   6  Arm B Negative Dislike-Risk grid     (5 runs x dataset, saves models)
#   7  Arm C  Hybrid combiner grid           (90 combinations x dataset)
#   8  Known-negative candidate injection eval (Krichene & Rendle 2020)
#   9  Generate all figures and tables

set -e

# Override from environment for quick test runs:
#   HPT_TRIALS=5 MAX_USERS=500 ./run_all.sh
HPT_TRIALS=${HPT_TRIALS:-200}
MAX_USERS=${MAX_USERS:-""}

CONFIGS=(
    configs/movielens_1m.yaml
    configs/movielens_10m.yaml
    configs/movielens_20m.yaml
    configs/spotify.yaml
)

ARM_A_THRESHOLDS=(3 4 5)
ARM_B_THRESHOLDS=(1 2 3 median modus)

max_users_flag() {
    [ -n "$MAX_USERS" ] && echo "--max_users $MAX_USERS" || echo ""
}


echo "=========================================================="
echo " Phase 1: Prepare datasets"
echo "=========================================================="
python main.py prepare --dataset 1m
python main.py prepare --dataset 10m
python main.py prepare --dataset 20m
python main.py prepare --dataset spotify


echo "=========================================================="
echo " Phase 2: Snapshot v1 results (post-hoc only)"
echo "=========================================================="
python scripts/snapshot_results.py --label v1_post_hoc


echo "=========================================================="
echo " Phase 3: Post-hoc inference-time grid (36 runs per dataset)"
echo "          Already cached runs are skipped automatically"
echo "=========================================================="
for cfg in "${CONFIGS[@]}"; do
    python main.py grid --config "$cfg"
done


echo "=========================================================="
echo " Phase 4: Hyperparameter tuning ($HPT_TRIALS trials per arm/threshold)"
echo "          Arm A objective: nDCG@10 on validation set"
echo "          Arm B objective: negative_detection_hit@10 (leave-one-negative-out)"
echo "          Results cached  rerun is a no-op if study.pkl already exists"
echo "=========================================================="

echo "--- Arm A HPT ---"
for cfg in "${CONFIGS[@]}"; do
    for t in "${ARM_A_THRESHOLDS[@]}"; do
        echo "  HPT arm=a  threshold=$t  config=$cfg"
        python scripts/run_hyperparameter_tuning.py \
            --config "$cfg" --arm a --threshold "$t" \
            --n_trials "$HPT_TRIALS" $(max_users_flag)
    done
done

echo "--- Arm B HPT ---"
for cfg in "${CONFIGS[@]}"; do
    for t in "${ARM_B_THRESHOLDS[@]}"; do
        echo "  HPT arm=b  threshold=$t  config=$cfg"
        python scripts/run_hyperparameter_tuning.py \
            --config "$cfg" --arm b --threshold "$t" \
            --n_trials "$HPT_TRIALS" $(max_users_flag)
    done
done


echo "=========================================================="
echo " Phase 5: Arm A Positive-Preference SVD grid"
echo "          Trains SVD on rating >= pos_threshold"
echo "          Uses tuned hyperparameters if available"
echo "          References: Rendle et al. 2009 (BPR), He et al. 2017 (NCF)"
echo "=========================================================="
for cfg in "${CONFIGS[@]}"; do
    python scripts/run_arm_a_positive_svd_grid.py \
        --config "$cfg" $(max_users_flag)
done


echo "=========================================================="
echo " Phase 6: Arm B — Negative Dislike-Risk Detector grid"
echo "          Trains SVD on rating <= neg_threshold"
echo "          Evaluated via leave-one-negative-out (LNO)"
echo "          Uses tuned hyperparameters if available"
echo "=========================================================="
for cfg in "${CONFIGS[@]}"; do
    python scripts/run_arm_b_negative_svd_grid.py \
        --config "$cfg" $(max_users_flag)
done


echo "=========================================================="
echo " Phase 7: Arm C Hybrid combiner grid"
echo "          final_score = minmax(pos) - alpha * minmax(neg)"
echo "          90 combinations per dataset (3 pos x 5 neg x 6 alpha)"
echo "=========================================================="
for cfg in "${CONFIGS[@]}"; do
    python scripts/run_arm_c_hybrid_grid.py \
        --config "$cfg" $(max_users_flag)
done


echo "=========================================================="
echo " Phase 8: Known-negative candidate injection evaluation"
echo "          (Krichene & Rendle 2020) injects user dislikes into pool"
echo "=========================================================="
for cfg in "${CONFIGS[@]}"; do
    python scripts/run_known_negative_eval.py --config "$cfg" $(max_users_flag)
done


echo "=========================================================="
echo " Phase 9: Generate all figures and tables"
echo "=========================================================="
python scripts/generate_all_figures.py


echo "=========================================================="
echo " DONE"
echo ""
echo " Figures:   reports/figures/"
echo " Tables:    reports/tables/"
echo " Results:"
echo "   outputs/*/grid_summary.json               (post-hoc)"
echo "   outputs/*/grid_summary_arm_a.json          (Arm A positive-preference)"
echo "   outputs/*/grid_summary_arm_b.json          (Arm B dislike-detector)"
echo "   outputs/*/grid_summary_arm_c.json          (Arm C hybrid)"
echo "   outputs/*/grid_summary_known_neg_eval.json"
echo ""
echo " To push results:"
echo "   git add outputs/ reports/ scripts/ run_all.sh"
echo "   git commit -m 'add three-arm architecture results'"
echo "   git push"
echo "=========================================================="
