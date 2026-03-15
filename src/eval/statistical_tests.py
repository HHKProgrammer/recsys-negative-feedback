# pipeline step 7 — statistical significance testing
# after collecting per-user metric values, tests whether differences are real or random
# with n=6040 users the tests have high power — small real effects become detectable
# paired = same user compared against itself, controls for user difficulty differences

# Statistical significance tests for thesis results.
# Usage:
# from src.eval.statistical_tests import compare_to_baseline
#  table = compare_to_baseline(baseline_per_user_df, variant_per_user_df, k=10)
#   print(table)

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def paired_t_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Tuple[float, float]:
    # Paired t-test: are the means of two per-user score arrays different?
    # di = score_variant(useri) - score_baseline(useri)  for each user i
    #
    # d_bar  = (1/n) * sum(di)                      mean of differences
    # s_d    = sqrt((1/(n-1)) * sum((di - d_bar)^2)) standard deviation
    #
    # t      = d_bar / (s_d / sqrt(n))              t-statistic
    #
    # p-value = P(|T| >= |t|  under H0: d_bar = 0)
    #
    # comparing user against itself
    # and everything under 0.5 is insignificant
    #
    # why paired? because the same user rated different things in each model
    # pairing removes between-user variance some users are just harder to recommend for
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return float(t_stat), float(p_value)


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Tuple[float, float]:
    # absoulte diffrenze
    # Wilcoxon signed-rank test: non-parametric alternative to paired t-test.
    # Suitable when the distribution of differences is not normal.
    # safer for hit@k and ndcg@k which have many zeros ,zeroinflated distributions
    # ranks absolute differences and checks if positive diffs dominate negative diffs
    stat, p_value = stats.wilcoxon(scores_a, scores_b)
    return float(stat), float(p_value)


def confidence_interval(scores: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    # 95% CI for the mean using the t-distribution.
    #
    # 95% CI = x_bar +/- t_{alpha/2, n-1} * SE
    # SE = s / sqrt(n)       s = standard deviation, n = number of users
    #
    # for n=6040: t_{0.025, 6039} ~= 1.96  (approaches normal distribution for large n)
    #
    # report as: ndcg@10 = 0.0434 +/- 0.0003 (95% ci)
    # shows the measurement uncertainty, not just a point estimate
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)  # standard error = s / sqrt(n)
    margin = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return float(mean - margin), float(mean + margin)


def compare_to_baseline(
    baseline_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    # Compare a variant to the baseline on all per-user metrics.
    # returns a table ready for the thesis with delta and significance per metric
    # columns: metric | baseline_mean | variant_mean | delta | t_stat | p_value | significant
    metric_cols = [
        f"precision@{k}", f"recall@{k}", f"ndcg@{k}",
        f"negative@{k}", f"hit@{k}", "mrr",
    ]
    # Keep only columns that exist in both frames
    metric_cols = [c for c in metric_cols if c in baseline_df.columns and c in variant_df.columns]

    rows = []
    for col in metric_cols:
        a = baseline_df[col].values
        b = variant_df[col].values
        t_stat, p_value = paired_t_test(a, b)
        rows.append({
            "metric": col,
            "baseline_mean": round(float(np.mean(a)), 4),
            "variant_mean": round(float(np.mean(b)), 4),
            "delta": round(float(np.mean(b)) - float(np.mean(a)), 4),  # positive = variant better
            "t_stat": round(t_stat, 3),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,  # standard alpha level
        })

    return pd.DataFrame(rows)