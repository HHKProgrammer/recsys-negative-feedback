# recsys-negative-feedback

bachelor thesis can negative feedback improve recommendations?

**research question:** most recsys only learn from likes. can low ratings push recommendations away from disliked content?

---

## datasets

| dataset | users | items | ratings | source |
|---------|-------|-------|---------|--------|
| MovieLens 1M | 6,040 | 3,416 | 993,571 | GroupLens |
| MovieLens 10M | 69,878 | 10,196 | 9,928,938 | GroupLens |
| MovieLens 20M | 138,493 | 18,345 | 19,845,531 | GroupLens |
| Spotify MSSD | 6,376 | 3,490 | 74,648 | Spotify via AIcrowd |

split: random leave-one-out per user — last item = test, second-last = val  
reference: Fan et al. 2024 (random split conservative and honest for recsys)

---

## architecture — four arms

**task: top-N ranking, not rating regression.** ratings = preference signal, goal = rank quality (nDCG@10, HR@10, MRR)

### Arm A — Positive Preference Model
- trains SVD on `rating >= pos_threshold` only
- removes noisy low-rating signal from latent space
- thresholds: `[3, 4, 5]`
- ref: Rendle et al. 2009 (BPR), He et al. 2017 (NCF)

### Arm B — Negative Dislike-Risk Detector
- trains SVD on `rating <= neg_threshold` only
- NOT a recommender a dislike detector
- high rank = item looks like something user dislikes
- thresholds: `[1, 2, 3, median, modus]`
- eval: leave-one-negative-out (LNO) held-out disliked item must appear in top-k
- ref: Cena et al. 2004, Paudel et al. 2016

### Arm C — Hybrid Combiner
- `final_score = minmax(arm_a_score) - alpha * minmax(arm_b_score)`
- min-max normalization per user per candidate list
- flat score range → fallback to 0.5 (graceful degradation to Arm A)
- alpha: `[0.05, 0.1, 0.2, 0.3, 0.5, 1.0]`
- grid: 3 pos × 5 neg × 6 alpha = **90 combinations per dataset**

### Arm D — Joint Positive-Negative SVD
- trains SVD on **both** positives and negatives with binary targets
- negative feedback incorporated directly into latent factors not post-hoc
- rating transformation:

| rating | target | meaning |
|--------|--------|---------|
| `>= pos_threshold` | `1.0` | explicit like |
| `<= neg_threshold` | `0.0` | explicit dislike |
| middle | dropped | ambiguous — excluded |

- `rating_scale=(0, 1)` so SVD pushes liked items toward 1.0, disliked toward 0.0
- grid: 4 (pos, neg) pairs  `(4,2)`, `(5,1)`, `(4,1)`, `(3,2)`
- **core comparison: Arm A vs Arm D**  does joint training improve top-N over positive-only?
  - Arm D > Arm A in nDCG@10 → training-time negatives help ranking
  - Arm D ≈ Arm A but lower `sim_to_neg@10` → negatives improve avoidance, not accuracy
  - Arm D < Arm A → binary signal loses information from full rating scale
- vs Arm B: Arm B trains on negatives only (dislike detector); Arm D trains on both (joint preference model)
- vs Arm C: Arm C penalises post-hoc; Arm D bakes negative signal into the latent factors themselves
- ref: Cena et al. 2004, Paudel et al. 2016

### Post-hoc variants (baseline comparison)
| variant | what |
|---------|------|
| `filter` | remove disliked items from candidates |
| `rerank` | subtract `alpha * max_sim_to_hated` from SVD score |
| `weighted` | like rerank but 1-star penalised more than 3-star |

---

## negative thresholds

| label | definition |
|-------|-----------|
| `fixed_1` / `neg_le_1` | rating <= 1 |
| `fixed_2` / `neg_le_2` | rating <= 2 |
| `fixed_3` / `neg_le_3` | rating <= 3 |
| `median` | rating < user median |
| `modus` | rating < user mode |

---

## metrics

| metric | description |
|--------|-------------|
| `nDCG@10` | position-sensitive ranking quality (primary) |
| `HR@10` | hit rate — did model find the right item? |
| `MRR` | mean reciprocal rank |
| `sim_to_neg@10` | avg cosine sim of top-10 to disliked items (lower = better) |
| `neg_detection_hit@10` | Arm B: did held-out disliked item appear in top-10? |
| `neg_detection_ndcg@10` | Arm B: nDCG where disliked item is "relevant" |
| `mean_dislike_rank` | Arm B: avg rank of disliked items (lower = better detection) |

---

## hyperparameter tuning (HPT)

Optuna TPE Bayesian search, falls back to random search if Optuna not installed

| param | Arm A search space | Arm B search space | Arm D search space |
|-------|-------------------|-------------------|--------------------|
| n_factors | 10–200 | 10–200 | 10–200 |
| n_epochs | 20, 30, 50, 75, 100 | same | same |
| lr_all | 0.001–0.02 | same | same |
| reg_all | 0.005–0.2 | 0.03–0.5 (higher — sparser data) | 0.005–0.2 |
| biased | True / False | True / False | True / False |

- Arm A objective: `nDCG@10` on val set
- Arm B objective: `negative_detection_hit@10` on LNO val split
- Arm D objective: `nDCG@10` on val set (binary-target model)
- default: 200 trials per arm × threshold × dataset

---

## setup

```bash
conda create -n recsys-conda python=3.8 -y
conda activate recsys-conda
conda install -c conda-forge pandas numpy pyarrow scikit-learn scikit-surprise scipy matplotlib seaborn tqdm -y
pip install optuna          # optional enables Bayesian HPT (recommended)
```

or:

```bash
conda env create -f environment.yml
```

---

## server

| spec | value |
|------|-------|
| CPU | 48 cores, Intel x86_64, AVX-512, no hyperthreading |
| RAM | 128 GB |
| GPU | none |
| OS | Debian GNU/Linux 13 (trixie) |
| Python | 3.8.20 (conda) |

parallelism:
- all 4 datasets run simultaneously per phase (bash `&` + `wait`)
- HPT: 32 background jobs at once (12 Arm A + 20 Arm B across all datasets)
- Arm C: `--n_parallel 12` workers per dataset (ProcessPoolExecutor, initializer loads data once per worker)
- known-neg eval: `--n_workers 12` threads per dataset (ThreadPoolExecutor, numpy releases GIL)
- Arm D HPT: 16 background jobs (4 pairs × 4 datasets), Arm D grid: 4 datasets in parallel

evaluation speed design:
- candidate set: 501 items per user (450 neutral + 50 injected + 1 test), not all items sampled eval (Krichene & Rendle 2020)
- scoring: vectorized `predict_batch()` one numpy matmul per user instead of per-item predict() calls (~50× faster)
- known-neg eval: SVD trained **once** per dataset, shared across all 36 grid configs (was: 36× redundant training)
- HPT: `HPT_MAX_USERS=5000` limits each trial to 5000 users; final grid uses all users

---

## running

### full pipeline (server, disconnect-safe)

```bash
# final results all users, 200 HPT trials, 12 Arm C workers per dataset
HPT_TRIALS=200 nohup ./run_all.sh > full_run.log 2>&1 &
echo $! > run_all.pid

# monitor
tail -f full_run.log
ps -p $(cat run_all.pid)      # check still running
```

```bash
# quick sanity check  5 trials, 500 users, 2 Arm C workers
HPT_TRIALS=5 MAX_USERS=500 N_PARALLEL=2 nohup ./run_all.sh > test_run.log 2>&1 &
```

### adding Arm D to an existing run (server)

if Phases 1–9 are already complete, run only the new Arm D phases without repeating completed work:

```bash
# full Arm D (200 HPT trials, all users)
HPT_TRIALS=200 HPT_MAX_USERS=5000 nohup ./run_arm_d.sh > arm_d_run.log 2>&1 &
echo $! > arm_d_run.pid

# monitor
tail -f arm_d_run.log
ps -p $(cat arm_d_run.pid)

# quick local test first (5 trials, 500 users)
HPT_TRIALS=5 MAX_USERS=500 ./run_arm_d.sh > arm_d_test.log 2>&1
```

resume-safe: re-run the same command if interrupted — all phases skip completed work.
per-job logs written to `logs/<timestamp>/` so parallel output does not interleave.

### pipeline phases

```
# run_all.sh full reproducible pipeline (all four arms):
Phase 1   prepare all datasets
Phase 2   snapshot v1 results (before new runs)
Phase 3   post-hoc grid (36 experiments x dataset)
Phase 4   HPT Arm A (12 jobs) + Arm B (20 jobs) + Arm D (16 jobs) all in parallel
Phase 5   Arm A grid (3 thresholds x 4 datasets = 12 jobs in parallel, saves models)
Phase 6   Arm B grid (5 thresholds x 4 datasets = 20 jobs in parallel, saves models)
Phase 7   Arm C hybrid grid (90 combinations x dataset, 12 workers per dataset)
Phase 8   known-negative injection eval (Krichene & Rendle 2020)
Phase 9   Arm D joint SVD grid (4 configurations x 4 datasets in parallel)
Phase 10  generate all figures and tables (A / B / C / D comparison)

# run_arm_d.sh Arm D only (when Phases 1–8 are already done on the server):
Phase 10  Arm D HPT (4 pairs x 4 datasets = 16 parallel jobs)
Phase 11  Arm D joint SVD grid (4 configurations x 4 datasets)
Phase 12  regenerate all figures and tables (includes Arm D comparison)
```

### individual scripts

```bash
# prepare one dataset
python main.py prepare --dataset 1m     # or 10m / 20m / spotify

# post-hoc grid (36 runs)
python main.py grid --config configs/movielens_1m.yaml

# hyperparameter tuning
python scripts/run_hyperparameter_tuning.py --config configs/movielens_1m.yaml --arm a --threshold 4
python scripts/run_hyperparameter_tuning.py --config configs/movielens_1m.yaml --arm b --threshold 2
python scripts/run_hyperparameter_tuning.py --config configs/movielens_1m.yaml --arm b --threshold median
python scripts/run_hyperparameter_tuning.py --config configs/movielens_1m.yaml --arm d --threshold 4 --neg_threshold 2

# arm grids
python scripts/run_arm_a_positive_svd_grid.py --config configs/movielens_1m.yaml
python scripts/run_arm_b_negative_svd_grid.py --config configs/movielens_1m.yaml
python scripts/run_arm_c_hybrid_grid.py       --config configs/movielens_1m.yaml
python scripts/run_arm_d_joint_svd_grid.py    --config configs/movielens_1m.yaml

# known-negative injection eval
python scripts/run_known_negative_eval.py --config configs/movielens_1m.yaml

# generate figures
python scripts/generate_all_figures.py
```

---

## project structure

```
src/
  data/
    prepare_movielens.py    cold-start filter, id remap, split
    prepare_spotify.py      Spotify MSSD skip-level weighting
    threshold_utils.py      label negative items per user
  models/
    svd_baseline.py         SVD training, id mapping, cosine similarity, predict_batch
    negative_variants.py    filter / rerank / weighted penalty
  eval/
    metrics.py              nDCG, HR, MRR, sim_to_neg, neg_detection_*, mean_dislike_rank
    ranking_metrics.py      full eval pipeline (500 candidates, threaded)
    statistical_tests.py    paired t-test, Wilcoxon, confidence intervals
  utils/
    config.py               YAML -> typed dataclasses
    io.py                   load/save parquet, JSON
    seed.py                 reproducible seeds

scripts/
  run_hyperparameter_tuning.py    Optuna HPT for Arm A / Arm B / Arm D
  run_arm_a_positive_svd_grid.py  Arm A grid (pos-only SVD)
  run_arm_b_negative_svd_grid.py  Arm B grid (neg-only SVD, LNO eval)
  run_arm_c_hybrid_grid.py        Arm C grid (90 combinations)
  run_arm_d_joint_svd_grid.py     Arm D grid (joint pos-neg SVD, binary targets)
  run_known_negative_eval.py      inject dislikes into candidate pool
  run_train_positive_grid.py      simple pos-only baseline (no HPT)
  snapshot_results.py             save current outputs before overwriting
  generate_all_figures.py         all figures and LaTeX tables

run_all.sh      full pipeline Phases 1–9 (all arms including Arm C)
run_arm_d.sh    Arm D only  Phases 10–12 (assumes Phases 1–9 done)

configs/
  movielens_1m.yaml
  movielens_10m.yaml
  movielens_20m.yaml
  spotify.yaml

outputs/
  movielens/ml-1m/
    grid_summary.json               post-hoc (36 experiments)
    grid_summary_arm_a.json         Arm A (3 thresholds)
    grid_summary_arm_b.json         Arm B (5 thresholds, LNO metrics)
    grid_summary_arm_c.json         Arm C (90 combinations)
    grid_summary_arm_d.json         Arm D (4 pos/neg pairs, binary targets)
    grid_summary_known_neg_eval.json
    models/arm_a/<label>/model.pkl  saved for Arm C
    models/arm_b/<label>/model.pkl  saved for Arm C
    models/arm_d/<label>/model.pkl  saved for potential Arm E
    tuning/arm_a/<label>/best_params.json
    tuning/arm_b/<label>/best_params.json
    tuning/arm_d/<label>/best_params.json
  snapshots/v1_post_hoc/           results before new runs

reports/
  figures/    PDF + PNG figures
  tables/     LaTeX tables
```

---

## figures and tables

| figure | what |
|--------|------|
| fig1 | baseline sanity check: actual NDCG/HR vs random-ranker reference (was: estimated RMSE) |
| fig2 | core tradeoff: sim_to_neg vs NDCG scatter per variant |
| fig3 | best NDCG per variant per dataset |
| fig4 | alpha sensitivity: any penalty level hurts NDCG |
| fig5 | negative threshold comparison |
| fig6 | dataset scaling (1M → 10M → 20M) |
| fig7 | MovieLens vs Spotify (skip→rating: skip-before-30s=1, skip-before-end=2, full-play=4, +queue=5) |
| fig8 | known-negative injection eval (hatched bars = 0 by construction, not measured) |
| fig9 | post-hoc vs train-positive SVD (relabelled: not "Hu et al. 2008" — different method) |
| fig_arm_b | Arm B dislike detection quality per threshold per dataset |
| fig_arm_c | Arm A vs Arm C headline: NDCG and sim_to_neg per dataset |
| fig_arm_d_comparison | Arm A vs Arm D vs Arm C: NDCG@10 and sim_to_neg@10 per dataset (Δ% vs baseline annotated on bars) |

| table | what |
|-------|------|
| A | dataset statistics (users, items, ratings, feedback type, split) |
| 1 | all experiments (appendix) |
| 2 | baseline summary + quality verdict |
| 3 | Surprise RMSE benchmark (rating prediction, different task) |
| 4 | key findings: best variant per dataset + delta |
| 5 | V1 post-hoc vs V2 train-positive SVD comparison |
| C | Arm B detection quality per threshold (LNO evaluation) |
| D | Arm A vs Arm C headline (main research question) |
| E | Arm D headline: best joint pos-neg config per dataset (nDCG@10, HR@10, MRR, sim_to_neg@10) |
| F | Arm A vs Arm D per dataset: absolute and relative delta for each metric |
| G | strategy summary: all 6 strategies compared across all datasets |

**known scientific limitations:**
- significance tests not yet added requires per-user CSV output from grid runs
- single seed  no variance across random seeds
- plausibility range in fig1b is own estimate, not a published benchmark (labeled as such)
- train-positive SVD ≠ Hu et al. 2008 (confidence-weighted ALS on all interactions)

---

## papers
Koren, Bell, Volinsky (2009) — Matrix Factorization Techniques for Recommender Systems
  https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf

- Järvelin, Kekäläinen (2002) — Cumulated Gain-Based Evaluation of IR Techniques (nDCG)
  https://dl.acm.org/doi/10.1145/582415.582418

- Krichene, Rendle (2020) — On Sampled Metrics for Item Recommendation (500 candidates)
  https://dl.acm.org/doi/10.1145/3383313.3412259

- He, Liao, Zhang, Nie, Hu, Chua (2017) — Neural Collaborative Filtering (cold-start threshold)
  https://dl.acm.org/doi/10.1145/3038912.3052569

- Koren (2010) — Collaborative Filtering with Temporal Dynamics (temporal split)
  https://dl.acm.org/doi/10.1145/1721654.1721677
  
| paper | used for |
|-------|---------|
| Koren, Bell, Volinsky 2009 | SVD matrix factorization baseline |
| Järvelin, Kekäläinen 2002 | nDCG metric |
| Krichene, Rendle 2020 (KDD) | sampled evaluation (500 candidates), known-neg injection |
| Rendle et al. 2009 (BPR, UAI) | positive-only training (Arm A) |
| He et al. 2017 (NCF, WWW) | positive-only training (Arm A) |
| Cena et al. 2004 | explicit dislikes as recommendation constraints (Arm B, Arm D) |
| Paudel et al. 2016 | dislike-avoidance in top-N (Arm B / C / D) |
| Fan et al. 2024 | random split for MovieLens |