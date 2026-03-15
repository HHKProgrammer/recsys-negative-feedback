# recsys-negative-feedback

bachelor thesis project  can negative feedback improve recommendations?

the main question i am investigating: most recommender systems only learn from what users liked, but users also tell us what they hate through low ratings, can i use that signal to push recommendations away from content the user dislikes?

**dataset:** MovieLens 1M — ~1 million ratings, 6040 users, 3706 movies

**research question:** does integrating negative signals (low ratings) into the ranking step improve recommendation quality over a standard SVD baseline?

---

## what i built

a complete experiment pipeline with 36 configurations, comparing one baseline against three negative feedback strategies across five threshold types and three penalty strengths

**baseline:** standard SVD matrix factorization  learns user and item latent vectors, predicts ratings, ranks by predicted score

**three negative feedback variants:**

| variant | what it does |
|---------|-------------|
| `filter` | removes disliked items from candidates before ranking |
| `rerank` | subtracts `alpha * max_similarity_to_hated_item` from the svd score |
| `weighted` | like rerank but weights each hated item by how bad its rating was — 1-star counts more than 3-star |

**key finding:** the weighted penalty variant with a broad threshold (rating <= 3) and gentle penalty strength (alpha=0.1) achieves the best nDCG@10 across all 36 configurations, showing that negative feedback helps when applied carefully, aggressive penalization (alpha=1.0) hurts performance

---

## how the pipeline works

```
raw data (ratings.dat)
    -> prepare_movielens.py   cold-start filter, id remap, temporal split
    -> threshold_utils.py     label which training ratings are "negative"
    -> svd_baseline.py        train SVD matrix factorization
    -> negative_variants.py   apply filter / rerank / weighted penalty at ranking time
    -> ranking_metrics.py     sample 500 candidates per user, rank, measure
    -> statistical_tests.py   paired t-test, wilcoxon, 95% confidence intervals
    -> run_grid.py             run all 36 experiments, save summary
```

---

## setup (Ubuntu / WSL)

### install miniconda (first time only)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
~/miniconda3/bin/conda init bash
exec bash
```

accept conda terms:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### create the environment

```bash
conda create -n recsys-conda python=3.8 -y
conda activate recsys-conda
conda install -c conda-forge pandas numpy pyarrow jupyter scikit-learn scikit-surprise scipy matplotlib seaborn ipykernel -y
```

or from the environment file:

```bash
conda env create -f environment.yml
```

### register the jupyter kernel for VS Code

```bash
python -m ipykernel install --user --name recsys-conda --display-name "Python 3.8 (recsys)"
```

### daily workflow

```bash
conda activate recsys-conda
cd /path/to/recsys-negative-feedback
```

---

## project structure

```
recsys-negative-feedback/
├── src/
│   ├── data/
│   │   ├── prepare_movielens.py   # step 2 — cold-start filter, remap, temporal split
│   │   ├── threshold_utils.py     # step 3 — label negative items per user
│   │   └── load_movielens.py      # simple loader for notebooks
│   ├── models/
│   │   ├── svd_baseline.py        # step 4 — SVD training, id mapping, cosine similarity
│   │   └── negative_variants.py   # step 5 — filter, rerank, weighted penalty variants
│   ├── eval/
│   │   ├── metrics.py             # step 6 — P@K, R@K, nDCG@K, Hit@K, MRR, sim_to_neg@K
│   │   ├── ranking_metrics.py     # step 6 — full evaluation pipeline with candidate sampling
│   │   └── statistical_tests.py   # step 7 — paired t-test, wilcoxon, confidence interval
│   ├── runners/
│   │   ├── run_experiment.py      # step 8 — single experiment (train + eval + save)
│   │   └── run_grid.py            # step 9 — full 36-run grid, resume-safe checkpoint
│   └── utils/
│       ├── config.py              # YAML config loaded into typed dataclasses
│       ├── experiment_tracker.py  # timestamped run directories with logs
│       ├── io.py                  # load/save parquet, JSON, YAML
│       └── seed.py                # reproducible random seeds
├── configs/
│   └── movielens_1m.yaml          # all hyperparameters for the ML-1M experiments
├── data/
│   ├── raw/movielens/ml-1m/       # ratings.dat goes here (not in git — too large)
│   └── processed/movielens/ml-1m/ # generated splits (not in git)
├── outputs/movielens/ml-1m/runs/  # one folder per experiment (not in git)
├── tests/
│   ├── test_metrics.py
│   └── test_thresholds.py
└── main.py                        # entry point for all commands
```

---

## running experiments

all commands assume you are in the project root with the conda env active

### step 1 — prepare data splits

downloads and processes the raw ratings into five parquet files:

```bash
python main.py prepare --dataset 1m
```

this produces: `train.parquet`, `test.parquet`, `train_inner.parquet`, `val.parquet`, `user_thresholds.parquet`

### step 2 — run a single experiment

```bash
# baseline (no negative feedback)
python main.py run --config configs/movielens_1m.yaml --variant baseline

# filter variant, fixed threshold=2
python main.py run --config configs/movielens_1m.yaml \
    --variant filter --threshold_type fixed --fixed_threshold 2

# rerank variant, median threshold, alpha=0.3
python main.py run --config configs/movielens_1m.yaml \
    --variant rerank --threshold_type median --alpha 0.3

# weighted variant (best result), fixed threshold=3, alpha=0.1
python main.py run --config configs/movielens_1m.yaml \
    --variant weighted --threshold_type fixed --fixed_threshold 3 --alpha 0.1
```

### step 3 — run the full grid (36 experiments)

```bash
python main.py grid --config configs/movielens_1m.yaml
```

if interrupted, re-run the same command — it picks up from where it stopped

### quick mode (4 experiments, ~5 min, for sanity checks)

```bash
python main.py grid --config configs/movielens_1m.yaml --quick
```

### run tests

```bash
python -m pytest tests/ -v
```

---

## experiment grid

| # | variant | threshold | alpha |
|---|---------|-----------|-------|
| 1 | baseline | — | — |
| 2–4 | filter | fixed {1, 2, 3} | — |
| 5–6 | filter | median, modus | — |
| 7–21 | rerank | all 5 thresholds | {0.1, 0.3, 1.0} |
| 22–36 | weighted | all 5 thresholds | {0.1, 0.3, 1.0} |

**total: 36 experiments**

### threshold semantics

negative feedback is defined differently by each threshold type:

- **fixed** (`threshold_type=fixed`): rating **<=** threshold is negative
  e.g. threshold=2 means ratings {1, 2} are negative
- **median** (`threshold_type=median`): rating **<** user's median is negative
  e.g. user with median=3.5 — only ratings {1, 2, 3} are negative (3.5 itself is not)
- **modus** (`threshold_type=modus`): rating **<** user's mostcommon rating is negative
  more conservative than median  ignores ratings that are at the user's usual level

the strict lessthan for median/modus is intentional  a user who typically gives 3 stars should not have 3-star ratings counted as negative

### penalty strength (alpha)

controls how much negative feedback influences the final score:

| alpha | effect |
|-------|--------|
| 0.1 | gentle — svd ranking mostly preserved, small nudge |
| 0.3 | moderate — noticeable reordering of similar items |
| 1.0 | aggressive — often hurts because it overwrites svd too strongly |

---

## results summary (33 completed experiments)

best configuration: **weighted_fixed_3_a0.1**

| metric | baseline | best variant | delta |
|--------|----------|-------------|-------|
| nDCG@10 | 0.0431 | 0.0434 | +0.0003 |
| Hit@10 | 0.0867 | 0.0869 | +0.0002 |
| MRR | 0.0419 | 0.0421 | +0.0002 |

key observations:
- filter variant produces identical results to baseline expected, because candidates are unseen items and negatives are already-rated items, so the filter never removes anything
- rerank and weighted both improve at alpha=0.1, degrade at alpha=1.0
- weighted consistently outperforms rerank across all thresholds and alpha values
- the sim_to_neg@10 metric confirms that weighted/rerank push recommendations away from the hated-item neighborhood in latent space

---

## output files

each experiment produces a folder in `outputs/movielens/ml-1m/runs/`:

```
2026-03-07_143012_weighted_fixed_3_a0.1/
├── config.json               # exact hyperparameters used
├── metrics.json              # P@10, R@10, nDCG@10, Hit@10, MRR, runtime
├── metrics_per_user.parquet  # per-user scores (needed for significance tests)
└── run.log                   # timestamped log of the entire run
```

after the full grid: `outputs/movielens/ml-1m/grid_summary.json` contains all results and best-by-metric lookup

---

## svd hyperparameters

tuned via GridSearchCV in `notebooks/learning/03_hyperparameter_tuning.ipynb`:

| param | value | meaning |
|-------|-------|---------|
| n_factors | 100 | number of latent dimensions in user/item vectors |
| n_epochs | 20 | passes through all training data during sgd |
| lr_all | 0.01 | learning rate for gradient descent updates |
| reg_all | 0.05 | l2 regularization to prevent overfitting |
| best RMSE | 0.8646 | rating prediction error on validation set |

---

## mathematical background

**svd prediction:**
```
r_hat(u,i) = mu + bu + bi + pu[u] dot qi[i]
```
where mu is global mean, bu/bi are user/item biases, pu and qi are the latent vectors

**sgd update rule** (for each training rating):
```
error e = r - r_hat(u,i)
bu  <- bu + lr * (e - reg * bu)
bi  <- bi + lr * (e - reg * bi)
pu  <- pu + lr * (e * qi - reg * pu)
qi  <- qi + lr * (e * pu - reg * qi)
```

**cosine similarity** (used to measure how similar two items are in latent space):
```
cos(a, b) = (a dot b) / (norm(a) * norm(b))
```
result of 1.0 = identical direction, 0.0 = unrelated, -1.0 = opposite

**weighted penalty** (variant C score formula):
```
weight(n) = (5 - rating_n) / 5
penalty(u,i) = mean over all negative items n of [cos(qi, qn) * weight(n)]
score(u,i) = r_hat(u,i) - alpha * penalty(u,i)
```

---

## literature references

- Koren, Bell, Volinsky (2009) — Matrix Factorization Techniques for Recommender Systems
  https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf

- Järvelin, Kekäläinen (2002) — Cumulated Gain-Based Evaluation of IR Techniques (nDCG)
  https://dl.acm.org/doi/10.1145/582415.582418

- Krichene, Rendle (2020) — On Sampled Metrics for Item Recommendation (500 candidates)
  https://dl.acm.org/doi/10.1145/3383313.3412259

- He, Liao, Zhang, Nie, Hu, Chua (2017) — Neural Collaborative Filtering (cold-start threshold)
  https://dl.acm.org/doi/10.1145/3038912.3052569

- Koren (2010) — Collaborative Filtering with Temporal Dynamics (temporal split)
  https://dl.acm.org/doi/10.1145/1721654.1721677