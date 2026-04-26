#!/bin/bash

set -e

echo "Phase 1: MovieLens 1M"
python main.py grid --config configs/movielens_1m.yaml

echo "Phase 2: MovieLens 10M"
python main.py prepare --dataset 10m
python main.py grid --config configs/movielens_10m.yaml

echo "Phase 3: MovieLens 20M"
python main.py prepare --dataset 20m
python main.py grid --config configs/movielens_20m.yaml

echo "Phase 4: Spotify"
python main.py prepare --dataset spotify --raw_dir data/raw/spotify/data
python main.py grid --config configs/spotify.yaml

echo "Phase 5: Final Figures"
python scripts/generate_all_figures.py

echo "=== Push results ==="
git add -f outputs/movielens/ml-1m/grid_summary.json \
        outputs/movielens/ml-10m/grid_summary.json \
        outputs/movielens/ml-20m/grid_summary.json \
        outputs/spotify/grid_summary.json \
        reports/
git commit -m "add experiment results and figures"
git push

echo "DONE"

