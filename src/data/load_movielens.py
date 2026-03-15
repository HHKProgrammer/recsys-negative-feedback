# pipeline step 1 — data loading
# simple loader that reads the processed parquet splits
# prepare_movielens.py must be run first to generate these files
# used by notebooks and quick scripts that just need train/test without full pipeline

from pathlib import Path
import pandas as pd

# Going up 2 levels (src/data → src → project root) gives us the project root
# parents[2] means: this file is at depth 2 from root, so walk up twice
# makes the path work no matter where the script is called from
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_movielens(baseDir=None):
    # If no directory is given  use the project root we computed above
    if baseDir is None:
        baseDir = _PROJECT_ROOT

    # Processed parquet files live under data/processed/movielens/
    procDir = baseDir / "data" / "processed" / "movielens"
    trainDf = pd.read_parquet(procDir / "train.parquet")
    testDf  = pd.read_parquet(procDir / "test.parquet")

    return trainDf, testDf
