# pipeline utility — file i/o helpers
# all read and write operations go through here so theres one place to change formats
# parquet is binary columnar format — much faster than csv for large dataframes
# for ~1m rows: parquet loads in ~0.3s vs csv ~3s, and uses 7x less disk space

#File IO utilities.
import json
import os
from typing import Any, Dict

import pandas as pd
import yaml

#csv was too slow so i changed to parquet and less storage
def load_yaml(path: str) -> Dict[str, Any]:
    # reads a yaml config file into a plain python dict
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    # creates parent dirs if missing, then dumps dict as indented json
    # default=str handles non-serializable types like datetime
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    # index=False — pandas row index is not part of the actual data
    # parquet preserves int32, float64 etc exactly, no type guessing on reload
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


##make dir  creates dir
def ensure_dir(path: str) -> str:
    # exist_ok=True means no error if dir already exists
    # returns the path so it can be used directly in assignments
    os.makedirs(path, exist_ok=True)
    return path
