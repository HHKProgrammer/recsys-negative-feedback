# pipeline utility — experiment logging and run management
# every experiment gets its own timestamped folder under outputs/
# this makes it easy to compare runs and always know exactly what settings were used
# the per-user parquet file is critical — needed for the statistical significance tests

# Experiment tracking: creates a timestamped run directory and logs everything.
import json
import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd

# for each Experiment there will be a config.json, metrics.json, metrics_per_user.parquet and a run.log.
class ExperimentTracker:
    def __init__(self, base_dir: str, experiment_name: str):
        # run_id = timestamp + experiment name, e.g. "2026-03-07_143012_weighted_fixed_3_a0.1"
        # this makes the folder name human-readable and sortable by time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_id = f"{timestamp}_{experiment_name}"
        self.run_dir = os.path.join(base_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, "run.log")
        self.start_time = datetime.now()
        self._metrics_path = os.path.join(self.run_dir, "metrics.json")
        self._init_log()

    def _init_log(self) -> None:
        # writes the header line to run.log at the start of the experiment
        with open(self.log_path, "w") as f:
            f.write(f"run_id: {self.run_id}\n")
            f.write(f"started: {self.start_time.isoformat()}\n")
            f.write("-" * 60 + "\n")

    def log(self, message: str) -> None:
        # prints to console and appends to run.log with a timestamp prefix
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        print(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def save_config(self, config: Dict[str, Any]) -> None:
        # saves the full config dict as config.json in the run folder
        # so in the future i can always see which exact parameters were used
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        self.log(f"Config saved → {path}")

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        # aggregated metrics — one number per metric, averaged over all test users
        with open(self._metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        self.log(f"Metrics saved → {self._metrics_path}")

    def save_per_user_metrics(self, df: pd.DataFrame) -> None:
        # one row per user — needed for paired t-test and wilcoxon significance tests
        # without this file i cant compare variants statistically
        path = os.path.join(self.run_dir, "metrics_per_user.parquet")
        df.to_parquet(path, index=False)
        self.log(f"Per-user metrics saved → {path}")

    #rhere meassure time using metrics.json + runtime
    def finish(self) -> None:
        elapsed = datetime.now() - self.start_time
        self.log(f"Finished. Runtime: {elapsed}")
        # Append runtime to metrics file if it exists
        if os.path.exists(self._metrics_path):
            with open(self._metrics_path, "r") as f:
                metrics = json.load(f)
            metrics["runtime_seconds"] = elapsed.total_seconds()
            with open(self._metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
