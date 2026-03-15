# pipeline utility — config loading
# reads the yaml file and maps all parameters into typed python dataclasses
# this way i can do config.model.n_factors instead of raw dict lookups
# every experiment saves its config to disk so results are fully reproducible

# Configuration management via dataclasses loaded from yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

# to document all parameters and ensure reptoducability
@dataclass
class DataConfig:
    # name of the dataset and where the raw/processed files live
    name: str
    raw_path: str
    processed_path: str


#splits in tain eval,.. test
@dataclass
class SplitConfig:
    # filenames for the five parquet splits produced by prepare_movielens
    # temporal leave-one-out — last item per user is test, second-last is val
    train_file: str = "train.parquet"
    train_inner_file: str = "train_inner.parquet"
    val_file: str = "val.parquet"
    test_file: str = "test.parquet"
    user_thresholds_file: str = "user_thresholds.parquet"


# svd config
@dataclass
class ModelConfig:
    # hyperparameters tuned in notebooks/learning/03_hyperparameter_tuning.ipynb
    # n_factors=100 latent dimensions, n_epochs=20 passes through training data
    # lr_all=0.01 learning rate for sgd, reg_all=0.05 l2 regularization
    # reference — koren bell volinsky 2009, matrix factorization techniques
    # https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf
    name: str = "SVD"
    n_factors: int = 100
    n_epochs: int = 20
    lr_all: float = 0.01
    reg_all: float = 0.05


#relaying on supervisor trying diffrent thresholds and modus and median
@dataclass
class NegativeFeedbackConfig:
    # Fixed thresholds: rating <= threshold is negative
    # threshold=1 only very bad ratings, threshold=3 includes anything below average
    fixed_thresholds: List[int] = field(default_factory=lambda: [1, 2, 3])
    use_median: bool = True  # negative if rating < user_median
    use_modus: bool = True   # negative if rating < user_mode
    # alpha controls how strong the penalty is — 0.1 gentle, 1.0 aggressive
    alphas: List[float] = field(default_factory=lambda: [0.1, 0.3, 1.0])
    pos_threshold: int = 4   # items rated >= 4 count as positive


@dataclass
class EvalConfig:
    # k=10 top-k recommendations, n_candidates=500 random unseen items per user
    # 500 candidates is standard in sampled evaluation for recsys
    # reference — krichene rendle 2020, on sampled metrics for item recommendation
    # https://dl.acm.org/doi/10.1145/3383313.3412259
    k: int = 10
    n_candidates: int = 500
    random_seed: int = 42


#just for testing
@dataclass
class QuickModeConfig:
    # reduced settings for fast sanity checks — 500 users instead of 6040
    # useful to verify the pipeline works without waiting 2+ hours
    n_candidates: int = 100
    max_users: int = 500
    thresholds: List[int] = field(default_factory=lambda: [2])
    alphas: List[float] = field(default_factory=lambda: [0.3])
    variants: List[str] = field(default_factory=lambda: ["baseline", "filter", "rerank"])


#config structure.. splits output dir
@dataclass
class ExperimentConfig:
    # top-level config — holds all sub-configs as nested dataclasses
    # from_yaml() is the main constructor, to_dict() serializes for saving to disk
    data: DataConfig
    splits: SplitConfig
    model: ModelConfig
    negative_feedback: NegativeFeedbackConfig
    eval: EvalConfig
    quick_mode: QuickModeConfig
    output_dir: str
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        # loads yaml file and maps each section to its dataclass
        # **raw["dataset"] unpacks the dict as keyword args into DataConfig(...)
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        data = DataConfig(**raw["dataset"])
        splits = SplitConfig(**raw.get("splits", {}))
        model = ModelConfig(**raw.get("model", {}).get("params", {}))
        nf_raw = raw.get("negative_feedback", {})
        nf = NegativeFeedbackConfig(**nf_raw)
        ev_raw = raw.get("evaluation", {})
        ev_raw.setdefault("random_seed", raw.get("random_seed", 42))
        ev = EvalConfig(**ev_raw)
        qm_raw = raw.get("quick_mode", {})
        qm = QuickModeConfig(**qm_raw) if qm_raw else QuickModeConfig()
        output_dir = raw.get("output", {}).get("base_dir", "outputs/")

        return cls(
            data=data,
            splits=splits,
            model=model,
            negative_feedback=nf,
            eval=ev,
            quick_mode=qm,
            output_dir=output_dir,
            random_seed=raw.get("random_seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        # serializes all sub-configs to a plain dict for saving to config.json
        return {
            "data": self.data.__dict__,
            "splits": self.splits.__dict__,
            "model": self.model.__dict__,
            "negative_feedback": self.negative_feedback.__dict__,
            "eval": self.eval.__dict__,
            "output_dir": self.output_dir,
            "random_seed": self.random_seed,
        }

    def split_path(self, filename: str) -> str:
        # helper to get full path to a processed split file
        return str(Path(self.data.processed_path) / filename)
