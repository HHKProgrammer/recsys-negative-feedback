import json
import optuna
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--study", required=True)
parser.add_argument("--study_name", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

storage = f"sqlite:///{args.study}"

# Load the study with the correct name
study = optuna.load_study(
    study_name=args.study_name,
    storage=storage
)

best = study.best_trial

params = {
    "n_factors": best.params.get("n_factors"),
    "n_epochs": best.params.get("n_epochs"),
    "lr_all": best.params.get("lr_all"),
    "reg_all": best.params.get("reg_all"),
    "biased": best.params.get("biased"),
}

with open(args.output, "w") as f:
    json.dump(params, f, indent=2)

print("Saved →", args.output)
