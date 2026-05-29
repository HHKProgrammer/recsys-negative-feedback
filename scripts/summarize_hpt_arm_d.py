"""
Arm D HPT Summary  reads ALL log directories and produces a consolidated
results table (markdown + CSV) without overwriting anything.

Reads from:  logs/<timestamp>_arm_d/hpt_d_p<P>_n<N>_<dataset>.log
             outputs/<dataset>/tuning/arm_d/<label>/best_params.json 
             outputs/<dataset>/tuning/arm_d/<label>/trials.csv         

Writes to:   logs/hpt_arm_d_summary.md   
             logs/hpt_arm_d_summary.csv  

Usage:
    python scripts/summarize_hpt_arm_d.py
    python scripts/summarize_hpt_arm_d.py --verbose   
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOGS_DIR = ROOT / "logs"
OUTPUTS_DIR = ROOT / "outputs"

DS_MAP = {
    "movielens_1m": ("movielens", "ml-1m"),
    "movielens_10m": ("movielens", "ml-10m"),
    "movielens_20m": ("movielens", "ml-20m"),
    "spotify": ("spotify", None),
}
CONFIGS = ["p4_n2", "p4_n1", "p5_n1", "p3_n2"]
DATASETS = ["movielens_1m", "movielens_10m", "movielens_20m", "spotify"]

THRESHOLD_LABELS = {
    "p4_n2": "pos≥4 / neg≤2",
    "p4_n1": "pos≥4 / neg≤1",
    "p5_n1": "pos≥5 / neg≤1",
    "p3_n2": "pos≥3 / neg≤2",
}
DS_LABELS = {
    "movielens_1m": "ML-1M",
    "movielens_10m": "ML-10M",
    "movielens_20m": "ML-20M",
    "spotify": "Spotify",
}


def parse_log(path: Path) -> dict:
    """Extract all useful info from one HPT log file."""
    result = {
        "log_path": str(path),
        "run_dir": path.parent.name,
        "engine": "unknown",
        "trials_done": 0,
        "trials_target": 0,
        "best_ndcg": None,
        "best_params": None,
        "train_rows": None,
        "pos_rows": None,
        "neg_rows": None,
        "skipped": False,
        "convergence": [],  
    }
    if not path.exists():
        return result

    text = path.read_text(errors="replace")

    if "SKIP:" in text:
        result["skipped"] = True
        return result

    # Engine
    m = re.search(r"Engine:\s+(.+)", text)
    if m:
        result["engine"] = m.group(1).strip()

    # Training data
    m = re.search(r"(\S[\d,]+) positives \+ (\S[\d,]+) negatives = (\S[\d,]+) rows", text)
    if m:
        result["pos_rows"] = int(m.group(1).replace(",", ""))
        result["neg_rows"] = int(m.group(2).replace(",", ""))
        result["train_rows"] = int(m.group(3).replace(",", ""))

    # Per-trial best lines: two formats
    # Random search:  [N/200] best nDCG@10=X
    # Optuna:         [N/200] nDCG@10=Y  best=X  trial_time=...
    for line in text.splitlines():
        m_rs = re.match(r"\s*\[(\d+)/(\d+)\]\s+best nDCG@10=([0-9.]+)", line)
        m_op = re.match(r"\s*\[(\d+)/(\d+)\]\s+nDCG@10=[0-9.]+\s+best=([0-9.]+)", line)
        m = m_rs or m_op
        if m:
            n = int(m.group(1))
            target = int(m.group(2))
            val = float(m.group(3))
            result["convergence"].append((n, val))
            result["trials_done"] = n
            result["trials_target"] = target
            if result["best_ndcg"] is None or val > result["best_ndcg"]:
                result["best_ndcg"] = val

    # Best params line
    m = re.search(r"Best params:\s*(\{.+\})", text)
    if m:
        try:
            result["best_params"] = eval(m.group(1))  
        except Exception:
            result["best_params"] = m.group(1)

    # Final best line 
    m = re.search(r"Best nDCG@10.*?:\s*([0-9.]+)", text)
    if m:
        result["best_ndcg"] = float(m.group(1))

    return result


def load_best_params_json(cfg: str, ds: str):
    """Load best_params.json from outputs if it exists."""
    label_map = {
        "p4_n2": "pos_ge_4_neg_le_2",
        "p4_n1": "pos_ge_4_neg_le_1",
        "p5_n1": "pos_ge_5_neg_le_1",
        "p3_n2": "pos_ge_3_neg_le_2",
    }
    label = label_map[cfg]
    parts = DS_MAP.get(ds)
    if parts is None:
        return None
    cat, sub = parts
    if sub:
        base = OUTPUTS_DIR / cat / sub / "tuning" / "arm_d" / label
    else:
        base = OUTPUTS_DIR / cat / "tuning" / "arm_d" / label
    p = base / "best_params.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def load_trials_csv(cfg: str, ds: str):
    """Load trials.csv from outputs if it exists."""
    label_map = {
        "p4_n2": "pos_ge_4_neg_le_2",
        "p4_n1": "pos_ge_4_neg_le_1",
        "p5_n1": "pos_ge_5_neg_le_1",
        "p3_n2": "pos_ge_3_neg_le_2",
    }
    label = label_map[cfg]
    parts = DS_MAP.get(ds)
    if parts is None:
        return []
    cat, sub = parts
    if sub:
        base = OUTPUTS_DIR / cat / sub / "tuning" / "arm_d" / label
    else:
        base = OUTPUTS_DIR / cat / "tuning" / "arm_d" / label
    p = base / "trials.csv"
    if not p.exists():
        return []
    rows = []
    try:
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    except Exception:
        pass
    return rows


def collect_all_results() -> dict:
    """
    For each (cfg, ds) pair, collect results from ALL log run directories
    and return the best across all runs.
    """
    results = {}

    run_dirs = sorted(LOGS_DIR.glob("*_arm_d"), key=lambda p: p.name)

    for cfg in CONFIGS:
        for ds in DATASETS:
            key = (cfg, ds)
            results[key] = {
                "cfg": cfg,
                "ds": ds,
                "best_ndcg": None,
                "best_run": None,
                "best_trials": 0,
                "best_engine": "unknown",
                "best_params": None,
                "all_runs": [],
                "train_rows": None,
                "pos_rows": None,
                "neg_rows": None,
                "skipped_in_latest": False,
            }

            for run_dir in run_dirs:
                log_path = run_dir / f"hpt_d_{cfg}_{ds}.log"
                r = parse_log(log_path)
                if r["skipped"]:
                    if run_dir == run_dirs[-1]:
                        results[key]["skipped_in_latest"] = True
                    continue

                entry = {
                    "run": run_dir.name,
                    "trials_done": r["trials_done"],
                    "trials_target": r["trials_target"],
                    "best_ndcg": r["best_ndcg"],
                    "engine": r["engine"],
                    "convergence": r["convergence"],
                }
                results[key]["all_runs"].append(entry)

                if r["train_rows"] and not results[key]["train_rows"]:
                    results[key]["train_rows"] = r["train_rows"]
                    results[key]["pos_rows"] = r["pos_rows"]
                    results[key]["neg_rows"] = r["neg_rows"]

                if r["best_ndcg"] is not None:
                    if results[key]["best_ndcg"] is None or r["best_ndcg"] > results[key]["best_ndcg"]:
                        results[key]["best_ndcg"] = r["best_ndcg"]
                        results[key]["best_run"] = run_dir.name
                        results[key]["best_trials"] = r["trials_done"]
                        results[key]["best_engine"] = r["engine"]
                        if r["best_params"]:
                            results[key]["best_params"] = r["best_params"]

            bp = load_best_params_json(cfg, ds)
            if bp and "best_params" in bp:
                results[key]["best_params"] = bp["best_params"]
                if results[key]["best_ndcg"] is None and "best_value" in bp:
                    results[key]["best_ndcg"] = bp["best_value"]

            trials_rows = load_trials_csv(cfg, ds)
            if trials_rows:
                results[key]["csv_total_trials"] = len(trials_rows)
                best_from_csv = max((float(r.get("score", 0)) for r in trials_rows), default=None)
                if best_from_csv and (results[key]["best_ndcg"] is None or best_from_csv > results[key]["best_ndcg"]):
                    results[key]["best_ndcg"] = best_from_csv
            else:
                results[key]["csv_total_trials"] = None

    return results


def fmt(v, decimals=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def params_short(p) -> str:
    if p is None:
        return "—"
    if isinstance(p, dict):
        f = p.get("n_factors", "?")
        ep = p.get("n_epochs", "?")
        lr = p.get("lr_all", "?")
        reg = p.get("reg_all", "?")
        biased = p.get("biased", "?")
        return f"f={f} ep={ep} lr={lr} reg={reg} bias={biased}"
    return str(p)


def build_markdown_table(results: dict) -> str:
    lines = []
    lines.append("| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |")
    lines.append("|--------|---------|--------|------------|--------|---------------------|")
    for cfg in CONFIGS:
        for ds in DATASETS:
            r = results[(cfg, ds)]
            trials_str = str(r["best_trials"]) if r["best_trials"] else "—"
            if r.get("csv_total_trials"):
                trials_str = str(r["csv_total_trials"])
            engine = "Optuna TPE" if "Optuna" in r["best_engine"] else "Random Search"
            engine_short = "TPE" if "Optuna" in r["best_engine"] else "RS"
            lines.append(
                f"| `{cfg}` {THRESHOLD_LABELS[cfg]} | {DS_LABELS[ds]} "
                f"| {trials_str} | **{fmt(r['best_ndcg'])}** "
                f"| {engine_short} | {params_short(r['best_params'])} |"
            )
    return "\n".join(lines)


def build_dataset_comparison(results: dict) -> str:
    """Best nDCG per config × dataset — compact overview."""
    lines = []
    header = "| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |"
    sep    = "|-------------------|---------|---------|---------|---------|-------|"
    lines.append(header)
    lines.append(sep)
    for cfg in CONFIGS:
        row_vals = []
        for ds in DATASETS:
            v = results[(cfg, ds)]["best_ndcg"]
            row_vals.append(v)
        best_v = max((v for v in row_vals if v is not None), default=None)
        cells = []
        for v in row_vals:
            s = fmt(v)
            if v is not None and v == best_v:
                s = f"**{s}**"
            cells.append(s)
        lines.append(
            f"| `{cfg}` {THRESHOLD_LABELS[cfg]:<12} | {' | '.join(cells)} | {fmt(best_v)} |"
        )
    return "\n".join(lines)


def build_convergence_block(results: dict) -> str:
    lines = []
    for cfg in CONFIGS:
        for ds in DATASETS:
            r = results[(cfg, ds)]
            for run in r["all_runs"]:
                if not run["convergence"]:
                    continue
                conv = run["convergence"]
                lines.append(f"\n**{cfg}/{ds}** ({run['run']}, {run['trials_done']} trials, {run['engine'].split()[0]})")
                lines.append("```")
                prev = None
                for trial, val in conv:
                    marker = " ← new best" if (prev is None or val > prev) else ""
                    lines.append(f"  [{trial:>3}/{run['trials_target']}]  {val:.4f}{marker}")
                    prev = max(val, prev) if prev is not None else val
                lines.append("```")
    return "\n".join(lines)


def write_csv(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in CONFIGS:
        for ds in DATASETS:
            r = results[(cfg, ds)]
            bp = r["best_params"] or {}
            rows.append({
                "config": cfg,
                "threshold_label": THRESHOLD_LABELS[cfg],
                "dataset": DS_LABELS[ds],
                "trials": r.get("csv_total_trials") or r["best_trials"] or "",
                "best_ndcg_10": fmt(r["best_ndcg"]) if r["best_ndcg"] else "",
                "engine": "Optuna TPE" if "Optuna" in r["best_engine"] else "Random Search",
                "best_run_dir": r["best_run"] or "",
                "n_factors": bp.get("n_factors", ""),
                "n_epochs": bp.get("n_epochs", ""),
                "lr_all": bp.get("lr_all", ""),
                "reg_all": bp.get("reg_all", ""),
                "biased": bp.get("biased", ""),
                "train_rows": r["train_rows"] or "",
                "pos_rows": r["pos_rows"] or "",
                "neg_rows": r["neg_rows"] or "",
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(content: str, path: Path, append: bool = True) -> None:
    """Write to path. If append=True and file exists, prepend a timestamp block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n---\n\n## Run: {ts}\n\n"
    if append and path.exists():
        old = path.read_text()
        path.write_text(header + content + "\n\n" + old)
    else:
        path.write_text(header + content + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Include convergence curves")
    parser.add_argument("--no-append", action="store_true", help="Overwrite summary markdown instead of prepending")
    args = parser.parse_args()

    print("Scanning log directories...")
    results = collect_all_results()

    # Print  stdout
    print("\n" + "=" * 72)
    print("ARM D HPT SUMMARY — best validation nDCG@10 across all runs")
    print("=" * 72)
    print()
    print("OVERVIEW TABLE (best per config × dataset):")
    print()
    print(build_dataset_comparison(results))
    print()

    print("DETAILED TABLE (trials, engine, hyperparameters):")
    print()
    print(build_markdown_table(results))
    print()

    # Training data sizes
    print("TRAINING SET SIZES (from logs):")
    for cfg in CONFIGS:
        for ds in DATASETS:
            r = results[(cfg, ds)]
            if r["train_rows"]:
                pos_pct = 100 * r["pos_rows"] / r["train_rows"] if r["train_rows"] else 0
                print(f"  {cfg}/{DS_LABELS[ds]}: {r['train_rows']:>12,} rows  "
                      f"({r['pos_rows']:,} pos / {r['neg_rows']:,} neg = {pos_pct:.0f}% positive)")

    # Convergence
    if args.verbose:
        print()
        print("CONVERGENCE DETAILS:")
        print(build_convergence_block(results))

    # Write files
    md_content = (
        "# Arm D HPT Results\n\n"
        "## Overview\n\n" + build_dataset_comparison(results) + "\n\n"
        "## Detailed Results\n\n" + build_markdown_table(results) + "\n\n"
    )
    if args.verbose:
        md_content += "## Convergence Curves\n\n" + build_convergence_block(results) + "\n\n"

    md_path = LOGS_DIR / "hpt_arm_d_summary.md"
    csv_path = LOGS_DIR / "hpt_arm_d_summary.csv"

    write_markdown(md_content, md_path, append=not args.no_append)
    write_csv(results, csv_path)

    print(f"\nSaved → {md_path}")
    print(f"Saved → {csv_path}")
    print("(Markdown is prepended — old runs preserved at bottom)")


if __name__ == "__main__":
    main()