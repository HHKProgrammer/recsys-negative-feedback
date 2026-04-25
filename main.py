"""
Entry point for all experiments.

Commands
--------
prepare   Generate train/test/val/user_thresholds parquet files
run       Run a single experiment
grid      Run the full experiment grid (36 runs for ML-1M)

Examples
--------
python main.py prepare --dataset 1m
python main.py prepare --dataset spotify --raw_dir data/raw/spotify
python main.py run --config configs/movielens_1m.yaml --variant baseline
python main.py run --config configs/movielens_1m.yaml --variant rerank --threshold_type fixed --fixed_threshold 2 --alpha 0.3
python main.py grid --config configs/movielens_1m.yaml
python main.py grid --config configs/movielens_1m.yaml --quick
"""
import argparse


def cmd_prepare(args):
    if args.dataset == "spotify":
        from src.data.prepare_spotify import prepare_spotify

        output_dir = args.output_dir or "data/processed/spotify"
        prepare_spotify(
            raw_dir=args.raw_dir,
            output_dir=output_dir,
        )
    else:
        from src.data.prepare_movielens import prepare_movielens

        output_dir = args.output_dir or f"data/processed/movielens/ml-{args.dataset}"
        prepare_movielens(
            raw_dir=args.raw_dir,
            output_dir=output_dir,
            dataset=args.dataset,
            min_ratings=args.min_ratings,
        )


def cmd_run(args):
    from src.runners.run_experiment import run_single_experiment
    from src.utils.config import ExperimentConfig

    config = ExperimentConfig.from_yaml(args.config)
    metrics = run_single_experiment(
        config=config,
        variant=args.variant,
        threshold_type=args.threshold_type,
        fixed_threshold=args.fixed_threshold,
        alpha=args.alpha,
        max_users=args.max_users,
    )

    print("\n--- Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def cmd_grid(args):
    from src.runners.run_grid import run_full_grid

    run_full_grid(config_path=args.config, quick=args.quick)


def main():
    parser = argparse.ArgumentParser(
        description="Negative Feedback RecSys Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # ── prepare ──────────────────────────────────────────────────────────────
    p_prep = sub.add_parser("prepare", help="Prepare dataset splits")
    p_prep.add_argument("--dataset", choices=["100k", "1m", "10m", "20m", "spotify"], default="1m")
    p_prep.add_argument("--raw_dir", default="data/raw/movielens")
    p_prep.add_argument("--output_dir", default=None)
    p_prep.add_argument("--min_ratings", type=int, default=5)

    # ── run ──────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run one experiment")
    p_run.add_argument("--config", required=True)
    p_run.add_argument(
        "--variant", required=True,
        choices=["baseline", "filter", "rerank", "weighted"],
    )
    p_run.add_argument(
        "--threshold_type", default=None,
        choices=["fixed", "median", "modus"],
    )
    p_run.add_argument("--fixed_threshold", type=int, default=None)
    p_run.add_argument("--alpha", type=float, default=None)
    p_run.add_argument("--max_users", type=int, default=None)

    # ── grid ─────────────────────────────────────────────────────────────────
    p_grid = sub.add_parser("grid", help="Run full experiment grid")
    p_grid.add_argument("--config", required=True)
    p_grid.add_argument("--quick", action="store_true",
                        help="Reduced grid for quick sanity check")

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "grid":
        cmd_grid(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
