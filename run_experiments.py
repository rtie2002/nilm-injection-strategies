"""
Run the full NILM experiment grid and append results to CSV.

Examples:
    python run_experiments.py --suite strategy
    python run_experiments.py --suite ratio --appliance kettle
    python run_experiments.py --suite all --skip-existing
    python run_experiments.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

from model_train import CODE_DIR, configure_run_paths, evaluate_run, load_config, run_training

RESULT_COLUMNS = [
    "run_id",
    "suite",
    "model",
    "appliance",
    "train_dataset",
    "strategy_label",
    "paper_strategy",
    "rho",
    "test_dataset",
    "val_mae",
    "val_sae",
    "val_f1",
    "test_mae",
    "test_sae",
    "test_f1",
    "best_epoch",
    "best_val_loss",
    "checkpoint",
    "elapsed_s",
    "timestamp",
    "status",
    "error",
]


def load_experiments(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_run_id(model: str, appliance: str, train_dataset: str) -> str:
    return f"{model}|{appliance}|{train_dataset}"


def build_run_cfg(
    exp_cfg: dict,
    *,
    model: str,
    appliance: str,
    train_dataset: str,
    suite: str,
    strategy_label: str,
    paper_strategy: str,
    rho: int | float,
) -> dict:
    base_path = CODE_DIR / exp_cfg.get("base_config", "hyperparameter.yaml")
    cfg = load_config(base_path)

    cfg["data"]["appliance"] = appliance
    cfg["data"]["train_dataset"] = train_dataset
    cfg["data"]["val_dataset"] = exp_cfg["data"]["val_dataset"]
    cfg["data"]["test_dataset"] = exp_cfg["data"]["test_dataset"]
    cfg["model"]["name"] = model

    cfg = configure_run_paths(cfg)
    cfg["_meta"] = {
        "suite": suite,
        "strategy_label": strategy_label,
        "paper_strategy": paper_strategy,
        "rho": rho,
        "run_id": make_run_id(model, appliance, train_dataset),
    }
    return cfg


def iter_runs(exp_cfg: dict, suite: str, appliances: list[str] | None, models: list[str] | None):
    appliances = appliances or exp_cfg["appliances"]
    models = models or exp_cfg["models"]

    suites: list[tuple[str, list[dict]]] = []
    if suite in ("strategy", "all"):
        suites.append(("strategy", exp_cfg["strategy_comparison"]))
    if suite in ("ratio", "all"):
        suites.append(("ratio", exp_cfg["ratio_sensitivity"]))

    for suite_name, runs in suites:
        for run in runs:
            for model in models:
                if model.lower() == "transformer":
                    print("WARNING: transformer is not implemented yet; skipping.")
                    continue
                for appliance in appliances:
                    yield build_run_cfg(
                        exp_cfg,
                        model=model,
                        appliance=appliance,
                        train_dataset=run["train_dataset"],
                        suite=suite_name,
                        strategy_label=run["label"],
                        paper_strategy=run.get("paper_strategy", run["label"]),
                        rho=run.get("rho", ""),
                    )


def load_results_csv(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["run_id"]: row for row in rows if row.get("run_id")}


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def result_row_from_run(cfg: dict, metrics: dict) -> dict:
    meta = cfg["_meta"]
    return {
        "run_id": meta["run_id"],
        "suite": meta["suite"],
        "model": metrics.get("model", cfg["model"]["name"]),
        "appliance": metrics.get("appliance", cfg["data"]["appliance"]),
        "train_dataset": metrics.get("train_dataset", cfg["data"]["train_dataset"]),
        "strategy_label": meta["strategy_label"],
        "paper_strategy": meta["paper_strategy"],
        "rho": meta["rho"],
        "test_dataset": metrics.get("test_dataset", cfg["data"]["test_dataset"]),
        "val_mae": metrics.get("val_mae", ""),
        "val_sae": metrics.get("val_sae", ""),
        "val_f1": metrics.get("val_f1", ""),
        "test_mae": metrics.get("test_mae", ""),
        "test_sae": metrics.get("test_sae", ""),
        "test_f1": metrics.get("test_f1", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "best_val_loss": metrics.get("best_val_loss", ""),
        "checkpoint": metrics.get("checkpoint", ""),
        "elapsed_s": metrics.get("elapsed_s", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": metrics.get("status", "ok"),
        "error": metrics.get("error", ""),
    }


def should_skip(run_id: str, existing: dict[str, dict], skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    row = existing.get(run_id)
    return row is not None and row.get("status") in {"ok", "eval_only"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeatable NILM experiment grid")
    parser.add_argument(
        "--experiments",
        default=str(CODE_DIR / "experiments.yaml"),
        help="Path to experiments.yaml",
    )
    parser.add_argument(
        "--suite",
        choices=["strategy", "ratio", "all"],
        default="strategy",
        help="Which paper table to populate",
    )
    parser.add_argument("--appliance", action="append", help="Limit to appliance(s); repeatable flag")
    parser.add_argument("--model", action="append", help="Limit to model(s)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs already marked ok in CSV")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate checkpoints without training")
    parser.add_argument("--no-plots", action="store_true", help="Disable per-run figure export")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs only")
    args = parser.parse_args()

    exp_cfg = load_experiments(Path(args.experiments))
    results_path = CODE_DIR / exp_cfg["results"]["dir"] / exp_cfg["results"]["csv"]
    existing = load_results_csv(results_path)

    appliances = args.appliance or None
    models = args.model or None
    planned = list(iter_runs(exp_cfg, args.suite, appliances, models))

    print(f"Suite: {args.suite}")
    print(f"Planned runs: {len(planned)}")
    print(f"Results CSV: {results_path}")

    if args.dry_run:
        for cfg in planned:
            meta = cfg["_meta"]
            print(
                f"  [{meta['suite']}] {meta['run_id']} | "
                f"strategy={meta['paper_strategy']} | rho={meta['rho']}"
            )
        return 0

    failed = 0
    skipped = 0
    completed = 0

    for i, cfg in enumerate(planned, start=1):
        meta = cfg["_meta"]
        run_id = meta["run_id"]
        print("\n" + "=" * 60)
        print(f"Run {i}/{len(planned)}: {run_id}")
        print("=" * 60)

        if should_skip(run_id, existing, args.skip_existing):
            print("skip (already in results CSV)")
            skipped += 1
            continue

        clean_cfg = copy.deepcopy(cfg)
        clean_cfg.pop("_meta", None)

        try:
            if args.eval_only:
                metrics = evaluate_run(clean_cfg)
            else:
                metrics = run_training(clean_cfg, plot=not args.no_plots)
            row = result_row_from_run(cfg, metrics)
            append_result(results_path, row)
            existing[run_id] = row
            completed += 1
            print(
                f"saved -> test MAE {row['test_mae']:.2f} W | "
                f"SAE {row['test_sae']:.2f}% | F1 {row['test_f1']:.4f}"
            )
        except Exception as exc:
            failed += 1
            err = traceback.format_exc()
            print(f"FAILED: {exc}")
            row = result_row_from_run(
                cfg,
                {
                    "status": "failed",
                    "error": str(exc),
                },
            )
            append_result(results_path, row)
            existing[run_id] = row
            if args.eval_only:
                print(err)

    print("\n" + "=" * 60)
    print(f"Done. completed={completed}, skipped={skipped}, failed={failed}")
    print(f"Results: {results_path}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
