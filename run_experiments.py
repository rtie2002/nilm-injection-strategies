"""
Run the full NILM experiment grid and append results to CSV.

Examples:
    python run_experiments.py --suite strategy
    python run_experiments.py --suite all --skip-existing --no-plots
    python run_experiments.py --summary-only --suite all
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
    "figure_dir",
    "loss_curve_png",
    "loss_curve_pdf",
    "mae_curve_png",
    "mae_curve_pdf",
    "f1_curve_png",
    "f1_curve_pdf",
    "on_samples_png",
    "on_samples_pdf",
    "timestamp",
    "status",
    "error",
]

APPLIANCE_ORDER = ["washingmachine", "dishwasher", "fridge", "microwave", "kettle"]
APPLIANCE_HEADERS = ["WM", "DW", "Fridge", "MW", "Kettle"]
METRIC_SPECS = [
    ("MAE", "test_mae", ".2f"),
    ("SAE (%)", "test_sae", ".2f"),
    ("F1", "test_f1", ".4f"),
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
    """Load results CSV; keep the latest row per run_id."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    latest: dict[str, dict] = {}
    for row in rows:
        run_id = row.get("run_id")
        if not run_id:
            continue
        prev = latest.get(run_id)
        if prev is None or row.get("timestamp", "") >= prev.get("timestamp", ""):
            latest[run_id] = row
    return latest


def format_metric(value: str | float | int | None, fmt: str) -> str:
    if value is None or value == "":
        return "--"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return "--"


def lookup_cell(
    results: dict[str, dict],
    *,
    model: str,
    appliance: str,
    train_dataset: str,
    metric_key: str,
    fmt: str,
) -> str:
    run_id = make_run_id(model, appliance, train_dataset)
    row = results.get(run_id)
    if row is None or row.get("status") not in {"ok", "eval_only"}:
        return "--"
    return format_metric(row.get(metric_key), fmt)


def _print_text_table(
    title: str,
    row_label_header: str,
    row_specs: list[tuple[str, str]],
    models: list[str],
    cell_fn,
) -> None:
    col_w = 10
    label_w = max(22, len(row_label_header) + 2)

    print()
    print(title)
    print("-" * (label_w + col_w * len(APPLIANCE_HEADERS)))

    header = f"{row_label_header:<{label_w}}" + "".join(f"{h:>{col_w}}" for h in APPLIANCE_HEADERS)
    print(header)
    print("-" * (label_w + col_w * len(APPLIANCE_HEADERS)))

    for model in models:
        model_title = model.upper()
        print(f"[{model_title}]")
        for row_label, row_key in row_specs:
            cells = []
            for appliance in APPLIANCE_ORDER:
                cells.append(cell_fn(model, appliance, row_key))
            line = f"  {row_label:<{label_w - 2}}" + "".join(f"{c:>{col_w}}" for c in cells)
            print(line)
        print()


def print_strategy_summary(exp_cfg: dict, results: dict[str, dict]) -> None:
    models = exp_cfg.get("models", ["cnn"])
    runs = exp_cfg["strategy_comparison"]
    row_specs = [(r["paper_strategy"], r["label"]) for r in runs]

    print("\n" + "=" * 72)
    print("STRATEGY COMPARISON (test set)")
    print("=" * 72)

    for metric_name, metric_key, fmt in METRIC_SPECS:
        def cell_fn_metric(model: str, appliance: str, strategy_label: str) -> str:
            run = next(r for r in runs if r["label"] == strategy_label)
            return lookup_cell(
                results,
                model=model,
                appliance=appliance,
                train_dataset=run["train_dataset"],
                metric_key=metric_key,
                fmt=fmt,
            )

        _print_text_table(
            f"Metric: {metric_name}",
            "Strategy",
            row_specs,
            models,
            cell_fn_metric,
        )


def print_ratio_summary(exp_cfg: dict, results: dict[str, dict]) -> None:
    models = exp_cfg.get("models", ["cnn"])
    runs = exp_cfg["ratio_sensitivity"]
    row_specs = [(f"{r['rho']}%", r["label"]) for r in runs]

    print("\n" + "=" * 72)
    print("RATIO SENSITIVITY — D3 BALANCED (test set)")
    print("=" * 72)

    for metric_name, metric_key, fmt in METRIC_SPECS:
        def cell_fn_metric(model: str, appliance: str, ratio_label: str) -> str:
            run = next(r for r in runs if r["label"] == ratio_label)
            return lookup_cell(
                results,
                model=model,
                appliance=appliance,
                train_dataset=run["train_dataset"],
                metric_key=metric_key,
                fmt=fmt,
            )

        _print_text_table(
            f"Metric: {metric_name}",
            "rho",
            row_specs,
            models,
            cell_fn_metric,
        )


def print_results_summary(exp_cfg: dict, results_path: Path, suite: str) -> None:
    results = load_results_csv(results_path)
    if not results:
        print("\nNo results in CSV yet.")
        return

    ok = sum(1 for r in results.values() if r.get("status") in {"ok", "eval_only"})
    print(f"\nResults loaded: {ok}/{len(results)} runs with metrics")

    if suite in ("strategy", "all"):
        print_strategy_summary(exp_cfg, results)
    if suite in ("ratio", "all"):
        print_ratio_summary(exp_cfg, results)


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
        "figure_dir": metrics.get("figure_dir", ""),
        "loss_curve_png": metrics.get("loss_curve_png", ""),
        "loss_curve_pdf": metrics.get("loss_curve_pdf", ""),
        "mae_curve_png": metrics.get("mae_curve_png", ""),
        "mae_curve_pdf": metrics.get("mae_curve_pdf", ""),
        "f1_curve_png": metrics.get("f1_curve_png", ""),
        "f1_curve_pdf": metrics.get("f1_curve_pdf", ""),
        "on_samples_png": metrics.get("on_samples_png", ""),
        "on_samples_pdf": metrics.get("on_samples_pdf", ""),
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
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip training; print tables from existing CSV",
    )
    args = parser.parse_args()

    exp_cfg = load_experiments(Path(args.experiments))
    results_path = CODE_DIR / exp_cfg["results"]["dir"] / exp_cfg["results"]["csv"]
    existing = load_results_csv(results_path)

    appliances = args.appliance or None
    models = args.model or None

    if args.summary_only:
        print_results_summary(exp_cfg, results_path, args.suite)
        return 0

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
                f"saved -> test MAE {format_metric(row['test_mae'], '.2f')} W | "
                f"SAE {format_metric(row['test_sae'], '.2f')}% | "
                f"F1 {format_metric(row['test_f1'], '.4f')}"
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
    print(f"Results CSV: {results_path}")
    print("=" * 60)
    print_results_summary(exp_cfg, results_path, args.suite)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
