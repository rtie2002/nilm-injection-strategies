"""
Run Geng-style injection-ratio experiment — train + test all rho × appliances.

Prerequisites:
  cd DiffusionModel_NILM
  python build_geng_rho_datasets.py

Then:
  cd NILM-main/NILM-main_pytorch
  python -m nilm_main_pytorch.run_injection_rho_scenarios
  python -m nilm_main_pytorch.run_injection_rho_scenarios --phase test --rho 0 100
  python -m nilm_main_pytorch.run_injection_rho_scenarios --model easy_s2s --no-ewma
"""

from __future__ import annotations

import sys
from pathlib import Path

_PYTORCH_ROOT = Path(__file__).resolve().parent.parent
if str(_PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTORCH_ROOT))

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime

from nilm_main_pytorch.models.params import ALL_APPLIANCES
from nilm_main_pytorch.test import test_one
from nilm_main_pytorch.train import train_one
from nilm_main_pytorch.utils import (
    add_device_cli_args,
    data_root_path,
    device_options_from_args,
    load_config,
    log_device,
    merge_rho_cli_config,
    portable_path_str,
    results_path,
    save_json,
)

DEFAULT_RHO_PCTS: tuple[int, ...] = (0, 25, 50, 100, 200)
DEFAULT_N_REAL = 100_000


@dataclass(frozen=True)
class RhoScenarioSpec:
    scenario_id: str
    label: str
    rho_pct: int
    model: str
    n_real: int

    @classmethod
    def from_rho(cls, rho_pct: int, *, model: str, n_real: int) -> RhoScenarioSpec:
        return cls(
            scenario_id=f"rho_{rho_pct}",
            label=f"rho={rho_pct}%",
            rho_pct=int(rho_pct),
            model=model,
            n_real=int(n_real),
        )


def build_scenarios(
    rho_pcts: tuple[int, ...],
    *,
    model: str,
    n_real: int,
) -> tuple[RhoScenarioSpec, ...]:
    return tuple(RhoScenarioSpec.from_rho(r, model=model, n_real=n_real) for r in rho_pcts)


def run_all(
    cfg: dict,
    *,
    scenarios: tuple[RhoScenarioSpec, ...],
    phase: str,
    appliances: tuple[str, ...],
    test_house: int,
    use_ewma: bool | None,
    skip_errors: bool,
    verbose: bool,
) -> list[dict]:
    results: list[dict] = []
    total = len(scenarios) * len(appliances)
    done = 0

    for spec in scenarios:
        for appliance in appliances:
            done += 1
            run_cfg = merge_rho_cli_config(
                cfg,
                model=spec.model,
                appliance=appliance,
                rho_pct=spec.rho_pct,
                n_real=spec.n_real,
            )
            tag = (
                f"[{done}/{total}] {spec.model} | {appliance} | "
                f"{spec.label} | n_real={spec.n_real:,}"
            )
            print(f"\n{'=' * 72}\n{tag}\n{'=' * 72}")

            row: dict = {
                "scenario_id": spec.scenario_id,
                "scenario_label": spec.label,
                "model": spec.model,
                "appliance": appliance,
                "rho_pct": spec.rho_pct,
                "n_real": spec.n_real,
                "n_syn": int(round(spec.rho_pct / 100.0 * spec.n_real)),
                "augmented": spec.rho_pct > 0,
                "test_house": test_house,
            }

            try:
                if phase in ("train", "train_test"):
                    train_row = train_one(run_cfg, verbose=verbose, show_device=False)
                    row.update(
                        {
                            "train_status": train_row.get("status"),
                            "best_epoch": train_row.get("best_epoch"),
                            "val_mae": train_row.get("val_mae"),
                            "val_sae": train_row.get("val_sae"),
                            "val_f1": train_row.get("val_f1"),
                            "checkpoint": train_row.get("checkpoint"),
                            "train_csv": train_row.get("train_csv"),
                            "figure_dir": train_row.get("figure_dir"),
                            "loss_curve_png": train_row.get("loss_curve_png"),
                            "loss_curve_pdf": train_row.get("loss_curve_pdf"),
                            "metric_summary_png": train_row.get("metric_summary_png"),
                            "metric_summary_pdf": train_row.get("metric_summary_pdf"),
                            "on_samples_png": train_row.get("on_samples_png"),
                            "on_samples_pdf": train_row.get("on_samples_pdf"),
                        }
                    )

                if phase in ("test", "train_test"):
                    test_row = test_one(
                        run_cfg,
                        test_house=test_house,
                        use_ewma=use_ewma,
                        verbose=verbose,
                    )
                    row.update(
                        {
                            "mae": test_row.get("mae"),
                            "sae": test_row.get("sae"),
                            "f1": test_row.get("f1"),
                            "recall": test_row.get("recall"),
                            "precision": test_row.get("precision"),
                            "raw_mae": test_row.get("raw_mae"),
                            "postprocess": test_row.get("postprocess"),
                            "inference": test_row.get("inference"),
                            "eval_csv": test_row.get("eval_csv"),
                            "checkpoint": test_row.get("checkpoint", row.get("checkpoint")),
                        }
                    )
                    row["status"] = "ok"
                    if not verbose:
                        print(
                            f"RESULT {spec.model}/{appliance} rho={spec.rho_pct}% | "
                            f"MAE {row['mae']:.2f}W | SAE {row['sae']:.4f} | F1 {row['f1']:.4f}",
                            flush=True,
                        )
                else:
                    row["status"] = row.get("train_status", "trained")

            except FileNotFoundError as exc:
                row["status"] = "skipped"
                row["error"] = str(exc)
                print(f"SKIP: {exc}", flush=True)
                results.append(row)
                if not skip_errors:
                    print(
                        "Hint: build rho CSVs with "
                        "cd DiffusionModel_NILM && python build_geng_rho_datasets.py",
                        flush=True,
                    )
                    raise
                continue
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                print(f"FAIL: {exc}", flush=True)
                results.append(row)
                if not skip_errors:
                    raise
                continue

            results.append(row)

    return results


def _fmt(value: object, width: int = 10) -> str:
    if value is None:
        return f"{'NA':>{width}}"
    if isinstance(value, float):
        if width <= 8:
            return f"{value:>{width}.4f}"
        return f"{value:>{width}.2f}"
    return f"{str(value):>{width}}"[:width].rjust(width)


def _pivot_table(
    results: list[dict],
    *,
    metric: str,
    title: str,
    appliances: tuple[str, ...],
    rho_labels: list[str],
    rho_ids: list[str],
) -> str:
    lookup: dict[tuple[str, str], dict] = {}
    for row in results:
        if row.get("status") not in ("ok", "trained", "train_test"):
            continue
        key = (row["appliance"], row["scenario_id"])
        lookup[key] = row

    col_w = max(8, max((len(s) for s in rho_labels), default=6))
    app_w = max(16, max((len(a) for a in appliances), default=10))

    lines = [title, "-" * (app_w + col_w * len(rho_ids) + 2)]
    header = f"{'appliance':<{app_w}}" + "".join(f"{lab:>{col_w}}" for lab in rho_labels)
    lines.append(header)
    lines.append("-" * len(header))

    for app in appliances:
        cells = [f"{app:<{app_w}}"]
        for sid in rho_ids:
            row = lookup.get((app, sid))
            val = row.get(metric) if row else None
            cells.append(_fmt(val, col_w))
        lines.append("".join(cells))

    return "\n".join(lines)


def format_paper_ratio_tables(
    results: list[dict],
    *,
    appliances: tuple[str, ...],
    scenarios: tuple[RhoScenarioSpec, ...],
    model: str,
) -> str:
    """ICSIMA-style blocks: Metric -> rho columns -> appliances rows."""
    rho_ids = [s.scenario_id for s in scenarios]
    rho_labels = [f"{s.rho_pct}%" for s in scenarios]
    app_short = {
        "washingmachine": "WM",
        "dishwasher": "DW",
        "fridge": "Fridge",
        "microwave": "MW",
        "kettle": "Kettle",
    }

    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    failed = sum(1 for r in results if r.get("status") == "failed")

    parts = [
        "",
        "=" * 72,
        f"INJECTION RATIO SUMMARY ({model.upper()} — Geng concat)",
        "=" * 72,
        f"Total runs: {len(results)} | OK: {ok} | Skipped: {skipped} | Failed: {failed}",
        f"rho = |D_s|/|D_r| | n_real = {scenarios[0].n_real:,}",
        "",
    ]

    for metric, metric_title in (
        ("mae", "MAE (W)"),
        ("sae", "SAE"),
        ("f1", "F1"),
    ):
        parts.append(f"--- {metric_title} ---")
        parts.append(
            _pivot_table(
                results,
                metric=metric,
                title="",
                appliances=appliances,
                rho_labels=rho_labels,
                rho_ids=rho_ids,
            )
        )
        parts.append("")

    parts.extend(
        [
            "DETAIL (all runs)",
            "-" * 72,
            f"{'rho':<8} {'appliance':<16} {'MAE':>8} {'SAE':>8} {'F1':>8} {'status':<8}",
            "-" * 72,
        ]
    )

    for row in sorted(
        results,
        key=lambda r: (r.get("rho_pct", -1), r.get("appliance", "")),
    ):
        app = row.get("appliance", "")
        short = app_short.get(app, app)
        parts.append(
            f"{row.get('rho_pct', ''):<8} "
            f"{short:<16} "
            f"{_fmt(row.get('mae'), 8)} "
            f"{_fmt(row.get('sae'), 8)} "
            f"{_fmt(row.get('f1'), 8)} "
            f"{row.get('status', ''):<8}"
        )

    parts.append("=" * 72)
    return "\n".join(parts)


def save_results_bundle(
    cfg: dict,
    *,
    model: str,
    phase: str,
    test_house: int,
    n_real: int,
    results: list[dict],
    scenarios: tuple[RhoScenarioSpec, ...],
) -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"injection_rho_{model}_{phase}_h{test_house}_n{n_real // 1000}k_{stamp}"

    json_path = results_path(cfg, f"{base}.json")
    csv_path = results_path(cfg, f"{base}.csv")
    txt_path = results_path(cfg, f"{base}_table.txt")

    payload = {
        "experiment": "geng_injection_ratio",
        "model": model,
        "phase": phase,
        "test_house": test_house,
        "n_real": n_real,
        "rho_definition": "|D_s| / |D_r|",
        "scenarios": [
            {
                "id": s.scenario_id,
                "label": s.label,
                "rho_pct": s.rho_pct,
                "n_syn": int(round(s.rho_pct / 100.0 * s.n_real)),
                "model": s.model,
                "n_real": s.n_real,
            }
            for s in scenarios
        ],
        "appliances": list(ALL_APPLIANCES),
        "runs": results,
    }
    save_json(json_path, payload)

    fieldnames = [
        "scenario_id",
        "scenario_label",
        "model",
        "appliance",
        "rho_pct",
        "n_real",
        "n_syn",
        "augmented",
        "test_house",
        "status",
        "mae",
        "sae",
        "f1",
        "recall",
        "precision",
        "raw_mae",
        "postprocess",
        "best_epoch",
        "val_mae",
        "val_sae",
        "val_f1",
        "checkpoint",
        "train_csv",
        "figure_dir",
        "loss_curve_png",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    appliances_in_run = tuple(dict.fromkeys(r["appliance"] for r in results))
    table_text = format_paper_ratio_tables(
        results,
        appliances=appliances_in_run if appliances_in_run else ALL_APPLIANCES,
        scenarios=scenarios,
        model=model,
    )
    txt_path.write_text(table_text, encoding="utf-8")

    return json_path, csv_path, txt_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train + test Geng injection-ratio grid (rho × appliances)",
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model", choices=["easy_s2s", "s2p", "fcn", "auglpn"], default="easy_s2s")
    p.add_argument(
        "--rho",
        type=int,
        nargs="+",
        default=list(DEFAULT_RHO_PCTS),
        help="Injection ratios in percent (default: 0 25 50 100 200)",
    )
    p.add_argument(
        "--n-real",
        type=int,
        default=DEFAULT_N_REAL,
        help="Real block size |D_r| — must match build_geng_rho_datasets.py",
    )
    p.add_argument(
        "--phase",
        choices=["train", "test", "train_test"],
        default="train_test",
    )
    p.add_argument("--appliance", choices=list(ALL_APPLIANCES), default=None)
    p.add_argument("--test-house", type=int, choices=[1, 2], default=2)
    p.add_argument("--ewma", action="store_true", help="Force EWMA when rho > 0")
    p.add_argument("--no-ewma", action="store_true", help="Disable EWMA (recommended for paper)")
    p.add_argument("--skip-errors", action="store_true")
    p.add_argument("--quiet", action="store_true")
    add_device_cli_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dev, gpu_id, require_cuda = device_options_from_args(args)
    cfg = load_config(args.config)
    if dev is not None:
        cfg["training"]["device"] = dev
    if gpu_id is not None:
        cfg["training"]["gpu_id"] = gpu_id
    if require_cuda is not None:
        cfg["training"]["require_cuda"] = require_cuda

    rho_pcts = tuple(sorted(set(int(r) for r in args.rho)))
    appliances = (args.appliance,) if args.appliance else ALL_APPLIANCES
    scenarios = build_scenarios(rho_pcts, model=args.model, n_real=args.n_real)

    use_ewma = None
    if args.ewma:
        use_ewma = True
    if args.no_ewma:
        use_ewma = False

    print(f"Experiment: Geng injection ratio | Model: {args.model}")
    print(f"Phase: {args.phase} | rho: {rho_pcts} | n_real: {args.n_real:,}")
    print(f"Appliances: {', '.join(appliances)}")
    print(f"Data root: {portable_path_str(data_root_path(cfg))}")
    log_device(cfg)
    print("Scenarios:")
    for s in scenarios:
        n_syn = int(round(s.rho_pct / 100.0 * s.n_real))
        print(f"  - {s.label} (real={s.n_real:,}, syn={n_syn:,})")

    results = run_all(
        cfg,
        scenarios=scenarios,
        phase=args.phase,
        appliances=appliances,
        test_house=args.test_house,
        use_ewma=use_ewma,
        skip_errors=args.skip_errors,
        verbose=not args.quiet,
    )

    summary = format_paper_ratio_tables(
        results,
        appliances=appliances,
        scenarios=scenarios,
        model=args.model,
    )
    print(summary)

    json_path, csv_path, txt_path = save_results_bundle(
        cfg,
        model=args.model,
        phase=args.phase,
        test_house=args.test_house,
        n_real=args.n_real,
        results=results,
        scenarios=scenarios,
    )
    print(f"\nSaved:\n  {json_path}\n  {csv_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
