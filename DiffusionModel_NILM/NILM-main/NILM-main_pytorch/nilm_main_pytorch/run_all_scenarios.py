"""
Run all Geng paper scenarios × all appliances — train, test, save metrics, print tables.

EasyS2S (Tables 5–7) — 6 data scenarios × 5 appliances:
  python -m nilm_main_pytorch.run_all_scenarios
  python -m nilm_main_pytorch.run_all_scenarios --phase test

Table 8–9 models (S2P, FCN, AugLPN) — origin 200k + augmented 200k+200k:
  python -m nilm_main_pytorch.run_all_scenarios --suite table8 --phase train_test

Full grid (all models, all scenarios):
  python -m nilm_main_pytorch.run_all_scenarios --suite all --phase train_test
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly (IDE Run / full path) without `python -m ...`
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
    merge_cli_config,
    portable_path_str,
    results_path,
    save_json,
)


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    label: str
    model: str
    train_percent: str
    augmented: bool


# Tables 5–7: EasyS2S × all mix conditions (REPRODUCTION_A.md)
EASY_S2S_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("origin_100k", "Origin 100k", "easy_s2s", "10", False),
    ScenarioSpec("origin_200k", "Origin 200k", "easy_s2s", "20", False),
    ScenarioSpec("mix_100k_100k", "100k+100k", "easy_s2s", "10", True),
    ScenarioSpec("mix_200k_200k", "200k+200k", "easy_s2s", "20", True),
    ScenarioSpec("mix_100k_200k", "100k+200k", "easy_s2s", "10_20", True),
    ScenarioSpec("mix_200k_100k", "200k+100k", "easy_s2s", "20_10", True),
)

# Tables 8–9: S2P, FCN, AugLPN — origin 200k + augmented 200k+200k
TABLE8_SCENARIOS: tuple[ScenarioSpec, ...] = tuple(
    ScenarioSpec(
        f"{'aug' if augmented else 'origin'}_200k_{model}",
        f"{'Aug 200k+200k' if augmented else 'Origin 200k'} ({model})",
        model,
        "20",
        augmented,
    )
    for model in ("s2p", "fcn", "auglpn")
    for augmented in (False, True)
)

SUITES: dict[str, tuple[ScenarioSpec, ...]] = {
    "easy_s2s": EASY_S2S_SCENARIOS,
    "table8": TABLE8_SCENARIOS,
    "all": EASY_S2S_SCENARIOS + TABLE8_SCENARIOS,
}


def _scenario_key(row: dict) -> str:
    return str(row.get("scenario_id", row.get("scenario", "")))


def run_all(
    cfg: dict,
    *,
    suite: str,
    phase: str,
    appliances: tuple[str, ...],
    test_house: int,
    use_ewma: bool | None,
    skip_errors: bool,
) -> list[dict]:
    scenarios = SUITES[suite]
    results: list[dict] = []
    total = len(scenarios) * len(appliances)
    done = 0

    for spec in scenarios:
        for appliance in appliances:
            done += 1
            run_cfg = merge_cli_config(
                cfg,
                model=spec.model,
                appliance=appliance,
                augmented=spec.augmented,
                train_percent=spec.train_percent,
                data_root=None,
                epochs=None,
            )
            tag = (
                f"[{done}/{total}] {spec.model} | {appliance} | "
                f"{spec.label} | pct={spec.train_percent} | "
                f"{'aug' if spec.augmented else 'origin'}"
            )
            print(f"\n{'=' * 72}\n{tag}\n{'=' * 72}")

            row: dict = {
                "scenario_id": spec.scenario_id,
                "scenario_label": spec.label,
                "model": spec.model,
                "appliance": appliance,
                "train_percent": spec.train_percent,
                "augmented": spec.augmented,
                "test_house": test_house,
            }

            try:
                if phase in ("train", "train_test"):
                    train_row = train_one(run_cfg, verbose=True, show_device=False)
                    row.update(
                        {
                            "train_status": train_row.get("status"),
                            "best_epoch": train_row.get("best_epoch"),
                            "val_mae": train_row.get("val_mae"),
                            "val_sae": train_row.get("val_sae"),
                            "val_f1": train_row.get("val_f1"),
                            "checkpoint": train_row.get("checkpoint"),
                            "train_csv": train_row.get("train_csv"),
                        }
                    )

                if phase in ("test", "train_test"):
                    test_row = test_one(
                        run_cfg,
                        test_house=test_house,
                        use_ewma=use_ewma,
                        verbose=True,
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
                else:
                    row["status"] = row.get("train_status", "trained")

            except FileNotFoundError as exc:
                row["status"] = "skipped"
                row["error"] = str(exc)
                print(f"SKIP: {exc}")
                if not skip_errors:
                    results.append(row)
                    raise
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                print(f"FAIL: {exc}")
                if not skip_errors:
                    results.append(row)
                    raise

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
    scenario_labels: list[str],
    scenario_ids: list[str],
) -> str:
    lookup: dict[tuple[str, str], dict] = {}
    for row in results:
        if row.get("status") not in ("ok", "trained", "train_test"):
            continue
        key = (row["appliance"], row["scenario_id"])
        lookup[key] = row

    col_w = max(12, max((len(s) for s in scenario_labels), default=10))
    app_w = max(16, max((len(a) for a in appliances), default=10))

    lines = [title, "-" * (app_w + col_w * len(scenario_ids) + 2)]
    header = f"{'appliance':<{app_w}}" + "".join(f"{lab:>{col_w}}" for lab in scenario_labels)
    lines.append(header)
    lines.append("-" * len(header))

    for app in appliances:
        cells = [f"{app:<{app_w}}"]
        for sid in scenario_ids:
            row = lookup.get((app, sid))
            val = row.get(metric) if row else None
            cells.append(_fmt(val, col_w))
        lines.append("".join(cells))

    return "\n".join(lines)


def format_summary_tables(
    results: list[dict],
    *,
    appliances: tuple[str, ...],
    scenarios: tuple[ScenarioSpec, ...],
) -> str:
    scenario_ids = [s.scenario_id for s in scenarios]
    scenario_labels = [s.label for s in scenarios]

    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    failed = sum(1 for r in results if r.get("status") == "failed")

    parts = [
        "",
        "=" * 72,
        "EXPERIMENT SUMMARY",
        "=" * 72,
        f"Total runs: {len(results)} | OK: {ok} | Skipped: {skipped} | Failed: {failed}",
        "",
        _pivot_table(
            results,
            metric="mae",
            title="MAE (W) — test house 2",
            appliances=appliances,
            scenario_labels=scenario_labels,
            scenario_ids=scenario_ids,
        ),
        "",
        _pivot_table(
            results,
            metric="sae",
            title="SAE (ratio)",
            appliances=appliances,
            scenario_labels=scenario_labels,
            scenario_ids=scenario_ids,
        ),
        "",
        _pivot_table(
            results,
            metric="f1",
            title="F1",
            appliances=appliances,
            scenario_labels=scenario_labels,
            scenario_ids=scenario_ids,
        ),
        "",
        "DETAIL (all runs)",
        "-" * 72,
        f"{'scenario':<16} {'appliance':<16} {'MAE':>8} {'SAE':>8} {'F1':>8} {'status':<8}",
        "-" * 72,
    ]

    for row in sorted(results, key=lambda r: (_scenario_key(r), r.get("appliance", ""))):
        parts.append(
            f"{row.get('scenario_label', _scenario_key(row)):<16} "
            f"{row.get('appliance', ''):<16} "
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
    suite: str,
    phase: str,
    test_house: int,
    results: list[dict],
    scenarios: tuple[ScenarioSpec, ...],
) -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{suite}_{phase}_h{test_house}_{stamp}"

    json_path = results_path(cfg, f"{base}.json")
    csv_path = results_path(cfg, f"{base}.csv")
    txt_path = results_path(cfg, f"{base}_table.txt")

    payload = {
        "suite": suite,
        "phase": phase,
        "test_house": test_house,
        "scenarios": [
            {
                "id": s.scenario_id,
                "label": s.label,
                "model": s.model,
                "train_percent": s.train_percent,
                "augmented": s.augmented,
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
        "train_percent",
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
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    appliances_in_run = tuple(dict.fromkeys(r["appliance"] for r in results))
    table_text = format_summary_tables(
        results,
        appliances=appliances_in_run if appliances_in_run else ALL_APPLIANCES,
        scenarios=scenarios,
    )
    txt_path.write_text(table_text, encoding="utf-8")

    return json_path, csv_path, txt_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all Geng scenarios × all appliances; save metrics and print tables",
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--suite", choices=list(SUITES.keys()), default="easy_s2s")
    p.add_argument(
        "--phase",
        choices=["train", "test", "train_test"],
        default="train_test",
        help="train_test = train each run then evaluate on test house (default)",
    )
    p.add_argument("--appliance", choices=list(ALL_APPLIANCES), default=None)
    p.add_argument("--test-house", type=int, choices=[1, 2], default=2)
    p.add_argument("--ewma", action="store_true", help="Force EWMA on augmented runs")
    p.add_argument("--no-ewma", action="store_true")
    p.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue if a run fails or data/checkpoint is missing",
    )
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
    appliances = (args.appliance,) if args.appliance else ALL_APPLIANCES
    scenarios = SUITES[args.suite]

    use_ewma = None
    if args.ewma:
        use_ewma = True
    if args.no_ewma:
        use_ewma = False

    print(f"Suite: {args.suite} | Phase: {args.phase} | Appliances: {', '.join(appliances)}")
    print(f"Data root: {portable_path_str(data_root_path(cfg))}")
    log_device(cfg)
    print(f"Scenarios ({len(scenarios)}):")
    for s in scenarios:
        print(f"  - {s.label} ({s.model}, pct={s.train_percent}, aug={s.augmented})")

    results = run_all(
        cfg,
        suite=args.suite,
        phase=args.phase,
        appliances=appliances,
        test_house=args.test_house,
        use_ewma=use_ewma,
        skip_errors=args.skip_errors,
    )

    appliances_in_run = tuple(dict.fromkeys(r["appliance"] for r in results))
    summary = format_summary_tables(
        results,
        appliances=appliances_in_run if appliances_in_run else appliances,
        scenarios=scenarios,
    )
    print(summary)

    json_path, csv_path, txt_path = save_results_bundle(
        cfg,
        suite=args.suite,
        phase=args.phase,
        test_house=args.test_house,
        results=results,
        scenarios=scenarios,
    )
    print(f"\nSaved:\n  {json_path}\n  {csv_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
