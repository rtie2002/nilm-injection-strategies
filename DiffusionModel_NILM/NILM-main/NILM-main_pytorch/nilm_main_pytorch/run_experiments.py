"""
Batch train / test for Geng paper tables — PyTorch version.

  python -m nilm_main_pytorch.run_experiments --suite table8 --phase train
  python -m nilm_main_pytorch.run_experiments --suite easy_s2s --phase test --test-house 2 --ewma
  python -m nilm_main_pytorch.run_experiments --suite all --phase train --appliance kettle
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from nilm_main_pytorch.models.params import ALL_APPLIANCES
from nilm_main_pytorch.test import test_one
from nilm_main_pytorch.train import train_one
from nilm_main_pytorch.utils import load_config, merge_cli_config, results_path, save_json

SUITES = {
  # Tables 8–9: S2P, FCN, AugLPN × origin + augmented
    "table8": {
        "models": ("s2p", "fcn", "auglpn"),
        "augmented_flags": (False, True),
    },
    # Tables 5–7: EasyS2S × origin + augmented (extend with more mix ratios later)
    "easy_s2s": {
        "models": ("easy_s2s",),
        "augmented_flags": (False, True),
    },
    "all": {
        "models": ("easy_s2s", "s2p", "fcn", "auglpn"),
        "augmented_flags": (False, True),
    },
}


def run_suite(
    cfg: dict,
    *,
    suite: str,
    phase: str,
    appliances: tuple[str, ...],
    test_house: int,
    use_ewma: bool | None,
    train_percent: str,
) -> list[dict]:
    spec = SUITES[suite]
    results: list[dict] = []

    for model in spec["models"]:
        for augmented in spec["augmented_flags"]:
            for appliance in appliances:
                run_cfg = merge_cli_config(
                    cfg,
                    model=model,
                    appliance=appliance,
                    augmented=augmented,
                    train_percent=train_percent,
                    data_root=None,
                    epochs=None,
                )
                label = f"{model}/{appliance}/{'aug' if augmented else 'origin'}"
                print(f"\n=== {phase.upper()} {label} ===")
                try:
                    if phase == "train":
                        row = train_one(run_cfg)
                    elif phase == "test":
                        row = test_one(
                            run_cfg,
                            test_house=test_house,
                            use_ewma=use_ewma,
                        )
                    else:
                        row = train_one(run_cfg)
                        row["train_status"] = row.get("status")
                        test_row = test_one(
                            run_cfg,
                            test_house=test_house,
                            use_ewma=use_ewma,
                        )
                        row.update({f"test_{k}": v for k, v in test_row.items() if k != "status"})
                        row["status"] = "train_test"
                except FileNotFoundError as exc:
                    print(f"SKIP {label}: {exc}")
                    row = {
                        "model": model,
                        "appliance": appliance,
                        "augmented": augmented,
                        "status": "skipped",
                        "error": str(exc),
                    }
                except Exception as exc:
                    print(f"FAIL {label}: {exc}")
                    row = {
                        "model": model,
                        "appliance": appliance,
                        "augmented": augmented,
                        "status": "failed",
                        "error": str(exc),
                    }
                results.append(row)

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch Geng NILM experiments (PyTorch)")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--suite", choices=list(SUITES.keys()), default="table8")
    p.add_argument(
        "--phase",
        choices=["train", "test", "train_test"],
        default="train",
    )
    p.add_argument("--appliance", choices=list(ALL_APPLIANCES), default=None)
    p.add_argument("--train-percent", default="20")
    p.add_argument("--test-house", type=int, choices=[1, 2], default=2)
    p.add_argument("--ewma", action="store_true")
    p.add_argument("--no-ewma", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    appliances = (args.appliance,) if args.appliance else ALL_APPLIANCES

    use_ewma = None
    if args.ewma:
        use_ewma = True
    if args.no_ewma:
        use_ewma = False

    results = run_suite(
        cfg,
        suite=args.suite,
        phase=args.phase,
        appliances=appliances,
        test_house=args.test_house,
        use_ewma=use_ewma,
        train_percent=args.train_percent,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = results_path(cfg, f"{args.suite}_{args.phase}_h{args.test_house}_{stamp}.json")
    save_json(out, {"suite": args.suite, "phase": args.phase, "runs": results})
    print(f"\nWrote {out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
