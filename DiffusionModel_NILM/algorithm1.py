"""
Algorithm 1 — Data Cleaning and Selection for Appliance Power Data (univariate).

Paper steps 1–10 (Geng et al., Energy 2025): filter ON excerpts in watts.
Default output: **real power (watts)** — CustomDataset in the diffusion repo
applies MinMaxScaler + [-1, 1] on load (see Utils/Data_utils/real_datasets.py).

Use --paper_minmax to also run paper steps 11–12 before saving (usually redundant).

Input:  {appliance}_training_.csv (from build_geng_mix.py --splits-only, or legacy prepare_all)
Output: Data/datasets/{appliance}.csv  (header: power, values in watts by default)

Usage:
  python algorithm1.py
  python algorithm1.py --paper_minmax   # paper-exact MinMax in CSV
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REL_TRAIN_ROOT = "NILM-main/dataset_preprocess/created_data/UK_DALE"
REL_OUTPUT_DIR = "Data/datasets"

L_WINDOW = 100
X_NOISE = 0

ALL_APPLIANCES = ("kettle", "microwave", "fridge", "dishwasher", "washingmachine")

APPLIANCE_PARAMS = {
    "kettle": {"on_power_threshold": 200, "mean": 700, "std": 1000},
    "microwave": {"on_power_threshold": 200, "mean": 500, "std": 800},
    "fridge": {"on_power_threshold": 50, "mean": 200, "std": 400},
    "dishwasher": {"on_power_threshold": 10, "mean": 700, "std": 1000},
    "washingmachine": {"on_power_threshold": 20, "mean": 400, "std": 700},
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return p.resolve()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


def find_training_csv(train_root: Path, appliance: str) -> Path:
    candidates = [
        train_root / appliance / f"{appliance}_training_.csv",
        train_root / f"{appliance}_training_.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    tried = "\n".join(f"    - {_rel(p)}" for p in candidates)
    raise FileNotFoundError(
        f"Training CSV not found for '{appliance}'. Tried:\n{tried}\n"
        f"  Run prepare_all_ukdale.py first."
    )


def algorithm1_select(
    power_watts: np.ndarray,
    x_threshold: float,
    l_window: int = L_WINDOW,
    x_noise: float = X_NOISE,
) -> np.ndarray:
    """Paper Algorithm 1 steps 2–10: return selected power in watts."""
    x = np.asarray(power_watts, dtype=np.float64).copy()
    x[x < x_noise] = 0
    t_start = np.where(x >= x_threshold)[0]

    t_selected: list[int] = []
    for index in t_start:
        start = max(0, index - l_window)
        end = min(len(x), index + l_window + 1)
        t_selected.extend(range(start, end))

    t_selected = sorted(set(t_selected))
    if len(t_selected) == 0:
        raise ValueError("Algorithm 1 selected zero timesteps — check threshold or input data.")
    return x[t_selected]


def load_training_power_zscore(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Expected 2 columns (aggregate, appliance), got {df.shape[1]} in {csv_path}")
    return df.iloc[:, 1].to_numpy(dtype=np.float64)


def process_appliance(
    appliance: str,
    train_root: Path,
    output_dir: Path,
    l_window: int,
    x_noise: float,
    paper_minmax: bool,
    input_file: Path | None,
    dry_run: bool,
) -> dict:
    params = APPLIANCE_PARAMS[appliance]
    x_threshold = params["on_power_threshold"]
    mean, std = params["mean"], params["std"]

    src = input_file if input_file is not None else find_training_csv(train_root, appliance)
    out_path = output_dir / f"{appliance}.csv"

    power_z = load_training_power_zscore(src)
    power_watts = power_z * std + mean
    selected_watts = algorithm1_select(
        power_watts,
        x_threshold=x_threshold,
        l_window=l_window,
        x_noise=x_noise,
    )

    if paper_minmax:
        scaler = MinMaxScaler()
        output_values = scaler.fit_transform(selected_watts.reshape(-1, 1)).ravel()
        output_format = "minmax_0_1"
    else:
        output_values = selected_watts
        output_format = "watts"

    manifest = {
        "appliance": appliance,
        "input": _rel(src),
        "output": _rel(out_path),
        "output_format": output_format,
        "config_data_root": f"./Data/datasets/{appliance}.csv",
        "input_rows": int(len(power_z)),
        "selected_rows": int(len(output_values)),
        "retention_pct": round(100.0 * len(output_values) / len(power_z), 2),
        "power_w_min": float(output_values.min()),
        "power_w_max": float(output_values.max()),
        "l_window": l_window,
        "x_threshold_w": x_threshold,
        "x_noise": x_noise,
    }

    if dry_run:
        print(
            f"  [dry-run] {appliance} ({output_format}): "
            f"{manifest['input_rows']:,} → {manifest['selected_rows']:,} rows → {_rel(out_path)}"
        )
        return manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"power": output_values}).to_csv(out_path, index=False)
    print(
        f"  {appliance} [{output_format}]: {manifest['input_rows']:,} → "
        f"{manifest['selected_rows']:,} rows ({manifest['retention_pct']}% kept)\n"
        f"           range: [{manifest['power_w_min']:.1f}, {manifest['power_w_max']:.1f}]\n"
        f"           in:  {_rel(src)}\n"
        f"           out: {_rel(out_path)}"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Algorithm 1 → Data/datasets/ for diffusion")
    p.add_argument("--appliance_name", type=str, choices=ALL_APPLIANCES, default=None)
    p.add_argument("--train_root", type=str, default=REL_TRAIN_ROOT)
    p.add_argument("--output_dir", type=str, default=REL_OUTPUT_DIR)
    p.add_argument("--input_file", type=str, default=None)
    p.add_argument("--window", type=int, default=L_WINDOW)
    p.add_argument("--x_noise", type=float, default=X_NOISE)
    p.add_argument(
        "--paper_minmax",
        action="store_true",
        help="Apply paper steps 11–12 (MinMax) before save; default is watts only",
    )
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    appliances = [args.appliance_name] if args.appliance_name else list(ALL_APPLIANCES)
    train_root = _resolve(args.train_root)
    output_dir = _resolve(args.output_dir)
    input_override = _resolve(args.input_file) if args.input_file else None

    if input_override and len(appliances) != 1:
        raise SystemExit("--input_file requires a single --appliance_name")
    if not train_root.is_dir():
        raise FileNotFoundError(f"train_root not found: {_rel(train_root)}")

    out_fmt = "minmax [0,1]" if args.paper_minmax else "watts (CustomDataset MinMax on load)"
    print("Algorithm 1 → diffusion training datasets")
    print(f"  output format: {out_fmt}")
    print(f"  train_root:    {_rel(train_root)}")
    print(f"  output_dir:    {_rel(output_dir)}")
    print(f"  l_window:      {args.window}")
    print(f"  appliances:    {appliances}\n")

    manifests = []
    for app in appliances:
        manifests.append(
            process_appliance(
                appliance=app,
                train_root=train_root,
                output_dir=output_dir,
                l_window=args.window,
                x_noise=args.x_noise,
                paper_minmax=args.paper_minmax,
                input_file=input_override,
                dry_run=args.dry_run,
            )
        )

    if not args.dry_run:
        manifest_path = output_dir / "algorithm1_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "l_window": args.window,
                    "output_format": "minmax_0_1" if args.paper_minmax else "watts",
                    "note": "CustomDataset fits MinMaxScaler on CSV load; neg_one_to_one maps to [-1,1]",
                    "appliances": manifests,
                },
                f,
                indent=2,
            )
        print(f"\nManifest: {_rel(manifest_path)}")
        print("\nDiffusion train:")
        for app in appliances:
            print(f"  python main.py --name {app} --config Config/{app}.yaml --gpu 0 --train")

    print("\nDone.")


if __name__ == "__main__":
    main()
