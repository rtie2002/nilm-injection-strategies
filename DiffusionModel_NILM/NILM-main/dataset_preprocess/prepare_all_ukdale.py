"""
Prepare all UK-DALE CSV scenarios needed for Geng et al. (Energy 2025) reproduction.

Derived from ukdale_processing.py. Automates:

  House 2 (train / val / test, paper 6:2:2 chronological split):
    - {appliance}_training_.csv       full training pool (60% of timeline)
    - {appliance}_validation_.csv     validation (20%)
    - {appliance}_test_.csv             origin-household test (20%) → Tables 5, 8
    - {appliance}_10training_.csv       first 100k train rows → 100k mix experiments
    - {appliance}_20training_.csv       first 200k train rows → Origin(200k), Tables 8–9 baseline

  House 1 (cross-household test only):
    - {appliance}_test_home1Small_.csv  full house-1 timeline → Tables 6, 9

Outputs are written under (relative to this script):
  created_data/UK_DALE/{appliance}/

Raw .dat layout expected (relative to this script):
  UK_DALE/house_1/channel_1.dat  (mains)
  UK_DALE/house_2/channel_1.dat  (mains)
  ... appliance channels per PARAMS below

Usage (can run from any working directory):
  python prepare_all_ukdale.py
  python path/to/prepare_all_ukdale.py
  python prepare_all_ukdale.py --appliances kettle microwave
  python prepare_all_ukdale.py --data_dir UK_DALE --dry_run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Paths relative to this script file (portable across machines / cwd).
SCRIPT_DIR = Path(__file__).resolve().parent
REL_DATA_DIR = "UK_DALE"
REL_SAVE_DIR = "created_data/UK_DALE"

# ---------------------------------------------------------------------------
# Constants (from ukdale_processing.py / paper Table 1)
# ---------------------------------------------------------------------------

AGG_MEAN = 522
AGG_STD = 814
SAMPLE_SECONDS = 6
DEFAULT_VAL_PCT = 20
DEFAULT_TEST_PCT = 20
TRAIN_CROP_SIZES = (100_000, 200_000)  # paper: 10^5 and 2×10^5

ALL_APPLIANCES = (
    "kettle",
    "microwave",
    "fridge",
    "dishwasher",
    "washingmachine",
)

# channel_map: {house_id: appliance_channel}; mains is always channel 1
PARAMS = {
    "kettle": {
        "mean": 700,
        "std": 1000,
        "channel_map": {1: 10, 2: 8},
    },
    "microwave": {
        "mean": 500,
        "std": 800,
        "channel_map": {1: 13, 2: 15},
    },
    "fridge": {
        "mean": 200,
        "std": 400,
        "channel_map": {1: 12, 2: 14},
    },
    "dishwasher": {
        "mean": 700,
        "std": 1000,
        "channel_map": {1: 6, 2: 13},
    },
    "washingmachine": {
        "mean": 400,
        "std": 700,
        "channel_map": {1: 5, 2: 12},
    },
}


def _resolve_dir(path: str | Path) -> Path:
    """Relative paths are resolved from SCRIPT_DIR, not the shell cwd."""
    p = Path(path)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return p.resolve()


def _rel_to_script(path: Path) -> str:
    """Display path relative to script dir when possible."""
    try:
        return str(path.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


def load_dat_channel(data_dir: Path, house: int, channel: int) -> pd.DataFrame:
    path = data_dir / f"house_{house}" / f"channel_{channel}.dat"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing UK-DALE file: {_rel_to_script(path)}\n"
            f"  Expected: {{data_dir}}/house_{house}/channel_{channel}.dat\n"
            f"  data_dir: {_rel_to_script(data_dir)}\n"
            f"  Put .dat files next to this script under: {REL_DATA_DIR}/house_{house}/"
        )
    return pd.read_table(
        path,
        sep=r"\s+",
        usecols=[0, 1],
        names=["time", "value"],
        dtype={"time": str},
    )


def align_and_resample(
    data_dir: Path,
    house: int,
    appliance: str,
    sample_seconds: int = SAMPLE_SECONDS,
) -> pd.DataFrame:
    """Load mains + appliance, align timestamps, resample to 6 s (paper 1/6 Hz)."""
    app_channel = PARAMS[appliance]["channel_map"][house]
    mains_df = load_dat_channel(data_dir, house, 1).rename(columns={"value": "aggregate"})
    app_df = load_dat_channel(data_dir, house, app_channel).rename(
        columns={"value": appliance}
    )

    for df in (mains_df, app_df):
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

    df_align = (
        mains_df.join(app_df, how="outer")
        .resample(f"{sample_seconds}s")
        .mean()
        .bfill(limit=1)
        .dropna()
    )

    if appliance == "fridge":
        df_align[appliance] = np.minimum(
            df_align["aggregate"].values, df_align[appliance].values
        )

    return df_align.reset_index(drop=True)


def zscore_normalize(
    df: pd.DataFrame,
    appliance: str,
    aggregate_mean: float = AGG_MEAN,
    aggregate_std: float = AGG_STD,
) -> pd.DataFrame:
    out = df[["aggregate", appliance]].copy()
    app_mean = PARAMS[appliance]["mean"]
    app_std = PARAMS[appliance]["std"]
    out["aggregate"] = (out["aggregate"] - aggregate_mean) / aggregate_std
    out[appliance] = (out[appliance] - app_mean) / app_std
    return out


def split_chronological(
    df: pd.DataFrame,
    validation_percent: float = DEFAULT_VAL_PCT,
    test_percent: float = DEFAULT_TEST_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Paper 6:2:2 — last test_percent = test, previous validation_percent = val, rest = train."""
    n = len(df)
    test_len = int(n * test_percent / 100)
    val_len = int(n * validation_percent / 100)

    test = df.iloc[-test_len:].reset_index(drop=True)
    remaining = df.iloc[:-test_len]
    val = remaining.iloc[-val_len:].reset_index(drop=True)
    train = remaining.iloc[:-val_len].reset_index(drop=True)
    return train, val, test


def save_csv(df: pd.DataFrame, path: Path, dry_run: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"  [dry-run] would write {path} ({len(df):,} rows)")
        return
    df.to_csv(path, index=False, header=False)


def crop_label(n_rows: int) -> str:
    """EasyS2S TrainPercent label: 100k → '10', 200k → '20'."""
    return str(n_rows // 10_000)


def process_appliance(
    appliance: str,
    data_dir: Path,
    save_root: Path,
    validation_percent: float,
    test_percent: float,
    train_crops: tuple[int, ...],
    skip_house1: bool,
    aggregate_mean: float,
    aggregate_std: float,
    dry_run: bool,
) -> dict:
    out_dir = Path(save_root) / appliance
    manifest: dict = {"appliance": appliance, "outputs": {}}

    # --- House 2: train / val / test (6:2:2) --------------------------------
    print(f"\n[{appliance}] House 2 — align, z-score, 6:2:2 split")
    raw_h2 = align_and_resample(data_dir, house=2, appliance=appliance)
    norm_h2 = zscore_normalize(raw_h2, appliance, aggregate_mean, aggregate_std)
    train, val, test = split_chronological(norm_h2, validation_percent, test_percent)

    files_h2 = {
        f"{appliance}_training_.csv": train,
        f"{appliance}_validation_.csv": val,
        f"{appliance}_test_.csv": test,
    }
    for name, frame in files_h2.items():
        save_csv(frame, out_dir / name, dry_run=dry_run)
        manifest["outputs"][name] = len(frame)

    for crop_n in train_crops:
        label = crop_label(crop_n)
        crop_name = f"{appliance}_{label}training_.csv"
        cropped = train.iloc[: min(crop_n, len(train))].reset_index(drop=True)
        save_csv(cropped, out_dir / crop_name, dry_run=dry_run)
        manifest["outputs"][crop_name] = len(cropped)
        if len(cropped) < crop_n:
            print(
                f"  warning: train pool has {len(train):,} rows, "
                f"requested {crop_n:,} for {crop_name}"
            )

    # --- House 1: cross-household test (full timeline, no 6:2:2 split) -------
    if not skip_house1:
        print(f"[{appliance}] House 1 — cross-household test")
        raw_h1 = align_and_resample(data_dir, house=1, appliance=appliance)
        norm_h1 = zscore_normalize(raw_h1, appliance, aggregate_mean, aggregate_std)
        h1_name = f"{appliance}_test_home1Small_.csv"
        save_csv(norm_h1, out_dir / h1_name, dry_run=dry_run)
        manifest["outputs"][h1_name] = len(norm_h1)

    manifest["house2_timeline_rows"] = len(norm_h2)
    manifest["split_ratio"] = f"{100 - validation_percent - test_percent}:{validation_percent}:{test_percent}"
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare all UK-DALE NILM CSVs for Geng et al. reproduction"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=REL_DATA_DIR,
        help=f"Input .dat root, relative to script dir (default: {REL_DATA_DIR})",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=REL_SAVE_DIR,
        help=f"Output CSV root, relative to script dir (default: {REL_SAVE_DIR})",
    )
    parser.add_argument(
        "--appliances",
        nargs="+",
        default=list(ALL_APPLIANCES),
        choices=ALL_APPLIANCES,
        help="Appliances to process (default: all five)",
    )
    parser.add_argument(
        "--validation_percent",
        type=float,
        default=DEFAULT_VAL_PCT,
        help="Validation fraction (default 20, paper 6:2:2)",
    )
    parser.add_argument(
        "--testing_percent",
        type=float,
        default=DEFAULT_TEST_PCT,
        help="Test fraction (default 20, paper 6:2:2)",
    )
    parser.add_argument(
        "--train_crops",
        nargs="+",
        type=int,
        default=list(TRAIN_CROP_SIZES),
        help="Crop sizes from start of train pool (default: 100000 200000)",
    )
    parser.add_argument(
        "--aggregate_mean",
        type=float,
        default=AGG_MEAN,
        help="Z-score mean for mains (default 522)",
    )
    parser.add_argument(
        "--aggregate_std",
        type=float,
        default=AGG_STD,
        help="Z-score std for mains (default 814)",
    )
    parser.add_argument(
        "--skip_house1",
        action="store_true",
        help="Skip House 1 cross-household test CSVs",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned outputs without writing files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = _resolve_dir(args.data_dir)
    save_path = _resolve_dir(args.save_path)

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"data_dir not found: {_rel_to_script(data_dir)}\n"
            f"  Put UK-DALE .dat files under: {REL_DATA_DIR}/house_2/channel_1.dat\n"
            f"  (paths are relative to script: {_rel_to_script(SCRIPT_DIR)})"
        )

    if args.validation_percent + args.testing_percent >= 100:
        raise ValueError("validation_percent + testing_percent must be < 100")

    t0 = time.time()
    print("UK-DALE batch preparation")
    print(f"  script dir: {_rel_to_script(SCRIPT_DIR)}")
    print(f"  data_dir:   {_rel_to_script(data_dir)}")
    print(f"  save_path:  {_rel_to_script(save_path)}")
    print(f"  appliances: {args.appliances}")
    print(
        f"  split: {100 - args.validation_percent - args.testing_percent}:"
        f"{args.validation_percent}:{args.testing_percent}"
    )

    all_manifests = []
    for appliance in args.appliances:
        manifest = process_appliance(
            appliance=appliance,
            data_dir=data_dir,
            save_root=save_path,
            validation_percent=args.validation_percent,
            test_percent=args.testing_percent,
            train_crops=tuple(args.train_crops),
            skip_house1=args.skip_house1,
            aggregate_mean=args.aggregate_mean,
            aggregate_std=args.aggregate_std,
            dry_run=args.dry_run,
        )
        all_manifests.append(manifest)

    manifest_path = save_path / "preprocessing_manifest.json"
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "paper": "Geng et al. Energy 2025",
                    "script_dir": _rel_to_script(SCRIPT_DIR),
                    "data_dir": _rel_to_script(data_dir),
                    "save_path": _rel_to_script(save_path),
                    "sample_seconds": SAMPLE_SECONDS,
                    "aggregate_mean": args.aggregate_mean,
                    "aggregate_std": args.aggregate_std,
                    "appliances": all_manifests,
                },
                f,
                indent=2,
            )
        print(f"\nManifest: {_rel_to_script(manifest_path)}")

    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.2f} min.")
    print("\nNext steps (not done by this script):")
    print("  1. Algorithm 1 → Data/datasets/{appliance}.csv (diffusion train)")
    print("  2. Sample synthetic watts + sum-of-5 aggregate")
    print(
        "  3. build_geng_mix.py → file10, file20, file10_20, file20_10 "
        "(or: python build_geng_mix.py --scenario missing)"
    )


if __name__ == "__main__":
    main()
