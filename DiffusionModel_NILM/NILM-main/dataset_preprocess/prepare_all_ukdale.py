"""
Prepare UK-DALE CSVs for Geng NILM / diffusion pipelines.

Modes
-----
**Paper-exact** (Geng et al. 2025 reproduction) — recommended for matching Tables 5–9:

  python prepare_all_ukdale.py --paper-exact

  Full house-2 timeline → one 6:2:2 split → fixed val/test + train crops:
    {app}_training_.csv          train pool (60% of full timeline)
    {app}_validation_.csv        validation (20%, same for all experiments)
    {app}_test_.csv              test house 2 (20%)
    {app}_10training_.csv        first 100k of train pool
    {app}_20training_.csv        first 200k of train pool
    {app}_test_home1Small_.csv   full house-1 timeline

  Val/test do NOT change when training goes 100k → 200k → mixed 400k.

**Pool-only** (stable val within first N rows) — thesis / debugging:

  python prepare_all_ukdale.py
  python build_geng_mix.py --splits-only

  Writes {app}_house2_pool.csv / {app}_house1_pool.csv (default N=400000).
  Splits deferred to build_geng_mix.py.

Raw .dat layout (relative to this script):
  UK_DALE/house_1/channel_1.dat  (mains)
  UK_DALE/house_2/channel_1.dat  (mains)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REL_DATA_DIR = "UK_DALE"
REL_SAVE_DIR = "created_data/UK_DALE"

AGG_MEAN = 522
AGG_STD = 814
SAMPLE_SECONDS = 6
DEFAULT_VAL_PCT = 20
DEFAULT_TEST_PCT = 20
TRAIN_CROP_SIZES = (100_000, 200_000)
DEFAULT_POOL_LIMIT = 400_000

ALL_APPLIANCES = (
    "kettle",
    "microwave",
    "fridge",
    "dishwasher",
    "washingmachine",
)

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
    p = Path(path)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return p.resolve()


def _rel_to_script(path: Path) -> str:
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


def rows_to_days(n_rows: int, sample_seconds: int = SAMPLE_SECONDS) -> float:
    return n_rows * sample_seconds / 86_400


def truncate_timeline(df: pd.DataFrame, limit: int | None) -> tuple[pd.DataFrame, int]:
    """Keep only the first `limit` chronological rows; return (subset, original length)."""
    original_len = len(df)
    if limit is None or limit <= 0 or limit >= original_len:
        return df, original_len
    return df.iloc[:limit].reset_index(drop=True), original_len


def save_csv(df: pd.DataFrame, path: Path, dry_run: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"  [dry-run] would write {path} ({len(df):,} rows)")
        return
    df.to_csv(path, index=False, header=False)


def crop_label(n_rows: int) -> str:
    """EasyS2S TrainPercent label: 100k → '10', 200k → '20'."""
    return str(n_rows // 10_000)


def print_paper_split_summary(appliance: str, manifest: dict, train_crops: tuple[int, ...]) -> None:
    """Print Geng paper timestep table for one appliance."""
    outputs = manifest.get("outputs", {})
    train_n = outputs.get(f"{appliance}_training_.csv", 0)
    val_n = outputs.get(f"{appliance}_validation_.csv", 0)
    test_n = outputs.get(f"{appliance}_test_.csv", 0)
    h1_n = outputs.get(f"{appliance}_test_home1Small_.csv", 0)
    total = manifest.get("house2_timeline_rows", train_n + val_n + test_n)

    print(f"\n  [{appliance}] paper-exact timesteps (@ {SAMPLE_SECONDS}s per row)")
    print(f"    house-2 timeline total : {total:,}  (~{rows_to_days(total):.0f} days)")
    print(f"    train pool (60%)       : {train_n:,}")
    print(f"    validation (20%)       : {val_n:,}  ← fixed for ALL experiments")
    print(f"    test house 2 (20%)     : {test_n:,}  ← fixed for ALL experiments")
    if h1_n:
        print(f"    test house 1 (full)    : {h1_n:,}")
    print("    experiment training crops (val/test unchanged):")
    for crop_n in train_crops:
        label = crop_label(crop_n)
        crop_name = f"{appliance}_{label}training_.csv"
        crop_n_actual = outputs.get(crop_name, 0)
        print(f"      Origin {crop_n // 1000}k ({crop_name}): {crop_n_actual:,}")
    print("    augmented training (built by build_geng_mix.py):")
    for crop_n in train_crops:
        label = crop_label(crop_n)
        mix_n = crop_n * 2
        print(f"      {crop_n // 1000}k+{crop_n // 1000}k (UK_DALECombined*_file{label}): {mix_n:,}")


def pool_csv_name(appliance: str, house: int) -> str:
    return f"{appliance}_house{house}_pool.csv"


def load_pool_csv(path: Path) -> pd.DataFrame:
    """Load a 2-column z-score pool CSV (no header)."""
    if not path.is_file():
        raise FileNotFoundError(f"Pool CSV not found: {path}")
    return pd.read_csv(path, header=None)


def process_appliance_pools(
    appliance: str,
    data_dir: Path,
    save_root: Path,
    pool_limit: int,
    skip_house1: bool,
    aggregate_mean: float,
    aggregate_std: float,
    dry_run: bool,
) -> dict:
    out_dir = Path(save_root) / appliance
    manifest: dict = {"appliance": appliance, "mode": "pool_only", "outputs": {}}

    for house in (2, 1):
        if house == 1 and skip_house1:
            continue
        print(f"\n[{appliance}] House {house} — align, first {pool_limit:,} rows, z-score")
        raw = align_and_resample(data_dir, house=house, appliance=appliance)
        raw, full_rows = truncate_timeline(raw, pool_limit)
        print(
            f"  pool: {len(raw):,} / {full_rows:,} rows "
            f"({rows_to_days(len(raw)):.1f} days @ {SAMPLE_SECONDS}s)"
        )
        norm = zscore_normalize(raw, appliance, aggregate_mean, aggregate_std)
        name = pool_csv_name(appliance, house)
        save_csv(norm, out_dir / name, dry_run=dry_run)
        manifest["outputs"][name] = len(norm)
        manifest[f"house{house}_pool_rows"] = len(norm)
        manifest[f"house{house}_timeline_rows_full"] = full_rows

    manifest["pool_limit"] = pool_limit
    return manifest


def process_appliance_paper_exact(
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
    """Geng paper: full house-2 timeline, one 6:2:2 split, fixed val/test, train crops."""
    out_dir = Path(save_root) / appliance
    manifest: dict = {"appliance": appliance, "mode": "paper_exact", "outputs": {}}

    print(f"\n[{appliance}] House 2 — full timeline, z-score, paper 6:2:2 split")
    raw_h2 = align_and_resample(data_dir, house=2, appliance=appliance)
    house2_full_rows = len(raw_h2)
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

    if not skip_house1:
        print(f"[{appliance}] House 1 — full timeline cross-household test")
        raw_h1 = align_and_resample(data_dir, house=1, appliance=appliance)
        norm_h1 = zscore_normalize(raw_h1, appliance, aggregate_mean, aggregate_std)
        h1_name = f"{appliance}_test_home1Small_.csv"
        save_csv(norm_h1, out_dir / h1_name, dry_run=dry_run)
        manifest["outputs"][h1_name] = len(norm_h1)

    manifest["house2_timeline_rows"] = len(norm_h2)
    manifest["house2_timeline_rows_full"] = house2_full_rows
    manifest["split_ratio"] = (
        f"{100 - validation_percent - test_percent}:{validation_percent}:{test_percent}"
    )
    manifest["experiment_training_rows"] = {
        "full_train_pool": len(train),
        "validation": len(val),
        "test_house2": len(test),
        "origin_crops": {
            f"{crop_n // 1000}k": min(crop_n, len(train)) for crop_n in train_crops
        },
        "augmented_mix_rows": {
            f"{crop_n // 1000}k+{crop_n // 1000}k": crop_n * 2 for crop_n in train_crops
        },
    }
    if not skip_house1:
        manifest["experiment_training_rows"]["test_house1"] = manifest["outputs"].get(
            f"{appliance}_test_home1Small_.csv", 0
        )

    print_paper_split_summary(appliance, manifest, train_crops)
    return manifest


def process_appliance_legacy_split(
    appliance: str,
    data_dir: Path,
    save_root: Path,
    validation_percent: float,
    test_percent: float,
    train_crops: tuple[int, ...],
    skip_house1: bool,
    aggregate_mean: float,
    aggregate_std: float,
    house2_timeline_limit: int | None,
    dry_run: bool,
) -> dict:
    out_dir = Path(save_root) / appliance
    manifest: dict = {"appliance": appliance, "mode": "legacy_paper_split", "outputs": {}}

    print(f"\n[{appliance}] House 2 — align, z-score, 6:2:2 split (legacy)")
    raw_h2 = align_and_resample(data_dir, house=2, appliance=appliance)
    raw_h2, house2_full_rows = truncate_timeline(raw_h2, house2_timeline_limit)
    if house2_timeline_limit is not None and len(raw_h2) < house2_full_rows:
        print(
            f"  timeline window: first {len(raw_h2):,} / {house2_full_rows:,} rows "
            f"({rows_to_days(len(raw_h2)):.1f} days @ {SAMPLE_SECONDS}s)"
        )
    norm_h2 = zscore_normalize(raw_h2, appliance, aggregate_mean, aggregate_std)
    train, val, test = split_chronological(norm_h2, validation_percent, test_percent)
    print(
        f"  split sizes: train {len(train):,} | val {len(val):,} | test {len(test):,} "
        f"({rows_to_days(len(train)):.1f} / {rows_to_days(len(val)):.1f} / "
        f"{rows_to_days(len(test)):.1f} days)"
    )

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

    if not skip_house1:
        print(f"[{appliance}] House 1 — cross-household test (legacy full timeline)")
        raw_h1 = align_and_resample(data_dir, house=1, appliance=appliance)
        norm_h1 = zscore_normalize(raw_h1, appliance, aggregate_mean, aggregate_std)
        h1_name = f"{appliance}_test_home1Small_.csv"
        save_csv(norm_h1, out_dir / h1_name, dry_run=dry_run)
        manifest["outputs"][h1_name] = len(norm_h1)

    manifest["house2_timeline_rows"] = len(norm_h2)
    manifest["house2_timeline_rows_full"] = house2_full_rows
    manifest["house2_timeline_limit"] = house2_timeline_limit
    manifest["split_ratio"] = (
        f"{100 - validation_percent - test_percent}:{validation_percent}:{test_percent}"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare UK-DALE pool CSVs (default) or legacy paper 6:2:2 split"
    )
    parser.add_argument("--data_dir", type=str, default=REL_DATA_DIR)
    parser.add_argument("--save_path", type=str, default=REL_SAVE_DIR)
    parser.add_argument(
        "--appliances",
        nargs="+",
        default=list(ALL_APPLIANCES),
        choices=ALL_APPLIANCES,
    )
    parser.add_argument(
        "--pool_limit",
        type=int,
        default=DEFAULT_POOL_LIMIT,
        help=f"First N resampled rows per house for pool CSVs (default {DEFAULT_POOL_LIMIT})",
    )
    parser.add_argument(
        "--paper-exact",
        action="store_true",
        help="Geng paper: full house-2 timeline 6:2:2 + 100k/200k crops + house-1 test",
    )
    parser.add_argument(
        "--legacy-paper-split",
        action="store_true",
        help="Alias for --paper-exact (optional --house2_timeline_limit to cap timeline)",
    )
    parser.add_argument("--validation_percent", type=float, default=DEFAULT_VAL_PCT)
    parser.add_argument("--testing_percent", type=float, default=DEFAULT_TEST_PCT)
    parser.add_argument(
        "--train_crops",
        nargs="+",
        type=int,
        default=list(TRAIN_CROP_SIZES),
    )
    parser.add_argument("--aggregate_mean", type=float, default=AGG_MEAN)
    parser.add_argument("--aggregate_std", type=float, default=AGG_STD)
    parser.add_argument(
        "--house2_timeline_limit",
        type=int,
        default=None,
        help="Legacy only: cap house-2 rows before 6:2:2 split",
    )
    parser.add_argument("--skip_house1", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = _resolve_dir(args.data_dir)
    save_path = _resolve_dir(args.save_path)

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"data_dir not found: {_rel_to_script(data_dir)}\n"
            f"  Put UK-DALE .dat files under: {REL_DATA_DIR}/house_2/channel_1.dat"
        )

    paper_exact = args.paper_exact or args.legacy_paper_split
    if paper_exact and args.validation_percent + args.testing_percent >= 100:
        raise ValueError("validation_percent + testing_percent must be < 100")

    t0 = time.time()
    print("UK-DALE batch preparation")
    print(f"  script dir: {_rel_to_script(SCRIPT_DIR)}")
    print(f"  data_dir:   {_rel_to_script(data_dir)}")
    print(f"  save_path:  {_rel_to_script(save_path)}")
    print(f"  appliances: {args.appliances}")

    all_manifests = []
    if paper_exact:
        use_timeline_cap = args.legacy_paper_split and args.house2_timeline_limit
        if use_timeline_cap:
            print("  mode: paper-exact with --house2_timeline_limit cap (non-standard)")
            print(
                f"  split: {100 - args.validation_percent - args.testing_percent}:"
                f"{args.validation_percent}:{args.testing_percent}"
            )
            for appliance in args.appliances:
                manifest = process_appliance_legacy_split(
                    appliance=appliance,
                    data_dir=data_dir,
                    save_root=save_path,
                    validation_percent=args.validation_percent,
                    test_percent=args.testing_percent,
                    train_crops=tuple(args.train_crops),
                    skip_house1=args.skip_house1,
                    aggregate_mean=args.aggregate_mean,
                    aggregate_std=args.aggregate_std,
                    house2_timeline_limit=args.house2_timeline_limit,
                    dry_run=args.dry_run,
                )
                all_manifests.append(manifest)
        else:
            print("  mode: paper-exact (full timeline, Geng 6:2:2)")
            print(
                f"  split: {100 - args.validation_percent - args.testing_percent}:"
                f"{args.validation_percent}:{args.testing_percent}"
            )
            for appliance in args.appliances:
                manifest = process_appliance_paper_exact(
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
    else:
        print(f"  mode: pool-only (first {args.pool_limit:,} rows per house)")
        print("  splits: deferred to build_geng_mix.py")
        for appliance in args.appliances:
            manifest = process_appliance_pools(
                appliance=appliance,
                data_dir=data_dir,
                save_root=save_path,
                pool_limit=args.pool_limit,
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
                    "mode": "paper_exact" if paper_exact else "pool_only",
                    "pool_limit": args.pool_limit,
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
    if not paper_exact:
        print("\nNext steps:")
        print("  1. build_geng_mix.py --splits-only   (6:2:2 + origin crops from pools)")
        print("  2. algorithm1.py                     (diffusion train data)")
        print("  3. build_geng_mix.py                  (auto split + mix in one command)")
    else:
        print("\nNext steps (paper-exact):")
        print("  1. algorithm1.py                     (diffusion: uses {app}_training_.csv)")
        print("  2. run_diffusion_all.py              (sample synthetic data)")
        print("  3. build_geng_mix.py --paper-exact   (mix CSVs only; splits already done)")


if __name__ == "__main__":
    main()
