"""
Geng et al. (Energy 2025) — splits, npy -> synthetic, paper mix scenarios.

Paper-exact (full timeline, Geng Tables 5–9):
  1. python NILM-main/dataset_preprocess/prepare_all_ukdale.py --paper-exact
  2. algorithm1.py + run_diffusion_all.py
  3. python build_geng_mix.py --paper-exact

Pool mode (first 400k rows, stable val):
  1. prepare_all_ukdale.py
  2. build_geng_mix.py --splits-only
  3. algorithm1.py + run_diffusion_all.py
  4. build_geng_mix.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "NILM-main" / "dataset_preprocess"))

from prepare_all_ukdale import (  # noqa: E402
    AGG_MEAN,
    AGG_STD,
    ALL_APPLIANCES,
    DEFAULT_TEST_PCT,
    DEFAULT_VAL_PCT,
    PARAMS,
    align_and_resample,
    crop_label,
    load_pool_csv,
    pool_csv_name,
    rows_to_days,
    split_chronological,
    zscore_normalize,
)

NILM_ON_THRESHOLDS_W = {
    "kettle": 2000,
    "microwave": 200,
    "fridge": 50,
    "dishwasher": 10,
    "washingmachine": 20,
}

DEFAULT_NPY_DIR = SCRIPT_DIR / "OUTPUT"
DEFAULT_GEN_DIR = SCRIPT_DIR / "generatedData"
DEFAULT_MIXED_DIR = SCRIPT_DIR / "NILM-main" / "dataset_preprocess" / "created_data" / "UK_DALE"
DEFAULT_TRAIN_ROOT = DEFAULT_MIXED_DIR
DEFAULT_UKDALE_RAW = SCRIPT_DIR / "NILM-main" / "dataset_preprocess" / "UK_DALE"
DEFAULT_ALG1_CSV = SCRIPT_DIR / "Data" / "datasets"


@dataclass(frozen=True)
class GengMixScenario:
    name: str
    n_real: int
    n_syn: int
    mix_file_label: str | None = None

    @property
    def file_label(self) -> str:
        """Combined CSV suffix (TrainPercent for augmented EasyS2S)."""
        if self.mix_file_label is not None:
            return self.mix_file_label
        return crop_label(self.n_real)

    @property
    def real_origin_label(self) -> str:
        """Origin crop label for loading {app}_{label}training_.csv."""
        return crop_label(self.n_real)

    @property
    def origin_csv(self) -> str:
        return f"{{app}}_{self.real_origin_label}training_.csv"

    @property
    def combined_csv(self) -> str:
        return f"UK_DALECombined{{app}}_file{self.file_label}.csv"


# Paper Tables 5–7 mix scenarios (augmented). Origin 100k/200k from prepare_all only.
GENG_MIX_SCENARIOS: tuple[GengMixScenario, ...] = (
    GengMixScenario("100k+100k", 100_000, 100_000),
    GengMixScenario("200k+200k", 200_000, 200_000),
    GengMixScenario("100k+200k", 100_000, 200_000, mix_file_label="10_20"),
    GengMixScenario("200k+100k", 200_000, 100_000, mix_file_label="20_10"),
)

# Pool CSVs from prepare_all_ukdale.py (pool-only mode).
GENG_POOL_CSVS = (
    "{app}_house2_pool.csv",
    "{app}_house1_pool.csv",
)

# Experiment CSVs written by ensure_geng_splits() or legacy prepare_all.
GENG_SUPPORT_CSVS = (
    "{app}_training_.csv",
    "{app}_validation_.csv",
    "{app}_test_.csv",
    "{app}_test_home1Small_.csv",
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return (SCRIPT_DIR / p).resolve() if not p.is_absolute() else p.resolve()


def combined_csv_path(train_root: Path, appliance: str, scenario: GengMixScenario) -> Path:
    name = scenario.combined_csv.format(app=appliance)
    return train_root / appliance / name


def scenarios_with_missing_outputs(
    train_root: Path,
    appliances: tuple[str, ...],
) -> tuple[GengMixScenario, ...]:
    """Return mix scenarios where at least one appliance is missing its combined CSV."""
    out: list[GengMixScenario] = []
    for scenario in GENG_MIX_SCENARIOS:
        for app in appliances:
            if not combined_csv_path(train_root, app, scenario).is_file():
                out.append(scenario)
                break
    return tuple(out)


def resolve_scenarios(
    args: argparse.Namespace,
    *,
    appliances: tuple[str, ...],
    train_root: Path,
) -> tuple[GengMixScenario, ...]:
    if args.scenario == "all":
        return GENG_MIX_SCENARIOS
    if args.scenario == "missing":
        missing = scenarios_with_missing_outputs(train_root, appliances)
        if not missing:
            print("All mix CSVs already exist for selected appliances.")
        else:
            print(
                "Missing mix scenarios to build: "
                + ", ".join(s.name for s in missing)
            )
        return missing
    if args.scenario == "10":
        return (GENG_MIX_SCENARIOS[0],)
    if args.scenario == "20":
        return (GENG_MIX_SCENARIOS[1],)
    if args.scenario == "10_20":
        return (GENG_MIX_SCENARIOS[2],)
    if args.scenario == "20_10":
        return (GENG_MIX_SCENARIOS[3],)
    if args.n_real is not None and args.n_syn is not None:
        return (GengMixScenario(f"{args.n_real//1000}k+{args.n_syn//1000}k", args.n_real, args.n_syn),)
    return GENG_MIX_SCENARIOS


def save_zscore_csv(df: pd.DataFrame, path: Path, dry_run: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"  [dry-run] would write {_rel(path)} ({len(df):,} rows)")
        return
    df.to_csv(path, index=False, header=False)
    print(f"  wrote {_rel(path)} ({len(df):,} rows)")


def pool_path(train_root: Path, appliance: str, house: int) -> Path:
    return train_root / appliance / pool_csv_name(appliance, house)


def has_paper_exact_splits(train_root: Path, appliance: str) -> bool:
    app_dir = train_root / appliance
    return (
        (app_dir / f"{appliance}_training_.csv").is_file()
        and (app_dir / f"{appliance}_validation_.csv").is_file()
        and (app_dir / f"{appliance}_10training_.csv").is_file()
        and (app_dir / f"{appliance}_20training_.csv").is_file()
    )


def all_have_paper_exact_splits(train_root: Path, appliances: tuple[str, ...]) -> bool:
    return all(has_paper_exact_splits(train_root, app) for app in appliances)


def print_paper_exact_status(train_root: Path, appliances: tuple[str, ...]) -> None:
    manifest_path = train_root / "preprocessing_manifest.json"
    if not manifest_path.is_file():
        print("  paper-exact CSVs detected (no preprocessing_manifest.json)")
        return
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if data.get("mode") != "paper_exact":
        return
    print(f"  preprocessing manifest: {_rel(manifest_path)}")
    for app in appliances:
        for entry in data.get("appliances", []):
            if entry.get("appliance") != app:
                continue
            exp = entry.get("experiment_training_rows", {})
            if not exp:
                break
            print(f"\n  [{app}] paper-exact timesteps")
            print(f"    train pool : {exp.get('full_train_pool', 0):,}")
            print(f"    validation : {exp.get('validation', 0):,}  (fixed for all experiments)")
            print(f"    test h2    : {exp.get('test_house2', 0):,}  (fixed for all experiments)")
            if "test_house1" in exp:
                print(f"    test h1    : {exp['test_house1']:,}")
            for k, v in exp.get("origin_crops", {}).items():
                print(f"    origin {k}  : {v:,}")
            for k, v in exp.get("augmented_mix_rows", {}).items():
                print(f"    mix {k}    : {v:,}  (training only)")
            break


def required_crop_sizes(scenarios: tuple[GengMixScenario, ...]) -> tuple[int, ...]:
    sizes = {s.n_real for s in scenarios}
    return tuple(sorted(sizes))


def ensure_geng_splits(
    appliances: tuple[str, ...],
    train_root: Path,
    scenarios: tuple[GengMixScenario, ...],
    *,
    validation_percent: float,
    test_percent: float,
    skip_house1: bool,
    force: bool,
    dry_run: bool,
) -> list[dict]:
    """Split house-2 pool 6:2:2 and write only the experiment CSVs needed."""
    manifests: list[dict] = []
    crop_sizes = required_crop_sizes(scenarios) if scenarios else (100_000, 200_000)

    for app in appliances:
        pool_h2 = pool_path(train_root, app, 2)
        if not pool_h2.is_file():
            legacy_train = train_root / app / f"{app}_training_.csv"
            if legacy_train.is_file():
                print(f"  [{app}] legacy split CSVs present — skip pool split")
                continue
            raise FileNotFoundError(
                f"Missing {_rel(pool_h2)}. Run:\n"
                f"  python NILM-main/dataset_preprocess/prepare_all_ukdale.py"
            )

        print(f"\n[{app}] split house-2 pool → train/val/test + origin crops")
        pool_z = load_pool_csv(pool_h2)
        train, val, test = split_chronological(pool_z, validation_percent, test_percent)
        print(
            f"  pool {len(pool_z):,} rows → train {len(train):,} | val {len(val):,} | "
            f"test {len(test):,} ({rows_to_days(len(train)):.1f} / "
            f"{rows_to_days(len(val)):.1f} / {rows_to_days(len(test)):.1f} days)"
        )

        app_dir = train_root / app
        split_files = {
            f"{app}_training_.csv": train,
            f"{app}_validation_.csv": val,
            f"{app}_test_.csv": test,
        }
        written: dict[str, int] = {}
        for name, frame in split_files.items():
            out = app_dir / name
            if force or not out.is_file():
                save_zscore_csv(frame, out, dry_run=dry_run)
            written[name] = len(frame)

        for crop_n in crop_sizes:
            label = crop_label(crop_n)
            crop_name = f"{app}_{label}training_.csv"
            out = app_dir / crop_name
            if force or not out.is_file():
                cropped = train.iloc[: min(crop_n, len(train))].reset_index(drop=True)
                save_zscore_csv(cropped, out, dry_run=dry_run)
                written[crop_name] = len(cropped)
                if len(cropped) < crop_n:
                    print(
                        f"  warning: train pool has {len(train):,} rows, "
                        f"requested {crop_n:,} for {crop_name}"
                    )

        if not skip_house1:
            pool_h1 = pool_path(train_root, app, 1)
            h1_out = app_dir / f"{app}_test_home1Small_.csv"
            if pool_h1.is_file() and (force or not h1_out.is_file()):
                pool_h1_z = load_pool_csv(pool_h1)
                print(f"  house-1 pool → {_rel(h1_out)} ({len(pool_h1_z):,} rows)")
                save_zscore_csv(pool_h1_z, h1_out, dry_run=dry_run)
                written[f"{app}_test_home1Small_.csv"] = len(pool_h1_z)
            elif not pool_h1.is_file():
                print(f"  warning: missing {_rel(pool_h1)} — skip house-1 test CSV")

        manifests.append({"appliance": app, "written": written})

    return manifests


def verify_pool_outputs(train_root: Path, appliances: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    print("Checking prepare_all_ukdale pool CSVs:")
    for app in appliances:
        for pattern in GENG_POOL_CSVS:
            path = train_root / app / pattern.format(app=app)
            if path.is_file():
                print(f"  OK  {_rel(path)}")
            else:
                print(f"  MISS {_rel(path)}")
                missing.append(_rel(path))
    print()
    return missing


def verify_prepare_all_outputs(train_root: Path, appliances: tuple[str, ...]) -> list[str]:
    """Check real CSVs from prepare_all_ukdale exist before mixing."""
    missing: list[str] = []
    print("Checking prepare_all_ukdale outputs:")
    for app in appliances:
        app_dir = train_root / app
        checks = [s.format(app=app) for s in GENG_SUPPORT_CSVS]
        checks += [s.origin_csv.format(app=app) for s in GENG_MIX_SCENARIOS]
        for name in checks:
            path = app_dir / name
            if path.is_file():
                print(f"  OK  {_rel(path)}")
            else:
                print(f"  MISS {_rel(path)}")
                missing.append(_rel(path))
    if missing:
        print()
        print("WARNING: missing experiment CSVs.")
        print("  Paper-exact: python NILM-main/dataset_preprocess/prepare_all_ukdale.py --paper-exact")
        print("  Pool mode:   python build_geng_mix.py --splits-only")
    print()
    return missing


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"{path.name}: expected [windows, timesteps, channels], got {arr.shape}")
    return arr


def detect_npy_scale(power: np.ndarray) -> str:
    pmax = float(np.nanmax(power))
    pmin = float(np.nanmin(power))
    if pmax <= 1.05 and pmin >= -0.05:
        return "unit01"
    return "watts"


def inverse_minmax_to_watts(power: np.ndarray, appliance: str, alg1_csv: Path) -> np.ndarray:
    if not alg1_csv.is_file():
        raise FileNotFoundError(
            f"Need {_rel(alg1_csv)} to inverse MinMax. Run algorithm1.py first."
        )
    raw = pd.read_csv(alg1_csv, header=0).values.astype(np.float64)
    scaler = MinMaxScaler().fit(raw)
    flat = power.reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(power.shape).astype(np.float32)


def windows_to_timesteps(windows: np.ndarray) -> np.ndarray:
    if windows.ndim == 3:
        windows = windows[:, :, 0]
    return windows.reshape(-1).astype(np.float32)


def post_filter_power(power: np.ndarray, threshold: float) -> np.ndarray:
    out = power.copy()
    out[out < threshold] = 0.0
    out[out < 0] = 0.0
    return out


def npy_to_watt_series(
    appliance: str,
    npy_dir: Path,
    alg1_dir: Path,
    post_filter: bool,
) -> np.ndarray:
    npy_path = npy_dir / appliance / f"ddpm_fake_{appliance}.npy"
    if not npy_path.is_file():
        raise FileNotFoundError(f"Missing {_rel(npy_path)} — run sampling first.")

    arr = load_npy(npy_path)
    power = arr[:, :, 0]

    if detect_npy_scale(power) == "unit01":
        print(f"  {appliance}: npy [0,1] MinMax -> inverse to watts")
        power = inverse_minmax_to_watts(power, appliance, alg1_dir / f"{appliance}.csv")
    else:
        print(f"  {appliance}: npy in watts ({power.min():.1f}-{power.max():.1f} W)")

    series = windows_to_timesteps(power)
    if post_filter:
        thr = NILM_ON_THRESHOLDS_W[appliance]
        series = post_filter_power(series, thr)
        print(f"  {appliance}: post-filter < {thr} W -> 0")
    return series


def save_watt_series(path: Path, appliance: str, series: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({appliance: series}).to_csv(path, index=False)


def build_all_watt_series(
    npy_dir: Path,
    gen_dir: Path,
    alg1_dir: Path,
    post_filter: bool,
) -> dict[str, np.ndarray]:
    series_map: dict[str, np.ndarray] = {}
    for app in ALL_APPLIANCES:
        series = npy_to_watt_series(app, npy_dir, alg1_dir, post_filter)
        out = gen_dir / f"{app}_syn_watts.csv"
        save_watt_series(out, app, series)
        print(f"  saved {_rel(out)} ({len(series):,} timesteps)")
        series_map[app] = series
    return series_map


def build_synthetic_xy_watts(
    target: str,
    series_map: dict[str, np.ndarray],
    n_syn: int,
) -> pd.DataFrame:
    lengths = {a: len(series_map[a]) for a in ALL_APPLIANCES}
    if min(lengths.values()) < n_syn:
        raise ValueError(
            f"Need {n_syn} synthetic timesteps; shortest series {min(lengths.values())} "
            f"({lengths}). Sample more windows or use a smaller scenario."
        )

    power_sum = np.zeros(n_syn, dtype=np.float32)
    for app in ALL_APPLIANCES:
        power_sum += series_map[app][:n_syn]

    y_syn = series_map[target][:n_syn].astype(np.float32)
    return pd.DataFrame({"aggregate": power_sum, target: y_syn})


def zscore_rows_to_watts(df_z: pd.DataFrame, appliance: str) -> pd.DataFrame:
    app_mean = PARAMS[appliance]["mean"]
    app_std = PARAMS[appliance]["std"]
    return pd.DataFrame(
        {
            "aggregate": df_z.iloc[:, 0].to_numpy(dtype=np.float64) * AGG_STD + AGG_MEAN,
            appliance: df_z.iloc[:, 1].to_numpy(dtype=np.float64) * app_std + app_mean,
        }
    )


def load_real_watts(
    appliance: str,
    n_real: int,
    train_root: Path,
    ukdale_raw: Path,
    real_csv: Path | None,
    scenario: GengMixScenario,
) -> pd.DataFrame:
    if real_csv is not None:
        df = pd.read_csv(real_csv)
        if "aggregate" not in df.columns or appliance not in df.columns:
            raise ValueError(f"--real-csv must have columns aggregate, {appliance}")
        return df[["aggregate", appliance]].iloc[:n_real].copy()

    label = scenario.real_origin_label
    z_path = train_root / appliance / f"{appliance}_{label}training_.csv"
    if z_path.is_file():
        print(f"  real: {_rel(z_path)} (denormalize z-score -> watts)")
        df_z = pd.read_csv(z_path, header=None)
        return zscore_rows_to_watts(df_z, appliance).iloc[:n_real].copy()

    if ukdale_raw.is_dir():
        try:
            raw = align_and_resample(ukdale_raw, house=2, appliance=appliance)
            if len(raw) >= n_real:
                print(f"  real: UK-DALE house-2 align ({_rel(ukdale_raw)})")
                return raw[["aggregate", appliance]].iloc[:n_real].copy()
        except FileNotFoundError:
            pass

    raise FileNotFoundError(
        f"Missing real data for {scenario.name}. Expected {_rel(z_path)} from prepare_all_ukdale.py."
    )


def geng_mix_labeled(
    real_watts: pd.DataFrame,
    syn_watts: pd.DataFrame,
    appliance: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Geng D_mix: concat real block then syn block (no row shuffle here).

    EasyS2S / ChunkS2S_Slider uses shuffle=True at train time — same as Geng original code.
    source: 0=real, 1=synthetic.
    """
    real_block = real_watts.copy()
    real_block["source"] = 0
    syn_block = syn_watts.copy()
    syn_block["source"] = 1
    labeled = pd.concat([real_block, syn_block], ignore_index=True)
    mixed_watts = labeled[["aggregate", appliance]].copy()
    return mixed_watts, labeled


def save_labeled_mix_csv(df_labeled: pd.DataFrame, appliance: str, path: Path) -> None:
    """Watts + source column for visualization (0=real, 1=synthetic)."""
    out = df_labeled[["aggregate", appliance, "source"]].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    n_real = int((out["source"] == 0).sum())
    n_syn = int((out["source"] == 1).sum())
    print(
        f"  labeled CSV: {_rel(path)} ({len(out):,} rows; "
        f"real={n_real:,}, syn={n_syn:,})"
    )


def save_geng_nilm_csv(df_watts: pd.DataFrame, appliance: str, path: Path) -> None:
    norm = zscore_normalize(df_watts, appliance, AGG_MEAN, AGG_STD)
    path.parent.mkdir(parents=True, exist_ok=True)
    norm.to_csv(path, index=False, header=False)
    print(f"  mixed NILM CSV: {_rel(path)} ({len(norm):,} rows, z-scored)")


def process_appliance_scenario(
    appliance: str,
    series_map: dict[str, np.ndarray],
    scenario: GengMixScenario,
    args: argparse.Namespace,
) -> dict:
    n_real, n_syn = scenario.n_real, scenario.n_syn

    syn_watts = build_synthetic_xy_watts(appliance, series_map, n_syn)
    syn_only_path = args.gen_dir / f"synthetic_{appliance}_{n_syn}.csv"
    if not args.dry_run:
        syn_watts.to_csv(syn_only_path, index=False)
    print(f"  synthetic XY: {_rel(syn_only_path)} (X=sum of 5 apps, Y={appliance})")

    real_watts = load_real_watts(
        appliance, n_real, args.train_root, args.ukdale_raw, args.real_csv, scenario
    )
    if len(real_watts) < n_real:
        raise ValueError(f"Only {len(real_watts)} real rows, need {n_real}")

    mixed_watts, labeled = geng_mix_labeled(
        real_watts.iloc[:n_real], syn_watts, appliance
    )
    out_name = scenario.combined_csv.format(app=appliance)
    out_path = args.mixed_dir / appliance / out_name
    labeled_path = args.mixed_dir / appliance / out_name.replace(".csv", "_labeled.csv")

    if getattr(args, "skip_existing", False) and out_path.is_file():
        print(f"  skip existing: {_rel(out_path)}")
        origin_path = args.train_root / appliance / scenario.origin_csv.format(app=appliance)
        return {
            "scenario": scenario.name,
            "appliance": appliance,
            "n_real": n_real,
            "n_syn": n_syn,
            "n_mix": None,
            "file_label": scenario.file_label,
            "origin_csv": _rel(origin_path),
            "mixed_csv": _rel(out_path),
            "labeled_csv": _rel(labeled_path),
            "syn_only_csv": _rel(syn_only_path),
            "skipped": True,
        }

    if args.dry_run:
        print(f"  [dry-run] would write {_rel(out_path)} ({len(mixed_watts):,} rows, concat order)")
        print(f"  [dry-run] would write {_rel(labeled_path)} (source 0=real block, 1=syn block)")
    else:
        save_geng_nilm_csv(mixed_watts, appliance, out_path)
        save_labeled_mix_csv(labeled, appliance, labeled_path)

    origin_path = args.train_root / appliance / scenario.origin_csv.format(app=appliance)
    return {
        "scenario": scenario.name,
        "appliance": appliance,
        "n_real": n_real,
        "n_syn": n_syn,
        "n_mix": len(mixed_watts),
        "file_label": scenario.file_label,
        "origin_csv": _rel(origin_path),
        "mixed_csv": _rel(out_path),
        "labeled_csv": _rel(labeled_path),
        "syn_only_csv": _rel(syn_only_path),
    }


def print_experiment_summary(train_root: Path, appliances: tuple[str, ...]) -> None:
    print("=" * 72)
    print("Geng experiment training CSV map (per appliance)")
    print("=" * 72)
    print(f"{'Scenario':<14} {'EasyS2S TrainPercent':<20} {'Training CSV'}")
    print("-" * 72)
    print(f"{'Origin 100k':<14} {'10':<20} {'{app}_10training_.csv'}")
    print(f"{'Origin 200k':<14} {'20':<20} {'{app}_20training_.csv'}")
    print(f"{'100k+100k':<14} {'10':<20} {'UK_DALECombined{app}_file10.csv'}")
    print(f"{'200k+200k':<14} {'20':<20} {'UK_DALECombined{app}_file20.csv'}")
    print(f"{'100k+200k':<14} {'10_20':<20} {'UK_DALECombined{app}_file10_20.csv'}")
    print(f"{'200k+100k':<14} {'20_10':<20} {'UK_DALECombined{app}_file20_10.csv'}")
    print()
    print("Val/test (all scenarios): {app}_validation_.csv, {app}_test_.csv, {app}_test_home1Small_.csv")
    print(f"Folder: {_rel(train_root)}/{{app}}/")
    print("=" * 72)
    for app in appliances:
        app_dir = train_root / app
        print(f"\n[{app}]")
        for pattern in (
            f"{app}_10training_.csv",
            f"{app}_20training_.csv",
            f"UK_DALECombined{app}_file10.csv",
            f"UK_DALECombined{app}_file20.csv",
            f"UK_DALECombined{app}_file10_20.csv",
            f"UK_DALECombined{app}_file20_10.csv",
            f"{app}_validation_.csv",
            f"{app}_test_.csv",
        ):
            path = app_dir / pattern
            mark = "OK " if path.is_file() else "MISS"
            print(f"  {mark}  {pattern}")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-click Geng mix: npy -> X/Y -> all paper scenarios"
    )
    p.add_argument(
        "--paper-exact",
        action="store_true",
        help="Use CSVs from prepare_all_ukdale.py --paper-exact (skip pool split step)",
    )
    p.add_argument(
        "--splits-only",
        action="store_true",
        help="Only split house pools → train/val/test + origin crops (no npy mix)",
    )
    p.add_argument(
        "--force-splits",
        action="store_true",
        help="Overwrite split/crop CSVs even if they already exist",
    )
    p.add_argument(
        "--validation-percent",
        type=float,
        default=DEFAULT_VAL_PCT,
        help="House-2 pool 6:2:2 validation fraction (default 20)",
    )
    p.add_argument(
        "--test-percent",
        type=float,
        default=DEFAULT_TEST_PCT,
        help="House-2 pool 6:2:2 test fraction (default 20)",
    )
    p.add_argument(
        "--skip-house1",
        action="store_true",
        help="Do not write {app}_test_home1Small_.csv from house-1 pool",
    )
    p.add_argument("--appliance", choices=list(ALL_APPLIANCES), default=None)
    p.add_argument("--all", action="store_true", help="All five appliances (default if no --appliance)")
    p.add_argument(
        "--scenario",
        choices=["all", "missing", "10", "20", "10_20", "20_10"],
        default="all",
        help=(
            "Mix scenario: all (default, all 4 mixes), missing (only absent CSVs), "
            "10, 20, 10_20 (100k+200k), 20_10 (200k+100k)"
        ),
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="With --scenario all/missing: do not overwrite mix CSVs that already exist",
    )
    p.add_argument("--n-real", type=int, default=None, help="Custom real timesteps (single custom scenario)")
    p.add_argument("--n-syn", type=int, default=None, help="Custom synthetic timesteps")
    p.add_argument("--npy-dir", type=Path, default=DEFAULT_NPY_DIR)
    p.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN_DIR)
    p.add_argument("--mixed-dir", type=Path, default=DEFAULT_MIXED_DIR)
    p.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    p.add_argument("--ukdale-raw", type=Path, default=DEFAULT_UKDALE_RAW)
    p.add_argument("--alg1-dir", type=Path, default=DEFAULT_ALG1_CSV)
    p.add_argument("--real-csv", type=Path, default=None)
    p.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Unused (CLI compat). Geng shuffles window indices in DataProvider at train time.",
    )
    p.add_argument("--no-post-filter", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-verify", action="store_true", help="Skip prepare_all CSV checks")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.npy_dir = _resolve(args.npy_dir)
    args.gen_dir = _resolve(args.gen_dir)
    args.mixed_dir = _resolve(args.mixed_dir)
    args.train_root = _resolve(args.train_root)
    args.ukdale_raw = _resolve(args.ukdale_raw)
    args.alg1_dir = _resolve(args.alg1_dir)
    if args.real_csv is not None:
        args.real_csv = _resolve(args.real_csv)

    # One-click default: all appliances, all scenarios.
    if not args.all and args.appliance is None:
        args.all = True
    if args.scenario == "missing":
        args.skip_existing = True

    appliances: tuple[str, ...] = ALL_APPLIANCES if args.all else (args.appliance,)  # type: ignore[arg-type]

    if args.validation_percent + args.test_percent >= 100:
        raise ValueError("validation-percent + test-percent must be < 100")

    paper_exact = args.paper_exact or all_have_paper_exact_splits(args.train_root, appliances)

    if args.splits_only:
        scenarios_for_crops: tuple[GengMixScenario, ...] = GENG_MIX_SCENARIOS
    else:
        scenarios_for_crops = resolve_scenarios(args, appliances=appliances, train_root=args.train_root)

    print("Geng mix builder")
    print(f"  train root:  {_rel(args.train_root)}")
    print(f"  appliances:  {appliances}")
    if paper_exact:
        print("  mode: paper-exact (full timeline splits from prepare_all_ukdale.py)")
        print_paper_exact_status(args.train_root, appliances)
    elif args.splits_only:
        print("  mode: splits-only (from house pools)")
    else:
        print(f"  scenarios:   {[s.name for s in scenarios_for_crops]}")
    print()

    if not paper_exact:
        pool_missing = verify_pool_outputs(args.train_root, appliances)
        if pool_missing and not any(
            has_paper_exact_splits(args.train_root, app) for app in appliances
        ):
            raise FileNotFoundError(
                "No pool CSVs and no paper-exact training CSVs.\n"
                "  Paper: python NILM-main/dataset_preprocess/prepare_all_ukdale.py --paper-exact\n"
                "  Pool:  python NILM-main/dataset_preprocess/prepare_all_ukdale.py"
            )

        ensure_geng_splits(
            appliances,
            args.train_root,
            scenarios_for_crops,
            validation_percent=args.validation_percent,
            test_percent=args.test_percent,
            skip_house1=args.skip_house1,
            force=args.force_splits,
            dry_run=args.dry_run,
        )
        print()
    elif not all_have_paper_exact_splits(args.train_root, appliances):
        missing_apps = [a for a in appliances if not has_paper_exact_splits(args.train_root, a)]
        raise FileNotFoundError(
            "Paper-exact mode but missing split CSVs for: "
            + ", ".join(missing_apps)
            + "\n  Run: python NILM-main/dataset_preprocess/prepare_all_ukdale.py --paper-exact"
        )

    if args.splits_only:
        if not args.skip_verify:
            verify_prepare_all_outputs(args.train_root, appliances)
        print_experiment_summary(args.train_root, appliances)
        return

    scenarios = scenarios_for_crops
    if not scenarios:
        print_experiment_summary(args.train_root, appliances)
        return

    print("Geng mix builder (synthetic)")
    print(f"  npy dir:     {_rel(args.npy_dir)}")
    print(f"  scenarios:   {[s.name for s in scenarios]}")
    print()

    if not args.skip_verify:
        verify_prepare_all_outputs(args.train_root, appliances)

    max_syn = max(s.n_syn for s in scenarios)
    print(f"Step 1 - npy to watt timesteps (need >= {max_syn:,} per appliance)")
    series_map = build_all_watt_series(
        args.npy_dir,
        args.gen_dir,
        args.alg1_dir,
        post_filter=not args.no_post_filter,
    )
    print()

    manifests: list[dict] = []
    for scenario in scenarios:
        print(f"Step 2 - scenario {scenario.name} (real {scenario.n_real:,} + syn {scenario.n_syn:,})")
        for app in appliances:
            print(f"  [{app}]")
            meta = process_appliance_scenario(app, series_map, scenario, args)
            manifests.append(meta)
        print()

    manifest_path = args.gen_dir / "geng_mix_manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"Manifest: {_rel(manifest_path)}")
    print_experiment_summary(args.train_root, appliances)
    print("Visualize mix: python ../../../data/geng_mix_visualize.py <labeled_csv>")
    print("Train NILM (PyTorch): cd ../NILM-main_pytorch && python -m nilm_main_pytorch.train ...")


if __name__ == "__main__":
    main()
