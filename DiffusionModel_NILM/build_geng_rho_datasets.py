"""
ICSIMA / thesis — Geng-style injection-ratio datasets (D4 / full-window append).

Builds training CSVs for rho in {0, 25, 50, 100, 200}% using the same method as Geng et al.:

  1. Synthetic Y: ddpm_fake_{app}.npy -> watt timesteps (per appliance)
  2. Synthetic X: sum of all 5 appliance synthetic watt series (aligned length)
  3. Real block: first n_real rows from origin training crop (watts)
  4. Mix: concat [real block | synthetic block]  (no shuffle; trainer shuffles windows)
  5. Z-score both columns (Geng aggregate + appliance stats) -> 2-col NILM CSV

Injection ratio (paper definition):

  rho = |D_s| / |D_r|  =>  n_syn = round(rho / 100 * n_real)

Prerequisites (same as build_geng_mix.py):
  python NILM-main/dataset_preprocess/prepare_all_ukdale.py [--paper-exact]
  python algorithm1.py
  python run_diffusion_all.py

Usage:
  cd DiffusionModel_NILM
  python build_geng_rho_datasets.py
  python build_geng_rho_datasets.py --rho 0 25 50 100 200 --n-real 100000
  python build_geng_rho_datasets.py --appliance washingmachine --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "NILM-main" / "dataset_preprocess"))

from prepare_all_ukdale import crop_label  # noqa: E402

from build_geng_mix import (  # noqa: E402
    ALL_APPLIANCES,
    DEFAULT_ALG1_CSV,
    DEFAULT_GEN_DIR,
    DEFAULT_MIXED_DIR,
    DEFAULT_NPY_DIR,
    DEFAULT_TRAIN_ROOT,
    DEFAULT_UKDALE_RAW,
    GengMixScenario,
    _rel,
    _resolve,
    build_all_watt_series,
    build_synthetic_xy_watts,
    geng_mix_labeled,
    load_real_watts,
    save_geng_nilm_csv,
    save_labeled_mix_csv,
    zscore_rows_to_watts,
)

DEFAULT_RHO_PCTS: tuple[int, ...] = (0, 25, 50, 100, 200)


@dataclass(frozen=True)
class RhoScenario:
    """One injection-ratio training set."""

    rho_pct: int
    n_real: int

    @property
    def n_syn(self) -> int:
        return int(round(self.rho_pct / 100.0 * self.n_real))

    @property
    def name(self) -> str:
        return f"rho{self.rho_pct}"

    @property
    def combined_csv(self) -> str:
        return f"UK_DALECombined{{app}}_rho{self.rho_pct}.csv"

    @property
    def labeled_csv(self) -> str:
        return f"UK_DALECombined{{app}}_rho{self.rho_pct}_labeled.csv"

    def geng_mix_scenario(self) -> GengMixScenario:
        """Adapter for load_real_watts() origin crop label."""
        return GengMixScenario(
            name=self.name,
            n_real=self.n_real,
            n_syn=self.n_syn,
            mix_file_label=f"rho{self.rho_pct}",
        )


def combined_path(train_root: Path, appliance: str, scenario: RhoScenario) -> Path:
    return train_root / appliance / scenario.combined_csv.format(app=appliance)


def labeled_path(train_root: Path, appliance: str, scenario: RhoScenario) -> Path:
    return train_root / appliance / scenario.labeled_csv.format(app=appliance)


def max_syn_timesteps(n_real: int, rho_pcts: tuple[int, ...]) -> int:
    return max(int(round(r / 100.0 * n_real)) for r in rho_pcts)


def load_real_watts_for_rho(
    appliance: str,
    n_real: int,
    train_root: Path,
    ukdale_raw: Path,
    real_csv: Path | None,
    geng_scenario: GengMixScenario,
) -> pd.DataFrame:
    """Load n_real real watts; fall back to full {app}_training_.csv if crop missing."""
    try:
        return load_real_watts(appliance, n_real, train_root, ukdale_raw, real_csv, geng_scenario)
    except FileNotFoundError:
        pass

    full_train = train_root / appliance / f"{appliance}_training_.csv"
    if full_train.is_file():
        print(f"  real: {_rel(full_train)} (crop missing; denorm first {n_real:,} rows)")
        df_z = pd.read_csv(full_train, header=None)
        return zscore_rows_to_watts(df_z, appliance).iloc[:n_real].copy()

    raise FileNotFoundError(
        f"Missing real training data for {appliance}. Run prepare_all_ukdale.py, expected "
        f"{appliance}_{crop_label(n_real)}training_.csv or {appliance}_training_.csv under {_rel(train_root / appliance)}"
    )


def build_one_appliance_rho(
    appliance: str,
    scenario: RhoScenario,
    series_map: dict,
    *,
    train_root: Path,
    ukdale_raw: Path,
    gen_dir: Path,
    real_csv: Path | None,
    dry_run: bool,
    skip_existing: bool,
) -> dict:
    n_real = scenario.n_real
    n_syn = scenario.n_syn
    out_path = combined_path(train_root, appliance, scenario)
    lab_path = labeled_path(train_root, appliance, scenario)
    geng_scenario = scenario.geng_mix_scenario()

    if skip_existing and out_path.is_file():
        print(f"  skip existing: {_rel(out_path)}")
        return {
            "appliance": appliance,
            "rho_pct": scenario.rho_pct,
            "n_real": n_real,
            "n_syn": n_syn,
            "n_mix": None,
            "training_csv": _rel(out_path),
            "labeled_csv": _rel(lab_path),
            "skipped": True,
        }

    real_watts = load_real_watts_for_rho(
        appliance,
        n_real,
        train_root,
        ukdale_raw,
        real_csv,
        geng_scenario,
    )
    if len(real_watts) < n_real:
        raise ValueError(f"{appliance} rho{scenario.rho_pct}: need {n_real} real rows, got {len(real_watts)}")

    real_block = real_watts.iloc[:n_real].copy()

    if n_syn == 0:
        mixed_watts = real_block
        labeled = real_block.copy()
        labeled["source"] = 0
        print(f"  {appliance} rho=0%: real only ({n_real:,} rows)")
    else:
        syn_watts = build_synthetic_xy_watts(appliance, series_map, n_syn)
        syn_only_path = gen_dir / f"synthetic_{appliance}_rho{scenario.rho_pct}_{n_syn}.csv"
        if not dry_run:
            syn_watts.to_csv(syn_only_path, index=False)
        print(
            f"  {appliance} rho={scenario.rho_pct}%: "
            f"real={n_real:,} + syn={n_syn:,} | X_syn=sum(5 apps), Y_syn={appliance}"
        )
        print(f"  synthetic XY: {_rel(syn_only_path)}")
        mixed_watts, labeled = geng_mix_labeled(real_block, syn_watts, appliance)

    if dry_run:
        print(f"  [dry-run] would write {_rel(out_path)} ({len(mixed_watts):,} rows)")
        print(f"  [dry-run] would write {_rel(lab_path)}")
    else:
        save_geng_nilm_csv(mixed_watts, appliance, out_path)
        save_labeled_mix_csv(labeled, appliance, lab_path)

    return {
        "appliance": appliance,
        "rho_pct": scenario.rho_pct,
        "n_real": n_real,
        "n_syn": n_syn,
        "n_mix": len(mixed_watts),
        "training_csv": _rel(out_path),
        "labeled_csv": _rel(lab_path),
        "skipped": False,
    }


def print_training_map(train_root: Path, rho_pcts: tuple[int, ...], n_real: int) -> None:
    print("=" * 72)
    print("Geng rho experiment — training CSV map")
    print(f"  n_real = {n_real:,}  |  rho = |D_s|/|D_r|  |  concat: real then synthetic")
    print("=" * 72)
    print(f"{'rho':<8} {'n_syn':<12} {'Training CSV'}")
    print("-" * 72)
    for rho in rho_pcts:
        n_syn = int(round(rho / 100.0 * n_real))
        print(f"{rho}%{'':<5} {n_syn:<12,} UK_DALECombined{{app}}_rho{rho}.csv")
    print()
    print(f"Val/test (unchanged): {{app}}_validation_.csv, {{app}}_test_.csv")
    print(f"Folder: {_rel(train_root)}/{{app}}/")
    print("Visualize: python ../data/geng_mix_visualize.py <labeled_csv>")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Geng-style NILM training CSVs for injection ratios rho (%%)."
    )
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
        default=100_000,
        help="Real training timesteps |D_r| (default: 100000 = Geng 10%% crop base)",
    )
    p.add_argument("--appliance", choices=list(ALL_APPLIANCES), default=None)
    p.add_argument("--npy-dir", type=Path, default=DEFAULT_NPY_DIR)
    p.add_argument("--alg1-dir", type=Path, default=DEFAULT_ALG1_CSV)
    p.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN_DIR)
    p.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    p.add_argument("--ukdale-raw", type=Path, default=DEFAULT_UKDALE_RAW)
    p.add_argument("--real-csv", type=Path, default=None, help="Optional real watts CSV override")
    p.add_argument("--no-post-filter", action="store_true", help="Disable ON-threshold post-filter on npy")
    p.add_argument("--skip-existing", action="store_true", help="Do not overwrite existing rho CSVs")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.npy_dir = _resolve(args.npy_dir)
    args.alg1_dir = _resolve(args.alg1_dir)
    args.gen_dir = _resolve(args.gen_dir)
    args.train_root = _resolve(args.train_root)
    args.ukdale_raw = _resolve(args.ukdale_raw)
    if args.real_csv is not None:
        args.real_csv = _resolve(args.real_csv)

    rho_pcts = tuple(sorted(set(int(r) for r in args.rho), key=lambda x: x))
    if any(r < 0 for r in rho_pcts):
        raise ValueError("rho must be >= 0")
    n_real = int(args.n_real)
    appliances = (args.appliance,) if args.appliance else ALL_APPLIANCES
    scenarios = tuple(RhoScenario(rho_pct=r, n_real=n_real) for r in rho_pcts)
    max_syn = max_syn_timesteps(n_real, rho_pcts)

    print("Geng rho dataset builder (ICSIMA table)")
    print(f"  train root:  {_rel(args.train_root)}")
    print(f"  npy dir:     {_rel(args.npy_dir)}")
    print(f"  n_real:      {n_real:,}")
    print(f"  rho values:  {', '.join(f'{r}%' for r in rho_pcts)}")
    print(f"  max n_syn:   {max_syn:,} (need this many synthetic timesteps per appliance)")
    print()

    if max_syn > 0 and not args.dry_run:
        print("Step 1 — npy -> watt timesteps (all appliances)")
        series_map = build_all_watt_series(
            args.npy_dir,
            args.gen_dir,
            args.alg1_dir,
            post_filter=not args.no_post_filter,
        )
        shortest = min(len(series_map[a]) for a in ALL_APPLIANCES)
        if shortest < max_syn:
            raise ValueError(
                f"Shortest synthetic series has {shortest:,} timesteps; need {max_syn:,} for rho={max(rho_pcts)}%. "
                "Re-run diffusion sampling with more windows."
            )
        print()
    elif max_syn > 0:
        print("Step 1 — [dry-run] would load npy -> watt series")
        series_map = {}
    else:
        print("Step 1 — rho=0 only; skip synthetic npy load")
        series_map = {}

    print("Step 2 — build per-appliance rho CSVs")
    entries: list[dict] = []
    for app in appliances:
        print(f"\n[{app}]")
        for scenario in scenarios:
            if scenario.n_syn > 0 and not series_map:
                raise RuntimeError("series_map missing; cannot build synthetic block")
            row = build_one_appliance_rho(
                app,
                scenario,
                series_map,
                train_root=args.train_root,
                ukdale_raw=args.ukdale_raw,
                gen_dir=args.gen_dir,
                real_csv=args.real_csv,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
            )
            entries.append(row)

    manifest = {
        "method": "geng_timestep_concat",
        "description": "X_syn=sum(5 synthetic apps), Y_syn=target app; concat real then syn; z-score",
        "rho_definition": "rho_pct = 100 * |D_s| / |D_r|",
        "n_real": n_real,
        "rho_pcts": list(rho_pcts),
        "appliances": list(appliances),
        "train_root": _rel(args.train_root),
        "entries": entries,
    }
    manifest_path = args.gen_dir / "geng_rho_manifest.json"
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nmanifest: {_rel(manifest_path)}")

    print()
    print_training_map(args.train_root, rho_pcts, n_real)


if __name__ == "__main__":
    main()
