"""
Train and/or sample diffusion models for all five NILM appliances.

Runs main.py sequentially per appliance (GPU memory). Expects Algorithm 1
outputs at Data/datasets/{appliance}.csv and Config/{appliance}.yaml.

Usage (from any directory):
  python run_diffusion_all.py              # train then sample (default)
  python run_diffusion_all.py --train      # train only
  python run_diffusion_all.py --sample     # sample only
  python run_diffusion_all.py --plan-syn   # print synthetic timestep budget, exit
  python run_diffusion_all.py --sample --milestone 10
  python run_diffusion_all.py --gpu 0 --appliances microwave kettle

Paper defaults: 20k epochs, checkpoint every 2000 steps → milestones 1..10.
Use --milestone latest (default for sampling) to load the newest checkpoint.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_PY = SCRIPT_DIR / "main.py"
DATA_DIR = SCRIPT_DIR / "Data" / "datasets"
CONFIG_DIR = SCRIPT_DIR / "Config"
OUTPUT_DIR = SCRIPT_DIR / "OUTPUT"

ALL_APPLIANCES = ("kettle", "microwave", "fridge", "dishwasher", "washingmachine")
DEFAULT_SEQ_LENGTH = 512
DEFAULT_SAVE_CYCLE = 2000
DEFAULT_MAX_EPOCHS = 20000

# Geng paper max mix is 200k+200k; 200% injection (D3 ratio 2.0) on 200k real → 400k syn.
GENG_MAX_SYN_TIMESTEPS = 200_000
INJECTION_200PCT_SYN_TIMESTEPS = 400_000
RECOMMENDED_SYN_BUFFER = 1.25  # keep 25% headroom after post-filter / alignment


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


def print_device_info(gpu_id: int) -> None:
    """Print CUDA / device info before launching main.py subprocesses."""
    print("Device")
    if not torch.cuda.is_available():
        print("  CUDA available:  False")
        print("  WARNING: No GPU detected — main.py will likely fail (model uses .cuda()).")
        print(f"  Requested --gpu {gpu_id} cannot be used.")
        return

    n = torch.cuda.device_count()
    print(f"  CUDA available:  True")
    print(f"  CUDA version:    {torch.version.cuda}")
    print(f"  PyTorch:         {torch.__version__}")
    print(f"  GPU count:       {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        marker = "  ← will use" if i == gpu_id else ""
        print(
            f"    [{i}] {props.name}  "
            f"({props.total_memory / 1024**3:.1f} GB){marker}"
        )

    if gpu_id < 0 or gpu_id >= n:
        print(f"  ERROR: --gpu {gpu_id} is invalid (valid: 0..{n - 1})")
        raise SystemExit(1)

    torch.cuda.set_device(gpu_id)
    print(f"  Selected device: cuda:{gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
    print()


def config_path(appliance: str) -> Path:
    return CONFIG_DIR / f"{appliance}.yaml"


def dataset_path(appliance: str) -> Path:
    return DATA_DIR / f"{appliance}.csv"


def count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def diffusion_windows_from_rows(n_rows: int, window: int = DEFAULT_SEQ_LENGTH) -> int:
    return max(n_rows - window + 1, 0)


def timesteps_from_windows(n_windows: int, window: int = DEFAULT_SEQ_LENGTH) -> int:
    """Non-overlapping stitch of sampled windows (stride = window)."""
    return n_windows * window


def windows_for_timesteps(n_timesteps: int, window: int = DEFAULT_SEQ_LENGTH) -> int:
    return (n_timesteps + window - 1) // window


def print_synthetic_plan(appliances: tuple[str, ...], max_pct: float) -> None:
    """Print how many diffusion windows / timesteps to keep for future mixing."""
    if max_pct <= 100:
        target_syn = GENG_MAX_SYN_TIMESTEPS
        label = "Geng 200k+200k (100% syn vs 200k real)"
    else:
        target_syn = INJECTION_200PCT_SYN_TIMESTEPS
        label = f"injection {max_pct:.0f}% (2x syn vs 200k real timesteps)"

    target_with_buffer = int(target_syn * RECOMMENDED_SYN_BUFFER)
    min_windows = windows_for_timesteps(target_with_buffer)

    print("Synthetic data budget (per appliance, after npy -> watt CSV post-process)")
    print(f"  scenario:     {label}")
    print(f"  min syn rows: {target_syn:,}  ->  store >= {target_with_buffer:,} with 25% buffer")
    print(f"  min windows:  {min_windows:,}  (non-overlap @ {DEFAULT_SEQ_LENGTH} steps/window)")
    print()
    print(f"{'appliance':<16} {'alg1 rows':>10} {'ddpm windows':>12} {'non-overlap ts':>16} {'>= target?':>10}")
    print("-" * 70)

    bottleneck_ts = None
    for app in appliances:
        path = dataset_path(app)
        if not path.is_file():
            print(f"{app:<16} {'(missing)':>10}")
            continue
        n_rows = count_csv_rows(path)
        n_win = diffusion_windows_from_rows(n_rows)
        n_ts = timesteps_from_windows(n_win)
        ok = "yes" if n_ts >= target_with_buffer else "NO"
        print(f"{app:<16} {n_rows:>10,} {n_win:>12,} {n_ts:>16,} {ok:>10}")
        bottleneck_ts = n_ts if bottleneck_ts is None else min(bottleneck_ts, n_ts)

    print()
    if bottleneck_ts is not None:
        print(
            f"Default sampling uses len(dataset) windows per appliance — "
            f"bottleneck non-overlap timesteps ~ {bottleneck_ts:,}."
        )
        if bottleneck_ts >= target_with_buffer:
            print("You already have enough raw ddpm windows; focus on post-process to >= target CSV rows.")
        else:
            shortfall = target_with_buffer - bottleneck_ts
            extra_windows = windows_for_timesteps(shortfall)
            print(
                f"Shortfall ~ {shortfall:,} timesteps - increase sampling by ~{extra_windows:,} "
                "windows for the bottleneck appliance."
            )
    print()
    print("For D4 sum-of-5 aggregate: all five {app}_synthetic.csv must have the same length >= target.")


def checkpoint_dir(appliance: str, seq_length: int = DEFAULT_SEQ_LENGTH) -> Path:
    return SCRIPT_DIR / ".Checkpoints" / f"Checkpoints_{appliance}_{seq_length}"


def read_seq_length(appliance: str) -> int:
    cfg = yaml.safe_load(config_path(appliance).read_text(encoding="utf-8"))
    return int(cfg["model"]["params"]["seq_length"])


def latest_milestone(appliance: str) -> int:
    """Highest checkpoint-N.pt in the appliance results folder."""
    seq_len = read_seq_length(appliance)
    folder = checkpoint_dir(appliance, seq_len)
    if not folder.is_dir():
        raise FileNotFoundError(f"No checkpoints in {_rel(folder)} — train first.")
    milestones = []
    for p in folder.glob("checkpoint-*.pt"):
        m = re.search(r"checkpoint-(\d+)\.pt$", p.name)
        if m:
            milestones.append(int(m.group(1)))
    if not milestones:
        raise FileNotFoundError(f"No checkpoint-*.pt files in {_rel(folder)}")
    return max(milestones)


def run_main(
    appliance: str,
    gpu: int,
    train: bool,
    milestone: int | None,
    extra_args: list[str],
) -> None:
    cmd = [
        sys.executable,
        str(MAIN_PY),
        "--name",
        appliance,
        "--config",
        str(config_path(appliance)),
        "--gpu",
        str(gpu),
        "--output",
        str(OUTPUT_DIR),
    ]
    if train:
        cmd.append("--train")
    else:
        if milestone is None:
            milestone = latest_milestone(appliance)
        cmd.extend(["--sample", "0", "--milestone", str(milestone)])
    cmd.extend(extra_args)

    print(f"\n{'=' * 60}")
    print(f"{'TRAIN' if train else 'SAMPLE'}: {appliance}")
    print(f"  command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/sample diffusion models for all appliances")
    p.add_argument("--train", action="store_true", help="Train all selected appliances")
    p.add_argument("--sample", action="store_true", help="Sample from trained checkpoints")
    p.add_argument(
        "--no-train",
        action="store_true",
        help="With default run: sample only (skip training)",
    )
    p.add_argument(
        "--plan-syn",
        action="store_true",
        help="Print synthetic timestep budget for mixing experiments and exit",
    )
    p.add_argument(
        "--max-injection-pct",
        type=float,
        default=200.0,
        help="Max injection %% for --plan-syn (100=Geng 200k+200k, 200=D3 2× syn)",
    )
    p.add_argument(
        "--appliances",
        nargs="+",
        choices=ALL_APPLIANCES,
        default=list(ALL_APPLIANCES),
    )
    p.add_argument("--gpu", type=int, default=0, help="CUDA device id")
    p.add_argument(
        "--milestone",
        type=str,
        default="latest",
        help="Checkpoint id for sampling (e.g. 10) or 'latest' (default)",
    )
    p.add_argument(
        "--skip_missing_data",
        action="store_true",
        help="Skip appliances without Data/datasets/{app}.csv",
    )
    p.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Extra args passed to main.py (e.g. --seed 2024)",
    )
    return p.parse_args()


def resolve_sample_milestone(appliance: str, milestone_arg: str) -> int:
    if milestone_arg.lower() == "latest":
        return latest_milestone(appliance)
    return int(milestone_arg)


def main() -> None:
    args = parse_args()

    if args.plan_syn:
        print_synthetic_plan(tuple(args.appliances), args.max_injection_pct)
        return

    do_train = args.train or (not args.train and not args.sample and not args.no_train)
    do_sample = args.sample or (not args.train and not args.sample and not args.no_train)
    if args.no_train and not args.sample and not args.train:
        do_train = False
        do_sample = True

    print("Diffusion batch runner")
    print(f"  script dir: {SCRIPT_DIR}")
    print(f"  appliances: {args.appliances}")
    if do_train:
        print(f"  train:      max_epochs={DEFAULT_MAX_EPOCHS}, save every {DEFAULT_SAVE_CYCLE} steps")
    if do_sample:
        print(f"  sample:     milestone={args.milestone}")
    print()
    print_device_info(args.gpu)

    # Preflight
    for app in args.appliances:
        if not config_path(app).is_file():
            raise FileNotFoundError(f"Missing config: {_rel(config_path(app))}")
        if not dataset_path(app).is_file():
            msg = f"Missing training data: {_rel(dataset_path(app))} — run algorithm1.py first"
            if args.skip_missing_data:
                print(f"  SKIP {app}: {msg}")
            else:
                raise FileNotFoundError(msg)

    appliances = [
        app
        for app in args.appliances
        if dataset_path(app).is_file() or not args.skip_missing_data
    ]
    appliances = [app for app in appliances if dataset_path(app).is_file()]

    if do_train:
        for app in appliances:
            run_main(app, args.gpu, train=True, milestone=None, extra_args=args.opts)

    if do_sample:
        for app in appliances:
            milestone = resolve_sample_milestone(app, args.milestone)
            print(f"  {app}: using checkpoint milestone {milestone}")
            run_main(app, args.gpu, train=False, milestone=milestone, extra_args=args.opts)

    print("\nAll jobs finished.")
    if do_sample:
        print(f"Samples saved under {_rel(OUTPUT_DIR)}/{{appliance}}/ddpm_fake_{{appliance}}.npy")
        print()
        print_synthetic_plan(tuple(appliances), args.max_injection_pct)


if __name__ == "__main__":
    main()
