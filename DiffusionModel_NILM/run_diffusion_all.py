"""
One-click train + sample for all five UK-DALE diffusion models.

Orchestrates the **original** Geng main.py (no changes to main.py / solver.py).
Default: train then sample every appliance that has Data/datasets/{app}.csv.

Usage:
  python run_diffusion_all.py
  python run_diffusion_all.py --no-train
  python run_diffusion_all.py --train --sample
  python run_diffusion_all.py --appliances kettle fridge --proportion 0.5
  python run_diffusion_all.py --plan-syn
"""

from __future__ import annotations

import argparse
import math
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
DEFAULT_SAMPLE_SIZE_EVERY = 400

GENG_MAX_SYN_TIMESTEPS = 200_000
INJECTION_200PCT_SYN_TIMESTEPS = 400_000
RECOMMENDED_SYN_BUFFER = 1.25


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


def print_device_info(gpu_id: int) -> None:
    print("Device")
    if not torch.cuda.is_available():
        print("  CUDA available:  False")
        print("  WARNING: No GPU detected — main.py will likely fail.")
        return

    n = torch.cuda.device_count()
    print("  CUDA available:  True")
    print(f"  CUDA version:    {torch.version.cuda}")
    print(f"  PyTorch:         {torch.__version__}")
    print(f"  GPU count:       {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        marker = "  <- will use" if i == gpu_id else ""
        print(f"    [{i}] {props.name}  ({props.total_memory / 1024**3:.1f} GB){marker}")

    if gpu_id < 0 or gpu_id >= n:
        raise SystemExit(f"ERROR: --gpu {gpu_id} is invalid (valid: 0..{n - 1})")

    torch.cuda.set_device(gpu_id)
    print(f"  Selected device: cuda:{gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
    print()


def config_path(appliance: str) -> Path:
    return CONFIG_DIR / f"{appliance}.yaml"


def dataset_path(appliance: str) -> Path:
    return DATA_DIR / f"{appliance}.csv"


def read_config(appliance: str) -> dict:
    return yaml.safe_load(config_path(appliance).read_text(encoding="utf-8"))


def read_window(appliance: str) -> int:
    cfg = read_config(appliance)
    return int(cfg["dataloader"]["train_dataset"]["params"].get("window", DEFAULT_SEQ_LENGTH))


def count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def diffusion_windows_from_rows(n_rows: int, window: int = DEFAULT_SEQ_LENGTH) -> int:
    return max(n_rows - window + 1, 0)


def non_overlap_windows(n_rows: int, window: int) -> int:
    return max(n_rows // window, 0)


def timesteps_from_windows(n_windows: int, window: int = DEFAULT_SEQ_LENGTH) -> int:
    return n_windows * window


def windows_for_timesteps(n_timesteps: int, window: int = DEFAULT_SEQ_LENGTH) -> int:
    return (n_timesteps + window - 1) // window


def print_synthetic_plan(appliances: tuple[str, ...], max_pct: float) -> None:
    if max_pct <= 100:
        target_syn = GENG_MAX_SYN_TIMESTEPS
        label = "Geng 200k+200k (100% syn vs 200k real)"
    else:
        target_syn = INJECTION_200PCT_SYN_TIMESTEPS
        label = f"injection {max_pct:.0f}% (2x syn vs 200k real timesteps)"

    target_with_buffer = int(target_syn * RECOMMENDED_SYN_BUFFER)
    min_windows = windows_for_timesteps(target_with_buffer)

    print("Synthetic data budget (per appliance, after npy -> watt CSV post-process)")
    print(f"  scenario:       {label}")
    print(f"  min syn rows:   {target_syn:,}  ->  store >= {target_with_buffer:,}")
    print(f"  min windows:    {min_windows:,}  (non-overlap @ 512)")
    print()
    print(
        f"{'appliance':<16} {'rows':>10} {'slide win':>10} {'non-ol win':>10} "
        f"{'orig ts':>12}"
    )
    print("-" * 64)

    for app in appliances:
        path = dataset_path(app)
        if not path.is_file():
            print(f"{app:<16} {'(missing)':>10}")
            continue
        window = read_window(app)
        n_rows = count_csv_rows(path)
        n_slide = diffusion_windows_from_rows(n_rows, window)
        n_non = non_overlap_windows(n_rows, window)
        orig_ts = timesteps_from_windows(n_slide, window)
        print(
            f"{app:<16} {n_rows:>10,} {n_slide:>10,} {n_non:>10,} {orig_ts:>12,}"
        )

    print()
    print(
        "Original main.py samples num=len(sliding_dataset) windows "
        f"(size_every={DEFAULT_SAMPLE_SIZE_EVERY}) -> [0,1] MinMax in ddpm_fake_{{app}}.npy"
    )


def checkpoint_dir(appliance: str) -> Path:
    cfg = read_config(appliance)
    base = cfg["solver"]["results_folder"]
    seq_len = int(cfg["model"]["params"]["seq_length"])
    folder = SCRIPT_DIR / f"{base}_{seq_len}"
    return folder


def latest_milestone(appliance: str) -> int:
    folder = checkpoint_dir(appliance)
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


def resolve_sample_milestone(appliance: str, milestone_arg: str) -> int:
    if milestone_arg.lower() == "latest":
        return latest_milestone(appliance)
    return int(milestone_arg)


def run_main(
    appliance: str,
    gpu: int,
    *,
    train: bool,
    milestone: int | None,
    proportion: float,
    tensorboard: bool,
    extra_opts: list[str],
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

    if tensorboard:
        cmd.append("--tensorboard")

    if train:
        cmd.append("--train")
        cmd.extend(
            [
                "dataloader.train_dataset.params.save2npy",
                "False",
                "dataloader.train_dataset.params.proportion",
                str(proportion),
            ]
        )
    else:
        if milestone is None:
            milestone = latest_milestone(appliance)
        cmd.extend(["--milestone", str(milestone)])

    cmd.extend(extra_opts)

    print(f"\n{'=' * 60}")
    print(f"{'TRAIN' if train else 'SAMPLE'}: {appliance}")
    print(f"  command: {' '.join(cmd)}")
    if not train:
        n_rows = count_csv_rows(dataset_path(appliance))
        window = read_window(appliance)
        n_windows = diffusion_windows_from_rows(n_rows, window)
        est_batches = math.ceil(n_windows / DEFAULT_SAMPLE_SIZE_EVERY)
        print(
            f"  original sampling: {n_windows:,} windows "
            f"(~{est_batches:,} batches @ {DEFAULT_SAMPLE_SIZE_EVERY})"
        )
    print(f"{'=' * 60}\n")
    subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=True)

    if not train:
        out = OUTPUT_DIR / appliance / f"ddpm_fake_{appliance}.npy"
        if out.is_file():
            arr_info = ""
            try:
                import numpy as np

                arr = np.load(out)
                arr_info = f"  shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]"
            except Exception:
                pass
            print(f"  OK: {_rel(out)}{arr_info}")
        else:
            print(f"  WARNING: expected output not found: {_rel(out)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train/sample diffusion models for all appliances (original main.py)"
    )
    p.add_argument("--train", action="store_true", help="Train all selected appliances")
    p.add_argument("--sample", action="store_true", help="Sample from trained checkpoints")
    p.add_argument("--no-train", action="store_true", help="Sample only (skip training)")
    p.add_argument("--plan-syn", action="store_true", help="Print synthetic budget and exit")
    p.add_argument(
        "--max-injection-pct",
        type=float,
        default=200.0,
        help="Max injection %% for --plan-syn",
    )
    p.add_argument(
        "--proportion",
        type=float,
        default=1.0,
        help="Training data proportion (1.0=full CSV; lower if RAM limited)",
    )
    p.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable tensorboard logging during training",
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
        help="Checkpoint id for sampling (e.g. 10) or 'latest'",
    )
    p.add_argument(
        "--skip_missing_data",
        action="store_true",
        help="Skip appliances without Data/datasets/{app}.csv",
    )
    p.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Extra config overrides passed to main.py (key value pairs)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    extra_opts = [x for x in args.opts if x != "--"]

    if args.plan_syn:
        print_synthetic_plan(tuple(args.appliances), args.max_injection_pct)
        return

    do_train = args.train or (not args.train and not args.sample and not args.no_train)
    do_sample = args.sample or (not args.train and not args.sample and not args.no_train)
    if args.no_train:
        do_train = False
        do_sample = True

    print("Diffusion batch runner (original main.py — no solver changes)")
    print(f"  script dir:   {SCRIPT_DIR}")
    print(f"  appliances:   {args.appliances}")
    if do_train:
        print(
            f"  train:        max_epochs={DEFAULT_MAX_EPOCHS}, "
            f"save_cycle={DEFAULT_SAVE_CYCLE}, proportion={args.proportion}"
        )
    if do_sample:
        print(f"  sample:       milestone={args.milestone}")
        print(
            f"  note:         uses len(sliding_dataset) windows, "
            f"size_every={DEFAULT_SAMPLE_SIZE_EVERY} (original Geng behavior)"
        )
    print()
    print_device_info(args.gpu)

    for app in args.appliances:
        if not config_path(app).is_file():
            raise FileNotFoundError(f"Missing config: {_rel(config_path(app))}")
        if not dataset_path(app).is_file():
            msg = f"Missing training data: {_rel(dataset_path(app))} — run algorithm1.py first"
            if args.skip_missing_data:
                print(f"  SKIP {app}: {msg}")
            else:
                raise FileNotFoundError(msg)

    appliances = [app for app in args.appliances if dataset_path(app).is_file()]

    if do_train:
        for app in appliances:
            run_main(
                app,
                args.gpu,
                train=True,
                milestone=None,
                proportion=args.proportion,
                tensorboard=args.tensorboard,
                extra_opts=extra_opts,
            )

    if do_sample:
        for app in appliances:
            milestone = resolve_sample_milestone(app, args.milestone)
            print(f"  {app}: checkpoint milestone {milestone}")
            run_main(
                app,
                args.gpu,
                train=False,
                milestone=milestone,
                proportion=args.proportion,
                tensorboard=False,
                extra_opts=extra_opts,
            )

    print("\nAll jobs finished.")
    if do_sample:
        print(f"Outputs: {_rel(OUTPUT_DIR)}/{{appliance}}/ddpm_fake_{{appliance}}.npy")
        print()
        print_synthetic_plan(tuple(appliances), args.max_injection_pct)


if __name__ == "__main__":
    main()
