"""
Train and/or sample diffusion models for all five NILM appliances.

Runs main.py sequentially per appliance (GPU memory). Expects Algorithm 1
outputs at Data/datasets/{appliance}.csv and Config/{appliance}.yaml.

Usage (from any directory):
  python run_diffusion_all.py --train
  python run_diffusion_all.py --sample
  python run_diffusion_all.py --train --sample
  python run_diffusion_all.py --sample --milestone 10
  python run_diffusion_all.py --train --gpu 0 --appliances microwave kettle

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
    if not args.train and not args.sample:
        raise SystemExit("Specify --train and/or --sample")

    print("Diffusion batch runner")
    print(f"  script dir: {SCRIPT_DIR}")
    print(f"  appliances: {args.appliances}")
    if args.train:
        print(f"  train:      max_epochs={DEFAULT_MAX_EPOCHS}, save every {DEFAULT_SAVE_CYCLE} steps")
    if args.sample:
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

    if args.train:
        for app in appliances:
            run_main(app, args.gpu, train=True, milestone=None, extra_args=args.opts)

    if args.sample:
        for app in appliances:
            milestone = resolve_sample_milestone(app, args.milestone)
            print(f"  {app}: using checkpoint milestone {milestone}")
            run_main(app, args.gpu, train=False, milestone=milestone, extra_args=args.opts)

    print("\nAll jobs finished.")
    if args.sample:
        print(f"Samples saved under {_rel(OUTPUT_DIR)}/{{appliance}}/ddpm_fake_{{appliance}}.npy")


if __name__ == "__main__":
    main()
