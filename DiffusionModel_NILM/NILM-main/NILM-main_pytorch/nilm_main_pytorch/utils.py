"""Shared utilities for NILM-main PyTorch package."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
PYTORCH_ROOT = PACKAGE_DIR.parent
NILM_MAIN_DIR = PYTORCH_ROOT.parent
DIFFUSION_DIR = NILM_MAIN_DIR.parent
REPO_ROOT = DIFFUSION_DIR.parent
DEFAULT_CONFIG = PYTORCH_ROOT / "config" / "default.yaml"

# Relative to NILM-main_pytorch/ (same as config/default.yaml)
DEFAULT_REL_DATA_ROOT = Path("../dataset_preprocess/created_data/UK_DALE")


def portable_path_str(path: str | Path) -> str:
    """Show repo-relative paths in logs/errors (works on any machine)."""
    p = Path(path).resolve()
    for base in (PYTORCH_ROOT, NILM_MAIN_DIR, DIFFUSION_DIR, REPO_ROOT):
        try:
            return str(p.relative_to(base))
        except ValueError:
            continue
    return str(p)


def resolve_path(path_str: str | Path, *, anchor: Path | None = None) -> Path:
    """Resolve a config path relative to the repo (not the shell cwd)."""
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()

    bases: list[Path] = []
    if anchor is not None:
        bases.append(anchor)
    bases.extend([PYTORCH_ROOT, NILM_MAIN_DIR, DIFFUSION_DIR, REPO_ROOT])

    seen: set[Path] = set()
    for base in bases:
        if base in seen:
            continue
        seen.add(base)
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate

    fallback = anchor or PYTORCH_ROOT
    return (fallback / p).resolve()


def data_root_path(cfg: dict) -> Path:
    """UK-DALE CSV folder — always resolved from repo layout."""
    raw = cfg.get("data", {}).get("data_root", DEFAULT_REL_DATA_ROOT)
    return resolve_path(raw, anchor=NILM_MAIN_DIR)


def checkpoint_dir_path(cfg: dict) -> Path:
    return resolve_path(cfg["paths"]["checkpoint_dir"], anchor=PYTORCH_ROOT)


def results_dir_path(cfg: dict) -> Path:
    return resolve_path(cfg["paths"]["results_dir"], anchor=PYTORCH_ROOT)


def to_config_path(path: Path, anchor: Path = PYTORCH_ROOT) -> str:
    """Express path relative to NILM-main_pytorch/ (portable yaml/checkpoints)."""
    p = path.resolve()
    a = anchor.resolve()
    try:
        return str(p.relative_to(a)).replace("\\", "/")
    except ValueError:
        pass
    # data_root lives under NILM-main/ (sibling of NILM-main_pytorch/)
    try:
        rel = p.relative_to(NILM_MAIN_DIR.resolve())
        return str(Path("..") / rel).replace("\\", "/")
    except ValueError:
        pass
    return portable_path_str(p)


def relativize_config(cfg: dict) -> dict:
    """Store relative paths in checkpoints so models work on another PC / drive."""
    out = copy.deepcopy(cfg)
    data = out.get("data", {})
    if "data_root" in data:
        data["data_root"] = to_config_path(data_root_path(out))
    paths = out.get("paths", {})
    if "checkpoint_dir" in paths:
        paths["checkpoint_dir"] = to_config_path(checkpoint_dir_path(out))
    if "results_dir" in paths:
        paths["results_dir"] = to_config_path(results_dir_path(out))
    return out


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_device(device: torch.device) -> str:
    """Human-readable device line for logs."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        name = torch.cuda.get_device_name(idx)
        mem_gb = props.total_memory / (1024**3)
        return f"cuda:{idx} — {name} ({mem_gb:.1f} GiB)"
    return "cpu"


def get_device(cfg: dict, *, require_cuda: bool | None = None) -> torch.device:
    """
    Pick compute device from config.

    training.device: auto | cuda | cpu  (default: auto)
    training.gpu_id: GPU index when using CUDA (default: 0)
    training.require_cuda: if true, error when CUDA unavailable (default: true)
    """
    train_cfg = cfg.get("training", {})
    pref = str(train_cfg.get("device", "auto")).lower().strip()
    want_cuda = pref in ("auto", "cuda", "gpu")
    must_cuda = bool(train_cfg.get("require_cuda", True)) if require_cuda is None else require_cuda

    if pref == "cpu":
        return torch.device("cpu")

    if want_cuda and torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        gpu_id = int(train_cfg.get("gpu_id", 0))
        if gpu_id < 0 or gpu_id >= n_gpu:
            print(f"WARNING: gpu_id {gpu_id} invalid ({n_gpu} GPU(s)); using cuda:0")
            gpu_id = 0
        return torch.device(f"cuda:{gpu_id}")

    if must_cuda or pref in ("cuda", "gpu"):
        raise RuntimeError(
            "GPU (CUDA) is required but not available.\n"
            f"  torch.cuda.is_available() = {torch.cuda.is_available()}\n"
            f"  torch.version.cuda          = {torch.version.cuda}\n"
            "Fix: install a CUDA build of PyTorch in your conda env, or set\n"
            "  training.device: cpu   in config/default.yaml\n"
            "  --cpu                  on the command line"
        )

    return torch.device("cpu")


def log_device(cfg: dict, *, prefix: str = "Device") -> torch.device:
    """Resolve device, print it, and return it."""
    device = get_device(cfg)
    print(f"{prefix}: {describe_device(device)}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    return device


def gpu_stats(device: torch.device) -> str:
    if device.type != "cuda":
        return "device=cpu"
    idx = device.index if device.index is not None else 0
    alloc = torch.cuda.memory_allocated(idx) / (1024**3)
    peak = torch.cuda.max_memory_allocated(idx) / (1024**3)
    return f"GPU {alloc:.2f} GiB used, {peak:.2f} GiB peak"


def norm_stats(appliance: str) -> dict:
    from nilm_main_pytorch.data.paths import AGGREGATE_MEAN, AGGREGATE_STD
    from nilm_main_pytorch.models.params import PARAMS_APPLIANCE

    p = PARAMS_APPLIANCE[appliance]
    return {
        "appliance_mean": float(p["mean"]),
        "appliance_std": float(p["std"]),
        "aggregate_mean": AGGREGATE_MEAN,
        "aggregate_std": AGGREGATE_STD,
    }


def experiment_run_suffix(cfg: dict, *, augmented: bool | None = None) -> str:
    """Checkpoint / figure suffix for Geng reproduction or rho injection runs."""
    rho = cfg.get("data", {}).get("injection_rho")
    if rho is not None:
        return f"rho{int(rho)}"
    is_aug = bool(cfg["data"]["augmented"]) if augmented is None else augmented
    pct = str(cfg["data"]["train_percent"])
    tag = "aug" if is_aug else "origin"
    return f"{tag}_{pct}"


def checkpoint_path(cfg: dict, model_name: str, appliance: str, augmented: bool) -> Path:
    ckpt_dir = checkpoint_dir_path(cfg)
    suffix = experiment_run_suffix(cfg, augmented=augmented)
    return ckpt_dir / f"{model_name.lower()}_{appliance}_{suffix}.pt"


def results_path(cfg: dict, name: str) -> Path:
    out_dir = results_dir_path(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def model_training_config(cfg: dict, model_name: str) -> dict[str, int | float]:
    """Resolve epochs / patience / lr / batch_size for a model (Geng Table 3)."""
    model = model_name.lower()
    train_cfg = cfg.get("training", {})
    per_model = train_cfg.get("per_model", {}).get(model, {})
    batch_size = int(cfg["dataloader"]["batch_size"][model])
    return {
        "batch_size": batch_size,
        "epochs": int(per_model.get("epochs", train_cfg.get("epochs", 100))),
        "patience": int(per_model.get("patience", train_cfg.get("patience", 20))),
        "lr": float(per_model.get("lr", train_cfg.get("lr", 0.001))),
        "early_stopping": bool(per_model.get("early_stopping", train_cfg.get("early_stopping", True))),
    }


def merge_cli_config(
    cfg: dict,
    *,
    model: str,
    appliance: str,
    augmented: bool,
    train_percent: str,
    data_root: str | None,
    epochs: int | None,
    device: str | None = None,
    gpu_id: int | None = None,
    require_cuda: bool | None = None,
) -> dict:
    out = copy.deepcopy(cfg)
    out["model"]["name"] = model
    out["data"]["appliance"] = appliance
    out["data"]["augmented"] = augmented
    out["data"]["train_percent"] = train_percent
    if data_root:
        out["data"]["data_root"] = data_root
    if device is not None:
        out["training"]["device"] = device
    if gpu_id is not None:
        out["training"]["gpu_id"] = gpu_id
    if require_cuda is not None:
        out["training"]["require_cuda"] = require_cuda
    # Keep relative paths in cfg; resolve at use-time via data_root_path()
    if epochs is not None:
        out["training"]["epochs"] = epochs
    return out


def merge_rho_cli_config(
    cfg: dict,
    *,
    model: str,
    appliance: str,
    rho_pct: int,
    n_real: int = 100_000,
    data_root: str | None = None,
    epochs: int | None = None,
    device: str | None = None,
    gpu_id: int | None = None,
    require_cuda: bool | None = None,
) -> dict:
    """Config for Geng-style injection-ratio experiment (build_geng_rho_datasets.py)."""
    out = merge_cli_config(
        cfg,
        model=model,
        appliance=appliance,
        augmented=int(rho_pct) > 0,
        train_percent=str(int(rho_pct)),
        data_root=data_root,
        epochs=epochs,
        device=device,
        gpu_id=gpu_id,
        require_cuda=require_cuda,
    )
    out["data"]["injection_rho"] = int(rho_pct)
    out["data"]["n_real"] = int(n_real)
    return out


def add_device_cli_args(parser) -> None:
    """--cpu / --cuda / --gpu-id / --allow-cpu on train & batch scripts."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu", action="store_true", help="Force CPU (no GPU)")
    group.add_argument("--cuda", action="store_true", help="Force CUDA; error if GPU missing")
    parser.add_argument("--gpu-id", type=int, default=None, help="CUDA device index (default: 0)")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not error when GPU is missing (use CPU instead)",
    )


def device_options_from_args(args) -> tuple[str | None, int | None, bool | None]:
    device: str | None = None
    require_cuda: bool | None = None
    if getattr(args, "cpu", False):
        device = "cpu"
        require_cuda = False
    elif getattr(args, "cuda", False):
        device = "cuda"
        require_cuda = True
    if getattr(args, "allow_cpu", False):
        require_cuda = False
    gpu_id = getattr(args, "gpu_id", None)
    return device, gpu_id, require_cuda
