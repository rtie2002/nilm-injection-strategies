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


def to_config_path(path: Path, anchor: Path) -> str:
    """Express an absolute path relative to anchor (for portable yaml/checkpoints)."""
    return str(path.resolve().relative_to(anchor.resolve())).replace("\\", "/")


def relativize_config(cfg: dict) -> dict:
    """Store relative paths in checkpoints so models work on another PC / drive."""
    out = copy.deepcopy(cfg)
    data = out.get("data", {})
    if "data_root" in data:
        data["data_root"] = to_config_path(data_root_path(out), PYTORCH_ROOT)
    paths = out.get("paths", {})
    if "checkpoint_dir" in paths:
        paths["checkpoint_dir"] = to_config_path(checkpoint_dir_path(out), PYTORCH_ROOT)
    if "results_dir" in paths:
        paths["results_dir"] = to_config_path(results_dir_path(out), PYTORCH_ROOT)
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


def get_device(cfg: dict) -> torch.device:
    train_cfg = cfg.get("training", {})
    if train_cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available():
        gpu_id = int(train_cfg.get("gpu_id", 0))
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


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


def checkpoint_path(cfg: dict, model_name: str, appliance: str, augmented: bool) -> Path:
    tag = "aug" if augmented else "origin"
    pct = str(cfg["data"]["train_percent"])
    ckpt_dir = checkpoint_dir_path(cfg)
    return ckpt_dir / f"{model_name.lower()}_{appliance}_{tag}_{pct}.pt"


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
) -> dict:
    out = copy.deepcopy(cfg)
    out["model"]["name"] = model
    out["data"]["appliance"] = appliance
    out["data"]["augmented"] = augmented
    out["data"]["train_percent"] = train_percent
    if data_root:
        out["data"]["data_root"] = data_root
    # Keep relative paths in cfg; resolve at use-time via data_root_path()
    if epochs is not None:
        out["training"]["epochs"] = epochs
    return out
