"""
Train / evaluate Geng NILM models in PyTorch (EasyS2S, S2P, FCN, AugLPN).

Examples:
  python geng_train.py --model easy_s2s --appliance kettle
  python geng_train.py --model s2p --appliance microwave --augmented
  python geng_train.py --model fcn --appliance kettle --epochs 100
  python geng_train.py --model auglpn --appliance fridge --eval-only
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from geng_data import build_geng_loaders, resolve_geng_paths
from metrics import evaluate_loader
from model.auglpn import AugLPN
from model.easy_s2s import EasyS2S
from model.fcn import FCN
from model.geng_params import FCN_OUTPUT_LENGTH, FCN_TARGET_LENGTH, PARAMS_APPLIANCE, norm_on_threshold
from model.s2p import S2P
from model_train import huber_loss_batch, switch_state_penalty

CODE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = (
    CODE_DIR
    / "DiffusionModel_NILM"
    / "NILM-main"
    / "dataset_preprocess"
    / "created_data"
    / "UK_DALE"
)


def load_geng_config(path: Path | None = None) -> dict:
    cfg_path = path or CODE_DIR / "hyperparameter_geng.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_geng_model(model_name: str, appliance: str, device: torch.device) -> nn.Module:
    name = model_name.lower()
    window_length = PARAMS_APPLIANCE[appliance]["window_length"]

    if name == "easy_s2s":
        model = EasyS2S(window_length=window_length)
    elif name == "s2p":
        model = S2P(window_length=window_length)
    elif name == "fcn":
        input_length = FCN_TARGET_LENGTH + (FCN_TARGET_LENGTH // 2) * 2
        model = FCN(input_length=input_length, output_length=FCN_OUTPUT_LENGTH)
    elif name == "auglpn":
        model = AugLPN(window_length=window_length, channels=32)
    else:
        raise ValueError(f"Unknown model: {model_name} (use easy_s2s, s2p, fcn, auglpn)")

    return model.to(device)


def geng_norm_stats(appliance: str) -> dict:
    p = PARAMS_APPLIANCE[appliance]
    return {
        "method": "zscore",
        "appliance_mean": float(p["mean"]),
        "appliance_std": float(p["std"]),
        "aggregate_mean": 522.0,
        "aggregate_std": 814.0,
    }


def to_watts_factory(stats: dict):
    def to_watts(t: torch.Tensor) -> np.ndarray:
        arr = t.detach().cpu().numpy()
        return arr * stats["appliance_std"] + stats["appliance_mean"]

    return to_watts


def build_loss_fn(cfg: dict, appliance: str, augmented: bool):
    train_cfg = cfg["training"]
    delta = float(train_cfg.get("huber_delta", 0.5))
    norm_thr = norm_on_threshold(appliance)
    on_thr_w = PARAMS_APPLIANCE[appliance]["on_power_threshold"]

    if augmented and cfg["model"]["name"].lower() == "easy_s2s":
        alpha = float(train_cfg.get("regression_alpha", 1.0))
        beta = float(train_cfg.get("switch_beta", 0.1))

        def loss_fn(pred, y):
            reg = huber_loss_batch(y, pred, delta)
            sw = switch_state_penalty(y, pred, norm_thr).mean()
            return alpha * reg + beta * sw

        return loss_fn, on_thr_w

    if cfg["model"]["name"].lower() == "easy_s2s" and not augmented:
        return lambda pred, y: nn.functional.mse_loss(pred, y), on_thr_w

    return lambda pred, y: huber_loss_batch(y, pred, delta), on_thr_w


def checkpoint_path(cfg: dict, appliance: str, augmented: bool) -> Path:
    model = cfg["model"]["name"].lower()
    tag = "origin" if not augmented else "aug"
    save_dir = CODE_DIR / cfg["training"].get("save_dir", "checkpoints_geng")
    return save_dir / f"{model}_{appliance}_{tag}_{cfg['data']['train_percent']}.pt"


def train_one(cfg: dict, *, plot: bool = False, verbose: bool = True) -> dict:
    appliance = cfg["data"]["appliance"]
    model_name = cfg["model"]["name"]
    augmented = bool(cfg["data"].get("augmented", False))
    device = torch.device(
        f"cuda:{cfg['training'].get('gpu_id', 0)}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    data_root = Path(cfg["data"]["data_root"])
    train_csv, val_csv = resolve_geng_paths(
        data_root,
        appliance,
        origin=not augmented,
        train_percent=str(cfg["data"]["train_percent"]),
        dataset_name=cfg["data"].get("dataset_name", "UK_DALE"),
    )

    batch_size = int(cfg["dataloader"]["batch_size"].get(model_name.lower(), 128))
    train_loader, val_loader = build_geng_loaders(
        model_name=model_name,
        appliance=appliance,
        train_csv=train_csv,
        val_csv=val_csv,
        batch_size=batch_size,
        num_workers=int(cfg["dataloader"].get("num_workers", 0)),
    )

    model = build_geng_model(model_name, appliance, device)
    loss_fn, on_thr_w = build_loss_fn(cfg, appliance, augmented)
    stats = geng_norm_stats(appliance)
    to_watts = to_watts_factory(stats)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))
    epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"].get("patience", 20))
    save_path = checkpoint_path(cfg, appliance, augmented)

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    since = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        val_metrics = evaluate_loader(model, val_loader, device, to_watts, on_thr_w)

        if verbose:
            print(
                f"epoch {epoch}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} | "
                f"val MAE {val_metrics['mae']:.2f} W | F1 {val_metrics['f1']:.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": cfg,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                    "appliance": appliance,
                    "augmented": augmented,
                },
                save_path,
            )
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch} (patience {patience})")
                break

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_metrics = evaluate_loader(model, val_loader, device, to_watts, on_thr_w)

    elapsed = time.time() - since
    return {
        "model": model_name.lower(),
        "appliance": appliance,
        "augmented": augmented,
        "train_csv": str(train_csv),
        "val_mae": val_metrics["mae"],
        "val_sae": val_metrics["sae"],
        "val_f1": val_metrics["f1"],
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "checkpoint": str(save_path),
        "elapsed_s": elapsed,
        "status": "ok",
    }


def evaluate_one(cfg: dict, verbose: bool = True) -> dict:
    appliance = cfg["data"]["appliance"]
    model_name = cfg["model"]["name"]
    augmented = bool(cfg["data"].get("augmented", False))
    device = torch.device(
        f"cuda:{cfg['training'].get('gpu_id', 0)}" if torch.cuda.is_available() else "cpu"
    )

    data_root = Path(cfg["data"]["data_root"])
    train_csv, val_csv = resolve_geng_paths(
        data_root,
        appliance,
        origin=not augmented,
        train_percent=str(cfg["data"]["train_percent"]),
    )
    batch_size = int(cfg["dataloader"]["batch_size"].get(model_name.lower(), 128))
    _, val_loader = build_geng_loaders(
        model_name=model_name,
        appliance=appliance,
        train_csv=train_csv,
        val_csv=val_csv,
        batch_size=batch_size,
    )

    model = build_geng_model(model_name, appliance, device)
    save_path = checkpoint_path(cfg, appliance, augmented)
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    stats = geng_norm_stats(appliance)
    on_thr_w = PARAMS_APPLIANCE[appliance]["on_power_threshold"]
    val_metrics = evaluate_loader(model, val_loader, device, to_watts_factory(stats), on_thr_w)

    if verbose:
        print(f"eval {model_name} {appliance} | val MAE {val_metrics['mae']:.2f} W")

    return {
        "model": model_name.lower(),
        "appliance": appliance,
        "val_mae": val_metrics["mae"],
        "val_sae": val_metrics["sae"],
        "val_f1": val_metrics["f1"],
        "checkpoint": str(save_path),
        "status": "eval_only",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Geng NILM models (PyTorch)")
    p.add_argument("--config", default=None, help="hyperparameter_geng.yaml path")
    p.add_argument("--model", choices=["easy_s2s", "s2p", "fcn", "auglpn"], required=True)
    p.add_argument("--appliance", choices=list(PARAMS_APPLIANCE.keys()), required=True)
    p.add_argument("--augmented", action="store_true", help="Use UK_DALECombined mix CSV")
    p.add_argument("--train-percent", default="20", help="10 or 20 for 100k/200k crops")
    p.add_argument("--data-root", default=None, help="UK_DALE created_data root")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_geng_config(Path(args.config) if args.config else None)
    cfg = copy.deepcopy(cfg)

    cfg["model"]["name"] = args.model
    cfg["data"]["appliance"] = args.appliance
    cfg["data"]["augmented"] = args.augmented
    cfg["data"]["train_percent"] = args.train_percent
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    if not Path(cfg["data"]["data_root"]).is_absolute():
        cfg["data"]["data_root"] = str(CODE_DIR / cfg["data"]["data_root"])
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    if args.eval_only:
        result = evaluate_one(cfg)
    else:
        result = train_one(cfg)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
