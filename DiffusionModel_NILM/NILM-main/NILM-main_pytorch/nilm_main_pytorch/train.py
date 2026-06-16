"""
Train Geng NILM models — PyTorch version.

  python -m nilm_main_pytorch.train --model easy_s2s --appliance kettle
  python -m nilm_main_pytorch.train --model s2p --appliance microwave --augmented
"""

from __future__ import annotations

import sys
from pathlib import Path

_PYTORCH_ROOT = Path(__file__).resolve().parent.parent
if str(_PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTORCH_ROOT))

import argparse
import json
import time

import torch

from nilm_main_pytorch.data.datasets import build_train_val_loaders
from nilm_main_pytorch.data.paths import require_csv, train_csv_path, validation_csv_path
from nilm_main_pytorch.losses import build_loss_fn
from nilm_main_pytorch.metrics import evaluate_loader
from nilm_main_pytorch.models import build_model
from nilm_main_pytorch.models.params import PARAMS_APPLIANCE
from nilm_main_pytorch.utils import (
    checkpoint_path,
    data_root_path,
    get_device,
    load_config,
    merge_cli_config,
    model_training_config,
    norm_stats,
    portable_path_str,
    relativize_config,
    set_seed,
)


def train_one(cfg: dict, *, verbose: bool = True) -> dict:
    appliance = cfg["data"]["appliance"]
    model_name = cfg["model"]["name"]
    augmented = bool(cfg["data"]["augmented"])
    train_cfg = cfg["training"]

    set_seed(int(train_cfg.get("seed", 2024)))
    device = get_device(cfg)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    data_root = data_root_path(cfg)
    if verbose:
        print(f"data_root: {portable_path_str(data_root)}")
    train_csv = require_csv(
        train_csv_path(
            data_root,
            appliance,
            origin=not augmented,
            train_percent=str(cfg["data"]["train_percent"]),
            dataset_name=cfg["data"].get("dataset_name", "UK_DALE"),
        ),
        "Training",
    )
    val_csv = require_csv(validation_csv_path(data_root, appliance), "Validation")

    train_params = model_training_config(cfg, model_name)
    batch_size = int(train_params["batch_size"])
    train_loader, val_loader = build_train_val_loaders(
        model_name=model_name,
        appliance=appliance,
        train_csv=train_csv,
        val_csv=val_csv,
        batch_size=batch_size,
        num_workers=int(cfg["dataloader"].get("num_workers", 0)),
    )

    model = build_model(model_name, appliance).to(device)
    loss_fn, on_thr_w = build_loss_fn(
        model_name,
        appliance,
        augmented=augmented,
        huber_delta=float(train_cfg.get("huber_delta", 0.5)),
        regression_alpha=float(train_cfg.get("regression_alpha", 1.0)),
        switch_beta=float(train_cfg.get("switch_beta", 0.1)),
    )
    stats = norm_stats(appliance)
    sample_second = float(cfg.get("evaluation", {}).get("sample_second", 6.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_params["lr"]))
    epochs = int(train_params["epochs"])
    patience = int(train_params["patience"])
    save_path = checkpoint_path(cfg, model_name, appliance, augmented)

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    since = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n_train += x.size(0)
        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item() * x.size(0)
                n_val += x.size(0)
        val_loss /= max(n_val, 1)

        val_metrics = evaluate_loader(
            model,
            val_loader,
            device,
            stats["appliance_mean"],
            stats["appliance_std"],
            on_thr_w,
            sample_second,
        )

        if verbose:
            print(
                f"epoch {epoch}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} | "
                f"val MAE {val_metrics['mae']:.2f} W | SAE {val_metrics['sae']:.4f} | "
                f"F1 {val_metrics['f1']:.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": relativize_config(cfg),
                    "best_val_loss": best_val,
                    "epoch": epoch,
                    "appliance": appliance,
                    "augmented": augmented,
                    "model_name": model_name,
                },
                save_path,
            )
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch}")
                break

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_metrics = evaluate_loader(
        model,
        val_loader,
        device,
        stats["appliance_mean"],
        stats["appliance_std"],
        on_thr_w,
        sample_second,
    )

    return {
        "model": model_name.lower(),
        "appliance": appliance,
        "augmented": augmented,
        "train_csv": portable_path_str(train_csv),
        "val_csv": portable_path_str(val_csv),
        "val_mae": val_metrics["mae"],
        "val_sae": val_metrics["sae"],
        "val_f1": val_metrics["f1"],
        "best_epoch": best_epoch,
        "checkpoint": portable_path_str(save_path),
        "elapsed_s": time.time() - since,
        "status": "trained",
    }


def parse_args() -> argparse.Namespace:
    from nilm_main_pytorch.models.params import ALL_APPLIANCES

    p = argparse.ArgumentParser(description="Train Geng NILM models (PyTorch)")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model", choices=["easy_s2s", "s2p", "fcn", "auglpn"], required=True)
    p.add_argument("--appliance", choices=list(ALL_APPLIANCES), required=True)
    p.add_argument("--augmented", action="store_true")
    p.add_argument("--train-percent", default="20")
    p.add_argument("--data-root", default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = merge_cli_config(
        load_config(args.config),
        model=args.model,
        appliance=args.appliance,
        augmented=args.augmented,
        train_percent=args.train_percent,
        data_root=args.data_root,
        epochs=args.epochs,
    )
    result = train_one(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
