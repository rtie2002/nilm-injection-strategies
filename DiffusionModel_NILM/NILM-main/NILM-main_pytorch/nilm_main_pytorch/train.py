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
    add_device_cli_args,
    checkpoint_path,
    data_root_path,
    device_options_from_args,
    get_device,
    gpu_stats,
    load_config,
    log_device,
    merge_cli_config,
    model_training_config,
    norm_stats,
    portable_path_str,
    relativize_config,
    set_seed,
)


def _run_epoch_batches(
    model,
    loader,
    device: torch.device,
    loss_fn,
    *,
    train: bool,
    optimizer=None,
    epoch: int = 0,
    total_epochs: int = 0,
    progress: bool = False,
    label: str = "train",
) -> float:
    n_batches = len(loader)
    progress_every = max(1, n_batches // 10)
    total_loss = 0.0
    n_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for bi, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=device.type == "cuda")
            y = y.to(device, non_blocking=device.type == "cuda")

            if train and bi == 0 and epoch == 1 and progress:
                pdev = next(model.parameters()).device
                print(
                    f"  [check] batch tensors on {x.device} | model weights on {pdev}",
                    flush=True,
                )

            if train:
                optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            if train:
                loss.backward()
                optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

            if progress and (bi % progress_every == 0 or bi == n_batches - 1):
                print(
                    f"  [{label}] epoch {epoch}/{total_epochs} "
                    f"batch {bi + 1}/{n_batches} loss {loss.item():.4f} | {gpu_stats(device)}",
                    flush=True,
                )

    return total_loss / max(n_samples, 1)


def train_one(
    cfg: dict,
    *,
    verbose: bool = True,
    show_device: bool = True,
) -> dict:
    appliance = cfg["data"]["appliance"]
    model_name = cfg["model"]["name"]
    augmented = bool(cfg["data"]["augmented"])
    train_cfg = cfg["training"]
    dl_cfg = cfg.get("dataloader", {})

    set_seed(int(train_cfg.get("seed", 2024)))
    device = log_device(cfg) if show_device else get_device(cfg)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    data_root = data_root_path(cfg)
    if verbose:
        print(f"data_root: {portable_path_str(data_root)}", flush=True)

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
    num_workers = int(dl_cfg.get("num_workers", 0))
    pin_memory = bool(dl_cfg.get("pin_memory", True))

    train_loader, val_loader = build_train_val_loaders(
        model_name=model_name,
        appliance=appliance,
        train_csv=train_csv,
        val_csv=val_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    if verbose:
        print(
            f"dataset: {n_train:,} train windows, {len(train_loader)} batches | "
            f"{n_val:,} val windows | batch_size={batch_size} | workers={num_workers}",
            flush=True,
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
    metrics_every = int(cfg.get("evaluation", {}).get("metrics_every_epochs", 5))
    progress_batches = bool(train_cfg.get("progress_batches", True)) and verbose

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_params["lr"]))
    epochs = int(train_params["epochs"])
    patience = int(train_params["patience"])
    save_path = checkpoint_path(cfg, model_name, appliance, augmented)

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    since = time.time()
    last_metrics: dict[str, float] | None = None

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.time()
        train_loss = _run_epoch_batches(
            model,
            train_loader,
            device,
            loss_fn,
            train=True,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=epochs,
            progress=progress_batches,
            label="train",
        )
        t_train = time.time() - t0

        model.eval()
        t1 = time.time()
        if verbose:
            print(f"  [val] epoch {epoch}/{epochs} running...", flush=True)
        val_loss = _run_epoch_batches(
            model,
            val_loader,
            device,
            loss_fn,
            train=False,
            epoch=epoch,
            total_epochs=epochs,
            progress=False,
            label="val",
        )
        t_val = time.time() - t1

        # Metrics requires another full pass over the validation loader; avoid doing it every epoch unless requested.
        run_metrics = (epoch % metrics_every == 0) or (epoch == epochs)
        t_metrics = 0.0
        if run_metrics:
            t2 = time.time()
            last_metrics = evaluate_loader(
                model,
                val_loader,
                device,
                stats["appliance_mean"],
                stats["appliance_std"],
                on_thr_w,
                sample_second,
            )
            t_metrics = time.time() - t2

        if verbose:
            parts = [
                f"epoch {epoch}/{epochs}",
                f"train {train_loss:.4f} ({t_train:.1f}s)",
                f"val {val_loss:.4f} ({t_val:.1f}s)",
            ]
            if last_metrics and run_metrics:
                parts.append(f"MAE {last_metrics['mae']:.1f}W")
                parts.append(f"F1 {last_metrics['f1']:.3f}")
                if t_metrics > 0:
                    parts.append(f"metrics {t_metrics:.1f}s")
            parts.append(gpu_stats(device))
            marker = " *best*" if val_loss < best_val else ""
            print(" | ".join(parts) + marker, flush=True)

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
            if verbose:
                print(f"  -> saved {portable_path_str(save_path)}", flush=True)
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch} (patience {patience})", flush=True)
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
    elapsed = time.time() - since
    if verbose:
        print(
            f"done {appliance} {'aug' if augmented else 'origin'} pct={cfg['data']['train_percent']} | "
            f"best epoch {best_epoch} | {elapsed / 60:.1f} min total | {gpu_stats(device)}",
            flush=True,
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
        "elapsed_s": elapsed,
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
    add_device_cli_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dev, gpu_id, require_cuda = device_options_from_args(args)
    cfg = merge_cli_config(
        load_config(args.config),
        model=args.model,
        appliance=args.appliance,
        augmented=args.augmented,
        train_percent=args.train_percent,
        data_root=args.data_root,
        epochs=args.epochs,
        device=dev,
        gpu_id=gpu_id,
        require_cuda=require_cuda,
    )
    result = train_one(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
