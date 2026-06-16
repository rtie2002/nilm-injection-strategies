"""
Test Geng NILM checkpoints — PyTorch version.

  python -m nilm_main_pytorch.test --model easy_s2s --appliance kettle --test-house 2
  python -m nilm_main_pytorch.test --model easy_s2s --appliance kettle --augmented --test-house 2 --ewma
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nilm_main_pytorch.data.datasets import build_loader, load_geng_csv
from nilm_main_pytorch.data.paths import require_csv, test_csv_path, validation_csv_path
from nilm_main_pytorch.inference import predict_geng_test
from nilm_main_pytorch.metrics import (
    DEFAULT_SAMPLE_SECOND,
    apply_postprocess,
    compute_metrics,
    denorm_appliance,
    prepare_predictions_geng,
)
from nilm_main_pytorch.models import build_model
from nilm_main_pytorch.models.params import ALL_APPLIANCES, PARAMS_APPLIANCE
from nilm_main_pytorch.utils import (
    checkpoint_path,
    get_device,
    load_config,
    merge_cli_config,
    model_training_config,
    norm_stats,
    results_path,
    save_json,
)


def _collect_predictions(model, loader, device, mean: float, std: float) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred_parts: list[np.ndarray] = []
    true_parts: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))
            pred_parts.append(denorm_appliance(pred.cpu(), mean, std).reshape(-1))
            true_parts.append(denorm_appliance(y, mean, std).reshape(-1))
    return np.concatenate(pred_parts), np.concatenate(true_parts)


def test_one(
    cfg: dict,
    *,
    test_house: int,
    use_ewma: bool | None = None,
    split: str = "test",
    geng_inference: bool = True,
    verbose: bool = True,
) -> dict:
    appliance = cfg["data"]["appliance"]
    model_name = cfg["model"]["name"]
    augmented = bool(cfg["data"]["augmented"])
    device = get_device(cfg)
    stats = norm_stats(appliance)
    on_thr_w = PARAMS_APPLIANCE[appliance]["on_power_threshold"]
    sample_second = float(cfg.get("evaluation", {}).get("sample_second", DEFAULT_SAMPLE_SECOND))

    data_root = Path(cfg["data"]["data_root"])
    if split == "val":
        eval_csv = require_csv(validation_csv_path(data_root, appliance), "Validation")
    elif split == "test":
        eval_csv = require_csv(test_csv_path(data_root, appliance, test_house), f"Test house {test_house}")
    else:
        raise ValueError("split must be 'test' or 'val'")

    batch_size = int(model_training_config(cfg, model_name)["batch_size"])
    window_length = PARAMS_APPLIANCE[appliance]["window_length"]

    ckpt_path = checkpoint_path(cfg, model_name, appliance, augmented)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(model_name, appliance).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if geng_inference and split == "test":
        agg, app = load_geng_csv(eval_csv)
        pred_z, true_z = predict_geng_test(
            model, model_name, agg, app, window_length, device, batch_size
        )
        pred_w = prepare_predictions_geng(
            denorm_appliance(torch.from_numpy(pred_z), stats["appliance_mean"], stats["appliance_std"])
        )
        true_w = prepare_predictions_geng(
            denorm_appliance(torch.from_numpy(true_z), stats["appliance_mean"], stats["appliance_std"])
        )
    else:
        loader = build_loader(
            model_name=model_name,
            appliance=appliance,
            csv_path=eval_csv,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(cfg["dataloader"].get("num_workers", 0)),
        )
        pred_w, true_w = _collect_predictions(
            model, loader, device, stats["appliance_mean"], stats["appliance_std"]
        )
        pred_w = prepare_predictions_geng(pred_w)
        true_w = prepare_predictions_geng(true_w)

    raw_metrics = compute_metrics(pred_w, true_w, on_thr_w, sample_second)

    apply_ema = use_ewma
    if apply_ema is None:
        apply_ema = augmented and bool(cfg.get("evaluation", {}).get("apply_ewma_when_augmented", True))

    if apply_ema:
        alpha = float(cfg.get("evaluation", {}).get("ewma_alpha", 0.9))
        pred_w = apply_postprocess(pred_w, augmented=True, threshold_w=on_thr_w, ewma_alpha=alpha)
        final_metrics = compute_metrics(pred_w, true_w, on_thr_w, sample_second)
        postprocess = f"ewma_alpha={alpha}"
    else:
        final_metrics = raw_metrics
        postprocess = "none"

    if verbose:
        print(
            f"{model_name} {appliance} | {'aug' if augmented else 'origin'} | "
            f"house {test_house} | MAE {final_metrics['mae']:.2f} W | "
            f"SAE {final_metrics['sae']:.4f} | F1 {final_metrics['f1']:.4f} | {postprocess}"
        )

    return {
        "model": model_name.lower(),
        "appliance": appliance,
        "augmented": augmented,
        "test_house": test_house,
        "split": split,
        "inference": "geng_full_sequence" if (geng_inference and split == "test") else "window_loader",
        "eval_csv": str(eval_csv),
        "checkpoint": str(ckpt_path),
        "postprocess": postprocess,
        "mae": final_metrics["mae"],
        "sae": final_metrics["sae"],
        "f1": final_metrics["f1"],
        "raw_mae": raw_metrics["mae"],
        "status": "tested",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test Geng NILM models (PyTorch)")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model", choices=["easy_s2s", "s2p", "fcn", "auglpn"], required=True)
    p.add_argument("--appliance", choices=list(ALL_APPLIANCES), required=True)
    p.add_argument("--augmented", action="store_true")
    p.add_argument("--train-percent", default="20")
    p.add_argument("--data-root", default=None)
    p.add_argument("--test-house", type=int, choices=[1, 2], default=2)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument(
        "--window-eval",
        action="store_true",
        help="Use window DataLoader instead of Geng full-sequence inference (test split only)",
    )
    p.add_argument("--ewma", action="store_true", help="Force EWMA post-process")
    p.add_argument("--no-ewma", action="store_true", help="Disable EWMA even if augmented")
    p.add_argument("--save", action="store_true", help="Write JSON to nilm_main_pytorch/results/")
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
        epochs=None,
    )

    use_ewma = None
    if args.ewma:
        use_ewma = True
    if args.no_ewma:
        use_ewma = False

    result = test_one(
        cfg,
        test_house=args.test_house,
        use_ewma=use_ewma,
        split=args.split,
        geng_inference=not args.window_eval,
    )

    if args.save:
        tag = "aug" if args.augmented else "origin"
        out = results_path(
            cfg,
            f"test_{args.model}_{args.appliance}_{tag}_h{args.test_house}.json",
        )
        save_json(out, result)
        print(f"saved {out}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
