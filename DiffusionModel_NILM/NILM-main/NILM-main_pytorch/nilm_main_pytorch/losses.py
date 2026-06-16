"""Geng training losses — PyTorch (Huber + switch-state combined loss)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nilm_main_pytorch.models.params import PARAMS_APPLIANCE, norm_on_threshold


def huber_elementwise(y_true: torch.Tensor, y_pred: torch.Tensor, delta: float) -> torch.Tensor:
    residual = torch.abs(y_true - y_pred)
    small = residual.pow(2)
    large = delta * residual - 0.5 * delta**2
    return torch.where(residual < delta, small, large)


def huber_batch(y_true: torch.Tensor, y_pred: torch.Tensor, delta: float) -> torch.Tensor:
    return huber_elementwise(y_true, y_pred, delta).mean(dim=-1).mean()


def switch_state_penalty(y_true: torch.Tensor, y_pred: torch.Tensor, norm_threshold: float) -> torch.Tensor:
    true_state = (y_true > norm_threshold).float()
    pred_state = (y_pred > norm_threshold).float()
    return (true_state - pred_state).pow(2).mean(dim=-1)


def build_loss_fn(
    model_name: str,
    appliance: str,
    *,
    augmented: bool,
    huber_delta: float = 0.5,
    regression_alpha: float = 1.0,
    switch_beta: float = 0.1,
):
    """Match Geng TF: augmented uses combined_loss; origin EasyS2S uses MSE."""
    name = model_name.lower()
    norm_thr = norm_on_threshold(appliance)
    on_thr_w = PARAMS_APPLIANCE[appliance]["on_power_threshold"]

    if augmented:
        def loss_fn(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            reg = huber_batch(y, pred, huber_delta)
            sw = switch_state_penalty(y, pred, norm_thr).mean()
            return regression_alpha * reg + switch_beta * sw

        return loss_fn, on_thr_w

    if name == "easy_s2s":
        return lambda pred, y: F.mse_loss(pred, y), on_thr_w

    return lambda pred, y: huber_batch(y, pred, huber_delta), on_thr_w
