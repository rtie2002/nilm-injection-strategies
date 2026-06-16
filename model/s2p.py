"""S2P (Seq2Point) — PyTorch port of NILM-main/NILM_Models.S2P_model."""

from __future__ import annotations

import torch
from torch import nn


def _seq_to_2d(x: torch.Tensor) -> torch.Tensor:
    """(B, L) -> (B, 1, L, 1) for Keras-style Conv2D over the time axis."""
    if x.dim() == 2:
        return x.unsqueeze(1).unsqueeze(-1)
    if x.dim() == 3:
        return x.unsqueeze(-1)
    return x


class S2P(nn.Module):
    """Mains window (B, L) -> midpoint appliance power (B, 1)."""

    def __init__(self, window_length: int = 600):
        super().__init__()
        self.window_length = window_length

        self.features = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(10, 1), padding=(4, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(30, 30, kernel_size=(8, 1), padding=(3, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(30, 40, kernel_size=(6, 1), padding=(2, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(40, 50, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 50, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * window_length, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(_seq_to_2d(x))
        return self.head(h)
