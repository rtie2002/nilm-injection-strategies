"""EasyS2S — PyTorch port of NILM-main/EasyS2S_Model.py (Conv1D seq-to-seq)."""

from __future__ import annotations

import torch
from torch import nn


class EasyS2S(nn.Module):
    """Mains window (B, L) -> appliance power sequence (B, L)."""

    def __init__(self, window_length: int = 600, n_dense: int = 1):
        super().__init__()
        self.window_length = window_length

        # Keras original: Conv1D(..., activation='relu') followed by BatchNormalization().
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
        )

        hidden = 512
        # Keras original: Dense(..., activation='relu') followed by BatchNormalization().
        dense_blocks: list[nn.Module] = [nn.Linear(16 * window_length, hidden), nn.ReLU(inplace=True), nn.BatchNorm1d(hidden)]
        for _ in range(max(0, n_dense - 1)):
            dense_blocks.extend([nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.BatchNorm1d(hidden)])
        self.dense = nn.Sequential(*dense_blocks)
        self.output = nn.Linear(hidden, window_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv(x).flatten(1)
        h = self.dense(h)
        return self.output(h)
