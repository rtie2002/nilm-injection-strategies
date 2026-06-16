"""FCN — PyTorch port of NILM-main/fcn_Model.py (dilated Conv1D, center crop)."""

from __future__ import annotations

import torch
from torch import nn


class FCN(nn.Module):
    """
    Long mains input (B, input_length) -> cropped appliance sequence (B, output_length).

    Default Geng setup: input_length=4105, output_length=1053 (center crop).
  Training targets are cropped to output_length in geng_data.
    """

    def __init__(self, input_length: int, output_length: int = 1053):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.crop_offset = (input_length - output_length) // 2

        layers: list[nn.Module] = [
            nn.Conv1d(1, 128, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        ]
        for dilation in (2, 4, 8, 16, 32, 64, 128, 256):
            layers.extend(
                [
                    nn.Conv1d(128, 128, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv1d(128, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 1, kernel_size=1),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        y = self.net(x).squeeze(1)
        off = self.crop_offset
        return y[:, off : off + self.output_length]
