"""AugLPN — PyTorch port of NILM-main/NILM_Models.AugLPN_NILM (channel=32)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _seq_to_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1).unsqueeze(-1)
    if x.dim() == 3:
        return x.unsqueeze(-1)
    return x


def _hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return F.relu6(x + 3.0) / 6.0


class SepConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: tuple[int, int]):
        kh, kw = kernel_size
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=(kh // 2, kw // 2), groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pointwise(self.depthwise(x)))


class AugLPN(nn.Module):
    """Seq2Point AugLPN_NILM: (B, L) -> (B, 1)."""

    def __init__(self, window_length: int = 600, channels: int = 32):
        super().__init__()
        self.window_length = window_length
        c = channels

        self.stem = nn.Conv2d(1, c, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.enc2 = nn.Conv2d(c, c * 2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.enc3 = nn.Conv2d(c * 2, c * 4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.enc4 = nn.Conv2d(c * 4, c * 8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.lat4 = nn.Conv2d(c * 8, c, kernel_size=1)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=(1, 2), mode="nearest"), nn.Conv2d(c, c, (1, 2)))
        self.lat3 = nn.Conv2d(c * 4, c, kernel_size=1)
        self.up2 = nn.Upsample(scale_factor=(1, 2), mode="nearest")
        self.lat2 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.up1 = nn.Upsample(scale_factor=(1, 2), mode="nearest")
        self.lat1 = nn.Conv2d(c, c, kernel_size=1)

        self.pan_conv1 = nn.Conv2d(c, c, kernel_size=1)
        self.pan_gate = nn.Conv2d(c, c, kernel_size=1)
        self.pan_b3 = nn.Conv2d(c, c, kernel_size=(1, 3), padding=(0, 1))
        self.pan_up2 = nn.Conv2d(c, c * 2, kernel_size=(1, 2), stride=(1, 2))
        self.pan_ref2 = nn.Conv2d(c * 2, c * 2, kernel_size=(1, 3), padding=(0, 1))
        self.pan_down3 = nn.Conv2d(c * 2, c, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.pan_dil1 = nn.Conv2d(c * 2, 96, kernel_size=(1, 3), padding=(2, 1), dilation=(2, 1))
        self.pan_dil2 = nn.Conv2d(96, 64, kernel_size=(1, 3), padding=(4, 1), dilation=(4, 1))
        self.pan_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.pan_refine = nn.Conv2d(64, c, kernel_size=(1, 3), padding=(0, 1))

        self.sep3a = SepConv2d(c * 4, 128, (1, 2))
        self.sep3b = SepConv2d(c * 4, 128, (1, 3))
        self.bigru3 = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.gru3_proj = nn.Conv2d(128, 64, kernel_size=(1, 2))

        self.sep2a = SepConv2d(c * 2, 64, (1, 2))
        self.sep2b = SepConv2d(c * 2, 64, (1, 3))
        self.sep2_down = nn.Conv2d(64, 64, kernel_size=(1, 2), stride=(1, 2))
        self.bigru2 = nn.GRU(64, 32, batch_first=True, bidirectional=True)
        self.gru2_proj = nn.Conv2d(64, 64, kernel_size=(1, 2))

        with torch.no_grad():
            dummy = _seq_to_2d(torch.zeros(1, window_length))
            flat_dim = self._encode_and_fuse(dummy).numel()

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 56),
            nn.ReLU(inplace=True),
            nn.Linear(56, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1),
        )

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        l1 = F.relu(self.stem(x))
        l2 = F.relu(self.enc2(l1))
        l3 = F.relu(self.enc3(l2))
        l4 = F.relu(self.enc4(l3))

        p4 = F.relu(self.lat4(l4))
        p3 = F.relu(self.up3(p4) + self.lat3(l3))
        p2 = F.relu(self.up2(p3) + self.lat2(l2))
        p1 = F.relu(self.up1(p2) + self.lat1(l1))
        return l2, l3, p1, p3

    def _pan(self, p1: torch.Tensor, l2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        t = F.normalize(self.pan_conv1(p1), p=2, dim=2)
        g = _hard_sigmoid(F.relu(self.pan_gate(F.adaptive_max_pool2d(p1, output_size=(1, 1)))))
        t = t * g
        b = F.normalize(self.pan_b3(p1), p=2, dim=2)
        t = t * b

        pan = self.pan_up2(t) + F.relu(self.pan_ref2(l2))
        pre = F.relu(self.pan_down3(pan)) + p3

        d = F.relu(self.pan_dil1(pan))
        d = F.relu(self.pan_dil2(d))
        d = F.relu(self.pan_refine(self.pan_pool(d)))
        return d + pre

    def _gru_branch(self, l2: torch.Tensor, l3: torch.Tensor) -> torch.Tensor:
        b, _, h3, _ = l3.shape
        s3 = self.sep3a(l3) + self.sep3b(l3)
        seq3, _ = self.bigru3(s3.squeeze(-1).permute(0, 2, 1))
        g3 = self.gru3_proj(seq3.permute(0, 2, 1).unsqueeze(-1))

        s2 = F.relu(self.sep2_down(self.sep2a(l2) + self.sep2b(l2)))
        _, h2, _ = s2.shape
        seq2, _ = self.bigru2(s2.squeeze(-1).permute(0, 2, 1))
        g2 = self.gru2_proj(seq2.permute(0, 2, 1).unsqueeze(-1))

        if g2.shape[2] != g3.shape[2]:
            g2 = F.interpolate(g2, size=g3.shape[2:], mode="nearest")
        return g2 + g3

    def _encode_and_fuse(self, x: torch.Tensor) -> torch.Tensor:
        l2, l3, p1, p3 = self._encode(x)
        pan = self._pan(p1, l2, p3)
        gru = self._gru_branch(l2, l3)
        if gru.shape[2:] != pan.shape[2:]:
            gru = F.interpolate(gru, size=pan.shape[2:], mode="nearest")
        return torch.cat([gru, pan], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._encode_and_fuse(_seq_to_2d(x)))
