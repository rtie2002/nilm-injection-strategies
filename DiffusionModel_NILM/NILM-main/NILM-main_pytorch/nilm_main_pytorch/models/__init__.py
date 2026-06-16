"""Geng NILM model zoo — PyTorch version."""

from __future__ import annotations

import torch.nn as nn

from .auglpn import AugLPN
from .easy_s2s import EasyS2S
from .fcn import FCN
from .params import ALL_APPLIANCES, FCN_OUTPUT_LENGTH, FCN_TARGET_LENGTH, PARAMS_APPLIANCE
from .s2p import S2P

MODEL_NAMES = ("easy_s2s", "s2p", "fcn", "auglpn")


def build_model(model_name: str, appliance: str) -> nn.Module:
    name = model_name.lower()
    window_length = PARAMS_APPLIANCE[appliance]["window_length"]

    if name == "easy_s2s":
        return EasyS2S(window_length=window_length)
    if name == "s2p":
        return S2P(window_length=window_length)
    if name == "fcn":
        input_length = FCN_TARGET_LENGTH + (FCN_TARGET_LENGTH // 2) * 2
        return FCN(input_length=input_length, output_length=FCN_OUTPUT_LENGTH)
    if name == "auglpn":
        return AugLPN(window_length=window_length, channels=32)
    raise ValueError(f"Unknown model {model_name!r}. Choose from {MODEL_NAMES}")
