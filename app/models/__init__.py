"""Model architectures (vendored from official repositories) and checkpoint loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from app.models.retinexformer_arch import RetinexFormer
from app.models.scunet_arch import SCUNet
from app.models.zerodce_arch import enhance_net_nopool


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        for key in ("params", "params_ema", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    return {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}


def _finalize(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    model.load_state_dict(_load_state_dict(path), strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)


def build_retinexformer(path: Path, arch: dict[str, Any], device: torch.device) -> RetinexFormer:
    arch = dict(arch)
    if "num_blocks" in arch:
        arch["num_blocks"] = list(arch["num_blocks"])
    return _finalize(RetinexFormer(**arch), path, device)


def build_zerodce(path: Path, device: torch.device) -> enhance_net_nopool:
    return _finalize(enhance_net_nopool(scale_factor=1), path, device)


def build_scunet(path: Path, device: torch.device) -> SCUNet:
    return _finalize(SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64), path, device)


__all__ = ["RetinexFormer", "SCUNet", "enhance_net_nopool", "build_retinexformer", "build_zerodce", "build_scunet"]
