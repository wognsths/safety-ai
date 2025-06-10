"""models.py

Utility to instantiate backbone models (ResNet, EfficientNet, …) with a custom
output dimension. Centralises model creation so that train/ scripts can call a
single `init_net` function.

Supported:
* resnet50
* resnet34
* efficientnet_b4
* efficientnet_b0 (fallback)

Add new models by extending `_MODEL_FACTORY` mapping.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torchvision.models as tvm


def _resnet50(output_dim: int, pretrained: bool = False) -> torch.nn.Module:
    model = tvm.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    model.fc = torch.nn.Linear(model.fc.in_features, output_dim)
    return model


def _resnet34(output_dim: int, pretrained: bool = False) -> torch.nn.Module:
    model = tvm.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
    model.fc = torch.nn.Linear(model.fc.in_features, output_dim)
    return model


def _efficientnet_b4(output_dim: int, pretrained: bool = False) -> torch.nn.Module:
    model = tvm.efficientnet_b4(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, output_dim)
    return model


def _efficientnet_b0(output_dim: int, pretrained: bool = False) -> torch.nn.Module:
    model = tvm.efficientnet_b0(weights="IMAGENET1K_V2" if pretrained else None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, output_dim)
    return model


# mapping
_MODEL_FACTORY: Dict[str, Callable[[int, bool], torch.nn.Module]] = {
    "resnet50": _resnet50,
    "resnet34": _resnet34,
    "efficientnet_b4": _efficientnet_b4,
    "efficientnet_b0": _efficientnet_b0,
}


# ──────────────────────────────────────────────────────────────
# public helpers
# ──────────────────────────────────────────────────────────────

def list_models() -> list[str]:
    """Return list of supported model names."""
    return sorted(_MODEL_FACTORY.keys())


def init_net(
    model_name: str,
    output_dim: int,
    *,
    pretrained: bool = False,
    device: torch.device | str | None = None,
) -> torch.nn.Module:
    """Create a model and move to device.

    Parameters
    ----------
    model_name : str
        One of list_models(). Case‑insensitive.
    output_dim : int
        Number of output classes.
    pretrained : bool, optional
        If True, load ImageNet pretrained weights.
    device : torch.device | str | None, optional
        If provided, move model to this device.
    """
    name = model_name.lower()
    if name not in _MODEL_FACTORY:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {list_models()}")

    model = _MODEL_FACTORY[name](output_dim, pretrained)
    if device is not None:
        model.to(device)
    return model
