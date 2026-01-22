"""Factory helpers for creating image models (custom / torchvision / timm)."""
from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision import models

try:
    import timm
except ImportError:  # pragma: no cover - handled at runtime
    timm = None

from src.models_image import create_model as create_custom_model, is_custom_model

ModelSource = Literal["custom", "torchvision", "timm"]

_TORCHVISION_RESNETS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
}


def is_torchvision_model(model_name: str) -> bool:
    return model_name.lower() in _TORCHVISION_RESNETS


def _create_torchvision_model(model_name: str, num_outputs: int, pretrained: bool) -> nn.Module:
    key = model_name.lower()
    if key not in _TORCHVISION_RESNETS:
        raise ValueError(f"Unsupported torchvision model_name: {model_name}")
    builder, weight_enum = _TORCHVISION_RESNETS[key]
    weights = weight_enum if pretrained else None
    model = builder(weights=weights)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
        return model
    raise ValueError(f"Unsupported torchvision architecture for regression head: {model_name}")


def _create_timm_model(model_name: str, num_outputs: int, pretrained: bool) -> nn.Module:
    if timm is None:
        raise ImportError("timm is required for model_name={model_name}")
    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_outputs)
    except Exception as exc:
        raise ValueError(f"Unable to create timm model '{model_name}': {exc}") from exc


def detect_model_source(model_name: str) -> ModelSource:
    if is_custom_model(model_name):
        return "custom"
    if is_torchvision_model(model_name):
        return "torchvision"
    return "timm"


def create_model_from_name(
    model_name: str,
    num_outputs: int,
    pretrained: bool,
    *,
    source: ModelSource | None = None,
) -> nn.Module:
    """Instantiate a model by name."""
    model_source = source or detect_model_source(model_name)
    if model_source == "custom":
        return create_custom_model(model_name, num_outputs=num_outputs)
    if model_source == "torchvision":
        return _create_torchvision_model(model_name, num_outputs=num_outputs, pretrained=pretrained)
    if model_source == "timm":
        return _create_timm_model(model_name, num_outputs=num_outputs, pretrained=pretrained)
    raise ValueError(f"Unknown model source {model_source}")


__all__ = ["create_model_from_name", "detect_model_source", "is_torchvision_model", "ModelSource"]
