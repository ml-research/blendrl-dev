from __future__ import annotations

from typing import List, Type, Optional

import torch
from torch import nn as nn

from utils import load_classes_in_package, load_module
from valuation.models.base import BaseValuationModelConfig, BaseValuationModel


def load_model_config_classes() -> List[Type[BaseValuationModelConfig]]:
    return load_classes_in_package("valuation/models", BaseValuationModelConfig)


def load_model_config_class(type: str) -> Optional[Type[BaseValuationModelConfig]]:
    classes = load_model_config_classes()
    for cls in classes:
        if cls.type == type:
            return cls
    return None


def load_model_classes() -> List[Type[BaseValuationModel]]:
    return load_classes_in_package("valuation/models", BaseValuationModel)


def load_model_class(type: str) -> Optional[Type[BaseValuationModel]]:
    classes = load_model_classes()
    for cls in classes:
        if cls.type == type:
            return cls
    return None


def get_default_valuation_model(env_name: str, device: torch.device) -> nn.Module:
    module = load_module(f"in/envs/{env_name}/valuation.py")
    model = nn.Module()
    model.forward = lambda predicate_name, *inputs: getattr(module, predicate_name)(*inputs)
    model = model.to(device)
    return model
