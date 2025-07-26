from __future__ import annotations

import math
from typing import List, Type, Optional

import torch
from torch import nn as nn

from utils import load_classes_in_package, load_module
from valuation.models.base import BaseValuationModel


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


def sample_inputs(num: int, mode: str, device: torch.device, x_range: (float, float) = (-1, 1), y_range: (float, float) = (-1, 1)) -> torch.Tensor:
    if mode == "random":
        positions = torch.rand(num, 2, device=device)
    elif mode == "grid":
        positions = torch.zeros(num, 2, device=device)

        num_per_axis = math.ceil(math.sqrt(num))
        num_all = num_per_axis ** 2
        cell_size = 1 / (num_per_axis + 1), 1 / (num_per_axis + 1)

        step = num_all / num
        for i in range(num):
            j = math.floor(step * i)
            row = j // num_per_axis
            col = j % num_per_axis
            positions[i, 0] = cell_size[0] * (col + 1)
            positions[i, 1] = cell_size[1] * (row + 1)
    else:
        assert True, f"Unknown sampling mode {mode})"

    positions[:, 0] = positions[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    positions[:, 1] = positions[:, 1] * (y_range[1] - y_range[0]) + y_range[0]

    inputs = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device).repeat(num, 1)
    inputs[:, 1:3] = positions

    return inputs