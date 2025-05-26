from typing import List

import torch as th
from dataclasses import dataclass, field

from nsfr.fol.language import Language
from nsfr.fol.logic import NeuralPredicate
from utils import FRAME_SIZE
from valuation.models.base import BaseValuationModel, BaseValuationModelConfig


@dataclass
class MLPConfig(BaseValuationModelConfig):
    type = "mlp"

    hidden_sizes: List[int] = field(default_factory=lambda: [64, 32])
    use_position_difference: bool = False
    discard_missing_objects: bool = False


class ValuationMLP(th.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int = 1):
        super().__init__()
        sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(sizes)-1):
            layers.append(th.nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:
                layers.append(th.nn.ReLU())

        self.model = th.nn.Sequential(
            *layers,
            th.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.model(x).squeeze(-1)
        return y


class ValuationModelMLP(BaseValuationModel):
    type = "mlp"

    def __init__(self, env_name: str, lang: Language, config: MLPConfig, device=None):
        super().__init__(env_name, lang, config, device)

        mlps = dict()

        for pred in self.lang.preds:
            if isinstance(pred, NeuralPredicate) and pred.name not in self.config.static_predicates:
                module_name = pred.name
                input_size = sum([dtype.num_features for dtype in pred.dtypes])
                if config.use_position_difference:
                    input_size -= 4
                mlp = ValuationMLP(input_size=input_size, hidden_sizes=config.hidden_sizes, output_size=1).to(self.device)
                mlps[module_name] = mlp

        self.heads = th.nn.ModuleDict(mlps)


    def forward_predicate(self, predicate_name, input):
        mlp = self.heads[predicate_name]

        num_objects = input.shape[1] // 4

        if self.config.discard_missing_objects:
            indices = (input[:, list(range(4, num_objects * 4, 4))] == 1.0).any(dim=1)
        else:
            indices = th.ones(input.shape[0], dtype=th.bool, device=input.device)

        result = th.zeros(input.shape[0], dtype=input.dtype, device=input.device)

        if indices.sum() == 0:
            return result

        # compute normalized differences
        if self.config.use_position_difference:
            x = input[indices, 4:]
            player_x = input[indices, 1]
            player_y = input[indices, 2]

            obj_index = 0
            while obj_index < x.shape[1] // 4:
                x[:, obj_index * 4 + 1] = (player_x - x[:, obj_index * 4 + 1]) / FRAME_SIZE[self.env_name][0]
                x[:, obj_index * 4 + 2] = (player_y - x[:, obj_index * 4 + 2]) / FRAME_SIZE[self.env_name][1]
                obj_index += 1
        else:
            x = input[indices]

        result[indices] = mlp(x)
        return result
