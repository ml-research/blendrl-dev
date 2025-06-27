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

        for pred in self.lang.neural_predicates:
            if pred.name not in self.config.static_predicates:
                module_name = pred.name
                input_size = sum([dtype.num_features for dtype in pred.dtypes])
                if config.use_position_difference:
                    input_size -= 4
                mlp = ValuationMLP(input_size=input_size, hidden_sizes=config.hidden_sizes, output_size=1).to(self.device)
                mlps[module_name] = mlp

        self.heads = th.nn.ModuleDict(mlps)


    def forward_predicate(self, predicate_name, input):
        mlp = self.heads[predicate_name]

        return mlp(input)
