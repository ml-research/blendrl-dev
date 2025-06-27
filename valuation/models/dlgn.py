from typing import List

import torch as th
from dataclasses import dataclass, field
from difflogic import LogicLayer, GroupSum

from nsfr.fol.language import Language
from valuation.models.base import BaseValuationModelConfig, BaseValuationModel

"""
    coordinate_system:      __________ x
                            |
                            |
                            | y
    frame_size:             [w=160, h=210]
    bbox:                   [1, center_x, center_y, orientation{0=N/None, 1=NNE, 2=NE, 3=ENE 4=E, ..., 15=NNW}]

    player:                 center, orientation
    ladder:                 center
    platform:               center
    fruit:                  center
    bell:                   center
    monkey:                 center, orientation?
    throwncoconut:          center, orientation?
    fallingcoconut:         center, orientation?

    nothing_around:         [bbox * 49]
    on_ladder:              [bbox player, bbox ladder]
    right_of_ladder:        [bbox player, bbox ladder]
    left_of_ladder:         [bbox player, bbox ladder]
    false_predicate:        [0]
    on_pl_ladder:           [bbox ladder?, bbox platform?]
    on_pl_player:           [bbox player?, bbox platform?]
    close_by_fruit:         [bbox player, bbox fruit]
    close_by_bell:          [bbox player, bbox bell]
    close_by_monkey:        [bbox player, bbox monkey]
    close_by_throwncoconut: [bbox player, bbox throwncoconut]
    close_by_fallingcoconut:[bbox player, bbox fallingcoconut]
    same_level_ladder:      [bbox player, bbox ladder]
"""


@dataclass
class DLGNConfig(BaseValuationModelConfig):
    type = "dlgn"

    hidden_sizes: List[int] = field(default_factory=lambda: [4096, 2048, 2048])


class ValuationDLGN(th.nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size=1, device='cuda'):
        super().__init__()
        sizes = [input_size] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(LogicLayer(sizes[i], sizes[i + 1], device=device))

        self.model = th.nn.Sequential(
            *layers,
            GroupSum(k=output_size, tau=sizes[-1])
        )

    def forward(self, x):
        y = self.model(x).squeeze(-1)
        return y


class ValuationModelDLGN(BaseValuationModel):
    type = "dlgn"

    def __init__(self, env_name: str, lang: Language, config: DLGNConfig, device=None):
        super().__init__(env_name, lang, config, device)

        dlgns = dict()

        for pred in self.lang.neural_predicates:
            if pred.name not in self.config.static_predicates:
                module_name = pred.name
                input_size = sum([dtype.num_features for dtype in pred.dtypes])
                if self.config.use_position_difference:
                    input_size -= 4
                dlgn = ValuationDLGN(input_size=input_size, hidden_sizes=config.hidden_sizes, output_size=1, device=self.device.type)
                dlgns[module_name] = dlgn

        self.heads = th.nn.ModuleDict(dlgns)

    def forward_predicate(self, predicate_name: str, input: th.Tensor) -> th.Tensor:
        dlgn = self.heads[predicate_name]

        # unset orientation
        input[:, 3::4] = 0

        return dlgn(input)
