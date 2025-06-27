from pathlib import Path
from typing import Dict, Literal

import numpy as np
import torch as th
from dataclasses import dataclass

from nsfr.fol.language import Language
from valuation.models.base import BaseValuationModel, BaseValuationModelConfig


@dataclass
class FixedStaticConfig(BaseValuationModelConfig):
    type = "fixed-static"

    program: str = "chatgpt_4o_v1_17x17"


class ValuationModelFixedStatic(BaseValuationModel):
    type = "fixed-static"

    force_relative_positions = True

    def __init__(self, env_name: str, lang: Language, config: FixedStaticConfig, device=None):
        super().__init__(env_name, lang, config, device)

        self._grid_values = self._init_grid_values()

    def _init_grid_values(self) -> Dict[str, th.Tensor]:
        """
        Returns: Dict (pred_name -> values tensor [H, W])
        """
        result = dict()
        for pred_name in self.learnable_predicates:
            prior_path = Path("in/envs") / self.env_name / "priors" / "programs" / self.config.program / (pred_name + ".txt")
            prior = np.loadtxt(prior_path, delimiter=" ", dtype=float)
            prior = th.as_tensor(prior, device=self.device, dtype=th.float32)
            result[pred_name] = prior

        return result

    def forward_predicate(self, predicate_name: str, input: th.Tensor) -> th.Tensor:
        # grid_sample requires 4-dimensional input tensor
        grid_values = self._grid_values[predicate_name].unsqueeze(0).unsqueeze(0)

        # grid_sample requires 4-dimensional output tensor
        sample_coords = input[:, 1:3].unsqueeze(0).unsqueeze(0)

        # sample from grid
        result = th.nn.functional.grid_sample(grid_values, sample_coords, align_corners=False)

        return result[0, 0, 0]