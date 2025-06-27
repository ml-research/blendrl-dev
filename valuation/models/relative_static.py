import torch as th
from dataclasses import dataclass

from nsfr.fol.language import Language
from nsfr.utils.common import bool_to_probs
from utils import FRAME_SIZE
from valuation.models.base import BaseValuationModel, BaseValuationModelConfig


@dataclass
class RelativeStaticConfig(BaseValuationModelConfig):
    type = "relative-static"


class ValuationModelRelativeStatic(BaseValuationModel):
    type = "relative-static"

    force_relative_positions = True

    def __init__(self, env_name: str, lang: Language, config: RelativeStaticConfig, device=None):
        super().__init__(env_name, lang, config, device)

    def forward_predicate(self, predicate_name, input) -> th.Tensor:
        return getattr(self, predicate_name)(input)

    def on_ladder(self, diff: th.Tensor) -> th.Tensor:
        x = diff[:, 1]
        return bool_to_probs(th.logical_and(-0.025 < x, x < 0.025)) * diff[:, 0] * self.same_level_ladder(diff)

    def left_of_ladder(self, diff: th.Tensor) -> th.Tensor:
        x = diff[:, 1]
        return bool_to_probs(x < -0.01875) * diff[:, 0] * self.same_level_ladder(diff)

    def right_of_ladder(self, diff: th.Tensor) -> th.Tensor:
        x = diff[:, 1]
        return bool_to_probs(0.01875 < x) * diff[:, 0] * self.same_level_ladder(diff)

    def same_level_ladder(self, diff: th.Tensor) -> th.Tensor:
        y = diff[:, 2] + 0.047
        is_same_level = th.logical_and(-0.114 < y, y < 0.114)
        return bool_to_probs(is_same_level)

    def close_by_monkey(self, diff: th.Tensor) -> th.Tensor:
        return self.close_by(diff)

    def close_by_throwncoconut(self, diff: th.Tensor) -> th.Tensor:
        return self.close_by(diff)

    def close_by(self, diff: th.Tensor) -> th.Tensor:
        th = 32
        obj_prob = diff[:, 0]
        frame_size = FRAME_SIZE[self.env_name]
        x_dist = (diff[:, 1] * frame_size[0]).pow(2)
        y_dist = (diff[:, 2] * frame_size[1]).pow(2)
        dist = (x_dist + y_dist).sqrt()
        return bool_to_probs(dist < th) * obj_prob
