import torch as th
from dataclasses import dataclass

from nsfr.fol.language import Language
from valuation.models.base import BaseValuationModel, BaseValuationModelConfig


@dataclass
class StaticConfig(BaseValuationModelConfig):
    type = "static"


class ValuationModelStatic(BaseValuationModel):
    type = "static"

    def __init__(self, env_name: str, lang: Language, config: StaticConfig, device=None):
        super().__init__(env_name, lang, config, device)

    def forward(self, predicate_name: str, *inputs) -> th.Tensor:
        return self.static_model(predicate_name, *inputs)

    def forward_predicate(self, predicate_name, *inputs) -> th.Tensor:
        pass
