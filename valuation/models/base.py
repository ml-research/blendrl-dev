from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Union, List

from dataclasses import dataclass, field

import torch as th
from hackatari import HackAtari

from nsfr.fol.language import Language


@dataclass
class BaseValuationModelConfig:
    type: str = field(init=False)
    static_predicates: List[str] = field(default_factory=lambda: [])


class BaseValuationModel(th.nn.Module, ABC):
    type: ClassVar[str]

    def __init__(self, env_name: str, lang: Language, config: BaseValuationModelConfig,
                 device: Optional[Union[th.device, str]]):
        super().__init__()

        self.env_name: str
        self.lang = lang
        self.config = config
        self.device = device

        if len(self.config.static_predicates) > 0:
            from valuation.utils import get_default_valuation_model
            self.static_model = get_default_valuation_model(env_name, self.device)

    def forward(self, predicate_name: str, *inputs) -> th.Tensor:
        if predicate_name in self.config.static_predicates:
            return self.static_model(predicate_name, *inputs)

        input_tensor = th.cat(inputs, dim=-1)
        batch_size = input_tensor.shape[0]
        input_tensor = input_tensor.view(batch_size, -1).float()
        return self.forward_predicate(predicate_name, input_tensor)

    @abstractmethod
    def forward_predicate(self, predicate_name: str, input: th.Tensor) -> th.Tensor:
        pass
