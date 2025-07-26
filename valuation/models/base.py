from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Union, List

import torch as th
from dataclasses import dataclass, field

from nsfr.fol.language import Language
from utils import optional, get_default_device, FRAME_SIZE


@dataclass
class BaseValuationModelConfig:
    type: str = field(init=False)

    static_predicates: List[str] = field(default_factory=lambda: [])
    use_position_difference: bool = False
    discard_missing_objects: bool = False
    clip_value: float = 0.0


class BaseValuationModel(th.nn.Module, ABC):
    type: ClassVar[str]

    force_relative_positions: bool = False

    def __init__(self, env_name: str, lang: Language, config: BaseValuationModelConfig,
                 device: Optional[Union[th.device, str]]):
        super().__init__()

        self.env_name = env_name
        self.lang = lang
        self.config = config
        self.device = optional(device, get_default_device())

        from valuation.utils import get_default_valuation_model
        self.static_model = get_default_valuation_model(env_name, self.device)

    @property
    def learnable_predicates(self) -> List[str]:
        return [pred.name for pred in self.lang.neural_predicates if pred.name not in self.config.static_predicates]

    def forward(self, predicate_name: str, *inputs) -> th.Tensor:
        # Forward to static valuation model
        if predicate_name in self.config.static_predicates:
            return self.static_model(predicate_name, *inputs)

        # Concatenate input tensors
        input_tensor = th.cat(inputs, dim=-1)
        batch_size = input_tensor.shape[0]
        input_tensor = input_tensor.view(batch_size, -1).float()

        # Find discarded objects
        num_objects = input_tensor.shape[1] // 4
        if self.config.discard_missing_objects:
            indices = (input_tensor[:, list(range(4, num_objects * 4, 4))] == 1.0).any(dim=1)
        else:
            indices = th.ones(input_tensor.shape[0], dtype=th.bool, device=input_tensor.device)

        result = th.zeros(input_tensor.shape[0], dtype=input_tensor.dtype, device=input_tensor.device)

        if indices.sum() == 0:
            return result

        # Compute relative positions
        if self.force_relative_positions or self.config.use_position_difference:
            x = input_tensor[indices, 4:]
            player_x = input_tensor[indices, 1]
            player_y = input_tensor[indices, 2]

            obj_index = 0
            while obj_index < x.shape[1] // 4:
                x[:, obj_index * 4 + 1] = (player_x - x[:, obj_index * 4 + 1]) / FRAME_SIZE[self.env_name][0]
                x[:, obj_index * 4 + 2] = (player_y - x[:, obj_index * 4 + 2]) / FRAME_SIZE[self.env_name][1]
                obj_index += 1
        else:
            x = input_tensor[indices]

        # Forward to valuation model
        result[indices] = self.forward_predicate(predicate_name, x)
        th.clip(result, self.config.clip_value, 1 - self.config.clip_value)
        return result

    @abstractmethod
    def forward_predicate(self, predicate_name: str, input: th.Tensor) -> th.Tensor:
        pass
