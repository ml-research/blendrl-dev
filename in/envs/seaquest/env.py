from typing import Sequence, Optional, List

import torch
import torch as th
from ocatari.ram.seaquest import MAX_NB_OBJECTS

from blendrl.env_utils import make_env
from hackatari.core import HackAtari
from nudge.env import NudgeBaseEnv
from utils import optional, DEFAULT_MODIFICATIONS


class NudgeEnv(NudgeBaseEnv):
    name = "seaquest"
    pred2action = {
        "noop": 0,
        "fire": 1,
        "up": 2,
        "right": 3,
        "left": 4,
        "down": 5,
    }
    pred_names: Sequence

    def __init__(
            self,
            mode: str,
            render_mode="rgb_array",
            render_oc_overlay=False,
            seed=None,
            modifications: Optional[List[str]] = None,
            reward_fn_path: Optional[str] = None,
            *args,
            **kwargs
    ):
        super().__init__(mode)
        self.env = HackAtari(
            env_name="ALE/Seaquest-v5",
            mode="ram",
            obs_mode="ori",
            modifs=optional(modifications, DEFAULT_MODIFICATIONS[NudgeEnv.name]),
            rewardfunc_path=optional(reward_fn_path, f"in/envs/{NudgeEnv.name}/reward/default.py"),
            render_mode=render_mode,
            render_oc_overlay=render_oc_overlay,
            *args,
            **kwargs
        )
        # for learning script from cleanrl
        self.env._env = make_env(self.env._env)
        self.n_actions = 6
        self.n_raw_actions = 18
        self.n_objects = 43 # there are only 42 objects, but we keep it at 43 for backward compatibility
        self.n_features = 4  # visible, x-pos, y-pos, right-facing
        self.seed = seed

        # Compute index offsets. Needed to deal with multiple same-category objects
        self.obj_offsets = {}
        offset = 0
        for obj, max_count in MAX_NB_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_NB_OBJECTS.keys())

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed)
        state = self.env.objects
        raw_state = obs  # self.env.dqn_obs
        logic_state, neural_state = self.extract_logic_state(
            state
        ), self.extract_neural_state(raw_state)
        logic_state = logic_state.unsqueeze(0)
        return logic_state, neural_state

    def step(self, action, is_mapped: bool = False):
        obs, reward, truncations, done, infos = self.env.step(action)

        state = self.env.objects
        raw_state = obs  # self.env.dqn_obs
        logic_state, neural_state = self.convert_state(state, raw_state)
        logic_state = logic_state.unsqueeze(0)
        return (logic_state, neural_state), reward, done, truncations, infos

    def extract_logic_state(self, input_state):
        state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)
        self.bboxes = th.zeros((self.n_objects, 4), dtype=th.int32)

        obj_count = {k: 0 for k in MAX_NB_OBJECTS.keys()}

        for obj in input_state:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            if obj.category == "OxygenBar":
                state[idx] = th.Tensor([1, obj.value, 0, 0])
            else:
                orientation = (
                    obj.orientation.value if obj.orientation is not None else 0
                )
                state[idx] = th.tensor([1, *obj.center, orientation])
            obj_count[obj.category] += 1
            self.bboxes[idx] = th.tensor(obj.xywh)

        return state

    def extract_neural_state(self, raw_input_state):
        return torch.Tensor(raw_input_state).unsqueeze(0)  # .float()

    def close(self):
        self.env.close()
