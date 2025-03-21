from typing import Sequence
import torch
from nudge.env import NudgeBaseEnv
from ocatari.core import OCAtari
from ocatari.ram.pong import MAX_NB_OBJECTS
from blendrl.env_utils import make_env
import numpy as np


class NudgeEnv(NudgeBaseEnv):
    name = "breakout"
    pred2action = {
        'noop': 0,
        'fire': 1,
        'right': 2,
        'left': 3
    }
    pred_names: Sequence

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False, seed=None):
        super().__init__(mode)
        self.env = OCAtari(env_name="ALE/Breakout-v5", mode="ram",obs_mode="ori",
                           render_mode=render_mode, render_oc_overlay=render_oc_overlay)

        self.env._env = make_env(self.env._env)
        self.n_actions = len(self.pred2action)
        self.n_raw_actions = 4
        self.n_objects = 110
        self.n_features = 4
        self.seed = seed
       
        # Compute index offsets. Needed to deal with multiple same-category objects
        self.obj_offsets = {}
        offset = 0
        for obj, max_count in MAX_NB_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_NB_OBJECTS.keys())

    def reset(self):
        raw_state, _ = self.env.reset(seed=self.seed)
        state = self.env.objects
        logic_state, neural_state = self.extract_logic_state(state), self.extract_neural_state(raw_state)
        logic_state = logic_state.unsqueeze(0)
        return logic_state, neural_state

    def step(self, action, is_mapped: bool = False):
        raw_state, reward, truncations, done, infos = self.env.step(action)
        state = self.env.objects
        logic_state, neural_state = self.convert_state(state, raw_state)
        logic_state = logic_state.unsqueeze(0)
        return (logic_state, neural_state), reward, done, truncations, infos

    def extract_logic_state(self, raw_state):
        logic_state = np.zeros((self.n_objects, self.n_features))
        for idx, obj in enumerate(raw_state):
            logic_state[idx][-4:] = np.array(obj.h_coords).flatten()

        if(raw_state[1].category == "NoObject"):
            logic_state[1][0] = -1
        
        return torch.tensor(logic_state, dtype=torch.float32)


    def extract_neural_state(self, raw_state):
        return torch.Tensor(raw_state).unsqueeze(0)

    def close(self):
        self.env.close()
