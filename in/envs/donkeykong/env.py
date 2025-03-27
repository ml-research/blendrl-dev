from typing import Sequence
import torch
from nudge.env import NudgeBaseEnv
from hackatari.core import HackAtari
import numpy as np
import torch as th
from ocatari.ram.donkeykong import MAX_NB_OBJECTS
import gymnasium as gym
from blendrl.env_utils import make_env


class NudgeEnv(NudgeBaseEnv):
    """
    NUDGE environment for DonkeyKong.
    
    Args:
        mode (str): Mode of the environment. Possible values are "train" and "eval".
        n_envs (int): Number of environments.
        render_mode (str): Mode of rendering. Possible values are "rgb_array" and "human".
        render_oc_overlay (bool): Whether to render the overlay of OC.
        seed (int): Seed for the environment.
    """
    name = "donkeykong"
    pred2action = {
        'noop': 0,
        'fire': 1,
        'up': 2,
        'right': 3,
        'left': 4,
        'down': 5,
    }
    pred_names: Sequence

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False, seed=None):
        """
        Constructor for the VectorizedNudgeEnv class.
        
        Args:
            mode (str): Mode of the environment. Possible values are "train" and "eval".
            n_envs (int): Number of environments.
            render_mode (str): Mode of rendering. Possible values are "rgb_array" and "human".
            render_oc_overlay (bool): Whether to render the overlay of OC.
            seed (int): Seed for the environment.
        """
        super().__init__(mode)
        self.env = HackAtari(env_name="ALE/DonkeyKong-v5", mode="ram", obs_mode="ori",\
            modifs=[("no_barrel"), ("change_level0")],\
            rewardfunc_path="in/envs/donkeykong/blenderl_reward.py",\
            render_mode=render_mode, render_oc_overlay=render_oc_overlay)
        
        
        # apply wrapper to _env
        self.env._env = make_env(self.env._env)
        self.n_actions = 6
        self.n_raw_actions = 18
        self.n_objects = 49
        self.n_features = 4  # visible, x-pos, y-pos, right-facing
        self.seed = seed

        # Compute index offsets. Needed to deal with multiple same-category objects
        self.obj_offsets = {}
        offset = 0
        for (obj, max_count) in MAX_NB_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_NB_OBJECTS.keys())

    def reset(self):
        """
        Reset the environment.
        
        Returns:
            logic_state (torch.Tensor): Logic state of the environment.
            neural_state (torch.Tensor): Neural state of the environment.
        """
        raw_state, _ = self.env.reset(seed=self.seed)
        state = self.env.objects
        logic_state, neural_state =  self.extract_logic_state(state), self.extract_neural_state(raw_state)
        logic_state = logic_state.unsqueeze(0)
        return logic_state, neural_state

    def step(self, action, is_mapped=False):
        """
        Perform a step in the environment.
        
        Args:
            action (torch.Tensor): Action to perform.
            is_mapped (bool): Whether the action is already mapped.
        Returns:
            logic_state (torch.Tensor): Logic state of the environment.
            neural_state (torch.Tensor): Neural state of the environment.
            reward (float): Reward obtained.
            done (bool): Whether the episode is done.
            truncations (dict): Truncations.
            infos (dict): Additional information.
        """
        raw_state, reward, truncations, done, infos = self.env.step(action)
        # save state in the original enviromnt to visualization
        state = self.env.objects
        logic_state, neural_state = self.convert_state(state, raw_state)
        logic_state = logic_state.unsqueeze(0)
        return (logic_state, neural_state), reward, done, truncations, infos

    def extract_logic_state(self, input_state):
        """ 
        Extracts the logic state from the input state.
        Args:
            input_state (list): List of objects in the environment.
        Returns:
            torch.Tensor: Logic state.
        
        """
        state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)

        obj_count = {k: 0 for k in MAX_NB_OBJECTS.keys()}
        self.bboxes = th.zeros((self.n_objects, 4), dtype=th.int32)

        for obj in input_state:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            if obj.category == "Time":
                state[idx] = th.Tensor([1, obj.value, 0, 0])
            else:
                orientation = obj.orientation.value if obj.orientation is not None else 0
                state[idx] = th.tensor([1, *obj.center, orientation])
            obj_count[obj.category] += 1
            self.bboxes[idx] = th.tensor(obj.xywh)
        return state

    def extract_neural_state(self, raw_input_state):
        """
        Extracts the neural state from the raw input state.
        Args:
            raw_input_state (torch.Tensor): Raw input state.
        Returns:
            torch.Tensor: Neural state.
        """
        return torch.Tensor(raw_input_state).unsqueeze(0)#.float()

    def close(self):
        """
        Close the environment.
        """
        self.env.close()
