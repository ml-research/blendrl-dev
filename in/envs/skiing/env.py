import abc
from typing import Sequence, Optional, List, Dict

import torch
import torch as th
from hackatari.core import HackAtari
from ocatari.ram.skiing import MAX_NB_OBJECTS_HUD, MAX_NB_OBJECTS

from blendrl.env_utils import make_env
from nudge.env import NudgeBaseEnv
from utils import optional, DEFAULT_MODIFICATIONS


class BlendRLEnv(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def ale_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def max_objects(self) -> Dict[str, int]:
        pass

    @property
    @abc.abstractmethod
    def pred2action(self) -> Dict[str, int]:
        pass

    @property
    def n_actions(self) -> int:
        return len(self.pred2action)

    @property
    def n_objects(self) -> int:
        return sum(self.max_objects.values())


class NudgeEnv(NudgeBaseEnv):
    """
    NUDGE environment for Skiing.

    Args:
        mode (str): Mode of the environment. Possible values are "train" and "eval".
        n_envs (int): Number of environments.
        render_mode (str): Mode of rendering. Possible values are "rgb_array" and "human".
        render_oc_overlay (bool): Whether to render the overlay of OC.
        seed (int): Seed for the environment.
    """

    name = "skiing"
    pred2action = {
        "noop": 0,
        "right": 1,
        "left": 2
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
        """
        Constructor for the NudgeEnv class.

        Args:
            mode (str): Mode of the environment. Possible values are "train" and "eval".
            n_envs (int): Number of environments.
            render_mode (str): Mode of rendering. Possible values are "rgb_array" and "human".
            render_oc_overlay (bool): Whether to render the overlay of OC.
            seed (int): Seed for the environment.
        """
        super().__init__(mode)
        self.env = HackAtari(
            env_name="ALE/Skiing-v5",
            mode="ram",
            obs_mode="ori",
            modifs=optional(modifications, DEFAULT_MODIFICATIONS[NudgeEnv.name]),
            rewardfunc_path=optional(reward_fn_path, f"in/envs/{NudgeEnv.name}/reward/default.py"),
            render_mode=render_mode,
            render_oc_overlay=render_oc_overlay,
            *args,
            **kwargs
        )
        # apply wrapper to _env
        self.env._env = make_env(self.env._env)
        # self.env_ori._env = make_env_ori(self.env_ori._env)
        self.n_actions = 3
        self.n_raw_actions = 18
        self.max_objects = MAX_NB_OBJECTS_HUD if kwargs.get("hud", False) else MAX_NB_OBJECTS
        self.n_objects = sum(self.max_objects.values())
        self.n_features = 4  # visible, x-pos, y-pos, right-facing
        self.seed = seed

        # Compute index offsets. Needed to deal with multiple same-category objects
        self.obj_offsets = {}
        offset = 0
        for obj, max_count in self.max_objects.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(self.max_objects.keys())

    def reset(self):
        """
        Reset the environment.

        Returns:
            logic_state (torch.Tensor): Logic state of the environment.
            neural_state (torch.Tensor): Neural state of the environment.
        """
        raw_state, _ = self.env.reset(seed=self.seed)
        # self.raw_state_ori, _ = self.env_ori.reset(seed=self.seed)
        state = self.env.objects
        self.ocatari_state = state
        logic_state, neural_state = self.extract_logic_state(
            state
        ), self.extract_neural_state(raw_state)
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
        # self.raw_state_ori, _, _, _, _ = self.env_ori.step(action)
        state = self.env.objects
        self.ocatari_state = state
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
        # seve bboxes for exlanation rendering
        self.bboxes = th.zeros((self.n_objects, 4), dtype=th.int32)

        obj_count = {k: 0 for k in self.max_objects.keys()}

        for obj in input_state:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            if obj.category == "Time":
                state[idx] = th.Tensor([1, obj.value, 0, 0])
            else:
                orientation = (
                    (obj.orientation.value if hasattr(obj.orientation, 'value') else obj.orientation) if obj.orientation is not None else 0
                )
                state[idx] = th.tensor([1, *obj.center, orientation])
            obj_count[obj.category] += 1
            self.bboxes[idx] = th.tensor(obj.xywh)
        return state

    # def object_id_to_ocatari_object(self, object_id):
    #     # obj28 -> Ladder at (x, y), (h, w)
    #     passa

    def extract_neural_state(self, raw_input_state):
        """
        Extracts the neural state from the raw input state.
        Args:
            raw_input_state (torch.Tensor): Raw input state.
        Returns:
            torch.Tensor: Neural state.
        """
        return torch.Tensor(raw_input_state).unsqueeze(0)  # .float()

    def close(self):
        """
        Close the environment.
        """
        self.env.close()
