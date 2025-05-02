from typing import Sequence
import torch
from nudge.env import NudgeBaseEnv
from ocatari.core import OCAtari
from hackatari.core import HackAtari
import numpy as np
import torch as th
from packaging import version
# from ocatari.ram.riverraid import MAX_NB_OBJECTS
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


from utils import load_cleanrl_envs

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

MAX_NB_OBJECTS = {
    "Player": 1,
    "PlayerMissile": 1,
    "FuelDepot": 4,
    "Tanker": 4,
    "Helicopter": 4,
    "House": 4,
    "Jet": 4,
    "Bridge": 1,
}
MAX_NB_OBJECTS_HUD = dict(MAX_NB_OBJECTS, **{"PlayerScore": 1, "Lives": 1})

"""gym.envs.registration.register(
    id="ALE/Riverraid-v5",
    entry_point="gymnasium.envs.atari:AtariEnv",
    kwargs={"rom_file": "roms/River Raid (1982) (Activision, Carol Shaw) (AX-020, AX-020-04) ~.bin"},
)"""

GYMNASIUM_VERSION = version.parse(gym.__version__)

if GYMNASIUM_VERSION <= version.parse("0.30.0"):
    def make_env(env):
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AutoResetWrapper(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env
else:
    def make_env(env):
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.Autoreset(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env


class NudgeEnv(NudgeBaseEnv):
    name = "riverraid"
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
        self, mode: str, render_mode="rgb_array", render_oc_overlay=False, seed=None
    ):
        super().__init__(mode)
        self.env = HackAtari(
            env_name="ALE/Riverraid",
            mode="ram",
            obs_mode="ori",
            modifs=[("LinearRiver"),],
            rewardfunc_path="in/envs/riverraid/blenderl_reward.py",
            render_mode=render_mode,
            render_oc_overlay=render_oc_overlay,
        )
        self.env._env = make_env(self.env._env)
        self.n_actions = 6
        self.n_raw_actions = 18
        self.n_objects = 24  # len(MAX_NB_OBJECTS)
        self.n_features = 4
        self.seed = seed

        self.obj_offsets = {}
        offset = 0
        for obj, max_count in MAX_NB_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_NB_OBJECTS.keys())

    def reset(self):
        raw_state, _ = self.env.reset(seed=self.seed)
        state = self.env.objects
        self.ocatari_state = state
        logic_state, neural_state = self.extract_logic_state(
            state
        ), self.extract_neural_state(raw_state)
        return logic_state.unsqueeze(0), neural_state

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
            if obj.category == "Time":
                state[idx] = th.tensor([1, obj.value, 0, 0])
            else:
                #print(obj.orientation)
                orientation = (
                    obj.orientation if obj.orientation is not None else 0
                )
                state[idx] = th.tensor([1, *obj.center, orientation])
            obj_count[obj.category] += 1
            self.bboxes[idx] = th.tensor(obj.xywh)
        return state

    def extract_neural_state(self, raw_input_state):
        return torch.tensor(raw_input_state, dtype=torch.float32).unsqueeze(0)

    def close(self):
        self.env.close()
