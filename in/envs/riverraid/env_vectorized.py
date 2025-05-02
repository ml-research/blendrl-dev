import time
from typing import Sequence
import torch
from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from hackatari.core import HackAtari
import torch as th
from packaging import version
# from ocatari.ram.riverraid import MAX_NB_OBJECTS, MAX_NB_OBJECTS_HUD
import gymnasium as gym

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
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


# python train_blenderl.py --env-name riverraid --joint-training --num-steps 16 --num-envs 1 --gamma 0.99
class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
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
        self,
        mode: str,
        n_envs: int,
        render_mode="rgb_array",
        render_oc_overlay=False,
        seed=None,
    ):
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
        # set up multiple envs
        self.n_envs = n_envs
        # initialize each HackAtari environment
        self.envs = [
            HackAtari(
                env_name="ALE/Riverraid-v5",
                mode="ram",
                obs_mode="ori",
                modifs=[("random_init"), ("change_difficulty")],
                rewardfunc_path="in/envs/riverraid/blenderl_reward.py",
                render_mode=render_mode,
                render_oc_overlay=render_oc_overlay,
            )
            for i in range(n_envs)
        ]
        # apply wrapper to _env
        for i in range(n_envs):
            self.envs[i]._env = make_env(self.envs[i]._env)

        self.n_actions = 6
        self.n_raw_actions = 18
        self.n_objects = 24  # len(MAX_NB_OBJECTS)
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
        """
        Reset the environment.

        Returns:
            logic_states (torch.Tensor): Logic states.
            neural_states (torch.Tensor): Neural states.
        """
        logic_states = []
        neural_states = []
        seed_i = self.seed
        print("Env is being reset...")
        for env in self.envs:
            obs, _ = env.reset(seed=seed_i)
            obs = torch.tensor(obs).float()
            state = env.objects
            raw_state = obs
            logic_state, neural_state = self.extract_logic_state(
                state
            ), self.extract_neural_state(raw_state)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            seed_i += 1
        print("Env reset is done.")
        return torch.stack(logic_states), torch.stack(neural_states)

    def step(self, actions, is_mapped=False):
        """
        Perform a step in the environment.

        Args:
            actions (torch.Tensor): Actions to be performed in the environment.
            is_mapped (bool): Whether the actions are already mapped.
        Returns:
            Tuple: Tuple containing:
                - torch.Tensor: Observations.
                - list: Rewards.
                - list: Truncations.
                - list: Dones.
                - list: Infos.
        """
        assert (
            len(actions) == self.n_envs
        ), "Invalid number of actions: n_actions is {} and n_envs is {}".format(
            len(actions), self.n_envs
        )
        observations = []
        rewards = []
        truncations = []
        dones = []
        infos = []
        logic_states = []
        neural_states = []

        start = time.time()
        for i, env in enumerate(self.envs):
            action = actions[i]
            obs, reward, truncation, done, info = env.step(action)
            raw_state = torch.tensor(obs).float()
            state = env.objects
            logic_state, neural_state = self.convert_state(state, raw_state)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            observations.append(obs)
            rewards.append(reward)
            truncations.append(truncation)
            dones.append(done)
            infos.append(info)
        end = time.time()

        #print("STEP OUTPUT:", type(logic_states), type(neural_states), type(rewards), type(truncations), type(dones), type(infos))
        return (
            (torch.stack(logic_states), torch.stack(neural_states)),
            rewards,
            truncations,
            dones,
            infos,
        )

    def extract_logic_state(self, raw_state):
        """
        Extracts the logic state from the input state.
        Args:
            raw_state (list): List of objects in the environment.
        Returns:
            torch.Tensor: Logic state.
        """
        state = th.zeros((self.n_objects, self.n_features), dtype=th.float32)
        obj_count = {k: 0 for k in MAX_NB_OBJECTS.keys()}

        for obj in raw_state:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]

            # Ensure idx is within bounds
            if idx >= self.n_objects:
                print(
                    f"Warning: Object index {idx} exceeds limit ({self.n_objects}). Skipping {obj.category}."
                )
                continue

            x, y = obj.center
            orientation = obj.orientation if obj.orientation is not None else 0

            state[idx] = th.tensor(
                [1.0, float(x), float(y), float(orientation)], dtype=th.float32
            )
            obj_count[obj.category] += 1

        return state

    def extract_neural_state(self, raw_input_state):
        """
        Extracts the neural state from the raw input state.
        Args:
            raw_input_state (torch.Tensor): Raw input state.
        Returns:
            torch.Tensor: Neural state.
        """
        return raw_input_state.to(dtype=torch.float32)

    def close(self):
        """
        Close the environment.
        """
        for env in self.envs:
            env.close()
