from __future__ import annotations

import glob
import inspect
import os
import pkgutil
import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, TypeVar, Type, Union, Dict, Iterable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NeuralBlenderActor(nn.Module):
    """
    Neural Blender Actor; 
    a neural network that takes an image as input and outputs a probability distribution over policies.
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, 2), std=0.01)
        
    def forward(self, x):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs
    
    
class CNNActor(nn.Module):
    """
    Neural Blender Actor;
    a neural network that takes an image as input and outputs a probability distribution over actions.
    """
    def __init__(self, n_actions=18, ):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def forward(self, x):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs

def get_blender(env, blender_rules, device, train=True, blender_mode='logic', reasoner='nsfr', explain=False, valuation_model=None, gamma: float = 0.01):
    """
    Load a Blender model. 
    Args:
        env (gym.Env): Environment.
        blender_rules (str): Path to Blender rules.
        device (torch.device): Device.
        train (bool): Whether to train the model.
        blender_mode (str): Mode of Blender. Possible values are "logic" and "neural".
        reasoner (str): Reasoner. Possible values are "nsfr" and "neumann".
        explain (bool): Whether to explain the model.
    Returns:
        Blender: Blender model.
    """
    assert blender_mode in ['logic', 'neural']
    if blender_mode == 'logic':
        if reasoner == 'nsfr':
            from nsfr.common import get_blender_nsfr_model
            return get_blender_nsfr_model(env.name, blender_rules, device, train=train, explain=explain, valuation_model=valuation_model, gamma=gamma)
        elif reasoner == 'neumann':
            from neumann.common import get_neumann_model, get_blender_neumann_model
            return get_blender_neumann_model(env.name, blender_rules, device, train=train, explain=explain)
    if blender_mode == 'neural':
        net = NeuralBlenderActor()
        net.to(device)
        return net
    
    
def load_cleanrl_envs(env_id, run_name=None, capture_video=False, num_envs=1):
    from cleanrl.cleanrl.ppo_atari import make_env
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, capture_video, run_name) for i in range(num_envs)],
    )
    return envs
    
def load_cleanrl_agent(pretrained, device):
    # from cleanrl.cleanrl.ppo_atari import Agent
    agent = CNNActor(n_actions=18) #, device=device, verbose=1)
    if pretrained:
        try:
            agent.load_state_dict(torch.load("cleanrl/out/ppo_Seaquest-v4_1.pth"))
            agent.to(device)
        except RuntimeError:
            agent.load_state_dict(torch.load("cleanrl/out/ppo_Seaquest-v4_1.pth", map_location=torch.device('cpu')))
    else:
        agent.to(device)
    return agent


def load_logic_ppo(agent, path):
    new_actor_dic = OrderedDict()
    new_critic_dic = OrderedDict()
    dic = torch.load(path)
    for name, value in dic.items():
        if 'actor.' in name:
            new_name = name.replace('actor.', '') 
            new_actor_dic[new_name] = value
        if 'critic.' in name:
            new_name = name.replace('critic.', '') 
            new_critic_dic[new_name] = value
    agent.logic_actor.load_state_dict(new_actor_dic)
    agent.logic_critic.load_state_dict(new_critic_dic)
    return agent


@dataclass
class Checkpoint:
    step: int
    path: Path


def get_all_checkpoints(checkpoints_dir: Path, sorted: bool = True) -> List[Checkpoint]:
    if not os.path.isdir(checkpoints_dir):
        return []

    checkpoint_filenames = os.listdir(checkpoints_dir)
    result = []
    pattern = re.compile("[0-9]+")
    for i, checkpoint_filename in enumerate(checkpoint_filenames):
        match = pattern.search(checkpoint_filename)
        if match is not None:
            step = int(match.group())
            path = checkpoints_dir / checkpoint_filename
            checkpoint = Checkpoint(step, path)
            result.append(checkpoint)

    if sorted:
        result.sort(key=lambda checkpoint: checkpoint.step)

    return result


def get_latest_checkpoint(checkpoints_dir: Path) -> Optional[Checkpoint]:
    checkpoints = get_all_checkpoints(checkpoints_dir, sorted=True)

    if len(checkpoints) > 0:
        return checkpoints[-1]

    return None


def load_module(*args, **kwargs):
    from nsfr.utils.common import load_module
    return load_module(*args, **kwargs)

T = TypeVar('T')
def optional(val: Optional[T], default: T) -> T:
    if val is None:
        return default

    return val


def get_default_device() -> torch.Device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_classes_in_package(package: str, subclass: Optional[Type[T]]) -> List[Type[T]]:
    classes = []

    # import each submodule
    for module in pkgutil.walk_packages([package]):
        submodule = load_module(f"{package}/{module.name}.py")

        for _, obj in inspect.getmembers(submodule, inspect.isclass):
            if (
                    (
                        subclass is None
                        or (
                            issubclass(obj, subclass)
                            and obj is not subclass
                        )
                    )
                    and obj.__module__ == submodule.__name__
            ):
                classes.append(obj)

    return classes

def get_model_state(checkpoint_path: Path, include_prefixes: Optional[List[str]] = None, exclude_prefixes: List[str] = [], discard_prefix_from_key: bool = False):
    with open(checkpoint_path, "rb") as f:
        device = torch.device('cpu')
        state_dict = torch.load(f, map_location=device, weights_only=True)

    new_state_dict = {}
    for k, v in state_dict.items():
        if any(k.startswith(exclude_prefix) for exclude_prefix in exclude_prefixes):
            continue

        if include_prefixes is None:
            new_state_dict[k] = v
        else:
            for include_prefix in include_prefixes:
                if k.startswith(include_prefix):
                    new_key = k if not discard_prefix_from_key else k[len(include_prefix):]
                    new_state_dict[new_key] = v
                    break

    return new_state_dict

def load_model_state(checkpoint_path: Path, model: any, strict: bool = False, include_prefixes: Optional[List[str]] = None, exclude_prefixes: List[str] = [], discard_prefix_from_key: bool = False) -> dict:
    new_state_dict = get_model_state(checkpoint_path, include_prefixes, exclude_prefixes, discard_prefix_from_key)
    model.load_state_dict(state_dict=new_state_dict, strict=strict)
    return new_state_dict


def save_model_state(model: nn.Module, checkpoint_path: Path, prefixes: List[str] = []):
    if len(prefixes) > 0:
        state_dict = {
            k: v for k, v in model.state_dict().items() if any(k.startswith(prefix) for prefix in prefixes)
        }
    else:
        state_dict = model.state_dict()

    torch.save(state_dict, checkpoint_path)


def reset_parameters(model: torch.nn.Module):
    for mod in model.modules():
        if hasattr(mod, 'reset_parameters'):
            mod.reset_parameters()


def masked_softmax(x, mask, dim, nan_fill = 0.0, *args, **kwargs):
    x = x.masked_fill(~mask, float('-inf'))
    y = torch.softmax(x, dim=dim, *args, **kwargs)
    y = y.nan_to_num(nan_fill)
    return y


def normalize(arr: Union[torch.Tensor, np.ndarray], eps: float = 1e-8) -> Union[torch.Tensor, np.ndarray]:
    return (arr - arr.mean()) / (arr.std() + eps)

FRAME_SIZE = {
    "seaquest": (160.0, 210.0),
    "kangaroo": (160.0, 210.0),
    "donkeykong": (160.0, 210.0)
}

DEFAULT_MODIFICATIONS = {
    "seaquest": [],
    "kangaroo": [
        "disable_coconut",
        "randomize_kangaroo_position",
        "change_level_0",
    ],
    "donkeykong": [
        "random_start",
        "change_level_0"
    ]
}


def to_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def running_average(arr: np.ndarray, window: int, center: bool = False) -> np.ndarray:
    result = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        if not center:
            start = max(0, i - window + 1)
            end = i+1
        else:
            start = max(0, i - window // 2)
            end = min(len(arr), i + window // 2)
        result[i] = np.mean(arr[start:end])
    return result


def round_decimals(x: torch.Tensor, decimals: int = 3) -> torch.Tensor:
    return torch.round(x * (10 ** decimals)) / (10 ** decimals)


class ArrayIO:
    """
    A helper class for chunked storage of numpy arrays with compression.

    This class manages large datasets by storing them in compressed chunks,
    allowing for efficient appending and retrieval operations.
    """

    def __init__(self, dir: Union[str, Path], chunk_size: int = 65536):
        """
        Initialize ArrayIO with directory and chunk size.

        Args:
            dir: Directory path where chunks will be stored
            chunk_size: Number of elements along dimension 0 before saving a chunk
        """
        self.dir = Path(dir)
        self.chunk_size = chunk_size
        self.buffers: Dict[str, np.ndarray] = {}
        self.chunk_counts: Dict[str, int] = {}

        # Create base directory if it doesn't exist
        self.dir.mkdir(parents=True, exist_ok=True)

        # Initialize chunk counts by scanning existing files
        self._initialize_chunk_counts()

    def _initialize_chunk_counts(self):
        """Initialize chunk counts by scanning existing chunk files."""
        for key_dir in self.dir.iterdir():
            if key_dir.is_dir():
                key = key_dir.name
                # Find all .npz files and get the highest numbered chunk
                chunk_files = glob.glob(str(key_dir / "*.npz"))
                if chunk_files:
                    chunk_nums = []
                    for f in chunk_files:
                        try:
                            chunk_num = int(Path(f).stem)
                            chunk_nums.append(chunk_num)
                        except ValueError:
                            continue
                    if chunk_nums:
                        self.chunk_counts[key] = max(chunk_nums) + 1
                    else:
                        self.chunk_counts[key] = 0
                else:
                    self.chunk_counts[key] = 0

    def _get_chunk_path(self, key: str, chunk_idx: int) -> Path:
        """Get the file path for a specific chunk."""
        key_dir = self.dir / key
        key_dir.mkdir(parents=True, exist_ok=True)
        return key_dir / f"{chunk_idx}.npz"

    def _save_chunk(self, key: str, array: np.ndarray):
        """Save a chunk to disk with compression."""
        chunk_idx = self.chunk_counts.get(key, 0)
        chunk_path = self._get_chunk_path(key, chunk_idx)

        # Save with compression
        np.savez_compressed(chunk_path, data=array)

        # Update chunk count
        self.chunk_counts[key] = chunk_idx + 1

    def _load_chunk(self, key: str, chunk_idx: int) -> Optional[np.ndarray]:
        """Load a specific chunk from disk."""
        chunk_path = self._get_chunk_path(key, chunk_idx)
        if chunk_path.exists():
            with np.load(chunk_path) as data:
                return data['data']
        return None

    def append(self, key: str, array: np.ndarray):
        """
        Append a numpy array to the dataset identified by key.

        Args:
            key: Dataset identifier
            array: Numpy array to append (concatenated along dimension 0)
        """
        if not isinstance(array, np.ndarray):
            raise ValueError("Input must be a numpy array")

        if array.ndim == 0:
            array = np.expand_dims(array, 0)
            array = np.expand_dims(array, 1)

        if array.ndim == 1:
            array = np.expand_dims(array, 1)

        # Initialize buffer if it doesn't exist
        if key not in self.buffers:
            self.buffers[key] = array.copy()
            if key not in self.chunk_counts:
                self.chunk_counts[key] = 0
        else:
            # Check shape compatibility (all dimensions except 0 must match)
            if array.shape[1:] != self.buffers[key].shape[1:]:
                raise ValueError(f"Shape mismatch: existing {self.buffers[key].shape[1:]} vs new {array.shape[1:]}")

            # Concatenate along dimension 0
            self.buffers[key] = np.concatenate([self.buffers[key], array], axis=0)

        # Check if we need to save chunks
        while self.buffers[key].shape[0] >= self.chunk_size:
            # Extract chunk
            chunk = self.buffers[key][:self.chunk_size]

            # Save chunk
            self._save_chunk(key, chunk)

            # Keep remaining data in buffer
            if self.buffers[key].shape[0] == self.chunk_size:
                # Buffer is exactly chunk_size, clear it
                del self.buffers[key]
                break
            else:
                # Keep remaining data
                self.buffers[key] = self.buffers[key][self.chunk_size:]

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Retrieve the complete dataset for a given key by loading and concatenating all chunks.

        Args:
            key: Dataset identifier

        Returns:
            Complete numpy array for the key
        """
        if key not in self.chunk_counts:
            raise KeyError(f"Key '{key}' not found")

        chunks = []

        # Load all saved chunks
        for i in range(self.chunk_counts[key]):
            chunk = self._load_chunk(key, i)
            if chunk is not None:
                chunks.append(chunk)

        # Add buffered data if it exists
        if key in self.buffers:
            chunks.append(self.buffers[key])

        if not chunks:
            raise ValueError(f"No data found for key '{key}'")

        # Concatenate all chunks
        return np.concatenate(chunks, axis=0)

    def close(self):
        """
        Save any remaining buffered data as final chunks.
        """
        for key, buffer in self.buffers.items():
            if buffer.size > 0:  # Only save non-empty buffers
                self._save_chunk(key, buffer)

        # Clear buffers after saving
        self.buffers.clear()

    def keys(self) -> list:
        """Return list of all dataset keys."""
        return list(self.chunk_counts.keys())

    def get_info(self, key: str) -> dict:
        """
        Get information about a dataset.

        Args:
            key: Dataset identifier

        Returns:
            Dictionary with dataset information
        """
        if key not in self.chunk_counts:
            raise KeyError(f"Key '{key}' not found")

        info = {
            'key': key,
            'num_chunks': self.chunk_counts[key],
            'buffered_size': self.buffers[key].shape[0] if key in self.buffers else 0,
        }

        # Get shape info from first chunk or buffer
        sample = None
        if self.chunk_counts[key] > 0:
            sample = self._load_chunk(key, 0)
        elif key in self.buffers:
            sample = self.buffers[key]

        if sample is not None:
            info['shape'] = sample.shape[1:]  # Shape excluding first dimension
            info['dtype'] = sample.dtype

        return info

    def __len__(self) -> int:
        """Return number of datasets."""
        return len(self.chunk_counts)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in the dataset."""
        return key in self.chunk_counts

    def __repr__(self) -> str:
        return f"ArrayIO(dir='{self.dir}', chunk_size={self.chunk_size}, datasets={len(self)})"


class ParameterSummary:

    def __init__(self):
        self.infos = dict()

    def set_from_state_dict(self, state_dict: dict, prefix: Optional[str] = None, **kwargs):
        for key, param in state_dict.items():
            fq_key = optional(prefix, "") + key
            self.infos[fq_key] = {**self.infos.get(fq_key, {}), **kwargs, "shape": list(param.shape)}

    def set_from_prefix(self, prefixes: Iterable[str], **kwargs):
        for key, info in self.infos.items():
            if any(key.startswith(prefix) for prefix in prefixes):
                self.infos[key] = {**info, **kwargs}

    @property
    def table(self):
        from prettytable import PrettyTable

        field_names = ["Parameter", "Shape", "Checkpoint", "Frozen", "Used"]
        table = PrettyTable()
        table.field_names = field_names
        table.align = "l"

        sorted_keys = sorted(self.infos.keys())
        bool_str = lambda b: "Yes" if b else "No"
        for key in sorted_keys:
            info = self.infos[key]
            row = [
                key,
                optional(info["shape"], "n/a"),
                optional(info["checkpoint_path"], "n/a"),
                bool_str(info.get("frozen", False)),
                bool_str(info.get("used", False))
            ]

            table.add_row(row)

        return table

    def print(self):
        table = self.table
        print(table)

    def save_as_csv(self, path: Path):
        table = self.table
        with open(str(path), "w") as f:
            f.write(table.get_csv_string(delimiter=";"))
