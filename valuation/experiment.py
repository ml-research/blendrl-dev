from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import torch
import yaml
from dataclasses import asdict
from torch import nn as nn

from blendrl.agents.blender_agent import BlenderActorCritic
from nsfr.fol.language import Language
from nsfr.utils.common import get_language
from nudge.agents.logic_agent import NsfrActorCritic
from nudge.agents.neural_agent import ActorCritic
from nudge.utils import load_model
from utils import Checkpoint, get_all_checkpoints, get_latest_checkpoint, \
    load_model_state, DEFAULT_MODIFICATIONS
from valuation.config import ValuationConfig
from valuation.models.base import BaseValuationModelConfig
from valuation.utils import load_model_config_class, load_model_class, get_default_valuation_model


class ValuationExperiment:

    base_dir = Path("out_val/runs")

    def __init__(self, dir: Path):
        self.dir = dir
        self.checkpoints_dir = self.dir / "checkpoints"
        self.images_dir = self.dir / "images"
        self.plots_dir = self.dir / "plots"
        self.config_path = self.dir / "config.yaml"
        self.logs_path = self.dir / "logs.json"
        self.sim_path = self.dir / "sim.npz"

        self.name = self.dir.name

        self._model = None
        self._valuation_model = None

        # Load config
        self.config = self.load_config()

    @staticmethod
    def from_name(name: str) -> ValuationExperiment:
        return ValuationExperiment(ValuationExperiment.base_dir / name)

    @staticmethod
    def from_path(path: Path) -> ValuationExperiment:
        return ValuationExperiment(path)

    @staticmethod
    def get_all(type: Optional[str] = None) -> List[ValuationExperiment]:
        experiment_dirs = list(ValuationExperiment.base_dir.iterdir())
        experiments = []
        for experiment_dir in experiment_dirs:
            if experiment_dir.is_dir():
                experiment = ValuationExperiment(experiment_dir)
                if type is None or type == experiment.valuation_model_type:
                    experiments.append(experiment)
        return experiments


    def init(self):
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def load_config(self) -> Optional[ValuationConfig]:
        if not self.config_path.exists():
            return None

        with open(self.config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

        model_type = config.get("valuation_model", {}).get("type")
        valuation_model_config = None
        if model_type is not None:
            model_config_cls = load_model_config_class(model_type)
            model_params = config["valuation_model"]
            del model_params["type"]
            valuation_model_config = model_config_cls(**model_params)

        config["valuation_model"] = valuation_model_config

        args = ValuationConfig(**config)
        return args

    def update_config(self, config: ValuationConfig, print_config=False):
        self.config = config

        with open(self.config_path, "w") as f:
            yaml.dump(asdict(self.config), f)

        if print_config:
            print("Hyperparameters:")
            print(open(self.config_path).read())

    @property
    def env_name(self) -> str:
        return self.config.env_name

    @property
    def valuation_model_type(self) -> Optional[str]:
        if self.config.valuation_model is not None:
            return self.config.valuation_model.type

        return None

    @property
    def valuation_model_config(self) -> BaseValuationModelConfig:
        return self.config.valuation_model

    @property
    def language(self) -> Language:
        return get_language(self.env_name, self.config.rules)

    def get_default_valuation_model(self, device: torch.device) -> nn.Module:
        return get_default_valuation_model(self.env_name, device)

    @property
    def env_config(self) -> dict:
        env_kwargs = {
            "modifications": DEFAULT_MODIFICATIONS[self.env_name] + self.config.extra_env_modifications,
            "frameskip": self.config.env_frameskip,
            "reward_fn_path": f"in/envs/{self.env_name}/reward/{self.config.reward_fn}.py",
        }
        if self.config.env_max_ep_steps is not None:
            env_kwargs["max_episode_steps"] = self.config.env_max_ep_steps

        return env_kwargs

    def get_model(self, device: torch.device, load_from_latest_checkpoint: bool = True) -> Union[NsfrActorCritic, ActorCritic, BlenderActorCritic]:
        if self._model is not None:
            return self._model

        valuation_model_type = self.valuation_model_type

        if valuation_model_type is None:
            valuation_model = self.get_default_valuation_model(device)
        else:
            model_cls = load_model_class(valuation_model_type)
            assert model_cls is not None, f"No valuation model of type '{valuation_model_type}' found"
            valuation_model = model_cls(env_name=self.env_name, lang=self.language, config=self.valuation_model_config, device=device)

        model = load_model(
            self.config.agent_path,
            env_kwargs_override=self.env_config,
            device=device,
            valuation_model=valuation_model,
            config_override=asdict(self.config)
        )

        if load_from_latest_checkpoint:
            checkpoint = self.latest_checkpoint

            if checkpoint is not None:
                load_model_state(checkpoint.path, model, strict=False)

        self._model = model
        return model

    @property
    def checkpoints(self) -> List[Checkpoint]:
        return get_all_checkpoints(self.checkpoints_dir)

    @property
    def latest_checkpoint(self) -> Optional[Checkpoint]:
        return get_latest_checkpoint(self.checkpoints_dir)


