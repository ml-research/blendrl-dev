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
    load_model_state, DEFAULT_MODIFICATIONS, optional
from valuation.config import ValuationConfig, load_model_config_class
from valuation.models.base import BaseValuationModelConfig, BaseValuationModel
from valuation.utils import load_model_class, get_default_valuation_model


class ValuationExperiment:

    base_dir = Path("out_val/runs")

    def __init__(self, dir: Path):
        self.dir = dir
        self.checkpoints_dir = self.dir / "checkpoints"
        self.images_dir = self.dir / "images"
        self.plots_dir = self.dir / "plots"
        self.config_path = self.dir / "config.yaml"
        self.logs_path = self.dir / "logs.json"
        self.logs_dir = self.dir / "logs"
        self.sim_path = self.dir / "sim.npz"

        self.name = self.dir.name

        self._model = None
        self._oracle = None

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
        os.makedirs(self.logs_dir, exist_ok=True)

    def _load_config(self, config: dict) -> Optional[ValuationConfig]:
        for valuation_model_field in ("valuation_model", "oracle_model"):
            model_type = optional(config.get(valuation_model_field), {}).get("type")
            valuation_model_config = None
            if model_type is not None:
                model_config_cls = load_model_config_class(model_type)
                model_params = config[valuation_model_field]
                del model_params["type"]
                valuation_model_config = model_config_cls(**model_params)

            config[valuation_model_field] = valuation_model_config

        args = ValuationConfig(**config)
        return args

    def load_config(self) -> Optional[ValuationConfig]:
        if not self.config_path.exists():
            return None

        with open(self.config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

        if config is None:
            return None

        return self._load_config(config)

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
    def language(self) -> Language:
        return get_language(self.env_name, self.config.rules)

    @property
    def valuation_model_type(self) -> Optional[str]:
        if self.config.valuation_model is not None:
            return self.config.valuation_model.type

        return None

    @property
    def valuation_model_config(self) -> BaseValuationModelConfig:
        return self.config.valuation_model

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

    def _get_valuation_model(self, config: BaseValuationModelConfig, device: torch.device) -> Optional[BaseValuationModel]:
        if config is None or config.type is None:
            return None
        else:
            model_cls = load_model_class(config.type)
            assert model_cls is not None, f"No valuation model of type '{config.type}' found"
            return model_cls(env_name=self.env_name, lang=self.language, config=config, device=device)

    def get_model(self, device: torch.device, load_from_latest_checkpoint: bool = True) -> Union[NsfrActorCritic, ActorCritic, BlenderActorCritic]:
        if self._model is not None:
            return self._model

        valuation_model = self._get_valuation_model(self.valuation_model_config, device)
        if valuation_model is None:
            valuation_model = self.get_default_valuation_model(device)

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
                load_model_state(checkpoint.path, model)

        if self.config.logic_critic_path is not None:
            logic_critic_experiment = ValuationExperiment.from_path(Path(self.config.logic_critic_path))
            logic_critic_checkpoint_path = logic_critic_experiment.latest_checkpoint.path
            load_model_state(logic_critic_checkpoint_path, model.logic_critic, include_prefixes=["logic_critic."], discard_prefix_from_key=True)

        self._model = model
        return model

    def get_oracle(self, device: torch.device) -> Optional[BaseValuationModel]:
        if self._oracle is not None:
            return self._oracle

        self._oracle = self._get_valuation_model(self.config.oracle_model, device)
        return self._oracle

    @property
    def checkpoints(self) -> List[Checkpoint]:
        return get_all_checkpoints(self.checkpoints_dir)

    @property
    def latest_checkpoint(self) -> Optional[Checkpoint]:
        return get_latest_checkpoint(self.checkpoints_dir)


