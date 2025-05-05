from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Type

import torch
import yaml
from torch import nn as nn

from nsfr.fol.data_utils import DataUtils
from nsfr.fol.language import Language
from utils import load_module, load_classes_in_package, Checkpoint, get_all_checkpoints, get_latest_checkpoint, \
    load_model_state
from valuation.models.base import BaseValuationModel, BaseValuationModelConfig


class ValuationExperiment:

    base_dir = Path("out_val/runs")

    def __init__(self, dir: Path):
        self.dir = dir
        self.checkpoints_dir = self.dir / "checkpoints"
        self.images_dir = self.dir / "images"
        self.plots_dir = self.dir / "plots"
        self.config_path = self.dir / "config.yaml"
        self.logs_path = self.dir / "logs.json"

        self.name = self.dir.name

    @staticmethod
    def from_name(name: str) -> ValuationExperiment:
        return ValuationExperiment(ValuationExperiment.base_dir / name)

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

    @property
    def config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return yaml.load(f, Loader=yaml.Loader)


        return {}

    def update_config(self, config: dict, print_config=False):
        new_config = self.config
        new_config.update(config)

        with open(self.config_path, "w") as f:
            yaml.dump(new_config, f)

        if print_config:
            print("Hyperparameters:")
            print(open(self.config_path).read())


    @property
    def env_name(self) -> str:
        return self.config["env_name"]

    @property
    def valuation_model_type(self) -> str:
        return self.config.get("valuation_model", {}).get("type")

    @property
    def valuation_model_config(self) -> Optional[BaseValuationModelConfig]:
        model_type = self.valuation_model_type
        if model_type is None:
            return None
        model_config_cls = load_model_config_class(model_type)
        model_params = self.config["valuation_model"]
        del model_params["type"]
        model_config = model_config_cls(**model_params)
        return model_config

    @property
    def language(self) -> Language:
        du = DataUtils(
            lark_path="nsfr/nsfr/lark/exp.lark",
            lang_base_path=f"in/envs/{self.env_name}/logic/",
            dataset=self.config.get("rules", "default")
        )
        return du.load_language()

    def get_valuation_model(self, device: torch.device, load_from_latest_checkpoint: bool = True) -> nn.Module:
        model_type = self.valuation_model_type

        if model_type is None:
            return self.get_default_valuation_model(device)

        model_cls = load_model_class(model_type)
        assert model_cls is not None, f"No valuation model of type '{model_type}' found"
        model = model_cls(env_name=self.env_name, lang=self.language, config=self.valuation_model_config, device=device)

        if load_from_latest_checkpoint:
            checkpoint = self.latest_checkpoint

            if checkpoint is not None:
                load_model_state(checkpoint.path, model)

        return model

    def get_default_valuation_model(self, device: torch.device) -> nn.Module:
        return get_default_valuation_model(self.env_name, device)

    @property
    def checkpoints(self) -> List[Checkpoint]:
        return get_all_checkpoints(self.checkpoints_dir)

    @property
    def latest_checkpoint(self) -> Optional[Checkpoint]:
        return get_latest_checkpoint(self.checkpoints_dir)


def load_model_config_classes() -> List[Type[BaseValuationModelConfig]]:
    return load_classes_in_package("valuation/models", BaseValuationModelConfig)

def load_model_config_class(type: str) -> Optional[Type[BaseValuationModelConfig]]:
    classes = load_model_config_classes()
    for cls in classes:
        if cls.type == type:
            return cls
    return None

def load_model_classes() -> List[Type[BaseValuationModel]]:
    return load_classes_in_package("valuation/models", BaseValuationModel)

def load_model_class(type: str) -> Optional[Type[BaseValuationModel]]:
    classes = load_model_classes()
    for cls in classes:
        if cls.type == type:
            return cls
    return None

def get_default_valuation_model(env_name: str, device: torch.device) -> nn.Module:
    module = load_module(f"in/envs/{env_name}/valuation.py")
    model = nn.Module()
    model.forward = lambda predicate_name, *inputs: getattr(module, predicate_name)(*inputs)
    model = model.to(device)
    return model