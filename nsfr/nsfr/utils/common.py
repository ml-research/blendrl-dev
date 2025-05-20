import os
from typing import Optional

import torch
import numpy as np
import importlib.util
import sys

from nsfr.fol.data_utils import DataUtils
from nsfr.fol.language import Language


def explaining_nsfr(NSFR, extracted_states):
    V_T = NSFR(extracted_states)
    # prednames = NSFR.prednames
    predicts = NSFR.predict_multi(v=V_T)
    explaining = NSFR.print_explaining(predicts)
    return explaining


def get_predictions(extracted_states, NSFR):
    predictions = NSFR(extracted_states)
    predicts = NSFR.print_explaining(predictions)
    return predicts


def extract_for_cgen_explaining(coin_jump):
    """
    extract state to metric
    input: coin_jump instance
    output: extracted_state to be explained

    x:  agent, key, door, enemy, position_X, position_Y
    y:  obj1(agent), obj2(key), obj3(door)，obj4(enemy)

    To be changed when using object-detection tech
    """
    # TODO
    num_of_feature = 6
    num_of_object = 4
    representation = coin_jump.level.get_representation()
    extracted_states = np.zeros((num_of_object, num_of_feature))
    for entity in representation["entities"]:
        if entity[0].name == 'PLAYER':
            extracted_states[0][0] = 1
            extracted_states[0][-2:] = entity[1:3]
            # 27 is the width of map, this is normalization
            # extracted_states[0][-2:] /= 27
        elif entity[0].name == 'KEY':
            extracted_states[1][1] = 1
            extracted_states[1][-2:] = entity[1:3]
            # extracted_states[1][-2:] /= 27
        elif entity[0].name == 'DOOR':
            extracted_states[2][2] = 1
            extracted_states[2][-2:] = entity[1:3]
            # extracted_states[2][-2:] /= 27
        elif entity[0].name == 'GROUND_ENEMY':
            extracted_states[3][3] = 1
            extracted_states[3][-2:] = entity[1:3]
            # extracted_states[3][-2:] /= 27

    if sum(extracted_states[:, 1]) == 0:
        key_picked = True
    else:
        key_picked = False

    def simulate_prob(extracted_states, num_of_objs, key_picked):
        for i, obj in enumerate(extracted_states):
            obj = add_noise(obj, i, num_of_objs)
            extracted_states[i] = obj
        if key_picked:
            extracted_states[:, 1] = 0
        return extracted_states

    def add_noise(obj, index_obj, num_of_objs):
        mean = torch.tensor(0.1)
        std = torch.tensor(0.05)
        noise = torch.abs(torch.normal(mean=mean, std=std)).item()
        rand_noises = torch.randint(1, 5, (num_of_objs - 1,)).tolist()
        rand_noises = [i * noise / sum(rand_noises) for i in rand_noises]
        rand_noises.insert(index_obj, 1 - noise)

        for i, noise in enumerate(rand_noises):
            obj[i] = rand_noises[i]
        return obj

    extracted_states = simulate_prob(extracted_states, num_of_object, key_picked)

    return torch.tensor(extracted_states, device="cuda:0")


def load_module(path: str, _module_name: Optional[str] = None):
    if _module_name is None:
        module_name = path.replace(os.path.sep, ".")
        if module_name.endswith(".py"):
            module_name = module_name[:-3]
    else:
        module_name = _module_name

    # check if module has already been loaded
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def bool_to_probs(bool_tensor: torch.Tensor):
    """Converts the values of a tensor from Boolean to probability values by
     slightly 'smoothing' them (1 to 0.99 and 0 to 0.01)."""
    return torch.where(bool_tensor, 0.99, 0.01)


def get_language(env_name: str, dataset: str = "default") -> Language:
    du = DataUtils(
        lark_path="nsfr/nsfr/lark/exp.lark",
        lang_base_path=f"in/envs/{env_name}/logic/",
        dataset=dataset
    )
    return du.load_language()