import math
import os
import random
import re
from functools import reduce
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from nsfr.nsfr import NSFReasoner
from nsfr.utils.torch import softor
from nudge.env import NudgeBaseEnv
from utils import optional, load_model_state, get_default_device
from valuation.models.base import BaseValuationModel


def to_proportion(dic):
    # Using reduce to get the sum of all values in the dictionary
    temp = reduce(lambda x, y: x + y, dic.values())
 
    # Using dictionary comprehension to divide each value by the sum of all values
    res = {k: v / temp for k, v in dic.items()}
    return res

def get_action_stats(env, actions):
    env_actions = env.pred2action.keys()
    frequency_dic = {}
    for action in env_actions:
        frequency_dic[action] = 0
        
    for i, action in enumerate(actions):
        frequency_dic[action] += 1
    
    action_proportion = to_proportion(frequency_dic)
    return action_proportion

def save_hyperparams(args, save_path, print_summary: bool = False):
    hyperparams = {}
    for key, value in vars(args).items():
        hyperparams[key] = value #local_scope[param]
    with open(save_path, 'w') as f:
        yaml.dump(hyperparams, f)
    if print_summary:
        print("Hyperparameter Summary:")
        print(open(save_path).read())


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def add_noise(obj, index_obj, num_of_objs):
    mean = torch.tensor(0.2)
    std = torch.tensor(0.05)
    noise = torch.abs(torch.normal(mean=mean, std=std)).item()
    rand_noises = torch.randint(1, 5, (num_of_objs - 1,)).tolist()
    rand_noises = [i * noise / sum(rand_noises) for i in rand_noises]
    rand_noises.insert(index_obj, 1 - noise)

    for i, noise in enumerate(rand_noises):
        obj[i] = rand_noises[i]
    return obj


def simulate_prob(extracted_states, num_of_objs, key_picked):
    for i, obj in enumerate(extracted_states):
        obj = add_noise(obj, i, num_of_objs)
        extracted_states[i] = obj
    if key_picked:
        extracted_states[:, 1] = 0
    return extracted_states


def build_model(model_dir: Path,
               env_kwargs_override: Optional[dict] = None,
               step: Optional[int] = None,
               device: Optional[torch.device] = None,
               explain: bool = False,
               valuation_model: Optional[BaseValuationModel] = None,
               config_override: Optional[dict] = None
               ):

    device = optional(device, get_default_device())

    # Determine all relevant paths
    model_dir = Path(model_dir)
    config_path = model_dir / "config.yaml"
    checkpoint_dir = model_dir / "checkpoints"
    if step is None:
        most_recent_step = get_most_recent_checkpoint_step(checkpoint_dir)
    else:
        most_recent_step = step
    checkpoint_path = checkpoint_dir / f"step_{most_recent_step}.pth"

    print("Loading model from", checkpoint_path)

    # Load model's configuration
    with open(config_path, "r") as f:
        config: dict = yaml.load(f, Loader=yaml.Loader)
        config.update(optional(config_override, {}))

    algorithm = config["algorithm"]
    environment = config["env_name"]

    # Setup the environment
    env = NudgeBaseEnv.from_name(environment, mode=algorithm, **optional(env_kwargs_override, {}))

    rules = config["rules"]

    # Initialize the model
    if algorithm == 'ppo':
        from nudge.agents.neural_agent import ActorCritic

        model = ActorCritic(env).to(device)
    elif algorithm == 'logic':
        from nudge.agents.logic_agent import NsfrActorCritic

        model = NsfrActorCritic(env, device=device, rules=rules, valuation_model=valuation_model).to(device)
    else:
        from blendrl.agents.blender_agent import BlenderActorCritic

        reasoner = config.get("reasoner", "nsfr")
        model = BlenderActorCritic(
            env, rules=rules, actor_mode=config["actor_mode"], blender_mode=config["blender_mode"],
            blend_function=config["blend_function"], reasoner=reasoner, device=device, explain=explain,
            valuation_model=valuation_model,
            gamma=config["softor_gamma"]
        ).to(device)

    return model, config, checkpoint_path

def load_model(model_dir: Path,
               env_kwargs_override: Optional[dict] = None,
               step: Optional[int] = None,
               device: Optional[torch.device] = None,
               explain: bool = False,
               valuation_model: Optional[BaseValuationModel] = None,
               config_override: Optional[dict] = None
               ):

    model, config, checkpoint_path = build_model(model_dir, env_kwargs_override, step, device, explain, valuation_model, config_override)

    # Load the model weights
    try:
        exclude_prefixes = []
        if config["reset_logic_critic"]:
            exclude_prefixes.append("logic_critic.")
        if config["reset_blending_weights"]:
            exclude_prefixes.extend(["actor.blender.im", "blender.im."])
        if config["reset_logic_actor"]:
            exclude_prefixes.extend(["actor.logic_actor.", "logic_actor."])
        load_model_state(checkpoint_path, model, exclude_prefixes=exclude_prefixes)
    except:
        assert False, (f"Failed to load model from {checkpoint_path}. "
                       f"Make sure that the logic critic of the specified 'agent_path' or 'logic_critic_path' has "
                       f"the same configuration for 'logic_critic_use_only_player_pos'.")

    return model


def load_model_train(model_dir,
                     n_envs,
               device=torch.device('cuda:0'),
               steps = None):
    # Determine all relevant paths
    model_dir = Path(model_dir)
    config_path = model_dir / "config.yaml"
    checkpoint_dir = model_dir / "checkpoints"
    if steps == None:
        most_recent_step = get_most_recent_checkpoint_step(checkpoint_dir)
    else:
        most_recent_step = steps
    checkpoint_path = checkpoint_dir / f"step_{most_recent_step}.pth"
    
    print("Loading model from", checkpoint_path)

    # Load model's configuration
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    algorithm = config["algorithm"]
    environment = config["env_name"]

    # Setup the environment
    env = VectorizedNudgeBaseEnv.from_name(environment, n_envs=n_envs, mode=algorithm)

    rules = config["rules"]

    print("Loading...")
    # Initialize the model
    if algorithm == 'ppo':
        from nudge.agents.neural_agent import ActorCritic

        model = ActorCritic(env).to(device)
    elif algorithm == 'logic':
        from nudge.agents.logic_agent import NsfrActorCritic

        model = NsfrActorCritic(env, device=device, rules=rules).to(device)
    else:
        from blendrl.agents.blender_agent import BlenderActorCritic

        model = BlenderActorCritic(env, rules=rules, actor_mode=config["actor_mode"],
                                   blender_mode=config["blender_mode"], blend_function=config["blend_function"],
                                   device=device).to(device)

    # Load the model weights
    with open(checkpoint_path, "rb") as f:
        model.load_state_dict(state_dict=torch.load(f, map_location=torch.device('cpu')))
    # model.logic_actor.im.W = torch.nn.Parameter(model.logic_actor.im.init_identity_weights(device))
    # print(model.logic_actor.im.W)

    return model, most_recent_step


def yellow(text):
    return "\033[93m" + text + "\033[0m"


def exp_decay(episode: int):
    """Reaches 2% after about 850 episodes."""
    return max(math.exp(-episode / 500), 0.02)


def get_most_recent_checkpoint_step(checkpoint_dir):
    checkpoints = os.listdir(checkpoint_dir)
    highest_step = 0
    pattern = re.compile("[0-9]+")
    for i, c in enumerate(checkpoints):
        match = pattern.search(c)
        if match is not None:
            step = int(match.group())
            if step > highest_step:
                highest_step = step
    return highest_step


def print_program(agent, mode="softor"):
    """Print a summary of logic programs using continuous weights."""
    # to adapt to be called from train and play scripts
    try:
        actor = agent.policy.actor
    except AttributeError:
        try:
            actor = agent.actor
        except AttributeError:
            actor = agent    
    if isinstance(actor, NSFReasoner):
        print_program_nsfr(actor, mode) 
    else:
        # the neumann reasoner
        print_program_neumann(actor, mode)
        
def print_program_nsfr(actor, mode):
    nsfr = actor
    if mode == "argmax":
        C = nsfr.clauses
        Ws_softmaxed = torch.softmax(nsfr.im.W, 1)
        for i, W_ in enumerate(Ws_softmaxed):
            max_i = np.argmax(W_.detach().cpu().numpy())
            print('C_' + str(i) + ': ',
                  C[max_i], 'W_' + str(i) + ':', round(W_[max_i].detach().cpu().item(), 3))
    elif mode == "softor":
        W_softmaxed = torch.softmax(nsfr.im.W, 1)
        w = softor(W_softmaxed, dim=0)
        for i, c in enumerate(nsfr.clauses):
            print('C_' + str(i) + ': ', np.round(w[i].detach().cpu().item(), 2), nsfr.clauses[i])
            

def print_program_neumann(actor, mode):
    neumann = actor
    if mode == "argmax":
        C = neumann.clauses
        Ws_softmaxed = torch.softmax(neumann.clause_weights, 1)
        for i, W_ in enumerate(Ws_softmaxed):
            max_i = np.argmax(W_.detach().cpu().numpy())
            print('C_' + str(i) + ': ',
                  C[max_i], 'W_' + str(i) + ':', round(W_[max_i].detach().cpu().item(), 3))
    elif mode == "softor":
        W_softmaxed = torch.softmax(neumann.clause_weights, 1)
        w = softor(W_softmaxed, dim=0)
        for i, c in enumerate(neumann.clauses):
            print('C_' + str(i) + ': ', np.round(w[i].detach().cpu().item(), 2), neumann.clauses[i])
            
            
            
def load_neuralppo_model(model_dir,
               env_kwargs_override: dict = None,
               steps = None,
               device=torch.device('cuda:0'),
               explain=False):
    # Determine all relevant paths
    model_dir = Path(model_dir)
    config_path = model_dir / "config.yaml"
    checkpoint_dir = model_dir / "checkpoints"
    if steps == None:
        most_recent_step = get_most_recent_checkpoint_step(checkpoint_dir)
    else:
        most_recent_step = steps
    checkpoint_path = checkpoint_dir / f"step_{most_recent_step}.pth"
    
    print("Loading model from", checkpoint_path)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    algorithm = config["algorithm"]
    environment = config["env_name"]
    # env_kwargs = config["env_kwargs"]
    # env_kwargs.update(env_kwargs_override)
    # env_kwargs = dict(render_oc_overlay=True)
    env_kwargs = {}

    # Setup the environment
    env = NudgeBaseEnv.from_name(environment, mode=algorithm, **env_kwargs)
    # model = ActorCritic(env).to(device)
    from utils import CNNActor
    model = CNNActor(n_actions=18) #, device=device, verbose=1)
    # Load the model weights
    with open(checkpoint_path, "rb") as f:
        model.load_state_dict(state_dict=torch.load(f, map_location=torch.device('cpu')))
    # model.logic_actor.im.W = torch.nn.Parameter(model.logic_actor.im.init_identity_weights(device))
    # print(model.logic_actor.im.W)

    return model


def get_objs(env: NudgeBaseEnv, category: str) -> list:
    obj_offset = env.obj_offsets.get(category)
    if obj_offset is None:
        return []

    result = []
    for obj in env.env.objects[obj_offset:]:
        if obj.category == category:
            result.append(obj)

    return result
