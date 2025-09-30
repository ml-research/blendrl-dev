from __future__ import annotations

from typing import Union, Annotated, Optional, Literal, List, Type

import tyro
from dataclasses import dataclass, field

from utils import load_classes_in_package
from valuation.models.base import BaseValuationModelConfig


def load_model_config_class(type: str) -> Optional[Type[BaseValuationModelConfig]]:
    classes = load_model_config_classes()
    for cls in classes:
        if cls.type == type:
            return cls
    return None


def load_model_config_classes() -> List[Type[BaseValuationModelConfig]]:
    return load_classes_in_package("valuation/models", BaseValuationModelConfig)


subcommands = tuple(
    Annotated[cls, tyro.conf.subcommand(cls.type)]
    for cls in load_model_config_classes()
)

ValuationModelType = Union[subcommands]
OracleSamplingMode = Literal["grid", "random"]


@dataclass
class ValuationConfig:
    valuation_model: ValuationModelType
    """the type and config of the valuation model"""
    oracle_model: Optional[ValuationModelType] = None
    """the type and config of the oracle model"""
    exp_name: str = "train_valuation"
    """the name of this experiment"""
    env_id: str = ""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "blendRL_val"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    # env_id: str = "Seaquest-v4"
    # """the id of the environment"""
    total_timesteps: int = 60_000_000
    """total timesteps of the experiments"""
    num_envs: int = 20
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy (higher = actions shall be distributed more equally; negative = actions shall be more disjunct)"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # added
    agent_path: str = "models/kangaroo_demo"
    """the path to the pretrained BlendRL agent"""
    env_name: str = "kangaroo"
    """the name of the environment"""
    algorithm: Literal["logic", "ppo", "blender"] = "blender"
    """the algorithm used in the agent"""
    blender_mode: Literal["logic", "neural"] = "logic"
    """the mode for the blend"""
    blend_function: Literal["softmax", "gumbel_softmax"] = "softmax"
    """the function to blend the neural and logic agents"""
    actor_mode: Literal["logic", "neural", "hybrid"] = "hybrid"
    """the mode for the agent"""
    rules: str = "default"
    """the ruleset used in the agent"""
    save_steps: int = 5_000_000
    """the number of steps to save models"""
    pretrained: bool = False
    """to use pretrained neural agent"""
    joint_training: bool = False
    """jointly train neural actor and logic actor and blender"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer (neural)"""
    logic_learning_rate: float = 2.5e-4
    """the learning rate of the optimizer (logic)"""
    blender_learning_rate: float = 2.5e-4
    """the learning rate of the optimizer (blender)"""
    blend_ent_coef: float = 0.01
    """coefficient of the blend entropy"""
    recover: bool = False
    """recover the training from the last checkpoint"""
    reasoner: str = "nsfr"
    """the reasoner used in the agent; nsfr or neumann"""

    # ==================================================================================================================
    # EXTRA ARGS
    # ==================================================================================================================

    # Auxiliary + Unused
    anneal_blend_ent_coef: bool = False
    """whether to gradually reduce the coefficient of the blend entropy"""
    reward_logic_subgoals: bool = False
    """whether to extend the reward function by logic subgoals"""
    regularize_ood_coef: float = 0.0
    regularize_ood_eps: float = 1.0
    logic_critic_use_only_player_pos: bool = False
    """whether the logic critic only uses the player position as input"""
    logic_actor_use_attention: bool = False
    """whether to use attention across available atoms in the logic actor"""
    logic_actor_use_permutation: bool = False
    """whether to permute the valuation functions across atoms"""
    object_dropout: float = 0.0
    """probability of dropping objects in the environment"""
    #atom_ent_coef: float = 0.00
    #"""coefficient of the atom values"""
    save_atom_gradient_data: bool = False
    """whether to save data of the atom gradients"""

    # Reset, finetune, overwrite and modify components
    ## Neural critic + actor
    reset_neural_component: bool = False
    """whether to randomize weights of the neural critic and actor"""
    learn_neural_component: bool = False
    """whether to finetune the neural critic and actor"""

    ## Blender
    reset_blending_weights: bool = False
    """whether to randomize the blending weights"""
    learn_blending_weights: bool = False
    """whether to finetune the blending weights"""
    neural_penalty_coef: float = 0.0
    """coefficient for penalizing activation of neural agent"""

    ## Logic Critic
    reset_logic_critic: bool = False
    """whether to randomize the weights of the logic critic"""
    learn_logic_critic: bool = False
    """whether to finetune the logic critic"""
    logic_critic_path: Optional[str] = None
    """path to the experiment from which the logic critic shall be initialized from"""

    ## Logic Actor
    reset_logic_actor: bool = False
    """whether to reset the weights of the logic actor"""
    learn_logic_actor: bool = False
    """whether to finetune the logic actor"""
    logic_actor_path: Optional[str] = None
    """path to the experiment from which the logic actor shall be initialized from"""
    softor_gamma: float = 0.01
    """Gamma coefficient for the softor operator"""

    ## Valuation Model
    valuation_model_path: Optional[str] = None
    """path to the experiment from which the valuation model shall be initialized from"""
    freeze_valuation_model: bool = False
    """whether to freeze the valuation model (can be used with `valuation_model_path` to use a pre-trained model)"""

    # Environment
    extra_env_modifications: List[str] = field(default_factory=list)
    """extra modifications that shall be applied to the environments"""
    env_max_ep_steps: Optional[int] = None
    """maximum steps after which an episode is reset"""
    env_frameskip: int = 4
    """frames to skip"""
    randomize_start_position: bool = False
    """whether to randomize the start position of the player after the episode has finished"""
    reward_fn: str = "default"
    """the reward function"""

    # Oracle
    anneal_concept_coef: bool = False
    """whether to anneal the concept coefficient"""
    concept_coef: float = 0.1
    """coefficient of the concept loss"""
    oracle_sampling_mode: OracleSamplingMode = "grid"
    """how to sample points for the oracle"""
    oracle_num_samples: int = 961
    """how many points to sample for the oracle"""

    # Logging
    log_heatmaps_steps: Optional[int] = None
    """steps after which heatmaps for e.g. critics and valuation models shall be logged (if None, logging will be skipped)"""
    save_train_data: bool = False
    """whether to save data used to train the model"""
