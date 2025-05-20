import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Annotated, Union, List, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import wandb
from dataclasses import dataclass, asdict, field
from rtpt import RTPT
from torch.utils.tensorboard import SummaryWriter

from blendrl.agents.blender_agent import BlenderActorCritic
from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from utils import DEFAULT_MODIFICATIONS
from valuation.utils import ValuationExperiment, load_model_config_classes
from nsfr.utils.logic import LogicState
from nudge.utils import load_model

IN_PATH = Path("in/")

torch.set_num_threads(5)


@dataclass
class Args:
    valuation_model: Union[
        tuple([
            Annotated[cls, tyro.conf.subcommand(cls.type)]
            for cls in load_model_config_classes()
        ])
    ]
    """the type and config of the valuation model"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
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
    """coefficient of the entropy"""
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
    anneal_blend_ent_coef: bool = False
    """whether to gradually reduce the coefficient of the blend entropy"""
    recover: bool = False
    """recover the training from the last checkpoint"""
    reasoner: str = "nsfr"
    """the reasoner used in the agent; nsfr or neumann"""

    learn_blending_weights: bool = False
    """whether to finetune the blending weights"""
    reset_blending_weights: bool = False
    """whether to randomize the blending weights at the start of the training"""
    reward_logic_subgoals: bool = False
    """whether to extend the reward function by logic subgoals"""
    extra_env_modifications: List[str] = field(default_factory=list)
    """extra modifications that shall be applied to the environments"""
    env_max_ep_steps: Optional[int] = None
    """maximum steps after which an episode is reset"""
    env_frameskip: int = 4
    """frames to skip"""
    reward_fn: str = "default"
    """the reward function"""
    #atom_ent_coef: float = 0.00
    #"""coefficient of the atom values"""


def main():

    # Parse arguments
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    config = asdict(args)

    # Setup process
    rtpt = RTPT(
        name_initials="HS",
        experiment_name="BlendRL",
        max_iterations=int(args.total_timesteps / args.save_steps),
    )

    # Initialize valuation experiment
    run_name = args.exp_name
    experiment = ValuationExperiment.from_name(run_name)
    experiment.init()
    experiment.update_config(config, print_config=True)
    logs_path = experiment.logs_path
    checkpoint_dir = experiment.checkpoints_dir

    # Setup metrics tracking
    writer_base_dir = ValuationExperiment.base_dir / ".." / "tensorboard"
    writer_dir = writer_base_dir / run_name

    if args.track:
        wandb.init(
            project=args.wandb_project_name + "_" + args.env_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
            id=run_name,
            resume="allow"
        )

    writer = SummaryWriter(str(writer_dir))

    # Set seeds (do not modify)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Setup game environments
    env_kwargs = {
        "modifications": DEFAULT_MODIFICATIONS[args.env_name] + args.extra_env_modifications,
        "frameskip": args.env_frameskip,
        "reward_fn_path": f"in/envs/{args.env_name}/reward/{args.reward_fn}.py",
    }
    if args.env_max_ep_steps is not None:
        env_kwargs["max_episode_steps"] = args.env_max_ep_steps

    envs = VectorizedNudgeBaseEnv.from_name(
        args.env_name,
        n_envs=args.num_envs,
        mode=args.algorithm,
        seed=args.seed,
        **env_kwargs,
    )

    # Load valuation model
    valuation_model = experiment.get_valuation_model(device, load_from_latest_checkpoint=args.recover)

    # Load logs from latest checkpoint
    global_step = 0
    save_step_bar = 0
    logs = defaultdict(list)
    if args.recover:
        # Get latest checkpoint
        latest_checkpoint = experiment.latest_checkpoint
        if latest_checkpoint is not None:
            latest_steps = latest_checkpoint.step
            global_step = latest_steps
            save_step_bar = global_step + args.save_steps
            print(f"Resuming training from step {global_step}")

        # Load training logs
        if os.path.exists(logs_path):
            with open(logs_path, "r") as logs_file:
                logs.update(json.load(logs_file))

    if not args.recover or logs is None:
        logs["total_timesteps"] = args.total_timesteps
        logs["num_envs"] = args.num_envs
        logs["num_steps"] = args.num_steps
        logs["num_iterations"] = args.num_iterations
        logs["batch_size"] = args.batch_size

    # Load agent model
    config_overrides = {
        "algorithm": args.algorithm,
        "rules": args.rules,
        "reasoner": args.reasoner,
        "actor_mode": args.actor_mode,
        "blender_mode": args.blender_mode,
        "blend_function": args.blend_function,
    }
    agent: BlenderActorCritic = load_model(
        args.agent_path,
        env_kwargs=env_kwargs,
        device=device,
        valuation_model=valuation_model,
        config_overrides=config_overrides
    )

    # Randomize blending weights
    if args.reset_blending_weights:
        im = agent.blender.im
        im.W = nn.Parameter(torch.Tensor(np.random.normal(size=(im.m, im.I.size(0)))).to(device))

    # Collect models that shall be trained
    trainable_models = [valuation_model]
    if args.learn_blending_weights:
        trainable_models.append(agent.blender.im)

    # Freeze agent
    agent.requires_grad_(False)
    for model in trainable_models:
        model.requires_grad_(True)

    # Rewards actually used to train model
    episodic_game_returns = torch.zeros((args.num_envs)).to(device)
    episodic_game_logic_blending_weights = [[] for _ in range(args.num_envs)]

    # Track models
    if args.track:
        wandb.watch(trainable_models, log="gradients")

    # Setup optimizer
    params = []
    for model in trainable_models:
        params.extend(list(model.parameters()))

    optimizer = optim.Adam([{"params": params, "lr": args.logic_learning_rate}],
        lr=args.logic_learning_rate,
        eps=1e-5,
    )

    # Start training
    agent._print()
    rtpt.start()

    # ALGO Logic: Storage setup
    observation_space = (4, 84, 84)
    # logic_observation_space = (84, 51, 4)
    logic_observation_space = (envs.n_objects, 4)
    # logic_observation_space = (84, 43, 4)
    action_space = ()
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space).to(device)
    logic_obs = torch.zeros(
        (args.num_steps, args.num_envs) + logic_observation_space
    ).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_space).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    next_logic_obs, next_obs = envs.reset()  # (seed=seed)
    # 1 env
    next_logic_obs = next_logic_obs.to(device)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    while global_step < args.total_timesteps:
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (global_step / args.total_timesteps)
            lrnow = frac * args.logic_learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            # update rtpt
            global_step += args.num_envs
            obs[step] = next_obs
            # print(logic_obs.shape)
            # print(next_logic_obs.shape)
            logic_obs[step] = next_logic_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # next_obs: (1, 4, 84, 84)
                # next_logic_obs: (1, 84, 51, 4)
                action, logprob, _, _, value, blending_weights = agent.get_action_and_value(
                    next_obs, next_logic_obs,
                    return_blending_weights=True
                )

            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            (next_logic_obs, next_obs), reward, terminations, truncations, infos = (
                envs.step(action.cpu().numpy())
            )
            next_logic_obs = next_logic_obs.float()
            terminations = np.array(terminations)
            truncations = np.array(truncations)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_logic_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_logic_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            episodic_game_returns += torch.tensor(reward).to(device).view(-1)
            for i in range(args.num_envs):
                episodic_game_logic_blending_weights[i].append(blending_weights[i, 1].item())

            for k, info in enumerate(infos):
                if "episode" in info:
                    print(
                        f"env={k}, global_step={global_step}, episodic_game_return={np.round(episodic_game_returns[k].detach().cpu().numpy(), 2)}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )
                    logs["episodic_returns"].append(info["episode"]["r"])
                    logs["episodic_lengths"].append(info["episode"]["l"])

                    # save the game reward
                    writer.add_scalar(
                        "charts/episodic_game_return",
                        episodic_game_returns[k],
                        global_step,
                    )

                    # save the min and mean logic blending weight
                    writer.add_scalar(
                        "charts/episodic_game_mean_logic_blending_weight",
                        np.mean(episodic_game_logic_blending_weights[k]),
                        global_step,
                    )

                    writer.add_scalar(
                        "charts/episodic_game_max_logic_blending_weight",
                        np.max(episodic_game_logic_blending_weights[k]),
                        global_step,
                    )

                    # reset game stats
                    episodic_game_returns[k] = 0
                    episodic_game_logic_blending_weights[k].clear()
                    print("Environment {} has been reset".format(k))

            # Save the model
            if global_step > save_step_bar:
                rtpt.step()
                checkpoint_path = checkpoint_dir / f"step_{save_step_bar}.pth"
                # save valuation model weights
                torch.save(valuation_model.state_dict(), checkpoint_path)
                # save agent weights
                # valuation_model.save(checkpoint_path, checkpoint_dir, [], [], [])
                print("\nSaved model at:", checkpoint_path)

                # save training data
                with open(logs_path, "w") as f:
                    json.dump(logs, f)

                # increase the updated bar
                save_step_bar += args.save_steps

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_logic_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + observation_space)
        b_logic_obs = logic_obs.reshape((-1,) + logic_observation_space)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # blend entropy coefficient
        blend_ent_coef = args.blend_ent_coef
        if args.anneal_blend_ent_coef:
            frac = 1.0 - (global_step / args.total_timesteps)
            blend_ent_coef = args.blend_ent_coef * frac

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # print(b_obs[mb_inds])
                _, newlogprob, entropy, blend_entropy, newvalue = (
                    agent.get_action_and_value(
                        b_obs[mb_inds], b_logic_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # high entropy => actions are more uniformly distributed
                entropy_loss = entropy.mean()

                # high blend entropy => neural and logic policies are more uniformly distributed
                blend_entropy_loss = blend_entropy.mean()

                # __import__('ipdb').set_trace()
                # the joint entropy loss incentivizes the action and blender distributions to be uniform
                joint_entropy_loss = (
                    -args.ent_coef * entropy_loss
                    -blend_ent_coef * blend_entropy_loss
                )
                loss = pg_loss + joint_entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(valuation_model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/loss", loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar(
            "losses/blend_entropy", blend_entropy_loss.item(), global_step
        )
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # the first SPS after the recovery is not accurate
        if int(global_step / (time.time() - start_time)) < 10000:
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )
        clause_weights = {f"{i+1}:{clause.head.pred.name}": agent.blender.im.get_clause_weights()[i].item() for i, clause in enumerate(agent.blender.clauses)}
        for clause_name, clause_weight in clause_weights.items():
            writer.add_scalar(
            f"charts/blending_clause_weights/{clause_name}",
                clause_weight,
                global_step
            )

        # save training data
        logs["value_losses"].append(v_loss.item())
        logs["policy_losses"].append(pg_loss.item())
        logs["entropies"].append(entropy_loss.item())
        logs["blend_entropies"].append(blend_entropy_loss.item())

        # print current agent information
        agent._print()

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
