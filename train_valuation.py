import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import wandb
from dataclasses import asdict
from rtpt import RTPT
from torch.nn import BCELoss
from torch.utils.tensorboard import SummaryWriter

from blendrl.agents.blender_agent import BlenderActorCritic
from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from plot.plot_utils import discretize_frame, get_predicate_heatmaps, get_cmap, create_prior_fig, fig_to_rgb
from utils import save_model_state, FRAME_SIZE, to_np, ArrayIO, normalize
from valuation.config import ValuationConfig
from valuation.experiment import ValuationExperiment
from valuation.utils import sample_inputs

IN_PATH = Path("in/")
DIVERGENT_CMAP = get_cmap("RdYlGn", "black")

torch.set_num_threads(5)


def main():

    # Parse arguments
    args = tyro.cli(ValuationConfig)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    config = asdict(args)

    # Setup process
    rtpt = RTPT(
        name_initials="HS",
        experiment_name=f"BlendRL_{args.exp_name}",
        max_iterations=max(1, int(args.total_timesteps / args.save_steps)),
    )

    # Initialize valuation experiment
    run_name = args.exp_name
    experiment = ValuationExperiment.from_name(run_name)
    experiment.init()
    experiment.update_config(args, print_config=True)
    has_oracle = experiment.config.oracle_model is not None
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
    env_kwargs = experiment.env_config

    envs = VectorizedNudgeBaseEnv.from_name(
        args.env_name,
        n_envs=args.num_envs,
        mode=args.algorithm,
        seed=args.seed,
        **env_kwargs,
    )

    # Load logs from latest checkpoint
    start_step = 0
    global_step = 0
    save_step_bar = 0
    log_step_bar = 0
    logs = defaultdict(list)
    training_is_resumed = False
    if args.recover:
        # Get latest checkpoint
        latest_checkpoint = experiment.latest_checkpoint
        if latest_checkpoint is not None:
            latest_steps = latest_checkpoint.step
            global_step = latest_steps
            start_step = global_step
            save_step_bar = global_step + args.save_steps
            log_step_bar = global_step + args.log_heatmaps_steps
            training_is_resumed = True
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

    # Create data writer
    if args.save_train_data:
        data_dir = experiment.logs_dir / "train"
        if not args.recover and data_dir.exists() and data_dir.is_dir():
            shutil.rmtree(data_dir)

        data_writer = ArrayIO(data_dir, chunk_size=100_000)

        max_ep_idx = 0
        if "ep_indices" in data_writer:
            max_ep_idx = data_writer["ep_indices"].max() + 1
        ep_indices = torch.arange(args.num_envs, device=device, dtype=torch.int64) + max_ep_idx

    # Load agent model
    agent: BlenderActorCritic = experiment.get_model(device, load_from_latest_checkpoint=args.recover)
    experiment.parameter_summary.print()
    experiment.parameter_summary.save_as_csv(experiment.parameter_summary_path)
    agent.env.reset()

    valuation_model = agent.valuation_model

    # Collect all relational predicates that are not static
    all_relational_predicates = set([pred.name for pred in experiment.language.neural_predicates if pred.arity == 2])
    static_predicates = set(experiment.valuation_model_config.static_predicates)
    learned_predicates = all_relational_predicates.difference(static_predicates)

    # Load oracle
    if has_oracle:
        oracle_valuation_model = experiment.get_oracle(device)
        oracle_valuation_model.requires_grad_(False)

    # Create static inputs for heatmaps
    player_input, grid_shape = discretize_frame(experiment.env_name, 32, device)
    frame_size = FRAME_SIZE[experiment.env_name]
    center_pos = (frame_size[0] / 2, frame_size[1] / 2)
    obj_inputs = [discretize_frame(experiment.env_name, 32, device, center_pos)[0]]

    # Rewards actually used to train model
    episodic_game_returns = torch.zeros((args.num_envs), device=device)
    episodic_game_logic_blending_weights = [[] for _ in range(args.num_envs)]

    # Track models
    # if args.track:
    #     wandb.watch(trainable_models, log="gradients")

    # Setup optimizer
    params = []
    param_names = []
    for name, param in agent.named_parameters():
        if param.requires_grad:
            params.append(param)
            param_names.append(name)

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
    frame_size = FRAME_SIZE[experiment.env_name]
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space, device=device)
    logic_obs = torch.zeros((args.num_steps, args.num_envs) + logic_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_space, device=device)
    positions = torch.zeros((args.num_steps, args.num_envs, 2), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # Compute prior from oracle
    if has_oracle:
        oracle_inputs = sample_inputs(args.oracle_num_samples, args.oracle_sampling_mode, device)
        oracle_values = dict()

        for pred_name in learned_predicates:
            oracle_v = oracle_valuation_model.forward_predicate(pred_name, oracle_inputs)
            oracle_values[pred_name] = oracle_v

            # Plot prior
            fig, ax = create_prior_fig(to_np(oracle_inputs[:, 1:3]), to_np(oracle_v))
            img = fig_to_rgb(fig)
            writer.add_image("oracle/priors/" + pred_name, img, global_step, dataformats="HWC")


    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    next_logic_obs, next_obs = envs.reset()  # (seed=seed)
    # 1 env
    next_logic_obs = next_logic_obs.to(device)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs, device=device)

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
            rewards[step] = torch.Tensor(reward).to(device).view(-1)
            positions[step] = next_logic_obs[:, 0, 1:3]
            next_obs, next_logic_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_logic_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            episodic_game_returns += torch.Tensor(reward).to(device).view(-1)
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

                    # reset player position
                    if args.randomize_start_position and experiment.env_name == "kangaroo":
                        noop_action = envs.pred2action['noop']
                        envs.envs[k].step(noop_action)
                        # this is a hack to re-apply the game modification after the episode has ended
                        envs.envs[k].step_modifs[0].__self__.already_reset = False

            # Save the model
            if global_step > save_step_bar:
                rtpt.step()

                # Save agent weights
                checkpoint_path = checkpoint_dir / f"step_{save_step_bar}.pth"
                save_model_state(agent, checkpoint_path, param_names)
                print("\nSaved model at:", checkpoint_path)

                # Save training data
                with open(logs_path, "w") as f:
                    json.dump(logs, f)

                # Increase the update bar
                save_step_bar += args.save_steps

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_logic_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
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

        # Flatten the batch
        b_obs = obs.reshape((-1,) + observation_space)
        b_logic_obs = logic_obs.reshape((-1,) + logic_observation_space)
        b_logprobs = logprobs.reshape(-1).detach()
        b_actions = actions.reshape((-1,) + action_space).detach()
        b_advantages = advantages.reshape(-1).detach()
        b_fin_advantages = torch.zeros_like(b_advantages)
        b_positions = positions.reshape((-1, 2))
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1).detach()

        # Save training data
        if args.save_train_data:
            data_writer.append("logprobs", to_np(b_logprobs))
            data_writer.append("actions", to_np(b_actions))
            data_writer.append("raw_advantages", to_np(b_advantages))
            data_writer.append("positions", to_np(b_positions))
            data_writer.append("values", to_np(b_values))
            data_writer.append("rewards", to_np(rewards.reshape(-1)))

            cur_ep_indices = torch.zeros_like(dones, dtype=torch.int64)

            for env_idx in range(args.num_envs):
                next_ep_idx = ep_indices.max().item() + 1
                ep_idx_offset = torch.cumsum(dones[:, env_idx].int(), dim=0) - 1
                zero_offset_indices = torch.argwhere(ep_idx_offset == -1)
                ep_idx_offset[zero_offset_indices] = ep_indices[env_idx] - next_ep_idx
                cur_ep_indices[:, env_idx] = next_ep_idx + ep_idx_offset
                ep_indices[env_idx] = cur_ep_indices[-1, env_idx]

            cur_ep_indices = to_np(cur_ep_indices.reshape(-1))
            data_writer.append("ep_indices", cur_ep_indices)

        # Anneal blend entropy coefficient
        blend_ent_coef = args.blend_ent_coef
        if args.anneal_blend_ent_coef:
            frac = 1.0 - (global_step / args.total_timesteps)
            blend_ent_coef = args.blend_ent_coef * frac

        # Anneal concept coefficient
        concept_coef = args.concept_coef
        if args.anneal_concept_coef:
            frac = 1.0 - (global_step / args.total_timesteps)
            concept_coef = args.concept_coef * frac

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Action probability
                _, newlogprob, entropy, blend_entropy, newvalue, new_blending_weights = (
                    agent.get_action_and_value(
                        b_obs[mb_inds], b_logic_obs[mb_inds], b_actions.long()[mb_inds], return_blending_weights=True
                    )
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Approximate KL-divergence (source: http://joschu.net/blog/kl-approx.html)
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                # Normalize advantages
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = normalize(mb_advantages, eps=1e-8)
                b_fin_advantages[mb_inds] = mb_advantages

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
                fin_v_loss = v_loss * args.vf_coef

                # Action entropy
                # (high = actions are more uniformly distributed)
                entropy_loss = entropy.mean()
                fin_entropy_loss = -args.ent_coef * entropy_loss

                # Blend entropy
                # (high = neural and logic policies are more uniformly distributed)
                blend_entropy_loss = blend_entropy.mean()
                fin_blend_entropy_loss = -blend_ent_coef * blend_entropy_loss

                # Joint entropy loss
                # (incentivizes the action and blender distributions to be uniform for better exploration)
                joint_entropy_loss = fin_entropy_loss + fin_blend_entropy_loss

                # Concept alignment loss
                fin_concept_loss = torch.tensor(0.0, device=device)
                if has_oracle:
                    for i, (pred_name, oracle_v) in enumerate(oracle_values.items()):
                        pred_v = valuation_model.forward_predicate(pred_name, oracle_inputs)
                        pred_concept_loss = BCELoss(reduction="mean")(pred_v, oracle_v)
                        if epoch == args.update_epochs - 1:
                            writer.add_scalar("losses/concept_loss/" + pred_name, pred_concept_loss.item(), global_step)
                        fin_concept_loss += pred_concept_loss

                fin_concept_loss *= concept_coef

                # Neural penalty loss
                neural_penalty_loss = new_blending_weights[:, 0].mean()
                fin_neural_penalty_loss = args.neural_penalty_coef * neural_penalty_loss

                # Total loss
                loss = pg_loss + joint_entropy_loss + fin_v_loss + fin_concept_loss + fin_neural_penalty_loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Log heatmaps
        if args.log_heatmaps_steps is not None and global_step > log_step_bar:
            # Save heatmap for each learnable predicate
            for pred_name in learned_predicates:
                heatmap = get_predicate_heatmaps(valuation_model, pred_name, player_input, obj_inputs, grid_shape,True)[-1]
                writer.add_image("predicates/" + pred_name, heatmap, global_step, dataformats="HW")

            # Save heatmaps for advantages per action
            # a_names = ["up", "right", "left"]
            # for a_name in a_names:
            #     a_idx = envs.pred2action[a_name]
            #     a_inds = torch.argwhere(b_actions == a_idx).flatten()
            #     a_positions = to_np(b_positions[a_inds])
            #     a_advantages = to_np(b_fin_advantages[a_inds])
            #     a_heatmap, _, _, _ = binned_statistic_2d(
            #         a_positions[:, 0], a_positions[:, 1], a_advantages,
            #         statistic='mean',
            #         bins=(32, 42),
            #         range=np.array([[0, frame_size[0]], [0, frame_size[1]]])
            #     )
            #     a_heatmap = np.nan_to_num(a_heatmap, nan=0.0)
            #     a_heatmap = np.ma.masked_where(a_heatmap == 0.0, a_heatmap)
            #     a_heatmap = (a_heatmap / 2.0) + 0.5 # map [-1, 1] to [0, 1]
            #     a_heatmap = DIVERGENT_CMAP(a_heatmap)[:, :, :3]
            #     writer.add_image("advantages/" + a_name, a_heatmap, global_step, dataformats="WHC")

            # Increase the step bar for logging heatmaps
            log_step_bar += args.log_heatmaps_steps

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/fin_value_loss", fin_v_loss.item(), global_step)
        writer.add_scalar("losses/fin_concept_loss", fin_concept_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/fin_entropy_loss", fin_entropy_loss.item(), global_step)
        writer.add_scalar("losses/blend_entropy", blend_entropy_loss.item(), global_step)
        writer.add_scalar("losses/fin_blend_entropy_loss", fin_blend_entropy_loss.item(), global_step)
        writer.add_scalar("losses/joint_entropy_loss", joint_entropy_loss.item(), global_step)
        writer.add_scalar("losses/neural_penalty_loss", neural_penalty_loss.item(), global_step)
        writer.add_scalar("losses/fin_neural_penalty_loss", fin_neural_penalty_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        sps = (global_step - start_step) / (time.time() - start_time)
        print(f"SPS: {sps:.2f}")
        writer.add_scalar("charts/SPS", sps, global_step)

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
    data_writer.close()


if __name__ == "__main__":
    main()
