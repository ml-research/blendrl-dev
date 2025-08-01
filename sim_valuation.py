import shutil
import time
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import torch
import tyro
from dataclasses import dataclass
from rtpt import RTPT

from blendrl.agents.blender_agent import BlenderActorCritic
from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from utils import get_default_device, to_np, ArrayIO
from valuation.experiment import ValuationExperiment


@dataclass
class Args:
    exp_name: Optional[str] = None
    """Name of the valuation experiment. If 'None', the model in `agent_path` will be evaluated."""
    seed: int = 0
    """Seed"""
    num_episodes: Optional[int] = 100
    """Minimum number of episodes to simulate"""
    num_steps: Optional[int] = None
    """Minimum number of steps to simulate"""
    num_envs: int = 64
    """Number of environments"""
    overwrite: bool = False
    "Whether to overwrite existing results"
    actor_mode: Optional[Literal["hybrid", "neural", "logic"]] = None
    """The actor mode"""

    # Used if no valuation experiment is specified
    agent_path: str = "models/kangaroo_demo"
    """Path to pretrained BlendRL model"""
    env_name: str = "kangaroo"
    """Name of environment"""


def main():
    # Parse arguments
    args = tyro.cli(Args)
    assert not (args.num_steps is None and args.num_episodes is None), f"Either --num-steps or --num-episodes must be specified"

    num_envs = args.num_envs

    # Get device and valuation model
    device = get_default_device()
    if args.exp_name is not None:
        experiment = ValuationExperiment.from_name(args.exp_name)
    else:
        experiment = ValuationExperiment.from_path(Path(args.agent_path))

    # Overwrite actor mode
    if args.actor_mode is not None:
        experiment.config.actor_mode = args.actor_mode

    # Overwrite max env steps
    experiment.config.env_max_ep_steps = None

    agent: BlenderActorCritic = experiment.get_model(device)
    env_kwargs = experiment.env_config

    agent.eval()

    # Load environments
    envs = VectorizedNudgeBaseEnv.from_name(
        experiment.env_name, n_envs=num_envs, mode="blender", seed=args.seed, **env_kwargs
    )

    # Collect data
    blender_predicates = agent.blender.prednames
    num_blender_predicates = len(blender_predicates)

    action_predicates = agent.logic_actor.prednames
    num_action_predicates = len(action_predicates)

    # Create process
    rtpt = RTPT(
        name_initials="HS",
        experiment_name="BlendRL_sim",
        max_iterations=args.num_episodes,
    )
    rtpt.start()

    # Start simulation
    data_dir = experiment.logs_dir / f"test_{experiment.config.actor_mode}"
    if args.overwrite and data_dir.exists():
       shutil.rmtree(data_dir)
    writer = ArrayIO(data_dir)
    global_ep = 0
    global_step = 0
    game_returns = np.zeros(num_envs)
    action = None
    iteration = -1
    env_dur = 0.0
    inf_dur = 0.0
    _t0 = time.time()
    start_time = _t0
    current_ep_indices = np.arange(num_envs)
    with torch.no_grad():
        while True:
            # Print progress
            iteration += 1
            if iteration % 100 == 0:
                it_dur = time.time() - _t0
                sim_dur = time.time() - start_time
                print(f"[{sim_dur:6.1f}s] Ep: {global_ep}/{args.num_episodes} | Step: {global_step}/{args.num_steps} | It Dur: {it_dur/num_envs:.3f}s | Env Dur: {env_dur/num_envs:.3f}s | Inf Dur: {inf_dur/num_envs:.3f}s")
                _t0 = time.time()
                env_dur = 0.0
                inf_dur = 0.0

            # Get next observations
            t0 = time.time()
            if action is None:
                _logic_obs, _obs = envs.reset()
                noop_action = envs.pred2action['noop']
                action = torch.ones(num_envs, dtype=int) * noop_action

            (_logic_obs, _obs), reward, terminations, truncations, infos = (
                envs.step(action.cpu().numpy())
            )
            env_dur += time.time() - t0

            # Gather pre-step data
            writer.append("logic_obs", to_np(_logic_obs))

            # Get next action
            _logic_obs = torch.tensor(_logic_obs, device=device)
            _obs = torch.tensor(_obs, device=device)

            t0 = time.time()
            action, _, _, _, _, _blending_weights = agent.get_action_and_value(
                _obs, _logic_obs, return_blending_weights=True
            )
            _neural_values = agent.get_neural_value(_obs).reshape(-1)
            _logic_values = agent.get_logic_value(_logic_obs).reshape(-1)
            inf_dur += time.time() - t0

            # Gather post-step data
            writer.append("ep_indices", current_ep_indices)
            writer.append("actions", to_np(action))
            writer.append("blending_weights", to_np(_blending_weights))
            writer.append("neural_action_probs", to_np(agent.actor.neural_action_probs))
            writer.append("logic_action_probs",to_np(agent.actor.logic_action_probs))
            writer.append("neural_values",to_np(_neural_values))
            writer.append("logic_values",to_np(_logic_values))
            writer.append("rewards",np.array(reward))

            if agent.logic_actor.V_T != []:
                writer.append("action_predicate_probs", to_np(agent.logic_actor.get_predictions(agent.logic_actor.V_T, action_predicates)))

            if agent.blender.V_T != []:
                writer.append("blender_predicate_probs", to_np(agent.blender.get_predictions(agent.blender.V_T, blender_predicates)))

            global_step += num_envs

            # Gather episodic info
            game_returns += np.array(reward)
            if global_ep < args.num_episodes:
                for k, info in enumerate(infos):
                    if "episode" in info:
                        writer.append("ep_lengths", np.array(info["episode"]["l"]))
                        writer.append("ep_returns", np.array(info["episode"]["r"]))
                        writer.append("ep_game_returns", np.array(game_returns[k]))
                        game_returns[k] = 0

                        current_ep_indices[k] = max(current_ep_indices) + 1

                        global_ep += 1
                        rtpt.step(f"{global_ep}/{args.num_episodes}")

                        if global_ep >= args.num_episodes:
                            break

            # Break program if end has reached
            if (args.num_episodes is None or global_ep >= args.num_episodes) and (args.num_steps is None or global_step >= args.num_steps):
                break

    writer.close()


if __name__ == "__main__":
    main()
