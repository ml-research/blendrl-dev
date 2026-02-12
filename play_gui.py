import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import tyro

from blendrl.renderer import Renderer
from valuation.experiment import ValuationExperiment


def main(
    env_name: str = "kangaroo",
    agent_path: str = "models/kangaroo_demo",
    exp_name: str = "",
    seed: int = 0,
    fps: int = 5,
    predicate_name: Optional[str] = None,
    use_oracle: bool = False,
    save_video: bool = False,
    save_num_episodes: int = 1,
    reset_frameskip: bool = True,
    reset_max_env_steps: bool = True,
) -> None:

    # load predicate model
    if exp_name != "":
        experiment = ValuationExperiment.from_name(exp_name)
    else:
        experiment = ValuationExperiment.from_path(Path(agent_path))

    if use_oracle:
        experiment.config.valuation_model = experiment.config.oracle_model

    if reset_frameskip:
        experiment.config.env_frameskip = 1

    if reset_max_env_steps:
        experiment.config.env_max_ep_steps = None

    video_path = None
    if save_video:
        os.makedirs(experiment.videos_dir, exist_ok=True)
        video_filename = experiment.env_name + "_" + experiment.name + "_" + ("oracle_" if use_oracle else "") + datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"
        video_path = experiment.videos_dir / video_filename

    # create renderer
    renderer = Renderer(
        experiment,
        fps=fps,
        deterministic=False,
        env_kwargs=dict(render_oc_overlay=True, **experiment.env_config),
        render_predicate_probs=experiment.config.actor_mode in ("hybrid", "logic"),
        seed=seed,
        predicate_name=predicate_name,
        save_video=video_path,
        save_num_episodes=save_num_episodes
    )

    # run renderer
    renderer.run()


if __name__ == "__main__":
    tyro.cli(main)
