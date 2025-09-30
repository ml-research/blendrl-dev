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
) -> None:

    # load predicate model
    if exp_name != "":
        experiment = ValuationExperiment.from_name(exp_name)
    else:
        experiment = ValuationExperiment.from_path(Path(agent_path))

    # create renderer
    renderer = Renderer(
        experiment,
        fps=fps,
        deterministic=False,
        env_kwargs=dict(render_oc_overlay=True, **experiment.env_config),
        render_predicate_probs=experiment.config.actor_mode in ("hybrid", "logic"),
        seed=seed,
        predicate_name=predicate_name,
    )

    # run renderer
    renderer.run()


if __name__ == "__main__":
    tyro.cli(main)
