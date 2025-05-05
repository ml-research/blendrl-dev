from typing import Optional

import torch
import tyro

from blendrl.renderer import Renderer
from valuation.utils import ValuationExperiment


def main(
    env_name: str = "kangaroo",
    agent_path: str = "models/kangaroo_demo",
    exp_name: str = "",
    seed: int = 0,
    fps: int = 5,
    predicate_name: Optional[str] = None,
) -> None:

    # load predicate model
    experiment = ValuationExperiment.from_name(exp_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valuation_model = experiment.get_valuation_model(device)

    # create renderer
    renderer = Renderer(
        agent_path=agent_path,
        env_name=env_name,
        fps=fps,
        deterministic=False,
        env_kwargs=dict(render_oc_overlay=True),
        render_predicate_probs=True,
        seed=seed,
        predicate_name=predicate_name,
        valuation_model=valuation_model,
    )

    # run renderer
    renderer.run()


if __name__ == "__main__":
    tyro.cli(main)
