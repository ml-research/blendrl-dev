import tyro
from blendrl.renderer import Renderer

from dataclasses import dataclass
import tyro


def main(
    env_name: str = "seaquest",
    agent_path: str = "models/kangaroo_demo",
    seed: int = 0,
    fps: int = 5,
) -> None:
    renderer = Renderer(
        agent_path=agent_path,
        env_name=env_name,
        fps=fps,
        deterministic=False,
        env_kwargs=dict(render_oc_overlay=True),
        render_predicate_probs=True,
        seed=seed,
    )
    renderer.run()


if __name__ == "__main__":
    tyro.cli(main)
