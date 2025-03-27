import tyro
from nudge.renderer import Renderer

from dataclasses import dataclass
import tyro


def main(
    env_name: str = "kangaroo",
    agent_path: str = "out/runs/kangaroo_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_20_steps_128_pretrained_False_joint_False_0",
    fps: int = 5,
    seed: int = 0
    ) -> None:
    renderer = Renderer(\
        agent_path=agent_path,
        env_name=env_name,
        fps=fps,
        deterministic=False,
        env_kwargs=dict(render_oc_overlay=True),
        #env_kwargs=dict(render_oc_overlay=False),
        render_predicate_probs=True,
        seed = seed)
    renderer.run()


if __name__ == "__main__":
    tyro.cli(main)
