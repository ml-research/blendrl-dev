import logging
from pathlib import Path
from typing import Literal, List, Optional

import numpy as np
import torch
import tyro
from dataclasses import dataclass, field

from nsfr.utils.common import get_language
from plot.plot_utils import get_cmap, _discretize_frame, create_heatmap_fig, save_fig
from utils import get_default_device, FRAME_SIZE, to_np, load_model_state
from valuation.experiment import ValuationExperiment
from valuation.models.fixed_static import FixedStaticConfig, ValuationModelFixedStatic
from valuation.utils import sample_inputs

logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)

PREDICATE_CMAP = get_cmap('Grays')
LOGIC_CRITIC_CMAP = get_cmap('gist_ncar')
DIVERGENT_CMAP = get_cmap('RdYlGn')
DIFFERENCE_CMAP = get_cmap('RdBu_r')

@dataclass
class Args:
    env_name: str
    rules: str
    program: str
    resolution: int = 500
    ignore_predicates: List[str] = field(default_factory=list)
    show_grid: bool = False
    normalize_frame: bool = False
    exp_name: Optional[str] = None
    checkpoint: Optional[int] = None
    equal_resolution_along_dimensions: bool = False

def main():
    args = tyro.cli(Args)
    config = FixedStaticConfig(static_predicates=args.ignore_predicates, program=args.program)
    lang = get_language(args.env_name, args.rules)
    device = get_default_device()
    plot_dir = Path("plots/oracle") / args.env_name / args.program
    plot_dir.mkdir(parents=True, exist_ok=True)

    frame_size = FRAME_SIZE[args.env_name]
    if args.equal_resolution_along_dimensions:
        input = sample_inputs(args.resolution ** 2, "grid", device)
        grid_shape = (args.resolution, args.resolution)
    else:
        input, grid_shape = _discretize_frame([[-frame_size[0], frame_size[0]], [-frame_size[1], frame_size[1]]], args.resolution, device, dtype=torch.float)
        input[:, 1] /= frame_size[0]
        input[:, 2] /= frame_size[1]
    aspect = float(frame_size[1]) / float(frame_size[0])
    extent = [-1, 1, 1, -1] if args.normalize_frame else [-frame_size[0], frame_size[0], frame_size[1], -frame_size[1]]

    if args.exp_name is None:
        valuation_model = ValuationModelFixedStatic(args.env_name, lang, config, device)
    else:
        exp = ValuationExperiment.from_name(args.exp_name)
        agent = exp.get_model(get_default_device(), load_from_latest_checkpoint=args.checkpoint is None)
        if args.checkpoint is not None:
            ckpt = None
            for checkpoint in exp.checkpoints:
                if checkpoint.step == args.checkpoint:
                    ckpt = checkpoint
            if ckpt is None:
                raise ValueError("No checkpoint found")
            load_model_state(ckpt.path, agent)
        valuation_model = agent.valuation_model


    for pred_name in valuation_model.learnable_predicates:
        print(pred_name, grid_shape)
        output = to_np(valuation_model.forward_predicate(pred_name, input).reshape(*grid_shape))
        if not args.equal_resolution_along_dimensions:
            output = output.T
        fig, ax, im = create_heatmap_fig(output, PREDICATE_CMAP, extent=extent, aspect=aspect)
        ax.set_xticks(np.linspace(extent[0], extent[1], grid_shape[0] + 1), minor=True)
        ax.set_yticks(np.linspace(extent[2], extent[3], grid_shape[1] + 1), minor=True)
        ax.grid(which="minor", color="k", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        #ax.set_title(pred_name, fontfamily="monospace")
        ax.set_xlabel(r"$\Delta x$", fontsize=16)
        ax.set_ylabel(r"$\Delta y$", labelpad=-10 if not args.normalize_frame else 0, fontsize=16)

        ax.set_xticks([extent[0], 0, extent[1]])
        ax.axhline(y=0, color="#ABA9A0", linestyle="--", linewidth=1.5)
        ax.set_yticks([extent[2], 0, extent[3]])
        ax.axvline(x=0, color="#ABA9A0", linestyle="--", linewidth=1.5)
        ax.tick_params(left=False, bottom=False, top=False, right=False, labelsize=16)

        # Add grid lines
        #ax.grid(True, which='both', linestyle='--', linewidth=1.0, color='gray')

        if args.exp_name is None:
            plot_path = plot_dir / f"{pred_name}.png"
        else:
            plot_path = exp.plots_dir / f"normalized_{pred_name}.png"
        save_fig(fig, plot_path)

if __name__ == '__main__':
    main()