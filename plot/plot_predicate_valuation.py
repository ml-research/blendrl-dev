import logging
import os
from io import BytesIO
from typing import List

import imageio.v2 as iio
import numpy as np
import torch as th
import tyro
from PIL import Image
from dataclasses import dataclass, field
from matplotlib import pyplot as plt, patches
from tqdm import tqdm

from nudge.env import NudgeBaseEnv
from nudge.utils import get_objs
from plot.plot_utils import get_cmap, create_heatmap_fig, fig_add_title, save_fig, get_predicate_heatmaps, \
    get_logic_critic_heatmap, discretize_frame
from utils import get_default_device, FRAME_SIZE, load_model_state
from valuation.experiment import ValuationExperiment

logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)

CMAP = get_cmap('jet')

@dataclass
class Args:
    exp_name: List[str]
    """List of experiment names"""
    predicate_name: List[str]
    """List of predicate names"""
    position: List[str] = field(default_factory=lambda: ["5"])
    """List of positions. If number between 1-9, the positions align like a numpad, e.g., 
       1=top left corner, 5=center, 9 bottom right corner. If [Object Category]_[i], 
       the position of the i-th object of the specified category in the environment will be used, e.g.,
       Ladder_1=position of the first ladder."""
    overlay_positions: bool = False
    """Whether a heatmap shall be created that overlays the heatmaps of all positions"""
    contour_objects: List[str] = field(default_factory=lambda: [])
    """Object categories that shall be plotted as contours in the heatmaps"""
    checkpoint: List[int] = field(default_factory=lambda: [])
    """List of individual checkpoint steps to plot. If empty, an animation will be created according to the 
       skip argument and additionally, the last checkpoint will be saved as still plot.
       If not empty, no animations will be created and only still plots for the specified steps will be saved."""
    save_all_checkpoints: bool = False
    """Whether to save all checkpoints (excluding skipped ones) as still plots. 
       If set, the checkpoint argument will be ignored."""

    plot_logic_critic: bool = False
    """Whether to additionally plot the values of the logic critic"""

    fps: int = 25
    """FPS of the animations"""
    skip: int = 1
    """How many checkpoint steps to move forward in the animation each frame. 1=all checkpoints; 2=every second checkpoint."""
    resolution: int = 32
    """Resolution of the heatmaps (number of cells in shortest dimension)"""


def main():
    args = tyro.cli(Args)

    device = get_default_device()

    for exp_name in args.exp_name:

        # Load valuation model
        experiment = ValuationExperiment.from_name(exp_name)
        agent = experiment.get_model(device, load_from_latest_checkpoint=False)
        valuation_model = agent.valuation_model

        # Load environment
        env = NudgeBaseEnv.from_name(experiment.env_name, mode=experiment.config.algorithm, **experiment.env_config)
        logic_obs, neural_obs = env.reset()

        # Load checkpoints
        checkpoints = experiment.checkpoints
        if len(checkpoints) == 0:
            print("No checkpoints found")
            continue
        checkpoint_steps = [checkpoints[-1].step] if len(args.checkpoint) == 0 else args.checkpoint

        # Create plots directory
        plots_dir = experiment.plots_dir / "predicate_valuation"
        os.makedirs(plots_dir, exist_ok=True)

        # Calculate player inputs
        player_input, grid_shape = discretize_frame(experiment.env_name, args.resolution, device)
        num_grid_cells = grid_shape[0] * grid_shape[1]

        # Calculate object inputs
        frame_size = FRAME_SIZE[experiment.env_name]
        object_inputs = []
        for position in args.position:
            if position.isdigit():
                position = int(position)
                x_mul = 0.25 * ((position + 2) % 3 + 1)
                y_mul = 0.25 * ((position + 2) // 3)
                obj_pos = (frame_size[0] * x_mul, frame_size[1] * y_mul)
            else:
                obj_type, obj_index = position.split("_")
                _obj_index = int(obj_index) - 1

                objs = get_objs(env, obj_type)
                if 0 <= _obj_index < len(objs):
                    obj = objs[_obj_index]
                    obj_pos = obj.center
                else:
                    assert True, f"No object {obj_type} with index {obj_index} found"

            obj_input, _ = discretize_frame(experiment.env_name, args.resolution, device, obj_pos)
            object_inputs.append((position, obj_input, obj_pos))

        # Calculate logic state
        logic_critic_input = th.tensor(logic_obs, device=device).repeat(num_grid_cells, 1, 1)

        # Determine object overlays
        obj_overlays = []
        for category in args.contour_objects:
            objs = get_objs(env, category)
            obj_overlays.extend([obj.xywh for obj in objs])

        # Initialize figures and animations
        artists = []
        writers = []

        # Iterate all checkpoints to create still heatmaps
        for checkpoint in tqdm(checkpoints):

            is_still_frame = (args.save_all_checkpoints and (checkpoint.step % (10_000 * args.skip) == 0)) or (checkpoint.step in checkpoint_steps)
            is_animation_frame = len(args.checkpoint) == 0 and (checkpoint.step % (10_000 * args.skip) == 0)

            # Compute heatmaps
            if is_still_frame or is_animation_frame:
                heatmaps = []  # list of dicts (name, position, heatmap, object positions)
                subtitle = f"{exp_name} (step {checkpoint.step})"

                load_model_state(checkpoint.path, agent, strict=False)

                for predicate_name in args.predicate_name:
                    predicate_heatmaps = get_predicate_heatmaps(
                        valuation_model,
                        predicate_name,
                        player_input,
                        [oi[1] for oi in object_inputs],
                        grid_shape,
                        include_overlay=args.overlay_positions
                    )
                    heatmaps.extend([{
                        "name": predicate_name,
                        "position": oi[0],
                        "heatmap": hm,
                        "obj_positions": [oi[2]],
                        "vmin": 0,
                        "vmax": 1,
                    } for oi, hm in zip(object_inputs, predicate_heatmaps)])

                    if args.overlay_positions:
                        heatmaps.append({
                            "name": predicate_name,
                            "position": "all",
                            "heatmap": predicate_heatmaps[-1],
                            "obj_positions": [oi[2] for oi in object_inputs],
                            "vmin": 0,
                            "vmax": 1,
                        })

                if args.plot_logic_critic:
                    heatmap = get_logic_critic_heatmap(agent.logic_critic, player_input, logic_critic_input, grid_shape)
                    heatmaps.append({
                        "name": "logic_critic",
                        "position": None,
                        "heatmap": heatmap,
                        "obj_positions": [],
                        "vmin": None,
                        "vmax": None,
                    })

                # Iterate all heatmaps
                for i, info in enumerate(heatmaps):
                    # Initialize figures and animations
                    if len(artists) < len(heatmaps):
                        # Draw heatmap
                        fig, ax, im = create_heatmap_fig(info["heatmap"], CMAP, vmin=info["vmin"], vmax=info["vmax"], extent=(0, frame_size[0], frame_size[1], 0))

                        # Draw object positions
                        for obj_grid_pos in info["obj_positions"]:
                            ax.scatter(*obj_grid_pos, color='white')

                        # Draw objects overlays
                        for (x, y, w, h) in obj_overlays:
                            rect = patches.Rectangle(
                                (x, y),
                                w, h,
                                linewidth=0.5,
                                edgecolor='white',
                                facecolor='none',
                                alpha=0.5
                            )
                            ax.add_patch(rect)

                        title_text, subtitle_text = fig_add_title(
                            fig,
                            info["name"],
                            subtitle
                        )
                        fig.tight_layout()

                        artists.append({
                            "fig": fig,
                            "ax": ax,
                            "im": im,
                            "subtitle": subtitle_text,
                        })

                        animation_filename = info["name"]
                        if info.get("position") is not None:
                            animation_filename += "_" + info["position"]
                        animation_path = plots_dir / (animation_filename + ".mp4")
                        writer = iio.get_writer(
                            animation_path,
                            fps=args.fps,
                            codec='libx264',
                            quality=8,  # Quality (lower is better; 0-10 range)
                        )
                        writers.append({
                            "writer": writer,
                            "path": animation_path,
                        })

                    artist = artists[i]
                    hm = info["heatmap"]
                    im = artist["im"]
                    im.set_array(hm)
                    im.set_clim(
                        vmin=hm.min() if info["vmin"] is None else info["vmin"],
                        vmax=hm.max() if info["vmax"] is None else info["vmax"]
                    )
                    artist["subtitle"].set_text(subtitle)

                    writer = writers[i]["writer"]
                    fig = artist["fig"]

                    if is_still_frame:
                        still_filename = info["name"] + (f"_{info['position']}" if info['position'] is not None else "")
                        if checkpoint.step != checkpoints[-1].step:
                            still_filename += f"_{checkpoint.step}"
                        still_path = plots_dir / f"{still_filename}.png"

                        save_fig(fig, still_path, close_fig=False)
                        print(f"Checkpoint {checkpoint.step} saved at {still_path}")

                    if is_animation_frame:
                        buf = BytesIO()
                        fig.savefig(buf, dpi=150, format='png', bbox_inches='tight')  # Tight bbox cropping here
                        buf.seek(0)
                        img = np.array(Image.open(buf).convert("RGB"))
                        writer.append_data(img)

        for artist, writer in zip(artists, writers):
            plt.close(artist["fig"])

            writer["writer"].close()
            print(f"Animation saved at {writer['path']}")



if __name__ == '__main__':
    main()
