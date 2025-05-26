import os
from argparse import ArgumentParser
from io import BytesIO
from typing import List

import imageio.v2 as iio
import numpy as np
import torch as th
from PIL import Image
from matplotlib import pyplot as plt, patches
from tqdm import tqdm

from nudge.env import NudgeBaseEnv
from plot.plot_utils import get_cmap, create_heatmap_fig, fig_add_title, save_fig
from utils import get_default_device, FRAME_SIZE, to_np, load_model_state
from valuation.experiment import ValuationExperiment
from valuation.models.base import BaseValuationModel
import logging

logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)

CMAP = get_cmap('jet')


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--exp-name", nargs='+', type=str, required=True)
    parser.add_argument("--predicate-name", nargs='+', type=str, required=True)
    parser.add_argument("--position", nargs='+', type=str, default=5)
    parser.add_argument("--obj-overlays", nargs='*', type=str)
    parser.add_argument("--overlay-positions", action='store_true')
    parser.add_argument("--checkpoint", nargs='+', type=int, default=None)

    parser.add_argument("--plot-logic-critic", action='store_true')

    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--cell-size", type=int, default=5)

    return parser.parse_args()

def get_predicate_heatmaps(valuation_model: BaseValuationModel, predicate_name: str, player_input: th.Tensor, obj_inputs: List[th.Tensor], frame_shape: (int, int), include_overlay: bool) -> List[np.ndarray]:
    heatmaps = []

    for obj_input in obj_inputs:
        # Compute heatmap
        output = valuation_model(predicate_name, player_input, obj_input).view(*frame_shape)
        heatmap = to_np(output).T
        heatmaps.append(heatmap)

    if include_overlay:
        # Compute overlayed heatmap
        overlayed_heatmap = np.maximum.reduce(heatmaps)
        heatmaps.append(overlayed_heatmap)

    return heatmaps


def get_logic_critic_heatmap(logic_critic: th.nn.Module, player_input: th.Tensor, obj_input: th.Tensor, frame_shape: (int, int)) -> np.ndarray:
    x = obj_input
    x[:, 0] = player_input
    output = logic_critic(x).view(*frame_shape)
    heatmap = to_np(output).T

    return heatmap


def get_objs(env: NudgeBaseEnv, category: str) -> list:
    obj_offset = env.obj_offsets.get(category)
    if obj_offset is None:
        return []

    result = []
    for obj in env.env.objects[obj_offset:]:
        if obj.category == category:
            result.append(obj)

    return result


def main():
    args = parse_args()

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
        checkpoint_steps = [checkpoints[-1].step] if args.checkpoint is None else args.checkpoint

        # Create plots directory
        plots_dir = experiment.plots_dir / "predicate_valuation"
        os.makedirs(plots_dir, exist_ok=True)

        # Calculate grid
        frame_size = FRAME_SIZE[experiment.env_name]
        frame_shape = (int(frame_size[0] // args.cell_size), int(frame_size[1] // args.cell_size))
        num_grid_cells = frame_shape[0] * frame_shape[1]

        # Calculate player inputs
        player_input = th.tensor([1, 0, 0, 0], device=device).repeat(num_grid_cells, 1)
        idx = 0
        for i in range(frame_shape[0]):
            x = (frame_size[0] / frame_shape[0]) * (i + 0.5)
            for j in range(frame_shape[1]):
                y = (frame_size[1] / frame_shape[1]) * (j + 0.5)
                player_input[idx, 1] = x
                player_input[idx, 2] = y
                idx += 1

        # Calculate object inputs
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

            obj_input = th.tensor([1, obj_pos[0], obj_pos[1], 0], device=device).repeat(num_grid_cells, 1)
            object_inputs.append((position, obj_input, obj_pos))

        # Calculate logic state
        logic_critic_input = th.tensor(logic_obs, device=device).repeat(num_grid_cells, 1, 1)

        # Determine object overlays
        obj_overlays = []
        for category in args.obj_overlays:
            objs = get_objs(env, category)
            obj_overlays.extend([obj.xywh for obj in objs])

        # Initialize figures and animations
        artists = []
        writers = []

        # Iterate all checkpoints to create still heatmaps
        for checkpoint in tqdm(checkpoints):

            is_still_frame = checkpoint.step in checkpoint_steps
            is_animation_frame = args.checkpoint is None and (checkpoint.step % (10_000 * args.skip) == 0)

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
                        frame_shape,
                        include_overlay=args.overlay_positions
                    )
                    heatmaps.extend([{
                        "name": predicate_name,
                        "position": oi[0],
                        "heatmap": hm,
                        "obj_positions": [oi[2]]
                    } for oi, hm in zip(object_inputs, predicate_heatmaps)])

                    if args.overlay_positions:
                        heatmaps.append({
                            "name": predicate_name,
                            "position": "all",
                            "heatmap": predicate_heatmaps[-1],
                            "obj_positions": [oi[2] for oi in object_inputs]
                        })

                if args.plot_logic_critic:
                    heatmap = get_logic_critic_heatmap(agent.logic_critic, player_input, logic_critic_input, frame_shape)
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
                        fig, ax, im = create_heatmap_fig(info["heatmap"], CMAP, vmin=info.get("vmin", 0), vmax=info.get("vmax", 1), extent=(0, frame_size[0], frame_size[1], 0))

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
                    im.set_clim(vmin=hm.min(), vmax=hm.max())
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
