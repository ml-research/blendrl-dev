import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Literal, get_args, Optional

import numpy as np
import torch
import torch as th
import tyro
from dataclasses import dataclass, field
from matplotlib import pyplot as plt, patches
from rtpt import RTPT
from tqdm import tqdm

from nsfr.valuation import ValuationModule
from nudge.env import NudgeBaseEnv
from nudge.utils import get_objs
from plot.plot_utils import get_cmap, create_heatmap_fig, fig_add_title, save_fig, get_predicate_heatmaps, \
    get_logic_critic_heatmap, discretize_frame, Animator, get_logic_actor_predicate_heatmap, make_transparent_cmap
from utils import get_default_device, FRAME_SIZE, load_model_state, to_np, optional
from valuation.experiment import ValuationExperiment

logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)

#PREDICATE_CMAP = get_cmap('jet')
PREDICATE_CMAP = get_cmap('inferno')
LOGIC_CRITIC_CMAP = get_cmap('gist_ncar')
DIVERGENT_CMAP = get_cmap('RdYlGn')
DIFFERENCE_CMAP = get_cmap('RdBu_r')
ACTIONS_CMAP = get_cmap("Dark2")
#DIFFERENCE_CMAP = get_cmap('berlin')

PlotType = Literal["all", "actions", "actions_overlay", "permuted_valuation", "logic_critic", "valuation", "valuation_overlay", "oracle", "oracle_overlay", "oracle_diff", "oracle_diff_overlay", "logic_actor_attention", "static_diff", "static_diff_overlay"]

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
    contour_objects: List[str] = field(default_factory=lambda: [])
    """Object categories that shall be plotted as contours in the heatmaps"""
    checkpoint: List[int] = field(default_factory=lambda: [])
    """List of individual checkpoint steps to plot. If empty, an animation will be created according to the 
       skip argument and additionally, the last checkpoint will be saved as still plot.
       If not empty, no animations will be created and only still plots for the specified steps will be saved."""
    save_all_checkpoints: bool = False
    """Whether to save all checkpoints (excluding skipped ones) as still plots. 
       If set, the checkpoint argument will be ignored."""
    logic_critic_extra_objects: List[str] = field(default_factory=lambda: [])
    """Extra objects placed in the center of the scene when plotting the values of the logic critic"""
    logic_actor_positions: List[str] = field(default_factory=lambda: ["1", "3", "5", "7", "9"])
    """Positions of the objects for which the attentions of the logic actor shall be computed"""
    save_animations: bool = False
    """Whether to save as animations"""

    plots: List[PlotType] = field(default_factory=lambda: ["all"])

    fps: int = 25
    """FPS of the animations"""
    skip: int = 1
    """How many checkpoint steps to move forward in the animation each frame. 1=all checkpoints; 2=every second checkpoint."""
    resolution: int = 32
    """Resolution of the heatmaps (number of cells in shortest dimension)"""
    format: str = "png"
    """Format of the plot"""

    save_grid: bool = False
    grid_labels: List[str] = field(default_factory=lambda: [])
    grid_filename: Optional[str] = None
    value_colorbar_index: Optional[int] = None
    diff_colorbar_index: Optional[int] = None
    grid_title: Optional[str] = None
    grid_add_max_actions: bool = False
    reweight: bool = False
    action_clauses: List[str] = field(default_factory=lambda: [])

    disable_title: bool = False
    transparent: bool = False
    zero_pad: bool = False


def main():
    args = tyro.cli(Args)

    # Setup process
    rtpt = RTPT(
        name_initials="HS",
        experiment_name="BlendRL",
        max_iterations=1,
    )
    rtpt.start()

    device = get_default_device()
    plot_all = "all" in args.plots
    plot = {
        literal: plot_all or (literal in args.plots)
        for literal in get_args(PlotType)
    }
    animation_enabled = args.save_animations and len(args.checkpoint) == 0

    # Plot grid
    num_predicates = len(args.predicate_name)
    num_rows = num_predicates + int(args.grid_add_max_actions)
    num_grid_columns = len(args.grid_labels)
    if args.save_grid:
        grid_fig, grid_axes = plt.subplots(
            num_rows, num_grid_columns,
            sharex=True, sharey=True,
            figsize=(1.5 + num_grid_columns * 1.8 + (num_grid_columns - 1) * 0.3, 0.8 + num_rows * 2.6 + (num_rows - 1) * 0.3),
        )
        grid_axes = np.reshape(grid_axes, (num_rows, num_grid_columns))

    for exp_index, exp_name in enumerate(args.exp_name):

        # Load valuation model
        experiment = ValuationExperiment.from_name(exp_name)
        agent = experiment.get_model(device, load_from_latest_checkpoint=False)
        valuation_model = agent.valuation_model
        oracle_model = experiment.get_oracle(device)
        static_model = experiment.get_default_valuation_model(device)
        plot_oracle = plot["oracle_diff"] or plot["oracle_diff_overlay"] or plot["oracle"] or plot["oracle_overlay"]
        if plot_oracle and oracle_model is None:
            print(f"Experiment {exp_name} has no oracle, plot will be skipped.")
            plot_oracle = False
            plot["oracle_diff"] = False
            plot["oracle_diff_overlay"] = False
        if plot["permuted_valuation"] and not experiment.config.logic_actor_use_permutation:
            print(f"Experiment {exp_name} did not use permutation, plot will be skipped.")
            plot["permuted_valuation"] = False

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

        def get_object_input(position: str):
            if position.isdigit():
                int_position = int(position)
                x_mul = 0.25 * ((int_position + 2) % 3 + 1)
                y_mul = 0.25 * ((int_position + 2) // 3)
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
            return obj_input, obj_pos

        object_inputs = {
            position: get_object_input(position)
            for position in set(args.position + args.logic_actor_positions)
        }

        # Calculate logic state
        logic_critic_input = th.tensor(logic_obs, device=device).repeat(num_grid_cells, 1, 1)

        if plot["logic_actor_attention"]:
            if not experiment.config.logic_actor_use_attention:
                print("Logic actor did not use attention, plot will be skipped.")
                plot["logic_actor_attention"] = False

        # Determine object overlays
        obj_overlays = []
        for category in args.contour_objects:
            objs = get_objs(env, category)
            obj_overlays.extend([obj.xywh for obj in objs])

        # Determine oracle heatmap
        oracle_heatmaps = defaultdict(list)
        if plot_oracle:
            oracle_heatmaps = {
                predicate_name: get_predicate_heatmaps(
                    oracle_model,
                    predicate_name,
                    player_input,
                    [object_inputs[pos][0] for pos in args.position],
                    grid_shape,
                    include_overlay=plot["oracle_diff_overlay"] or plot["oracle_overlay"]
                )
                for predicate_name in args.predicate_name
            }

        # Determine static heatmap
        static_heatmaps = defaultdict(list)
        if plot["static_diff"] or plot["static_diff_overlay"]:
            static_heatmaps = {
                predicate_name: get_predicate_heatmaps(
                    static_model,
                    predicate_name,
                    player_input,
                    [object_inputs[pos][0] for pos in args.position],
                    grid_shape,
                    include_overlay=plot["static_diff_overlay"]
                )
                for predicate_name in args.predicate_name
            }


        # Initialize figures and animations
        artists = []
        animators: List[Animator] = []

        # Iterate all checkpoints to create still heatmaps
        for checkpoint in tqdm(checkpoints):

            is_still_frame = (args.save_all_checkpoints and (checkpoint.step % (10_000 * args.skip) == 0)) or (checkpoint.step in checkpoint_steps)
            is_animation_frame = animation_enabled and (checkpoint.step % (10_000 * args.skip) == 0)

            # Compute heatmaps
            if is_still_frame or is_animation_frame:
                # update process title
                rtpt._variable_part = f"{exp_name}_{checkpoint.step}"
                rtpt._update_title()

                heatmaps = []  # list of dicts (name, position, heatmap, object positions)
                subtitle = f"{exp_name} (step {checkpoint.step})"

                load_model_state(checkpoint.path, agent)

                action_clauses = agent.logic_actor.prednames if len(args.action_clauses) == 0 else args.action_clauses

                clause_weights = to_np(agent.logic_actor.im.get_clause_weights())
                clause_weights /= clause_weights.max()
                clause_weights = [clause_weights[agent.logic_actor.prednames.index(ac)] for ac in action_clauses]

                for pidx, predicate_name in enumerate(args.predicate_name):
                    if plot["valuation"] or plot["valuation_overlay"] or plot["oracle_diff"] or plot["oracle_diff_overlay"] or plot["static_diff"] or plot["static_diff_overlay"]:
                        predicate_heatmaps = get_predicate_heatmaps(
                            valuation_model,
                            predicate_name,
                            player_input,
                            [object_inputs[pos][0] for pos in args.position],
                            grid_shape,
                            include_overlay=plot["valuation_overlay"] or plot["oracle_diff_overlay"] or plot["static_diff_overlay"]
                        )

                    diff_heatmaps = dict()
                    if plot["oracle_diff"] or plot["oracle_diff_overlay"]:
                        diff_heatmaps["oracle"] = [predicate_hm - oracle_hm for predicate_hm, oracle_hm in zip(predicate_heatmaps, oracle_heatmaps[predicate_name])]

                    if plot["static_diff"] or plot["static_diff_overlay"]:
                        diff_heatmaps["static"] = [predicate_hm - static_hm for predicate_hm, static_hm in zip(predicate_heatmaps, static_heatmaps[predicate_name])]

                    if plot["valuation"]:
                        heatmaps.extend([{
                            "name": predicate_name,
                            "position": pos,
                            "heatmap": hm,
                            "obj_positions": [object_inputs[pos][1]],
                            "vmin": 0,
                            "vmax": 1,
                            "cmap": PREDICATE_CMAP
                        } for pos, hm in zip(args.position, predicate_heatmaps)])

                    if plot["valuation_overlay"]:
                        hm = predicate_heatmaps[-1]
                        if args.reweight:
                            clause_weight = clause_weights[pidx]
                            hm *= clause_weight
                        heatmaps.append({
                            "name": predicate_name,
                            "position": "all",
                            "heatmap": hm,
                            "obj_positions": [object_inputs[pos][1] for pos in args.position],
                            "vmin": 0,
                            "vmax": 1,
                            "cmap": PREDICATE_CMAP
                        })

                    if plot["oracle"]:
                        heatmaps.extend([{
                            "name": predicate_name,
                            "position": pos,
                            "heatmap": hm,
                            "obj_positions": [object_inputs[pos][1]],
                            "vmin": 0,
                            "vmax": 1,
                            "cmap": PREDICATE_CMAP,
                            "filename": f"oracle_{predicate_name}"
                        } for pos, hm in zip(args.position, oracle_heatmaps[predicate_name])])

                    if plot["oracle_overlay"]:
                        heatmaps.append({
                            "name": predicate_name,
                            "position": "all",
                            "heatmap": oracle_heatmaps[predicate_name][-1],
                            "obj_positions": [object_inputs[pos][1] for pos in args.position],
                            "vmin": 0,
                            "vmax": 1,
                            "cmap": PREDICATE_CMAP,
                            "filename": f"oracle_{predicate_name}"
                        })

                    for diff_model, diff_heatmaps in diff_heatmaps.items():
                        if plot[f"{diff_model}_diff"]:
                            heatmaps.extend([{
                                "name": f"{predicate_name} (diff to {diff_model})",
                                "position": pos,
                                "heatmap": hm,
                                "obj_positions": [object_inputs[pos][1]],
                                "vmin": -1,
                                "vmax": 1,
                                "cmap": DIFFERENCE_CMAP,
                                "filename": f"{diff_model}_diff_{predicate_name}",
                                "contour_color": "black"
                            } for pos, hm in zip(args.position, diff_heatmaps)])

                        if plot[f"{diff_model}_diff_overlay"]:
                            heatmaps.append({
                                "name": f"{predicate_name} (diff to {diff_model})",
                                "position": "all",
                                "heatmap": diff_heatmaps[-1],
                                "obj_positions": [object_inputs[pos][1] for pos in args.position],
                                "vmin": -1,
                                "vmax": 1,
                                "cmap": DIFFERENCE_CMAP,
                                "filename": f"{diff_model}_diff_{predicate_name}",
                                "contour_color": "black"
                            })

                if plot["permuted_valuation"]:
                    perm_heatmaps = get_logic_actor_predicate_heatmap(agent.logic_actor, player_input, logic_critic_input, args.predicate_name, grid_shape)
                    for pred_name, heatmap in zip(args.predicate_name, perm_heatmaps):
                        heatmaps.append({
                            "name": f"{pred_name} (permuted)",
                            "position": None,
                            "heatmap": heatmap,
                            "obj_positions": [],
                            "vmin": 0,
                            "vmax": 1,
                            "cmap": PREDICATE_CMAP,
                            "filename": f"perm_{pred_name}"
                        })

                if plot["actions"] or plot["actions_overlay"] or args.grid_add_max_actions:
                    pred_names = action_clauses
                    action_heatmaps = get_logic_actor_predicate_heatmap(agent.logic_actor, player_input, logic_critic_input, pred_names, grid_shape)

                    if plot["actions"]:
                        for pred_name, action_hm in zip(pred_names, action_heatmaps):
                            heatmaps.append({
                                "name": pred_name,
                                "position": None,
                                "heatmap": action_hm,
                                "obj_positions": [],
                                "vmin": 0,
                                "vmax": 1,
                                "cmap": PREDICATE_CMAP,
                                "filename": f"action_{pred_name}"
                            })

                    if plot["actions_overlay"] or args.grid_add_max_actions:
                        stacked_action_heatmaps = np.stack(action_heatmaps, axis=2)
                        max_actions = np.argmax(stacked_action_heatmaps, axis=2) % 8
                        heatmaps.append({
                            "name": "max_actions",
                            "position": None,
                            "heatmap": max_actions,
                            "obj_positions": [],
                            "vmin": 0,
                            "vmax": 8,
                            "cmap": ACTIONS_CMAP
                        })


                if plot["logic_critic"]:
                    heatmap = get_logic_critic_heatmap(agent.logic_critic, player_input, logic_critic_input, grid_shape)
                    heatmaps.append({
                        "name": "logic_critic",
                        "position": None,
                        "heatmap": heatmap,
                        "obj_positions": [],
                        "vmin": None,
                        "vmax": None,
                        "cmap": LOGIC_CRITIC_CMAP
                    })

                    for lc_extra_object in args.logic_critic_extra_objects:
                        lc_input = logic_critic_input.clone()
                        obj_index = env.obj_offsets.get(lc_extra_object)
                        if obj_index is None:
                            continue
                        obj_pos = (int(frame_size[0] * 0.5), int(frame_size[1] * 0.5))
                        lc_input[:, obj_index] = th.tensor([1, *obj_pos, 0], device=device)

                        heatmap = get_logic_critic_heatmap(agent.logic_critic, player_input, lc_input, grid_shape)
                        heatmaps.append({
                            "name": "logic_critic",
                            "position": "5",
                            "heatmap": heatmap,
                            "obj_positions": [obj_pos],
                            "vmin": None,
                            "vmax": None,
                            "cmap": LOGIC_CRITIC_CMAP,
                            "filename": f"logic_critic_{lc_extra_object}"
                        })


                    # v_heatmap = heatmap[:-1] - heatmap[1:]
                    # h_heatmap = heatmap[:, 1:] - heatmap[:, :-1]
                    # for heatmap_suffix, heatmap in (("v", v_heatmap), ("h", h_heatmap)):
                    #     vabsmax = np.abs(heatmap).max()
                    #     heatmaps.append({
                    #         "name": "logic_critic_" + heatmap_suffix,
                    #         "position": None,
                    #         "heatmap": heatmap,
                    #         "obj_positions": [],
                    #         "vmin": -vabsmax,
                    #         "vmax": vabsmax,
                    #         "cmap": DIVERGENT_CMAP
                    #     })

                if plot["logic_actor_attention"]:
                    vm: ValuationModule = agent.logic_actor.fc.vm
                    object_dtypes = vm.object_types
                    player_dtype = experiment.language.consts[0].dtype

                    reference_object_positions = [object_inputs[pos][1] for pos in args.logic_actor_positions]

                    for object_dtype in object_dtypes:
                        if player_dtype == object_dtype:
                            continue

                        x1 = player_input.unsqueeze(1).repeat(1, len(args.logic_actor_positions), 1)
                        x2 = torch.stack(
                            tuple(object_inputs[pos][0] for pos in args.logic_actor_positions),
                            dim=1
                        )

                        attention_weights, attention_scores = vm.get_attention_weights(x1, x2, player_dtype, object_dtype)
                        min_attention_score, max_attention_score = attention_scores.min().item(), attention_scores.max().item()
                        for i, pos in enumerate(args.logic_actor_positions):
                            attention_weights_hm = to_np(attention_weights[:, i].reshape(grid_shape)).T
                            heatmaps.append({
                                "name": f"attention_weights_{object_dtype.name}",
                                "position": pos,
                                "heatmap": attention_weights_hm,
                                "obj_positions": [object_inputs[pos][1]],
                                "reference_obj_positions": reference_object_positions,
                                "vmin": 0,
                                "vmax": 1,
                                "cmap": PREDICATE_CMAP
                            })

                            #sm_attention_scores = torch.softmax(attention_scores[:, i], dim=0)
                            #attention_scores_hm = to_np(sm_attention_scores.reshape(grid_shape)).T
                            attention_scores_hm = to_np(attention_scores[:, i].reshape(grid_shape)).T
                            heatmaps.append({
                                "name": f"attention_scores_{object_dtype.name}",
                                "position": pos,
                                "heatmap": attention_scores_hm,
                                "obj_positions": [object_inputs[pos][1]],
                                "vmin": min_attention_score,
                                "vmax": max_attention_score,
                                "cmap": PREDICATE_CMAP
                            })

                # Iterate all heatmaps
                for i, info in enumerate(heatmaps):

                    grid_row = (i * num_rows) // (len(heatmaps))  # predicate
                    grid_col = (i % (len(heatmaps) // num_rows)) + (len(heatmaps) // num_rows) * exp_index

                    filename = info.get("filename", info["name"]) + (f"_{info['position']}" if info['position'] is not None else "")
                    save_kwargs = dict(dpi = 200, pad_inches = 0 if args.zero_pad else None, transparent = args.transparent)

                    # Initialize figures and animations
                    if len(artists) < len(heatmaps):
                        # Draw heatmap
                        kwargs = dict()
                        if args.save_grid:
                            kwargs["ax"] = grid_axes[grid_row][grid_col]

                        hm = info["heatmap"]
                        cmap = info["cmap"]
                        if args.transparent:
                            cmap = make_transparent_cmap(cmap)
                        fig, ax, im = create_heatmap_fig(hm, cmap, vmin=info["vmin"], vmax=info["vmax"], extent=(0, frame_size[0], frame_size[1], 0), **kwargs)
                        if args.transparent:
                            ax.axis('off')

                        # Draw object positions
                        contour_color = info.get("contour_color", "white")
                        for obj_grid_pos in info.get("reference_obj_positions", []):
                            ax.scatter(*obj_grid_pos, edgecolors=contour_color, facecolors='none')

                        for obj_grid_pos in info.get("obj_positions", []):
                            circle = patches.Circle(obj_grid_pos, radius=2, facecolor=contour_color, alpha=1.0)
                            ax.add_patch(circle)

                        # Draw objects overlays
                        for (x, y, w, h) in obj_overlays:
                            rect = patches.Rectangle(
                                (x, y),
                                w, h,
                                linewidth=1.0,
                                edgecolor=contour_color,
                                facecolor='none',
                                alpha=0.5
                            )
                            ax.add_patch(rect)

                        title_text, subtitle_text = None, None
                        if not args.disable_title:
                            title_text, subtitle_text = fig_add_title(
                                fig,
                                info["name"],
                                None, #subtitle
                                fontfamily="monospace",
                                fontsize=15,
                                pad=12
                            )
                        fig.tight_layout()

                        artists.append({
                            "fig": fig,
                            "ax": ax,
                            "im": im,
                            "subtitle": subtitle_text,
                        })

                        if animation_enabled:
                            ext = ".mp4" if not args.transparent else ".mov"
                            animation_path = plots_dir / (filename + ext)
                            animator_kwargs = dict()
                            if args.transparent:
                                animator_kwargs = dict(codec="prores_ks", ffmpeg_params=["-pix_fmt", "yuva444p10le"])
                            animator = Animator(fig, animation_path, args.fps, **animator_kwargs)
                            animators.append(animator)

                    artist = artists[i]
                    hm = info["heatmap"]
                    im = artist["im"]
                    im.set_array(hm)
                    im.set_clim(
                        vmin=hm.min() if info["vmin"] is None else info["vmin"],
                        vmax=hm.max() if info["vmax"] is None else info["vmax"]
                    )
                    fig = artist["fig"]

                    if is_still_frame:
                        still_filename = filename
                        if checkpoint.step != checkpoints[-1].step:
                            still_filename += f"_{checkpoint.step}"
                        still_path = plots_dir / f"{still_filename}.{args.format}"

                        save_fig(fig, still_path, close_fig=False, **save_kwargs)
                        print(f"Checkpoint {checkpoint.step} saved at {still_path}")

                    if is_animation_frame:
                        animator = animators[i]
                        animator.append(**save_kwargs)

        if not args.save_grid:
            for i, artist in enumerate(artists):
                plt.close(artist["fig"])

                if animation_enabled:
                    animator = animators[i]
                    animator.close()
                    print(f"Animation saved at {animator.path}")

    # Plot grid
    if args.save_grid:
        grid_filename = "_".join(args.exp_name) + "_" + "_".join(args.plots) if args.grid_filename is None else args.grid_filename
        grid_path = Path(f"plots/predicate_valuation/{grid_filename}.{args.format}")
        grid_fig.suptitle(optional(args.grid_title, ""), fontsize=16)
        for grid_row in range(num_rows):
            for grid_col in range(num_grid_columns):
                if grid_col == 0:
                    if grid_row < num_predicates:
                        grid_axes[grid_row, 0].set_ylabel(args.predicate_name[grid_row], fontsize=13, labelpad=8, fontfamily="monospace")
                    else:
                        grid_axes[grid_row, 0].set_ylabel("Most Likely Actions", fontsize=13, labelpad=8)
                if grid_row == 0:
                    grid_axes[0, grid_col].set_title(args.grid_labels[grid_col], fontsize=14, pad=8)
                if grid_row > 0:
                    grid_axes[grid_row, grid_col].set_title("")

        cax_y_offset = 0.57
        cax_label_offset = -8
        if args.value_colorbar_index is not None:
            cax1 = grid_axes[0, -1].inset_axes([1.07, cax_y_offset, 0.08, 0.43])
            cbar_kwargs = dict() if args.diff_colorbar_index is None else dict(format="{x:+}")
            cbar = grid_fig.colorbar(artists[args.value_colorbar_index]["im"], location="right", cax=cax1, ticks=[0, 1], **cbar_kwargs)
            cbar.set_label(r"$v$", labelpad=cax_label_offset)
            cax_y_offset -= 0.57
        if args.diff_colorbar_index is not None:
            cax2 = grid_axes[0, -1].inset_axes([1.07, cax_y_offset, 0.08, 0.43])
            cbar = grid_fig.colorbar(artists[args.diff_colorbar_index]["im"], location="right", cax=cax2, ticks=[-1, 1], format="{x:+}")
            cbar.set_label(r"$\Delta v$", labelpad=cax_label_offset)
            cax_y_offset -= 0.57
        if args.grid_add_max_actions:
            handles = [patches.Patch(color=ACTIONS_CMAP(i), label=label) for i, label in enumerate([ac.split("_")[0] for ac in action_clauses])]
            grid_axes[-1, -1].legend(handles=handles, bbox_to_anchor=(1.045, 1), loc='upper left', frameon=False, borderaxespad=0.0, handlelength=0.5, handleheight=1, handletextpad=0.4)

        save_fig(grid_fig, grid_path, close_fig=True, dpi=200, tight_layout=True)
        print(f"Saved grid path at {grid_path}")

if __name__ == '__main__':
    main()
