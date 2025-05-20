import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch as th
from matplotlib import animation
from tqdm import tqdm

from utils import load_model_state, get_default_device, FRAME_SIZE
from valuation.utils import ValuationExperiment


CMAP = 'jet'


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--exp-name", nargs='+', type=str, required=True)
    parser.add_argument("--predicate-name", nargs='+', type=str, required=True)
    parser.add_argument("--position", nargs='+', type=int, default=5)
    parser.add_argument("--checkpoint", nargs='+', type=int, default=None)

    parser.add_argument("--interval", type=int, default=40)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--cell-size", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()

    device = get_default_device()

    for exp_name in args.exp_name:
        # Load experiment
        experiment = ValuationExperiment.from_name(exp_name)

        # Load valuation model
        valuation_model = experiment.get_valuation_model(device, load_from_latest_checkpoint=False)

        checkpoints = experiment.checkpoints
        if len(checkpoints) == 0:
            print("No checkpoints found")
            continue
        checkpoint_steps = [checkpoints[-1].step] if args.checkpoint is None else args.checkpoint

        plots_dir = experiment.plots_dir / "predicate_valuation"
        os.makedirs(plots_dir, exist_ok=True)

        # Calculate grid
        frame_size = FRAME_SIZE[experiment.env_name]
        frame_grid = (int(frame_size[0] // args.cell_size), int(frame_size[1] // args.cell_size))
        num_grid_cells = frame_grid[0] * frame_grid[1]

        # Calculate player positions
        player_input = th.tensor([1, 0, 0, 0], device=device).repeat(num_grid_cells, 1)
        idx = 0
        for i in range(frame_grid[0]):
            x = (frame_size[0] / frame_grid[0]) * (i + 0.5)
            for j in range(frame_grid[1]):
                y = (frame_size[1] / frame_grid[1]) * (j + 0.5)
                player_input[idx, 1] = x
                player_input[idx, 2] = y
                idx += 1

        for position in args.position:
            x_mul = 0.25 * ((position + 2) % 3 + 1)
            y_mul = 0.25 * ((position + 2) // 3)
            obj_pos = (int(frame_size[0] * x_mul), int(frame_size[1] * y_mul))
            obj_grid_pos = (obj_pos[0] / frame_size[0] * frame_grid[0], obj_pos[1] / frame_size[1] * frame_grid[1])
            obj_input = th.tensor([1, obj_pos[0], obj_pos[1], 0], device=device).repeat(num_grid_cells, 1)

            for predicate_name in args.predicate_name:
                for checkpoint in checkpoints:
                    if checkpoint.step in checkpoint_steps:
                        load_model_state(checkpoint.path, valuation_model)

                        fig, ax = plt.subplots()
                        output = valuation_model(predicate_name, player_input, obj_input).view(*frame_grid).detach().cpu().numpy()
                        im = ax.imshow(output.T, cmap=CMAP, vmin=0, vmax=1)
                        ax.scatter(*obj_grid_pos, color='white')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        fig.suptitle(predicate_name, fontsize=10, y=0.95)
                        subtitle = ax.set_title(f"{exp_name} (step {checkpoint.step})", fontsize=8)
                        fig.tight_layout()

                        still_filename = f"{predicate_name}_{position}"
                        if checkpoint.step != checkpoints[-1].step:
                            still_filename += f"_{checkpoint.step}"
                        still_out_path = plots_dir / f"{still_filename}.png"
                        fig.savefig(still_out_path, dpi=150, bbox_inches='tight')
                        print(f"Checkpoint {checkpoint.step} saved at {still_out_path}")

                if args.checkpoint is None:
                    ani_out_path = plots_dir / f"{predicate_name}_{position}.gif"

                    _checkpoints = checkpoints[::args.skip]
                    num_checkpoints = len(_checkpoints)

                    with tqdm(desc=f"{exp_name} - Animating {predicate_name} {position}", total=num_checkpoints) as pbar:
                        def update(frame):
                            pbar.update()
                            checkpoint = _checkpoints[frame]
                            load_model_state(checkpoint.path, valuation_model)
                            output = valuation_model(predicate_name, player_input, obj_input).view(*frame_grid).detach().cpu().numpy()
                            im.set_array(output.T)
                            subtitle.set_text(f"{exp_name} (step {checkpoint.step})")
                            return [im, subtitle]

                        ani = animation.FuncAnimation(fig, update, frames=num_checkpoints, blit=True, interval=args.interval)

                        ani.save(ani_out_path, writer='pillow')

                    print(f"Animation saved at {ani_out_path}")
                    plt.close(fig)



if __name__ == '__main__':
    main()
