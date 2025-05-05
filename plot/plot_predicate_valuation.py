import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
import torch as th
from matplotlib import animation
from tqdm import tqdm

from utils import load_model_state
from valuation.utils import ValuationExperiment


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--predicate-name", type=str, required=True)
    parser.add_argument("--position", type=int, default=5)
    parser.add_argument("--interval", type=int, default=40)
    parser.add_argument("--skip", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cpu')
    experiment = ValuationExperiment.from_name(args.exp_name)

    predicate_model = experiment.get_valuation_model(device, load_from_latest_checkpoint=False)
    plots_dir = experiment.plots_dir

    os.makedirs(plots_dir, exist_ok=True)

    frame_size = (160, 210)
    frame_grid = (16, 21)
    num_grid_cells = frame_grid[0] * frame_grid[1]
    grid_cell_size = (frame_size[0] / frame_grid[0], frame_size[1] / frame_grid[1])

    x_mul = 0.25 * ((args.position + 2) % 3 + 1)
    y_mul = 0.25 * ((args.position + 2) // 3)
    obj_pos = (int(frame_size[0] * x_mul), int(frame_size[1] * y_mul))
    obj_grid_pos = (obj_pos[0] / frame_size[0] * frame_grid[0], obj_pos[1] / frame_size[1] * frame_grid[1])
    obj_input = th.tensor([1, obj_pos[0], obj_pos[1], 0]).repeat(num_grid_cells, 1)

    player_input = th.tensor([1, 0, 0, 0]).repeat(num_grid_cells, 1)
    idx = 0
    for i in range(frame_grid[0]):
        x = (frame_size[0] / frame_grid[0]) * (i + 0.5)
        for j in range(frame_grid[1]):
            y = (frame_size[1] / frame_grid[1]) * (j + 0.5)
            player_input[idx, 1] = x
            player_input[idx, 2] = y
            idx += 1

    checkpoints = experiment.checkpoints[args.skip::args.skip+1]
    num_checkpoints = len(checkpoints)

    if len(checkpoints) == 0:
        print("No checkpoints found")
        return

    load_model_state(checkpoints[0].path, predicate_model)

    fig, ax = plt.subplots()
    output = predicate_model(args.predicate_name, player_input, obj_input).view(*frame_grid).detach().numpy()
    im = ax.imshow(output.T, cmap='Reds', vmin=0, vmax=1)
    ax.scatter(*obj_grid_pos, color='black')
    title = ax.set_title(f"Checkpoint step {checkpoints[0].step}")

    out_path = plots_dir / f"activation_{args.predicate_name}_{args.position}.gif"

    with tqdm(desc="Animating frame", total=num_checkpoints) as pbar:
        def update(frame):
            pbar.update()
            checkpoint = checkpoints[frame]
            load_model_state(checkpoint.path, predicate_model)
            output = predicate_model(args.predicate_name, player_input, obj_input).view(*frame_grid).detach().numpy()
            im.set_array(output.T)
            title.set_text(f"Checkpoint step {checkpoint.step}")
            return [im, title]

        ani = animation.FuncAnimation(fig, update, frames=num_checkpoints, blit=True, interval=args.interval)

        ani.save(out_path, writer='pillow')

    plt.close(fig)

    print(f"Animation saved at {out_path}")


if __name__ == '__main__':
    main()
