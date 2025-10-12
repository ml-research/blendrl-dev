from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
import wandb
from dataclasses import dataclass
from matplotlib import ticker

from plot.plot_utils import save_fig
from utils import running_average

NUM_SAMPLES = 1000
NUM_RUNNING_AVG = 70


@dataclass
class Args:
    project_name: str
    """Name of the WandB project."""
    entity_name: str
    """Name of the WandB entity."""

def human_format(x, pos):
    return f"{x * 1e-6:.0f}M"

def main():
    # Parse args
    args = tyro.cli(Args)

    # Initialize W&B API
    api = wandb.Api()

    def get_data(run_name, metric, revert_annealing=False):
        # Get all runs in the project
        run = api.runs(
            f"{args.entity_name}/{args.project_name}",
            filters={
                "$or": [
                    {"name": run_name}
                ]
            },
        )[0]

        history = run.history(keys=[metric, "global_step"], samples=NUM_SAMPLES)

        # Skip runs without data
        if metric not in history or "global_step" not in history:
            return np.array([])

        x = history["global_step"]
        y = history[metric].values
        if revert_annealing:
            y /= (1 - (x / 10_000_000))
        return x, running_average(y, NUM_RUNNING_AVG, center = True)

    metrics = ["losses/fin_concept_loss", "charts/episodic_game_return"]
    exps_labels = [r"$\lambda_{\operatorname{CA}} = 1$", r"$\lambda_{\operatorname{CA}} = 0$"]
    metrics_labels = [r"$L^{\operatorname{CA}}$ (Log Scale)", "Goals"]

    exp1 = ["mlp_257_cl_cc0_3_s1", "mlp_257_cl_cc1_0_s2"]
    exp2 = ["mlp_209_cl_cc0_3_s1", "mlp_209_cl_cc1_0_s2"]

    fig, axes = plt.subplots(2, 2, figsize=(2.5 + 2 * 3.5, 2 * 2.5), sharey=True, sharex=True)
    axes = np.array([axes[1], axes[0]]).flatten()
    axes2 = np.array([ax.twinx() for ax in axes])

    for i in [0, 2]:
        axes[i].set_ylabel(metrics_labels[0], color="blue")
        axes[i].tick_params(axis="y", colors="blue")

    for i in [0, 1]:
        axes[i].set_xlabel("Step")


    x11_0, y11_0 = get_data(exp1[0], metrics[0], revert_annealing=True)
    y11_0 /= 0.3
    x11_1, y11_1 = get_data(exp1[0], metrics[1])

    x12_0, y12_0 = get_data(exp1[1], metrics[0], revert_annealing=True)
    x12_1, y12_1 = get_data(exp1[1], metrics[1])

    x21_0, y21_0 = get_data(exp2[0], metrics[0], revert_annealing=False)
    y21_0 /= 0.3
    x21_1, y21_1 = get_data(exp2[0], metrics[1])

    x22_0, y22_0 = get_data(exp2[1], metrics[0], revert_annealing=False)
    x22_1, y22_1 = get_data(exp2[1], metrics[1])


    lw = 1.0
    axes[0].plot(x11_0, y11_0, color="blue", linestyle="-", linewidth=lw, label=exps_labels[0])
    axes2[0].plot(x11_1, y11_1, color="purple", linestyle="-", linewidth=lw)

    axes[1].plot(x12_0, y12_0, color="blue", linestyle="-", linewidth=lw, label=exps_labels[0])
    axes2[1].plot(x12_1, y12_1, color="purple", linestyle="-", linewidth=lw)

    axes[2].plot(x21_0, y21_0, color="blue", linestyle="-", linewidth=lw, label=exps_labels[0])
    axes2[2].plot(x21_1, y21_1, color="purple", linestyle="-", linewidth=lw)

    axes[3].plot(x22_0, y22_0, color="blue", linestyle="-", linewidth=lw, label=exps_labels[0])
    axes2[3].plot(x22_1, y22_1, color="purple", linestyle="-", linewidth=lw)

    axes[0].set_title(r"$c_{\operatorname{CA}} = 0.3, \gamma_{\operatorname{CA}} = 1.0$")
    axes[1].set_title(r"$c_{\operatorname{CA}} = 1.0, \gamma_{\operatorname{CA}} = 1.0$")
    axes[2].set_title(r"$c_{\operatorname{CA}} = 0.3, \gamma_{\operatorname{CA}} = 0.0$")
    axes[3].set_title(r"$c_{\operatorname{CA}} = 1.0, \gamma_{\operatorname{CA}} = 0.0$")

    for ax in axes:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(human_format))
        ax.set_yscale("log")

    for i in [1, 3]:
        axes2[i].set_ylabel(metrics_labels[1], color="purple")
        axes2[i].tick_params(axis="y", colors="purple")

    for ax2 in axes2:
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(human_format))
        ax2.tick_params(axis="y", colors="purple")
        ax2.set_ylim([-0.2, 2.2])

    fig.suptitle("Kangaroo", fontsize=14)

    path = Path(f"plots/curves/kangaroo_ca_loss_vs_goals.png")
    save_fig(fig, path, dpi=200)


if __name__ == "__main__":
    main()
