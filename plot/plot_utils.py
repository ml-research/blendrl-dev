from pathlib import Path
from typing import Optional, Union, TypeAlias, List, Tuple, Any

import torch as th
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.patches import Patch

from utils import to_np, FRAME_SIZE
from valuation.models.base import BaseValuationModel

Cmap: TypeAlias = Union[Colormap, str]

def get_cmap(cmap: Cmap, masked_color=None) -> Colormap:
    if isinstance(cmap, str):
        _cmap = plt.get_cmap(cmap).copy()
        if masked_color:
            _cmap.set_bad(masked_color)
        return _cmap

    return cmap

def create_heatmap_fig(H: np.ndarray, cmap: Optional[Colormap] = None, *args, **kwargs) -> (plt.Figure, plt.Axes, AxesImage):
    # Create figure
    fig, ax = plt.subplots()

    # Plot image
    im = ax.imshow(H, cmap=cmap, *args, **kwargs)

    # Do not show ticks
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax, im

def fig_add_legend(fig: plt.Figure, labels: List[Tuple[str, Any]]) -> Legend:
    ax = fig.gca()

    patches = [
        Patch(facecolor=color, edgecolor='black', label=name)
        for name, color in labels
    ]
    legend = ax.legend(
        handles=patches,
        loc='upper left',
        bbox_to_anchor=(1.05, 1.0),
        borderaxespad=0.0,
        handleheight=2.0,
        handlelength=2.0,
        fontsize='medium',
        frameon=False
    )

    return legend

def fig_add_title(fig: plt.Figure, title: str, subtitle: Optional[str] = None) -> (plt.Text, Optional[plt.Text]):
    ax = fig.gca()

    if subtitle is not None:
        title_text = fig.suptitle(title, fontsize=10, y=0.95)
        subtitle_text = ax.set_title(subtitle, fontsize=8)
    else:
        title_text = ax.set_title(title)
        subtitle_text = None

    return (title_text, subtitle_text)

def save_fig(fig: plt.Figure, path: Path, close_fig: bool = True, *args, **kwargs):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", *args, **kwargs)
    if close_fig:
        plt.close(fig)


def discretize_frame(env_name: str, resolution: int, device: th.device, position: Optional[Tuple[float, float]] = None) -> Tuple[th.Tensor, Tuple[int, int]]:
    frame_size = FRAME_SIZE[env_name]
    min_dim = min(frame_size)
    cell_size = min_dim / resolution
    grid_shape = (int(frame_size[0] // cell_size), int(frame_size[1] // cell_size))
    num_grid_cells = grid_shape[0] * grid_shape[1]

    result = th.tensor([1, 0, 0, 0], device=device).repeat(num_grid_cells, 1)
    idx = 0
    for i in range(grid_shape[0]):
        x = cell_size * (i + 0.5) if position is None else position[0]
        for j in range(grid_shape[1]):
            y = cell_size * (j + 0.5) if position is None else position[1]
            result[idx, 1] = x
            result[idx, 2] = y
            idx += 1

    return result, grid_shape


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
