from io import BytesIO
from pathlib import Path
from typing import Optional, Union, TypeAlias, List, Tuple, Any

import torch as th
from PIL import Image
from matplotlib.collections import LineCollection
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.patches import Patch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.stats import binned_statistic_2d
import imageio.v2 as iio
from matplotlib.colors import ListedColormap

from nsfr import NSFReasoner
from nsfr.utils.logic import get_indices_by_predname
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

def make_transparent_cmap(cmap, N=256, K=32):
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap, N)
    else:
        cmap = cmap

    colors = cmap(np.linspace(0, 1, N))
    colors[:K, -1] = np.linspace(0, 1, K)  # modify alpha channel
    return ListedColormap(colors)

def create_heatmap_fig(
        H: np.ndarray,
        cmap: Optional[Colormap] = None,
        ax: Optional[plt.Axes] = None,
        *args,
        **kwargs
) -> (plt.Figure, plt.Axes, AxesImage):
    # Create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Plot image
    im = ax.imshow(H, cmap=cmap, *args, **kwargs)

    # Do not show ticks
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax, im

def mask_array(arr: np.ndarray, value: float) -> np.ndarray:
    if np.isnan(value):
        return np.ma.masked_where(np.isnan(arr), arr)

    return np.ma.masked_where(arr == value, arr)

def create_prior_fig(positions: np.ndarray, values: np.ndarray) -> (plt.Figure, plt.Axes):

    fig, ax = plt.subplots(figsize=(6, 6))

    # Set limits and aspect
    ax.set_xlim(-1, 1)
    ax.set_ylim(1, -1)
    ax.set_aspect('equal', adjustable='box')

    # Remove axis labels
    ax.set_xticks(np.arange(-1, 1.01, 0.25))
    ax.set_yticks(np.arange(-1, 1.01, 0.25))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False, top=False, right=False)

    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # Plot positions
    x = positions[:, 0]
    y = positions[:, 1]
    scatter = ax.scatter(x, y, c=values, cmap='binary', s=50, marker='s', edgecolors='w', vmin=0, vmax=1, linewidths=0.2)

    # Mark the origin
    ax.plot(0, 0, marker='x', color='black', markersize=10)

    fig.tight_layout()

    return fig, ax


def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=1.0, alpha=1.0, ax=None):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc

def create_multipath_fig(coords: List[np.array], frame_size: (float, float), moving_average: int = 1, ax: Optional[plt.Axes] = None, cmap: Optional[Colormap] = plt.get_cmap('RdYlGn_r'), *args, **kwargs) -> (plt.Figure, plt.Axes):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.set_xlim(0, frame_size[0])
    ax.set_ylim(frame_size[1], 0)
    ax.set_aspect('equal', adjustable='box')

    for path in coords:
        if moving_average > 1:
            # path = np.apply_along_axis(
            #     lambda a: np.convolve(a, np.ones(moving_average) / moving_average, mode='valid'),
            #     axis=0, arr=path
            # )
            path = path[::moving_average]
        x, y = path[:, 0], path[:, 1]

        if cmap is None:
            ax.plot(x, y, c='black', linewidth=0.5)
        else:
            colorline(x, y, cmap=cmap, linewidth=0.5, ax=ax)

    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    return fig, ax


def fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvas(fig)  # Bind canvas to figure
    canvas.draw()  # Render the figure

    # Get the RGBA buffer from the canvas
    buf = np.asarray(canvas.buffer_rgba())

    # Drop the alpha channel
    rgb = buf[:, :, :3]

    return rgb

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

def get_heatmap(x: np.ndarray, y: np.ndarray, values: Optional[np.ndarray], bins: (int, int), range: ((float, float), (float, float)), mode: str = "mean") -> np.ndarray:
    heatmap, _, _, _ = binned_statistic_2d(
        x, y, values,
        statistic=mode,
        bins=bins,
        range=range
    )
    return heatmap

def fig_add_title(fig: plt.Figure, title: str, subtitle: Optional[str] = None, **kwargs) -> (plt.Text, Optional[plt.Text]):
    ax = fig.gca()

    if subtitle is not None:
        title_text = fig.suptitle(title, fontsize=10, y=0.95, **kwargs)
        subtitle_text = ax.set_title(subtitle, fontsize=8)
    else:
        title_text = ax.set_title(title, **kwargs)
        subtitle_text = None

    return (title_text, subtitle_text)

def save_fig(fig: plt.Figure, path: Path, close_fig: bool = True, tight_layout: bool = True, *args, **kwargs):
    default_kwargs = dict(dpi=150)
    if tight_layout:
        fig.tight_layout()
        default_kwargs["bbox_inches"] = "tight"
    default_kwargs.update(kwargs)
    fig.savefig(path, *args, **default_kwargs)
    print(f"Saved figure to {path}")
    if close_fig:
        plt.close(fig)


def _discretize_frame(frame, resolution: int, device: th.device, position: Optional[Tuple[float, float]] = None, dtype=th.int) -> Tuple[th.Tensor, Tuple[int, int]]:
    frame_size = (frame[0][1] - frame[0][0], frame[1][1] - frame[1][0])
    min_dim = min(frame_size)
    cell_size = min_dim / resolution
    grid_shape = (int(frame_size[0] // cell_size), int(frame_size[1] // cell_size))
    num_grid_cells = grid_shape[0] * grid_shape[1]

    result = th.tensor([1, 0, 0, 0], device=device, dtype=dtype).repeat(num_grid_cells, 1)
    idx = 0
    for i in range(grid_shape[0]):
        x = cell_size * (i + 0.5) + frame[0][0] if position is None else position[0]
        for j in range(grid_shape[1]):
            y = cell_size * (j + 0.5) + frame[1][0] if position is None else position[1]
            result[idx, 1] = x
            result[idx, 2] = y
            idx += 1

    return result, grid_shape

def discretize_frame(env_name: str, resolution: int, device: th.device, position: Optional[Tuple[float, float]] = None, dtype=th.int) -> Tuple[th.Tensor, Tuple[int, int]]:
    frame_size = FRAME_SIZE[env_name]
    return _discretize_frame([[0, frame_size[0]], [0, frame_size[1]]], resolution, device, position, dtype)


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


def get_logic_actor_predicate_heatmap(logic_actor: NSFReasoner, player_input: th.Tensor, obj_input: th.Tensor, pred_names: List[str], frame_shape: (int, int)) -> List[np.ndarray]:
    x = obj_input
    x[:, 0] = player_input
    logic_actor(x)

    heatmaps = []
    for pred_name in pred_names:
        indices = get_indices_by_predname(pred_name, logic_actor.atoms)
        values = th.max(logic_actor.V_T[:, indices], dim=1)[0].view(*frame_shape)
        heatmap = to_np(values).T
        heatmaps.append(heatmap)

    return heatmaps


def get_logic_critic_heatmap(logic_critic: th.nn.Module, player_input: th.Tensor, obj_input: th.Tensor, frame_shape: (int, int)) -> np.ndarray:
    x = obj_input
    x[:, 0] = player_input
    output = logic_critic(x).view(*frame_shape)
    heatmap = to_np(output).T

    return heatmap


class Animator:

    def __init__(self, fig: plt.Figure, path: Path, fps: float = 25, codec: str = 'libx264', quality: int = 8, dpi: int = 150, **kwargs):
        self.fig = fig
        self.path = path
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self.dpi = dpi

        self._writer = iio.get_writer(
            self.path,
            fps=self.fps,
            codec=self.codec,
            quality=self.quality,  # Quality (lower is better; 0-10 range)
            **kwargs
        )

    def append(self, **kwargs):
        buf = BytesIO()
        default_kwargs = dict(dpi=self.dpi, format='png', bbox_inches='tight')
        default_kwargs.update(kwargs)
        self.fig.savefig(buf, **default_kwargs)
        buf.seek(0)
        img = np.array(Image.open(buf).convert("RGBA"))
        self._writer.append_data(img)

    def save_still(self, path: Path):
        save_fig(self.fig, path, close_fig=False)

    def close(self):
        self._writer.close()
