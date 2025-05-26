from pathlib import Path
from typing import Optional, Union, TypeAlias, List, Tuple, Any

from matplotlib.image import AxesImage
from matplotlib.legend import Legend
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.patches import Patch

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
