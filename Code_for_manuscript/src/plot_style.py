"""Shared publication plotting style for Code 10.4.

All generated paper figures use the Viridis family.  Captions belong in the
manuscript or asset manifest; figures therefore do not carry decorative titles.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def configure() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.usetex": False,
            "mathtext.fontset": "stix",
        }
    )


def colors(n: int, *, start: float = 0.12, stop: float = 0.92):
    if n <= 0:
        return []
    return plt.get_cmap("viridis")(np.linspace(start, stop, n))


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


configure()
