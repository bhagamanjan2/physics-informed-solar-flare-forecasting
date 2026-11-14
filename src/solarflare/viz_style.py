"""
Visualization style configuration and helper functions.
Owned by Agent A6 (Visualization Lead - EDA).
Adheres strictly to capstone visual standards:
  - Color-blind safe palette (Okabe-Ito / ColorBrewer inspired).
  - Consistent flare class colors (C, M, X).
  - Publication typography (font size >= 11, light grid, readable units).
  - Standardized 3-line Markdown takeaway boxes (What you see / Why it matters / Say this in the viva).
  - Standardized figure saver at 200 dpi.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Color-blind safe colors for flare classes (Okabe-Ito palette)
CLASS_COLORS = {
    "A": "#9E9E9E",  # Light grey
    "B": "#0072B2",  # Muted blue
    "C": "#009E73",  # Bluish green (Target threshold C-class)
    "M": "#E69F00",  # Amber / Orange (Moderate flares)
    "X": "#D55E00",  # Vermillion / Red-orange (Major/Extreme flares)
    "quiet": "#78909C"
}

# Clue group colors (G0: Brightness, G1: History, G2: QPP/STFT, G3: Phase)
CLUE_COLORS = {
    "G0": "#0072B2",  # Blue (Recent brightness)
    "G1": "#E69F00",  # Orange (Flare history)
    "G2": "#CC79A7",  # Reddish purple (Pulsations / STFT)
    "G3": "#009E73",  # Green (Phase flag)
}

# Data partition colors
SPLIT_COLORS = {
    "train": "#1F77B4",
    "val": "#FF7F0E",
    "test": "#2CA02C",
    "gap": "#7F7F7F",
    "skip": "#B0BEC5"
}

# General UI palette
PALETTE = {
    "primary": "#1E88E5",
    "secondary": "#FFC107",
    "background": "#FAFAFA",
    "dark": "#212121",
    "light": "#ECEFF1"
}


def set_viz_style() -> None:
    """Set global matplotlib and seaborn style for clean, publication-quality figures."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "figure.dpi": 150,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "grid.color": "#B0BEC5",
        "axes.edgecolor": "#78909C",
        "axes.linewidth": 1.0,
    })


def get_class_color(cls: str) -> str:
    """Return consistent color for a flare class."""
    return CLASS_COLORS.get(str(cls).strip().upper(), "#9E9E9E")


def setup_log_flux_axis(ax: plt.Axes, ymin: float = 1e-8, ymax: float = 5e-4, show_class_labels: bool = True) -> None:
    """
    Format a matplotlib axis for GOES X-ray flux with log10 scale
    and standard NOAA flare class threshold labels (A, B, C, M, X).
    """
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(r"GOES 1-8 $\mathrm{\AA}$ Flux ($\mathrm{W/m^2}$)")
    
    # Class threshold guidelines
    class_thresholds = [
        ("A", 1e-8, CLASS_COLORS["A"]),
        ("B", 1e-7, CLASS_COLORS["B"]),
        ("C", 1e-6, CLASS_COLORS["C"]),
        ("M", 1e-5, CLASS_COLORS["M"]),
        ("X", 1e-4, CLASS_COLORS["X"]),
    ]
    
    for cls, thresh, col in class_thresholds:
        if ymin <= thresh <= ymax:
            ax.axhline(thresh, color=col, linestyle=":", alpha=0.55, linewidth=1.0, zorder=1)
            if show_class_labels:
                ax.text(
                    0.01, thresh * 1.15, f"{cls}-class ($10^{{{int(np.log10(thresh))}}}$)",
                    transform=ax.get_yaxis_transform(),
                    color=col, fontsize=9, fontweight="bold", alpha=0.85,
                    va="bottom"
                )


def save_figure(fig: plt.Figure, filename: str, save_dir: str = "results/figures", dpi: int = 200) -> str:
    """Save figure to results/figures/ at specified DPI (default 200)."""
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def format_takeaway_box(what_you_see: str, why_it_matters: str, viva_say: str) -> str:
    """
    Format standard 3-line Markdown box following capstone guidelines:
      - What you see
      - Why it matters
      - Say this in the viva
    """
    box = (
        f"> **What you see:** {what_you_see}\n>\n"
        f"> **Why it matters:** {why_it_matters}\n>\n"
        f"> **Say this in the viva:** *\"{viva_say}\"*"
    )
    return box

# Aliases for interface compliance
add_takeaway_box = format_takeaway_box
three_line_box = format_takeaway_box
