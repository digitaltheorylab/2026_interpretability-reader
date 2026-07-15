"""Plotting utilities."""

from cycler import cycler

import matplotlib.pyplot as plt
import seaborn as sns

SQUARE_FIGSIZE = (4.5, 4.5)
FACET_HEIGHT = 4.5
FACET_ASPECT = 7 / 4.5

def set_plot_theme():
    """Set book-wide plotting defaults."""
    colorblind = sns.color_palette("colorblind")

    sns.set_theme(
        context="notebook",
        style="white",
        palette=colorblind,
        rc={
            # Figures
            "figure.figsize": (7, 4.5),
            "figure.dpi": 150,

            # Color cycle for matplotlib calls
            "axes.prop_cycle": cycler(color=colorblind.as_hex()),

            # Titles and labels
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,

            # Ticks
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,

            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,

            # Legend
            "legend.frameon": False,

            # Saving/export
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",

            # Better text handling in vector formats
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )

    plt.rcParams.update(
        {
            "figure.autolayout": True,
        }
    )
