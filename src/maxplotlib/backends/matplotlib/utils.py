# import sys; from os.path import dirname; sys.path.append(f'{dirname(__file__)}/../../')

# import matplotlib.pylab as pylab
import math
import pickle
from pathlib import Path

import _pickle as cPickle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

def setup_tex_fonts(fontsize=14):
    """
    Sets up LaTeX fonts for plotting.
    """
    tex_fonts = {
        "text.usetex": True,
        "font.family": "serif",
        "pgf.rcfonts": False,
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    }
    plt.rcParams.update(tex_fonts)
    return tex_fonts

def setup_plotstyle(
    tex_fonts=None,
    axes_grid=False,
    axes_grid_which="major",
    grid_alpha=1.0,
    grid_linestyle="dotted",
):
    """
    Configures the plot style.
    """
    if tex_fonts:
        plt.rcParams.update(tex_fonts)
    plt.rcParams["axes.grid"] = axes_grid
    plt.rcParams["axes.grid.which"] = axes_grid_which
    plt.rcParams["grid.alpha"] = grid_alpha
    plt.rcParams["grid.linestyle"] = grid_linestyle
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.pad"] = 8
    plt.rcParams["ytick.major.pad"] = 8

def set_size(width, fraction=1, ratio="golden"):
    """
    Sets figure dimensions to avoid scaling in LaTeX.
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    fig_width_pt = width_pt * fraction
    #inches_per_pt = 1 / 72.27

    # Calculate the figure height based on the desired ratio
    if ratio == "golden":
        golden_ratio = (5**0.5 - 1) / 2
        fig_height_pt = fig_width_pt * golden_ratio
    elif ratio == "square":
        fig_height_pt = fig_width_pt
    elif isinstance(ratio, (int, float)):
        fig_height_pt = fig_width_pt * ratio
    else:
        raise ValueError("Invalid ratio specified.")
    fig_dim = (fig_width_pt, fig_height_pt)
    return fig_dim

def create_lineplot(
    nx_subplots=1,
    ny_subplots=1,
    width=426.79135,
    figsize=None,
    dpi=300,
    ratio="golden",
    gridspec_kw=None,
):
    """
    Creates a line plot figure and axes.
    """
    if figsize is not None:
        fig_width, fig_height = figsize
    else:
        fig_width, fig_height = set_size(width, ratio=ratio)

    if gridspec_kw is None:
        gridspec_kw = {"wspace": 0.08, "hspace": 0.1}

    fig, axs = plt.subplots(
        ny_subplots,
        nx_subplots,
        figsize=(fig_width, fig_height),
        dpi=dpi,
        constrained_layout=False,
        gridspec_kw=gridspec_kw,
    )
    return fig, axs