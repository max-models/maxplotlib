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


class Color:
    def __init__(self, hex_color):
        self.hx = hex_color
        self.rgb = tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

        self.rgb_dec = [i / 255 for i in self.rgb]
        self.rgb_dec_str = ["{:.6f}".format(i) for i in self.rgb_dec]

        self.rgb_inv = tuple(np.subtract((256, 256, 256), self.rgb))
        self.rgb_dec_inv = [1.0 - c for c in self.rgb_dec]
        # self.pgf_col_str = '\definecolor{currentstroke}{rgb}{'
        self.pgf_col_str = "{rgb}{"
        self.pgf_col_str += self.rgb_dec_str[0] + ","
        self.pgf_col_str += self.rgb_dec_str[1] + ","
        self.pgf_col_str += self.rgb_dec_str[2] + "}%"

    def invert(self):
        inverted_color = Color(self.hx)

    def define_color_str(self, name):
        hex_str = self.hx.replace("#", "")
        out_str = "\\definecolor{" + name + "}{HTML}{" + hex_str + "}"
        out_str += " " * (70 - len(out_str)) + "% https://www.colorhexa.com/" + hex_str
        return out_str

    def __str__(self):
        return self.hx


# https://matplotlib.org/stable/gallery/color/named_colors.html
def mcolors2mplcolors(colors):
    names = sorted(colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    col_dict = dict()
    for name in names:
        col_dict[name] = Color(colors[name])
    return col_dict


def import_colors(cmap="pastel"):
    col_dict = dict()
    col_dict["pastel"] = mcolors2mplcolors(mcolors.CSS4_COLORS)
    col_dict["cmap2"] = mcolors2mplcolors(mcolors.CSS4_COLORS)
    col_dict["thesis_colors"] = mcolors2mplcolors(mcolors.CSS4_COLORS)

    col_dict["pastel"]["white"] = Color("#ffffff")
    col_dict["pastel"]["black"] = Color("#000000")
    col_dict["pastel"]["yellow"] = Color("#FFFFB3")
    col_dict["pastel"]["dkyellow"] = Color("#FFED6F")
    col_dict["pastel"]["purple"] = Color("#BEBADA")
    col_dict["pastel"]["dkpurple"] = Color("#BC80BD")
    col_dict["pastel"]["red"] = Color("#FB8072")
    col_dict["pastel"]["ltred"] = Color("#FFCCCB")
    col_dict["pastel"]["dkred"] = Color("#CB0505")
    col_dict["pastel"]["orange"] = Color("#FDB462")
    col_dict["pastel"]["dkgold"] = Color("#B8860B")
    col_dict["pastel"]["blue"] = Color("#80B1D3")
    col_dict["pastel"]["dkblue"] = Color("#00008B")
    col_dict["pastel"]["deepskyblue"] = Color("#1f78b4")
    col_dict["pastel"]["green"] = Color("#B3DE69")
    col_dict["pastel"]["ltgreen"] = Color("#CCEBC5")
    col_dict["pastel"]["dkgreen"] = Color("#006400")
    col_dict["pastel"]["bluegreen"] = Color("#8DD3C7")
    col_dict["pastel"]["pink"] = Color("#FCCDE5")
    col_dict["pastel"]["ltgray"] = Color("#D9D9D9")
    col_dict["pastel"]["dkgray"] = Color("#515151")
    col_dict["pastel"]["brown"] = Color("#D2691E")

    col_dict["cmap2"]["white"] = Color("#ffffff")
    col_dict["cmap2"]["black"] = Color("#000000")
    col_dict["cmap2"]["yellow"] = Color("#ffff99")
    col_dict["cmap2"]["dkyellow"] = Color("#FFED6F")
    col_dict["cmap2"]["ltpurple"] = Color("#cab2d6")
    col_dict["cmap2"]["purple"] = Color("#6a3d9a")
    col_dict["cmap2"]["dkpurple"] = Color("#BC80BD")
    col_dict["cmap2"]["red"] = Color("#e31a1c")
    col_dict["cmap2"]["ltred"] = Color("#fb9a99")
    col_dict["cmap2"]["dkred"] = Color("#CB0505")
    col_dict["cmap2"]["ltorange"] = Color("#fdbf6f")
    col_dict["cmap2"]["orange"] = Color("#ff7f00")
    col_dict["cmap2"]["blue"] = Color("#1f78b4")
    col_dict["cmap2"]["dkblue"] = Color("#00008B")
    col_dict["cmap2"]["deepskyblue"] = Color("#1f78b4")
    col_dict["cmap2"]["green"] = Color("#33a02c")
    col_dict["cmap2"]["ltgreen"] = Color("#b2df8a")
    col_dict["cmap2"]["dkgreen"] = Color("#006400")
    col_dict["cmap2"]["bluegreen"] = Color("#8DD3C7")
    col_dict["cmap2"]["pink"] = Color("#FCCDE5")
    col_dict["cmap2"]["ltgray"] = Color("#D9D9D9")
    col_dict["cmap2"]["dkgray"] = Color("#515151")
    col_dict["cmap2"]["brown"] = Color("#b15928")


def import_col_list(cmap="cmap2"):
    col_dict = import_colors(cmap)
    col_list = [
        col_dict["black"],
        col_dict["blue"],
        col_dict["red"],
        col_dict["green"],
        col_dict["orange"],
        col_dict["purple"],
        col_dict["gray"],
        col_dict["brown"],
        # col_dict['green'],
    ]
    return col_list


def id2color(id, cmap="cmap2"):
    col_list = import_col_list(cmap=cmap)
    return col_list[id % len(col_list)].hx


#
#  ,------.,--.                                                 ,--.
#  |  .---'`--' ,---. ,--.,--.,--.--. ,---.      ,---.  ,---. ,-'  '-.,--.,--. ,---.
#  |  `--, ,--.| .-. ||  ||  ||  .--'| .-. :    (  .-' | .-. :'-.  .-'|  ||  || .-. |
#  |  |`   |  |' '-' ''  ''  '|  |   \   --.    .-'  `)\   --.  |  |  '  ''  '| '-' '
#  `--'    `--'.`-  /  `----' `--'    `----'    `----'  `----'  `--'   `----' |  |-'
#              `---'                                                          `--'


linestyles = dict()
linestyles["solid"] = "solid"  # Same as (0, ()) or '-'
linestyles["dotted"] = "dotted"  # Same as (0, (1, 1)) or '.'
linestyles["dashed"] = "dashed"  # Same as '--'
linestyles["dashdot"] = "dashdot"  # Same as '-.'
linestyles["loosely dotted"] = (0, (1, 10))
linestyles["dotted"] = (0, (1, 1))
linestyles["densely dotted"] = (0, (1, 1))

linestyles["loosely dashed"] = (0, (5, 10))
linestyles["dashed"] = (0, (5, 5))
linestyles["densely dashed"] = (0, (5, 1))

linestyles["loosely dashdotted"] = (0, (3, 10, 1, 10))
linestyles["dashdotted"] = (0, (3, 5, 1, 5))
linestyles["densely dashdotted"] = (0, (3, 1, 1, 1))

linestyles["dashdotdotted"] = (0, (3, 5, 1, 5, 1, 5))
linestyles["loosely dashdotdotted"] = (0, (3, 10, 1, 10, 1, 10))
linestyles["densely dashdotdotted"] = (0, (3, 1, 1, 1, 1, 1))

linestyle_list = [linestyles[l] for l in linestyles]
linestyle_list_ordered = [
    linestyles["solid"],
    linestyles["densely dashdotted"],
    linestyles["dashed"],
    linestyles["dotted"],
]


class figure:
    def __init__(
        self,
        load_file="",
        nx_subplots=1,
        ny_subplots=1,
        width=426.79135,
        figsize=None,
        scale_width=1,
        dpi=300,
        threeD=False,
        ratio="golden",
        legend=True,
        axes_grid=False,
        gridspec_kw={"wspace": 0.08, "hspace": 0.1},
        legend_position="upper right",
        filename="MaxFigureClassInstance",
        directory="./",
        cmap="cmap2",
        fontsize=14,
        tex_fonts=True,
    ):
        # if width == 'singlecol':
        #    width = 426.79135 / 2.0
        # if width == 'doublecol':
        #    width = 426.79135

        self.nx_subplots = nx_subplots
        self.ny_subplots = ny_subplots
        self.width = width * scale_width
        self.dpi = dpi
        self.threeD = threeD
        self.ratio = ratio
        self.legend = legend
        self.filename = filename
        self.directory = directory
        self.axes_grid = axes_grid
        self.gridspec_kw = gridspec_kw
        self.cmap = cmap
        self.col_list = import_colors(self.cmap)

        self.axes_grid_which = "major"
        self.grid_alpha = 1.0
        self.grid_linestyle = linestyles["densely dotted"]
        self.fontsize = fontsize
        # print(self.directory)
        if len(self.directory) > 0:
            if not self.directory[-1] == "/":
                self.directory += "/"

        #
        # plt.style.use('seaborn')
        #
        if not figsize == None:
            self.width = figsize[0]
            self.ratio = figsize[0] / figsize[1]

        if tex_fonts:
            self.setup_tex_fonts()

        if not load_file == "":
            self.load(load_file)
        elif threeD:
            self.create_3dplot()
        else:
            self.create_lineplot()

    def setup_tex_fonts(self):
        self.tex_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
            "font.family": "serif",
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": self.fontsize,
            "font.size": self.fontsize,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": self.fontsize,
            "xtick.labelsize": self.fontsize,
            "ytick.labelsize": self.fontsize,
        }
        self.setup_plotstyle(
            tex_fonts=self.tex_fonts,
            axes_grid=self.axes_grid,
            axes_grid_which=self.axes_grid_which,
            grid_alpha=self.grid_alpha,
            grid_linestyle=self.grid_linestyle,
        )

    def create_lineplot(self):
        self.fig, self.axs = plt.subplots(
            self.ny_subplots,
            self.nx_subplots,
            figsize=self.set_size(
                self.width,
                ratio=self.ratio,  #
            ),  # sharex=True,#sharex='all', sharey='all',
            dpi=self.dpi,
            constrained_layout=False,
            gridspec_kw=self.gridspec_kw,
        )

    def create_3dplot(self):
        self.fig = plt.figure(figsize=self.set_size(self.width), dpi=self.dpi)
        self.axs = self.fig.add_subplot(111, projection="3d")

    def setup_plotstyle(
        self,
        tex_fonts=True,
        axes_grid=True,
        axes_grid_which="major",
        grid_alpha=0.0,
        grid_linestyle="dotted",
    ):
        if tex_fonts:
            plt.rcParams.update(self.tex_fonts)

        plt.rcParams["axes.grid"] = axes_grid  # False     ## display grid or not
        # gridlines at major, minor or both ticks
        plt.rcParams["axes.grid.which"] = axes_grid_which
        plt.rcParams["grid.alpha"] = grid_alpha  # transparency, between 0.0 and 1.0
        plt.rcParams["grid.linestyle"] = grid_linestyle

        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

        # This is to avoid the overlapping tick labels.
        plt.rcParams["xtick.major.pad"] = 8
        plt.rcParams["ytick.major.pad"] = 8
        # plt.rc('text.latex', preamble=r'\usepackage{wasysym}')

    def set_common_xlabel(self, xlabel="common X"):
        self.fig.text(
            0.5, -0.075, xlabel, va="center", ha="center", fontsize=self.fontsize
        )
        # fig.text(0.04, 0.5, 'common Y', va='center', ha='center', rotation='vertical', fontsize=rcParams['axes.labelsize'])

    def set_size(self, width, fraction=1, ratio="golden"):
        """Set figure dimensions to avoid scaling in LaTeX.
        Parameters
        ----------
        width: float
                Document textwidth or columnwidth in pts
        fraction: float, optional
                Fraction of the width which you wish the figure to occupy
        Returns
        -------
        fig_dim: tuple
                Dimensions of figure in inches
        """
        #
        # Width of figure (in pts)
        if width == "thesis":
            width_pt = 426.79135
        elif width == "beamer":
            width_pt = 307.28987
        else:
            width_pt = width

        # Width of figure
        fig_width_pt = width_pt * fraction

        # Convert from pt to inches
        inches_per_pt = 1 / 72.27

        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        golden_ratio = (5**0.5 - 1) / 2

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt

        if ratio == "golden":
            # Figure height in inches
            fig_height_in = fig_width_in * golden_ratio

        elif ratio == "square":
            # Figure height in inches
            fig_height_in = fig_width_in
        # print('ratio',ratio)
        if type(ratio) == int or type(ratio) == float:
            fig_height_in = fig_width_in * ratio

        fig_dim = (fig_width_in, fig_height_in)

        return fig_dim

    def get_axis(self, subfigure):
        if subfigure == -1:
            return self.axs
        elif not isinstance(subfigure, list):
            return self.axs[subfigure]
        elif isinstance(subfigure, list) and len(subfigure) == 2:
            return self.axs[subfigure[0], subfigure[1]]

    def get_limits(self, ax=None):
        if ax == None:
            xxmin, xxmax = self.axs.get_xlim()
            yymin, yymax = self.axs.get_ylim()
        else:
            xxmin, xxmax = ax.get_xlim()
            yymin, yymax = ax.get_ylim()
        arr = [xxmin, xxmax, yymin, yymax]
        return arr

    def set_labels(self, delta, point, subfigure=-1, axis="x"):
        ax = self.get_axis(subfigure)
        plt.sca(ax)
        if axis == "x":
            xmin, xmax = ax.get_xlim()
            width = int((xmax - xmin) / delta + 1) * delta
            locs, labels = plt.xticks()
            i0 = int(xmin / delta)
            i1 = int(xmax / delta)
            xvec = []
            xvec = np.arange(point - width, point + width + delta, delta)
            xvec += point
            xvec = xvec[xvec >= xmin]
            xvec = xvec[xvec <= xmax]
            new_labels = [i * delta for i in range(i0, i1)]
            # if precision == 0: new_labels = [int(x) for x in new_labels]
            plt.xticks(xvec, xvec)
        if axis == "y":
            return

    def scale_axis(
        self,
        subfigure=-1,
        axis="x",
        axs_in=None,
        scale=1.0,
        shift=0,
        precision=2,
        delta=-1,
        includepoint=-1,
        nticks=5,
        locs_labels=None,
    ):
        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.xticks.html
        # if subfigure_x == -1 and subfigure_y == -1 and nx_subplots > 1 and ny_subplots > 1:
        #    print('enter subfigure_x and subfigure_y!')
        #    return
        # if subfigure_x == -1 and subfigure_y == -1:
        #    ax = self.axs[]
        if subfigure == -1:
            ax = self.axs
        elif not isinstance(subfigure, list):
            ax = self.axs[subfigure]
        elif isinstance(subfigure, list) and len(subfigure) == 2:
            ax = self.axs[subfigure[0], subfigure[1]]

        if not axs_in == None:
            ax = axs_in
        # print("precision", precision, precision, precision)
        plt.sca(ax)
        if axis == "x":
            if locs_labels == None:
                xmin, xmax = ax.get_xlim()
                locs, labels = plt.xticks()
                if delta == -1 and includepoint == -1:
                    new_labels = [round((x + shift) * scale, precision) for x in locs]
                    if precision == 0:
                        new_labels = [int(x) for x in new_labels]
                else:
                    if delta == -1:
                        delta = (xmax - xmin) / (nticks - 1)
                    if includepoint == -1:
                        includepoint = xmin
                    width = int((xmax - xmin) / delta + 1) * delta
                    i0 = int(xmin / delta)
                    i1 = int(xmax / delta + 1)
                    locs = np.arange(
                        includepoint - width, includepoint + width + delta, delta
                    )
                    locs = locs[locs >= xmin - 1e-12]
                    locs = locs[locs <= xmax + 1e-12]

                    new_labels = [round((x + shift) * scale, precision) for x in locs]
                    if precision == 0:
                        new_labels = [int(y) for y in new_labels]
                new_labels = [f"${l}$" for l in new_labels]
                # plt.xticks(locs,new_labels)
                ax.set_xticks(locs)
                ax.set_xticklabels(new_labels)
                ax.axis(xmin=xmin, xmax=xmax)
            else:
                # plt.xticks(locs_labels['locs'],locs_labels['labels'])
                ax.set_xticks(locs_labels["locs"])
                ax.set_xticklabels(locs_labels["labels"])

        if axis == "y":
            if locs_labels == None:
                ymin, ymax = ax.get_ylim()
                locs, labels = plt.yticks()
                if delta == -1 and includepoint == -1:
                    new_labels = [round((y + shift) * scale, precision) for y in locs]
                    if precision == 0:
                        new_labels = [int(y) for y in new_labels]
                else:
                    if delta == -1:
                        delta = (ymax - ymin) / (nticks - 1)
                    if includepoint == -1:
                        includepoint = ymin
                    width = int((ymax - ymin) / delta + 1) * delta
                    i0 = int(ymin / delta)
                    i1 = int(ymax / delta + 1)
                    locs = np.arange(
                        includepoint - width, includepoint + width + delta, delta
                    )
                    locs = locs[locs >= ymin - 1e-12]
                    locs = locs[locs <= ymax + 1e-12]

                    new_labels = [round((y + shift) * scale, precision) for y in locs]
                    if precision == 0:
                        new_labels = [int(y) for y in new_labels]
                new_labels = [f"${l}$" for l in new_labels]
                # plt.yticks(locs,new_labels)

                ax.set_yticks(locs)
                ax.set_yticklabels(new_labels)

                ax.axis(ymin=ymin, ymax=ymax)
            else:
                # plt.yticks(locs_labels['locs'],locs_labels['labels'])
                ax.set_yticks(locs_labels["locs"])
                ax.set_yticklabels(locs_labels["labels"])

    def adjustFigAspect(self, aspect=1):
        """
        Adjust the subplot parameters so that the figure has the correct
        aspect ratio.
        """
        xsize, ysize = self.fig.get_size_inches()
        minsize = min(xsize, ysize)
        xlim = 0.4 * minsize / xsize
        ylim = 0.4 * minsize / ysize
        if aspect < 1:
            xlim *= aspect
        else:
            ylim /= aspect
        self.fig.subplots_adjust(
            left=0.5 - xlim, right=0.5 + xlim, bottom=0.5 - ylim, top=0.5 + ylim
        )

    def add_figure_label(
        self,
        label,
        pos="top left",
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round"),
        ha="left",
        va="top",
        ax=None,
    ):
        limits = self.get_limits(ax)
        # print(limits)
        lx = limits[1] - limits[0]
        ly = limits[3] - limits[2]

        if isinstance(pos, str):
            if "top" in pos:
                y = limits[2] + 0.90 * ly
            elif "center in pos":
                y = limits[2] + 0.5 * ly
            else:
                y = limits[2] + 0.05 * ly

            if "left" in pos:
                x = limits[0] + 0.1 * lx
            else:
                x = limits[0] + 0.95 * lx
        else:
            x = limits[0] + pos[0] * lx
            y = limits[2] + pos[1] * ly

        # print(x,y)
        if ax == None:
            ax = self.axs
        ax.text(
            x,
            y,
            f"{label}",
            rotation=0,
            ha=ha,
            va=va,
            bbox=bbox,
            fontsize=self.fontsize,
        )

    def savefig(
        self,
        filename="",
        formats=["png"],
        format="",
        create_sh_file=False,
        print_imgcat=True,
        format_folder=False,
        tight_layout=True,
    ):
        # self.update_figure()
        # self.fig.tight_layout()
        # print(self.directory)

        if "/" in self.filename:
            tmp = self.filename
            spl = tmp.split("/")
            self.filename = spl[-1]
            self.directory = tmp.replace(spl[-1], "")

        if not self.directory == "":
            Path(self.directory).mkdir(parents=True, exist_ok=True)

        if isinstance(format, list):
            formats = format
            format = ""
        if not format == "":
            formats = [format]
        if filename == "":
            filename = self.filename

        if formats == "all" or formats == ["all"]:
            self.dump()
            formats = ["jpg", "pdf", "pgf", "png", "svg", "txt", "pickle", "tex"]

        if isinstance(formats, str):
            formats = [formats]

        if format_folder:
            for format in formats:
                # print('self.directory',self.directory)
                Path(self.directory + "/" + format).mkdir(parents=True, exist_ok=True)

        _dir = self.directory
        for format in formats:
            if format_folder:
                self.directory = "{}{}/".format(_dir, format)
            # print('pl',format)
            if format in [
                "eps",
                "jpeg",
                "jpg",
                "pdf",
                "png",
                "ps",
                "raw",
                "rgba",
                "svg",
                "svgz",
                "tif",
                "tiff",
            ]:
                # self.fig.savefig(self.directory + filename + '.' + format,bbox_inches='tight', transparent=False)
                if tight_layout:
                    self.fig.savefig(
                        self.directory + filename + "." + format, bbox_inches="tight"
                    )
                else:
                    self.fig.savefig(self.directory + filename + "." + format)
            elif format == "pgf":
                # Save pgf figure
                self.fig.savefig(
                    self.directory + filename + "." + format, bbox_inches="tight"
                )

                # Replace pgf figure colors with colorlet
                # This is based af.
                # col_list = self.col_list
                file_str = "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                file_str += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                file_str += "%% Do not forget to add the following lines"
                for cmap in ["pastel", "cmap2"]:
                    col_list = import_colors(cmap)
                    file_str += "\n\n%% Definitions for " + cmap + "\n\n"
                    for col in col_list:
                        file_str += "%" + col_list[col].define_color_str(col) + "\n"
                file_str += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                file_str += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n"
                with open(self.directory + filename + "." + format, "r") as f:
                    for line in f:
                        file_str += line
                        for cmap in ["pastel", "cmap2"]:
                            col_list = import_colors(cmap)
                            for col in col_list:
                                if col_list[col].pgf_col_str in line:
                                    # print(line)
                                    file_str += (
                                        "\\colorlet{currentstroke}{" + col + "}%\n"
                                    )
                                    file_str += (
                                        "\\colorlet{currentfill}{" + col + "}%\n"
                                    )
                                    file_str += "\\colorlet{textcolor}{" + col + "}%\n"

                with open(self.directory + filename + "." + format, "w") as f:
                    f.write(file_str)
            elif format in ["txt", "dat", "csv"]:
                self.matplotlib2txt(self.directory + filename, format)
            elif format == "pickle":
                pickle.dump(self.fig, open(self.directory + filename + ".pickle", "wb"))
            elif format == "tex":
                import tikzplotlib

                # tikzplotlib.clean_figure()
                tikzplotlib.save(self.directory + filename + ".tex")
            else:
                try:
                    plt.savefig(
                        self.directory + filename + "." + format, bbox_inches="tight"
                    )
                except Exception as e:
                    print(
                        "ERROR: Could not save figure: "
                        + self.directory
                        + filename
                        + "."
                        + format
                    )
                    print(e)

        imgcat_formats = ["png"]
        if create_sh_file:
            with open("show_latest_image.sh", "w") as f:
                for format in formats:
                    if format in imgcat_formats:
                        f.write(
                            "imgcat " + self.directory + filename + "." + format + "\n"
                        )

        if print_imgcat and ("png" in formats or "pdf" in formats):
            if format_folder:
                self.directory = "{}{}/".format(_dir, "png")
            self.imgcat(formats)
        self.directory = _dir

        # if format in formats:
        #    print('imgcat ' + filename + '.' + format)

    def imgcat(self, formats="png"):
        imgcat_formats = ["png"]
        if isinstance(formats, str):
            formats = [formats]
        for format in formats:
            if format in imgcat_formats:
                print("imgcat " + self.directory + self.filename + "." + format)

    def matplotlib2txt(self, filename, format="txt"):
        # ax = plt.gca() # get axis handle

        x_unit = "mm"
        y_unit = "mm"

        max_len_arr = 0

        # Create a vector with each axis
        axs_vec = []
        if self.nx_subplots > 1 and self.ny_subplots > 1:
            for i in range(self.nx_subplots):
                for j in range(self.ny_subplots):
                    axs_vec.append(self.axs[i, j])
        elif self.nx_subplots > 1 or self.ny_subplots > 1:
            for i in range(self.nx_subplots * self.ny_subplots):
                axs_vec.append(self.axs[i])
        else:
            axs_vec.append(self.axs)

        # Save all the data
        line_names = []
        line_xdata = []
        line_ydata = []
        for iax, ax in enumerate(axs_vec):
            for line in ax.lines:
                line_names.append(line)
                line_xdata.append(line.get_xdata())
                line_ydata.append(line.get_ydata())
                max_len_arr = max(max_len_arr, len(line.get_xdata()))
        # print(line_names)
        data_mat = np.empty((len(line_names) * 2, max_len_arr))
        data_mat[:, :] = np.NaN
        i = 0

        header = ""

        for name, xdata, ydata in zip(line_names, line_xdata, line_ydata):
            data_mat[2 * i, 0 : len(xdata)] = xdata
            data_mat[2 * i + 1, 0 : len(ydata)] = ydata

            header += "Axial position, " + str(name) + ","

            i += 1

        np.savetxt(filename + "." + format, data_mat.T, header=header, delimiter=",")

    def dump(
        self,
        filename="",
    ):
        if filename == "":
            filename = self.filename + "_dump.txt"
        Path(self.directory + "/dump").mkdir(parents=True, exist_ok=True)
        with open(self.directory + "dump/" + filename, "wb") as file:
            file.write(cPickle.dumps(self.__dict__))

    def load(self, filename):
        with open(filename, "rb") as file:
            self.__dict__ = cPickle.loads(file.read())

    def get_lines(self):
        lines = plt.gca().lines
        out = []
        for i, line in enumerate(lines):
            line_dict = dict()
            line_dict["line"] = line
            line_dict["line_name"] = str(line)
            line_dict["line_xdat"] = line.get_xdata()
            line_dict["line_ydat"] = line.get_ydata()
            out.append(line_dict)
        return out

    def add_label_box(
        self,
        label="Test",
        xpos=0.05,
        ypos=0.95,
        rotation=0,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round"),
    ):
        self.axs.text(
            xpos,
            ypos,
            label,
            rotation=rotation,
            ha=ha,
            va=va,
            transform=self.axs.transAxes,
            bbox=bbox,
        )
        # print(label)


def fmt_scientific(x, pos):
    a, b = "{:.1e}".format(x).split("e")
    b = int(b)
    return r"${} \times 10^{{{}}}$".format(a, b)


def fmt_10pow(x, pos):
    a, b = "{:.1e}".format(x).split("e")
    b = int(b)
    return r"$10^{{{}}}$".format(b)


def fmt_int(x, pos, num_dec):
    return r"${}$".format(int(x))


def fmt_1dec(x, pos):
    return r"${}$".format(round(x, 1))


def fmt_2dec(x, pos):
    return r"${}$".format(round(x, 2))


if __name__ == "__main__":
    print("Testing maxplotlib")
    mfig = figure(filename="mpl_test")
    mfig.axs.plot([0, 1, 2, 3], [0, 0, 1, 1])
    mfig.savefig(formats=["png"])
