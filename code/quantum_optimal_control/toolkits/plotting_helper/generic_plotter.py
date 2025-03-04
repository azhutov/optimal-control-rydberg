import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Get the absolute path to the style file
current_dir = os.path.dirname(os.path.abspath(__file__))
style_path = os.path.join(current_dir, "styles", "default.mplstyle")
styles = [style_path]

def getStyles():
    return styles


def getStylishFigureAxes(nrows, ncols, axes_list = False, **style):
    with plt.style.context(styles):
        fig, axes = plt.subplots(nrows, ncols, **style)

    fig.set_size_inches((4.3 / 2.54 * ncols, 2.8 / 2.54 * nrows))
    plt.subplots_adjust(
        left=(4.3-3.48-0.0235)/4.3,
        right=(4.3-0.05)/4.3,
        bottom=(2.8-2.21-0.0235)/2.8,
        top=(2.8-0.05)/2.8,
        wspace=0.5,
        hspace=1
    )

    if isinstance(axes, np.ndarray):
        return fig, axes.ravel()
    else:
        return fig, np.array([axes]) if axes_list else axes
    
def getTwinAxis(ax, **style):
    with plt.style.context(styles):
        ax2 = ax.twinx()
    return ax2


class GenericPlotter:

    def __init__(self,
                 fig: matplotlib.figure.Figure,
                 ax: matplotlib.axes,
                 xlim = None,
                 ylim = None,
                 grid = False,
                 style = None,
                 xticks = None,
                 yticks = None,
                 xlabel = None,
                 ylabel = None,
                 title = None,
                 xlabel_font = None,
                 ylabel_font = None,
                 title_font = None):
        self.fig = fig
        self.ax = ax
        self.xlim = xlim
        self.ylim = ylim
        self.grid = grid
        self.style = style
        self.xticks = xticks
        self.yticks = yticks
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.xlabel_font = xlabel_font
        self.ylabel_font = ylabel_font

        self.title_font = {} if title_font is None else title_font
        self.title_font["size"] = self.title_font.get("size", 7)

    def draw(self):
        with plt.style.context(styles):
            result = self._draw()
            
        if self.grid:
            self.ax.grid()

        if self.xlabel is not None:
            if self.xlabel_font is None:
                self.ax.set_xlabel(self.xlabel)
            else:
                self.ax.set_xlabel(self.xlabel, font=self.xlabel_font)
        if self.ylabel is not None:
            if self.ylabel_font is None:
                self.ax.set_ylabel(self.ylabel)
            else:
                self.ax.set_ylabel(self.ylabel, font=self.ylabel_font)

        if self.title is not None:
            if self.title_font is None:
                self.ax.set_title(self.title)
            else:
                self.ax.set_title(self.title, font=self.title_font)

        self.setTicks()

        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)

        return result, self.fig, self.ax

    def setTicks(self):
        if self.xticks is not None:
            self.ax.set_xticks(self.xticks)
        if self.yticks is not None:
            self.ax.set_yticks(self.yticks)

    def _draw(self):
        raise NotImplementedError("Not implemented for the generic class")