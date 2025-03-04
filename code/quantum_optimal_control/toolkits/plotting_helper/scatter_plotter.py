from .generic_plotter import GenericPlotter
import matplotlib
import numpy as np


class ScatterPlotter(GenericPlotter):

    def __init__(self,
                 fig: matplotlib.figure.Figure, 
                 ax: matplotlib.axes, 
                 xdata, 
                 ydata, 
                 **inputs):
        super().__init__(fig, ax, **inputs)
        self.xdata = xdata
        self.ydata = ydata

    def _draw(self):
        if self.style is None:
            return self.ax.scatter(self.xdata, self.ydata)
        else:
            return self.ax.scatter(self.xdata, self.ydata, **self.style)