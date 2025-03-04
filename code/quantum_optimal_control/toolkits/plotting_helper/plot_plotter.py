from .generic_plotter import GenericPlotter


class PlotPlotter(GenericPlotter):

    def __init__(self, fig, ax, xdata, ydata, **inputs):
        super().__init__(fig, ax, **inputs)
        self.xdata = xdata
        self.ydata = ydata

    def _draw(self):
        if self.style is None:
            return self.ax.plot(self.xdata, self.ydata)
        else:
            return self.ax.plot(self.xdata, self.ydata, **self.style)