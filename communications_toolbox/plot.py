"""This class encapsulates tools for plotting several plot types.

This class encapsulates tools for plotting several plot types including plots
for in-phase quadrature data using scatter plots and spectral waterfalls using
heat maps.
"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


class ScatterPlotter:
    """Scatter plotter.

    Scatter plotter for plotting live scatter display.
    """

    def __init__(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        xlimit: Tuple[float, float],
        ylimit: Tuple[float, float],
        figure_size: Tuple[float, float] = (5.0, 5.0),
    ) -> None:
        """Initializes scatter plotter.

        Args:
            title: Title of plot.
            xlabel: X-axis label of scatter plot.
            ylabel: Y-axis label of scatter plot.
            xlimit: X-axis limit of scatter plot.
            ylimit: Y-axis limit of scatter plot.
            figure_size: Size of figure. Defaults to (10.0, 10.0).
        """
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._xlimit = xlimit
        self._ylimit = ylimit

        self._fig = plt.figure(figsize=figure_size)
        self._axes = self._fig.add_subplot()

    def update(self, data: np.ndarray) -> None:
        """Updates scatter plot with 2-dimensional data.

        Args:
            data: Data to plot. Data is assumed to be in the [N x 2], where N
                is the number of data point samples to plot.
        """
        self._axes.cla()
        self._set_axes()
        self._axes.scatter(data[:, 0], data[:, 1])
        plt.figure(self._fig.number)
        plt.pause(0.05)

    def _set_axes(self) -> None:
        """Sets axes labels, view, and plot."""
        self._axes.set_title(self._title)
        self._axes.set_xlabel(self._xlabel)
        self._axes.set_ylabel(self._ylabel)
        self._axes.set_xlim(self._xlimit)
        self._axes.set_ylim(self._ylimit)
        self._axes.grid()


class HeatmapPlotter:
    """Heatmap plotter.

    Heatmap plotter for plotting live heatmap display.
    """

    def __init__(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        figure_size: Tuple[float, float] = (5.0, 5.0),
    ) -> None:
        """Initializes heatmap plotter.

        Args:
            title: Title of plot.
            xlabel: X-axis label of scatter plot.
            ylabel: Y-axis label of scatter plot.
            xlimit: X-axis limit of scatter plot.
            ylimit: Y-axis limit of scatter plot.
            figure_size: Size of figure. Defaults to (10.0, 10.0).
        """
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel

        self._fig = plt.figure(figsize=figure_size)
        self._axes = self._fig.add_subplot()

    def update(self, data: np.ndarray) -> None:
        """Updates heatmap plot with 2-dimensional data.

        Args:
            data: Data to plot as a heatmap. Data is assumed to be in the
                [H x W], where H is the height and W is the width.
        """
        self._axes.cla()
        self._set_axes()
        self._axes.imshow(data)
        plt.figure(self._fig.number)
        plt.pause(0.05)

    def _set_axes(self) -> None:
        """Sets axes labels, view, and plot."""
        self._axes.set_title(self._title)
        self._axes.set_xlabel(self._xlabel)
        self._axes.set_ylabel(self._ylabel)
        self._axes.grid()
