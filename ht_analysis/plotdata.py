#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utils to plot data (thinking about pyplot)
"""
__author__ = "Miguel Hernández Cabronero <miguel.hernandez@uab.cat>"
__date__ = "10/09/2019"

import collections
from matplotlib import pyplot as plt

class PlottableData:
    alpha = 0.75

    def __init__(self, data=None, axis_labels=None, label=None, extra_kwargs=None):
        self.data = data
        self.axis_labels = axis_labels
        self.label = label
        self.extra_kwargs = extra_kwargs if extra_kwargs is not None else {}

    def render(self):
        """Render data in current figure
        """
        raise NotImplementedError()

    def render_axis_labels(self):
        """Add axis labels in current figure - don't show or save the result
        """
        raise NotImplementedError()


class PlottableData2D(PlottableData):
    """Plot 2D data using plt.plot()
    """

    def __init__(self, x_values, y_values,
                 x_label=None, y_label=None,
                 label=None, extra_kwargs=None,
                 remove_duplicates=True,
                 alpha=None):
        """
        :param x_values, y_values: values to be plotted (only a reference is kept)
        :param x_label, y_label: axis labels
        :param label: line legend label
        :param extra_kwargs: extra arguments to be passed to plt.plot
        """
        assert len(x_values) == len(y_values)
        if remove_duplicates:
            found_pairs = collections.OrderedDict()
            for x, y in zip(x_values, y_values):
                found_pairs[(x, y)] = (x, y)
            x_values, y_values = zip(*found_pairs.values())

        super().__init__(data=(x_values, y_values), axis_labels=(x_label, y_label),
                         label=label, extra_kwargs=extra_kwargs)
        self.x_values = x_values
        self.y_values = y_values
        self.x_label = x_label
        self.y_label = y_label
        self.alpha = alpha if alpha is not None else self.alpha

    def render(self):
        """Plot 2D data using plt.plot()

        :param anchor: if not None, the difference self-anchor is rendered instead of self
        """
        plt.plot(self.x_values, self.y_values, label=self.label, alpha=self.alpha,
                 **self.extra_kwargs)
        self.render_axis_labels()
        if self.label is not None:
            plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1))

    def render_axis_labels(self):
        """        Show the labels in label list (if not None) or in self.axis_label_list
          (if label_list None) in the current figure.
        """
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

    def diff(self, other, ylabel_affix=" difference"):
        assert len(self.x_values) == len(other.x_values)
        assert len(self.y_values) == len(other.y_values)
        assert all(s == o for s, o in zip(self.x_values, other.x_values))
        return PlottableData2D(x_values=self.x_values,
                               y_values=[s - o for s, o in zip(self.y_values, other.y_values)],
                               x_label=self.x_label,
                               y_label=f"{self.y_label}{ylabel_affix}")


LineData = PlottableData2D

class ScatterData(PlottableData2D):
    alpha = 0.5

    def render(self):
        plt.scatter(self.x_values, self.y_values, label=self.label, alpha=self.alpha,
                    **self.extra_kwargs)
        self.render_axis_labels()
        if self.label:
            plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
