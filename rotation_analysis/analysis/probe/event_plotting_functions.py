import numpy as np
from matplotlib import pyplot as plt

from margrie_libs.margrie_libs.signal_processing.list_utils import flatten as flatten_list


def plot_raster(events, label=None):  # TODO: extract for use in many modules
    plt.eventplot(events)
    if label is not None:
        plt.title(label)


def plot_histogram(events, duration_in_samples, ax, label=None, color=None, scale_factor=1, n_bins=10):
    hist_events = np.array(flatten_list(events))
    hist, bin_edges = np.histogram(hist_events, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    ax.step(bin_centers, hist / scale_factor, color=color)
    plt.xlim([0, duration_in_samples])
    plt.title(label)


def normalise_axes(axes, space=0):
    lower = -1
    upper = max(max([ax.get_ylim()[1] for ax in axes]), 10) + space
    for ax in axes:
        ax.set_ylim([lower, upper])
