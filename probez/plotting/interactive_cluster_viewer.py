import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ipywidgets import interact, fixed
from plotting import generic_plotting_functions
from util import generic_functions


class InteractiveClusterViewer(object):
    """
    Module for use in IPython: generates interactive plots based on a list of spike_handling.

    """

    def __init__(self, spike_struct):
        self.good_clusters = []
        self.MUA_clusters = []
        self.spike_struct = spike_struct

    def plot_all_cluster_qualities(self, clusters, cmap, s=70, zorder=0):
        for c in clusters:
            self.plot_cluster_quality(c, cmap=cmap, s=s, zorder=zorder, edgecolor='None')

    def plot_cluster_quality_and_waveform(self, c):  # TODO: extract properly
        plt.close('all')
        plt.figure()
        plt.clf()
        plt.subplot(221)
        plt.cla()
        plt.plot(c.normalised_avg_waveforms)
        plt.text(2, 10, 'cluster: {}, channel: {}'.format(c.cluster_id, c.best_channel))

        plt.subplot(222)
        plt.cla()
        plt.imshow(c.normalised_avg_waveforms.T, aspect='auto', origin='lower')

        plt.subplot(223)
        plt.cla()
        x, y = generic_functions.rewrite_array_as_list_for_plotting(c.best_channel_waveforms.T) # reshape for plot speed
        plt.plot(x, y, color='k', linewidth=0.2, alpha=0.7)
        plt.plot(c.avg_waveforms[:, c.best_channel], color='r')

        plt.subplot(224)
        plt.cla()
        self.plot_all_cluster_qualities(self.good_clusters, cmap='Blues')
        self.plot_all_cluster_qualities(self.MUA_clusters, cmap='Reds')
        self.plot_cluster_quality(c, cmap='Greys', s=80, zorder=1, edgecolor='k')
        plt.ylim([-0.05, 1])
        plt.xlim([-0.05, 1])

    def interactive_quality_and_waveform_browsing(self):
        interact(self._browse_qualities_and_waveforms, idx=(0, len(self.good_clusters)-1, 1))

    def _browse_qualities_and_waveforms(self, idx):
        c = self.good_clusters[idx]
        self.plot_cluster_quality_and_waveform(c)

    @staticmethod
    def plot_cluster_quality(c, cmap, s, zorder, edgecolor):
        c.plot_quality(cmap, s, zorder, edgecolor)

    @staticmethod
    def _browse_waveforms(cluster):
        plt.cla()
        plt.plot(cluster.avg_waveforms)

    def interactive_waveform_browsing(self, clusters):
        fig = plt.figure()
        interact(self._browse_waveforms, cluster=clusters, fig=fixed(fig))

    def plot_heat_map(self, channel_spike_counts, boundaries=None):
        """

        :param np.array channel_spike_counts:
        :param dict boundaries: depths to draw on the figure, e.g. cortical layer boundaries
        :return fig: a figure for all the world to see

        """

        channel_indices = np.arange(self.spike_struct.n_chan)
        info = np.vstack([self.spike_struct.x_coords, self.spike_struct.y_coords, channel_spike_counts[::-1]])
        df = pd.DataFrame(info.T, index=channel_indices, columns=['x_pos', 'y_pos', 'spike_count'])
        plot_view_df = df.pivot('y_pos', 'x_pos', 'spike_count')
        fig = sns.heatmap(plot_view_df, annot=True, linewidths=1)

        if boundaries is not None:
            for bound in boundaries.values():
                channel_boundary = bound/10
                plt.hlines(channel_boundary, 0, 4, linewidth=2, linestyles='dashed')
        return fig
