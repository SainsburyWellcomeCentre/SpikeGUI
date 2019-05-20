import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from cached_property import cached_property
from probez.spike_handling import waveforms, cluster_exceptions, cluster
from probez.util import generic_functions
from probez.sorting_quality import load_quality_measures
from bisect import bisect_left

# from analysis.probe.config import FS_probe

class SpikeIo(object):

    """
    class for loading and manipulating KiloSort output files and processed data

    example usage:
    >>> sp = SpikeIo(root, traces_path, n_chan)  # create the SpikeStruct and load all variables
    >>> good_cluster_ids = sp.get_clusters_in_group('good')  # get all the clusters in a specific group
    >>> depth_ordered_good_clusters =  sp.sort_cluster_ids_by_depth(good_cluster_ids, descend=True)  # order them


    """

    def __init__(self, root, traces_path, n_chan, quality_path=None):

        self.n_chan = n_chan
        self.root = root
        self.traces_path = traces_path
        self.quality_path = quality_path

        load = self.load_data

        self.all_spike_times = np.array(generic_functions.flatten_list(load('spike_times')))
        self.spike_clusters = load('spike_clusters')
        self.unique_cluster_ids = np.unique(self.spike_clusters)

        channel_positions = load('channel_positions')
        self.x_coords = channel_positions[:, 0]
        self.y_coords = channel_positions[:, 1]

        self.groups = self.read_groups()
        if self.read_groups() is not None:
            self.good_cluster_ids = self.get_clusters_in_group('good')
            self.MUA_cluster_ids = self.get_clusters_in_group('mua')
            self.noise_cluster_ids = self.get_clusters_in_group('noise')
            self.unsorted_cluster_ids = self.get_clusters_in_group('unsorted')

        if quality_path is not None:
            self.groups, self.contamination_rates, self.isi_violations, \
                         self.isolation_distances = load_quality_measures.load_from_matlab(quality_path)

    def load_data(self, name):
        path = os.path.join(self.root, name) + '.npy'
        if not os.path.isfile(path):
            raise cluster_exceptions.SpikeStructLoadDataError('file: {} does not exist'.format(path))
        return np.load(path)

    @cached_property
    def traces(self):
        """
        Traces_path should be the path to the raw or processed binary data. This is used primarily for extracting
        waveforms so it helps if the data are high pass filtered or processed in some way, but it shouldn't be essential
        :param limit: restrict the size of the data used to improve speed
        :return shaped_data:
        """
        if not os.path.isfile(self.traces_path):
            raise cluster_exceptions.SpikeStructLoadDataError('file: {} does not exist'.format(self.traces_path))
        data = np.memmap(self.traces_path, dtype=np.int16)
        if data.shape[0] % self.n_chan != 0:
            raise cluster_exceptions.IncorrectNchanTracesStructError(data.shape[0], self.n_chan)
        print('reshaping data... this may take a while')
        shape = (int(data.shape[0] / self.n_chan), self.n_chan)
        shaped_data = np.memmap(self.traces_path, shape=shape, dtype=np.int16)
        print('data reshaping completed! :)')
        return shaped_data

    def get_traces_median(self, n_samples=1500000):
        return np.median(self.traces[0:n_samples, :], axis=0)

    def read_groups(self):
        """
        manual sorting output (from e.g. phy) is stored as cluster_groups.csv
        :return dictionary manually_labelled_cluster_groups: a dictionary of group_ids for every cluster
        """
        path = os.path.join(self.root, 'cluster_groups.csv')
        if not os.path.isfile(path):
            print('no cluster groups file')
            return None
        with open(path, 'rt') as f:
            reader = csv.reader(f)
            manually_labelled_cluster_groups = {}
            for i, row in enumerate(reader):
                if i != 0:  # skip first row (headers)
                    cluster_id = row[0].split()[0]
                    cluster_group = row[0].split()[1]
                    manually_labelled_cluster_groups.setdefault(int(cluster_id), cluster_group)
        return manually_labelled_cluster_groups

    def get_clusters_in_group(self, group):
        """
        :param group: the group that the user wants clusters from
        :return clusters: a list of all cluster_ids classified to a user defined group:
        """
        if self.groups is None:
            return
        return [key for key in self.groups.keys() if self.groups[key] == group]

    def get_cluster_group_mask(self, group):  # ?
        clusters = self.get_clusters_in_group(group)
        return [cid in clusters for cid in self.spike_clusters]

    def get_spike_times_in_interval(self, start_t, end_t, spike_times=None):
        """

        :param start_t: start point (n_samples)
        :param end_t: end point (n_samples)
        :param spike_times: the set of spikes from which to get subset from
        :return spike_times: all spike times within a user specified interval
        """
        if spike_times is None:
            spike_times = self.all_spike_times
        start_idx = bisect_left(spike_times, start_t)
        end_idx = bisect_left(spike_times, end_t)
        return spike_times[start_idx:end_idx]

    def get_spike_cluster_ids_in_interval(self, start_t, end_t, spike_times=None, spike_clusters=None):
        """

        :param start_t: start point (n_samples)
        :param end_t: end point (n_samples)
        :param spike_clusters: the set of spikes from which to get subset from
        :param spike_times:
        :return spike_clusters: all cluster ids within a user specified interval
        """
        if spike_times is None:
            spike_times = self.all_spike_times
        if spike_clusters is None:
            spike_clusters = self.spike_clusters
        start_idx = bisect_left(spike_times, start_t)
        end_idx = bisect_left(spike_times, end_t)
        return spike_clusters[start_idx:end_idx]

    def get_spike_times_in_cluster(self, cluster_id):
        """

        :param int cluster_id:
        :return spike times: an array of all spike times within a user specified cluster
        """

        return self.all_spike_times[self.spikes_in_cluster_mask(cluster_id)]

    def spikes_in_cluster_mask(self, cluster_id):
        """
        a boolean mask of all indices of spikes that belong to a group of user defined clusters

        :param int cluster_id:
        :return:
        """

        return self.spike_clusters == cluster_id

    def cluster_spike_times_in_interval(self, cluster_id, start, end, spike_times=None, spike_clusters=None):
        if spike_times is None:
            spike_times = self.all_spike_times
            spike_clusters = self.spike_clusters

        spike_times_in_interval = self.get_spike_times_in_interval(start, end, spike_times)
        cluster_ids_in_interval = self.get_spike_cluster_ids_in_interval(start, end, spike_times, spike_clusters)
        spikes_in_cluster_and_interval = spike_times_in_interval[cluster_ids_in_interval == cluster_id]
        return spikes_in_cluster_and_interval

    # def cluster_spike_times_in_interval_time(self, cluster_id, start, end, spike_times=None, spike_clusters=None):
    #     return self.cluster_spike_times_in_interval(cluster_id, start, end, spike_times, spike_clusters) / FS_probe

    def sort_cluster_ids_by_depth(self, cluster_ids=None, cluster_depths=None, descend=True):
        """
        :param cluster_ids:
        :param cluster_depths:
        :param descend: if descending order is required then set this to true
        :return:
        """
        if cluster_ids is None:
            cluster_ids = self.good_cluster_ids
        if cluster_depths is None:
            cluster_depths = self.get_cluster_depths

        sorted_cluster_ids, cluster_depths = generic_functions.sort_by(cluster_ids, cluster_depths, descend=descend)
        return sorted_cluster_ids, cluster_depths

    def plot_probe_as_image(self, start, end):
        plt.imshow(self.traces[start:end, :].T, aspect='auto', origin=0)

    def get_channel_spike_counts(self, cluster_ids, start, end):
        """
        clusters are allocated to their 'best channel' using the average waveforms of their spikes
        the number of spikes in each channel is then computed and returned

        :param cluster_ids:
        :param start: the start of the window in which you want to count spikes
        :param end: the end of the window in which you want to count spikes
        :return spike_counts: the number of spikes in the window for each cluster in each channel on the probe
        """
        spike_counts = np.zeros(self.n_chan)

        for cluster_id in cluster_ids:
            depth = self.get_cluster_channel_from_avg_waveforms(cluster_id)
            cluster_spikes = self.cluster_spike_times_in_interval(cluster_id, start, end)
            spike_counts[depth] += len(cluster_spikes)

        return spike_counts

    def clusters_in_depth_range(self, lower, upper, cluster_ids=None):
        if cluster_ids is None:
            cluster_ids = self.good_cluster_ids

        cluster_ids = np.array(cluster_ids)
        clusters = [cluster.Cluster(cid) for cid in cluster_ids]
        cluster_channels = np.array([c.best_channel for c in clusters])

        return self.good_cluster_ids[np.logical_and(cluster_channels > lower, cluster_channels < upper)]

    def get_clusters_in_depth_range(self, cluster_ids, lower, upper):
        """
        :param cluster_ids:
        :param lower: the lower layer (channel) bound
        :param upper: the upper layer (channel) bound
        :return cluster_ids: a list of clusters within the two bounds
        """
        cluster_depths = self.get_cluster_depths(cluster_ids)
        clusters_in_depth_range = [idx for idx, depth in zip(cluster_ids, cluster_depths) if lower < depth < upper]
        return clusters_in_depth_range

    def get_cluster_depths(self, cluster_ids):
        cluster_depths = []
        for cluster_id in cluster_ids:
            cluster_depth = self.get_cluster_channel_from_avg_waveforms([cluster_id])
            cluster_depths.append(cluster_depth)
        return cluster_depths

    def get_cluster_channel_from_avg_waveforms(self, cluster_id, n_spikes=100):
        """
        :param cluster_id:
        :param n_spikes: number of spikes used to form average waveform
        :return:
        """
        spike_times = self.get_spike_times_in_cluster(cluster_id)
        cluster_channel = waveforms.get_channel_of_max_amplitude_avg_waveform(spike_times[:n_spikes], self.traces,
                                                                              self.n_chan)

        return cluster_channel