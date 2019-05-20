import os
import numpy as np
from probez.spike_handling import cluster_exceptions
from probez.util import generic_functions
from cached_property import cached_property

import time

class ClusterRead(object):
    def __init__(self, spike_io, cluster_id):
        self.spike_io = spike_io
        self.cluster_id = cluster_id

    @property
    def channel_waveforms(self):
        """

        :return array best_channel_waveforms: every spike waveform for this cluster from the channel that is calculated
        as being closest to the source
        """

        return self.waveforms

    @property
    def best_channel(self):
        """

        :return best_channel: the index of the channel that is calculated as being closest to the source
        """

        n_spike_max = 10000

        t_spike = self.spike_times[:n_spike_max]
        v_spike = self.spike_io.traces[t_spike, :]
        t_spike_avg = np.mean(v_spike, axis=0)

        return np.argmin(t_spike_avg)

    @property
    def avg_waveforms(self):
        """

        :return avg_waveforms: the average waveform of the cluster's spikes, calculated for every channel and organised
         in a n_chan x t array where n_chan is the number of channels on the probe and t the number of samples in each
         waveform
        """
        return np.array(np.mean(self.waveforms, axis=2))

    @cached_property
    def waveforms(self, subtract_baseline=False, limit=None):
        """

        :param subtract_baseline: if the data are not normalised, some median correction needs to be applied to correct
        for the different baselines of the channels
        :return array waveforms: all spike waveforms across all channels and n_chan x t x n_waveforms
        """
        waveforms = self._get_channel_spike_waveforms(n_samples_before_peak=40,
                                                      n_samples_after_peak=60,
                                                      limit=limit)
        return waveforms

    def _get_channel_spike_waveforms(self, n_samples_before_peak, n_samples_after_peak, limit=None):
        """

        :param int n_samples_before_peak:
        :param int n_samples_after_peak:
        :return np.array all_waveforms:
        """

        n_pts_expt = np.size(self.spike_io.traces, axis=0)
        spike_times = self.spike_times[:limit]
        n_waveforms = len(spike_times)
        n_waveform_samples = n_samples_before_peak + n_samples_after_peak

        # waveform start/finish time point indices
        t_start = np.array([max(0, int(x - n_samples_before_peak)) for x in spike_times])
        t_finish = np.array([min(n_pts_expt, int(x + n_samples_after_peak)) for x in spike_times])

        best_channel_wform = self.spike_io.traces[:, self.best_channel]
        ww = [best_channel_wform[ts:tf] for ts, tf in zip(t_start, t_finish)]
        all_waveforms = np.zeros((n_waveform_samples, n_waveforms))

        for i in range(len(spike_times)):
            all_waveforms[:len(ww[i]), i] = ww[i]

        return all_waveforms

    def get_spike_times_in_interval(self, start, end):

        """

        :param start:
        :param end:
        :return spike_times: the time (n_samples) of all spikes that occur between the given start and end samples
        """
        spike_mask = np.logical_and(self.spike_times > start, self.spike_times < end)
        spike_times = self.spike_times[spike_mask]
        return spike_times

    @property
    def spike_times(self):
        """

        :return spike_times: the time of all spikes that belong to this cluster
        """
        return self.spike_io.get_spike_times_in_cluster(self.cluster_id)

