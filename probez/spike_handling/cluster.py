import os
import numpy as np
from probez.spike_handling import cluster_exceptions
from probez.util import generic_functions
from cached_property import cached_property

import time

class Cluster(object):
    def __init__(self, spike_io, cluster_id, use_best_only=False):
        self.spike_io = spike_io
        self.cluster_id = cluster_id
        self.depth = self.spike_io.get_cluster_channel_from_avg_waveforms(self.cluster_id)
        self.use_best_only = use_best_only

        try:
            self.contamination_rate, self.isi_violations, self.isolation_distance = self.get_quality()
        except cluster_exceptions.QualityNotLoadedError as e:  # FIXME:
            pass

    def get_quality(self):
        """

        :return contamination_rate: an estimation of the number of noise events that are wrongly classified as being in
        the cluster, based on overlap in feature space
        :return isi_violations: an estimation of the number of false positives based on the number of biologically
        implausible events i.e. violations of the refractory period of the neuron
        :return isolation distance: the distance in feature space separating this cluster from all other spikes recorded
        """

        sp = self.spike_io
        cid = np.where(sp.unique_cluster_ids == self.cluster_id)[0][0]

        if sp.quality_path is None:
            raise cluster_exceptions.QualityNotLoadedError('cluster.spike_io does not have a quality path')

        contamination_rate = sp.contamination_rates[cid][0]
        isi_violations = sp.isi_violations[cid][0]
        isolation_distance = sp.isolation_distances[cid][0]

        return contamination_rate, isi_violations, isolation_distance

    @property
    def group(self):
        """

        :return group: the group label assigned by manual clustering
        (usually done in phy: https://github.com/kwikteam/phy-contrib)
        """
        return self.spike_io.groups[self.cluster_id]

    @property
    def best_channel_waveforms(self):
        """

        :return array best_channel_waveforms: every spike waveform for this cluster from the channel that is calculated
        as being closest to the source
        """

        t0 = time.time()

        if self.use_best_only:
            waveforms = self.waveforms
            print('Wave Form Time = {0}'.format(time.time()-t0))
            return waveforms
        else:
            best_waveforms = np.squeeze(self.waveforms[:, self.best_channel, :])
            print('Wave Form Time = {0}'.format(time.time() - t0))
            return best_waveforms

    @property
    def best_channel(self):
        """

        :return best_channel: the index of the channel that is calculated as being closest to the source
        """
        return self._get_channel_with_greatest_negative_deflection()

    def _get_channel_with_greatest_negative_deflection(self):

        n_spike_max = 10000
        t_spike = self.spike_times[:n_spike_max]
        v_spike = self.spike_io.traces[t_spike, :]
        t_spike_avg = np.mean(v_spike, axis=0)
        return np.argmin(t_spike_avg)

        # _, min_channel = np.unravel_index(self.normalised_avg_waveforms.argmin(), self.avg_waveforms.shape)
        # return min_channel

    @property
    def avg_waveforms(self):
        """

        :return avg_waveforms: the average waveform of the cluster's spikes, calculated for every channel and organised
         in a n_chan x t array where n_chan is the number of channels on the probe and t the number of samples in each
         waveform
        """
        return np.array(np.mean(self.waveforms, axis=2))

    @property
    def normalised_avg_waveforms(self):
        wfm = self.avg_waveforms
        med = np.median(wfm, axis=0)
        return wfm-med

    def _subtract_baseline_from_waveforms(self, waveforms):
        """

        :param waveforms:
        :return: median normalised waveforms
        """
        return waveforms - self.spike_io.get_traces_median(n_samples=100000)

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
        if subtract_baseline:
            waveforms = self._subtract_baseline_from_waveforms(waveforms)
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

        if self.use_best_only:
            best_channel_wform = self.spike_io.traces[:, self.best_channel]
            ww = [best_channel_wform[ts:tf] for ts, tf in zip(t_start, t_finish)]
            all_waveforms = np.zeros((n_waveform_samples, n_waveforms))
        else:
            ww = [self.spike_io.traces[ts:tf, :] for ts, tf in zip(t_start, t_finish)]
            all_waveforms = np.zeros((n_waveform_samples, self.spike_io.n_chan, n_waveforms))

        # dt_wave = t_finish - t_start
        # if np.any(dt_wave < n_waveform_samples):
        #     for i_wave in np.where(dt_wave < n_waveform_samples)[0]:
        #         t_pts_add = n_waveform_samples - dt_wave[i_wave]
        #         if t_start[i_wave] == 0:
        #             ww[i_wave] = np.array(list(np.zeros(t_pts_add)) + list(ww[i_wave]))
        #         else:
        #             ww[i_wave] = np.array(list(ww[i_wave]) + list(np.zeros(t_pts_add)))


        # if self.use_best_only:
        #     t0 = time.time()
        #     ww_final = np.array(ww).T
        #     print('Waveform Conversion = {0}'.format(time.time() - t0))
        #
        #     return ww_final
        # else:
        #     a = 1

        for i in range(len(spike_times)):
            # if not (np.issubdtype(type(spike_time), np.int32) or np.issubdtype(type(spike_time), np.uint64)):
            #     raise cluster_exceptions.SpikeTimeTypeError('got spike time:{}, type:{} expected '
            #                                                 'type: {}'.format(spike_time, type(spike_time), int))

            if self.use_best_only:
                all_waveforms[:len(ww[i]), i] = ww[i]
            else:
                all_waveforms[:len(ww[i]), :, i] = ww[i]

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

    def plot_waveforms_and_quality(self):
        fig = plt.figure()
        plt.clf()
        plt.subplot(221)
        plt.cla()
        plt.plot(self.normalised_avg_waveforms)
        plt.text(2, 10, 'cluster: {}, channel: {}'.format(self.cluster_id, self.best_channel))

        plt.subplot(222)
        plt.cla()
        plt.imshow(self.normalised_avg_waveforms.T, cmap='Reds_r', aspect='auto', origin='lower')
        plt.vlines(0, 0, 277, 'k')
        plt.vlines(60, 0, 277, 'k')
        plt.xlim([-50, 110])

        plt.subplot(223)
        plt.cla()
        x, y = generic_functions.rewrite_array_as_list_for_plotting(self.best_channel_waveforms.T) # reshape for plot speed
        plt.plot(x, y, color='k', linewidth=0.2, alpha=0.7)
        plt.plot(self.avg_waveforms[:, self.best_channel], color='r')

        plt.subplot(224)
        plt.cla()
        self.plot_quality(cmap='Greys', s=80, zorder=1, edgecolor='k')
        plt.ylim([-0.05, 1])
        plt.xlim([-0.05, 1])
        return fig

    def save_cluster_summary_figure(self, save_dir=None):
        if save_dir is None:
            save_dir = os.path.join(self.spike_io.root, 'cluster_figures')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        fig = self.plot_waveforms_and_quality()
        save_path = os.path.join(save_dir, str(self.cluster_id))
        fig.savefig(save_path, format='png')

    def pass_criteria(self, contamination_limit=1, isi_violation_limit=0.05, isolation_distance_limit=0.2):
        contamination_rate, isi_violations, isolation_distance = self.get_quality()
        if contamination_rate < contamination_limit and isi_violations < isi_violation_limit and isolation_distance > isolation_distance_limit:
            return True
        return False
