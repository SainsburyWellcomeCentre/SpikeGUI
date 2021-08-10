import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from margrie_libs.margrie_libs.signal_processing.filters import box_smooth, diff
from margrie_libs.margrie_libs.utils.utils import colors

from rotation_analysis.analysis.event_detection.event_plot import _plot_trace
from rotation_analysis.analysis.probe.ctxt_manager_decorator import temp_setattr
from rotation_analysis.analysis.pseudo_shuffle_analysis import plot_sd_shuffles


def _get_psth(raster, duration, n_bins_per_second=1, start_x=0):
    period = 1 / n_bins_per_second
    max_x = start_x + duration + period
    n_bins = (max_x - start_x) / n_bins_per_second
    bins = np.linspace(start_x, max_x, n_bins)
    psth_y, psth_x = np.histogram(raster, bins)
    psth_y = np.hstack((psth_y, np.array(psth_y[-1])))
    return psth_x, psth_y


class BlockPlotter(object):

    def __init__(self, block):
        """

        :param analysis.block.Block block:
        """
        self.block = block

        self.default_cmap = plt.get_cmap('inferno')

        self.imshow_params = {'interpolation': 'none',
                              'aspect': 'auto',
                              'origin': 'lower',
                              'cmap': self.default_cmap}

    @property
    def trials(self):
        return self.block.kept_traces  # Change here to change type of processing

    @property
    def detected_events(self):
        return self.block.kept_events_collections

    def __make_fig_path(self, fig_type):
        fig_name = '{}_{}.{}'.format(self.block, fig_type, self.block.cell.ext)
        return os.path.join(self.block.cell.dir, fig_name)

    def __make_main_graph(self, n_plots_above=2):
        # Two subplots, the axes array is 1-d
        n_trials = len(self.trials)
        f, axarr = plt.subplots(n_trials + n_plots_above, sharex=True, figsize=(8, 11))
        # f, axarr = plt.subplots(n_trials + n_plots_above, figsize=(8, 11))
        f.suptitle('{}'.format(self.block))
        f.text(0.51, 0.06, 'time (s)', ha='center', fontsize=12)
        f.text(0.06, 0.5, 'F/F0', va='center', rotation='vertical', fontsize=12)
        ax1 = axarr[0]
        ax1.tick_params(axis='y', labelsize=12)
        ax1.locator_params(axis='y', nbins=3)
        return f, axarr

    def __plot_detection_individual_trials(self, axarr, offset=2):
        """

        :param axarr:
        :param int offset: number of plots above
        :return:
        """
        trials_min, trials_max = self.block.get_trials_min_max()  # WARNING: gives min max of trial kind of processing
        max_range = trials_max - trials_min
        for i, trial in enumerate(self.block.kept_trials):
            trace = trial.processed_trace
            ax = axarr[i+offset]
            events_params = self.detected_events[i].get_events_point_params()
            events_pos, peaks_pos, half_rises, _ = events_params
            label = '{}_{}'.format(self.block, i)
            trace = signal.medfilt(trace.copy(), 3)
            _plot_trace(trace, label, events_pos, peaks_pos,
                        color='grey', x=trial.stimulus.x,
                        ax=ax)  # SK: trying to plot the medfiltered traces instead of raw
            ax.locator_params(axis='y', nbins=3)
            ax.set_xlim([0, max(trial.stimulus.x)])
            ax.set_ylim((0 - (.2 * max_range)), (trials_max + (.2 * max_range)))
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

    def plot_detection(self):
        """
        For each trace in the Cell, plots the trace
        and adds the events detection elements as a scatter plot on top
        Each figure will be saved
        """
        stimulus = self.block.trials[0].stimulus  # FIXME: should be fore each != stimulus
        n_plots_above = 2
        f, axarr = self.__make_main_graph(n_plots_above=n_plots_above)
        ax1 = axarr[0]
        ax2 = ax1.twinx()  # TODO: check if necessary to return ax2 previously
        avg = np.mean(np.array([t for t in self.trials]), 0)
        ax1.plot(stimulus.x, avg,
                 lw=2.0,
                 color=colors['dark_gray'],
                 label='Average F/F0')
        avg_range = avg.max() - avg.min()
        ax1.set_ylim((avg.min() - (.2 * avg_range)), (avg.max() + (.5 * avg_range)))

        # Command
        ax2.plot(stimulus.x, stimulus.cmd,
                 lw=1.0,
                 color=colors['orange'],
                 label='Position (degree)')
        ax2.plot(stimulus.x, stimulus.velocity,
                 lw=1.0,
                 color='blue',
                 label='Velocity(degree/s)')
        plt_min = stimulus.cmd.min()
        plt_max = stimulus.cmd.max()
        for period, color in zip(('c_wise', 'c_c_wise', 'spin', 'bsl_short'), ('r', 'g', 'b', 'orange')):
            if period == 'bsl_short':
                ranges = stimulus.get_ranges_by_type(period, 'spin')
            else:
                ranges = stimulus.get_ranges_by_type(period)
            for rng in ranges:
                start, end = np.array(rng, dtype=np.float64) * stimulus.sampling_interval
                plt.plot((start, start), (plt_min, plt_max), '-', color=color, label=period)
                plt.plot((end, end), (plt_min, plt_max), '-', color=color, label=period)
        # plt.legend()

        # ax2.set_ylabel('Command', fontsize=12, color='blue')  #SK: changing label size
        ax2.tick_params(axis='y', labelsize=12)
        ax2.locator_params(axis='y', nbins=3)
        # # Debugging
        # ax2.set_xlim((35, 50))
        # lines1, labels1 = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # f.legend(lines1+lines2, labels1+labels2, loc='upper center', bbox_to_anchor=(0.51, 0.94), ncol=3, fontsize=10)

        # Superimposed traces
        self.__plot_superimposed_traces(axarr)

        # Stacked traces
        self.__plot_detection_individual_trials(axarr, offset=n_plots_above)
        fig_path = self.__make_fig_path('detection')
        plt.savefig(fig_path)
        plt.close(f)

    def __plot_superimposed_traces(self, axarr):
        ax = axarr[1]
        for trial in self.block.kept_trials:
            trace = signal.medfilt(trial.processed_trace.copy(), 3)
            ax.plot(trial.stimulus.x, trace, color='grey')
            ax.locator_params(axis='y', nbins=3)
            ax.set_xlim([0, max(trial.stimulus.x)])
            # ax.set_ylim((0 - (.2 * max_range)), (trials_max + (.2 * max_range)))
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

    # REFACTOR: should be attributes of spiking_matrix object
    def plot_frequencies_histogram(self, spiking_matrix, degrees, degrees_type, matrix_unit_str):
        """

        :param np.array spiking_matrix: n_degrees x n_trials (trial = repeated angle measure)
        :param str matrix_unit_str: The description used to label the graph x axis
        :return:
        """
        f = plt.figure()

        spiking_means = spiking_matrix.mean(axis=1)
        spiking_stds = spiking_matrix.std(axis=1)

        bar_width = degrees[1] - degrees[0]
        plot_degrees = degrees[:len(spiking_means)]  # FIXME: should always be -1
        plt.bar(plot_degrees, spiking_means, width=bar_width, yerr=spiking_stds)
        plt.ylabel('frequency (Hz)')
        plt.xticks(plot_degrees)

        fig_path = self.__make_fig_path('{}_spiking_histogram'.format(degrees_type))
        plt.savefig(fig_path)
        plt.close(f)

    def plot_frequencies_heatmap(self, freqs, degrees, degrees_type):
        f = plt.figure()

        n_trials = len(self.detected_events) * self.block.trials[0].stimulus.n_repeats  # FIXME: incorrect
        plt.imshow(freqs.T, **self.imshow_params,
                   extent=(degrees.min(), degrees.max(),
                           0, n_trials))

        plt.title('{}'.format(self.block))
        plt.colorbar()

        fig_path = self.__make_fig_path('{}_spiking_heatmap'.format(degrees_type))
        plt.savefig(fig_path)
        plt.close(f)

    def plot_trials_as_image(self, median_filter_size=3):
        f = plt.figure()
        plt.title('{}'.format(self.block))
        data = np.array([signal.medfilt(t, median_filter_size) for t in self.trials], dtype=np.float64)
        plot_extent = (0, self.block.trials[0].stimulus.x[-1], 0, data.shape[0])  # FIXME: incorrect
        plt.imshow(data,
                   **self.imshow_params,
                   extent=plot_extent,
                   clim=(-0.25, 1))  # FIXME: Change me
        plt.colorbar()
        fig_path = self.__make_fig_path('deltaF_image')
        plt.savefig(fig_path)
        plt.close(f)

    def plot_raster(self, color='gray'):
        f = plt.figure()
        plt.title('{}'.format(self.block))

        x_axis = self.block.trials[0].stimulus.x  # FIXME: incorrect
        # loop backwards to have plots in same order as detection
        for i, trial_events in enumerate(reversed(self.detected_events)):
            raster_x = trial_events.peak_times
            raster_y = np.ones(raster_x.size) * i
            for x, y in zip(raster_x, raster_y):
                plt.plot((x, x), (y, y + 1), '-', c=color)
        plt.xlim(x_axis[0], x_axis[-1])
        plt.ylim(0, len(self.detected_events))  # TODO: check if +1
        f.axes[0].get_yaxis().set_visible(False)

        plt.xlabel('time')
        plt.ylabel('trial')

        fig_path = self.__make_fig_path('raster')
        plt.savefig(fig_path)
        plt.close(f)

    def plot_psth(self, plot=False):
        if plot:
            f = plt.figure()
        stimulus = self.block.trials[0].stimulus  # FIXME: incorrect

        spin_start = stimulus.get_spin_start() * stimulus.sampling_interval  # value in time not samples
        spin_end = stimulus.get_spin_end() * stimulus.sampling_interval

        bsl_duration, bsl_psth_x, bsl_psth_y = self.__plot_psth(plot, 'bsl_short', 'spin', 'blue', start_x=0)  # FIXME: this is not necessarily 0
        spin_duration, spin_psth_x, spin_psth_y = self.__plot_psth(plot, 'spin', None, 'green', start_x=spin_start)
        bsl2_duration, bsl2_psth_x, bsl2_psth_y = self.__plot_psth(plot, 'bsl2', 'spin', 'red', start_x=spin_end)

        return (bsl_psth_x, bsl_psth_y), (spin_psth_x, spin_psth_y), (bsl2_psth_x, bsl2_psth_y)

    def __plot_psth(self, plot, range_name, constraining_range_name, color, start_x):
        stimulus = self.block.trials[0].stimulus  # FIXME: incorrect
        rngs = stimulus.get_ranges_by_type(range_name, constraining_range_name)
        period_duration = stimulus.get_ranges_duration(rngs)
        period_events = self.block.get_events_peaks(range_name, constraining_range_name)  # TODO: rename get event times
        period_psth_x, period_psth_y = _get_psth(period_events, period_duration, start_x=start_x)
        if plot:
            plt.step(period_psth_x, period_psth_y, where='post', color=color)
        return period_duration, period_psth_x, period_psth_y

    def plot_single_trial_detection(self, angle, trial_id):  # FIXME: trial plotter
        with temp_setattr(self.block, 'current_condition', {'angle': angle}):
            trial = self.block.trials[trial_id]
            trace = trial.processed_trace
            params = self.block.detection_params[angle]
            delta_wave = diff(box_smooth(trace, params.n_pnts_peak),
                              box_smooth(trace, params.n_pnts_bsl),
                              params.n_pnts_rise_t)
            plt.plot(trial.stimulus.x[:len(delta_wave)], delta_wave, linewidth=0.3)  # FIXME: find why different length # REFACTOR: extract

            trace = signal.medfilt(trace.copy(), params.median_kernel_size)
            events_params = trial.events.get_events_point_params()  # WARNING: not analysed_events because for GUI
            events_pos, peaks_pos, half_rises, _ = events_params
            _plot_trace(trace, 'trial_{}'.format(trial_id), events_pos, peaks_pos, color='grey', x=trial.stimulus.x, y_shift=0)

    def plot_sd_shuffles(self, freqs, levels, levels_name, n_shuffles=1000):
        f = plt.figure()
        _, real_sd, right_tail, _, p_val = plot_sd_shuffles(freqs, levels, n_shuffles=n_shuffles, show=False)

        fig_path = self.__make_fig_path('{}_shuffles'.format(levels_name))
        plt.savefig(fig_path)
        plt.close(f)

        return right_tail, real_sd, p_val
