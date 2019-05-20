import os
import warnings

import numpy as np
import pandas as pd

from margrie_libs.margrie_libs.stats.stats import paired_t_test, wilcoxon
from scipy import signal

from rotation_analysis.analysis.block_plotter import BlockPlotter
from rotation_analysis.analysis.event_detection.detection_params import DetectionParams
from rotation_analysis.analysis.event_detection.events_collection import EventsCollection
from rotation_analysis.analysis.probe.ctxt_manager_decorator import temp_setattr

from rotation_analysis.analysis.trial import Trial
from rotation_analysis.resample.resampling import count_discontinuities_in_the_matrix, fix_timetable


# TODO:  2- Add event counts (instead of only event rates)
# TODO:  3- Add end of post-stim bit to baseline for Poisson comparison (to ensure duration is equal):
# TODO:       define duration from the end (walk backwards) to add to baseline as an argument
# TODO:  4- Add area under event:
# TODO:       Need to define with Troy what event end we choose


class BlockException(Exception):
    pass


class Block(object):
    """
    A block of trials and the functions that operate on them. All analysis at this level considers groupings of trials,
    set by the context manager and a dictionary of conditions see temp_setattr for further information.

    Analysis is generally done on pairs of conditions and all possible combinations are considered. If at least one of
    the conditions in the pair is arbitrary in duration the comparison group is used to match the durations.

    """

    def __init__(self, cell, recordings, use_bsl_2):
        self.conditions = []
        self.current_condition = {'keep': True}  # dictionary of conditions, e.g. {'keep': True, 'angle': 45, 'is_cw_first': True}  # TODO: check if keep actually does anything anymore
        self.shuffles_results = pd.DataFrame()
        self.cell = cell

        self.detection_params = DetectionParams()

        self.metrics_functions_dict = {
            'frequency': self.get_trials_freqs,
            'amplitude': self.get_trials_weighted_amplitudes,
            'delta_f': self.get_trials_fluo
        }

        self.recording_conditions = list(recordings.keys())  # Needs to be list to be pickleable
        self.trials = []
        self.detection_params = {}

        for condition, recordings_in_condition in recordings.items():
            self.get_detection_params(condition)
            trials_in_conditions = [Trial(self, i, rec, condition, use_bsl_2) for i, rec in enumerate(recordings_in_condition)]
            self.trials.extend(trials_in_conditions)

        self.plotter = BlockPlotter(self)

        self.stats_df = pd.DataFrame()

    def get_detection_params(self, condition):
        detection_params_csv_path = self.get_detection_params_path(condition)
        if detection_params_csv_path:
            self.detection_params[condition] = DetectionParams(config_file_path=detection_params_csv_path)
        else:
            self.detection_params[condition] = DetectionParams()

    def set_conditions(self, conditions):
        self.conditions = conditions

    def get_detection_params_path(self, angle):
        with temp_setattr(self, 'current_condition', {'keep': True, 'angle': angle}):
            expected_path = os.path.join(self.cell.dir, '{}_detection_params.csv'.format(self))
            if os.path.exists(expected_path):
                return expected_path
            else:
                warnings.warn('File {} not found to reload detection parameters, creating new file'
                              .format(expected_path))
                return None

    def __len__(self):
        """
        n columns

        :return:
        """
        return len(self.kept_trials)

    def __str__(self):
        return '{}_{}'.format(self.cell, self.formatted_current_condition)  # TODO: self.stimulus.name

    @property
    def formatted_current_condition(self):
        return '_'.join(['{}_{}'.format(k, v) for k, v in self.current_condition.items()])

    @property
    def kept_trials(self):
        return [t for t in self.trials if t.matches_attributes(self.current_condition)]

    @property
    def kept_traces(self):
        return [t.processed_trace for t in self.kept_trials]

    @property
    def kept_raw_traces(self):
        return [t.raw_trace for t in self.kept_trials]

    @property
    def kept_events_collections(self):
        return [t.events for t in self.kept_trials]

    @property
    def kept_noises(self):
        return [t.noises for t in self.kept_trials]

    @property
    def condition_pairs(self):
        return self.kept_trials[0].stimulus.condition_pairs

    @property
    def analysed_metrics(self):
        return self.cell.analysed_metrics

    def remove_trials(self, bad_trials_list):  # TODO: see if can be dynamic instead
        """
        Given a list of trials indices, remove the trials in the list
        """
        if bad_trials_list is not None:
            for i, t in enumerate(self.trials):  # WARNING: all trials, should be by angle
                if i in bad_trials_list:
                    t.keep = False

    def get_events_peaks(self, period_name, constraining_period_name=None):
        events_collections = [t.get_events_in_period(period_name, constraining_period_name) for t in self.kept_trials]
        events = EventsCollection.from_concatenation_of_events_collections(events_collections)
        return events.peak_times

    def get_freqs_from_timetable(self, levels_var_name):  # TODO: rename
        """

        :param str levels_var_name:
        :return:
        """

        # FIXME: this cmd and timetable should be computed inside the loop per trial to match that trial
        stim = self.kept_trials[0].stimulus
        levels, timetable, cmd, cmd_x = stim._get_timetable(levels_var_name)

        gaps = count_discontinuities_in_the_matrix(timetable)
        if gaps:
            median_diff = np.median(np.abs(np.diff(timetable, axis=0)))
            median_threshold = 2
            timetable = fix_timetable(cmd, cmd_x, timetable,
                                      t_min=stim.get_full_cycles_spin_range()[0][0] * stim.sampling_interval + 1)  # FIXME: Hacky
            warnings.warn('Gaps found in the {} timetable of {}'.format(levels_var_name, self))

        n_degrees = timetable.shape[0] - 1  # -1 because n-1 intervals (Can be velocity or acceleration too)
        n_trials = len(self.kept_trials) * stim.n_repeats
        freqs = np.full((n_degrees, n_trials), np.nan, dtype=np.float64)
        for i, trial in enumerate(self.kept_trials):
            events_collection = trial.events
            for bin_start in range(n_degrees):
                for k in range(trial.stimulus.n_repeats):
                    t1, t2 = timetable[bin_start: bin_start+2, k]
                    duration = abs(t2 - t1)
                    if gaps and 0 < bin_start < n_degrees - 1 and duration > median_threshold*median_diff:
                        print('Trial {} skipping bin from {} to {}, repeat {}, median delta_t = {}'
                              .format(i, t1, t2, k, median_diff))
                        continue
                    current_events = events_collection.in_unordered_time_range(t1, t2)
                    n_events = len(current_events.peak_times)
                    freqs[bin_start, (i*trial.stimulus.n_repeats+k)] = n_events / duration
        if gaps:
            freqs = freqs[~np.isnan(freqs)].reshape((n_degrees - 1, n_trials))
        self.plotter.plot_frequencies_heatmap(freqs, levels, levels_var_name)
        self.plotter.plot_frequencies_histogram(freqs, levels, levels_var_name, '')  # TODO: add unit
        n_shuffles = 1000
        right_tail, real_sd, p_val = self.plotter.plot_sd_shuffles(freqs, levels, levels_var_name, n_shuffles)
        self.shuffles_results = pd.concat([self.shuffles_results,
                                           pd.DataFrame(
            {
                '{} n shuffles'.format(levels_var_name): n_shuffles,
                '{} right tail'.format(levels_var_name): right_tail,
                '{} real sd'.format(levels_var_name): real_sd,
                '{} p value'.format(levels_var_name): p_val
            }, index=[0])], axis=1)
        return freqs, levels, cmd

    def get_trials_min_max(self):
        """
        Get the min([min(t) for t in self.analysed_trials])
        And the max([max(t) for t in self.analysed_trials])
        :return:
        """
        trials_min = min([min(t) for t in self.kept_traces])
        trials_max = max([max(t) for t in self.kept_traces])
        return trials_min, trials_max

    def get_trials_freqs(self, period_name, constraining_period_name=None):
        """
        Trials based

        :param str period_name:
        :param str constraining_period_name:
        :return:
        """
        return np.array([t.get_frequency(period_name, constraining_period_name) for t in self.kept_trials])

    def get_average_freq(self, period_name, constraining_period_name=None):
        """
        Cell based

        :param str period_name:
        :param str constraining_period_name:
        :return:
        """
        freqs = self.get_trials_freqs(period_name, constraining_period_name)
        return freqs.mean()

    def get_trials_weighted_amplitudes(self, period_name, constraining_period_name=None):
        """
        Trials based

        :param str period_name:
        :param str constraining_period_name:
        :return:
        """
        weighted_amplitudes = [t.get_weighted_amplitude(period_name, constraining_period_name)
                               for t in self.kept_trials]
        return weighted_amplitudes

    def get_weighted_average_ampl(self, period_name, constraining_period_name=None):
        """
        Cell based (all averaged at once)

        :param str period_name:
        :param str constraining_period_name:
        :return:
        """
        ampls = np.array(self.get_trials_weighted_amplitudes(period_name, constraining_period_name))
        return ampls.mean()

    def get_trials_fluo(self, period_name, constraining_period_name=None):
        trials_average_fluo = [t.extract_period(period_name, constraining_period_name).mean() for t in self.kept_trials]
        return trials_average_fluo

    def get_average_fluo(self, period_name, constraining_period_name=None):
        fluos = np.array(self.get_trials_fluo(period_name, constraining_period_name))
        return fluos.mean()

    def get_psth(self):
        return self.plotter.plot_psth(plot=False)

    def plot(self):
        self.plotter.plot_detection()
        self.plotter.plot_trials_as_image()
        self.plotter.plot_raster()
        self.plotter.plot_psth(True)

    def get_events_integral(self, angle, trial_id):
        """
        All trials but one angle (because for detection)

        :param trial_id:
        :return:
        """
        with temp_setattr(self, 'current_condition', {'angle': angle}):
            integrals = []
            trial = self.trials[trial_id]
            trace = trial.processed_trace
            filtered_trace = signal.medfilt(trace.copy(), 3)
            for event in trial.events:
                integrals.append(event.get_integral(filtered_trace))
            if integrals:
                if __debug__:
                    print(integrals)
                return np.array(integrals).mean()
            else:
                return 0

    def analyse(self, angle, processed=True):  # WARNING: all trials # TODO: rename?  + DOUBLE WARNING should be only one angle
        """
        All trials but one angle

        :param processed:
        :return:
        """
        with temp_setattr(self, 'current_condition', {'angle': angle}):
            self.__reset_detection()
            for trial in self.kept_trials:
                trial.detect(self.detection_params[angle], processed)  # detect all whether kept or not

    def save_detection(self, condition, processing_type):
        """
        Save detected events parameters and detection parameters to 2 csv files
        """

        events_params_df = pd.DataFrame()
        for trial in self.kept_trials:
            events_params_df = pd.concat([events_params_df, trial.events.to_df(trial.idx)])

        dest_name = '{}_{}_events.csv'.format(self, processing_type)
        dest_path = os.path.join(self.cell.dir, dest_name)
        events_params_df.to_csv(dest_path)

        detection_params = self.detection_params[condition].to_df()
        params_file_name = '{}_detection_params.csv'.format(self)
        params_file_path = os.path.join(self.cell.dir, params_file_name)
        detection_params.to_csv(params_file_path)

    def __reset_detection(self):
        """
        All trials but one angle

        :return:
        """
        for trial in self.kept_trials:
            trial.reset_detection()

    def condition_str(self):
        all_conditions = []
        for k, v in self.current_condition.items():
            condition_str = '{}: {}'.format(k, v)
            all_conditions.append(condition_str)
        return ', '.join(all_conditions)

    def to_df(self):
        df_dict = {}
        for c1, c2 in self.condition_pairs:
            for metric in self.analysed_metrics:
                colname = '{}_{}'.format(c1, metric)
                df_dict[colname] = self.metrics_functions_dict[metric](c1, c2)
                colname = '{}_{}'.format(c2, metric)
                df_dict[colname] = self.metrics_functions_dict[metric](c2)
        df = pd.DataFrame(df_dict)  # TODO: order columns
        return df

    def save_stats(self, parametric=False):
        """
        Analyse and compare frequency and amplitude of events for each trial between condition 1 and condition 2
        e.g. baseline vs spinning

        :return:
        """
        if parametric:
            stats_func_name = 'paired_t_test'
            stats_func = paired_t_test
        else:
            stats_func_name = 'wilcoxon_test'
            stats_func = wilcoxon

        df = self.to_df()

        csv_filename = '{}_trials.csv'.format(self)
        csv_file_path = os.path.join(self.cell.main_dir, csv_filename)
        df.to_csv(csv_file_path)

        stats = {}
        for c1, c2 in self.condition_pairs:
            for metric in self.analysed_metrics:
                c1_metric = df['{}_{}'.format(c1, metric)]
                c2_metric = df['{}_{}'.format(c2, metric)]
                p_val = stats_func(c1_metric, c2_metric)
                print('{} {} vs {} {} {}: p_value={}'
                      .format(self, c1, c2, metric, stats_func_name, p_val))
                stats['{}_vs_{}_{}'.format(c1, c2, metric)] = p_val

        self.stats_df = pd.DataFrame(stats, index=[0])

    def get_averages_df(self):
        out_dict = {}
        metrics_funcs = {
            'frequency': self.get_average_freq,
            'amplitude': self.get_weighted_average_ampl,
            'delta_f': self.get_average_fluo
        }
        mapping = {  # FIXME: e.g. delta_f > fluo shouldn't be needed
            'frequency': 'freq',
            'amplitude': 'weighted_ampl',
            'delta_f': 'fluo'
        }
        for c1, c2 in self.condition_pairs:
            for metric in self.analysed_metrics:
                metric_func = metrics_funcs[metric]
                metric = mapping[metric]
                if c1 == 'bsl_short':
                    col1 = '{}_{}_{}'.format(c1, c2, metric)
                else:
                    col1 = '{}_{}'.format(c1, metric)
                col2 = '{}_{}'.format(c2, metric)
                out_dict[col1] = metric_func(c1, c2)
                out_dict[col2] = metric_func(c2)
        return pd.DataFrame(out_dict, index=[0])

    def get_results_df(self):
        return pd.concat([self.get_averages_df(), self.stats_df, self.shuffles_results], axis=1)
