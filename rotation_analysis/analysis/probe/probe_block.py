import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from rotation_analysis.analysis.probe.event_plotting_functions import plot_raster, plot_histogram
from rotation_analysis.analysis.probe.probe_block_plotter import ProbeBlockPlotter
from rotation_analysis.analysis.probe.probe_events_collection import ProbeEventsCollection
from rotation_analysis.analysis.probe.probe_trial import ProbeTrial


from rotation_analysis.analysis.block import Block


class ProbeBlock(Block):
    """
    A block of trials. All analysis at this level considers groupings of trials, set by the context manager and
    a dictionary of conditions see temp_setattr for further information.

    """

    def __init__(self, cluster, use_bsl_2, spike_struct, bonsai_io, igor_io, trigger_trace_io):
        self.bonsai_io = bonsai_io
        self.shuffles_results = pd.DataFrame()
        self.cell = cluster
        self.metrics_functions_dict = {
            'frequency': self.get_trials_freqs,
        }

        self.trials = []

        # assert bonsai_io.n_triggers == igor_io.n_triggers == trigger_trace_io.n_triggers

        self.trials = [ProbeTrial(self, i, use_bsl_2, spike_struct,
                                  bonsai_io, igor_io, trigger_trace_io) for i in range(trigger_trace_io.n_triggers)]

        self.plotter = ProbeBlockPlotter(self)
        self.spike_struct = spike_struct
        self.stats_df = pd.DataFrame()

    def get_metadata(self, recordings):
        return None

    def plot(self):
        self.plotter.plot_raster()

    def get_averages_df(self):
        df_dict = {}
        metrics_funcs = {
            'frequency': self.get_average_freq,
        }
        mapping = {
            'frequency': 'freq',
        }
        for c1, c2 in self.condition_pairs:
            for metric in self.analysed_metrics:

                df_dict['cell id'] = self.cell.id
                df_dict['condition'] = self.condition_str()
                df_dict['best channel'] = self.cell.depth

                metric_func = metrics_funcs[metric]
                metric = mapping[metric]
                if c1 == 'bsl_short':
                    col1 = '{}_{}_{}'.format(c1, c2, metric)
                else:
                    col1 = '{}_{}'.format(c1, metric)
                col2 = '{}_{}'.format(c2, metric)
                df_dict[col1] = metric_func(c1, c2)
                df_dict[col2] = metric_func(c2)

                col3 = 'delta_{}_{}'.format(c2, c1)
                df_dict[col3] = metric_func(c2) - metric_func(c1, c2)

        return pd.DataFrame(df_dict, index=[0])

    def get_events_in_period(self, period, constraining_period=None, relative=True):

        all_events = []
        for t in self.kept_trials:
            start, end = t.stimulus.get_ranges_by_type(period, constraining_period)[0]
            events = t.get_events_in_period(period, constraining_period)
            if relative:
                events = ProbeEventsCollection(events - start)
            all_events.append(events)
        return all_events

    def plot_all_sub_stimuli(self, n_bins=30, time_match_group='c_wise', fig=None):
        """

        :param n_bins: the binning size of the histogram
        :param time_match_group: the condition_pair to use to determine the duration of the group being compared
        (e.g. when comparing a baseline that can be of arbitrary length
        :return:
        """
        if fig is None:
            fig = plt.figure(facecolor='w', figsize=(8, 3))

        PLOT_SPACE = 5000

        labels = self.kept_trials[0].stimulus.histogram_plot_labels
        time_match_group = labels[-1]

        assert 'bsl' not in time_match_group

        for i, label in enumerate(labels):
            events_in_period = self.get_events_in_period(label, time_match_group)
            start, end = self.kept_trials[0].stimulus.get_ranges_by_type(label, time_match_group)[0]
            duration_in_samples = end - start

            ax = fig.add_subplot(2, len(labels), i + 1)
            plot_raster(events_in_period, label=label)
            plt.xlim([0 - PLOT_SPACE, duration_in_samples + PLOT_SPACE])

            ax = fig.add_subplot(2, len(labels), i + 1 + len(labels))
            plot_histogram(events_in_period, ax=ax, duration_in_samples=duration_in_samples, label=label, n_bins=n_bins)
            plt.xlim([0 - PLOT_SPACE, duration_in_samples + PLOT_SPACE])

        plt.tight_layout()
        return fig

    def plot_all_stimuli(self):
        for t in self.kept_trials:
            t.stimulus.plot()
        plt.show()

    def to_df(self):
        df_dict = {}
        for c1, c2 in self.condition_pairs:
            for metric in self.analysed_metrics:

                df_dict['cell_id'] = self.cell.id
                df_dict['condition'] = self.condition_str()
                df_dict['best channel'] = self.cell.depth

                colname = '{}_{}'.format(c1, metric)
                df_dict[colname] = self.metrics_functions_dict[metric](c1, c2)
                colname = '{}_{}'.format(c2, metric)
                df_dict[colname] = self.metrics_functions_dict[metric](c2)

                colname = 'delta_{}_{}'.format(c2, c1)
                delta_c2_c1 = self.metrics_functions_dict[metric](c2) - self.metrics_functions_dict[metric](c1, c2)  # TODO: add to metrics dictionary
                df_dict[colname] = delta_c2_c1

        df = pd.DataFrame(df_dict)  # TODO: order columns

        return df

    def generate_db(self):
        
        db = pd.DataFrame()
        df_dict = {}

        for metric in self.analysed_metrics:
            for t in self.trials:

                values_dict = {}
                events_list_dict = {}

                for c1, c2 in t.stimulus.condition_pairs:
                    val_c1 = t.get_frequency(c1, c2)
                    val_c2 = t.get_frequency(c2)

                    values_dict.setdefault(c1, val_c1)
                    values_dict.setdefault(c2, val_c2)

                    event_list_c1 = t.get_events_in_period(c1, c2, relative=True).events
                    event_list_c2 = t.get_events_in_period(c2, None, relative=True).events

                    events_list_dict.setdefault('{}_event_times'.format(c1), event_list_c1)
                    events_list_dict.setdefault('{}_event_times'.format(c2), event_list_c2)

                n_datapoints = len(values_dict.keys())

                df_dict['trial_id'] = [t.idx] * n_datapoints
                df_dict['cell_id'] = [self.cell.id] * n_datapoints
                df_dict['within_stimulus_condition'] = list(values_dict.keys())
                df_dict['metric'] = [metric] * n_datapoints
                df_dict['values'] = list(values_dict.values())
                df_dict['event_locs'] = list(events_list_dict.values())
                df_dict['experiment'] = [t.bonsai_data()['Condition']] * n_datapoints

                for k, v in t.bonsai_data().items():  # TODO: convert bonsai naming to snake case
                    ignore_keys = ['Experiment', 'Timestamp', 'Condition']

                    if k in ignore_keys:
                        continue

                    if not isinstance(v, str) and np.isnan(v):
                        continue

                    df_dict['between_stimuli_condition_metric'] = [k] * n_datapoints
                    df_dict['between_stimuli_condition'] = [v] * n_datapoints
                    df = pd.DataFrame(df_dict)
                    db = db.append(df, ignore_index=True)
        return db
