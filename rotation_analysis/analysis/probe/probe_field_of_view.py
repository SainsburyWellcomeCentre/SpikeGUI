import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from rotation_analysis.analysis.field_of_view import FieldOfView
from rotation_analysis.analysis.probe.probe_cell import ProbeCell
from rotation_analysis.analysis.probe.ctxt_manager_decorator import temp_setattr
import matplotlib.pyplot as plt

from margrie_libs.margrie_libs.signal_processing.list_utils import flatten as flatten_list
from rotation_analysis.analysis.probe.event_plotting_functions import normalise_axes


class ProbeFieldOfView(FieldOfView):
    """
    This class abstracts functionality that the user wants to apply to a group of cells/clusters.
    """
    def __init__(self, exp_id, cells, trials_to_remove, main_dir, depth, bonsai_io, igor_io, trigger_trace_io, sp,
                 cluster_ids=None):

        #super().__init__(cells, trials_to_remove, main_dir, depth)

        self.stats = pd.DataFrame()
        self.exp_id = exp_id
        self.trials_to_remove = trials_to_remove
        self.main_dir = main_dir
        self.depth = depth
        self.cluster_ids = cluster_ids
        self.bonsai_io = bonsai_io
        self.igor_io = igor_io
        self.trigger_trace_io = trigger_trace_io
        self.sp = sp
        self.cells = self.get_cluster
        self.detection_params = {}
        self.df_all_conditions = pd.DataFrame()

    def get_str(self, angle):
        return 'condition_{}'.format(angle)

    def get_cluster(self):

        if self.cluster_ids is None:
            cluster_ids = self.sp.good_cluster_ids
        else:
            cluster_ids = self.cluster_ids

        for cid in cluster_ids:
            c = ProbeCell(exp_id=self.exp_id, src_dir=self.main_dir, depth=0, cell_idx=cid,
                          extension='eps', use_bsl_2=False, spike_io=self.sp, bonsai_io=self.bonsai_io,
                          igor_io=self.igor_io, trigger_trace_io=self.trigger_trace_io)
            yield c

    @property
    def analysed_metrics(self):
        return ['freq']

    def save_all_clusters_trials(self, matching_criteria_dicts, save_path='/home/skeshav/all_trials.csv'):
        with temp_setattr(self, 'df_all_conditions', pd.DataFrame()):
            for i, c in enumerate(self.cells()):
                for matching_criteria_dict in matching_criteria_dicts:
                    with temp_setattr(c.block, 'current_condition', matching_criteria_dict):
                        self.df_all_conditions = self.df_all_conditions.append(c.block.to_df(), ignore_index=True)

            self.df_all_conditions.to_csv(save_path)

    def save_all_clusters_avgs(self, matching_criteria_dicts, save_path='/home/skeshav/all_trials_avgs.csv'):
        with temp_setattr(self, 'df_all_conditions', pd.DataFrame()):

            for i, c in enumerate(self.cells()):
                for matching_criteria_dict in matching_criteria_dicts:
                    with temp_setattr(c.block, 'current_condition', matching_criteria_dict):
                        self.df_all_conditions = self.df_all_conditions.append(c.block.get_averages_df(), ignore_index=True)

            self.df_all_conditions.to_csv(save_path)

    def plot_all_sub_stimuli_histograms(self, matching_dictionary, n_bins=30, save=False, format='png'):

        for c in tqdm(self.cells()):
            with temp_setattr(c.block, 'current_condition', matching_dictionary):
                fig = plt.figure(facecolor='w', figsize=(8, 3))
                plt.title('Cluster {}_Channel {}'.format(c.id, c.depth))
                c.block.plot_all_sub_stimuli(n_bins=n_bins, fig=fig)

                if save:
                    save_string = self.matching_dictionary_to_title_string(matching_dictionary)
                    save_dir = os.path.join(self.main_dir, 'histogram_plots', save_string)
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, 'cluster_{}_channel_{}.{}'.format(c.id, c.depth, format))
                    fig.savefig(save_path, format=format)
                else:
                    plt.show()

    def plot_overlay_histogram(self, matching_dictionaries, n_bins=20, save=False, format='png'):  # FIXME:
        for c in self.cells():

            fig = plt.figure(facecolor='w', figsize=(8, 3))

            for matching_dictionary in matching_dictionaries:
                with temp_setattr(c.block, 'current_condition', matching_dictionary):
                    fig = c.block.plot_all_sub_stimuli(n_bins=n_bins, fig=fig)

            plt.suptitle('Cluster {}_ Channel {}'.format(c.id, c.depth))
            histogram_axes = [fig.axes[i] for i in range(len(fig.axes)) if i % 2 == 1]
            normalise_axes(histogram_axes, space=3)
            #plt.show()

            if save:
                save_string = self.get_save_string(matching_dictionaries)
                save_dir = os.path.join(self.main_dir, 'histogram_plots', save_string)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, 'cluster_{}_channel_{}.{}'.format(c.id, c.depth, format))
                fig.savefig(save_path, format=format)
            else:
                plt.show()

    def get_save_string(self, matching_dictionaries):
        return '_'.join([self.matching_dictionary_to_title_string(matching_dictionary) for matching_dictionary in
                         matching_dictionaries])

    def do_stats(self, matching_dictionary):
        for c in self.cells():

            with temp_setattr(c.block, 'current_condition', matching_dictionary):
                c.set_main_dir(self.main_dir)
                c.block.save_stats()
                self.stats.append(c.block.stats_df)

        print(self.stats)

    @staticmethod
    def matching_dictionary_to_title_string(matching_criteria):
        all_sub_strings = []

        for k, v in matching_criteria.items():
            string = '{}_{}'.format(k, v)
            all_sub_strings.append(string)
        title_string = '_'.join(all_sub_strings)

        return title_string

    def generate_db(self):
        """
        Builds a database over all trials, conditions and cells, implemented in pandas. The idea is that this can be
        the user can query this database
        :return:
        """

        db = pd.DataFrame()
        for c in self.cells():
            db = db.append(c.block.generate_db(), ignore_index=True)
        return db

    def get_population_histogram_matrices(self, labels=('bsl_short', 'c_wise', 'c_c_wise'), n_bins=30):

        mats = [np.full(len(self.cluster_ids), n_bins, np.nan) for lbl in labels]
        population_histogram_matrices = {}
        for i, c in enumerate(self.cells()):
            for label, mat in zip(labels, mats):
                events_in_period = c. block.get_events_in_period(label, 'c_wise')
                hist = self.get_histogram(events_in_period, n_bins=n_bins)
                mat[i] = hist
            [population_histogram_matrices.setdefault(label, mat) for label, mat in zip(labels, mats)]
        return population_histogram_matrices

    @staticmethod
    def get_histogram(events, scale_factor=1, n_bins=10):
        hist_events = np.array(flatten_list(events))
        hist, bin_edges = np.histogram(hist_events, bins=n_bins)
        return hist / scale_factor

    def population_data(self, matching_dictionary, labels, time_match_label):
        all_groups = {}
        for label in labels:
            spikes_in_label = []
            for c in self.cells():
                with temp_setattr(c.block, 'current_condition', matching_dictionary):
                    spikes_in_label.extend(c.block.get_events_in_period(label, time_match_label))
            all_groups.setdefault(label, spikes_in_label)
        return all_groups
