import numpy as np
from spike_handling.cluster import Cluster

import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
pandas2ri.activate()
rstats = importr('stats')


class ClusterStats(object):
    def __init__(self, cid, stimuli_struct, rates=None, labels=None, stimuli=None):
        self.stimuli_struct = stimuli_struct

        if stimuli is None:
            self.stimuli = self.stimuli_struct.stimuli
        else:
            self.stimuli = stimuli
        if rates is not None:
            self.rates = rates
            self.labels = labels
        else:
            self.rates, self.labels = self.stimuli_struct.get_rates_all_sub_stimuli(cid, self.stimuli)

        self.data_df = self.initialise_condition_df()
        self.stats_df = self.initialise_stats_df()
        self.add_rates_to_df(self.rates)
        self._compare_all_categories()
        self.stats_df = self.filter_out_irrelevant_comparisons()

    def initialise_condition_df(self):
        data = ['id']
        data.extend(self.labels)
        data = np.array([data])
        df = pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])
        return df

    @staticmethod
    def initialise_stats_df():
        data = np.array([['id', 'p_value_wilcoxon', 'label']])
        df = pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])
        return df

    def add_rates_to_df(self, rates):
        for r in zip(*rates):
            current_index = self.data_df.index.max()+1
            if np.isnan(current_index):
                current_index = 0
            self.data_df.loc[current_index] = r

    def add_p_value(self, p_value, label):
        current_index = self.stats_df.index.max()+1
        if np.isnan(current_index):
            current_index = 0
        self.stats_df.loc[current_index] = p_value, label

    def _compare_all_categories(self):
        import itertools
        all_combinations = list(itertools.combinations(self.data_df.columns, 2))

        for combination in all_combinations:
            label = combination
            data_1 = self.data_df[combination[0]]
            data_2 = self.data_df[combination[1]]
            p_value = r_wilcoxon(data_1, data_2)
            self.add_p_value(p_value, label)

    def mean_rates(self):
        return np.mean(self.rates, axis=0)

    def filter_out_irrelevant_comparisons(self):
        short_stimuli = ['cw', 'acw', 'baseline_pre_short', 'baseline_post_short']  # TODO generalise
        long_stimuli = ['baseline_pre', 'stimulus', 'baseline_post']
        irrelevant_indices = []
        for i, (value, labels) in enumerate(self.stats_df.values):
            if not (all(l in short_stimuli for l in labels) or all(l in long_stimuli for l in labels)):
                irrelevant_indices.append(i)
        return self.stats_df.drop(self.stats_df.index[irrelevant_indices])


class PopulationStats(object):
    def __init__(self, stimuli_struct, stimuli=None, cluster_ids=None):
        self.stimuli_struct = stimuli_struct

        if stimuli is None:
            self.stimuli = self.stimuli_struct.stimuli
        else:
            self.stimuli = stimuli
        self.labels = self.stimuli[0].labels
        self.data_df = None
        self.stats_df = None
        self.cluster_ids = cluster_ids

    def initialise_condition_df(self, additional_labels=None):
        data = ['id', 'cid', 'channel']
        data.extend(self.labels)
        if additional_labels is not None:
            data.extend(additional_labels)
        data = np.array([data])
        df = pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])
        return df

    @staticmethod
    def initialise_stats_df():
        data = np.array([['id', 'p_value_wilcoxon', 'label']])
        df = pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])
        return df

    def add_to_condition_df(self, data):
        for r in zip(*data):
            current_index = self.data_df.index.max()+1
            if np.isnan(current_index):
                current_index = 0
            self.data_df.loc[current_index] = r

    def add_p_value(self, p_value, label):
        current_index = self.stats_df.index.max()+1
        if np.isnan(current_index):
            current_index = 0
        self.stats_df.loc[current_index] = p_value, label

    def compare_all_categories(self, paired=True):
        import itertools
        all_combinations = list(itertools.combinations(self.data_df.columns, 2))
        stats_df = self.initialise_stats_df()

        for combination in all_combinations:
            label = combination
            data_1 = self.data_df[combination[0]]
            data_2 = self.data_df[combination[1]]
            p_value = r_wilcoxon(data_1, data_2, paired=paired)
            self.add_p_value(p_value, label)
        return stats_df

    def add_cluster_to_df(self, cid):
        rates, labels = self.stimuli_struct.get_rates_all_sub_stimuli(cid=cid, stimuli=self.stimuli)
        c = Cluster(self.stimuli_struct.sp, cid)
        cluster_stats = ClusterStats(cid, self.stimuli_struct, stimuli=self.stimuli)
        relevant_p_values, relevant_labels = self.get_relevant_comparisons(cluster_stats.stats_df)

        data = [[cid], [c.best_channel]]
        mean_rates = np.mean(rates, axis=1)
        nested_means = [[mean] for mean in mean_rates]

        data.extend(nested_means)
        data.extend(relevant_p_values)

        if self.data_df is None:
            self.data_df = self.initialise_condition_df(relevant_labels)
        self.add_to_condition_df(data)

    def compute_p_values(self, cid, rates=None, labels=None):
        cluster_stats = ClusterStats(cid, self.stimuli_struct, stimuli=self.stimuli)
        cluster_stats.add_rates_to_df(rates)
        cluster_stats._compare_all_categories()
        return cluster_stats.stats_df

    @staticmethod
    def get_relevant_comparisons(stats_df):  # TODO: refactor to allow user defined groups
        short_stimuli = ['cw', 'acw', 'baseline_pre_short', 'baseline_post_short']
        long_stimuli = ['baseline_pre', 'stimulus', 'baseline_post']
        relevant_data = []
        relevant_labels = []
        for value, labels in stats_df.values:
            if all(l in short_stimuli for l in labels) or all(l in long_stimuli for l in labels):
                relevant_data.append([value])
                relevant_labels.append('_vs_'.join(list(labels)))
        return relevant_data, relevant_labels

    def add_clusters_to_df(self):
        for cid in self.cluster_ids:
            self.add_cluster_to_df(cid)


def r_wilcoxon(arr1, arr2, paired=True):
    pd_arr1 = pd.Series(arr1)
    pd_arr2 = pd.Series(arr2)
    result = rstats.wilcox_test(pd_arr1, pd_arr2, paired=paired, exact=True)
    p_val = result[2][0]
    return p_val
