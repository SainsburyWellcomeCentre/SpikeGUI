import operator

import numpy as np
import pandas as pd
from margrie_libs.stats.stats import wilcoxon


def abs_eq(a, b):
    return abs(a) == b


class DatabaseIo(object):
    def __init__(self, db_path):
        self.db = pd.read_csv(db_path)

    @staticmethod
    def filter_by_idx(db, key, idx_list):
        if idx_list is None:
            return db

        results_df = pd.DataFrame()
        for idx in idx_list:
            results_df = results_df.append(db[db[key] == idx], ignore_index=True)
        return results_df

    @staticmethod
    def filter_by(db, key, condition):

        if condition is None:
            return db

        return db[db[key] == condition]

    def get_filter_function(self, key):
        filter_functions = {
                            'cell_id': self.filter_by_idx,
                            'trial_id': self.filter_by_idx,
                            'within_stimulus_condition': self.filter_by,
                            'between_stimuli_condition': self.filter_by,
                            'between_stimuli_condition_metric': self.filter_by,
                            'metric': self.filter_by
                            }

        return filter_functions[key]

    def compare_groups(self, query_dict, other_query_dict):
        trial_group = self.filter_df(query_dict)
        other_trial_group = self.filter_df(other_query_dict)

        if len(trial_group) == 0:
            raise ValueError('there are no trials with these parameters: {}'.format(query_dict))

        if len(other_trial_group) == 0:
            raise ValueError('there are no trials with these parameters: {}'.format(other_query_dict))

        group_vals = self.get_group_values(trial_group)
        other_group_vals = self.get_group_values(other_trial_group)

        group_avg = self.get_group_mean(trial_group)
        other_group_avg = self.get_group_mean(other_trial_group)

        wilcox_p = wilcoxon(group_vals, other_group_vals)

        return group_avg, other_group_avg, wilcox_p

    @staticmethod
    def get_group_values(trials):
        return np.array(trials['values'])

    def get_group_mean(self, trials):
        return self.get_group_values(trials).mean()

    def comparison_from_str(self, a, cmp_str, b):
        """

        :param a: a subset dataframe
        :param cmp_str: a comparison string (e.g. == something)
        :param b:
        :return:
        """

        comparators = self.get_comparator_functions()

        b = self.cast_to_same_type(b, a)
        if isinstance(b, list):
            return a.isin(b)
        return comparators[cmp_str](a, b)

    @staticmethod
    def get_comparator_functions():
        comparators = {
            '==': operator.eq,
            '!=': operator.ne,
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            'and': operator.and_,
            'or': operator.or_,
            'in': operator.contains,
            'abs': abs_eq
        }
        return comparators

    def cast_to_same_type(self, b, a):
        """
        Converts the query key from a string to the appropriate type in the data series

        assumes all elements of data series are of the same type

        :param b:
        :param a:
        :return:
        """

        a_type_function = type(a.iloc()[0])
        if b.startswith('['):
            b = b.strip('[]').split(',')
            return [a_type_function(b_element.strip()) for b_element in b]
        return a_type_function(b)

    def filter_df(self, filter_dict):

        f_df = self.db.copy()

        for k, v in filter_dict.items():

            if k not in f_df.keys():
                raise ValueError(k)

            components = v.split(' ', 1)
            cmp_str = components[0]
            cmp_val = components[-1]
            query_result = self.comparison_from_str(f_df[k], cmp_str, cmp_val)
            n_results = np.count_nonzero(query_result)

            if n_results == 0:
                return False

            f_df = f_df[query_result]

        return f_df
