import warnings

import numpy as np
import pandas as pd
from datetime import datetime

from cached_property import cached_property

from analysis.probe.probe_io.io_exceptions import IgorFileKeyError, BonsaiQueryError
from analysis.stimulus import DEGREES_PER_VOLT
from pyphys.pyphys import PxpParser


class ProbeIo(object):
    def __init__(self, path):
        self.path = path


class TriggerTraceIo(ProbeIo):
    """
    Handling TTL inputs recording on *usually* channel -1 on the probe. Can be a single trace with digital multiple
    inputs (denoted by different values) that need to be split. Or can be multiple analog traces, depending on the probe
    """

    def __init__(self, combined_trigger_path, stimulus_onset_trigger_path=None, photodiode_trigger_path=None):
        super(TriggerTraceIo, self).__init__(combined_trigger_path)

        if stimulus_onset_trigger_path is None and photodiode_trigger_path is None:
            self.stimulus_onset_trigger_trace, self.photodiode_trigger_trace = self._split_combined_trigger()
        else:
            self.stimulus_onset_trigger_trace = np.load(stimulus_onset_trigger_path).squeeze()
            self.photodiode_trigger_trace = np.load(photodiode_trigger_path).squeeze()

    @cached_property
    def _raw_data(self):
        return np.load(self.path)

    def _all_trigger_locs(self):
        return np.where(np.diff(self.stimulus_onset_trigger_trace) > 0)[0][::2]

    def _split_combined_trigger(self):
        trigger = self._raw_data
        split_trigger_traces = []

        if min(trigger) != 0:
            trigger -= trigger.min()

        for input_value in np.unique(trigger)[1:]:
            trigger_input = trigger == input_value
            split_trigger_traces.append(trigger_input)
        return split_trigger_traces

    @cached_property
    def ordered_triggers(self):
        return self._all_trigger_locs()

    def get_trigger(self, idx):
        return self.ordered_triggers[idx]

    @property
    def n_triggers(self):
        return len(self.ordered_triggers)

    def get_corrected_trigger(self, idx):
        return self.get_trigger(idx) + self.get_photodiode_offset(idx)

    def get_photodiode_offset(self, idx):
        this_trigger_onset = self.get_trigger(idx)

        next_trigger_onset = self.get_trigger(idx + 1) if idx < (self.n_triggers-1) else len(self.stimulus_onset_trigger_trace)

        for i, val in enumerate(self.photodiode_trigger_trace[this_trigger_onset:next_trigger_onset]):
            if val:
                print('the offset for trigger number: {} was found to be {}'.format(idx, i))
                return i
        print('no photodiode trigger found in this stimulus')
        return 0


def test_get_corrected_trigger(trigger_trace_io, idx=0):

    original_trigger_location = trigger_trace_io.get_trigger(idx)
    correction = trigger_trace_io.get_photodiode_offset(idx)
    corrected_trigger = trigger_trace_io.get_corrected_trigger(idx)

    assert corrected_trigger == original_trigger_location + correction
    assert trigger_trace_io.photodiode_trigger_trace[corrected_trigger]


class BonsaiIo(ProbeIo):

    """
    BonsaiIO deals with all metadata surrounding a given stimulus - i.e. the parameters of stimuli presented by bonsai.
    It reads this from a csv file and orders the list according to the time stamp. Currently any false triggers need to
    be deleted before creating a BonsaiIo instance.
    """

    def __init__(self, path: object) -> object:
        super(BonsaiIo, self).__init__(path)

    @cached_property
    def data(self):
        return pd.read_csv(self.path)

    def get_datapoint(self, idx):
        return self.data.loc()[idx]

    def get_condition(self, idx):
        return self.get_datapoint(idx)['Condition']

    @property
    def ordered_conditions(self):
        return list(self.data['Condition'])

    @property
    def n_triggers(self):
        return len(self.ordered_conditions)

    def keys(self, exclude=None):
        if exclude is None:
            exclude = ['Timestamp']
        keys = list(self.data.keys())
        [keys.remove(key) for key in exclude]
        return keys

    def values(self, idx):
        values = []
        for key in self.keys():
            values.append(self.get_datapoint(idx)[key])
        return values

    def matches_attributes(self, attributes_dict, idx):
        f_df = self.filter(self.relevant_attributes_dictionary(attributes_dict))
        if idx in f_df.index.values:
            return True
        else:
            return False

    def get_relevant_attributes_for_comparison(self, attributes_dict):
        return set(self.data.keys()).intersection(attributes_dict.keys())

    def relevant_attributes_dictionary(self, attributes_dict):
        relevant_keys = self.get_relevant_attributes_for_comparison(attributes_dict)
        relevant_dictionary = {k: attributes_dict[k] for k in attributes_dict if k in relevant_keys}
        return relevant_dictionary

    def compare(self, a, cmp_str, b):

        b = self.cast(b, a)

        if cmp_str == 'abs':
            return abs(a) == b
        if cmp_str == '==':
            return a == b
        elif cmp_str == '!=':
            return a != b
        elif cmp_str == '>':
            return a > b
        elif cmp_str == '<':
            return a < b
        elif 'and' in cmp_str:
            raise NotImplementedError('AND operation is not yet implemented')

    def cast(self, b, a):
        """
        Converts the query key from a string to the appropriate type in the data series

        assumes all elements of data series are of the same type

        :param b:
        :param a:
        :return:
        """

        return type(a.iloc()[0])(b)

    def filter(self, filter_dict):

        f_df = self.data.copy()

        for k, v in filter_dict.items():

            if k not in f_df.keys():
                raise BonsaiQueryError(k)

            cmp_str = v.split(' ', 1)[0]
            cmp_val = v.rsplit()[-1]
            query_result = self.compare(f_df[k], cmp_str, cmp_val)
            n_results = np.count_nonzero(query_result)

            if n_results == 0:
                return False

            f_df = f_df[query_result]

        return f_df


class IgorIo(ProbeIo):

    """
    IgorIo deals with the stimulus waveforms themselves - i.e. how is the stimulus broken into different sections.
    These waveforms need to be rescaled in y and the sampling frequency is currently an important determinant of later analysis.
    """

    DEGREES_PER_VOLT = 20

    def __init__(self, path, keys=None):
        """

        :param path: path to the igor file
        :param list keys: a list of string keys (MUST BE ALL KEYS) of all conditions of interest
        """
        super(IgorIo, self).__init__(path)
        self.all_keys = keys

    @cached_property
    def data(self):
        return PxpParser(self.path)

    def sorted_keys(self):
        all_times = {}
        for condition in self.all_keys:
            condition_time = self.get_timestamp_from_condition_block(condition)
            if condition_time is not None:
                all_times.setdefault(condition, condition_time)
        return sorted(all_times, key=lambda k: all_times[k])  # sorted dictionary all_keys by value

    def get_sampling_rate(self, idx):
        condition = self.get_condition(idx)
        return self.get_sampling_rate_from_condition(condition)

    def get_condition(self, idx):
        n_waveforms = 0
        for condition in self.sorted_keys():
            waveforms = self.get_waveforms_in_condition(condition)
            n_waveforms += len(waveforms)
            if idx <= n_waveforms:
                return condition

    def get_timestamp_from_condition_block(self, condition_key):
        """
        Each set of igor waveforms should be saved in order of acquisition
        triggers will be sorted on this basis. This function gets the timestamp for a condition block

        :return:

        """
        file_time_key = 'FileTime'

        if 'FileTime' not in self.data.data[condition_key]['vars'] and 'fileTime' not in self.data.data[condition_key]['vars']:
            raise IgorFileKeyError('FileTime not present in this igor file, '
                                   'please ensure this is added in acquisition order')

        if 'FileTime' in self.data.data[condition_key]['vars']:
            file_time_key = 'FileTime'

        condition_block_time_str = self.data.data[condition_key]['vars'][file_time_key][0]
        condition_block_time_dt = datetime.strptime(condition_block_time_str, '%H:%M:%S').time()

        return condition_block_time_dt

    def get_sampling_rate_from_condition(self, condition):
        return self.data.data[condition]['vars']['CPG_samplingrate']

    @property
    def ordered_waveforms(self):
        return self._get_ordered_cmd_waveform_list()

    def _get_ordered_cmd_waveform_list(self):
        all_waveforms = []

        for condition in self.sorted_keys():
            all_waveforms.extend(self.get_waveforms_in_condition(condition))

        upscaled_waveforms = [wfm * DEGREES_PER_VOLT for wfm in all_waveforms]

        return upscaled_waveforms

    def get_group_trigger_trace(self, condition):

        """
        the TTL given with the command. used for splitting command into multiple
        :param condition:
        :return:
        """
        return self.data.data[condition]['cpg_ttlStim']

    def get_command(self, condition):
        return self.data.data[condition]['command']

    def get_waveforms_in_condition(self, condition):
        """
        extracts individual waveforms from igor command using the igor_ttl

        :param condition:
        :return:
        """

        all_waveforms = []
        triggers = self.get_group_igor_trigger_locs(condition)
        command = self.get_command(condition)
        waveform_length = np.diff(triggers)[0]
        #assert len(np.unique(np.diff(triggers))) == 1

        if len(np.unique(np.diff(triggers))) != 1:
            warnings.warn('there seems to be a dropped sample in the stimulus from igor, go check this')

        for i, trigger in enumerate(triggers):
            wfm_end = (i+1)*waveform_length
            waveform = command[trigger:wfm_end]
            all_waveforms.append(waveform)

        return all_waveforms

    def get_group_igor_trigger_locs(self, condition):
        trigger_trace = self.get_group_trigger_trace(condition)
        triggers_in_group = [0]  # first trigger incorrectly starts at 1 so this is set to 0
        triggers = np.where(np.diff(trigger_trace) > 1)[0] + 1
        triggers_in_group.extend(triggers)
        return triggers_in_group

    def get_waveform(self, idx):
        print(idx)
        return self._get_ordered_cmd_waveform_list()[idx]

    @property
    def n_triggers(self):
        return len(self.ordered_waveforms)
