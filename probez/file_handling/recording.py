import os
import re

import numpy as np
from configobj import ConfigObj

from file_handling import recording_io
from util import detrending


class Recording:

    """
    class for dealing with individual files acquired with a probe, specifically for managing all files that
    are associated with a particular recording session, defined as one continuous recording period with a probe.


    example usage:

    >>> input_folder = "./my/path/to/the/directory"
    >>> rec = Recording(input_folder, n_chan=385,trigger_channel=384)
    >>> path = rec.get_path('.imec.ap.bin')
    >>> my_raw_data = rec.get_data(path)

    My data contain a trigger channel in addition to the recording sites, which in this case is the largest channel
    number (384, when 0 indexed).

    >>> ext_in, ext_out = '.imec.ap.bin', '.imec.ap_no_trig.bin'
    >>> n_chan = 385
    >>> trigger_channel_index = 384
    >>> rec.save_trigger(ext_in, n_chan, trigger_channel_index)
    >>> rec.remove_channels_from_data(ext_in, ext_out, 385, channels_to_discard=[trigger_channel_index])
    >>> path_no_trig = rec.get_path('ap_no_trig')
    >>> my_data_without_trigger_channel = rec.get_data(path_no_trig, 384)

    """

    # TODO: profile, file operations are slow

    def __init__(self, input_folder, output_folder, file_name, n_chan=None, dtype='h', trigger_channel=-1):
        self.name = file_name.split('.')[0]
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.dtype = dtype
        self.n_chan = n_chan
        self.trigger_index = trigger_channel
        self.path = os.path.join(self.input_folder, file_name)
        self.start = None
        self.end = None

    def __gt__(self, other, pattern='_\d\d_'):
        """
        enables np.sort to work on this class when instances are in a list
        NOTE: for different file names the pattern needs to be changed (test to make sure it works as desired)
        :param other:
        :param pattern: the expression that the sort function looks for
        :return:
        """
        index_self = re.findall(pattern, self.name)[0][1:-1]
        index_other = re.findall(pattern, other.name)[0][1:-1]
        if int(index_self) > int(index_other):
            return True
        else:
            return False

    @property
    def data(self):
        return self.get_data()[0]

    @staticmethod
    def rec_file(path, n_chan):
        return recording_io.RecordingIo(path, n_chan)

    @staticmethod
    def file_size(path):
        """
        :param :
        :return:
        """
        info = os.stat(path)
        return info.st_size

    def meta_data(self, ext='.imec.ap.meta'):
        try:
            file_path = self.get_path(ext)
            config = ConfigObj(file_path)
            return config
        except OSError('file not present'):
            return

    def get_path(self, ext):
        return os.path.join(self.input_folder + self.name + ext)

    def get_data(self):
        shaped_data, shape = self.load_probe_data_memmap(self.path, self.n_chan)
        return shaped_data, shape

    def load_probe_data_memmap(self, path, n_chan):
        data = np.memmap(path, dtype='int16', mode='r')
        if data.shape[0] % n_chan != 0:
            raise ValueError('n_chan is incorrect, try again')
        shape = (int(data.shape[0] / n_chan), n_chan)
        shaped_data = np.memmap(path, shape=shape, dtype='int16', mode='r')
        return shaped_data, shape

    def remove_channels_from_data(self, n_chan, channels_to_discard=[], start_sample=0, end_sample=None, out_root=None):
        # TODO: test what happens if channel to discard too high

        if end_sample is None:
            end_sample = self.data.shape[0]

        if out_root is None:
            out_root = self.input_folder
        else:
            out_root = out_root

        path_in = self.path
        path_out = os.path.join(out_root, '{}_{}_to_{}_{}.imec.ap.bin'.format(self.name, start_sample,
                                                                              end_sample, channels_to_discard))  # FIXME: should it use self.output_folder ?
        rec_file = self.rec_file(path_in, n_chan)
        rec_file.process_to_file(path_in, path_out, n_chan, channels_to_discard,
                                 processing_func=rec_file.remove_channels_from_chunk,
                                 on_data=False, start_sample=start_sample, end_sample=end_sample)

    def subtract_common_avg_reference(self, n_chan=None, start_sample=0, end_sample=None, save_shortened_trigger=False,
                                      out_root=None, trigger_idx=(-1,), processing_func=detrending.median_subtract_car_and_bandpass):  # TODO: move out_root to class
        """
        SK: calls for the specified function to pre-process data

        :param in n_chan: total number of channels
        :param in trigger_idx: indices of triggers channels
        :param in start_sample:
        :param in end_sample:
        :param save_shortened_trigger: whether to save the triggers or not
        :param out_root: the folder destination of the output data
        :return:
        """
        if n_chan is None:
            n_chan = self.n_chan

        # TODO: n_chan_per_chunk input
        self.start = start_sample
        self.end = end_sample
        path_in = self.path

        if out_root is None:
            out_root = self.input_folder
        else:
            out_root = out_root

        path_out = os.path.join(out_root, '{}_{}_to_{}.imec.ap.bin'.format(self.name, start_sample, end_sample))
        rec_file = self.rec_file(path_in, n_chan)
        rec_file.process_to_file(path_in, path_out, n_chan, [],
                                 processing_func=processing_func, on_data=True,
                                 start_sample=start_sample, end_sample=end_sample)

        if save_shortened_trigger:
            for trigger_index in trigger_idx:
                self.save_trigger(start_sample, end_sample, out_root, trigger_index)

    def save_trigger(self, start_sample, end_sample, out_root, trigger_index=-1):
        if start_sample is None:
            start_sample = 0

        if end_sample is None:
            end_sample = self.data.shape[0]

        trigger_path = os.path.join(out_root, '{}_trigger_{}_to_{}_chan{}.npy').format(self.name, start_sample, end_sample, trigger_index)
        np.save(trigger_path, self.data[start_sample:end_sample, trigger_index].squeeze())  # hack remove to function/attribute

    # def save_trigger(self):
    #     """
    #
    #     :param path:
    #     :param n_chan:
    #     :param trigger_index:
    #     :return:
    #     """
    #     trigger = self.get_data()[0][:, self.trigger_index]
    #     trigger.tofile(self.get_path('trigger'))

    def find_boundaries(self, data, trigger_index, jump_forward=0):
        """
        defines data start and end points based on trigger (start) and artifacts (end)
        :return start: the beginning of the desired recording, defined as the trigger
        :return end: the end of the desired recording, defined as the end of the file unless there is a weird artefact
        that I had, which would mean suddenly only 0s
        """
        trigger_trace = data[:, trigger_index]
        start = self.find_trigger_start(trigger_trace)
        if start == 0:
            start += jump_forward  # arbitrary, but enough samples to avoid probe initialisation

        end = self.find_end(data, channel_index=0)

        return start, end

    @staticmethod
    def is_trigger(trace):
        n_different_numbers = len(np.unique(trace))
        if n_different_numbers == 1:
            return False
        return True

    @staticmethod
    def is_artefact(trace):
        n_different_numbers = len(np.unique(trace))
        if n_different_numbers == 3:
            return True

    def find_trigger_start(self, trigger_trace):
        """
        takes a trigger trace and returns the first index where value is different to the starting value
        :param np.array trigger_trace:
        :return:
        """

        if not self.is_trigger(trigger_trace):
            print('not trigger')
            return 0

        p0 = -1

        for i, p in enumerate(trigger_trace):
            if p != p0:
                if all(trigger_trace[i:i+10] != p0):
                    return max(0, i)
                else:
                    continue

    def find_end(self, data, channel_index):

        trace = self.trace(data, channel_index)

        if self.is_artefact(trace):
            for i, p in enumerate(trace):
                if p == 0:
                    if all(trace[i:i+10] == 0):
                        return i
                    else:
                        continue
        return len(trace)

    @staticmethod
    def trace(data, channel_index, trigger=False):
        trace = data[:, channel_index]
        if trigger:
            if len(np.unique(trace[0:10000])) > 2:  # check values present to ensure trigger
                raise ValueError('trace not trigger')  # is actually a trigger
        return trace

    @property
    def processed_data(self):
        processed_name = '{}_{}_to_{}.imec.ap.bin'.format(self.name, self.start, self.end)
        path_to_processed_data = os.path.join(self.output_folder, processed_name)
        return self.load_probe_data_memmap(path_to_processed_data, self.n_chan)[0]

    @property
    def trigger(self):  # FIXME: convert to numpy first
        raw_trigger = self.data[:, self.trigger_index]
        zero_centred_trigger = raw_trigger - raw_trigger.min()  # trigger can have different resting values depending on
                                                                # the hardware configuration
        return zero_centred_trigger

