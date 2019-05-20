import math
import os

import numpy as np
from cached_property import cached_property
from scipy import io

from rotation_analysis.analysis.probe.config import FS_probe
from rotation_analysis.analysis.stimulus import Stimulus, RotationStimulus, DEGREES_PER_VOLT
from rotation_analysis.resample import resampling
from margrie_libs.margrie_libs.signal_processing.list_utils import flatten as flatten_list
import matplotlib.pyplot as plt


ANALYSED_WINDOW_DURATION = 1


class StaticStimulus(Stimulus):

    def __init__(self, idx, loc, waveform, bonsai_io, igor_io, use_bsl_2=False, min_range_n_pnts=300, baseline_duration_before_trigger_s=2):
        """

        :param idx:
        :param loc:
        :param waveform:
        :param bonsai_io:
        :param igor_io:
        :param use_bsl_2:
        :param min_range_n_pnts:
        :param baseline_duration_before_trigger_s:
        """

        self.idx = idx

        self.FS_probe = FS_probe
        self.FS_igor = igor_io.get_sampling_rate(self.idx)
        self.n_samples_before_trigger = baseline_duration_before_trigger_s * self.FS_probe

        self.loc = loc - self.n_samples_before_trigger
        self.min_range_n_points = min_range_n_pnts
        self.waveform = waveform


        self.use_bsl_2 = use_bsl_2
        self.bonsai_io = bonsai_io
        self.n_samples_before_trigger = baseline_duration_before_trigger_s


    condition_pairs = (('bsl_short', 'stimulus'),
                       ('whole_trial', 'whole_trial'))

    histogram_plot_labels = ('bsl_short', 'stimulus')  # determines plotting order on histogram

    @cached_property
    def cmd(self):
        return self.upsample_waveform(self.waveform)

    def matches_attributes(self, attributes_dict):

        print('checking static stimulus attributes')
        return self.bonsai_io.matches_attributes(attributes_dict, self.idx)

    def get_bsl_ranges(self):
        start = 0
        end = self.n_samples_before_trigger
        return [[start, end]]

    def get_condition_ranges(self):
        start = self.n_samples_before_trigger
        end = 100000
        return [[start, end]]

    def get_whole_stimulus_ranges(self):
        start = 0
        end = len(self.cmd + self.n_samples_before_trigger)
        return [[start, end]]

    @property
    def sampling_interval(self):
        """
        The sampling interval in seconds

        Same for all angles
        """
        return 1 / self.FS_probe

    def get_ranges_by_type(self, range_type, range_type2=None):
        expected_range_types = ('bsl_short', 'stimulus', 'whole_trial')
        if range_type not in expected_range_types:
            raise ValueError('Range must be one of {}, got "{}"'.format(expected_range_types, range_type))

        range_functions = {
            'bsl_short': self.get_bsl_ranges,
            'stimulus': self.get_condition_ranges,
            'whole_trial': self.get_whole_stimulus_ranges,
        }

        ranges = range_functions[range_type]()
        ranges = [rng for rng in ranges if (rng[1] - rng[0]) > self.min_range_n_points]
        return ranges

    def upsample_waveform(self, wfm):
        scale_factor = self.FS_probe/self.FS_igor
        x = np.linspace(0, len(wfm)*scale_factor, len(wfm))
        y = wfm.flatten()
        x_vals = np.linspace(0,  len(wfm)*scale_factor, len(wfm)*scale_factor)
        upsampled_command_waveform = np.interp(x_vals, x, y)
        return upsampled_command_waveform

    def get_ranges_duration(self, ranges):  # WARNING: private
        return sum([(end - start) * self.sampling_interval for start, end in ranges])

    def plot(self):
        plt.axvline(self.get_bsl_ranges()[0][0], 0, 1, color='r')
        plt.axvline(self.get_bsl_ranges()[0][1], 0, 1, color='r', linewidth=1.5)
        plt.axvline(self.get_condition_ranges()[0][0], 0, 1, color='r')
        plt.axvline(self.get_condition_ranges()[0][1], 0, 1, color='r')


class ProbeRotationStimulus(RotationStimulus):
    def __init__(self, idx, loc, waveform, bonsai_io, igor_io, use_bsl_2=False, min_range_n_pnts=1000):
        """

        :param idx:
        :param loc:
        :param waveform:
        :param bonsai_io:
        :param igor_io:
        :param use_bsl_2:
        :param min_range_n_pnts: used to remove unrealistically short sub-stimulus regions. Can cause unexpected behaviour if incorrect
        """

        self.idx = idx
        self.loc = loc

        self.waveform = waveform  # non-upsampled
        self.FS_igor = igor_io.get_sampling_rate(self.idx)
        self.FS_probe = FS_probe

        self.angle = math.ceil(max(waveform))
        self.bonsai_io = bonsai_io
        self.n_samples_before_trigger = 0
        super(ProbeRotationStimulus, self).__init__(use_bsl_2, min_range_n_pnts)


    histogram_plot_labels = ('bsl_short', 'c_wise', 'c_c_wise')

    @cached_property
    def cmd(self):
        return self.upsample_waveform(self.waveform)

    def matches_attributes(self, attributes_dict):
        print('checking rotation stimulus attributes')

        return self.bonsai_io.matches_attributes(attributes_dict, self.idx)

    @property
    def sampling_interval(self):
        """
        The sampling interval in seconds

        Same for all angles
        """
        return 1 / FS_probe  # FIXME: hacky AF

    def upsample_waveform(self, wfm):  # FIXME: should be in parent class
        scale_factor = self.FS_probe/self.FS_igor
        x = np.linspace(0, len(wfm)*scale_factor, len(wfm))
        y = wfm.flatten()
        x_vals = np.linspace(0,  len(wfm)*scale_factor, len(wfm)*scale_factor)
        upsampled_command_waveform = np.interp(x_vals, x, y)
        return upsampled_command_waveform

    @staticmethod
    def load_waveform_in_ms_from_matlab(path):
        if not os.path.exists(path):
            raise FileNotFoundError('Could not find file at path: {}'.format(path))
        wfm = io.loadmat(path)
        return wfm['waveform']['data'][0][0]

    def raw_cmd(self):
        cmd_path = os.path.join(self.rec.dir, 'cmd', 'waveform_180.mat')  # FIXME:
        wfm = self.load_waveform_in_ms_from_matlab(cmd_path)
        return np.concatenate([np.zeros(177), np.array(flatten_list(wfm)) * DEGREES_PER_VOLT])  #FIXME: this code exists in two places

    def get_command(self, rec_dir):  # FIXME: rec_dir is not used yet
        return self.cmd

    @property
    def x(self):
        return self.upsample_waveform(self.raw_x())  # FIXME:

    def raw_x(self):
        n_pnts_cmd = len(self.waveform)
        _cmd_x = np.linspace(0, n_pnts_cmd - 1, n_pnts_cmd)
        return _cmd_x / self.FS_probe

    @property
    def velocity(self):  # this is recomputed from downsampled because fitting on upsampled is slow
        pos = self.waveform
        start_p, end_p = self._get_full_cycles_spin_range(pos)[0]
        undersampled_vel = resampling.get_velocity_from_position(pos, self.raw_x(), start_p, end_p)
        velocity = self.upsample_waveform(undersampled_vel)
        return velocity

    @property
    def acceleration(self):  # this is recomputed from downsampled because fitting on upsampled is slow
        pos = self.waveform
        start_p, end_p = self._get_full_cycles_spin_range(pos)[0]
        undersampled_vel = resampling.get_acceleration_from_position(pos, self.raw_x(), start_p, end_p)
        acceleration = self.upsample_waveform(undersampled_vel)
        return acceleration

    def get_acceleration_cmd(self):
        pos = self.raw_cmd()
        start_p, end_p = RotationStimulus._get_full_cycles_spin_range(pos)[0]
        acceleration = resampling.get_acceleration_from_position(pos, self.x, start_p, end_p)
        return self.upsample_waveform(acceleration)  # FIXME: check amplitude

    @staticmethod
    def get_angle_from_waveform(waveform, degrees_per_volt=20):
        return round(max(waveform), 2) * degrees_per_volt

    @staticmethod
    def _get_full_cycles_spin_range(sine):
        peaks_list = ProbeRotationStimulus.get_sine_peaks(sine)
        start = peaks_list[0]
        end = peaks_list[-1]
        return ((start, end), )

    def get_cmd_peaks(self):
        return ProbeRotationStimulus.get_sine_peaks(self.cmd)

    def get_full_cycles_spin_range(self):  # TEST:
        """
        The start and end points of the part of the spinning that is analysed for cwise/ccwise (removes ramps)

        :return:

        """

        # FIXME: for velocity self is an ndarray!

        return ProbeRotationStimulus._get_full_cycles_spin_range(self.cmd)

    def plot(self):  # FIXME: why is this runnning many times
        plt.plot(np.arange(self.rec.loc, self.rec.loc + len(self.cmd)), self.cmd)
        plt.plot(self.get_ranges_by_type('bsl')[0][0] + self.loc, 0, 'o', color='r')
        plt.plot(self.get_ranges_by_type('bsl')[0][1] + self.loc, 1, 'o', color='r')
        plt.plot(self.get_ranges_by_type('c_wise')[0][0] + self.loc, 2, 'o', color='k')
        plt.plot(self.get_ranges_by_type('c_wise')[0][1] + self.loc, 3, 'o', color='k')
        plt.plot(self.get_ranges_by_type('c_c_wise')[0][0] + self.loc, 4, 'o', color='b')
        plt.plot(self.get_ranges_by_type('c_c_wise')[0][1] + self.loc, 5, 'o', color='b')

