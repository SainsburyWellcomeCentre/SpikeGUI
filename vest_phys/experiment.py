import os
import sys
from collections import OrderedDict as ODict
from math import floor
from copy import copy, deepcopy

import shutil
import errno

import matplotlib
import numpy as np
from cached_property import cached_property
from scipy import stats

from experiment_resampler import ExperimentResampler
from plotting.experiment_plotter import ExperimentPlotter
from signal_processing.resampled_matrix import ResampledMatrix

#sys.path.append('/home/crousse/code/pyphys/pyphys')
# noinspection PyUnresolvedReferences
#from pyphys import PxpParser as Parser

from signal_processing.signal_processing import low_pass, count_points_between_values
from signal_processing import mat_utils
from utils.utils import dprint, shell_hilite

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
r_stats = importr("stats")


class Experiment(object):
    """
    .. warning:: clockwise is the first direction regardless of the real direction

    """

    def __init__(self, path, ext='png', channel='A', cell_type='', layer=''):
        """

        :param path:
        :param string ext: File extension for the figures
        :param string channel: The igor recording channel
        :param string cell_type: e.g. pyramid, ct, cc ...
        :param string layer: cortical layer
        """
        self.path = path
        self.name = os.path.splitext(os.path.basename(self.path))[0]
        self.parent_dir = os.path.dirname(path)
        self.dir = os.path.join(self.parent_dir, self.name)
        self.create_dir()

        self.ext = ext

        self.cell_type = cell_type  # TODO: use metadata ?
        self.layer = layer

        self.exp_id, self.data = self.get_data()

        self.raw_data = [self.data[name] for name in self.get_raw_names()]
        self.fix_nans(self.raw_data)
        self.raw_clipped_baselined = [self.data[name] for name in self.get_raw_clipped_baselined_names()]
        self.re_baseline()  # WARNING: baselined on pseudo minimum not mean hence re-baseline on mean
        self.fix_nans(self.raw_clipped_baselined)
        self.raw_clipped_data = self.get_raw_clipped_data()
        self.raw_clipped_baselined_avg = mat_utils.avg_waves(self.raw_clipped_baselined)

        self.bsl_spiking_freq = self.get_baseline_spiking()

        self.resampler = ExperimentResampler(self)
        self.plotter = ExperimentPlotter(self)
        self.write_tables()

    def get_pooled_vms_bsl_cw_ccw(self):
        bsls = []
        cws = []
        ccws = []
        for w in self.raw_clipped_baselined:
            bsl = self.extract_bsl(w)
            bsls.extend(bsl)
            cycles = mat_utils.cutAndGetMultiple(self.cmd, w)
            for c in cycles:
                mid = int(len(c)/2)
                cw = c[:mid]
                cws.extend(cw)
                ccw = c[mid:]
                ccws.extend(ccw)
        bsls = np.array(bsls, dtype=np.float64)
        cws = np.array(cws, dtype=np.float64)
        ccws = np.array(ccws, dtype=np.float64)
        return bsls, cws, ccws

    def re_baseline(self):
        """
        Fixes baseline of raw_clipped_baseline to baseline at 0 not at minimum (that was used for polar plots)

        :return:
        """
        for i in range(len(self.raw_clipped_baselined)):
            bsl = self.extract_bsl(self.raw_clipped_baselined[i])
            offset = bsl.mean()
            self.raw_clipped_baselined[i] = self.raw_clipped_baselined[i] - offset

    def create_dir(self):
        try:
            os.mkdir(self.dir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(self.dir):
                pass
            else:
                raise

    def move_to_folder(self):
        # exp_files = [os.path.join(self.parent_dir, f) for f in os.listdir(self.parent_dir) if self.name in f]
        # exp_files = [f for f in exp_files if os.path.isfile(f)]
        shutil.move(self.path, os.path.join(self.dir, self.name+'.pxp'))

    def fix_nans(self, waves_list, warning_threshold=10, max_nans=35):
        """
        Removes up to max_nans NaNs at the end of the last wave in waves_list.
        If the number of removed NaNs exceed warning_threshold, a Warning is issued.
        If the number would exceed max_nans, an exception is raised.
        Threshold set rather high because in case a spike occurs at the end and the last points are NaN
        then the spike interpolation will give NaN values.

        .. warning:
            Modifies in place

        :param list waves_list:
        :param int warning_threshold:
        :param int max_nans:
        :return:
        """
        nans_indices = np.where(np.isnan(waves_list[-1]))[0]
        n_nans = nans_indices.size
        if n_nans > max_nans:
            raise ValueError("Wave has too many NaNs, {} found ({})".format(n_nans, nans_indices))
        elif n_nans > warning_threshold:
            print("{}: Wave has too many NaNs, {} found ({})".format(
                shell_hilite('WARNING: ', 'red', True),
                n_nans,
                nans_indices)
            )
        expected_nans = np.arange(waves_list[-1].size - n_nans, waves_list[-1].size, 1)
        if not np.array_equal(nans_indices, expected_nans):
            raise ValueError("The NaNs are expected to be located at the end, indices found: {}, on wave of {} points"
                             .format(nans_indices, waves_list[-1].size))
        else:
            if n_nans > 0:
                waves_list[-1][nans_indices] = waves_list[-1][nans_indices[0] - 1]

    @property
    def sampling(self):
        return self.data['vars']['samplingInterval'][0]

    def point_to_ms(self, p):
        """
        Convert a point information to time (in ms)

        :param int p: The point to convert
        :return: The value in milliseconds
        """
        return p * self.sampling * 1000

    @property
    def cmd(self):
        return self.data['cpgCommand']

    @property
    def velocity(self):
        return self.data['polarVelocity']

    @property
    def acceleration(self):
        return self.data['polarAcceleration']

    def get_data(self):
        parser = Parser(self.path)
        protocols = list(parser.data.keys())[2:]  # Discard the 2 first folders
        # exp_id = self.__prompt_id(protocols)  # TODO: make optional if used from batch or not
        exp_id = self._get_id(protocols)
        if exp_id not in self.path:
            raise ValueError("Name mismatch between experiment {} and path {}".format(exp_id, self.path))
        data = parser.data[exp_id]
        return exp_id, data

    def _get_waves_names(self, match_str, good_ids=None):
        """
        Gets the list of names of the waves of a certain type in the current experiment
        Assumes channel A or B only in recording (-2 stripping)
        """

        names = [name for name in list(self.data.keys()) if name.startswith(match_str) and name.endswith('0')]  # FIXME: use channel here

        # Python doesn't do natural sorting of non 0 padded strings
        waves_ids = [int(name[len(match_str):-2]) for name in names]

        if good_ids is not None:
            waves_names = [name for rid, name in sorted(zip(waves_ids, names)) if rid in good_ids]
        else:
            waves_names = [name for rid, name in sorted(zip(waves_ids, names))]
        return waves_names

    def get_raw_names(self):  # TODO: cached_property
        """
        Return the list of raw data waves in the current experiment
        """
        return self._get_waves_names('CombRaw', good_ids=self.keep_ids)

    def get_raw_clipped_baselined_names(self):
        """
        Return the list of raw data waves with spikes clipped in the current experiment
        """
        return self._get_waves_names('CombFS', good_ids=self.keep_ids)

    def get_rasters_names(self):
        """
        Returns a sorted list of names of the form occurrenceCombFSxxx of the spike times waves
        """
        return self._get_waves_names('occurrenceCombFS', good_ids=self.keep_ids)

    def get_raw_clipped_data(self):  # TODO: Test that minima of strides are the same
        raw_clipped_data = []
        for raw_wave, raw_clipped_baselined_wave in zip(self.raw_data, self.raw_clipped_baselined):
            diff = raw_clipped_baselined_wave.min() - raw_wave.min()
            raw_clipped_data.append(raw_clipped_baselined_wave - diff)
        return raw_clipped_data

    def get_clipped_cycles(self):
        """

        .. warning::
            Will only work with 2 cycles (cutInHalf)

        :return columns: t1bsl1+t1cycle1, t1bsl2+t1cycle2, t2bsl1+t2cycle1, t2bsl2+t2cycle2
        :rtype: list
        """
        columns = []
        for trial in self.raw_clipped_data:
            bsl = self.extract_bsl(trial)
            bsls = mat_utils.cutInHalf(bsl)  # WARNING: works only with 2 cycles
            cycles = mat_utils.cutAndGetMultiple(self.cmd, trial)  # WARNING: works only with 2 cycles
            for segment_id in range(self.n_segments):
                column = np.hstack((bsls[segment_id], cycles[segment_id]))
                columns.append(column)
        return columns

    def cut_and_average(self, wave):
        return mat_utils.cutAndAvgSine(self.cmd, wave)

    @property
    def baseline_end(self):
        """
        Index of the last baseline point

        :return:
        """
        return self.recording_start - 1

    @property
    def baseline_length(self):
        return self.baseline_end + 1

    @property
    def baseline_duration(self):
        """
        Baseline duration in seconds

        :return:
        """
        return self.baseline_length * self.sampling

    @property
    def bsl_plot_segment_length(self):
        """

        :return: The width of a single baseline segment in points
        """
        return int(floor((self.baseline_length / self.n_segments)))

    @property
    def bsl_plot_segment_duration(self):
        """
        typically half of baseline duration (because plot segments)
        :return:
        """
        return self.bsl_plot_segment_length * self.sampling

    def extract_bsl(self, wave):
        """
        Extract the portion of wave that corresponds to the first baseline
        """
        return deepcopy(wave[:self.baseline_length])

    def extract_baseline_plot_segment(self, wave):
        """
        averaged 2 halves of baseline

        :param wave:
        :return:
        """
        return mat_utils.cut_and_avg_halves(self.extract_bsl(wave))

    @cached_property
    def segment_length(self):
        cmd_segment = mat_utils.cutAndAvgSine(self.cmd, self.cmd)
        n_pnts_segment = cmd_segment.shape[0]
        return n_pnts_segment

    @cached_property
    def segment_duration(self):
        """
        The duration in time of a segment (e.g. a single clockwise cycle)

        :return:
        """
        segment_duration = self.segment_length * self.sampling
        return segment_duration

    @property
    def half_segment_duration(self):
        """
        Return the duration of a half display segment (i.e. a single clockwise or counter_clockwise ramp) in seconds
        :return:
        """
        mid = self.segment_duration / 2.
        return mid

    def get_command_plot_baselines(self):  # TODO: merge with method below
        cmd_bsl = self.extract_baseline_plot_segment(self.cmd)
        vel_bsl = self.extract_baseline_plot_segment(self.velocity)
        acc_bsl = self.extract_baseline_plot_segment(self.acceleration)
        return cmd_bsl, vel_bsl, acc_bsl

    def get_command_plot_segments(self):
        cmd_segment = self.cut_and_average(self.cmd)  # WARNING: not correct if amplitude of sine decreases or increases
        vel_segment = self.cut_and_average(self.velocity)
        acc_segment = self.cut_and_average(self.acceleration)
        return cmd_segment, vel_segment, acc_segment

    @property
    def data_plot_segment_half(self):
        mid = int(self.data_plot_segment.size / 2)
        return mid

    @cached_property
    def n_segments(self):
        """
        Number of cycles cut from trial

        :return:
        """
        return len(mat_utils.cutAndGetMultiple(self.cmd, self.cmd))

    @cached_property
    def data_plot_segment(self):
        """
        clipped, baselined, averaged (and cut in segments)

        :return:
        """
        return self.cut_and_average(self.raw_clipped_baselined_avg)

    @cached_property
    def bsl_clipped_baselined_mean(self):
        bsl_mean = self.extract_bsl(self.raw_clipped_baselined_avg)
        bsl_mean = mat_utils.cut_and_avg_halves(bsl_mean)  # WARNING: n segments should not be halves but self.n_segments
        return bsl_mean

    @cached_property
    def clock_wise_clipped_baselined_mean(self):
        return deepcopy(self.data_plot_segment[:self.data_plot_segment_half])

    @cached_property
    def c_clock_wise_clipped_baselined_mean(self):
        return deepcopy(self.data_plot_segment[self.data_plot_segment_half:])

    def _get_trend(self, wave):
        return low_pass(wave, 5001)

    @cached_property
    def bsl_trend(self):  # TODO: use
        return self._get_trend(self.bsl_clipped_baselined_mean)

    @cached_property
    def clock_wise_trend(self):
        return self._get_trend(self.clock_wise_clipped_baselined_mean)

    @cached_property
    def c_clock_wise_trend(self):
        return self._get_trend(self.c_clock_wise_clipped_baselined_mean)

    def get_peaks_indices(self):
        return np.array(mat_utils.findSinePeaks(self.cmd))

    def _get_segment_raster(self, raster, segment_start_p, segment_end_p):
        """
        Return the part of the raster that falls between start and end points
        Raster is scaled in ms

        :param raster:
        :param int segment_start_p:
        :param int segment_end_p:
        :return:
        """
        segment_start_t = self.point_to_ms(segment_start_p)  # rasters from Neuromatic are in ms
        segment_end_t = self.point_to_ms(segment_end_p)
        segment_raster = raster[np.logical_and(raster >= segment_start_t, raster < segment_end_t)]
        return segment_raster.copy()

    def get_rasters(self):
        """
        Return the rasters as a list of raster segments (t1s1, t1s2, t1s3, t1s4t2s1, t2s2, t2s3...)
        The rasters are values in seconds of spikes occurences in absolute since sweep start.
        The start time of each segment within the sweep is stored in rasters_start_ts (in seconds)

        .. glossary::
            t: trial
            s: segment

        :return: bsl_rasters, rasters, rasters_start_ts
        """
        peaks_pos = self.get_peaks_indices()  # In points
        negative_peaks = peaks_pos[::2]

        bsl_rasters = []
        rasters = []
        rasters_start_ts = []
        for name in self.get_rasters_names():
            raster = self.data[name]
            for i in range(self.n_segments):
                bsl_start_p = i * self.bsl_plot_segment_length
                bsl_end_p = bsl_start_p + self.bsl_plot_segment_length
                bls_segt_raster = self._get_segment_raster(raster, bsl_start_p, bsl_end_p)
                bls_segt_raster = np.array(bls_segt_raster) / 1000.  # NeuroMatic rasters in ms
                bsl_rasters.append(bls_segt_raster)

                start_p = negative_peaks[i]
                end_p = negative_peaks[i + 1]
                segt_raster = self._get_segment_raster(raster, start_p, end_p)
                segt_raster = np.array(segt_raster) / 1000.  # NeuroMatic rasters in ms
                rasters.append(segt_raster)
                rasters_start_ts.append(start_p * self.sampling)
        return bsl_rasters, rasters, rasters_start_ts

    def get_spiking_freq_lists_per_trial(self):
        """
        Non stacked rasters (list of spiking frequencies) per trial

        :return:
        """
        bsl_rasters, rasters, rasters_starts = self.get_rasters()
        bsl_freqs = []
        clock_wise_freqs = []
        c_clock_wise_freqs = []
        first_quarter_freqs = []
        second_quarter_freqs = []
        third_quarter_freqs = []
        fourth_quarter_freqs = []

        # Using bsl_plot_segment_duration since bsl_rasters split  to compare with clockwise/counterclock
        # BASELINE
        for r in bsl_rasters:
            bsl_freqs.append(r.size / self.bsl_plot_segment_duration)

        # CW CCW
        mid = self.half_segment_duration
        for s, raster in zip(rasters_starts, rasters):
            r = raster - s  # tODO: check if needs np.array
            n_spikes_clock_wise = r[r < mid].size
            clock_wise_freqs.append(n_spikes_clock_wise / mid)
            n_spikes_c_clock_wise = r[r >= mid].size
            c_clock_wise_freqs.append(n_spikes_c_clock_wise / mid)

        # CONTRA IPSI
        quarter_duration = mid / 2
        for s, raster in zip(rasters_starts, rasters):
            r = raster - s  # tODO: check if needs np.array
            first_quarter_freqs.append(r[r < quarter_duration].size / quarter_duration)

            fourth_quarter_freqs.append(r[r >= (quarter_duration*3)].size /quarter_duration)

            second_quarter_n_spikes = r[np.logical_and(r >= quarter_duration, r < mid)].size
            second_quarter_freqs.append(second_quarter_n_spikes / quarter_duration)

            third_quarter_n_spikes = r[np.logical_and(r >= mid, r < (quarter_duration*3))].size
            third_quarter_freqs.append(third_quarter_n_spikes / quarter_duration)

        table = ODict()
        table['bsl'] = bsl_freqs
        table['clockWise'] = clock_wise_freqs
        table['cClockWise'] = c_clock_wise_freqs
        table['cw_contra'] = first_quarter_freqs
        table['ccw_contra'] = fourth_quarter_freqs
        table['cw_ipsi'] = second_quarter_freqs
        table['ccw_ipsi'] = third_quarter_freqs
        return table

    def get_spiking_frequencies(self):
        """
        Uses stacked version of rasters to produce only 3 integers necessary to compute global osi/dsi

        :return tuple(int): baseline spiking frequency, clockwise and counter_clockwise
        """
        bsl_rasters, rasters, raster_starts = self.get_rasters()  # Keep absolute values for mid

        bsl_raster = self._concatenate_rasters(bsl_rasters)
        bsl_freq = bsl_raster.size / self.bsl_plot_segment_duration

        mid = self.half_segment_duration

        raster = self._concatenate_rasters([r - s for r, s in zip(rasters, raster_starts)])
        c_wise_freq = raster[raster < mid].size / mid
        c_c_wise_freq = raster[raster >= mid].size / mid

        return bsl_freq, c_wise_freq, c_c_wise_freq

    def _concatenate_rasters(self, rasters):
        """
        Concatenates the list of rasters

        :param rasters:
        :return:
        """
        raster = np.zeros(0)  # empty array
        for rstr in rasters:
            raster = np.hstack((raster, deepcopy(rstr)))  # TODO: check if flattent would not work
        return raster

    @cached_property
    def keep_ids(self):
        try:
            remove_ids = self.data['ind']
        except KeyError:
            print("{} Experiment {} 'ind' wave missing, assuming keep all.".format(
                shell_hilite("WARNING:", 'yellow', True),
                shell_hilite("{}".format(self.exp_id), 'magenta')
            ))
            remove_ids = []
        all_ids = list(range(len(self._get_waves_names('CombRaw'))))  # Number of raw waves before filtering
        good_ids = [_id for _id in all_ids if _id not in remove_ids]
        good_ids = np.array(good_ids, dtype=np.uint16) + 1  # Neuromatic indexes from 1
        return good_ids

    @cached_property
    def clipped_avgs_per_trial_table(self):
        """
        .. csv-table:: table
            :delim: space

            bsl_trial_0_part1   c_wise_trial_0_part1    c_c_wise_trial_0_part1
            bsl_trial_0_part2   c_wise_trial_0_part2    c_c_wise_trial_0_part2
            bsl_trial_1_part1   c_wise_trial_1_part1    c_c_wise_trial_1_part1
            bsl_trial_1_part2   c_wise_trial_1_part2    c_c_wise_trial_1_part2

        :return ODict: table
        """
        avgs_c_wise, avgs_c_c_wise = self.extract_clipped_avgs()
        table = ODict()
        table['bsl'] = self.extract_clipped_avgs_bsl()
        table['clockWise'] = avgs_c_wise
        table['cClockWise'] = avgs_c_c_wise
        table['cw_contra'] = self.extract_cw_contra_clipped_avgs()
        table['ccw_contra'] = self.extract_ccw_contra_clipped_avgs()
        table['cw_ipsi'] = self.extract_cw_ipsi_clipped_avgs()
        table['ccw_ipsi'] = self.extract_ccw_ipsi_clipped_avgs()
        return table

    def extract_clipped_avgs_bsl(self):
        """
        To be used by compund method clipped_avgs_per_trial_table

        :return: The list of means of each baseline for each trial (2 baseline halves to have same dimension as clockwise/counterclockwise
        """
        avgs_bsl = []
        for trial in self.raw_clipped_data:
            bsl = self.extract_bsl(trial)
            halves = mat_utils.cutInHalf(bsl)
            for half in halves:
                avgs_bsl.append(half.mean())
        return avgs_bsl

    def extract_clipped_avgs(self):
        """
        To be used by clipped_avgs_per_trial_table

        :return: list of pairs of clockwise/counter_clockwise averages per trial
        """
        avgs_c_wise = []
        avgs_c_c_wise = []
        for trial in self.raw_clipped_data:
            segments = mat_utils.cutAndGetMultiple(self.cmd, trial)
            for segment in segments:
                avgs_c_wise.append(segment[:self.data_plot_segment_half].mean())
                avgs_c_c_wise.append(segment[self.data_plot_segment_half:].mean())
        return avgs_c_wise, avgs_c_c_wise

    def extract_cw_contra_clipped_avgs(self):
        avgs = []
        for trial in self.raw_clipped_data:
            segments = mat_utils.cutAndGetMultiple(self.cmd, trial)
            for segment in segments:
                avgs.append(self.extract_cw_contra_from_plot_segment(segment).mean())
        return avgs

    def extract_ccw_contra_clipped_avgs(self):
        avgs = []
        for trial in self.raw_clipped_data:
            segments = mat_utils.cutAndGetMultiple(self.cmd, trial)
            for segment in segments:
                avgs.append(self.extract_ccw_contra_from_plot_segment(segment).mean())
        return avgs

    def extract_cw_ipsi_clipped_avgs(self):
        avgs = []
        for trial in self.raw_clipped_data:
            segments = mat_utils.cutAndGetMultiple(self.cmd, trial)
            for segment in segments:
                avgs.append(self.extract_cw_ipsi_from_plot_segment(segment).mean())
        return avgs

    def extract_ccw_ipsi_clipped_avgs(self):
        avgs = []
        for trial in self.raw_clipped_data:
            segments = mat_utils.cutAndGetMultiple(self.cmd, trial)
            for segment in segments:
                avgs.append(self.extract_ccw_ipsi_from_plot_segment(segment).mean())
        return avgs

    def extract_cw_contra_from_plot_segment(self, segment):
        quarter_len = int(len(segment) / 4.0)  # OPTIMISE: extract
        first_quarter = segment[:quarter_len]
        return deepcopy(first_quarter)  # OPTIMISE: check if deepcopy necessary

    def extract_ccw_contra_from_plot_segment(self, segment):
        quarter_len = int(len(segment) / 4.0)  # OPTIMISE: extract
        last_quarter = segment[-quarter_len:]
        return deepcopy(last_quarter)  # OPTIMISE: check if deepcopy necessary

    def extract_cw_ipsi_from_plot_segment(self, segment):
        quarter_len = int(len(segment) / 4.0)  # OPTIMISE: extract
        mid = int(len(segment) / 2.0)  # TODO: use built in method
        second_quarter = segment[quarter_len:mid]
        return deepcopy(second_quarter)

    def extract_ccw_ipsi_from_plot_segment(self, segment):
        quarter_len = int(len(segment) / 4.0)  # OPTIMISE: extract
        mid = int(len(segment) / 2.0)  # TODO: use built in method
        third_quarter = segment[mid:mid+quarter_len]
        return deepcopy(third_quarter)

    def independant_t_test(self, vect1, vect2):
        """
        Performs an independant t test and returns only the p value
        :param vect1:
        :param vect2:
        :return:
        """
        return stats.ttest_ind(vect1, vect2)[1]

    def paired_t_test(self, vect1, vect2):
        """
        Performs a paired t_test and returns only the p value

        :param vect1:
        :param vect2:
        :return:
        """
        try:
            p_value = stats.ttest_rel(vect1, vect2)[1]
        except ValueError as err:
            raise ValueError("{}; array lengths: {}, {}".format(err, len(vect1), len(vect2)))
        return p_value

    def wilcoxon_test(self, vect1, vect2):
        if len(vect1) != len(vect2):
            raise ValueError("Arrays have different length: {}, {} (exp: {})".
                             format(len(vect1), len(vect2), self.name))
        results = r_stats.wilcox_test(FloatVector(vect1), FloatVector(vect2), paired=True, exact=True)
        return results[results.names.index('p.value')][0]

    def get_deltas(self, bsl_avg, c_wise_avg, cc_wise_avg, cw_contra_avg, ccw_contra_avg, cw_ipsi_avg, ccw_ipsi_avg):
        bsl_delta = 0
        c_wise_delta = c_wise_avg - bsl_avg
        cc_wise_delta = cc_wise_avg - bsl_avg
        cw_contra_delta = cw_contra_avg - bsl_avg
        ccw_contra_delta = ccw_contra_avg - bsl_avg
        cw_ipsi_delta = cw_ipsi_avg - bsl_avg
        ccw_ipsi_delta = ccw_ipsi_avg - bsl_avg
        return bsl_delta, c_wise_delta, cc_wise_delta, cw_contra_delta, ccw_contra_delta, cw_ipsi_delta, ccw_ipsi_delta
        
    def get_dsi(self, bsl_avg, c_wise_avg, cc_wise_avg):
        """

        :param c_wise_avg:
        :param cc_wise_avg:
        :return:
        """
        _, c_wise_delta, cc_wise_delta = self.get_deltas(bsl_avg, c_wise_avg, cc_wise_avg, 0, 0, 0, 0)[:3]  # HACK: to avoid rewriting function without last 4 args
        if abs(c_wise_delta) + abs(cc_wise_delta) != abs(c_wise_delta + cc_wise_delta):  # different signs
            return 1.
        preferred_response = max(abs(c_wise_delta), abs(cc_wise_delta))
        non_preferred_response = min(abs(c_wise_delta), abs(cc_wise_delta))
        if (preferred_response + non_preferred_response) == 0:
            return 'NaN'
        else:
            return (preferred_response - non_preferred_response) / (preferred_response + non_preferred_response)

    def get_max_diff(self):
        """
        The maximum duration bewtween two elements (e.g. angles) in the spike normalisation.
        This function is only to avoid hard coding the number (1000ms or 1 s).

        :return: 1000
        """
        return 1000

    def normalise_spiking(self, levels_w_name):
        """
        levels is of the form:

        .. csv-table::
            :delim: space
            :header: segment, 0, 1, 2, 3, ..., 360

            clockwise_segment1  t0deg   t1deg   t2deg   t3deg    " "   t360deg
            c_clockwise_segmt1  t0deg   t1deg   t2deg   t3deg    " "   t360deg
            clockwise_segment2  t0deg   t1deg   t2deg   t3deg    " "   t360deg
            c_clockwise_segmt2  t0deg   t1deg   t2deg   t3deg    " "   t360deg

        The result is of the form (transposed 1, 0) with a 3rd dimension n_trials

        .. csv-table:: spiking
            :delim: space
            :header: segment, 0, 1, 2, 3, ..., 360

            clockwise_segment1  n_spikes0deg   n_spikes1deg   n_spikes2deg     " "     n_spikes360deg
            c_clockwise_segmt1  n_spikes0deg   n_spikes1deg   n_spikes2deg     " "     n_spikes360deg
            clockwise_segment2  n_spikes0deg   n_spikes1deg   n_spikes2deg     " "     n_spikes360deg
            c_clockwise_segmt2  n_spikes0deg   n_spikes1deg   n_spikes2deg     " "     n_spikes360deg

        .. csv-table:: times
            :delim: space
            :header: segment, 0, 1, 2, 3, ..., 360

            clockwise_segment1  duration0deg   duration1deg   duration2deg      " "    duration360deg
            c_clockwise_segmt1  duration0deg   duration1deg   duration2deg      " "    duration360deg
            clockwise_segment2  duration0deg   duration1deg   duration2deg      " "    duration360deg
            c_clockwise_segmt2  duration0deg   duration1deg   duration2deg      " "    duration360deg

        :param string levels_w_name: The name in self.data (igor data) of the levels_wave we want (e.g. degrees.)
        :return: (spiking, times)
        """
        levels = deepcopy(self.data[levels_w_name]).transpose((1, 0))  # dimensions = (nOrientations, nDegrees)
        spiking, times = self.normalise_spiking_sampling_method_2(levels)
        return spiking, times

    def normalise_spiking_sampling_method_2(self, levels_wave):
        """
        Normalise the spiking of each trial (by degrees, degrees/sec... depending on levels_wave)
        and convert durations to seconds

        :param levels_wave:
        :return:
        """
        norm_raster = []
        durations = []
        for name in self.get_rasters_names():  # For each trial
            raster = np.squeeze(deepcopy(self.data[name]))
            out = self._normalise_spike_sampling_method_2(raster, levels_wave, self.get_max_diff())
            trial_norm_raster, trial_norm_durations = out
            norm_raster.append(trial_norm_raster)
            durations.append(trial_norm_durations)
        norm_raster = np.array(norm_raster).transpose( (2, 0, 1) )
        durations = np.array(durations).transpose( (2, 0, 1) )
        durations /= 1000.  # NM uses ms
        return norm_raster, durations

    def _normalise_spike_sampling_method_2(self, raster, levels_wave, max_diff):  # WARNING: explain
        """
        return the number of spikes in each degree (or degree/sec, segre/sec/sec) bin and the duration of the bin

        :param raster:
        :param np.array levels_wave:
        :param float max_diff:
        :return:
        """
        norm_raster = np.zeros(levels_wave.shape)
        durations = np.zeros(levels_wave.shape)

        for i in range(levels_wave.shape[0]):
            for j in range(levels_wave.shape[1] - 1):
                start_time = levels_wave[i, j]
                end_time = levels_wave[i, j + 1]
                duration = abs(end_time - start_time)  # abs because levels_wave not sorted by time (but by level)
                if duration < max_diff:
                    n_levels = count_points_between_values(start_time, end_time, raster)
                else:
                    n_levels = np.nan
                norm_raster[i, j] = n_levels
                durations[i, j] = duration
        return norm_raster, durations
    
    def get_baseline_spiking(self):
        """
        Returns a list of spikes frequencies computed as :math:`n_spikes / bsl_duration`
        for all rasters matched to self.get_rasters_names.
        Used for normalised matrices.
        """

        spike_ns = []
        i = 0
        bsl_rasters, rasters, rasters_start_ts = self.get_rasters()
        for raster in bsl_rasters:
            if len(raster) > 0:
                try:
                    n_spikes = raster.size
                except RuntimeWarning:
                    print('{} Could not get spikes in baseline from the following wave: {}'.format(
                        shell_hilite('Error:', 'red', True),
                        i)
                    )
                    n_spikes = 0
            else:
                n_spikes = 0
            dprint('Wave: {}, number of spikes in baseline: {}.'.format(i, n_spikes))
            spike_ns.append(n_spikes)
            i += 1
        spike_freqs = np.array(spike_ns) / self.baseline_duration
        return spike_freqs

    @cached_property
    def recording_start(self):
        """
        Returns the index of the fist non 0 point
        """
        for i in range(len(self.cmd)):
            if self.cmd[i] != 0:
                return i

    @cached_property
    def recording_end(self):   # TODO: check if used
        for i in range(len(self.cmd), 0, -1):
            if self.cmd[i] != 0:
                return i
    
    def analyse(self, do_spiking_difference=False, do_spiking_ratio=False):
        """
        Analyse (stats and plots) all matrices in self
        """
        for mat in self.matrices:
            mat.analyse(do_spiking_difference=do_spiking_difference, do_spiking_ratio=do_spiking_ratio)

    def write(self):
        """
        Save all matrices to csv
        """
        for mat in self.matrices:
            mat.save_binned_data(mat.matrix_name)
    
    def __prompt_id(self, keys):
        """
        Prompts user for protocols and validates response
        """
        exp_ids = list(range(len(keys)))
        while True:
            print('Experiments available:')
            for _id, key in zip(exp_ids, keys):
                print('\t{}: {}'.format(_id, key))
            prompt = "Please type in the number corresponding to the protocol: "
            exp_id = int(input(prompt))
            if exp_id in exp_ids:
                return exp_id
            else:
                'Please select a valid experiment id (from {})'.format(exp_ids)

    def _get_id(self, protocols):
        return [k for k in protocols if k.startswith('m')][0]

    def write_tables(self):
        csv_file_path = os.path.join(self.dir, '{}_clipped_traces.csv'.format(self.name))
        self.write_avgs_across_trials_table(csv_file_path)
        csv_file_path = os.path.join(self.dir, '{}_clipped_avgs_per_trial.csv'.format(self.name))
        self.write_avgs_per_trial_table(csv_file_path, self.clipped_avgs_per_trial_table)
        csv_file_path = os.path.join(self.dir, '{}_spiking_frequencies_per_trial.csv'.format(self.name))
        self.write_avgs_per_trial_table(csv_file_path, self.get_spiking_freq_lists_per_trial(), True)

    def write_avgs_per_trial_table(self, path, table, is_spiking=False):
        """
        Used for avg vm per trial and avg spiking per trial

        :param string path: The path to save the figure
        :param ODict table:
        :param bool is_spiking: Whether Vm or spiking data
        :return:
        """
        header = ('bsl', 'clockWise', 'cClockWise', 'cw_contra', 'ccw_contra', 'cw_ipsi', 'ccw_ipsi')
        for k in header:
            assert k in list(table.keys()), 'key {} not in {}'.format(k, list(table.keys()))
            table[k] = np.array(table[k])
        with open(path, 'w') as csv_file:

            # ALL TRIALS
            csv_file.write('\t'.join(header) + '\n')
            for elements in zip(*[table[k] for k in header]):  # REFACTOR: rename elements
                csv_file.write('{},{},{},{},{},{},{}\n'.format(*elements))  # trial1 cycle1, t1c2, t2c1, t2c2, ... tnc1, tnc2
            csv_file.write('\n')

            averages = [table[k].mean() for k in header]
            sds = [table[k].std() for k in header]
            csv_file.write('Mean:\n{},{},{},{},{},{},{}\n'.format(*averages))
            csv_file.write('Delta:\n{},{},{},{},{},{},{}\n\n'.format(*self.get_deltas(*averages)))
            csv_file.write('SD:\n{},{},{},{},{},{},{}\n\n'.format(*sds))

            # STATS
            csv_file.write('Stats (Wilcoxon signed-rank):\n')
            import warnings
            warnings.filterwarnings('ignore')
            csv_file.write('baseline/clockwise,p-value:,{}\n'.
                           format(self.wilcoxon_test(table[header[0]], table[header[1]])))
            csv_file.write('clockwise/cClockWise,p-value:,{}\n'.
                           format(self.wilcoxon_test(table[header[1]], table[header[2]])))
            csv_file.write('baseline/cClockwise,p-value:,{}\n'.
                           format(self.wilcoxon_test(table[header[0]], table[header[2]])))
            csv_file.write('baseline/cw_contra,p-value:,{}\n'.
                           format(self.wilcoxon_test(table[header[0]], table[header[3]])))
            csv_file.write('baseline/ccw_contra,p-value:,{}\n'.
                           format(self.wilcoxon_test(table[header[0]], table[header[4]])))
            csv_file.write('baseline/cw_ipsi,p-value:,{}\n'.
                           format(self.wilcoxon_test(table[header[0]], table[header[5]])))
            csv_file.write('baseline/ccw_ipsi,p-value:,{}\n'.
                           format(self.wilcoxon_test(table[header[0]], table[header[6]])))
            warnings.filterwarnings('error')
            csv_file.write('\n')

            # DSI
            if is_spiking:
                dsi = self.get_dsi(*self.get_spiking_frequencies())
                self.dsi = dsi  # WARNING: set outside __init__
            else:
                dsi = self.get_dsi(self.bsl_clipped_baselined_mean.mean(),
                                   self.clock_wise_clipped_baselined_mean.mean(),
                                   self.c_clock_wise_clipped_baselined_mean.mean())
            csv_file.write('\n')
            csv_file.write('DSI:,{}\n'.format(dsi))

    def write_avgs_across_trials_table(self, path, sep=','):
        with open(path, 'w') as csv_file:
            columns = self.get_clipped_cycles()

            cycles_header = sep.join(['cycle{}'.format(i) for i in range(len(columns))])
            csv_file.write(cycles_header + sep + 'average\n')

            for l in zip(*columns):  # transpose columns to lines
                csv_file.write(sep.join([str(p) for p in l]) + sep + str(np.mean(l)))
                csv_file.write('\n')
            csv_file.write('\n')

    @property
    def duration(self):
        return self.sampling * self.cmd.shape[0]

    def resample_matrices(self):
        """
        Resample the data in position, velocity or acceleration to get even representation of each degree, degree/s...

        :return:
        """

        self.position_spiking, self.position_durations = self.normalise_spiking('degreesLocs')  # FIXME: investigate
        self.velocity_spiking, self.velocity_durations = self.normalise_spiking('velocitiesLocs')
        self.acceleration_spiking, self.acceleration_durations = self.normalise_spiking('accelerationsLocs')

        stats_path = self._init_stats_file()
        vm_mat = ResampledMatrix(self, 'normalisedMatrix', self.ext, stats_path)
        spiking = ResampledMatrix(self, 'rasterMatrix', self.ext, stats_path)
        vm_vel_mat = ResampledMatrix(self, 'velocityNormalisedMatrix', self.ext, stats_path)
        spiking_vel_mat = ResampledMatrix(self, 'velocityRasterMatrix', self.ext, stats_path)
        vm_acc_mat = ResampledMatrix(self, 'accelerationNormalisedMatrix', self.ext, stats_path)
        spiking_acc_mat = ResampledMatrix(self, 'accelerationRasterMatrix', self.ext, stats_path)
        self.matrices = (vm_mat, spiking, vm_vel_mat, spiking_vel_mat, vm_acc_mat, spiking_acc_mat)

    def _init_stats_file(self):
        stats_path = os.path.join(self.dir, '{}_stats.txt'.format(self.name))
        with open(stats_path, 'w') as out_file:  # Ensure file is empty
            out_file.write('')
        return stats_path

