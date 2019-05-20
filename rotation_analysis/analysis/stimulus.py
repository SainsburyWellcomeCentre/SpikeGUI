import os
import warnings

import numpy as np
from cached_property import cached_property
from configobj import ConfigObj
from margrie_libs.margrie_libs.signal_processing import mat_utils
from margrie_libs.margrie_libs.utils.print_utils import print_rule
from scipy import io

from rotation_analysis.resample import resampling
from rotation_analysis.resample.resampling import make_time_table

DEGREES_PER_VOLT = 20


class MissingAttributeError(Exception):
    pass


class Stimulus(object):

    def matches_attributes(self, attributes_dict):
        print('checking stimulus parent attibutes')

        for k, v in attributes_dict.items():
            if not hasattr(self, k):
                raise MissingAttributeError()
            if getattr(self, k) != v:
                return False
        return True


class RotationStimulus(Stimulus):  # FIXME: some functionality is only defined in the subclass i.e. sampling_interval
    default_config_path = 'C:\\Work\\EPhys\\Code\\Git Respositories\\.sepi_analysis.conf'
    # default_config_path = os.path.normpath(os.path.expanduser('~/.sepi_analysis.conf'))
    default_conf = ConfigObj(default_config_path, encoding="UTF8", indent_type=' ' * 4, unrepr=True,
                             create_empty=True, write_empty_values=True)
    default_conf.reload()

    position_spacing = default_conf['binning']['position']  # 15
    velocity_spacing = default_conf['binning']['velocity']  # 20
    acceleration_spacing = default_conf['binning']['acceleration']  # 5

    condition_pairs = (  # TODO: extract to cfg
        ('bsl_short', 'spin'),
        ('bsl2', 'spin'),
        ('bsl_short', 'bsl2'),
        ('bsl_short', 'c_wise'),
        ('bsl_short', 'c_c_wise'),
        ('c_wise', 'c_c_wise'),
        ('bsl_short', 'c_wise_long'),
        ('bsl_short', 'c_c_wise_long'),
        ('c_wise_long', 'c_c_wise_long')
    )

    @staticmethod
    def get_spacing_from_ref_var_name(ref_var_name):
        spacing_dictionnary = {
            'position': RotationStimulus.position_spacing,
            'distance': RotationStimulus.position_spacing,
            'velocity': RotationStimulus.velocity_spacing,
            'acceleration': RotationStimulus.acceleration_spacing
        }
        return spacing_dictionnary[ref_var_name]

    def __init__(self, use_bsl_2, min_range_n_pnts=10):
        self.use_bsl_2 = use_bsl_2
        self.n_repeats = self.get_n_repeats()
        self.min_range_n_points = min_range_n_pnts

    def get_n_repeats(self):
        """
        Gets the number of repeats of values which corresponds in this case to the number of full cycles

        .. warning::
            Makes assumptions on cmd shape

        :return:
        """
        return len(self.get_cmd_peaks()) - 1

    def get_spin_start(self):  # FIXME: make more robust
        """
        Start point of spin period
        """
        for i, p in enumerate(self.cmd):
            if p != 0:
                return i

    def get_spin_end(self):  # FIXME: make more robust
        """
        End point of spin period
        """
        #         if cmd[-1] != 0:
        # #            raise ValueError("No end baseline detected for cell {}".format(self.id))
        for i in reversed(range(len(self.cmd))):
            if self.cmd[i] != self.cmd[-1]:
                return i

    @staticmethod
    def get_sine_peaks(sine):
        return np.array(mat_utils.find_sine_peaks(sine))

    def get_cmd_peaks(self):
        return CalciumRotationStimulus.get_sine_peaks(self.cmd)

    def get_bsl_ranges(self):
        """
        Return 2 tuples of start,end for the first and second baseline periods
        Range in points
        """
        bsl1_range_start = 0
        bsl1_range_end = self.get_spin_start() - 1

        bsl_ranges = [(bsl1_range_start, bsl1_range_end), ]
        if self.use_bsl_2:
            bsl_ranges.append(self.get_bsl_2_ranges()[0])
        print_rule()
        print("Normal baseline ranges: {}".format(np.array(bsl_ranges) * self.sampling_interval))
        print_rule()
        return bsl_ranges

    def get_bsl_2_ranges(self):
        bsl2_range_start = self.get_spin_end() + 1
        bsl2_range_end = len(self.cmd) - 1
        print_rule()
        print("baseline 2 ranges: {}".format((bsl2_range_start * self.sampling_interval, bsl2_range_end * self.sampling_interval)))
        print_rule()
        return ((bsl2_range_start, bsl2_range_end), )

    def get_bsl_short_ranges(self, other_period):  # For ranges and PSTH
        bsl_short_end = self.get_spin_start()
        other_period_n_pnts = sum([(end - start) for start, end in self.get_ranges_by_type(other_period)])
        bsl_short_start = bsl_short_end - other_period_n_pnts
        bsl_short_start = bsl_short_start if bsl_short_start >= 0 else 0
        print_rule()
        print("baseline short for {} ranges: {}".format(other_period, (bsl_short_start * self.sampling_interval, bsl_short_end * self.sampling_interval)))
        print_rule()
        return ((bsl_short_start, bsl_short_end), )

    def get_spin_ranges(self):
        """
        Range in points of the spinning period (ramps included)

        .. warning:
            Returns tuple for interface consistency with other ranges methods
        """
        return (self.get_spin_start(), self.get_spin_end()),

    def get_spin_duration(self):
        """
        Duration of the spinning period (ramps included)
        :return:
        """
        spin_start, spin_end = self.get_spin_ranges()[0]
        return (spin_end - spin_start) * self.sampling_interval

    def get_full_cycles_spin_n_pnts(self):
        ranges = self.get_full_cycles_spin_range()
        start, end = ranges[0]
        return end - start

    @staticmethod
    def _get_full_cycles_spin_range(sine):
        peaks_list = CalciumRotationStimulus.get_sine_peaks(sine)
        start = peaks_list[0]
        end = peaks_list[-1]
        return ((start, end), )

    def get_full_cycles_spin_range(self):  # TEST:
        """
        The start and end points of the part of the spinning that is analysed for cwise/ccwise (removes ramps)

        :return:
        """
        return CalciumRotationStimulus._get_full_cycles_spin_range(self.cmd)

    def get_ranges_duration(self, ranges):  # WARNING: private
        return sum([(end - start) * self.sampling_interval for start, end in ranges])

    # def get_c_wise_duration(self):  # TEST:
    #     return self.get_ranges_duration(self.__get_clock_wise_ranges())

    def get_one_cycle_n_pnts(self):  # WARNING: Unused
        return int(self.get_full_cycles_spin_n_pnts() / self.n_repeats)

    def __find_range_starts_and_ends(self, src_mask):
        """
        For a binary mask of the form:
        (0,0,0,1,0,1,1,1,0,0,0,1,1,0,0,1)
        returns:
        (0,0,0,1,0,1,0,1,0,0,0,1,1,0,0,1)
        """
        output_mask = np.hstack(([src_mask[0]], np.diff(src_mask)))  # reintroduce first element
        starts = np.where(output_mask == 1)[0] + self.get_spin_start()
        ends = np.where(output_mask == -1)[0] + self.get_spin_start()
        return starts, ends

    def __get_velocity_during_spin(self):
        velocity_during_spin = self.velocity[self.get_spin_start():self.get_spin_end()]
        return velocity_during_spin

    def __get_diffed_velocity_during_spin(self):
        velocity_during_spin = self.diffed_velocity[self.get_spin_start():self.get_spin_end()]
        return velocity_during_spin

    def __find_clock_wise_ranges(self, vel):
        mask = np.array(vel < 0, dtype=np.int64)
        mask[-1] = 0
        mask[0] = 0
        return self.__find_range_starts_and_ends(mask)

    def __find_counter_clock_wise_ranges(self, vel):
        mask = np.array(vel > 0, dtype=np.int64)
        mask[-1] = 0  # FIXME: the problem is that the trace never fully starts at or reaches 0
        mask[0] = 0
        return self.__find_range_starts_and_ends(mask)

    def __get_clock_wise_ranges(self):  # FIXME: seems to also get counter clockwise
        """
        Values in points

        :return:
        """
        velo = self.__get_velocity_during_spin()
        starts, ends = self.__find_clock_wise_ranges(velo)
        assert len(starts) == len(ends), 'Starts and ends length must match, got {} and {}'.format(len(starts), len(ends))
        ranges = []
        for start, end in zip(starts, ends):
            ranges.append([start, end])
        return ranges

    def __get_counter_clock_wise_ranges(self):
        starts, ends = self.__find_counter_clock_wise_ranges(self.__get_velocity_during_spin())
        assert len(starts) == len(ends), 'Starts and ends length must match, got {} and {}'.format(len(starts),
                                                                                                   len(ends))
        ranges = []
        for start, end in zip(starts, ends):
            ranges.append([start, end])
        return ranges

    def __get_clock_wise_long_ranges(self):
        """
        Values in points

        :return:
        """

        diffed_velocity_during_spin = self.__get_diffed_velocity_during_spin()  # FIXME: this never returns to exactly 0
        starts, ends = self.__find_clock_wise_ranges(diffed_velocity_during_spin)
        assert len(starts) == len(ends), 'Starts and ends length must match, got {} and {}'.format(len(starts),
                                                                                                   len(ends))
        ranges = []
        for start, end in zip(starts, ends):
            ranges.append([start, end])
        return ranges

    def __get_counter_clock_wise_long_ranges(self):
        starts, ends = self.__find_counter_clock_wise_ranges(self.__get_diffed_velocity_during_spin())
        assert len(starts) == len(ends), 'Starts and ends length must match, got {} and {}'.format(len(starts),
                                                                                                   len(ends))
        ranges = []
        for start, end in zip(starts, ends):
            ranges.append([start, end])
        return ranges

    def get_ranges_by_type(self, range_type, range_type2=None):
        expected_range_types = ('bsl', 'bsl2', 'bsl_short', 'spin', 'full_cycles',
                                'c_wise', 'c_c_wise', 'c_wise_long', 'c_c_wise_long')
        if range_type not in expected_range_types:
            raise ValueError('Range must be one of {}, got "{}"'.format(expected_range_types, range_type))

        range_functions = {
            'bsl': self.get_bsl_ranges,
            'bsl2': self.get_bsl_2_ranges,
            # 'bsl_short': self.get_bsl_short_ranges,
            'full_cycles': self.get_full_cycles_spin_range,
            'spin': self.get_spin_ranges,
            'c_wise': self.__get_clock_wise_ranges,
            'c_c_wise': self.__get_counter_clock_wise_ranges,
            'c_wise_long': self.__get_clock_wise_long_ranges,
            'c_c_wise_long': self.__get_counter_clock_wise_long_ranges
        }
        if range_type == 'bsl_short':
            ranges = self.get_bsl_short_ranges(range_type2)
        else:
            ranges = range_functions[range_type]()
        ranges = [rng for rng in ranges if (rng[1] - rng[0]) > self.min_range_n_points]
        return ranges

    def __make_timetable(self, cmd, spacing):
        """

        :param cmd: one of cmd, velocity, acceleration
        :param spacing: the binning of the above
        :return: The timetable for the given command in seconds
        """
        _min = cmd.min()
        _max = cmd.max()
        if abs(_max) - abs(_min) > abs(_max - _min) / 500:
            warnings.warn('Command not symmetric around 0, got min: {0:.3f}, max: {1:.3f}'.format(_min, _max))
        n_levels = int(round((_max - _min) / spacing))
        # levels controls the binning of the timetable
        epsilon = 0.001  # WARNING: may need to adjust epsilon
        levels = np.linspace(_min+epsilon, _max-epsilon, n_levels + 1)

        start_p, end_p = self.get_full_cycles_spin_range()[0]
        n_crossings = self.n_repeats  # check if this needs to be calculated from cmd inputs
        timetable = make_time_table(cmd, self.x, levels, n_crossings, start_p, end_p)
        return levels, timetable

    def _get_timetable(self, levels_var_name):
        timetables_functions = {
            'position': self.position_timetable,
            'distance': self.get_distance_timetable,
            'velocity': self.velocity_timetable,
            'acceleration': self.acceleration_timetable
        }
        levels, levels_timetable = timetables_functions[levels_var_name]()
        cmd = self.get_ref_var(levels_var_name)
        return levels, levels_timetable, cmd, self.x

    def get_ref_var(self, ref_var_name):
        cmd_functions_dict = {
            'position': self.cmd,
            'distance': self.cmd,
            'velocity': self.velocity,
            'acceleration': self.acceleration
        }
        return cmd_functions_dict[ref_var_name]

    def position_timetable(self):
        return self.__make_timetable(self.cmd, RotationStimulus.position_spacing)

    def velocity_timetable(self):
        return self.__make_timetable(self.velocity, RotationStimulus.velocity_spacing)

    @cached_property
    def acceleration(self):
        start_p, end_p = self.get_full_cycles_spin_range()[0]
        acceleration = resampling.get_acceleration_from_position(self.cmd, self.x, start_p, end_p)
        return acceleration

    def acceleration_timetable(self):
        return self.__make_timetable(self.acceleration, RotationStimulus.acceleration_spacing)

    def get_distance_timetable(self):
        degrees, positions_timetable = self.position_timetable()
        distances = degrees + abs(degrees.min())
        distances_table = np.sort(positions_timetable.copy(), axis=0)  # TEST:
        return distances, distances_table

    @cached_property
    def velocity(self):
        start_p, end_p = self.get_full_cycles_spin_range()[0]
        velocity = resampling.get_velocity_from_position(self.cmd, self.x, start_p, end_p)
        return velocity

    @cached_property
    def diffed_velocity(self):
        return np.diff(self.cmd) / self.sampling_interval

    @property
    def x(self):  # TODO: if sampling interval is too big
        n_pnts_cmd = len(self.cmd)
        t_max = self.sampling_interval * n_pnts_cmd
        cmd_x = np.linspace(0, t_max, n_pnts_cmd)
        return cmd_x

    def get_command(self, arg):
        raise NotImplementedError

    # def get_bsl2_short_ranges(self, other_period):  # For PSTH
    #     bsl_short_start = self.get_spin_end()
    #     other_period_n_pnts = sum([(end - start) for start, end in self.get_ranges_by_type(other_period)])
    #     bsl_short_end = bsl_short_start + other_period_n_pnts
    #     bsl_short_end = min(bsl_short_end, len(self.cmd))
    #     print_rule()
    #     print("baseline 2 for {} short ranges: {}".format(other_period, (bsl_short_start * self.sampling_interval, bsl_short_end * self.sampling_interval)))
    #     print_rule()
    #     return ((bsl_short_start, bsl_short_end), )

    # def get_bsl_duration(self):
    #     return self.get_ranges_duration(self.get_bsl_ranges())

    # TODO: implement bsl2_short


class CalciumRotationStimulus(RotationStimulus):
    def __init__(self, rec, angle, use_bsl_2=False, min_range_n_pnts=10):
        """

        :param calcium_recordings.recording.Recording rec:
        """
        self.rec_dir = rec.dir
        self.metadata = self.get_metadata(rec)
        self.cmd = self.get_command(self.rec_dir)
        self.angle = angle

        super(CalciumRotationStimulus, self).__init__(use_bsl_2, min_range_n_pnts=min_range_n_pnts)

    def get_metadata(self, rec):
        return rec.vars

    @property
    def is_cw_first(self):
        return self.get_cmd_peaks()[0] > 0

    def get_command(self, rec_dir):  # FIXME: needs to change with Igor
        """
        The position command of the motor
        """
        mat_path = os.path.join(rec_dir, 'image_ticks.mat')
        mat = io.loadmat(mat_path)
        return mat['tmpPosList'][0]

    @property
    def sampling_interval(self):
        """
        The sampling interval in seconds

        Same for all angles
        """
        return self.metadata['frame.duration']

    def get_n_points(self):
        """
        n lines

        :return:
        """
        return self.metadata['frame.count']

