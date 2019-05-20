import math
import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from margrie_libs.utils.print_utils import dprint

from shuffle_anova_mateo import analyse_split, DIRECTION_NUMS
from signal_processing.mat_utils import linearise, get_uniques


class ExperimentResampler(object):
    def __init__(self, exp):
        self.exp = exp

    def _normalise_trace_sampling(self, sweep, cmd_levels_locs):
        """
        Pick values in sweep at cmd_levels_locs time points (using self.exp.sampling to convert)
        interpolating as needed

        :param np.array sweep: The vms values to pick from
        :param cmd_levels_locs: # in time (ms) absolute from sweep start. Shape is n_levels, n_half_cycles
        :return:
        """
        n_levels, n_half_cycles = cmd_levels_locs.shape[:2]

        normalised_matrix = np.zeros((n_levels, n_half_cycles))
        max_time_ms = len(sweep) * self.exp.sampling * 1000  # WARNING: in ms because of NeuroMatic
        time_axis = np.linspace(0, max_time_ms, len(sweep))
        for i in range(n_half_cycles):
            level_times = cmd_levels_locs[:, i]
            normalised_matrix[:, i] = np.interp(level_times, time_axis, sweep)
        return normalised_matrix

    def _get_velocity_matrix(self):
        return self.__get_matrix('velocities', 'velocitiesLocs')

    def _get_acceleration_matrix(self):
        return self.__get_matrix('accelerations', 'accelerationsLocs')

    def __get_matrix(self, ref_var_name, locs_name, test=False):
        ref_var = self.exp.data[ref_var_name]
        ref_var_locs = self.exp.data[locs_name]  # in ms absolute from sweep start

        data = self.exp.raw_clipped_baselined  # WARNING: computed on raw_clipped_baselined

        n_trials = len(data)
        # n_repeats_per_trial = how many times we hit the same velocity or acc or position value in trimmed sweep
        n_repeats_per_trial = self.exp.n_segments * 2  # 2 times the value per full cycle
        n_ref_vars = len(ref_var)
        matrix = np.zeros((n_trials, n_ref_vars, n_repeats_per_trial))  # TODO: maybe stack 2nd cycle as new sweep

        for i, sweep in enumerate(data):
            if test:
                matrix[i, :, :] = np.vstack([ref_var]*n_repeats_per_trial).T
            else:
                matrix[i, :, :] = self._normalise_trace_sampling(sweep, ref_var_locs)
        return ref_var, matrix

    # FIXME: rename with velocity and derive from generalised version
    def shuffle_stats(self, velocities, velocity_matrix_collapsed, direction_mat_collapsed, shuffle=True):
        self.make_velocity_df(velocity_matrix_collapsed)

        if shuffle:
            dprint("Computing velocities permutations")
            fig_path = os.path.join(self.exp.dir, "{}_resampled_distributions.{}".format(self.exp.name, self.exp.ext))
            plt.savefig(fig_path)  # FIXME: maybe part of plotter

            print("Velocity")
            f, signif_ccw, signif_cw, signif_r2_ccw, signif_r2_cw = analyse_split(velocities, velocity_matrix_collapsed,
                                                     direction_mat_collapsed,
                                                     self.exp.name, 'velocity')
            self.exp.significant_counter_clockwise = signif_ccw
            self.exp.significant_clockwise = signif_cw

            self.exp.significant_r2_counter_clockwise = signif_r2_ccw
            self.exp.significant_r2_clockwise = signif_r2_cw

            fig_path = os.path.join(self.exp.dir, "{}_split_velocity_plot.{}".format(self.exp.name, self.exp.ext))
            plt.savefig(fig_path)
            plt.close()

    def shuffle_stats_acceleration(self, accelerations, acceleration_matrix_collapsed, direction_mat_collapsed, shuffle=True):
        # self.make_acceleration_df(acceleration_matrix_collapsed)

        if shuffle:
            dprint("Computing accelerations permutations")
            fig_path = os.path.join(self.exp.dir, "{}_acceleration_resampled_distributions.{}".
                                    format(self.exp.name, self.exp.ext))
            plt.savefig(fig_path)  # FIXME: maybe part of plotter

            analyse_split(accelerations, acceleration_matrix_collapsed,
                          direction_mat_collapsed, self.exp.name, 'acceleration')

            fig_path = os.path.join(self.exp.dir, "{}_split_acceleration_plot.{}".format(self.exp.name, self.exp.ext))
            plt.savefig(fig_path)
            plt.close()

    def _average_acceleration_matrix(self, accelerations, acceleration_matrix):
        return self._average_matrix("acceleration", accelerations, acceleration_matrix)

    def _average_velocity_matrix(self, velocities, velocity_matrix):
        return self._average_matrix("velocity", velocities, velocity_matrix)

    def _average_matrix(self, ref_var_name, ref_var_unique_values, input_matrix):
        """
        Average by bins in 3 dimensions (bin_size, semi_cycles, trials) at once and get sd and se

        :param str ref_var_name: The name of the variable to study (position, velocity or acceleration)
        :param np.array ref_var_unique_values:
        :param np.ndarray input_matrix:
        :return: matrix_average, matrix_sd, matrix_se, bin_edges
        """
        if ref_var_name == 'velocity':
            ref_var_max = 80
        elif ref_var_name == 'acceleration':
            ref_var_max = 71
        else:
            raise ValueError("Unknown ref var {}".format(ref_var_name))
        bin_edges, n_bins, n_repeats, collapsed_mat, collapsed_direction_mat = self.collapse_matrix(ref_var_max,
                                                                                                    ref_var_unique_values,
                                                                                                    input_matrix)
        # collapsed_mat has shape: n_ref_var_values, n_repeats

        # BIN AND AVERAGE
        # n_micro_bins_per_bin = int(input_matrix.shape[1]/n_bins) - 3  # FIXME: see why
        # if n_micro_bins_per_bin < 1:  # FIXME: see above
        #     n_micro_bins_per_bin = 1
        # n_samples = n_micro_bins_per_bin * n_repeats
        if n_bins == collapsed_mat.shape[1]:  # bin size of 1
            matrix_average = collapsed_mat.mean(axis=1)
            matrix_sd = collapsed_mat.std(axis=1)
        else:
            matrix_average = np.zeros(n_bins)
            matrix_sd = np.zeros(n_bins)
            for i in range(n_bins):
                # micro_bins_mask = (main_dimension_unique_values >= bin_edges[i]) & (main_dimension_unique_values < bin_edges[i+1])
                micro_bins_mask = ref_var_unique_values == ref_var_unique_values[i]
                bin_data = collapsed_mat[micro_bins_mask, :].copy()
                # bin_data_str = ','.join([str(d) for d in bin_data])  # OPTIMISE: slow bit
                # csv_file.write("{},{},{}\n".format(i, ref_var_unique_values[i], bin_data_str))  # TODO: extract

                matrix_average[i] = bin_data.mean()
                matrix_sd[i] = bin_data.std()
        matrix_se = matrix_sd / (math.sqrt(n_repeats))

        tmp_df = pd.DataFrame({
            'i': np.arange(n_bins),
            ref_var_name: ref_var_unique_values
        })
        for repeat in range(n_repeats):
            tmp_df['s{}'.format(repeat)] = collapsed_mat[:, repeat]
        csv_path = os.path.join(self.exp.dir, '{}_vm_vs_{}_detail.{}'.format(self.exp.name, ref_var_name, 'csv'))
        tmp_df.to_csv(csv_path)

        if ref_var_name == 'velocity':
            self.shuffle_stats(ref_var_unique_values, collapsed_mat, collapsed_direction_mat)
        elif ref_var_name == 'acceleration':
            self.shuffle_stats_acceleration(ref_var_unique_values, collapsed_mat, collapsed_direction_mat)
        return matrix_average, matrix_sd, matrix_se, bin_edges

    def collapse_matrix(self, max_ref_var_value, ref_var_unique_values, input_matrix):
        """
        Collapse the trials and repeats dimensions of the input matrix into one

        :param float max_ref_var_value:
        :param np.array ref_var_unique_values:
        :param np.ndarray input_matrix:
        :return: bin_edges, n_bins, n_repeats, collapsed_matrix
        """
        n_bins = input_matrix.shape[1]
        bin_edges = np.linspace(-max_ref_var_value, max_ref_var_value, n_bins + 1)  # WARNING: 1 more since edges
        n_repeats = input_matrix.shape[0] * input_matrix.shape[2]  # number of sample for the same ref_var value

        direction_mat = np.full(input_matrix.shape, DIRECTION_NUMS['counter_clockwise'], np.int64)
        for i in range(0, input_matrix.shape[2], 2):
            direction_mat[:, :, i] = DIRECTION_NUMS['clockwise']
        # FIXME: see why len(ref_var_unique_values) and not n_bins
        collapsed_matrix = input_matrix.view().transpose(1, 0, 2).reshape(len(ref_var_unique_values), n_repeats)
        collapsed_direction_matrix = direction_mat.view().transpose(1, 0, 2).reshape(n_bins, n_repeats)
        return bin_edges, n_bins, n_repeats, collapsed_matrix, collapsed_direction_matrix

    def make_baseline_velocity_df(self, frame, trials, velocities):
        return self.make_baseline_variable_df(frame, trials, velocities, ref_var_name='velocity')

    def make_baseline_acceleration_df(self, frame, trials, accelerations):
        return self.make_baseline_variable_df(frame, trials, accelerations, ref_var_name='acceleration')

    def make_baseline_variable_df(self, frame, trials, variable_values, ref_var_name):
        """

        :param pd.DataFrame frame:
        :param trials:
        :param np.array variable_values: velocities or accelerations... but form DF (with repeats)
        :param str ref_var_name:
        :return:
        """
        # BASELINE
        n_pts_movement = int(len(frame))
        n_pnts_bsl = int(n_pts_movement / 2)  # length of 1 direction
        baseline_data = np.array([self.exp.extract_bsl(trace) for trace in self.exp.raw_clipped_baselined])
        baseline_data = np.random.choice(linearise(baseline_data), n_pnts_bsl)  # To match point number of data

        unique_variable_values = np.array(get_uniques(variable_values), dtype=np.int64)
        n_variable_repeats = int(n_pnts_bsl / len(unique_variable_values))
        bsl_variable = np.repeat(unique_variable_values, n_variable_repeats)

        bsl_directions = np.full(n_pnts_bsl, 'none', dtype='<U4')
        unique_trials = np.array(get_uniques(trials), dtype=np.int64)
        trial_repeats = int(n_pnts_bsl / len(unique_trials))
        baseline_trials = np.tile(unique_trials, trial_repeats)

        # same number of points as 1 direction, direction='none'
        baseline_df = pd.DataFrame({ref_var_name: bsl_variable,
                                    'direction': bsl_directions,
                                    'vm': baseline_data,
                                    'trial': baseline_trials})
        return baseline_df

    def make_velocity_df(self, velocity_matrix_collapsed):
        self.exp.frame = self.make_df(velocity_matrix_collapsed, 'velocity')  # FIXME: replace by velocity_frame

    def make_acceleration_df(self, acceleration_matrix_collapsed):
        self.exp.acceleration_frame = self.make_df(acceleration_matrix_collapsed, 'acceleration')

    def make_df(self, collapsed_matrix, ref_var_name='velocity', test=False):
        """
        Assumes collapsed_matrix with monotonously increasing ref_var values on axis 0

        .. warning:: uses fact that velocity+ == CW, velocity- == CCW

        :param np.array collapsed_matrix: shape = (n_unique_ref_var_values, n_repeats)
        :param str ref_var_name: one of ('velocity', 'position', 'acceleration')
        :param bool test:
        :return: frame_with_baseline (column ref_var_name = []) # one of 'velocities', 'positions', 'accelerations'
        """
        dprint("Making {} DataFrame".format(ref_var_name))

        matrix_mid = int(math.ceil(collapsed_matrix.shape[0] / 2))
        directions, vms, reference_var, trials = [], [], [], []
        # Remove sign of reference_var put in direction
        for i in range(0, matrix_mid - 1):  # step half from middle
            for step, direction in zip((+1, -1), ('clockwise', 'counter_clockwise')):  # step positive and negative each time
                submat = collapsed_matrix[matrix_mid + i * step].copy()  # take column that matches step  # WARNING: implicit [mid, :]
                n_trials = len(submat)
                directions.extend([direction] * n_trials)
                reference_var.extend([i] * n_trials)
                trials.extend(list(range(n_trials)))
                if test:
                    submat = [random.random() for i in range(n_trials)]
                    if step == -1:
                        submat = [s+10 for s in submat]
                vms.extend(submat)  # WARNING: this is an approximation (1/2 sampling points)
        frame = pd.DataFrame({ref_var_name: np.array(reference_var, dtype=np.int64),
                              # TODO: check if OK or better to double positive_velocities[0]
                              'direction': np.array(directions),
                              'vm': np.array(vms, dtype=np.float64),
                              'trial': np.array(trials, dtype=np.int64)})

        baseline_df = self.make_baseline_variable_df(frame, trials, reference_var, ref_var_name=ref_var_name)
        frame_with_baseline = pd.concat((frame, baseline_df))
        csv_path = os.path.join(self.exp.dir, '{}_vm_vs_{}_data_frame.{}'.format(self.exp.name, ref_var_name, 'csv'))
        frame_with_baseline.to_csv(csv_path)
        return frame_with_baseline

