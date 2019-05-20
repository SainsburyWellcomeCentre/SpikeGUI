import numpy as np
from scipy.optimize import leastsq, curve_fit
from tqdm import trange

from matplotlib import pyplot as plt
from margrie_libs.margrie_libs.signal_processing.mat_utils import find_sine_peaks


def crossing_idx(trace, a, b):
    _crossing_idx = np.where(np.logical_or(np.logical_and(a <= trace, trace < b),
                                           np.logical_and(a >= trace, trace > b)))[0]
    if len(_crossing_idx) > 0:
        return _crossing_idx[0]
    else:
        return None


class ResamplingError(Exception):
    pass


def make_time_table(cmd, cmd_x, degrees, n_crossings, start_p=0, end_p=-1):
    output = np.full((len(degrees), n_crossings), np.nan)
    first_found = False
    if start_p != 0:
        start_p -= 1
    if end_p == -1:
        end_p = len(cmd)
    for i in trange(len(cmd) - 1):  # Loop over command to loop only once (instead of searching it for each crossing)
        if not (start_p <= i < end_p):
            continue
        a, b = cmd[i:i + 2]
        if a == b:
            continue
        if first_found:
            degrees_range = degrees[idx_of_degree - 1: idx_of_degree + 2]
        else:
            degrees_range = degrees
        if len(degrees_range) == 0:  # FIXME: check why disappears
            degrees_range = degrees
            first_found = False
            # raise ValueError('Missing degrees_range for index {} (x={}, y={})'.format(i, x[i], cmd[i]))
        relative_idx_in_degrees = crossing_idx(degrees_range, a, b)
        if relative_idx_in_degrees is not None:
            if not first_found:
                idx_of_degree = np.searchsorted(degrees, degrees_range[relative_idx_in_degrees])
                first_found = True
            else:
                idx_of_degree += relative_idx_in_degrees - 1
            for j in range(n_crossings):  # OPTIMISE: could be optimised to check only current range
                if np.isnan(output[idx_of_degree][j]):  # Find the crossing which hasn't been filled yet
                    current_degree = degrees[idx_of_degree]
                    sub_y = cmd[i: i + 2]  # (a, b)
                    sub_x = cmd_x[i: i + 2]
                    val_time = np.interp(current_degree, sub_y, sub_x)
                    output[idx_of_degree, j] = val_time
                    break  # otherwise, will fill multiple times with the same value
    if np.isnan(output).any():
        plt.plot(cmd_x[start_p:end_p], cmd[start_p:end_p])
        plt.show()
        raise ResamplingError('There should be no NaN left in the matrix')
    return output


def get_velocity_from_position(cmd, cmd_x, start_p=0, end_p=-1):
    """
    derives velocity from a position trace. NOTE: strange behaviour at edges

    :param cmd:
    :param cmd_x:
    :param start_p:
    :param end_p:
    :return:
    """
    params = fit_sine_wave(cmd_x[start_p:end_p], cmd[start_p:end_p])

    def vel_func(t):
        return params['amp'] * params['omega'] * np.cos(params['omega'] * t + params['phase'])  # cos is the derivative

    if start_p != 0:  # 1 point extra to find the values
        start_p -= 1
    if end_p != -1:
        end_p += 1
    tmp_velocity = vel_func(cmd_x[start_p:end_p])
    velocity = np.zeros(len(cmd))
    velocity[start_p:end_p] = tmp_velocity
    return velocity


def get_acceleration_from_position(cmd, cmd_x, start_p=0, end_p=-1):
    params = fit_sine_wave(cmd_x[start_p:end_p], cmd[start_p:end_p])

    def acc_func(t):
        return - params['amp'] * params['omega'] ** 2 * np.sin(
            params['omega'] * t + params['phase'])  # -sin is the 2nd derivative

    if start_p != 0:
        start_p -= 1
    if end_p != -1:
        end_p += 1
    tmp_acc = acc_func(cmd_x[start_p:end_p])
    acceleration = np.zeros(len(cmd))
    acceleration[start_p:end_p] = tmp_acc
    return acceleration


def fit_sine_wave(x_data, data):
    """Fit sin to the input time sequence, and return fitting parameters
    amp, omega, phase, offset, freq, period and fitfunc

    from https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
    """

    delta_x = x_data[1] - x_data[0]
    ff = np.fft.fftfreq(len(x_data), delta_x)  # assume uniform spacing
    Fyy = abs(np.fft.fft(data))

    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(data) * 2. ** 0.5
    guess_offset = np.mean(data)
    guessed_params = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    def sin_func(t, ampl, w, phase, offset):
        return ampl * np.sin(w * t + phase) + offset

    popt, pcov = curve_fit(sin_func, x_data, data, p0=guessed_params)
    ampl, w, phase, offset = popt
    f = w / (2. * np.pi)

    fitfunc = lambda t: ampl * np.sin(w * t + phase) + offset

    params = {'amp': ampl,
              'omega': w,
              'phase': phase,
              'offset': offset,
              'freq': f,
              'period': 1. / f,
              'fitfunc': fitfunc,
              'maxcov': np.max(pcov),
              'rawres': (guessed_params, popt, pcov)}

    return params


class TimeTableJumpError(object):
    pass


def find_discontinuities_in_the_matrix(mat, axis=0, median_threshold=2):
    diff = np.abs(np.diff(mat, axis=axis))
    median_diff = np.median(diff, axis=axis)
    discontinuities_locs = np.where(diff > median_threshold * median_diff)
    if discontinuities_locs:
        discontinuities_locs = discontinuities_locs[0]
        discontinuities_locs = [d for d in discontinuities_locs if d != 0]
        discontinuities_locs = [d for d in discontinuities_locs if d != mat.shape[axis] - 2]
        return discontinuities_locs
    else:
        return []


def count_discontinuities_in_the_matrix(mat, axis=0):
    return len(find_discontinuities_in_the_matrix(mat, axis=axis)) / mat.shape[1 - axis]


def find_zero_crossing_times(sine, x_axis):
    #return x_axis[find_sine_peaks(np.abs(np.diff(sine)))][1:-1]
    return x_axis[find_sine_peaks(np.abs(np.diff(sine)))][1:]


def fix_timetable(cmd, cmd_x, mat, t_min=0):  # WARNING assumes that as many missing in all columns
    discontinuities_locs = find_discontinuities_in_the_matrix(mat, axis=0)
    discontinuities_locs = list(set(discontinuities_locs))
    out_list = list(mat.copy())
    zero_crossings = find_zero_crossing_times(cmd, cmd_x)
    zero_crossings = [c for c in zero_crossings if c >= t_min]  # FIXME: Hacky AF

    out_list.insert(discontinuities_locs[0] + 1, np.array(zero_crossings))

    out_mat = np.array(out_list)
    # if __debug__:
    #     plt.plot(cmd_x, cmd)
    #     for j in range(out_mat.shape[0]):
    #         for i in range(out_mat.shape[1]):
    #             if i == 0 and j <= 4:
    #                 color = 'red'
    #             if i == 0 and j > 4:
    #                 color = 'blue'
    #             if i == 1 and j <= 4:
    #                 color = 'green'
    #             if i == 1 and j > 4:
    #                 color = 'magenta'
    #             plt.axvline(out_mat[j, i], color=color)
    #     plt.show()
    return out_mat
