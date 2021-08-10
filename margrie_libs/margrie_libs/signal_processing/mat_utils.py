"""
Collection of generic numpy array functions
"""
import math
import warnings

import numpy as np

from margrie_libs.margrie_libs.signal_processing.exceptions import BadRandomError, PeakDetectionError


def _get_decimate_new_n_pnts(trace, window_width, end_method):
    methods = ("drop", "strict", "pad")
    n_remaining_points = trace.size % window_width
    if end_method == 'strict' and n_remaining_points != 0:
        raise ValueError(
            "The decimation factor does not create an exact point numbers and you have selected 'strict'")
    elif end_method == 'drop':
        n_samples_last_window = 0
    elif end_method == 'pad':
        n_samples_last_window = n_remaining_points if n_remaining_points <= 2 else 2  # TODO: find better name for 'pad'
    else:
        raise ValueError("end_method should be one of {}, got {}".
                         format(methods, end_method))
    n_complete_windows = trace.size // window_width
    new_n_pnts = n_complete_windows * 2 + n_samples_last_window
    return new_n_pnts


def decimate(trace, decimation_factor=10, end_method="drop"):
    """
    Decimate (reduce the number of points) of the source trace to plot the trace.
    To preserve the visual aspect of the trace, the algorithm takes the min and max on a sliding window defined by
    decimation_factor.

    .. important:
        This function is intended for plotting only. For other uses, see more appropriate downsampling methods.

    :param trace: The trace to decimate
    :param int decimation_factor: the number X such that trace.size = X * out.size
    :param string end_method: How to deal with the last points
    :return: A decimated copy of the trace
    """

    if not isinstance(decimation_factor, int):
        raise TypeError("Decimation factor should be an integer number. Got {}.".format(decimation_factor))
    if decimation_factor < 1:
        raise ValueError("Decimation factor needs to be at least 1 to get a window of 2. Got {}.".
                         format(decimation_factor))

    window_width = decimation_factor * 2

    new_n_pnts = _get_decimate_new_n_pnts(trace, window_width, end_method)

    out = np.zeros(new_n_pnts)
    for i, j in enumerate(range(0, new_n_pnts, 2)):  # by 2 because 1 min and 1 max for each window
        window_start_p = i * window_width
        window_end_p = int(window_start_p + window_width)
        if window_start_p > trace.size:
            raise RuntimeError("Array {}, of size {}, iteration {}, from {} to {} ({} points)".
                               format(trace, trace.shape, i, window_start_p, window_end_p, window_width))
        if window_end_p > trace.size:
            window_end_p = -1
        segment = trace[window_start_p:window_end_p]
        out[j] = segment.min()
        try:
            out[j+1] = segment.max()
        except IndexError:  # If trace.size % window_width == 1
            break
    return out


def decimate_x(x_trace, decimation_factor=10, end_method="drop"):
    window_width = decimation_factor * 2
    new_n_pnts = _get_decimate_new_n_pnts(x_trace, window_width, end_method)
    return np.linspace(x_trace[0], x_trace[-1], new_n_pnts)  # FIXME: adjust not exactly x_trace[-1] because of drop


def find_sine_peaks_ranges(sine_trace):
    """
    Sine has to be zero centered
    """
    return abs(sine_trace) > (0.9 * sine_trace.max())


def find_sine_peaks(sine_trace):
    """
    Returns the indices (points) of the peaks
    Sine has to be zero centered
    """
    peak_ranges = find_sine_peaks_ranges(sine_trace)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        boundaries = np.diff(peak_ranges)
    boundaries_indices = (np.where(boundaries == True))[0]
    
    peak_starts = boundaries_indices[::2]
    peak_starts += 1  # add 1 because of diff
    peak_ends = boundaries_indices[1::2]
    peak_ends += 1  # add 1 because of diff
    
    peaks_pos = []
    for peak_start, peak_end in zip(peak_starts, peak_ends):
        peak = abs(sine_trace[peak_start:peak_end])  # abs because positive and negative peaks
        peak_pos = np.where(peak == peak.max())[0]   # because there may be several points at max
        if 0 in peak:
            raise PeakDetectionError('There should be no 0 in the peak, found {}'.format(peak))
        middle = int(math.floor(peak_pos.size / 2))
        if peak_pos.size % 2 == 0:
            middle -= 1
        peak_pos = peak_pos[middle]
        peak_pos += peak_start  # absolute position
        peaks_pos.append(peak_pos)
    return peaks_pos


def cut_and_avg_sine(sine_trace, trace, scaling=1):
    """
    sineTrace and trace must have same number of points
    Cut trace based on peaks of sine. to extract one period and averages all corresponding segments
    """
    # error_msg = "sineTrace and trace must have same number of points, got {} and {}"
    # .format(sineTrace.shape, trace.shape)
    # assert sineTrace.size == trace.size, error_msg
    segments = cut_and_get_multiple(sine_trace, trace, scaling=scaling)
    segments = np.array(segments, dtype=np.float64)
    return segments.mean(0)


def cut_and_sum_sine(sine_trace, trace, scaling=1):
    segments = cut_and_get_multiple(sine_trace, trace, scaling=scaling)
    segments = np.array(segments, dtype=np.float64)
    return segments.sum(0)


def cut_in_half(trace):
    middle = int(trace.size/2)
    first_half = trace[:middle]
    second_half = trace[middle:]
    if first_half.size != second_half.size:
        second_half = second_half[:-1]
    assert first_half.size == second_half.size, "Length of first half and second half differ: {} and {}".format(first_half.size, second_half.size)
    return first_half, second_half


def cut_and_avg_halves(trace):
    segments = cut_in_half(trace)
    return np.array(segments, dtype=np.float64).mean(0)


def cut_and_get_multiple(sine_trace, trace, scaling=1):
    """
    sineTrace and trace must have same number of points
    Cut trace based on peaks of sine. to extract one period and returns all corresponding segments

    .. warning:
        If the number of clockwise and counterclockwise segments differs, will only return the first N segments of each
        kind such that N = min(nClockWise, nCounterClockWise).
    """
    
    peaks_locs = np.array(find_sine_peaks(sine_trace), dtype=np.int64)  # full peaks ==> not the ramp peak
    peaks_locs *= scaling

    segments = []
    lengths = []
    for i in range(0, (peaks_locs.size - 1), 2):
        start_p = peaks_locs[i]
        try:
            end_p = peaks_locs[i+2]
        except IndexError:
            break
        segment = trace[start_p:end_p]
        lengths.append(segment.size)
        segments.append(segment)
    min_length = min(lengths)  # TODO: put criterion on max number of points diff
    segments = [s[:min_length] for s in segments]
    return segments


def avg(mat):
    """
    Returns the vector corresponding to mat averaged accross 2nd and 3rd dims.
    Assumes that the matrix is all filled (no NaN since avg of avg).
    """
    # if __debug__:
    #     print(mat.shape)
    if mat.ndim > 1:
        return avg(np.average(mat, axis=1))
    else:
        return mat


def avg_waves(waves):
    """
    Transforms the input list into a numpy array and returns the average across the first dimension

    :param list waves:
    :return:
    """
    matrix = np.array(waves)
    return matrix.mean(0)  # TODO: check dimension


def sd(mat):
    """
    Returns the vector corresponding to mat averaged accross 2nd and 3rd dims.
    Assumes that the matrix is all filled (no NaN since avg of avg).
    """
    # if __debug__:
    #     print(mat.shape)
    if mat.ndim > 2:
        return sd(np.concatenate([mat[:,:,k] for k in range(mat.shape[2])], axis=1))
    elif mat.ndim > 1:
        return np.std(mat, axis=1)
    else:
        return mat


def out_of_place_shuffle(src_array):
    """
    A function to clean the numpy shuffle function
    and not modify in place
    """
    if (src_array == 0).all():
        return src_array
    tmp = src_array.copy()
    np.random.shuffle(tmp)
    if (tmp == src_array).all():
        raise BadRandomError('srcArray: {}\ntmpArray: {}'.format(src_array, tmp))
    return tmp


def shuffle(mat):
    """
    Returns the randomly shuffled version of the input array
    The shuffle is performed on a First dimension basis
    """
    if (mat == 0).all():
        return mat
    out = mat.copy()
    if mat.ndim == 1:
        out[:] = np.nan  # For safety
        out = out_of_place_shuffle(mat)
    elif mat.ndim == 2:
        out[:, :] = np.nan
        for y in range(mat.shape[1]):
            out[:, y] = out_of_place_shuffle(mat[:, y])
    elif mat.ndim == 3:
        out[:, :, :] = np.nan
        for z in range(mat.shape[2]):
            for y in range(mat.shape[1]):
                out[:, y, z] = out_of_place_shuffle(mat[:, y, z])
    else:
        raise NotImplementedError('Number of dimension {} is not implemented'.format(mat.ndim))
    if (mat == out).all():
        raise BadRandomError('srcArray: {}\ntmp: {}'.format(mat, out))
    return out


def linearise(array):
    return array.reshape(array.size)


def get_uniques(items):
    return sorted(list(set(items)))


def make_uniques(values):
    return sorted(list(set(values)))


def flip_odd_rows(mat):
    out = mat.copy()
    for i in range(1, mat.shape[1], 2):  # TEST: check that rows/col correct
        out[:, i] = mat[::-1, i]  # TEST: check that rows/col correct
    return out


def make_mask_from_indices(size, idx_true=None):
    """
    creates a boolean mask from a list of indices
    :param int size: the total size of the array
    :param list idx_true: a list of indices to be returned as True in the mask
    :return:
    """

    # TODO: make work for n dimensional? is this something the np.ma module could do better?

    if idx_true is None:
        idx_true = list(range(size))

    mask = []
    for i in range(size):
        if i in idx_true:
            mask += [True]
        else:
            mask += [False]
    return np.array(mask)
