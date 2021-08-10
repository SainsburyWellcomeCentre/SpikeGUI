import math
import numpy as np
from scipy.signal import butter, filtfilt


def detrend_all_channels_gaussian(array):
    from scipy.ndimage.filters import gaussian_filter1d
    detrended_data = array - gaussian_filter1d(array, sigma=100, axis=0)
    return detrended_data


def denoise_detrend(array, n_chan, detrending_func=detrend_all_channels_gaussian):
    detrend_array = detrending_func(array)
    new_data = common_average_reference_subtraction_blockwise(detrend_array, n_chan)
    return new_data.astype(int)


def detrend_single_channel(trace):
    from scipy.signal import savgol_filter
    return trace - savgol_filter(trace, 1001, 3)


def detrend_all_channels_savgol(array):
    from scipy.signal import savgol_filter
    detrended_data = array - savgol_filter(array, 1001, 3, axis=0)
    return detrended_data.astype(int)


def median_subtract(array):
    trace_medians = np.median(array, axis=0)
    return (array - trace_medians).astype(int)


def median_subtract_and_car(array, n_chan):
    detrend_array = median_subtract(array)
    denoised_median_subtracted = common_average_reference_subtraction_blockwise(detrend_array, n_chan)
    return denoised_median_subtracted


def butter_bandpass(lowcut, highcut, fs, order=3):

    """
    creating the butterworth filter

    :param int lowcut: low frequency cutoff
    :param int highcut: high frequency cutoff
    :param int fs: sampling frequency in Hz
    :param int order: filter order
    :return: b and a on default

    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(array, lowcut, highcut, fs, order=3):
    """

    apply a butterworth filter to 2d array, option of filtfilt for zero phase filtering

    :param np.array array:
    :param int lowcut:
    :param int highcut:
    :param int fs:
    :param int order:
    :return:
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = filtfilt(b, a, array, axis=0)
    return filtered_data


def median_subtract_car_and_bandpass(array, n_chan, lowcut=300, highcut=10000, fs=25000, order=3):

    """
    temporal and spatial median subtraction and bandpass filtering

    :param np.array array:
    :param int n_chan:
    :param int lowcut:
    :param int highcut:
    :param int fs:
    :param int order:
    :return:
    """

    detrend_array = median_subtract(array)
    denoised_median_subtracted = common_average_reference_subtraction_blockwise(detrend_array, n_chan)
    filtered_denoised = butter_bandpass_filter(denoised_median_subtracted, lowcut, highcut, fs, order=order)
    return filtered_denoised.astype(np.int16)


def common_average_reference_subtraction_blockwise(array, n_chan, n_chan_per_block=32):  # 77 fro neuropix
    """

    :param np.array array:
    :param int n_chan: number of channels in total including triggers (i.e. for determining the shape of your data)
    :param int n_chan_per_block: the number of channels to de-noise together: NOTE: must be a factor of n_chan

    :return np.array denoised: a de-noised array
    """
    # TODO: n_chan_per_chunk needs to be automatically adjusted for different numbers of channels.
    # TODO: (cont.) enforce minimum number of channels

    if n_chan % n_chan_per_block != 0:
        raise ValueError('n_chan_per_chunk {} must be multiple of n_chan_tot {}'.format(n_chan_per_block, n_chan))

    n_chunks = math.ceil(n_chan / n_chan_per_block)

    median_array = np.zeros_like(array)
    for i in range(n_chunks):
        block_median = common_average_reference(array[:, n_chan_per_block * i:n_chan_per_block * (i + 1)])
        median_array[:, (n_chan_per_block * i):(n_chan_per_block * (i + 1))] = block_median

    denoised = array - median_array

    return denoised


def common_average_reference(array):
    return np.median(array, axis=1).reshape(array.shape[0], 1)
