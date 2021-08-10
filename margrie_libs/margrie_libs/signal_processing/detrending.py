import math
import os

import numpy as np
from file_handling import binary_classes

from margrie_libs.signal_processing.metadata_handling import store_meta_data


def detrend_single_channel(trace):
    from scipy.signal import savgol_filter
    return trace - savgol_filter(trace, 1001, 3)


def detrend_all_channels_savgol(array):
    from scipy.signal import savgol_filter
    detrended_data = array - savgol_filter(array, 1001, 3, axis=0)
    return detrended_data.astype(int)


def detrend_all_channels_gaussian(array):
    from scipy.ndimage.filters import gaussian_filter1d
    detrended_data = array - gaussian_filter1d(array, sigma=100, axis=0)
    return detrended_data


def denoise_detrend(array, n_chan, detrending_func=detrend_all_channels_gaussian):
    detrend_array = detrending_func(array)
    new_data = chunk_wise_common_average_ref(detrend_array, n_chan)
    return new_data.astype(int)


def median_subtract(array):
    trace_medians = np.median(array, axis=0)
    return (array-trace_medians).astype(int)


def median_subtract_and_car(array, n_chan):
    detrend_array = median_subtract(array)
    denoised_median_subtracted = chunk_wise_common_average_ref(detrend_array, n_chan)
    return denoised_median_subtracted

#
# def normalise_denoise_detrend(array, n_chan):
#     detrended = denoise_detrend(array, n_chan=n_chan)
#     normalised_detrended = detrended/np.std(detrended, axis=0)
#     return normalised_detrended


def common_average_reference(array):
    return np.median(array, axis=1).reshape(array.shape[0], 1)


def chunk_wise_common_average_ref(array, n_chan, n_chan_per_block=77): # 40
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


def _get_file_size(f_in_path):
    file_info = os.stat(f_in_path)
    file_size = file_info.st_size
    return file_size


def _reshape_data(data, chunk):
    n_samples = int(chunk.size/chunk.byte_width/chunk.n_chan)
    reshaped_data = np.array(data).reshape(n_samples, chunk.n_chan)
    return reshaped_data


def get_alpha(file_path, shape):

    data = np.memmap(file_path, shape=shape, dtype='int16')
    total_size = 2**15  # signed
    alpha = total_size/max(data.max(), np.abs(data.min()))
    return alpha


def _scale_to_int16(file_path, data, alpha):
    store_meta_data(file_path, 'alpha', alpha)  # scale factor needs to be stored in meta data for loading back
    data_int16 = (data*alpha)
    return data_int16.astype(int)


def _make_packable(file_path, data, chunk, alpha):
    data = _scale_to_int16(file_path, data, alpha)  # data must be scaled properly to avoid data loss
    data = data.reshape(int(chunk.size/chunk.byte_width))
    return tuple(data)


def process_to_file(f_in_path, f_out_path, path_to_meta_data, n_chan, process_data_func=denoise_detrend,
                    n_samples_to_process=50000):

    file_size = _get_file_size(f_in_path)

    data_point = binary_classes.DataPoint()
    time_point = binary_classes.TimePoint(data_point, n_chan)
    chunk = binary_classes.Chunk(time_point, n_samples_to_process)  # define normal chunk

    leftover_bytes = file_size % chunk.size  # this is a problem if the last chunk is very small
    leftover_samples = int(leftover_bytes/time_point.size)
    last_chunk = binary_classes.Chunk(time_point, leftover_samples)  # define final chunk

    print('chunk size is {}, last chunk is {} bytes:'.format(chunk.size, leftover_bytes))
    print('leftover samples = {}'.format(leftover_samples))

    n_chunks = math.ceil(file_size/chunk.size)

    with open(f_in_path, 'rb') as f_in:
        with open(f_out_path, 'wb') as f_out:
            for i in range(n_chunks):
                print('chunk: {} of {}'.format(i+1, n_chunks))
                try:
                    chunk_in = f_in.read(chunk.size)  # read 1 set of channels
                except EOFError:
                    break

                if int(len(chunk_in)) == last_chunk.size:
                    current_chunk = last_chunk
                else:
                    current_chunk = chunk

                if not chunk_in:
                    print('break line reached')
                    break
                if not current_chunk:
                    print('break line reached')
                    break

                data = current_chunk.s.unpack_from(chunk_in)
                data = _reshape_data(data, current_chunk)  # reshape data to array
                processed_data = process_data_func(data, n_chan)  # carry out processing steps
                packable_new_data = _make_packable(path_to_meta_data, processed_data, current_chunk)  # put data into packable form
                chunk_out = current_chunk.s.pack(*packable_new_data)  # pack



                bytes_written = f_out.write(bytes(chunk_out))


