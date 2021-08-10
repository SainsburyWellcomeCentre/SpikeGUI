import numpy as np


def get_spike_waveforms(spike_times, traces, n_chan, n_samples_before_peak=20, n_samples_after_peak=40):

    n_waveform_samples = n_samples_before_peak + n_samples_after_peak
    waveforms = np.zeros((n_waveform_samples, n_chan, len(spike_times)))

    for i, spike_time in enumerate(spike_times):
        spike_time = int(spike_time)
        start_of_waveform = int(spike_time - n_samples_before_peak)
        end_of_waveform = int(spike_time + n_samples_after_peak)
        waveform_on_all_channels = traces[start_of_waveform:end_of_waveform, :]
        waveforms[:, :, i] = waveform_on_all_channels
    return waveforms


def get_avg_waveforms(waveforms):
    """

    :param np.ndarray waveforms: all spike waveforms (list of np.arrays/memmaps across all channels)
    :return np.array avg_waveforms: the average waveforms on all channels
    """
    avg_waveforms = np.mean(waveforms, axis=2)
    return avg_waveforms


def get_channel_of_max_amplitude_avg_waveform(spike_times, traces, n_chan):

    all_waveforms = get_spike_waveforms(spike_times, traces, n_chan)
    avg_waveforms = get_avg_waveforms(all_waveforms)
    min_time_point, min_channel = np.unravel_index(avg_waveforms.argmin(), avg_waveforms.shape)
    return min_channel
