import os

from analysis.probe import analyse_functions
from analysis.probe.trigger_handling.trigger_handler import TriggerIo
from spike_handling import spike_io

SRC_DIR = os.path.expanduser('~/probe_data/')


def main(experiment_id, raw_traces_extension='g0_t0.imec.ap.bin', trigger_extension='trigger.npy', n_chan=385, angle=90, stimulus_condition='black'):
    """
    
    :param experiment_id: name of the mouse (should be the same as the parent folder for this data, 
    assumes that the file name is the experiment_id and raw_traces extension combined)
    :param raw_traces_extension: 
    :param n_chan: number of channels on the full probe
    :param angle: 
    :param stimulus_condition:
    :param start_sample:
    :param end_sample:
    :return:
    """

    experiment_directory = os.path.join(SRC_DIR, experiment_id)

    traces_filename = '{}_{}'.format(experiment_id, raw_traces_extension)
    probe_trigger_filename = '{}_{}'.format(experiment_id, trigger_extension)

    traces_path = os.path.join(experiment_directory, traces_filename)
    probe_trigger_trace_path = os.path.join(experiment_directory, probe_trigger_filename)

    stimuli_path = os.path.join(experiment_directory, 'randwaves_clean')
    timestamps_path = os.path.join(experiment_directory, 'bonsai_timestamps.csv')

    condition = get_condition_from_keys(angle, experiment_id, stimulus_condition)

    sp = spike_io.SpikeIo(experiment_directory, traces_path, n_chan)

    failed_trials_bonsai_idx = [34, 35, 52, 53, 54, 55, 56]
    failed_trials_probe_idx = [0, 1, 2, 3, 36, 37, 54, 55, 56, 57, 58]

    th = TriggerIo(stimuli_path, timestamps_path, probe_trigger_trace_path, failed_trials_bonsai_idx,
                   failed_trials_probe_idx)

    th.plot_split_triggers()  # sanity check step to visually check how triggers map onto the probe trace

    #analyse_functions.plot_all_histograms(condition, experiment_id, th.actual_triggers, sp, experiment_directory)

    #analyse_functions.plot_resampling_heatmaps(experiment_directory, exp_id=experiment_id, successful_triggers=th.actual_triggers, sp=sp, condition=condition)
    analyse_functions.do_stats(experiment_directory, exp_id=experiment_id, successful_triggers=th.actual_triggers, sp=sp)

    # analyse_functions.generate_heatmap_arrays(sp, experiment_id, th.actual_triggers, experiment_directory, condition, n_bins=30)
    # analyse_functions.plot_sorted_population_heatmaps(sp, '/home/skeshav/probe_data/CA_242_A/population_histograms_heatmap_matrices/CA_242_A_90_black/')


def get_condition_from_keys(angle, experiment_id, stimulus_condition):
    return '{}_{}_{}'.format(experiment_id, angle, stimulus_condition)


if __name__ == '__main__':
    main('CA_242_A')
