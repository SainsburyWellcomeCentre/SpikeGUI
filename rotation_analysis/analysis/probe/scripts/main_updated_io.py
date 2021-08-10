import os

from analysis.probe.probe_io.probe_io import TriggerTraceIo, BonsaiIo, IgorIo
from analysis.probe.probe_field_of_view import ProbeFieldOfView
from spike_handling import spike_io

SRC_DIR = os.path.expanduser('/home/skeshav/swc_glusterfs/Sepi/Probe/')


def main(experiment_id, raw_traces_extension='g0_t0_4000000_to_168953184.imec.ap.bin',
         trigger_extension='trigger_4000000_to_168953184.npy', n_chan=385):
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

    probe_trigger_trace_path = os.path.join('/home/skeshav/swc_glusterfs/Sepi/Probe/CA312_B_processed/CA312_B_g0_t0_trigger_4000000_to_168953184.npy')

    igor_waveforms_path = '/home/skeshav/swc_glusterfs/Sepi/Probe/CA312_B_processed/Igor/CA312_B_editedFileTime'
    bonsai_metadata_path = '/home/skeshav/swc_glusterfs/Sepi/Probe/CA312_B_processed/Bonsai/CA312_B_all.csv'

    traces_path = os.path.join(experiment_directory, traces_filename)
    sp = spike_io.SpikeIo(experiment_directory, traces_path, n_chan)

    trigger_trace_io = TriggerTraceIo(probe_trigger_trace_path)
    bonsai_io = BonsaiIo(bonsai_metadata_path)
    igor_io = IgorIo(igor_waveforms_path)

    fov = ProbeFieldOfView(experiment_id, [], [], experiment_directory, 0, bonsai_io, igor_io, trigger_trace_io, sp, cluster_ids=[406, 426, 262, 310, 319, 338, 516, 646, 129, 188, 301, 621])

    matching_criteria_1 = {'Condition': '== LandmarkLeft',

                           }

    matching_criteria_2 = {'Condition': '== LandmarkRight',

                           }

    matching_criteria_dicts = [matching_criteria_1, matching_criteria_2]

    #fov.plot_all_sub_stimuli_histograms(matching_criteria_1, save=True, format='png')
    fov.plot_all_sub_stimuli_histograms(matching_criteria_2, save=True, format='png')

    fov.save_all_clusters_avgs(matching_criteria_dicts)
    fov.save_all_clusters_trials(matching_criteria_dicts)


if __name__ == '__main__':
    main('CA312_B_processed')
    # program_name = os.path.basename(__file__)
    # parser = ArgumentParser(prog=program_name,
    #                         description='Program to recursively analyse all recordings in a probe experiment')
    # 
    # parser.add_argument('source_directory', type=str, help='The source directory of the experiment')
    # parser.add_argument('-f', '--file-extension', dest='extension', type=str, choices=('png', 'eps', 'pdf', 'svg'),
    #                     default='png',
    #                     help='The file extension to save the figures. One of: %(choices)s. Default: %(default)s.')
    # 
    # parser.add_argument('-s', '--stats', action='store_true', help='Whether to do the statistics of each cell')
    # parser.add_argument('-p', '--plot', action='store_true', help='Whether to plot each cell')
    # parser.add_argument('-c', '--remove-cells', dest='remove_cells', action='store_true',
    #                     help='Whether to remove some cells from the analysis')
    # 
    # parser.add_argument('-t', '--remove-trials', dest='remove_trials', action='store_true',
    #                     help='Whether to remove some trials from the analysis')
    # 
    # parser.add_argument('--use-bsl-2', dest='use_bsl_2', action='store_true',
    #                     help='Whether to pool the first and second baselines in the analysis of baseline')
    # 
    # args = parser.parse_args()
    # 
    # main(args)

