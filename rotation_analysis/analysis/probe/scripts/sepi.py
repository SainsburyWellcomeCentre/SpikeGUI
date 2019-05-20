import matplotlib

from analysis.probe.ctxt_manager_decorator import temp_setattr

matplotlib.use('QT5Agg')


import os

import matplotlib.pyplot as plt
from tqdm import tqdm

from analysis.probe.probe_io.probe_io import TriggerTraceIo, BonsaiIo, IgorIo
from analysis.probe.probe_field_of_view import ProbeFieldOfView
from spike_handling import spike_io

SRC_DIR = os.path.expanduser('/home/skeshav/swc-winstor/margrie/Sepi/Probe/CA282_5_processed/')


def main(experiment_id, raw_traces_extension='g1_t0_13000000_to_175000000.imec.ap.bin',
        trigger_extension='g1_t0_trigger_13000000_to_175000000.npy', n_chan=385):
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


    experiment_directory = os.path.join(SRC_DIR)  #SK: fixing src directory errors

    traces_filename = '{}_{}'.format(experiment_id, raw_traces_extension)
    probe_trigger_filename = '{}_{}'.format(experiment_id, trigger_extension)

    probe_trigger_trace_path = os.path.join('/home/skeshav/swc-winstor/margrie/Sepi/Probe/CA282_5_processed/CA282_5_g1_t0_trigger_13000000_to_175000000.npy')

    igor_waveforms_path = '/home/skeshav/swc-winstor/margrie/Sepi/Probe/CA282_5_processed/Igor/CA_282_5'
    bonsai_metadata_path = '/home/skeshav/swc-winstor/margrie/Sepi/Probe/CA282_5_processed/Bonsai/CA282_5_all.csv'

    traces_path = os.path.join(experiment_directory, traces_filename)
    sp = spike_io.SpikeIo(experiment_directory, traces_path, n_chan)


    trigger_trace_io = TriggerTraceIo(probe_trigger_trace_path, stimulus_onset_trigger_path= None, photodiode_trigger_path= None)
    bonsai_io = BonsaiIo(bonsai_metadata_path)
    igor_io = IgorIo(igor_waveforms_path, ['visual','black40', 'black', 'uniform', 'landmarkleft', 'landmarkright'])
    fov = ProbeFieldOfView(experiment_id, [], [], experiment_directory, 0, bonsai_io, igor_io, trigger_trace_io, sp,
                           cluster_ids=sp.good_cluster_ids[0:5])

    # for c in fov.get_cluster():
    #     fig = plt.figure(facecolor='w', figsize=(8, 3))
    #     plt.plot(c.best_channel_waveforms, alpha=0.2)
    #     plt.plot(c.best_channel_waveforms.mean(1), linewidth=2)
    #     plt.show()
    #     save_dir = os.path.join(experiment_directory, 'average_waveforms')
    #     if not os.path.isdir(save_dir):
    #         os.makedirs(save_dir)
    #     save_path = os.path.join(save_dir, 'cluster_{}_channel_{}.{}'.format(c.id, c.depth,'png'))
    #     fig.savefig(save_path, format='png')




    #FS_igor = 500  #SK: to use in congif
    #FS_probe = 25000   #SK: to use in congif
    #n_samples_before_trigger = 50000 #SK: to use in congif, also start and end.

    #db = fov.generate_db()
    #db.to_csv('/home/skeshav/Desktop/db.csv')
    matching_criteria_1 = {'Condition': '== Black'}
    matching_criteria_2 = {'Condition': '== Uniform'}
    #matching_criteria_3 = {'Condition': '== LandmarkLeft'}
    #matching_criteria_4 = {'Condition': '== LandmarkRight'}
    #matching_criteria_5 = {'Condition': '== MotorDrifting2'}
    #matching_criteria_6 = {'Condition': '== UniformDrifting', 'TemporalFrequency': '== 4'}
    #matching_criteria_7 = {'Condition': '== UniformDrifting', 'TemporalFrequency': '== -4'}
    #matching_criteria_8 = {'Condition': '== UniformDrifting', 'TemporalFrequency': '== 0.5'}
    #matching_criteria_9 = {'Condition': '== UniformDrifting', 'TotalCycles': '== 15'}
    #matching_criteria_10 = {'Condition': '== UniformDrifting', 'TotalCycles': '== 120'}
    #matching_criteria_11 = {'Condition': '== MotorDrifting'}


    matching_criteria_dicts1 = [matching_criteria_1, matching_criteria_2]
    #matching_criteria_dicts2 = [matching_criteria_2, matching_criteria_3]
    #matching_criteria_dicts3 = [matching_criteria_3, matching_criteria_4]
    #matching_criteria_dicts4 = [matching_criteria_2, matching_criteria_4]
    #matching_criteria_dicts5 = [matching_criteria_1, matching_criteria_5]
    #matching_criteria_dicts6 = [matching_criteria_1, matching_criteria_2, matching_criteria_5]
    #matching_criteria_dicts7 = [matching_criteria_6, matching_criteria_7]
    #matching_criteria_dicts8 = [matching_criteria_9, matching_criteria_10]
    #matching_criteria_dicts9 = [matching_criteria_1, matching_criteria_2, matching_criteria_3, matching_criteria_4]


    fov.plot_overlay_histogram(matching_criteria_dicts1)
    #fov.plot_overlay_histogram(matching_criteria_dicts1, save = True)
    #fov.plot_overlay_histogram(matching_criteria_dicts2)
    #fov.plot_overlay_histogram(matching_criteria_dicts3)
    #fov.plot_overlay_histogram(matching_criteria_dicts4)
    #fov.plot_overlay_histogram(matching_criteria_dicts5)
    #fov.plot_overlay_histogram(matching_criteria_dicts6, save=True)
    #fov.plot_overlay_histogram(matching_criteria_dicts7, save=True)
    #fov.plot_overlay_histogram(matching_criteria_dicts8, save=True)
    #fov.plot_overlay_histogram(matching_criteria_dicts9, save=True)


    #fov.save_all_clusters_avgs(matching_criteria_dicts, save_path='/home/skeshav/Desktop/all_trials_avg_steve.csv')
    #fov.save_all_clusters_trials(matching_criteria_dicts, save_path='/home/skeshav/Desktop/all_trials_steve.csv')
    #fov.plot_all_sub_stimuli_histograms(matching_criteria_1, save=True)
    #fov.plot_all_sub_stimuli_histograms(matching_criteria_2, save=True)
    #fov.plot_all_sub_stimuli_histograms(matching_criteria_3, save=True)
    #fov.plot_all_sub_stimuli_histograms(matching_criteria_4, save=True)
    #fov.plot_all_sub_stimuli_histograms(matching_criteria_5, save=True)
    #fov.plot_all_sub_stimuli_histograms(matching_criteria_6, save=True)
    #fov.plot_all_sub_stimuli_histograms(matching_criteria_7, save=True)
    #fov.plot_all_sub_stimuli_histograms(matching_criteria_11, save=True)


if __name__ == '__main__':
    tqdm(main('CA282_5'))
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
    # main(args)