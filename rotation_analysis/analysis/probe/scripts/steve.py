import matplotlib

from rotation_analysis.analysis.probe.ctxt_manager_decorator import temp_setattr

matplotlib.use('QT5Agg')


import os

import matplotlib.pyplot as plt
from dateutil import parser
from tqdm import tqdm

from rotation_analysis.analysis.probe.probe_io.probe_io import TriggerTraceIo, BonsaiIo, IgorIo
from rotation_analysis.analysis.probe.probe_field_of_view import ProbeFieldOfView
from probez.spike_handling import spike_io

# SRC_DIR = os.path.expanduser('/home/skeshav/swc-winstor/margrie/Sepi/Probe/Data/CA326_C/CA326_C_day3/processed/denoised_and_filtered2/')
SRC_DIR = 'G:\\Seagate\\Work\\EPhys\\Data\\CA326_C_day3\\processed\\denoised_and_filtered2'

def main(experiment_id, raw_traces_extension='g0_t0_0_to_160778264_32_33_34_35_13000000_to_160778264.imec.ap.bin',
        trigger_extension='trigger_13000000_to_300000000_chan[34, 35].npy', n_chan=32):
    """

    :param trigger_extension:
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

    #experiment_directory = os.path.join(SRC_DIR, experiment_id)
    experiment_directory = os.path.join(SRC_DIR)

    traces_filename = '{}_{}'.format(experiment_id, raw_traces_extension)

    # Initialisation File
    #  - change names to "Probe Trigger Trace Path"
    #  - Also invlude "Stimulus Onset Trigger Path" (stimulus_onset_trigger_path)
    #                 "Photo Diode Trigger Path" (photodiode_trigger_path)

    # If photodiode_trigger_path and stimulus_onset_trigger_path is empty, then use probe_trigger_trace_path

    probe_trigger_trace_path = 'G:\\Seagate\\Work\\EPhys\\Data\\CA326_C_day3\\processed\\CA326_C_day3_g0_t0_trigger_13000000_to_300000000_chan[34_35].npy'
    igor_waveforms_path = 'G:\\Seagate\\Work\\EPhys\\Data\\CA326_C_day3\\Igor\\CA326_C_day3'
    bonsai_metadata_path = 'G:\\Seagate\\Work\\EPhys\\Data\\CA326_C_day3\\Bonsai\\CA326_C_day3_all.csv'

    # probe_trigger_trace_path = os.path.join('/home/skeshav/swc-winstor/margrie/Sepi/Probe/Data/CA326_C/CA326_C_day3/processed/CA326_C_day3_g0_t0_trigger_13000000_to_300000000_chan[34_35].npy')
    # igor_waveforms_path = '/home/skeshav/swc-winstor/margrie/Sepi/Probe/Data/CA326_C/CA326_C_day3/Igor/CA326_C_day3'
    # bonsai_metadata_path = '/home/skeshav/swc-winstor/margrie/Sepi/Probe/Data/CA326_C/CA326_C_day3/Bonsai/CA326_C_day3_all.csv'

    traces_path = os.path.join(experiment_directory, traces_filename)
    sp = spike_io.SpikeIo(experiment_directory, traces_path, n_chan)

    stimulus_onset_trigger_path = 'G:\\Seagate\\Work\\EPhys\\Data\\CA326_C_day3\\processed\\CA326_C_day3_g0_t013000000to_160778264_trigger_stim_onset_trigger.npy'
    photodiode_trigger_path = 'G:\\Seagate\\Work\\EPhys\\Data\\CA326_C_day3\\processed\\CA326_C_day3_g0_t013000000to_160778264_photodiode_trigger.npy'


    trigger_trace_io = TriggerTraceIo(probe_trigger_trace_path,
                                      stimulus_onset_trigger_path=stimulus_onset_trigger_path,
                                      photodiode_trigger_path=photodiode_trigger_path)
    bonsai_io = BonsaiIo(bonsai_metadata_path)
    igor_io = IgorIo(igor_waveforms_path, ['UniformDrifting', 'MotorDrifting', 'MotorDrifting2', 'Black',
                                           'Uniform', 'LandmarkLeft', 'LandmarkRight'])
    igor2_io = IgorIo(igor_waveforms_path, ['UniformDrifting', 'MotorDrifting', 'Black',
                                           'Uniform', 'LandmarkLeft', 'LandmarkRight'])


    #
    t_bonsai = [parser.parse(x) for x in bonsai_io.data['Timestamp']]
    t_bonsai_trial = []


    fov = ProbeFieldOfView(experiment_id, [], [], experiment_directory, 0, bonsai_io, igor_io, trigger_trace_io, sp,
                           cluster_ids=sp.good_cluster_ids)

    matching_criteria_1 = {'Condition': '== Black'}
    matching_criteria_2 = {'Condition': '== Uniform'}
    matching_criteria_3 = {'Condition': '== MotorDrifting'}
    matching_criteria_4 = {'Condition': '== LandmarkLeft'}
    matching_criteria_5 = {'Condition': '== LandmarkRight'}
    matching_criteria_6 = {'Condition': '== UniformDrifting'}

    matching_criteria_dicts1 = [matching_criteria_1, matching_criteria_2]
    matching_criteria_dicts2 = [matching_criteria_3, matching_criteria_1, matching_criteria_2]
    matching_criteria_dicts3 = [matching_criteria_1, matching_criteria_2, matching_criteria_4, matching_criteria_5]
    matching_criteria_dicts4 = [matching_criteria_6]

    # for c in fov.cells():
    #     with temp_setattr(c.block, 'current_condition', matching_criteria_11):
    #         stimulus = c.block.kept_trials[0].stimulus
    #         # levels, positions = stimulus.position_timetable()
    #         plt.plot(c.block.kept_trials[0].stimulus.x, c.block.kept_trials[0].stimulus.cmd);
    #
    #         levels, vel = stimulus.velocity_timetable()
    #         plt.axvline(vel[0][0], color='k')
    #         plt.axvline(vel[-1][0], color='k')
    #         plt.axvline(vel[0][1], color='r')
    #         plt.axvline(vel[-1][1], color='r')
    #         plt.show()

    fov.plot_overlay_histogram(matching_criteria_dicts1, save = False)
    #fov.plot_overlay_histogram(matching_criteria_dicts2, save=True)
    #fov.plot_overlay_histogram(matching_criteria_dicts3, save=True)
    #fov.plot_all_sub_stimuli_histograms(matching_criteria_3, save=True)

    #db = fov.generate_db()
    #db.to_csv('/home/skeshav/Desktop/db_test_fix_relative_events.csv')


if __name__ == '__main__':
    tqdm(main('CA326_C_day3'))
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

