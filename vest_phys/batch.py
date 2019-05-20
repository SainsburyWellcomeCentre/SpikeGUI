import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import shutil
from matplotlib import gridspec
from matplotlib import pyplot as plt

import sys

try:
    from src.utils.utils import shell_hilite
except ImportError:
    from utils.utils import shell_hilite

sys.path.append(os.path.join(os.path.dirname(sys.modules[__name__].__file__), '..'))
from experiment import Experiment

#  src_dir = '/alzymr/buffer/zizAreNotBrains/CR/dataClubMateo/'
exp_extension = '.pxp'


def plot_total_psth(exps, src_dir, extension):
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 7])  # FIXME: compute from durations
    f = plt.figure()
    f.suptitle('Average PSTHs')
    ax1 = f.add_subplot(gs[0])
    ax2 = f.add_subplot(gs[1], sharey=ax1)

    psth_x = exps[0].plotter.psth_x
    bsl_psth_x = exps[0].plotter.bsl_psth_x

    colors = {'L6_CC': 'b',
              'L6_CT': 'g',
              'L2-3_Pyr': 'r',
              'L5_Pyr': 'purple',
              'L6_Pyr': 'orange'
              }
    cell_specs = set(((e.layer, e.cell_type) for e in exps))
    for cell_spec in cell_specs:
        psths = [e.plotter.psth_y for e in exps if (e.layer, e.cell_type) == cell_spec]
        bsl_psths = [e.plotter.bsl_psth_y for e in exps if (e.layer, e.cell_type) == cell_spec]

        formatted_cell_spec = "{}_{}".format(*cell_spec)
        ax1.step(bsl_psth_x, np.mean(bsl_psths, 0),
                 where='post',
                 color=colors[formatted_cell_spec],
                 label=formatted_cell_spec)
        ax2.step(psth_x, np.mean(psths, 0),
                 where='post',
                 color=colors[formatted_cell_spec],
                 label=formatted_cell_spec)

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Spikes (n)')
    ax2.set_xlabel('Time (s)')
    #    plt.legend() # FIXME: does not yet work

    plt.savefig(os.path.join(src_dir, 'total_PSTH.{}'.format(extension)))
    plt.close(f)


if __name__ == '__main__':
    program_name = os.path.basename(__file__)
    supported_extensions = tuple([str(k) for k in plt.gcf().canvas.get_supported_filetypes().keys()])
    parser = ArgumentParser(prog=program_name,
                            description='Program to batch process vestibular physiology '
                                        'recordings from the roller coaster')
    parser.add_argument("-f", "--source-file-path", dest="src_file_path", type=str,
                        help="The file where the parameters are stored")
    parser.add_argument("-e", "--extension", choices=supported_extensions, default='png', type=str,
                        help="The extension to save the graphs. One of %(choices)s. Default: %(default)s.")
    parser.add_argument("-s", "--resample-matrices", dest='resample_matrices', action='store_true',
                        help="Whether or not to do the analysis with the matrices resampled in degrees, vel...")
    parser.add_argument("-d", "--do-spiking-difference", dest='do_spiking_difference', action='store_true',
                        help="FIXME")
    parser.add_argument("-r", "--do-spiking-ratio", dest='do_spiking_ratio', action='store_true',
                        help="FIXME")
    parser.add_argument('-t', '--plot-total-psth', dest="plot_total_psth", action='store_true',
                        help='Plot a summary PSTH for each cell type')
    parser.add_argument('-m', '--move-source', dest="move_source", action='store_true',
                        help='Whether we want to move the source file to the experiment directory with the graphs.')

    args = parser.parse_args()

    src_dir = os.path.dirname(args.src_file_path)
    print("Source directory: {}".format(src_dir))
    
    with open(args.src_file_path, 'r') as in_file:
        lines = in_file.readlines()
    lines = [l.strip().split('\t') for l in lines if not l.startswith('#')]
    
    exps = []
    i = 0
    j = 0
    for exp_name, cell_type, layer in lines:
        exp_path = os.path.join(src_dir, '{}{}'.format(exp_name, exp_extension))
        if not os.path.exists(exp_path):
            raise ValueError("File {} does not exist".format(exp_path))
        else:
            print(shell_hilite('Processing: {}'.format(exp_path), 'green'))
        os.chdir(os.path.dirname(exp_path))
        exp = Experiment(exp_path, ext=args.extension, cell_type=cell_type, layer=layer)
        exps.append(exp)
        if args.resample_matrices:
            exp.resample_matrices()
            exp.analyse(do_spiking_difference=args.do_spiking_difference, do_spiking_ratio=args.do_spiking_ratio)
            exp.write()
        exp.plotter.plot(True)
        significant_directions = []
        significant_directions_r2 = []
        if exp.significant_counter_clockwise:
            significant_directions.append("counter_clockwise")
        if exp.significant_clockwise:
            significant_directions.append("clockwise")

        if exp.significant_r2_counter_clockwise:
            significant_directions_r2.append('counter_clockwise')
        if exp.significant_r2_clockwise:
            significant_directions_r2.append('clockwise')
        print("Significant directions: {}".format(significant_directions))
        if len(significant_directions) == 2:
            df = exp.frame[(exp.frame['direction'] == significant_directions[0]) | (exp.frame['direction'] == significant_directions[1])]
        elif len(significant_directions) == 1:
            df = exp.frame[exp.frame['direction'] == significant_directions[0]]

        if significant_directions:
            df_cp = df.copy()
            df_cp.loc[:, 'cell'] = i
            if i == 0:
                total_df = df_cp
            else:
                # df_cp.loc[:, 'trial'] = df['trial'] + (total_df['trial'].max() + 1)
                total_df = pd.concat([total_df, df_cp])
            i += 1

        print("Significant directions r2: {}".format(significant_directions_r2))
        if len(significant_directions_r2) == 2:
            df_r2 = exp.frame[(exp.frame['direction'] == significant_directions_r2[0]) | (exp.frame['direction'] == significant_directions_r2[1])]
        elif len(significant_directions_r2) == 1:
            df_r2 = exp.frame[exp.frame['direction'] == significant_directions_r2[0]]
        if significant_directions_r2:
            df_r2_cp = df_r2.copy()
            df_r2_cp.loc[:, 'cell'] = j
            if j == 0:
                total_df_r2 = df_r2_cp
            else:
                total_df_r2 = pd.concat([total_df_r2, df_r2_cp])
            j += 1

        # _, _, vel_to_plot = exp.matrices[2]._pool_orientations(exp.matrices[2].binned_data)
        # exp.matrices[2].plot_scatter(np.nan_to_num(vel_to_plot), name='{}_test_scatter.{}'.format(exp.name, exp.ext))
    if i > 0:
        total_df.to_csv(os.path.join(src_dir, "significant_trials.csv"))
    if j > 0:
        total_df_r2.to_csv(os.path.join(src_dir, "significant_trials_r2.csv"))

    if args.plot_total_psth:
        plot_total_psth(exps, src_dir, args.extension)

    if args.move_source:
        for exp in exps:
            exp.move_to_folder()

    # dsi_out = ['{}\t{}\t{}\t{}\n'.format(e.exp_id, e.cell_type, e.layer, e.dsi) for e in exps]  # FIXME: no dsi
    # with open(os.path.join(src_dir, 'dsis.csv'), 'w') as out_file:
    #     out_file.writelines(dsi_out)
