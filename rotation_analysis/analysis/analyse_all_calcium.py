import os
import sys
import pickle
from argparse import ArgumentParser
from collections import OrderedDict
from zipfile import ZipFile

if sys.platform == "win32":
    sys.path.append(os.path.abspath(os.path.normpath('c:\\Users\\Administrator\\Desktop\\sepi_rotation_analysis_steve\\')))

from analysis.cell import Cell
from analysis.field_of_view import FieldOfView
from calcium_recordings.register_all import getRecsTree, getDepthDirs, getDepth

from analysis.event_detection.gui import gui_launcher


def get_n_cells(rec):
    """
    Return the number of cells detected in that recording
    """
    try:
        rec0_path = rec.registeredImgFilesPaths[0]
    except IndexError:
        print("Recording {}, no registered image".format(rec))
        raise
    rois_path = rec0_path[:-4] + "_roi.zip"
    with ZipFile(rois_path, 'r') as rois:
        n_cells = len([f for f in rois.namelist() if f.endswith('.roi')])
    return n_cells


def get_first_rec(recs, angle):
    for i in range(len(recs[angle])):
        first_rec = recs[angle][i]
        try:
            n_cells = get_n_cells(first_rec)
            break  # Stop as soon as we have a good recording
        except IndexError:
            pass
    return first_rec, n_cells


def extract_cell_idx(path):
    cell_name = os.path.splitext(os.path.basename(path))[0]
    cell_idx = int(cell_name.split('_')[-1])
    return cell_idx


def prompt_cells_removal(depth):
    answer = input("Please give a comma separated list of cells to remove for depth: {}\n".format(depth))
    if answer:
        cells_to_remove = answer.split(',')
        cells_to_remove = [int(c) for c in cells_to_remove if c]  # TODO: check if should have if c != ''
    else:
        cells_to_remove = []
        print("User chose not to remove cells")
    print('Cells selected: {}'.format(cells_to_remove))
    return cells_to_remove


def save_removed_cells(main_dir, depth, cells_to_remove):
    with open(os.path.join(main_dir, 'cells_removed.txt'), 'a') as out_file:
        out_file.write('Depth: {}\n\tcells: {}\n'.format(depth, ','.join([str(c) for c in cells_to_remove])))


def load_cells(main_dir, remove_cells):
    cells = {}
    depths_dirs = getDepthDirs(main_dir)
    for depth_dir in depths_dirs:
        depth = getDepth(depth_dir)
        cells[depth] = []
        files_list = os.listdir(depth_dir)
        cells_pkls = [f for f in files_list if f.endswith('.pkl')]
        cells_idx_list = [extract_cell_idx(f) for f in cells_pkls]

        cells_idx_list, cells_pkls = zip(*sorted(zip(cells_idx_list, cells_pkls)))

        if remove_cells:
            cells_to_remove = prompt_cells_removal(depth)
            save_removed_cells(main_dir, depth, cells_to_remove)
        else:
            cells_to_remove = []
        for idx, pkl in zip(cells_idx_list, cells_pkls):  # filter_df for that depth
            if idx not in cells_to_remove:
                cell_pkl_path = os.path.join(depth_dir, pkl)
                print(cell_pkl_path)
                with open(cell_pkl_path, 'rb') as pkl_file:
                    cells[depth].append(pickle.load(pkl_file))
    return cells


def make_cells_from_recordings(main_dir, depth, recs, extension, use_bsl_2):
    first_rec, n_cells = get_first_rec(recs, list(recs.keys())[0])
    depth_dir = os.path.join(main_dir, first_rec.ini.replace(main_dir, "").split(os.path.sep)[1])
    exp_id = (main_dir.split(os.sep))[-1]
    cells = [Cell(depth_dir, exp_id, depth, i, recs, extension, use_bsl_2) for i in range(n_cells)]
    return cells


def get_cells(main_dir, extension, use_bsl_2):
    cells = {}
    recs_tree = getRecsTree(main_dir)
    for depth, recs_at_depth in recs_tree.items():
        recs = OrderedDict()
        angles = recs_at_depth.keys()
        for angle in angles:
            tmp_recs = recs_at_depth[angle]['recs']  # Skip the highRes and stacks
            recs[angle] = [r for r in tmp_recs if r._getProfilesPaths()]  # Skip if no csv fluorescence profile

        if not recs_at_depth:  # TODO: put as first line
            continue
        tmp_cells = make_cells_from_recordings(main_dir, depth, recs, extension, use_bsl_2)

        for angle in angles:
            blocks = [cell.block for cell in tmp_cells]
            gui_launcher.main(blocks, angle)  # FIXME: use processing type

        for cell in tmp_cells:
            processing_type = 'deltaF'  # TODO: deltaFDeltaColor, + put as argument to program
            cell.save_detection(processing_type)
            cell.pickle()
        cells[depth] = tmp_cells
    return cells


def promp_trial_removal(main_dir, depth):
    answer = input("Please give a comma separated list of trials to remove for depth: {}\n".format(depth))
    trials_to_remove = [int(t) for t in answer.split(',') if t != '']
    print('Trials selected: {}'.format(trials_to_remove))
    with open(os.path.join(main_dir, 'trials_removed.txt'), 'a') as out_file:
        out_file.write('Depth: {}\n')
        out_file.write('\ttrials: {}\n'.format(depth, answer))
    return trials_to_remove


def main(_args):
    """
    MAIN FUNCTION
    """
    main_dir = os.path.abspath(_args.source_directory)

    if _args.load_cells:  # Using pickle
        cells = load_cells(main_dir, _args.remove_cells)
    else:
        cells = get_cells(main_dir, _args.extension, _args.use_bsl_2)

    depths = cells.keys()
    for depth in depths:
        current_cells = cells[depth]

        if _args.remove_cells and not _args.load_cells:
            cells_to_remove = prompt_cells_removal(depth)
            save_removed_cells(main_dir, depth, cells_to_remove)
            current_cells = [c for c in current_cells if c.id not in cells_to_remove]
        if _args.remove_trials:
            trials_to_remove = promp_trial_removal(main_dir, depth)
        else:
            trials_to_remove = []

        fov = FieldOfView(current_cells, trials_to_remove, main_dir, depth)
        fov.analyse(_args.stats, _args.plot, _args.extension)


if __name__ == "__main__":
    program_name = os.path.basename(__file__)
    parser = ArgumentParser(prog=program_name,
                            description='Program to recursively analyse all recordings in an experiment')
#    parser.add_argument("-p", "--channels-to-process", dest="channels_to_process", type=int, nargs='+',
    #  default=[1, 2], help="The list of channels to process. Default: %(default)s")
    parser.add_argument('source_directory', type=str, help='The source directory of the experiment')
    parser.add_argument('-f', '--file-extension', dest='extension', type=str, choices=('png', 'eps', 'pdf', 'svg'),
                        default='png',
                        help='The file extension to save the figures. One of: %(choices)s. Default: %(default)s.')
    parser.add_argument('-l', '--load-cells', dest='load_cells', action='store_true',
                        help='Whether to load the cells (defaults to creating them instead)')
    parser.add_argument('-s', '--stats', action='store_true', help='Whether to do the statistics of each cell')
    parser.add_argument('-p', '--plot', action='store_true', help='Whether to plot each cell')
    parser.add_argument('-c', '--remove-cells', dest='remove_cells', action='store_true',
                        help='Whether to remove some cells from the analysis')
    parser.add_argument('-t', '--remove-trials', dest='remove_trials', action='store_true',
                        help='Whether to remove some trials from the analysis')
    parser.add_argument('--use-bsl-2', dest='use_bsl_2', action='store_true',
                        help='Whether to pool the first and second baselines in the analysis of baseline')

    args = parser.parse_args()
    
    main(args)
