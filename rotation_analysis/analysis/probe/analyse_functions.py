import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from analysis.probe.probe_cell import get_cluster, get_all_clusters
from analysis.probe.ctxt_manager_decorator import temp_setattr
from analysis.probe.probe_field_of_view import ProbeFieldOfView
from spike_handling import cluster


def plot_all_stimuli(condition, experiment_id, sp, experiment_directory, cluster_ids, bonsai_io, igor_io, trigger_trace_io):
    for c in tqdm(get_cluster(experiment_id, sp, experiment_directory, bonsai_io, igor_io, trigger_trace_io, cluster_ids)):
        with temp_setattr(c.block, 'current_condition', {'keep': True, 'condition': condition}):
            for t in c.block.kept_trials:
                t.stimulus.plot()
            plt.show()


def plot_resampling_heatmaps(src_dir=None, exp_id=None, successful_triggers=None, sp=None, condition=None, cluster_ids=None):

    if cluster_ids is None:
        cluster_ids = sp.good_cluster_ids

    cells = get_all_clusters(exp_id, sp, src_dir, successful_triggers, cluster_ids=cluster_ids)

    fov = ProbeFieldOfView(cells, None, src_dir, 100)  # TODO: PASS IN A GENERATOR FUNCTION INSTEAD
    fov.plot_spiking_heatmaps(condition)


def plot_sorted_population_heatmaps(sp, path_to_folder_containing_arrays, cluster_ids):

    clusters = [cluster.Cluster(sp, cid) for cid in cluster_ids]
    cluster_depths = [c.best_channel for c in clusters]

    baseline_path = os.path.join(path_to_folder_containing_arrays, 'bsl.npy')
    baseline_array = np.load(baseline_path)

    for condition in ['bsl', 'c_wise', 'c_c_wise']:
        fname = '{}.npy'.format(condition)

        path = os.path.join(path_to_folder_containing_arrays, fname)

        median_array = np.tile(np.median(baseline_array, axis=1), baseline_array.shape[1]).reshape(baseline_array.shape[1], baseline_array.shape[0]).T
        unsorted_array = np.load(path) - median_array

        sorted_array, sorted_depths = sort_array_by(cluster_depths, unsorted_array)

        fig = plt.figure()
        plt.imshow(sorted_array - median_array, vmin=0, vmax=10)
        fig.savefig(os.path.join(path_to_folder_containing_arrays, '{}.eps'.format(condition)), format='eps')
        np.save(os.path.join(path_to_folder_containing_arrays, 'ordered_channel_numbers'), sorted_depths)
        np.savetxt(os.path.join(path_to_folder_containing_arrays, 'ordered_channel_numbers.txt'), sorted_depths,
                   delimiter=',', fmt='%u')


def do_stats(src_dir, exp_id, successful_triggers, sp, cluster_ids):
    clusters = get_all_clusters(exp_id, sp, src_dir, successful_triggers, cluster_ids=cluster_ids)

    fov = ProbeFieldOfView(clusters, None, src_dir, 100)  # TODO: PASS IN A GENERATOR FUNCTION INSTEAD
    fov.analyse(do_stats=True, do_plots=False, extension='eps')


def sort_array_by(idx_to_sort_by, array_to_sort):
    sorted_array = np.full_like(array_to_sort, np.nan)
    order = np.argsort(idx_to_sort_by)
    ordered_idx_to_sort_by = np.sort(idx_to_sort_by)
    for i, order_item in enumerate(order):
        sorted_array[order_item, :] = array_to_sort[i, :]

    return sorted_array, ordered_idx_to_sort_by

