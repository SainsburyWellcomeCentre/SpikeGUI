import numpy as np
from spike_handling import spike_io
from spike_handling.cluster import Cluster
from util.generic_functions import sort_by
import matplotlib.pyplot as plt
import os


def main():
    src_dir = '/home/slenzi/Desktop/CA242_B/'
    traces_path = '/home/slenzi/Desktop/sepi_subset_probe_data/CA242_B_g0_t0_shortened_sample_22000000_to_168000000.imec.ap.bin'
    base_path = '/home/slenzi/Desktop/CA242_B/population_histograms_heatmap_npy_arrays/CA_242_B_90_uniform'
    sp = spike_io.SpikeIo(src_dir, traces_path, 385)
    cluster_ids = sp.good_cluster_ids
    clusters = [Cluster(sp, cid) for cid in cluster_ids]
    cluster_depths = [c.best_channel for c in clusters]

    baseline_path = os.path.join(base_path, 'bsl_population_heatmap.npy')
    baseline_array = np.load(baseline_path)

    for condition in ['bsl', 'c_wise', 'c_c_wise']:
        fname = '{}_population_heatmap.npy'.format(condition)

        path = os.path.join(base_path, fname)
        #unsorted_array = (np.full((10, 10), 1)*np.arange(10)).T

        median_array = np.tile(np.median(baseline_array, axis=1), 30).reshape(30, 57).T
        unsorted_array = np.load(path) - median_array

        sorted_array, sorted_depths = sort_array_by(cluster_ids, cluster_depths, unsorted_array)
        # plt.imshow(unsorted_array, vmin=0, vmax=70)
        # plt.colorbar()
        # plt.show()
        fig = plt.figure()
        plt.imshow(sorted_array - median_array, vmin=0, vmax=10)
        fig.savefig(os.path.join(base_path, '{}.eps'.format(condition)), format='eps')
        np.save(os.path.join(base_path, 'ordered_channel_numbers'), sorted_depths)
        np.savetxt(os.path.join(base_path, 'ordered_channel_numbers.txt'), sorted_depths, delimiter=',', fmt='%u')



def sort_array_by(idx_to_sort, idx_to_sort_by, array_to_sort):  # TODO: extract to margrielibs

    sorted_array = np.full_like(array_to_sort, np.nan)
    #ordered_idx, ordered_by_idx = sort_by(idx_to_sort, idx_to_sort_by)
    order = np.argsort(idx_to_sort_by)
    ordered_idx_to_sort_by = np.sort(idx_to_sort_by)
    for i, order_item in enumerate(order):
        sorted_array[order_item, :] = array_to_sort[i, :]

    # for i, (cid_unsorted, cid_sorted_by_depth) in enumerate(zip(idx_to_sort, ordered_idx)):
    #     idx = np.where(ordered_idx == cid_unsorted)[0][0]
    #     sorted_array[idx, :] = array_to_sort[i, :]

    return sorted_array, ordered_idx_to_sort_by


def test_sort_matrix_by():
    idx_to_sort = np.arange(10)
    idx_to_sort_by = [10, 4, 5, 6, 2, 1, 3, 9, 8, 7]
    array_to_be_sorted = (np.full((10, 10), 1) * np.arange(10)).T

    sorted_array = sort_array_by(idx_to_sort, idx_to_sort_by, array_to_be_sorted)

    assert sorted_array[0][0] == 0
    assert sorted_array[1][0] == 7


if __name__ == '__main__':
    main()