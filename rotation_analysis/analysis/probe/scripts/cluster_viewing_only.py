import matplotlib
matplotlib.use('QT5Agg')


import os

import matplotlib.pyplot as plt
from tqdm import tqdm
from spike_handling import spike_io, cluster

SRC_DIR = os.path.expanduser('/home/skeshav/swc-winstor/margrie/Sepi/Probe/CA326_C/CA326_C_day3/processed/denoised_and_filtered2/')


def main(experiment_id, raw_traces_extension='g0_t0_0_to_160778264_32_33_34_35_13000000_to_160778264.imec.ap.bin', n_chan=32):

    experiment_directory = SRC_DIR

    traces_filename = '{}_{}'.format(experiment_id, raw_traces_extension)
    traces_path = os.path.join(experiment_directory, traces_filename)
    spike_io = spike_io.SpikeStruct(experiment_directory, traces_path, n_chan)
    clusters = [cluster.Cluster(spike_io, cid) for cid in spike_io.good_cluster_ids]

    for c in clusters:
        fig = plt.figure(facecolor='w', figsize=(8, 3))
        plt.plot(c.best_channel_waveforms, alpha=0.2)
        plt.plot(c.best_channel_waveforms.mean(1), linewidth=2)
        plt.show()
        save_dir = os.path.join(experiment_directory, 'average_waveforms')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'cluster_{}_channel_{}.{}'.format(c.cluster_id, c.depth,'png'))
        fig.savefig(save_path, format='png')


if __name__ == '__main__':
    tqdm(main('CA326_C_day3'))
