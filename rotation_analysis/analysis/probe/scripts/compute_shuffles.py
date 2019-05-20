import numpy as np
from analysis import pseudo_shuffle_analysis

def main():
    path_to_hm = "/home/slenzi/Desktop/CA242_B/spiking_heatmaps/CA_242_B_90_black/condition_CA_242_B_90_black_position_spiking_heatmap.csv"
    spiking_mat = np.loadtxt(open(path_to_hm, "rb"), delimiter=",", skiprows=1)[:, 1:]
    pseudo_shuffle_analysis.plot_sd_shuffles(spiking_mat, )

if __name__ == '__main__':
    main()