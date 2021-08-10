import numpy as np
from scipy.signal import savgol_filter


def low_pass(vector, n_pnts=51):
    polynomial_order = 3
    return savgol_filter(vector, n_pnts, polynomial_order)


def count_points_between_values(pnt_a, pnt_b, src_array):
    start = min(pnt_a, pnt_b)
    end = max(pnt_a, pnt_b)
    src_array = np.array(src_array, ndmin=1)
    levels = src_array[(src_array >= start) & (src_array < end)]
    n_levels = len(levels)
    return n_levels
