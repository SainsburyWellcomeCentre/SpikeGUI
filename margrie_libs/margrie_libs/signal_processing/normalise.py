import numpy as np


def normalise(src_array):
    """
    Normalise srcArray from 0 to 1
    """
    out_array = src_array.copy()
    out_array -= out_array.min()
    out_array /= out_array.max()
    return out_array


def normalise_around_zero(src_array, bsl_end=15000):
    """
    Normalise trace by centering average(bsl) on 0 and to max == 1
    """
    out_array = src_array.copy()
    
    bsl = np.mean(out_array[:bsl_end])
    out_array -= bsl
    out_array /= out_array.max()
    
    return out_array
