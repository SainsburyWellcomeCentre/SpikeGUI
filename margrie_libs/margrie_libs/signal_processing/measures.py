import numpy as np

from margrie_libs.margrie_libs.signal_processing.filters import high_pass


def get_sd(trace, n_pnts_high_pass_filter):
    """
    Get the SD of the detrended trace

    :param trace:
    :param n_pnts_high_pass_filter:
    :return:
    """
    return np.std(high_pass(trace, n_pnts_high_pass_filter))
