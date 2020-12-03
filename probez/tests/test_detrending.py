import numpy as np


def permute_traces(traces, shift=2):
    return np.roll(traces, shift, axis=1)

