import numpy as np


def find_levels(wave, threshold):
    mask = wave < threshold  # TODO: do for positive or negative
    starts_mask = find_range_starts(mask)
    indices = np.where(starts_mask == 1)  # convert to indices
    return indices[0]


def find_range_starts(src_mask):
    """
    For a binary mask of the form:
    (0,0,0,1,0,1,1,1,0,0,0,1,1,0,0,1)
    returns:
    (0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1)
    """
    tmp_mask = np.logical_and(src_mask[1:], np.diff(src_mask))
    output_mask = np.hstack(([src_mask[0]], tmp_mask))  # reintroduce first element
    return output_mask


def find_level_increase(trace, value):  # TODO: find better numpy built in function (try numpy.where(condition)[0][0]
    for i in range(1, len(trace)):
        if trace[i-1] < value <= trace[i]:
            return i
    raise StopIteration("value {} not found".format(value))


def find_level_decrease(trace, value):  # TODO: find better numpy built in function (try numpy.where(condition)[0][0]
    for i in range(1, len(trace)):
        if trace[i-1] > value >= trace[i]:
            return i
    raise StopIteration("value {} not found".format(value))
