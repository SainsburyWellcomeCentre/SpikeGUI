import numpy as np


def flatten(lst):
    """
    this is the fastest implementation

    >>> flatten([[1, 2, 3], [2, 4, 6], [4, 6, 8]])
    [1, 2, 3, 2, 4, 6, 4, 6, 8]
    """
    out = []
    for sublist in lst:
        out.extend(sublist)
    return out


def sort_by(list_to_sort, list_to_sort_by, descend=True):
    """
    sort one list by another list
    :param list list_to_sort:
    :param list list_to_sort_by:
    :param bool descend:
    :return list sorted_list:
    """

    sorted_lists = [(cid, did) for did, cid in sorted(zip(list_to_sort_by, list_to_sort))]
    if descend:
        sorted_lists = sorted_lists[::-1]
    ordered = np.array(sorted_lists)[:, 0]
    ordered_by = np.array(sorted_lists)[:, 1]

    return list(ordered), list(ordered_by)
