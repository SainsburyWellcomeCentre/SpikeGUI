def rewrite_array_as_list_for_plotting(array):
    """function for restructuring data so it can be plotted more efficiently by matplotlib. Converts the input array
    into a list of lines separated by None values. NOTE: this means that all lines are together represented as a single
    line. Using this data structure to plot will, therefore, reduce the extent to which individual lines can be accessed
    by matplotlib.

==================   ==================
line number              data
==================   ==================
0                    1.8, 1.2, 1.6, 0.7
1                    0.8, 1.2, 1.5, 0.9
==================   ==================

will become

==================   ============================================
line number              data
==================   ============================================
0                    1.8, 1.2, 1.6, 0.7, None, 0.8, 1.2, 1.5, 0.9
==================   ============================================

:param array array: the data you wish to plot
:return x: the data as one None-separated line
:return y: a list of indices to plot alongside

"""

    y = []
    x = []
    for item in array:  # FIXME: shape restriction/adjustment necessary
        y.extend(list(item))
        y.append(None)
        x.extend(list(range(len(item))))
        x.append(None)
    return x, y