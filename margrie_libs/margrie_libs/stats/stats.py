from rpy2.robjects import FloatVector
from scipy import stats

from rpy2.robjects.packages import importr

r_stats = importr("stats")


def paired_t_test(vect1, vect2):
    try:
        p_val = stats.ttest_rel(vect1, vect2)[1]
    except ValueError as err:
        raise ValueError("vect1: {} ({} elements), vect2: {} ({} elements); {}"
                         .format(vect1, len(vect1), vect2, len(vect2), err))
    return p_val


def wilcoxon(vect1, vect2):  # TODO: use closure as same pattern as above
    if len(vect1) == 0 or len(vect2) == 0:
        return float('nan')
    try:
        results = r_stats.wilcox_test(FloatVector(vect1), FloatVector(vect2), paired=True, exact=True)
        # p_val = stats.wilcoxon(vect1, vect2)[1]
    except ValueError as err:
        raise ValueError("vect1: {} ({} elements), vect2: {} ({} elements); {}"
                         .format(vect1, len(vect1), vect2, len(vect2), err))
    wilcox_stat = results[results.names.index('statistic')][0]
    p_val = results[results.names.index('p.value')][0]
    return p_val
