from scipy import io
from probez.util import generic_functions


def load_from_matlab(path):
    quality = io.loadmat(path)
    cluster_groups = generic_functions.flatten_list(quality['cgs'].T)
    unit_quality = quality['uQ']
    isi_violations = quality['isiV'].T
    contamination_rate = quality['cR']
    return cluster_groups, isi_violations, contamination_rate, unit_quality
