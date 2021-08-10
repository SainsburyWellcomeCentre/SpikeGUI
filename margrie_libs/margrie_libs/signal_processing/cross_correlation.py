import numpy as np


def cross_cor(a1, a2):
    return np.correlate(a1, a2, 'full')    


def _fix_periodic_cross_cor(xcor):
    """
    Will recenter the cross correlogram as the zero shift ends up at the beginning
    otherwise
    """
    mid = int(xcor.size/2)
    first_half = xcor[mid:0:-1]  # FIXME: numpyise
    second_half = xcor[-1:mid:-1]
    centered_cross_cor = np.hstack((first_half, second_half))
    return centered_cross_cor


def periodic_cross_cor(x, y):
    """Periodic correlation, implemented using np.correlate.

    x and y must be real sequences with the same length.
    """
    cross_correlogram = np.correlate(x, np.hstack((y[1:], y)), mode='valid')
    centered_cross_cor = _fix_periodic_cross_cor(cross_correlogram)
    return centered_cross_cor


def normalised_periodic_cross_cor(x, y):
    return periodic_cross_cor((x / np.linalg.norm(x)), (y / np.linalg.norm(y)))


def normalised_periodic_cross_cor_shuffled(x, y):
    x_copy = x.copy()
    np.random.shuffle(x_copy)
    return normalised_periodic_cross_cor(x_copy, y)
