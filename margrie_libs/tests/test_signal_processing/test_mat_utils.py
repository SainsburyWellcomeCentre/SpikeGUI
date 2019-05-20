import pytest
from src.signal_processing import mat_utils

import numpy as np


@pytest.fixture
def trace(request):
    n_pnts = request.param
    return np.random.rand(n_pnts)

@pytest.mark.parametrize("decimation_factor", (10, 20, 30))
def test__get_decimate_new_n_pnts(decimation_factor):
    trace = np.random.rand(1001)
    with pytest.raises(ValueError):
        mat_utils._get_decimate_new_n_pnts(trace, decimation_factor * 2, 'strict')
    assert mat_utils._get_decimate_new_n_pnts(trace, decimation_factor * 2, 'drop') == (trace.size // (decimation_factor*2)) * 2


@pytest.mark.parametrize("decimation_factor", (10, 20, 30))
def test_decimate_x(decimation_factor):
    trace = np.random.rand(1001)
    decimate_x_size = mat_utils.decimate_x(trace, decimation_factor).size
    decimate_size = mat_utils.decimate(trace, decimation_factor).size
    assert decimate_x_size == decimate_size


@pytest.mark.parametrize("ratio", (10, 20, 40))
@pytest.mark.parametrize("trace", (100, ), indirect=True)
def test_decimate(ratio, trace):
    assert trace.mean() == pytest.approx(mat_utils.decimate(trace, ratio).mean(), 0.1)


@pytest.mark.parametrize("trace", (100, 150), indirect=True)  # TODO: test with odd number and check which segment longest
def test_cut_in_half(trace):
    segments = mat_utils.cutInHalf(trace)
    assert len(segments) == 2
    assert len(segments[0]) == len(segments[1])
#
# def avg(mat):
#     """
#     Returns the vector corresponding to mat averaged accross 2nd and 3rd dims.
#     Assumes that the matrix is all filled (no NaN since avg of avg).
#     """
#     if __debug__:
#         print(mat.shape)
#     if mat.ndim > 1:
#         return avg(np.average(mat, axis=1))
#     else:
#         return mat
#
#
# def avg_waves(waves):
#     """
#     Transforms the input list into a numpy array and returns the average across the first dimension
#
#     :param list waves:
#     :return:
#     """
#     matrix = np.array(waves)
#     return matrix.mean(0)  # TODO: check dimension


def test_out_of_place_shuffle():
    test_array = np.random.rand(33, 42, 3)
    result = mat_utils.outOfPlaceShuffle(test_array)
    assert result.shape == test_array.shape
    assert result.sum() == pytest.approx(test_array.sum(), 0.000000000001)
    assert not (np.array_equal(result, test_array))

