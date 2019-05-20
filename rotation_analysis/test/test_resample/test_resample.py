
import numpy as np


def test_interpolation_result():
    curr_deg = [100.001256703]
    sub_x = [0.93740937, 0.93750938]
    sub_y = [99.99411959, 100.00352355]
    interpolated_value = np.interp(curr_deg, sub_y, sub_x)
    assert sub_x[0] < interpolated_value < sub_x[1]
