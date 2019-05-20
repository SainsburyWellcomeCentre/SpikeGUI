import math

import numpy as np
import pytest
from analysis.event_detection import events_collection

from analysis.stimulus import CalciumRotationStimulus


class FakeStim(CalciumRotationStimulus):
    def __init__(self, sampling_interval, n_repeats=2):
        super(CalciumRotationStimulus, self).__init__(sampling_interval)
        self.use_bsl_2 = False
        self.min_range_n_points = 10
        self.n_repeats = n_repeats


@pytest.fixture
def empty_stim():
    stim = FakeStim(0.001)
    stim.cmd = np.zeros(100)
    return stim


def test_get_spin_start(empty_stim):
    spin_start_p = 50
    empty_stim.cmd[spin_start_p:] = 1
    assert empty_stim.get_spin_start() == spin_start_p


def test_get_spin_end(empty_stim):
    spin_end_p = 50
    empty_stim.cmd[:spin_end_p+1] = 1
    assert empty_stim.get_spin_end() == spin_end_p


@pytest.fixture
def sinusoidal_stim():
    stim = FakeStim(0.001)
    cmd, cmdx = make_cmd(1000)
    stim.cmd = cmd
    return stim


def make_cmd(n_points=1000, duration=21, phy=0):
    cmd_x = np.linspace(0, duration, n_points)
    max_angle = 180
    freq = 0.07075
    cmd = max_angle * np.sin(2*np.pi*freq*cmd_x + phy)
    return cmd, cmd_x


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


@pytest.mark.xfail
def test_get_n_repeats(sinusoidal_stim):
    n_repeats = sinusoidal_stim.get_n_repeats()
    # plt.plot(sinusoidal_stim.cmd); plt.show()
    peaks = sinusoidal_stim.get_cmd_peaks()
    full_cycle_start = peaks[0]
    full_cycle_end = peaks[1]
    sub_command = sinusoidal_stim.cmd[full_cycle_start:full_cycle_end]
    for i in range(179):
        assert len(np.where(np.diff(np.sign(sub_command - i)))[0]) == n_repeats


@pytest.mark.parametrize("peak_width", [1, 2, 5, 7, 9, 6])
def test_get_cmd_peaks(empty_stim, peak_width):
    peaks = list(range(10, 90, 10))
    for p in peaks:
        empty_stim.cmd[p:p+peak_width] = 1
    half_n_points_peak = int(math.floor(peak_width / 2))
    expected_peak_locs = [p + half_n_points_peak for p in peaks]
    if peak_width % 2 == 0:
        expected_peak_locs = [p-1 for p in expected_peak_locs]
    print('Peak width = {}'.format(peak_width))
    assert list(empty_stim.get_cmd_peaks()) == expected_peak_locs


def test_get_clockwise_ranges(sinusoidal_stim):
    pass


def test_get_counter_clockwise_ranges(sinusoidal_stim):
    pass


@pytest.fixture
def resampling_test_data(sinusoidal_stim):
    max_angle = 180
    min_angle = -180
    step_size = 30
    lower_ranges = range(min_angle, max_angle, step_size)
    upper_ranges = range(min_angle + step_size, max_angle + step_size, step_size)

    cmd = sinusoidal_stim.cmd
    all_events = []
    for i, (lower, upper) in enumerate(zip(lower_ranges, upper_ranges)):
        bin_mask = np.logical_and(cmd < upper, cmd > lower)
        bin_locs = np.where(bin_mask)[0]
        spikes = np.random.choice(bin_locs, i+1)
        all_events.extend(spikes)

    return events_collection.EventsCollection(all_events)


def test_resampling(sinusoidal_stim, resampling_test_data):
    max_angle = 180
    min_angle = -180
    step_size = 30
    lower_ranges = range(min_angle, max_angle, step_size)
    upper_ranges = range(min_angle + step_size, max_angle + step_size, step_size)
    n_steps = len(list(lower_ranges))
    freqs_table = get_frequency_table(sinusoidal_stim.cmd, sinusoidal_stim.x, resampling_test_data)
    assert freqs_table == np.arange(1, n_steps)  # TODO: this entire thing probably needs a dummy class or two
