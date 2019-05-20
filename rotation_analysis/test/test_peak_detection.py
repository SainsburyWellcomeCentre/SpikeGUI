import numpy as np
from matplotlib import pyplot as plt
from scipy import io
from scipy.signal import resample
from resample.resampling import get_velocity_from_position
from analysis import stimulus
from margrie_libs.signal_processing.mat_utils import find_sine_peaks

path = '/home/slenzi/spine_shares/buffer/sepi_rotation/CA23_3/cmd/waveform_180.mat'

cmd = io.loadmat(path)['waveform']['data'][0][0]
cmd_sampling_interval = 0.001
cmd = cmd.squeeze()
cmd *= 20

resampling_factor = 20
cmd_small = resample(cmd, int(len(cmd) / resampling_factor))
plt.plot(cmd_small)

resampled_sampling_interval = cmd_sampling_interval * resampling_factor
cmd_x = np.linspace(0, len(cmd_small) * resampled_sampling_interval, len(cmd_small))

#stimulus.RotationStimulus.get_counter_clock_wise_ranges(cmd)

plt.plot(cmd_x, cmd_small)

coarse_vel = np.diff(cmd_small) / resampled_sampling_interval
coarse_vel = np.hstack((coarse_vel, [coarse_vel[-1]]))
plt.plot(cmd_x, coarse_vel)

pks = find_sine_peaks(cmd_small)
print(np.array(pks) * resampled_sampling_interval)

vel = get_velocity_from_position(cmd_small, cmd_x, pks[0], pks[-1])


franken_vel = coarse_vel.copy()
franken_vel[pks[0] - 1:pks[-1] -1] = vel[pks[0]:pks[-1]]


for i, p in enumerate(cmd):
    if p != 0:
        spin_start = int(i / resampling_factor)
        break


for i in reversed(range(len(cmd))):
    if cmd[i] != cmd[-1]:
        spin_end = int(i / resampling_factor)
        break

mask = np.array(vel[spin_start:spin_end] < 0, dtype=np.int64)

def __find_range_starts_and_ends(src_mask):
    """
    For a binary mask of the form:
    (0,0,0,1,0,1,1,1,0,0,0,1,1,0,0,1)
    returns:
    (0,0,0,1,0,1,0,1,0,0,0,1,1,0,0,1)
    """
    output_mask = np.hstack(([src_mask[0]], np.diff(src_mask)))  # reintroduce first element
    starts = np.where(output_mask == 1)[0] + spin_start
    ends = np.where(output_mask == -1)[0] + spin_end
    return starts, ends

print(__find_range_starts_and_ends(mask))
# (array([1766, 2121]), array([3181, 3887]))

plt.plot(vel)

plot_mask = np.zeros(cmd_small.shape)
plot_mask[1414:2827] = mask
plt.plot(plot_mask*180)
plt.plot(cmd_small)
plt.plot(vel)

