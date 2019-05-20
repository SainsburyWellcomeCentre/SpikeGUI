# coding: utf-8
# from time import time

import math
import pprint

import numpy as np
from matplotlib import pyplot as plt

from resample.resampling import get_velocity_from_position, get_acceleration_from_position, fit_sine_wave
from resample.resampling import make_time_table
from margrie_libs.utils.sample_data import make_cmd


def main():
    n_crossings = 2
    cmd, cmd_x = make_cmd()
    plt.plot(cmd_x, cmd)
    plt.show()

    min_deltax = np.diff(abs(cmd)).max()
    n_degrees = math.ceil(cmd.max() - cmd.min())
    degrees = np.linspace(cmd.min(), cmd.max(), n_degrees / min_deltax)

    output = make_time_table(cmd, cmd_x, degrees, n_crossings)  # Degrees controls the binning of the timetable

    plt.imshow(output, aspect='auto', interpolation='none', origin='lower', extent=(0, 1,
                                                                                    cmd.min(), cmd.max()))
    plt.colorbar()
    plt.show()


def test_derivatives():
    cmd, cmd_x = make_cmd()
    plt.plot(cmd_x, cmd, color='pink', linewidth=10, label='cmd')
    params = fit_sine_wave(cmd_x, cmd)

    def pos_func(t):
        return params['amp'] * np.sin(params['omega'] * t + params['phase']) + params['offset']

    plt.plot(cmd_x, pos_func(cmd_x), linewidth=2, color='red', label='fit')
    # plt.legend()
    # plt.show()

    plt.plot(cmd_x, get_velocity_from_position(cmd, cmd_x), color='blue', label='vel')
    # plt.legend()
    # plt.show()

    plt.plot(cmd_x, get_acceleration_from_position(cmd, cmd_x), color='green', label='acc')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
    pp = pprint.PrettyPrinter(indent=4)

    n_crossings = 2  # TODO: extract from wave params ?
    cmd, cmd_x = make_cmd()
    plt.plot(cmd_x, cmd)
    plt.show()

    # POSITION
    min_deltax = np.diff(abs(cmd)).max()
    degree_spacing = 30
    n_degrees = int(round((cmd.max() - cmd.min()) / degree_spacing))
    print(cmd.min(), cmd.max(), degree_spacing, n_degrees)
    degrees = np.linspace(cmd.min(), cmd.max(), n_degrees+1)
    print(n_degrees, ['{:.2f}'.format(d) for d in degrees])
    positions_table = make_time_table(cmd, cmd_x, degrees, n_crossings)  # Degrees controls the binning of the timetable

    # DISTANCE
    distances_table = positions_table.copy()
    distances_table = np.sort(distances_table, axis=0)  # TEST:

    # VELOCITY
    velocity = get_velocity_from_position(cmd, cmd_x)
    min_vel = velocity.min()
    max_vel = velocity.max()
    velocity_spacing = 20
    n_vels = int(round((velocity.max() - velocity.min()) / velocity_spacing))
    velocities = np.linspace(min_vel, max_vel, n_vels+1)
    print(n_vels, ['{:.2f}'.format(v) for v in velocities])
    velocities_table = make_time_table(velocity, cmd_x, velocities, n_crossings)
    # assert velocities.min() >= min_vel
    # assert velocities.max() <= max_vel

    # ACCELERATION
    acceleration = get_acceleration_from_position(cmd, cmd_x)
    min_acc = acceleration.min()
    max_acc = acceleration.max()
    acceleration_spacing = 5
    n_accs = int(round((acceleration.max() - acceleration.min()) / acceleration_spacing))
    accelerations = np.linspace(min_acc, max_acc, n_accs+1)
    print(n_accs, ['{:.2f}'.format(a) for a in accelerations])
    accelerations_table = make_time_table(acceleration, cmd_x, accelerations, n_crossings)


# Make position table DONE
# Flip odd rows of position to make distance  DONE
# Make velocity curve  DONE
# Make acceleration curve  DONE
# Make degrees, velocities, accelerations with proper spacing  TODO: define degrees
# make corresponding timetables  DONE

# TODO: how do we deal with segment transitions (velocity)
# TODO: add statistics
