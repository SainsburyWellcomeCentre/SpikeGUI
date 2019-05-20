# coding: utf-8

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from tqdm import trange

DIRECTION_NUMS = {
    'clockwise': 1,
    'counter_clockwise': 0
}

DEBUG = False


def reset_df(df):
    """

    :param pd.DataFrame df:
    """
    df.direction = ''
    if 'velocity' in df.columns:
        df.velocity = 0
    if 'acceleration' in df.columns:
        df.acceleration = 0
    df.vm = 0


def clean_df(frame):
    if 'Unnamed: 0' in frame.keys():
        frame = frame.drop('Unnamed: 0', axis=1)
    return frame


def duplicate_empty(df):
    out_df = df.copy()  # TODO: see id needs deepcopy
    reset_df(out_df)
    return out_df


def get_n_samples_per_column(df, column_name):
    """e.g. number of repeats of same velocity, acceleration ..."""
    # uses the fact that they have the same number of element per repeat
    # So [0] has same value as the others
    return df.groupby([column_name]).count().vm[0]


def get_means(df, var_name='velocity'):
    # df_cp = df.copy()
    # df_cp['vm'] = abs(df_cp['vm'])  # WARNING: no need if split velocities
    # print("Minimum value: {}".format(df_cp['vm'].min()))
    means = df.groupby([var_name])['vm'].mean()
    return means


def filter_direction(frame, direction):
    """
    >>> f = filter_direction(frame, 'clockwise')
    """
    return frame[frame['direction'] == direction].copy()


def filter_rotation(frame):
    """
    All but the baseline
    """
    return frame[frame['direction'] != 'none'].copy()


def pseudo_shuffle_mat(ref_var, mat, replace=False, debug=False):
    """
    Shuffles the data but keeps the time information (i.e. shuffles the velocity while keeping the
    time information intact)

    :param np.array ref_var: shape: n_accelerations
    :param np.array mat: shape: (n_trials, n_accelerations)
    :return: shuffled_mat
    :rtype: np.array
    """
    shuffled_mat = np.zeros(mat.shape)

    n_accs, n_samples = mat.shape
    # TODO: use replace=True for velocity as well
    seeds = np.random.choice(np.arange(len(ref_var)), n_samples, replace=replace)

    for i in range(n_samples):
        seed = seeds[i]
        trial_vms = mat[:, i]
        shuffled_vms = np.hstack((trial_vms[seed:], trial_vms[:seed]))  # FIXME: could be done with np.roll (TEST:)
        if debug: plt.plot(shuffled_vms)  # (with fake traces)
        shuffled_mat[:, i] = shuffled_vms.copy()
    if debug: plt.show()
    return shuffled_mat


def plot_shuffles_histogram(ref_var, mat, n_iter=100, replace=False, do_r_squared=True, plot=False):
    """

    :param np.array ref_var: shape: n_accelerations
    :param np.array mat: shape: (n_trials, n_accelerations)
    :param int n_iter: The number of times to shuffle
    :return: data_means, shuffles_means, randomised_sds (shuffles_means = list of n_trials * n_accelerations) others: n_accelerations
    """
    randomised_sds = np.zeros(n_iter)
    shuffles_means = []
    rs_squared = []
    for i in trange(n_iter, desc='Shuffling'):
        shuffled_mat = pseudo_shuffle_mat(ref_var, mat, replace=replace)
        shuffled_mat_means = shuffled_mat.mean(axis=1)
        shuffles_means.append(shuffled_mat_means)  # across trials
        randomised_sds[i] = shuffled_mat_means.std()
        if do_r_squared:
            r_squared, _ = _get_r_squared(ref_var, shuffled_mat_means, plot=True)
            rs_squared.append(r_squared)
        if plot:
            plt.hist(shuffled_mat_means, 50, histtype='step', linewidth=0.25, color='gray')
    data_means = mat.mean(axis=1)
    if plot:
        plt.hist(data_means, 50, histtype='step', linewidth=2, color='red')
        plt.show()
    return data_means, shuffles_means, randomised_sds, np.array(rs_squared, dtype=np.float64)


def plot_sds(means, randomised_sds, show=True):
    real_sd = means.std()
    plt.scatter(np.zeros(len(randomised_sds)), randomised_sds, color='gray', label='rndm')
    plt.scatter(0, real_sd, color='red', label='data')
    if show:
        plt.show()


def plot_sds_histo(means, randomised_sds):
    real_sd = means.std()
    plt.hist(randomised_sds, 100, histtype='step', linewidth=0.5, color='gray')
    plt.plot((real_sd, real_sd), (0, 1), color='red')
    plt.show()


# REFACTOR: Extract to other module
def get_percentiles(means, randomised_metric, percentage=5, two_tailed=False, verbose=True):  # FIXME: not ready for 2 tailed (only 1 tailed >)
    real_sd = means.std()
    if two_tailed:
        percentage /= 2
    left_tail = np.percentile(randomised_metric, percentage)
    right_tail = np.percentile(randomised_metric, 100 - percentage)

    percentile = stats.percentileofscore(randomised_metric, real_sd, kind='rank')
    # if percentile > 50:  # FIXME: only for two tailed
    p_val = (100 - percentile) / 100
    # else:
    #     p_val = percentile / 100
    if two_tailed:
        p_val *= 2  # FIXME: check
    if verbose:
        print("Left tail: {}".format(left_tail))
        print("Real value: {}".format(real_sd))
        print("Right tail: {}".format(right_tail))
    else:
        return left_tail, real_sd, right_tail, percentile, p_val


def sanitise_p_value(n_iter, p_val):
    """
    Fixes p value too small or == 0 when computed form distribution
    (if not even 1 sample on the other side of measured value)

    :param int n_iter: The number of iterations that were performed to compute the distribution
    :param float p_val:
    :return:
    """
    if p_val == 0:
        p_val = "< {}".format(1 / n_iter)
    else:
        p_val = "{:.4f}".format(p_val)
    return p_val


def _get_r_squared(x, y, plot=False, plot_color='black', linewidth=0.25, legend=False):
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    start_y = slope * x[0] + intercept
    end_y = slope * x[-1] + intercept
    fit = np.linspace(start_y, end_y, len(x))
    if plot:
        if legend:
            plt.plot(x, fit, color=plot_color, linewidth=linewidth, label='fit')
        else:
            plt.plot(x, fit, color=plot_color, linewidth=linewidth)
    return r_value ** 2, p_value


def x_ticks_off():
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')


def validate_names(direction, ref_var_name):
    assert direction in ('clockwise', 'counter_clockwise'), \
        'Unknown direction, got {} expected on of "clockwise", "counter_clockwise"'.format(direction)
    assert ref_var_name in ('velocity', 'acceleration'), \
        'Unknown direction, got {} expected on of "velocity", "acceleration"'.format(ref_var_name)


def get_single_direction_mat(mat, direction_mat, ref_var_name, direction):
    n_refs, n_samples = mat.shape
    mid = int(n_refs / 2)
    if ref_var_name == 'velocity':
        if direction == 'counter_clockwise':  # negative velocities = counter_clockwise
            # return mat[:, :int(n_samples / 2)]  # DEBUG: remove (and do random.choice(np.arange(n_samples), int(n_samples/2))
            return mat[:mid, :]
        elif direction == 'clockwise':
            # return mat[:, int(n_samples / 2):]  # DEBUG: remove (and do random)
            return mat[mid:, :]
    elif ref_var_name == 'acceleration':
        single_direction_mat = mat[direction_mat == DIRECTION_NUMS[direction]]
        single_direction_mat = single_direction_mat.reshape(n_refs, int(n_samples / 2))
        return single_direction_mat


def get_single_direction_ref(ref_var, ref_var_name, direction):
    # ref_var_name = 'acceleration'     # DEBUG: remove

    n_refs = ref_var.shape[0]
    mid = int(n_refs / 2)
    if ref_var_name == 'velocity':
        if direction == 'counter_clockwise':  # negative velocities = counter_clockwise
            return ref_var[:mid]
        elif direction == 'clockwise':
            return ref_var[mid:]
    elif ref_var_name == 'acceleration':  # entire set of accelerations for each direction
        return ref_var


def shuffle_and_analyse_direction(axarr, mat, ref_var, direction_mat, ref_var_name, n_shuffles, side_hist_bins, direction):
    shuffle_hist_plot_name = '{}_hist_shuffle'.format(direction)
    fit_hist_plot_name = '{}_hist_fit'.format(direction)
    if ref_var_name == 'acceleration' and direction == 'clockwise':
        main_plot_name = 'bottom_plot'
    else:
        main_plot_name = 'top_plot'
    x_offsets = {'counter_clockwise': 0.135,  # left  # TODO: improve
                 'clockwise': 0.815}  # right

    plt.axes(axarr[shuffle_hist_plot_name])
    validate_names(direction, ref_var_name)
    mat_single_direction = get_single_direction_mat(mat, direction_mat, ref_var_name, direction)
    ref_var_single_direction = get_single_direction_ref(ref_var, ref_var_name, direction)

    # TODO: constrain ref_var to -71, 71 for acceleration
    plt.axes(axarr[main_plot_name])
    means, shuffles_means, randomised_sds, randomised_rs_squared = plot_shuffles_histogram(ref_var_single_direction,
                                                                                           mat_single_direction,
                                                                                           n_shuffles)

    # side distribution plot
    plt.axes(axarr[shuffle_hist_plot_name])
    plt.hist(randomised_sds, side_hist_bins, histtype='step', orientation='horizontal', color='gray', label='sds')
    plt.scatter(0, means.std(), color='red', label='data')
    plt.ylim((0, 2.5))
    plt.xlabel('{}'.format(direction))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1,
               mode="expand", borderaxespad=0.)

    plt.axes(axarr[fit_hist_plot_name])
    plt.hist(randomised_rs_squared, np.linspace(0, 1, 100), histtype='step', orientation='horizontal', color='green', label='rndm r^2')
    plt.ylim((0, 1))

    left_tail, real_sd, right_tail, _, p_val = get_percentiles(means, randomised_sds, verbose=False)
    direction_significant = p_val < 0.05
    p_val = sanitise_p_value(n_shuffles, p_val)

    plt.figtext(x_offsets[direction], 0.85, "0.05 threshold: {:.2f}\ndata sd: {:.2f}\np_value: {}"  # TODO: use axis.text instead
                .format(right_tail, real_sd, p_val))
    x_ticks_off()

    # main plot
    axis = axarr[main_plot_name]
    plt.axes(axis)
    n_example_shuffles = 1 if DEBUG else 20
    for shuffles_mean in shuffles_means[:n_example_shuffles]:
        plt.plot(ref_var_single_direction, shuffles_mean, color='gray', linewidth=0.1, alpha=0.3)
    # Mean at the end to be on top
    plt.plot(ref_var_single_direction, mat_single_direction.mean(axis=1), color='red', linewidth=2, label='mean')

    # R^2 labels
    x_offset = ref_var_single_direction[int(len(ref_var_single_direction) / 3)]
    y_range = axis.get_ylim()
    y_offset = 0.85 * (y_range[1] - y_range[0]) + y_range[0]
    # y_offset_2 = 0.80 * (y_range[1] - y_range[0]) + y_range[0]
    r_squared, _ = _get_r_squared(ref_var_single_direction,
                                  mat_single_direction.mean(axis=1),
                                  plot=True,
                                  plot_color='green',
                                  linewidth=2)
    axis.text(x_offset, y_offset, "R^2 {0}: {1:.2f}".format(direction, r_squared))
    # axis.text(x_offset, y_offset_2, "P value {0}: {1:.6f}".format(direction, r_squared_p_value))

    # R squared histo
    plt.axes(axarr[fit_hist_plot_name])
    plt.scatter(0, r_squared, color='green', label='r^2')
    percentile = stats.percentileofscore(randomised_rs_squared, r_squared, kind='rank')  # FIXME: check if 1 or 2 tailed
    # if percentile > 50:
    r_squared_p_val = (100 - percentile) / 100
    # else:
    #     r_squared_p_val = percentile / 100
    print("R squared distro p value: {}".format(r_squared_p_val))
    direction_significant_r2 = r_squared_p_val < 0.05
    r_squared_p_val = sanitise_p_value(n_shuffles, r_squared_p_val)
    plt.figtext(x_offsets[direction], 0.82, 'r^2 p_val = {}'.format(r_squared_p_val))

    return direction_significant, direction_significant_r2, shuffles_means


def analyse_split(ref_var, mat, direction_mat, dest_name='', ref_var_name='velocity', n_shuffles=10000):
    if ref_var_name == 'velocity':
        ref_var = np.linspace(-80, 80, mat.shape[0])
        # TODO: accelerations should be with even values (resample from max)
    average_vms = mat.mean(axis=1)

    axarr, fig = make_figure(dest_name, ref_var_name)

    side_hist_bins = np.linspace(0, 2.5, 100)

    significant_counter_clockwise, significant_r2_counter_clockwise, _ = shuffle_and_analyse_direction(axarr, mat, ref_var, direction_mat,
                                                                     ref_var_name,
                                                                     n_shuffles, side_hist_bins, 'counter_clockwise')

    significant_clockwise, significant_r2_clockwise, shuffles_means = shuffle_and_analyse_direction(axarr, mat, ref_var, direction_mat,
                                                                          ref_var_name,
                                                                          n_shuffles, side_hist_bins, 'clockwise')

    # To have 1 label only
    plt.plot(ref_var[0], shuffles_means[0][0], color='gray', linewidth=0.1, alpha=0.3, label='shuffled')

    plt.axes(axarr['top_plot'])

    plot_min = average_vms.min()
    plot_max = average_vms.max()
    plt.plot(np.zeros(2), (plot_min, plot_max), color='black', linestyle='dotted')
    plt.xlabel(ref_var_name)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0.)
    return fig, significant_counter_clockwise, significant_clockwise, significant_r2_counter_clockwise, significant_r2_clockwise


def make_figure(dest_name, ref_var_name):
    fig = plt.figure(figsize=(18, 18))

    if ref_var_name == 'velocity':
        n_rows = 1
    else:
        n_rows = 2

    grid_shape = (n_rows, 7)
    axarr = {
        'counter_clockwise_hist_shuffle': plt.subplot2grid(grid_shape, (0, 0), rowspan=n_rows),  # counter on left
        'clockwise_hist_shuffle': plt.subplot2grid(grid_shape, (0, 6), rowspan=n_rows),
        'top_plot': plt.subplot2grid(grid_shape, (0, 1), colspan=5),  # top main plot
    }
    if n_rows > 1:
        axarr['bottom_plot'] = plt.subplot2grid(grid_shape, (1, 1), colspan=5)  # bottom main plot

    axarr['counter_clockwise_hist_fit'] = axarr['counter_clockwise_hist_shuffle'].twinx()
    axarr['clockwise_hist_fit'] = axarr['clockwise_hist_shuffle'].twinx()

    plt.suptitle(dest_name)
    return axarr, fig


# #######################  TESTS  #############################
def make_fake_frame(n_repeats=10):
    """

    :param int n_repeats: has to be even
    :return:
    """
    velocities = np.repeat(np.arange(0, 80, 1), n_repeats)  # 4 repeats
    n_velocities = len(set(velocities))
    n_directions_per_vel = int(n_repeats / 2)
    base_direction_array = np.array([['clockwise'] * n_directions_per_vel +
                                     ['counter'] * n_directions_per_vel]
                                    * n_velocities)
    directions = base_direction_array.reshape(len(velocities))
    vms = np.zeros(velocities.shape)
    for i in range(len(velocities)):
        v = velocities[i]
        coeff = i % n_repeats + 1  # (avoid 0)
        vms[i] = v + (coeff * n_velocities)
    # vms = velocities.copy()
    frame = pd.DataFrame({'velocity': velocities,
                          'direction': directions,
                          'vm': vms
                          })
    return frame


# WARNING: UNUSED BUT PROBABLY VALID
# def pseudo_shuffle_df_direction(df):
#     """
#     Shuffles the data frame but keeps the time information (i.e. shuffles the velocity while keeping the
#     time information intact)
#     """
#     out_df = df.copy()  # TODO: see id needs deepcopy
#     #     reset_df(out_df)
#
#     n_samples = get_n_samples_per_column(df, 'velocity')
#     velocities = sorted(list(set(df['velocity'])))
#     n_velocities = len(velocities)
#
#     seeds = list(range(n_samples))
#     random.shuffle(seeds)  # where we shift the "lines"
#     shuffled_directions_list = ['clockwise' if seed < (n_samples / 2) else 'counter_clockwise' for seed in seeds]
#     shuffled_directions = np.tile(shuffled_directions_list, n_velocities)
#     #     print(shuffled_directions)
#     out_df['direction'] = shuffled_directions
#     #     plt.show()  # DEBUG (with fake traces)
#     return out_df
#
#
# def analyse_direction(frame, n_iter=10):
#     frame = filter_rotation(frame)
#     randomised_sds = []
#     for i in trange(n_iter, desc='shuffles'):
#         shuffled_frame = pseudo_shuffle_df_direction(frame)
#         shuffle_means = np.array(shuffled_frame.groupby(['direction'])['vm'].mean())
#         plt.scatter((0, 1), shuffle_means, color='blue')
#         randomised_sds.append(shuffle_means.std())
#     data_means = np.array(frame.groupby(['direction'])['vm'].mean())
#     plt.scatter((0, 1), data_means, color='red')
#     plt.show()
#
#     plt.hist(randomised_sds, 50, histtype='step', color='gray')
#     data_sd = data_means.std()
#     plt.plot((data_sd, data_sd), (0, 1), color='red')
#     plt.show()
#
#     get_percentiles(data_means, randomised_sds, 1)

# def plot_velocity_shuffles(frame, n_iter=20):
#     for i in trange(n_iter, desc='Shuffling'):
#         shuffled_frame = pseudo_shuffle_df(frame)
#         plt.plot(get_means(shuffled_frame), linewidth=0.25, color='gray')
#     plt.plot(get_means(frame), linewidth=2, color='blue')
#     plt.show()


# def plot_velocity(frame):
#     plt.plot(frame.groupby(['velocity'])['vm'].mean(), linewidth=3, color='gray')
#     plt.show()


# def plot_velocity_hist(frame):
#     plt.hist(frame.groupby(['velocity'])['vm'].mean(), 50, histtype='step')
#     plt.show()


# def plot_histograms(frame, show=True):
#     means = []
#     sds = []
#     for data in frame.groupby(['direction'])['vm']:
#         plt.hist(data[1], 50, linewidth=2, histtype='step', label=data[0])
#         means.append(data[1].mean())
#         sds.append(data[1].std())
#     #     print(data[1])
#     plt.legend()
#     if show:
#         plt.show()
#     return means, sds


# def plot_distributions_means_and_sds(means, sds):
#     plt.scatter([0]*len(means), means, color=('blue', 'green', 'red'))
#     plt.scatter([1]*len(sds), sds, color=('blue', 'green', 'red'))
#     plt.show()

# def get_signed_sds(frame, ref_var_name='velocity'):
#     """
#     Returns vms_sds corresponding to ref_var from -max to +max (back to back backwards)
#     Also returns r squared for clockwise and counterclockwise portion
#     """
#     clockwise_vm_sds = filter_direction(frame, 'clockwise').groupby(ref_var_name)['vm'].std().values
#     counter_clockwise_vm_sds = filter_direction(frame, 'counter_clockwise').groupby(ref_var_name)['vm'].std().values
#     clockwise_vm_sds = clockwise_vm_sds.flatten()
#     counter_clockwise_vm_sds = counter_clockwise_vm_sds.flatten()
#
#     # Stack back to back backwards to start negative to positive ref_variable
#     vms_sds = np.hstack((counter_clockwise_vm_sds[::-1], clockwise_vm_sds))
#     return vms_sds


















# WARNING: delete
# def plot_velocity_shuffles_histogram(frame, n_iter=10, plot=True):
#     """
#     Computes the shuffles (n_iter times) and plots the histograms of the average of the shuffle if plot==True
#
#     :param pd.DataFrame frame:
#     :param int n_iter: The number of times to shuffle
#     :param bool plot:
#     :return: data_means, shuffles_means, randomised_sds
#     """
#     randomised_sds = np.zeros(n_iter)
#     shuffles_means = []
#     for i in trange(n_iter, desc='Shuffling'):
#         shuffled_frame = pseudo_shuffle_df(frame)
#         data_means = get_means(shuffled_frame)
#         shuffles_means.append(data_means)
#         if plot:
#             plt.hist(data_means, 50, histtype='step', linewidth=0.25, color='gray')
#         randomised_sds[i] = data_means.std()
#
#     data_means = get_means(frame)
#     if plot:
#         plt.hist(data_means, 50, histtype='step', linewidth=2, color='red')
#         plt.show()
#     return data_means, shuffles_means, randomised_sds

# def pseudo_shuffle_df(df, debug=False, ref_var_name='velocity'):
#     """
#     Shuffles the data frame but keeps the time information (i.e. shuffles the velocity while keeping the
#     time information intact)
#
#     The samples are all arranged linearly in the vm column of the df with repeated series of velocity
#
#     """  # TODO: Confirm
#     out_df = duplicate_empty(df)
#
#     n_samples = get_n_samples_per_column(df, ref_var_name)
#     ref_var = make_uniques(df[ref_var_name])
#     n_ref_vars = len(ref_var)
#
#     seeds = random.sample(ref_var, n_samples)  # which random ref_var we start from
#
#     shuffled_vms = np.zeros(len(df))
#     shuffled_ref_vars = np.zeros(len(df))
#     for i in range(n_samples):
#         offset = (seeds[i] * n_samples) + i
#         vms = df['vm'].copy()
#         # rotate # REFACTOR: extract
#         selected_vms = np.hstack((vms[offset::n_samples], vms[i:offset:n_samples]))[:n_ref_vars]
#         if debug: plt.plot(selected_vms)  # (with fake traces)
#
#         start = i * n_ref_vars  # current idx in output vector (no append since np)
#         shuffled_vms[start:start + n_ref_vars] = selected_vms.copy()
#         shuffled_ref_vars[start:start + n_ref_vars] = ref_var.copy()
#
#     out_df['vm'] = shuffled_vms
#     out_df[ref_var_name] = shuffled_ref_vars
#
#     if debug: plt.show()  # (with fake traces)
#     return out_df


# def analyse_velocity(frame):
#     means, shuffles_means, randomised_sds = plot_velocity_shuffles_histogram(frame, 20)
#     plot_sds(means, randomised_sds)
#     plot_sds_histo(means, randomised_sds)
#     get_percentiles(means, randomised_sds)


