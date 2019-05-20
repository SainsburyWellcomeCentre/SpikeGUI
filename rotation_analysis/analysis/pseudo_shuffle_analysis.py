from matplotlib import pyplot as plt


from vest_phys.shuffle_anova_mateo import plot_shuffles_histogram, get_percentiles, sanitise_p_value


def plot_sd_shuffles(spiking_mat, ref_var_vector, n_shuffles=1000, show=True):
    means, shuffles_means, randomised_sds, randomised_rs_squared = plot_shuffles_histogram(ref_var_vector,
                                                                                           spiking_mat,
                                                                                           n_shuffles,
                                                                                           do_r_squared=False,
                                                                                           replace=True)  # WARNING: needs replace because too few columns

    left_tail, real_sd, right_tail, percentile, p_val = get_percentiles(means, randomised_sds, verbose=False)
    p_val = sanitise_p_value(n_shuffles, p_val)
    plt.hist(randomised_sds, 50, histtype='step', linewidth=2, color='gray')
    lower_xlim = min(left_tail-1, means.std() + 0.2)
    upper_xlim = max(right_tail+1, means.std() + 0.2)
    plt.xlim(lower_xlim, upper_xlim)  # TEST:
    plt.axvline(x=means.std(), color='red')
    if show:
        plt.show()
    return left_tail, real_sd, right_tail, percentile, p_val
