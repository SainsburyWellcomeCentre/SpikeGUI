# REMOVE ME LATER
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import numpy as np

import analysis_guis.calc_functions as cfcn

def ccgram_plot(ccG_xi, ccG_N, ciN_lo, ciN_hi, t_min, t_max, is_bar=True, ax=None):
    '''

    :param xi:
    :param ccG:
    :param ci_lo:
    :param ci_hi:
    :param t_min:
    :param t_max:
    :return:
    '''

    if ax is None:
        _, ax = plt.subplots()

    if is_bar:
        plt.bar(ccG_xi, height=ccG_N)
    else:
        plt.plot(ccG_xi, ccG_N,'bo')

        ii = np.logical_or(ccG_N < ciN_lo, ccG_N > ciN_hi)
        jj = np.logical_and(abs(ccG_xi) >= t_min, abs(ccG_xi) <= t_max)
        kk = np.logical_and(ii, jj)

        plt.plot(ccG_xi[kk], ccG_N[kk], 'ro')


    plt.plot(ccG_xi, ciN_lo, 'r--')
    plt.plot(ccG_xi, ciN_hi, 'r--')
    y_lim = plt.ylim()
    lim_patch = [Rectangle((-t_max, y_lim[0]), (t_max - t_min), y_lim[1]),
                 Rectangle((t_min, y_lim[0]), (t_max - t_min), y_lim[1])]
    pc = PatchCollection(lim_patch, facecolor='r', alpha=0.25, edgecolor='k')
    ax.add_collection(pc)
    plt.show()

def check_final_soln(i_ref, i_comp, t_spike, ccG, ccG_xi, filt, p_lim, t_min, t_max, c_id, i_ofs=0, is_bar=True):
    '''

    :param i_ref:
    :param i_comp:
    :param t_spike:
    :param ccG:
    :return:
    '''

    n_plt = min(len(np.array(range(i_ofs,len(i_ref)))), 16)
    plt.figure(figsize=(18, 9), dpi=100)
    ii = np.arange(75,125).astype(int)


    for i in range(n_plt):
        j = i + i_ofs
        ax = plt.subplot(4, 4, i+1)

        f_scale_ref = len(t_spike[i_ref[j]]) / 1000.0
        ccG_N = f_scale_ref * ccG[i_ref[j], i_comp[j], :]
        ciN_lo, ciN_hi = cfcn.calc_ccgram_prob(ccG_N, filt, p_lim)

        ccgram_plot(ccG_xi[ii], ccG_N[ii], ciN_lo[ii], ciN_hi[ii], t_min, t_max, ax=ax, is_bar=is_bar)
        ax.set_title('{0} vs {1}'.format(c_id[i_ref[j]], c_id[i_comp[j]]))

    plt.tight_layout()
    plt.show()