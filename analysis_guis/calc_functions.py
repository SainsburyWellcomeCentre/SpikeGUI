# module import
import os
import neo
import time
import pywt
import copy
import random
import peakutils
import math as m
import numpy as np
import quantities as pq
from fastdtw import fastdtw
from numpy.matlib import repmat
import shapely.geometry as geom

# scipy module imports
from scipy import stats, signal
from scipy.stats import poisson as p
from scipy.stats import pearsonr as pr
from scipy.spatial.distance import *
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# sklearn module imports
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import mutual_info_score

# elephant module imports
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef

# custom module imports
import analysis_guis.common_func as cf
from analysis_guis.dialogs.rotation_filter import RotationFilteredData

try:
    import analysis_guis.test_plots as tp
except:
    pass

# other function declarations
dcopy = copy.deepcopy
diff_dist = lambda x, y: np.sum(np.sum((x - y) ** 2, axis=0)) ** 0.5

########################################################################################################################
########################################################################################################################


def opt_time_to_y0(args, bounds):
    '''

    :param x:
    :param pp:
    :param y0:
    :return:
    '''

    def opt_func(x, pp, y0):
        '''

        :param x:
        :param pp:
        :param y0:
        :return:
        '''

        return np.power(pp(x[0]) - y0, 2.0)

    # parameters
    iter, iter_max, f_tol = 0, 50, 1e-6
    if np.diff(bounds)[0][0] < f_tol:
        return bounds[0][0]

    # keep looping until a satifactory result has been achieves
    while iter < iter_max:
        # sets the random bound
        x0 = bounds[0][0] + random.random() * np.diff(bounds[0])
        m_opt = minimize(opt_func, x0, args=args, bounds=bounds, tol=f_tol)

        #
        if m_opt.fun < f_tol:
            x_opt = m_opt.x[0]
            break
        else:
            iter += 1
            if iter == iter_max:
                x_opt = bounds[0][0]
                break

    if x_opt is None:
        return bounds[0][0]
    else:
        return x_opt

###########################################
####    CROSS-CORRELOGRAM FUNCTIONS    ####
###########################################


def calc_ccgram(ts1, ts2, win_sz0=50, bin_size=0.5, return_freq=True):
    '''

    :param ts1:
    :param ts2:
    :param win_sz:
    :return:
    '''

    # initialisations and memory allocation
    i_start, i_spike, win_sz = 0, 1, [-win_sz0, win_sz0]
    ccInfo = np.nan * np.zeros((len(ts1), 3), dtype=float)

    # keep looping until all spikes have been searched
    while i_spike < len(ts1):
        # Seek to the beginning of the current spike's window
        i = i_start
        while ((i + 1) < len(ts2)) and (ts2[i] <= (ts1[i_spike] + win_sz[0])):
            i += 1

        # sets the start of the window (for later)
        i_start = i
        if (ts2[i] > (ts1[i_spike] + win_sz[1])):
            i_spike += 1
            continue

        # Find all the spike indices that fall within the window
        while ((i + 1) < len(ts2)) and (ts2[i] <= (ts1[i_spike] + win_sz[1])):
            i += 1

        # sets the cross-correlogram information
        i_end = i - 1
        ccInfo[i_spike, 0] = i_start
        ccInfo[i_spike, 1] = i_end
        ccInfo[i_spike, 2] = ts1[i_spike]

        # increments the spike index
        i_spike += 1

    # memory allocation and initialisations
    t_ofs, i_ts1, i_ts2, incr = [], [], [], 0
    d_index = np.diff(ccInfo[:, :2], axis=1).ravel()

    while True:
        # determines which spikes belong to the current window
        has_ind = d_index >= incr
        if np.any(has_ind):
            tmp = np.where(has_ind)[0]
        else:
            break

        # sets the overall indices and the centre-times of the spikes
        idx = ccInfo[tmp, 0].astype(int) + incr
        c_times = ccInfo[tmp, 2]

        #
        t_ofs_new = ts2[idx] - c_times
        is_ok = np.abs(t_ofs_new) > 1e-6

        # appends the time offsets and increments the counter
        t_ofs.append(t_ofs_new[is_ok])
        incr += 1

        # possibly include later?
        # i_ts1.append(tmp[is_ok])
        # i_ts2.append(idx[is_ok])

    # returns the offsets
    t_ofs = cf.flat_list(t_ofs)
    xi_bin = np.arange(win_sz[0]+bin_size/2, win_sz[1] + bin_size/2, bin_size)
    hh = np.histogram(t_ofs, xi_bin)

    if return_freq:
        return hh[0] * (1000.0 / (bin_size * len(ts1))), (hh[1][:-1] + hh[1][1:]) / 2
    else:
        return hh[0], (hh[1][:-1] + hh[1][1:]) / 2

####################################################
####    CLUSTER MATCHING METRIC CALCULATIONS    ####
####################################################


def calc_isi_corr(data_fix, data_free, is_feas):
    '''

    :param data_fix:
    :param data_free:
    :param is_feas:
    :return:
    '''

    # memory allocation
    isi_corr = np.zeros((data_fix['nC'], data_free['nC']))

    # calculates the correlation coefficient for each feasible cluster
    for i_fix in range(data_fix['nC']):
        for i_free in range(data_free['nC']):
            if is_feas[i_fix, i_free]:
                C = np.corrcoef(data_fix['isiHist'][i_fix],data_free['isiHist'][i_free])
                isi_corr[i_fix, i_free] = C[0, 1]
            else:
                isi_corr[i_fix, i_free] = np.nan

    # returns the final array
    return isi_corr, norm_array_rows(isi_corr)


def calc_isi_hist_metrics(data_fix, data_free, is_feas, calc_fcn, is_norm=True, max_norm=True):
    '''

    :param data_fix:
    :param data_free:
    :param is_feas:
    :param calc_fcn:
    :return:
    '''

    # memory allocation
    hist_metric = np.zeros((data_fix['nC'], data_free['nC']))

    # calculates the correlation coefficient for each feasible cluster
    for i_fix in range(data_fix['nC']):
        for i_free in range(data_free['nC']):
            if is_feas[i_fix, i_free]:
                h_fix = data_fix['isiHist'][i_fix]
                h_free = data_free['isiHist'][i_free]

                if is_norm:
                    h_fix = norm_signal_sum(h_fix)
                    h_free = norm_signal_sum(h_free)

                hist_metric[i_fix, i_free] = calc_fcn(h_fix, h_free)
            else:
                hist_metric[i_fix, i_free] = np.nan

    # returns the final array
    return hist_metric, norm_array_rows(hist_metric, max_norm)


def calc_signal_hist_metrics(data_fix, data_free, is_feas, calc_fcn, is_norm=True, max_norm=True):
    '''

    :param data_fix:
    :param data_free:
    :param is_feas:
    :param calc_fcn:
    :param is_norm:
    :return:
    '''

    # memory allocation
    hist_metric = np.zeros((data_fix['nC'], data_free['nC']))

    # calculates the metrics for each time point between each
    for i_fix in range(data_fix['nC']):
        # loops through each of the free clusters calculating the metrics for the current time point
        for i_free in range(data_free['nC']):
            if is_feas[i_fix, i_free]:
                # calculates
                single_metric = calc_single_hist_metric(data_fix, data_free, i_fix, i_free, is_norm, calc_fcn)

                # calculates the mean of all the time points
                hist_metric[i_fix, i_free] = np.mean(single_metric)
            else:
                # if not feasible, then set a NaN value
                hist_metric[i_fix, i_free] = np.nan

    # returns the final metric array
    return hist_metric


def calc_single_hist_metric(data_fix, data_free, i_fix, i_free, is_norm, calc_fcn):
    '''

    :param data_fix:
    :param data_free:
    :param i_fix:
    :param i_free:
    :param is_norm:
    :param calc_fcn:
    :return:
    '''

    # retrieves/normalises the fixed histogram for the current point
    metric_temp = np.zeros(data_fix['nPts'])

    for i_pts in range(data_fix['nPts']):
        # retrieves fixed/free histograms for the current point
        h_fix = data_fix['ptsHist'][i_fix][i_pts, :]
        h_free = data_free['ptsHist'][i_free][i_pts, :]

        # normalises the histograms (if required)
        if is_norm:
            h_fix = norm_signal_sum(h_fix)
            h_free = norm_signal_sum(h_free)

        # calculates the metric
        metric_temp[i_pts] = calc_fcn(h_fix, h_free)

    return metric_temp

################################################
####    SIGNAL BASED METRICS CALCULATIONS   ####
################################################


def calc_signal_feature_diff(data_fix, data_free, is_feas):
    '''

    :param data_fix:
    :param data_free:
    :param is_feas:
    :return:
    '''

    # memory allocations
    sig_feat = np.zeros((data_fix['nC'], data_free['nC'], 4))

    #
    for i_fix in range(data_fix['nC']):
        # sets the fixed landmark points
        i_max1_fix = data_fix['sigFeat'][i_fix, 0]
        i_min_fix = data_fix['sigFeat'][i_fix, 1]
        i_max2_fix = data_fix['sigFeat'][i_fix, 2]
        t_hw_fix = data_fix['sigFeat'][i_fix, 4] - data_fix['sigFeat'][i_fix, 3]

        a_fix = data_fix['vMu'][int(i_max1_fix), i_fix]
        b_fix = data_fix['vMu'][int(i_max2_fix), i_fix]
        c_fix = i_max2_fix - i_min_fix
        d_fix = data_fix['vMu'][int(i_min_fix), i_fix]
        x_fix, y_fix = a_fix - d_fix, b_fix - d_fix


        for i_free in range(data_free['nC']):
            # sets the free landmark points
            if is_feas[i_fix, i_free]:
                i_max1_free = data_free['sigFeat'][i_free, 0]
                i_min_free = data_free['sigFeat'][i_free, 1]
                i_max2_free = data_free['sigFeat'][i_free, 2]
                t_hw_free = data_free['sigFeat'][i_free, 4] - data_free['sigFeat'][i_free, 3]

                a_free = data_free['vMu'][int(i_max1_free), i_free]
                b_free = data_free['vMu'][int(i_max2_free), i_free]
                c_free = i_max2_free - i_min_free
                d_free = data_free['vMu'][int(i_min_free), i_free]
                x_free, y_free = a_free - d_free, b_free - d_free

                # calculates the feature differences
                sig_feat[i_fix, i_free, 0] = prop_diff(b_fix, b_free)
                sig_feat[i_fix, i_free, 1] = prop_diff(c_fix, c_free)
                sig_feat[i_fix, i_free, 2] = prop_diff(x_fix, x_free)
                sig_feat[i_fix, i_free, 3] = prop_diff(y_fix, y_free)
            else:
                for i_feat in range(np.size(sig_feat, axis=2)):
                    sig_feat[i_fix, i_free, i_feat] = np.nan

    # returns the feature difference array
    return sig_feat


def calc_array_euclidean(arr):
    '''

    :param sig_feat:
    :return:
    '''

    #
    arr_euc = np.array([
            np.sum([
                arr[j, :, i] ** 2 for i in range(np.size(arr, axis=2))
            ],axis=0) ** 0.5 for j in range(np.size(arr, axis=0))
        ]) / np.size(arr, axis=2)**0.5

    #
    return arr_euc


def calc_signal_corr(i_dtw, data_fix, data_free, is_feas):
    '''

    :param data_fix:
    :param data_free:
    :param is_feas:
    :return:
    '''

    # memory allocation
    cc = np.zeros((data_fix['nC'], data_free['nC']))
    dd_dtw = np.zeros((data_fix['nC'], data_free['nC']))
    dtw_scale = np.zeros((data_fix['nC'], data_free['nC']))

    # calculates the min/max mean signal values
    y_min = [np.min(data_fix['vMu'][:, i]) for i in range(data_fix['nC'])]
    y_max = [np.max(data_fix['vMu'][:, i]) for i in range(data_fix['nC'])]

    #
    for i_fix in range(data_fix['nC']):
        for i_free in range(data_free['nC']):
            if is_feas[i_fix, i_free]:
                # sets up the dynamic time-warped signals
                i_dtw_nw = i_dtw[i_fix, i_free]
                y_fix_norm = norm_signal(data_fix['vMu'][:, i_fix], y_min=y_min[i_fix], y_max=y_max[i_fix])
                y_free_norm = norm_signal(data_free['vMu'][:, i_free], y_min=y_min[i_fix], y_max=y_max[i_fix])
                y_fix_dtw, y_free_dtw = y_fix_norm[i_dtw_nw[:, 0]], y_free_norm[i_dtw_nw[:, 1]]

                # calculates the correlation coefficient
                C = np.corrcoef(y_fix_dtw, y_free_dtw)

                # calculates the total distance between the mean signals
                d_tot_dtw = calc_total_distance(y_fix_dtw, y_free_dtw, data_fix['nPts'])

                # dtw_scale[i_fix, i_free] = (data_fix['nPts'] / len(i_dtw_nw))
                cc[i_fix, i_free] = C[0, 1]
                dd_dtw[i_fix, i_free] = np.max(d_tot_dtw)
                dtw_scale[i_fix, i_free] = data_fix['nPts'] / len(y_fix_dtw)
            else:
                cc[i_fix, i_free] = np.nan
                dd_dtw[i_fix, i_free] = np.nan
                dtw_scale[i_fix, i_free] = np.nan

    # returns the final arrays
    return cc, dd_dtw, dtw_scale


def calc_dtw_indices(comp, data_fix, data_free, is_feas):
    '''

    :param data_fix:
    :param data_free:
    :return:
    '''

    #
    for i_fix in range(data_fix['nC']):
        for i_free in range(data_free['nC']):
            if is_feas[i_fix, i_free]:
                _, p = fastdtw(data_fix['vMu'][:, i_fix], data_free['vMu'][:, i_free])
                comp.i_dtw[i_fix, i_free] = np.array(p)

    return comp

############################################
####    HISTOGRAM SIMILARITY METRICS    ####
############################################

def calc_total_distance(y_fix, y_free, n_pts):
    '''

    :return:
    '''

    # sets the point arrays
    xi = np.array(range(len(y_fix))) / n_pts
    pp = [np.vstack((xi, y_fix)).T, np.vstack((xi, y_free)).T]
    lines = [geom.LineString(pp[1]), geom.LineString(pp[0])]

    #
    d_tot = np.zeros((len(xi), 2))
    for j in range(len(pp)):
        for i, p in enumerate(pp[j]):
            pp_g = geom.Point(p[0], p[1])
            d_tot[i, j] = pp_g.distance(lines[j])

    return d_tot

def calc_kldiverge(hist_1, hist_2):
    '''

    :param hist_1:
    :param hist_2:
    :return:
    '''

    return stats.entropy(pk=hist_1+1, qk=hist_2+1)


def calc_bhattacharyya(hist_1, hist_2):
    '''

    :param hist_1:
    :param hist_2:
    :return: Bhattacharyya distance between histogram distributions
    '''

    return -np.log(sum((np.sqrt(u * w) for u, w in zip(hist_1, hist_2))))


def calc_hist_intersect(hist_1, hist_2):
    '''

    :param hist_2:
    :param hist_2:
    :return:
    '''

    return np.true_divide(np.sum(np.minimum(hist_1, hist_2)), np.max([np.sum(hist_1),np.sum(hist_2)]))


def calc_kw_stat(hist_1, hist_2):
    '''

    :param hist_1:
    :param hist_2:
    :return:
    '''

    ks_stat, _ = stats.kruskal(hist_1, hist_2)
    return ks_stat


def calc_wasserstein(hist_1, hist_2):
    '''

    :param hist_1:
    :param hist_2:
    :return:
    '''

    return stats.wasserstein_distance(hist_1, hist_2)

###################################################
####    NORMALISATION CALCULATION FUNCTIONS    ####
###################################################

def norm_signal(y_signal, y_max=None, y_min=None):
    '''

    :param y_signal:
    :param y_max:
    :param y_min:
    :return:
    '''

    # calculates the maximum signal value
    if y_max is None:
        y_max = np.max(y_signal)

    # calculates the minimum signal value
    if y_min is None:
        y_min = np.min(y_signal)

    # returns the normalised signal
    return (y_signal - y_min) / (y_max - y_min)


def norm_signal_sum(y_signal):
    '''

    :param y_signal:
    :return:
    '''

    return y_signal / np.sum(y_signal)


def norm_array_rows(metric, max_norm=True):
    '''

    :param max_norm:
    :return:
    '''

    # memory allocation
    metric_norm = np.zeros((np.size(metric, axis=0), np.size(metric, axis=1)))

    #
    for i_row in range(np.size(metric, axis=0)):
        if np.all([np.isnan(x) for x in metric[i_row, :]]):
            metric_norm[i_row, :] = np.nan
        else:
            if max_norm:
                metric_norm[i_row, :] = np.abs(metric[i_row, :]) / np.nanmax(np.abs(metric[i_row, :]))
            else:
                metric_norm[i_row, :] = np.nanmin(np.abs(metric[i_row, :])) / np.abs(metric[i_row, :])

    # returns the array
    return metric_norm

###################################################
####    MISCELLANEOUS CALCULATION FUNCTIONS    ####
###################################################

def cluster_distance(data_fix, data_free, n_shuffle=100, n_spikes=10, i_cluster=[1]):
    '''

    :return:
    '''

    # memory allocation and other initialisations
    n_cluster = len(i_cluster)
    rperm = np.random.permutation
    mu_dist = np.zeros((n_cluster, data_free['nC'] + 1, n_shuffle))

    # calculates the distances between the free/fixed cluster shuffled means (over all clusters/shuffles)
    for i_fix in range(n_cluster):
        # REMOVE ME LATER for waitbar
        print('Calculating Distances for Fixed Cluster #{0}'.format(i_fix + 1))

        # progress update update
        for i_free in range(data_free['nC'] + 1):
            # sets the number of free/fixed spikes for the current cluster
            i_cl = i_cluster[i_fix] - 1
            n_fix = np.size(data_fix['vSpike'][i_cl], axis=1)
            if i_free < data_free['nC']:
                n_free = np.size(data_free['vSpike'][i_free], axis=1)

            #
            for i_shuff in range(n_shuffle):
                # calculates the shuffled fixed mean signal
                ind_fix = rperm(n_fix)[:n_spikes]
                ws_fix = np.mean(data_fix['vSpike'][i_cl][:, ind_fix], axis=1)

                # sets the comparison signal based on the type
                if i_free == data_free['nC']:
                    # case is the fixed signal population mean
                    ws_free = data_fix['vMu'][:, i_cl]
                else:
                    # case is the shuffled free mean signal
                    ind_free = rperm(n_free)[:n_spikes]
                    ws_free = np.mean(data_free['vSpike'][i_free][:, ind_free], axis=1)

                # calculates the euclidean distance between the signals
                mu_dist[i_fix, i_free, i_shuff] = euclidean(ws_free, ws_fix)

    #
    return mu_dist

    # # closes the waitbar dialog
    # h.close()

def calc_ccgram_types(ccG, ccG_xi, t_spike, c_id=None, calc_para=None, w_prog=None, expt_id=None):
    '''

    :param ccG:
    :param ccG_xi:
    :return:
    '''

    def add_list_signals(ccG_T, ci_loT, ci_hiT, ccG_N, ciN_lo, ciN_hi, ib, ind):
        '''

        :return:
        '''

        # appends on the lower/upper confidence interval limits
        if ib == 0:
            ccG_T[ind].append(ccG_N[::-1])
            ci_loT[ind].append(ciN_lo[::-1])
            ci_hiT[ind].append(ciN_hi[::-1])
        else:
            ccG_T[ind].append(ccG_N)
            ci_loT[ind].append(ciN_lo)
            ci_hiT[ind].append(ciN_hi)

    def det_event_time(ccG_xi, i_grp, ib):
        '''

        '''

        if ib == 0:
            # case is for the t < 0 search band, so take the last point from the last group
            return -ccG_xi[i_grp[-1][-1]]
        else:
            # case is for the t > 0 search band, so take the first point from the first group
            return ccG_xi[i_grp[0][0]]

    def calc_event_duration(i_grp, dt, ib):
        '''
        '''

        #
        if ib == 0:
            return (len(i_grp[-1]) - 1) * dt
        else:
            return (len(i_grp[0]) - 1) * dt

    def is_excite_grp_feas(z_ccG, ccG_N, ind_grp, i_side):
        '''

        :param ccG:
        :param ci_mn:
        :param i_grp:
        :return:
        '''

        # parameters
        n_wid, pmx_tol, imx_rng, cc_peak_tol, t_mn_max = 7, 0.90, 1, 0.5, 5

        # sets the indices for the synchronisation zone (will be removed from the calculations)
        ind_side = np.zeros(len(z_ccG), dtype=bool)
        ind_side[i_side] = True

        # if the maximum value
        i_peak_zccg = peakutils.indexes(z_ccG * ind_side, thres=0.01)
        if not np.any([x in i_peak_zccg for x in ind_grp]):
            return False
        else:
            i_mx = np.argmax(z_ccG[ind_grp])
            not_ok_1 = np.any(np.diff(ccG_N[ind_grp[:i_mx]]) < 0)
            not_ok_2 = np.any(np.diff(ccG_N[ind_grp[i_mx:]]) > 0)

            if not_ok_1 or not_ok_2:
                return False
            elif ((ccG_N[ind_grp[0] - 1] / ccG_N[ind_grp[i_mx]]) > pmx_tol) or \
                 ((ccG_N[ind_grp[-1] + 1] / ccG_N[ind_grp[i_mx]]) > pmx_tol):
                return False

        # calculates the cwt coefficients for the range. from this calculate the overall signal
        cc = pywt.cwt(z_ccG * (z_ccG > 0), np.arange(1, n_wid), 'mexh')[0]

        # if the
        cc_avg = np.mean(cc * (cc > 0), axis=0)
        cc_grp_mx = cc[:, i_mx]

        #
        imx_ind_rng = np.arange(ind_grp[0] - imx_rng, ind_grp[-1] + (imx_rng + 1))
        if np.argmax(cc_avg) not in imx_ind_rng:
            return False
        else:
            cc_avg /= np.max(cc_avg)
            ind_peaks = np.arange(50,150)

            i_peak = peakutils.indexes(cc_avg[ind_peaks], thres=cc_peak_tol, min_dist=n_wid)
            if len(i_peak) == 1:
                return True
            else:
                return False

        # cc_avg = np.mean(np.abs(cc), axis=0)
        # cc_avg /= np.max(cc_avg)

        # # if the overall maximum is not in the max index range, then return a false value
        # if np.argmax(cc_avg) not in imx_ind_rng:
        #     return False
        # else:
        #     # otherwise, determine if any of the other peaks are significant (return false value if more than one)
        #     i_peak = peakutils.indexes(cc_avg, thres=cc_peak_tol, min_dist=n_cc)
        #     return len(i_peak) == 1

    # fixed parameters
    p_lower = 0.25
    ccG_median_min = 10
    min_count = 3
    t_min = 1.5
    t_max = 4.0
    n_hist, nC = int(np.size(ccG, axis=2) / 2), np.size(ccG, axis=1)
    dt = ccG_xi[1] - ccG_xi[0]

    # calculation parameters
    if calc_para is None:
        # calculations parameters have not been provided
        n_min = [3, 2]
        p_lim = 99.9999/100.0
        f_cutoff = 5
    else:
        # case is the calculation parameters have been provided
        n_min = [calc_para['n_min_lo'], calc_para['n_min_hi']]
        p_lim = calc_para['p_lim'] / 100.0
        f_cutoff = calc_para['f_cutoff']

    # memory allocation
    d_copy, A, B, C = copy.deepcopy, [[] for _ in range(5)], [[] for _ in range(4)], np.zeros((nC, nC))
    i_grp, c_type_arr, t_event_arr = np.empty(2, dtype=object), d_copy(C).astype(int), d_copy(C)
    ci_loT, ci_hiT, ccG_T = d_copy(A), d_copy(A), d_copy(A)
    c_type, t_event, t_dur = d_copy(A), d_copy(B), d_copy(B)

    # sets up the pre/post spike indices
    i_side = [np.where(ccG_xi < -dt)[0], np.where(ccG_xi > dt)[0]]
    i_band_outer = np.where(np.abs(ccG_xi) > 2*t_max)[0]
    i_band_tot = np.where(np.logical_and(ccG_xi >= -t_max, ccG_xi <= t_max))[0]

    # sets the binary array of the search band
    i_band = [np.where(np.logical_and(ccG_xi <= -t_min, ccG_xi >= -t_max))[0],
              np.where(np.logical_and(ccG_xi >= t_min, ccG_xi <= t_max))[0]]

    # sets up the gaussian signal filter
    freq_range = np.arange(0,len(ccG_xi)) * (1 / dt)
    freq = next((i for i in range(len(ccG_xi)) if freq_range[i] > f_cutoff), len(ccG_xi))

    # calculates the
    for i_ref in range(nC):
        # if the progress bar object is provided, then update the progress
        if w_prog is not None:
            if expt_id is None:
                p_str = 'Analysing Cluster #{0}/{1}'.format(i_ref + 1, nC)
            else:
                p_str = 'Analysing Expt #{0}/{1}, Cluster #{2}/{3}'.format(expt_id[0], expt_id[1], i_ref + 1, nC)

            # print(p_str)
            w_prog.emit(p_str, 100.0 * i_ref / nC)

        # calculates the cc-gram scale factor
        f_scale_ref = len(t_spike[i_ref]) / 1000.0
        for i_comp in range(i_ref, nC):
            # only calculate the comparison if reference != comparison
            if i_ref != i_comp:
                # converts the cc-gram frequencies to counts
                ccG_N = f_scale_ref * ccG[i_ref, i_comp, :]
                ciN_lo, ciN_hi, z_ccG = calc_ccgram_prob(ccG_N, freq, p_lim)
                ind_grp = [[i_comp, i_ref], [i_ref, i_comp]]

                # determines if A) there are several points within the search time band that are much less than the
                # median spike count, and B) if the median spike count is above a certain level
                ccG_median = np.median(ccG_N[i_band_outer])
                if (np.sum(ccG_N[i_band_tot] < p_lower * ccG_median) > min_count) and (ccG_median > ccG_median_min):
                    # if so, then the cc-gram is probably an anomaly
                    c_type[4].append([ind_grp[0][0], ind_grp[0][1]])

                    # appends on the lower/upper confidence interval limits
                    ccG_T[2].append(ccG_N)
                    ci_loT[2].append(ciN_lo)
                    ci_hiT[2].append(ciN_hi)
                else:
                    # otherwise, calculate the cc-gram lower/upper confidence levels and calculates the relative
                    # values of the cc-gram values to these limits (within the search time band)
                    for ib in range(len(i_band)):
                        d_sig_lo = ciN_lo[i_band[ib]] - ccG_N[i_band[ib]]
                        d_sig_hi = ccG_N[i_band[ib]] - ciN_hi[i_band[ib]]

                        # determines the time points that are above/below the upper/lower confidence intervals
                        is_sig_lo, is_sig_hi = d_sig_lo > 0, d_sig_hi > 0
                        if (not np.any(is_sig_lo)) and (not np.any(is_sig_hi)):
                            continue

                        # removes any small groupings
                        i_grp[0], i_grp[1] = cf.get_index_groups(is_sig_lo), cf.get_index_groups(is_sig_hi)
                        for i in range(len(i_grp)):
                            if len(i_grp[i]):
                                i_grp[i] = [x for x in i_grp[i] if len(x) >= n_min[i]]

                        # if there are no valid groups, then continue
                        if all([len(x) == 0 for x in i_grp]):
                            continue

                        # determines the bands to which the significant groups belong to
                        i1, i2 = ind_grp[ib][0], ind_grp[ib][1]
                        for i in range(len(i_grp)):
                            if len(i_grp[i]):
                                # if potential groups
                                if i == 0:
                                    # case is an inhibitory group
                                    is_ok = True
                                else:
                                    # case is an excitatory group
                                    if ib == 0:
                                        is_ok = is_excite_grp_feas(z_ccG, ccG_N, i_band[ib][i_grp[i][-1]], i_side[ib])
                                    else:
                                        is_ok = is_excite_grp_feas(z_ccG, ccG_N, i_band[ib][i_grp[i][0]], i_side[ib])

                                # determines if the type has been set
                                if not is_ok:
                                    #
                                    c_type[2+i].append([i1, i2])
                                    t_dur[2+i].append(calc_event_duration(i_grp[i], dt, ib))
                                    t_event[2+i].append(t_event_arr[i1, i2])

                                    # appends on the lower/upper confidence interval limits
                                    add_list_signals(ccG_T, ci_loT, ci_hiT, ccG_N, ciN_lo, ciN_hi, ib, 2+i)

                                elif c_type_arr[i1, i2] == 0:
                                    # case is the value has not yet been set
                                    c_type_arr[i1, i2] = i + 1
                                    t_event_arr[i1, i2] = det_event_time(ccG_xi[i_band[ib]], i_grp[i], ib)

                                    c_type[i].append([i1, i2])
                                    t_dur[i].append(calc_event_duration(i_grp[i], dt, ib))
                                    t_event[i].append(t_event_arr[i1, i2])

                                    # appends on the lower/upper confidence interval limits
                                    add_list_signals(ccG_T, ci_loT, ci_hiT, ccG_N, ciN_lo, ciN_hi, ib, i)

                                elif c_type_arr[i1, i2] != 0:
                                    # if the current event precedes the previously stored event, then swap the classifcation
                                    # to the current type
                                    t_event_new = det_event_time(ccG_xi[i_band[ib]], i_grp[i], ib)
                                    if t_event_new < t_event_arr[i1, i2]:
                                        #
                                        j = c_type_arr[i1, i2] - 1
                                        N = len(c_type[j]) - 1

                                        # removes the existing signals from the stored list
                                        c_type[j].pop(N)
                                        t_dur[j].pop(N)
                                        t_event[j].pop(N)
                                        ccG_T[j].pop(N)
                                        ci_loT[j].pop(N)
                                        ci_hiT[j].pop(N)

                                        # resets the other values to the current type
                                        c_type[i].append([i1, i2])
                                        t_dur[i].append(calc_event_duration(i_grp[i], dt, ib))
                                        t_event[i].append(t_event_new)

                                        # appends on the lower/upper confidence interval limits
                                        add_list_signals(ccG_T, ci_loT, ci_hiT, ccG_N, ciN_lo, ciN_hi, ib, i)

                                    # appends on the lower/upper confidence interval limits
                                    c_type[4].append([i1, i2])
                                    add_list_signals(ccG_T, ci_loT, ci_hiT, ccG_N, ciN_lo, ciN_hi, ib, 4)

    # returns the data arrays
    return c_type, t_dur, t_event, ci_hiT, ci_loT, ccG_T


def calc_ccgram_prob(ccG, freq, p_lim, ind=None):
    '''

    :param ccG:
    :param freq:
    :return:
    '''

    # case is fourier smoothing
    rft = np.fft.rfft(np.hstack((ccG, ccG[-1])))
    rft[freq:] = 0
    ccG_lo = np.fft.irfft(rft)[:-1]

    #
    n_win, n_bin = 50, len(ccG)
    ccG_hi, ii = ccG - ccG_lo, np.array(list(range(n_win)) + list(range(n_bin-n_win,n_bin)))
    ccG_mn, ccG_sd = np.mean(ccG_hi[ii]), np.std(ccG_hi[ii])

    # returns the lower/upper confidence levels
    return p.ppf(1.0 - p_lim, ccG_lo), p.ppf(p_lim, ccG_lo), (ccG_hi - ccG_mn) / ccG_sd


def prop_diff(x, y):
    '''

    :param x:
    :param y:
    :return:
    '''

    return max(0.0, 1.0 - np.abs((x - y) / x))


def ind2sub(n_cols, ind):
    '''

    :param n_cols:
    :param ind:
    :return:
    '''

    return (np.mod(ind.astype('int'), n_cols), (ind / n_cols).astype('int'))


def calc_weighted_mean(metrics, W=None):
    '''

    :return:
    '''

    # memory allocation
    metric_weighted_mean = np.zeros((np.size(metrics, axis=0), np.size(metrics, axis=1), np.size(metrics, axis=2)))

    # sets the relative weighting regime
    if W is None:
        # no weights provided, so use equal weights
        W = np.ones(np.size(metrics, axis=2)) / np.size(metrics, axis=2)
    else:
        # otherwise, ensure the weights sum up to 1
        W = W / np.sum(W)

    # sets the weighted values into the overall array (only for the non-NaN values)
    for i_row in range(np.size(metrics, axis=0)):
        ii = np.array([np.isnan(x) for x in metrics[i_row, :, 0]])
        for i_met in range(np.size(metrics, axis=2)):
            metric_weighted_mean[i_row, :, i_met] = W[i_met] * metrics[i_row, :, i_met]

    # returns the weighted sum
    return np.sum(metric_weighted_mean, axis=2)


def det_gmm_cluster_groups(grp_means):
    '''

    :param g_means:
    :return:
    '''

    # lambda function declaration
    mu_tol = 1e-3

    # initialisations
    i_grp, i_ind, n_grp = [grp_means[0]], [[0]], np.size(grp_means, axis=0)

    #
    for i in range(1, n_grp):
        # calculates the distance between the currently stored groups
        mu_diff = np.vstack([[diff_dist(x, grp_means[i]), diff_dist(x, grp_means[i][:, ::-1])] for x in i_grp])
        if np.any(mu_diff < mu_tol):
            # if there is a match, then determine which group/direction the new mean
            i_match = np.where(mu_diff < mu_tol)
            i_ind[i_match[0][0]].append(i)
        else:
            # otherwise, create a new cluster grouping
            i_grp.append(grp_means[i])
            i_ind.append([i])

    # returns the index
    return i_grp, i_ind


def arr_range(y, dim=None):
    '''

    :param y:
    :return:
    '''

    if dim is None:
        return np.max(y) - np.min(y)
    else:
        return np.max(y, axis=dim) - np.min(y, axis=dim)


def calc_avg_roc_curve(roc_xy):
    '''

    :param roc_xy:
    :return:
    '''

    # initialisations
    xi = np.linspace(0, 1, 101)

    # calculates the new interpolated x/y locations
    x_nw = np.stack([interp1d(np.linspace(0, 1, len(x[:, 0])), x[:, 0], kind='nearest')(xi) for x in roc_xy]).T
    y_nw = np.stack([interp1d(np.linspace(0, 1, len(x[:, 1])), x[:, 1], kind='nearest')(xi) for x in roc_xy]).T

    # returns the average trace
    return np.vstack((np.mean(x_nw, axis=1), np.mean(y_nw, axis=1))).T


def get_inclusion_filt_indices(c, exc_gen_filt):
    '''

    :param c:
    :param exc_gen_filt:
    :return:
    '''

    # applies the general exclusion filter (for the fields that have been set)
    cl_inc = dcopy(c['expInfo']['clInclude'])
    for exc_gen in exc_gen_filt:
        ex_g = exc_gen_filt[exc_gen]
        for ex_gt in ex_g:
            if exc_gen == 'region_name':
                cl_inc[c['chRegion'] == ex_gt] = False
            elif exc_gen == 'record_layer':
                cl_inc[c['chLayer'] == ex_gt] = False

    # returns the cluster inclusion index array
    return cl_inc


def calc_roc_curves_pool(p_data):
    '''

    :param p_data:
    :return:
    '''

    x, y = p_data[0], p_data[1]
    return cf.calc_roc_curves(None, None, x_grp=x, y_grp=y)


def get_rot_phase_offsets(calc_para, is_vis=False):
    '''

    :param calc_para:
    :return:
    '''

    # retrieves the parameters based on the experiment type
    if is_vis:
        # case is the visual experiment analysis
        t_ofs_str, t_phase_str, use_full_str = 't_ofs_vis', 't_phase_vis', 'use_full_vis'
    else:
        # case is the rotation experiment analysis
        t_ofs_str, t_phase_str, use_full_str = 't_ofs_rot', 't_phase_rot', 'use_full_rot'

    if t_ofs_str in calc_para:
        # if the offset parameters are present, then determine if the offsets are to be used
        if use_full_str in calc_para:
            # if the entire phase parameter is present, then return the offset values based on this parameters value
            if calc_para[use_full_str]:
                # if using the entire phase, then return None values
                return None, None
            else:
                # otherwise, use the partial phase parameters
                return float(calc_para[t_ofs_str]), float(calc_para[t_phase_str])
        else:
            # otherwise, use the partial phase parameters
            return float(calc_para[t_ofs_str]), float(calc_para[t_phase_str])
    else:
        # case is the parameters are not parameters
        return None, None


def calc_noise_correl(d_data, n_sp):
    '''

    :param d_data:
    :return:
    '''

    def calc_pw_noise_correl(n_spt):
        '''

        :param n_spt:
        :return:
        '''

        # array dimensioning
        n_c = np.size(n_spt, axis=1)
        if len(np.shape(n_spt)) == 2:
            # if a 2D array, then convert to a 3D array by including a redundant 3rd axis
            n_t = np.size(n_spt, axis=0)
            n_spt = np.reshape(n_spt, (n_t, n_c, 1))

        # memory allocation
        r_pair = np.nan * np.ones((n_c, n_c))

        # calculates the pair-wise pearson correlations between each cell pair (over all trials)
        for i_c0 in range(n_c):
            for i_c1 in range(i_c0 + 1, n_c):
                # sets up the pairwise array and from this calculates the mean/std dev
                n_sp_pair = n_spt[:, np.array([i_c0, i_c1]), :]
                n_sp_mn, n_sp_sd = np.mean(n_sp_pair), np.std(n_sp_pair)

                # calculates the pair-wise pearson correlations
                z_sp_pair = (n_sp_pair - n_sp_mn) / n_sp_sd
                x, y = z_sp_pair[:, 0, :].flatten(), z_sp_pair[:, 1, :].flatten()
                r_pair[i_c0, i_c1] = r_pair[i_c1, i_c0] = pr(x, y)[0]

        # returns the pairwise correlation array
        return r_pair

    # array dimensioning and memory allocation
    n_cond = len(d_data.ttype)
    n_ex, n_t, n_dim = len(n_sp), int(np.size(n_sp[0], axis=0) / (2 * n_cond)), len(np.shape(n_sp[0]))
    d_data.pw_corr = np.empty(n_ex, dtype=object)

    for i_ex in range(n_ex):
        if n_dim == 2:
            # case is analysing non-shuffled data
            d_data.pw_corr[i_ex] = [
                calc_pw_noise_correl(n_sp[i_ex][np.arange(i * n_t, (i + 1) * n_t), :]) for i in range(2 * n_cond)
            ]
        else:
            # case is analysing shuffled data
            d_data.pw_corr[i_ex] = [
                calc_pw_noise_correl(n_sp[i_ex][np.arange(i * n_t, (i + 1) * n_t), :, :]) for i in range(2 * n_cond)
            ]


def run_part_lda_pool(p_data):
    '''

    :param p_data:
    :return:
    '''

    # retrieves the pool data
    data, calc_para, r_filt = p_data[0], p_data[1], p_data[2]
    i_expt, i_cell, n_trial_max, n_cell = p_data[3], dcopy(p_data[4]), p_data[5], p_data[6]

    # sets the required number of cells for the LDA analysis
    for i_ex in range(len(i_expt)):
        # determines the original valid cells for the current experiment
        ii = np.where(i_cell[i_ex])[0]

        # from these cells, set n_cell cells as being valid (for analysis purposes)
        i_cell[i_ex][:] = False
        i_cell[i_ex][ii[np.random.permutation(len(ii))][:n_cell]] = True

    # runs the LDA
    results = run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max)

    # returns the cell count and LDA decoding accuracy values
    return [n_cell, results[1]]


def run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max, d_data=None,
                is_shuffle=False, w_prog=None, calc_corr=False):
    '''

    :param data:
    :param calc_para:
    :param r_filt:
    :param i_expt:
    :param i_cell:
    :param n_trial_max:
    :param d_data:
    :param is_shuffle:
    :return:
    '''


    def reduce_cluster_data(data, i_expt):
        '''

        :param data:
        :param i_expt:
        :param cell_ok:
        :return:
        '''

        # creates a copy of the data and removes any
        data_tmp = dcopy(data)

        # reduces down the number of
        if len(i_expt) != len(data_tmp.cluster):
            data_tmp.cluster = [data_tmp.cluster[i_ex] for i_ex in i_expt]
            data_tmp._cluster = [data_tmp._cluster[i_ex] for i_ex in i_expt]

        # returns the reduced data class object
        return data_tmp

    def run_lda_predictions(w_prog, r_obj, lda_para, i_cell, ind_t, i_ex, is_shuffle):
        '''

        :param r_obj:
        :param cell_ok:
        :param n_trial_max:
        :param i_ex:
        :return:
        '''

        def shuffle_trial_counts(n_sp, n_t):
            '''

            :param n_sp:
            :param n_trial_max:
            :return:
            '''

            # initialisation
            n_cell = np.size(n_sp, axis=1)

            # shuffles the trials for each of the cells
            for i_cell in range(n_cell):
                # sets the permutation array ensures the following:
                #  * CW/CCW trials are shuffled the same between conditions
                #  * Trials are shuffled independently between conditions
                ind_c0, ind_c1 = np.random.permutation(n_t), np.random.permutation(n_t)
                ind_c = np.hstack((ind_c0, ind_c0 + n_t, ind_c1 + (2 * n_t), ind_c1 + (3 * n_t)))

                # shuffles the trials for the current cell
                n_sp[:, i_cell] = n_sp[ind_c, i_cell]

            # returns the array
            return n_sp

        # memory allocation and initialisations
        n_sp, t_sp, i_grp = [], [], []
        n_t, n_grp, N = n_trial_max, 2 * r_obj.n_filt, 2 * r_obj.n_filt * n_trial_max

        # retrieves the cell indices (over each condition) for the current experiment
        ind_c = [np.where(i_expt == i_ex)[0][i_cell] for i_expt in r_obj.i_expt]
        n_cell, pW = len(ind_c[0]), 1 / r_obj.n_expt

        ####################################
        ####    LDA DATA ARRAY SETUP    ####
        ####################################

        # sets up the LDA data/group index arrays across each condition
        for i_filt in range(r_obj.n_filt):
            # retrieves the time spikes for the current filter/experiment, and then combines into a single
            # concatenated array. calculates the final spike counts over each cell/trial and appends to the
            # overall spike count array
            A = dcopy(r_obj.t_spike[i_filt][ind_c[i_filt], :, :])[:, ind_t, :]
            if r_obj.rot_filt['t_type'][i_filt] == 'MotorDrifting':
                # case is motordrifting (swap phases)
                t_sp_tmp = np.hstack((A[:, :, 2], A[:, :, 1]))
            else:
                # case is other experiment conditions
                t_sp_tmp = np.hstack((A[:, :, 1], A[:, :, 2]))

            # calculates the spike counts and appends them to the count array
            n_sp.append(np.vstack([np.array([len(y) for y in x]) for x in t_sp_tmp]))
            t_sp.append(t_sp_tmp)

            # sets the grouping indices
            ind_g = [2 * i_filt, 2 * i_filt + 1]
            i_grp.append(np.hstack((ind_g[0] * np.ones(n_t), ind_g[1] * np.ones(n_t))).astype(int))

        # combines the spike counts/group indices into the final combined arrays
        n_sp, t_sp, i_grp = np.hstack(n_sp).T, np.hstack(t_sp).T, np.hstack(i_grp)

        # shuffles the trials (if required)
        if is_shuffle:
            n_sp = shuffle_trial_counts(n_sp, n_trial_max)

        # normalises the spike count array (if required)
        n_sp_calc = dcopy(n_sp)
        if lda_para['is_norm']:
            n_sp_mn, n_sp_sd = np.mean(n_sp_calc, axis=0), np.std(n_sp_calc, axis=0)
            n_sp_calc = np.divide(n_sp_calc - repmat(n_sp_mn, N, 1), repmat(n_sp_sd, N, 1))

            # any cells where the std. deviation is zero are set to zero (to remove any NaNs)
            n_sp_calc[:, n_sp_sd == 0] = 0

        ###########################################
        ####    LDA PREDICTION CALCULATIONS    ####
        ###########################################

        # memory allocation
        lda_pred, c_mat = np.zeros(N, dtype=int), np.zeros((n_grp, n_grp), dtype=int)
        lda_pred_chance, c_mat_chance = np.zeros(N, dtype=int), np.zeros((n_grp, n_grp), dtype=int)
        p_mat = np.zeros((N, n_grp), dtype=float)

        # sets the LDA solver type
        if lda_para['solver_type'] == 'svd':
            # case the SVD solver
            lda = LDA()
        elif lda_para['solver_type'] == 'lsqr':
            # case is the LSQR solver
            if lda_para['use_shrinkage']:
                lda = LDA(solver='lsqr', shrinkage='auto')
            else:
                lda = LDA(solver='lsqr')
        else:
            # case is the Eigen solver
            if lda_para['use_shrinkage']:
                lda = LDA(solver='eigen', shrinkage='auto')
            else:
                lda = LDA(solver='eigen')

        # fits the LDA model and calculates the prediction for each
        for i_pred in range(len(i_grp)):
            # updates the progress bar
            if w_prog is not None:
                w_str = 'Running LDA Predictions (Expt {0} of {1})'.format(i_ex + 1, r_obj.n_expt)
                w_prog.emit(w_str, 100. * pW * (i_ex + i_pred / len(i_grp)) )

            # fits the one-out-trial lda model
            ii = np.array(range(len(i_grp))) != i_pred
            try:
                lda.fit(n_sp_calc[ii, :], i_grp[ii])
            except:
                e_str = 'There was an error running the LDA analysis with the current solver parameters. ' \
                        'Either choose a different solver or alter the solver parameters before retrying'
                w_prog.emit(e_str, 'LDA Analysis Error')
                return None, False

            # calculates the model prediction from the remaining trial and increments the confusion matrix
            lda_pred[i_pred] = lda.predict(n_sp_calc[i_pred, :].reshape(1, -1))
            p_mat[i_pred, :] = lda.predict_proba(n_sp_calc[i_pred, :].reshape(1, -1))
            c_mat[i_grp[i_pred], lda_pred[i_pred]] += 1

            # fits the one-out-trial shuffled lda model
            ind_chance = np.random.permutation(len(i_grp) - 1)
            lda.fit(n_sp_calc[ii, :], i_grp[ii][ind_chance])

            # calculates the chance model prediction from the remaining trial and increments the confusion matrix
            lda_pred_chance[i_pred] = lda.predict(np.reshape(n_sp_calc[i_pred, :], (1, n_cell)))
            c_mat_chance[i_grp[i_pred], lda_pred_chance[i_pred]] += 1

        # calculates the LDA transform values (uses svd solver to accomplish this)
        if lda_para['solver_type'] != 'lsqr':
            # memory allocation
            lda_X, lda_X0 = np.empty(n_grp, dtype=object), lda.fit(n_sp_calc, i_grp)

            # calculates the variance explained
            if len(lda_X0.explained_variance_ratio_) == 2:
                lda_var_exp = np.round(100 * lda_X0.explained_variance_ratio_.sum(), 2)
            else:
                lda_var_sum = lda_X0.explained_variance_ratio_.sum()
                lda_var_exp = np.round(100 * np.sum(lda_X0.explained_variance_ratio_[:2] / lda_var_sum), 2)

            # separates the transform values into the individual groups
            lda_X0T = lda_X0.transform(n_sp_calc)
            for ig in range(n_grp):
                lda_X[ig] = lda_X0T[(ig * n_t):((ig + 1) * n_t), :2]
        else:
            # transform values are not possible with this solver type
            lda_X, lda_var_exp = None, None

        # returns the final values in a dictionary object
        return {
            'c_mat': c_mat, 'p_mat': p_mat, 'lda_pred': lda_pred,
            'c_mat_chance': c_mat_chance, 'lda_pred_chance': lda_pred_chance,
            'lda_X': lda_X, 'lda_var_exp': lda_var_exp, 'n_cell': n_cell
        }, n_sp, True

    # initialisations
    lda_para = calc_para['lda_para']
    t_ofs, t_phase = get_rot_phase_offsets(calc_para)

    # creates a reduce data object and creates the rotation filter object
    data_tmp = reduce_cluster_data(data, i_expt)
    r_obj = RotationFilteredData(data_tmp, r_filt, None, None, True, 'Whole Experiment', False,
                                 t_ofs=t_ofs, t_phase=t_phase)

    # memory allocation and other initialisations
    A = np.empty(r_obj.n_expt, dtype=object)
    lda, exp_name, n_sp = dcopy(A), dcopy(A), dcopy(A)
    ind_t, n_ex = np.array(range(n_trial_max)), r_obj.n_expt

    # memory allocation for accuracy binary mask calculations
    n_c = len(r_filt['t_type'])
    BG, BD = np.zeros((2 * n_c, 2 * n_c), dtype=bool), np.zeros((2, 2 * n_c), dtype=bool)
    y_acc = np.zeros((n_ex, 1 + n_c), dtype=float)

    # sets up the binary masks for the group/direction types
    for i_c in range(n_c):
        BG[(2 * i_c):(2 * (i_c + 1)), (2 * i_c):(2 * (i_c + 1))] = True
        BD[0, 2 * i_c], BD[1, 2 * i_c + 1] = True, True

    # sets the experiment file names
    f_name0 = [os.path.splitext(os.path.basename(x['expFile']))[0] for x in data_tmp.cluster]

    # loops through each of the experiments performing the lda calculations
    for i_ex in range(n_ex):
        exp_name[i_ex] = f_name0[i_ex]
        lda[i_ex], n_sp[i_ex], ok = run_lda_predictions(w_prog, r_obj, lda_para,
                                                        i_cell[i_ex], ind_t, i_ex, is_shuffle)
        if not ok:
            # if there was an error, then exit with a false flag
            return False

        # calculates the grouping accuracy values
        c_mat = lda[i_ex]['c_mat'] / n_trial_max
        y_acc[i_ex, 0] += np.sum(np.multiply(BG, c_mat)) / (2 * n_c)

        # calculates the direction accuracy values (over each condition)
        for i_c in range(n_c):
            y_acc[i_ex, 1 + i_c] = np.sum(np.multiply(BD, c_mat[(2 * i_c):(2 * (i_c + 1)), :])) / 2

    if d_data is not None:
        # sets the lda values
        d_data.lda = lda
        d_data.y_acc = y_acc
        d_data.exp_name = exp_name

        # sets the solver parameters
        set_lda_para(d_data, lda_para, r_filt, n_trial_max)

        # sets the phase duration/offset parameters
        d_data.tofs = t_ofs
        d_data.tphase = t_phase
        d_data.usefull = calc_para['use_full_rot']

        # calculates the noise correlation (if required)
        if calc_corr:
            calc_noise_correl(d_data, n_sp)

        # returns a true value
        return True
    elif is_shuffle:
        # otherwise, return the calculated values
        return [lda, y_acc, exp_name, n_sp]
    else:
        # otherwise, return the calculated values
        return [lda, y_acc, exp_name]


def init_lda_solver_para():
    return {
        'n_cell_min': 10,
        'n_trial_min': 10,
        'is_norm': True,
        'use_shrinkage': True,
        'solver_type': 'eigen',
        'comp_cond': ['Black', 'Uniform'],
        'cell_types': 'All Cells',
        'y_acc_max': 100,
    }


def init_lda_para(d_data):
    '''

    :param d_data:
    :return:
    '''

    def set_lda_para(para_def, para_curr):
        '''

        :param para0:
        :param paraC:
        :return:
        '''

        if para_curr is None:
            return para_def
        elif isinstance(para_curr, str) or isinstance(para_curr, list):
            return para_curr
        else:
            return para_def if (para_curr < 0) else para_curr

    def set_def_lda_para(d_data, p_str):
        '''

        :param d_data:
        :param p_str:
        :return:
        '''

        # initialisations
        def_para = {}

        # retrieves the values from the data class
        for ps in p_str:
            p_val = d_data.__getattribute__(ps)
            if p_val is None:
                continue
            elif p_val != -1:
                def_para[ps] = p_val

        # returns the lda default parameter dictionary
        return def_para

    # retrieves the default parameter values
    lda_para = init_lda_solver_para()

    # if the lda has been calculated, then use these values
    if d_data.lda is not None:
        # sets the LDA parameters
        lda_para['n_cell_min'] = set_lda_para(lda_para['n_cell_min'], d_data.cellmin)
        lda_para['n_trial_min'] = set_lda_para(lda_para['n_trial_min'], d_data.trialmin)
        lda_para['is_norm'] = set_lda_para(lda_para['is_norm'], d_data.norm)
        lda_para['use_shrinkage'] = set_lda_para(lda_para['use_shrinkage'], d_data.shrinkage)
        lda_para['solver_type'] = set_lda_para(lda_para['solver_type'], d_data.solver)
        lda_para['comp_cond'] = set_lda_para(lda_para['comp_cond'], d_data.ttype)
        lda_para['cell_types'] = set_lda_para(lda_para['cell_types'], d_data.ctype)
        lda_para['y_acc_max'] = set_lda_para(lda_para['y_acc_max'], d_data.yaccmx)

    # sets the default parameters based on the type
    if d_data.type in ['Direction', 'Individual']:
        # case is the default rotational LDA analysis
        def_para = set_def_lda_para(d_data, ['tofs', 'tphase', 'usefull'])

    elif d_data.type == 'Temporal':
        # case is the temporal LDA analysis
        def_para = set_def_lda_para(d_data, ['dt_phs', 'dt_ofs', 'phs_const'])

    elif d_data.type == 'TrialShuffle':
        # case is the shuffled trial LDA analysis
        def_para = set_def_lda_para(d_data, ['tofs', 'tphase', 'usefull', 'nshuffle'])

    elif d_data.type == 'Partial':
        # case is the partial LDA analysis
        def_para = set_def_lda_para(d_data, ['tofs', 'tphase', 'usefull', 'nshuffle', 'cellminpart'])

    # returns the lda solver/default parameter dictionaries
    return lda_para, def_para


def set_def_para(para, p_str, def_val):
    '''

    :param rot_para:
    :param p_str:
    :param def_val:
    :return:
    '''

    # returns the parameter value if present in the para dictionary, otherwise return the default value
    return para[p_str] if p_str in para else def_val


def set_lda_para(d_data, lda_para, r_filt, n_trial_max, ignore_list=[]):
    '''

    :param d_data:
    :param lda_para:
    :param r_filt:
    :param n_trial_max:
    :return:
    '''

    # sets the parameter to class conversion strings
    conv_str = {
        'n_cell_min': 'cellmin',
        'n_trial_min': 'trialmin',
        'solver_type': 'solver',
        'use_shrinkage': 'shrinkage',
        'is_norm': 'norm',
        'cell_types': 'ctype',
        'y_acc_max': 'yaccmx',
        'comp_cond': 'ttype',
    }

    # sets the trial count and trial types
    d_data.ntrial = n_trial_max

    # sets the LDA solver parameters
    for ldp in lda_para:
        if ldp not in ignore_list:
            setattr(d_data, conv_str[ldp], lda_para[ldp])