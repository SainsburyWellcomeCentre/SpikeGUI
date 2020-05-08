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
import pickle as _p
import quantities as pq
from fastdtw import fastdtw
import scikit_posthocs as sp
from numpy.matlib import repmat
import shapely.geometry as geom

# scipy module imports
from scipy import stats
from scipy.stats import poisson as p
from scipy.signal import medfilt
from scipy.stats import pearsonr as pr
from scipy.spatial.distance import *
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d
from scipy.stats.distributions import t

# rpy2 module imports
import rpy2.robjects as robjects
from rpy2.robjects.methods import RS4
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
r_stats = importr("stats")

# sci-kit learn module import
from sklearn.linear_model import LinearRegression

# try:
#     r_art = importr("ARTool")
# except:
#     pass

# sklearn module imports
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# PyQt5 module imports
from PyQt5.QtCore import QRect

# custom module imports
import analysis_guis.common_func as cf
import analysis_guis.rotational_analysis as rot
from analysis_guis.dialogs.rotation_filter import RotationFilteredData

try:
    import analysis_guis.test_plots as tp
except:
    pass

# other function declarations
dcopy, scopy = copy.deepcopy, copy.copy
diff_dist = lambda x, y: np.sum(np.sum((x - y) ** 2, axis=0)) ** 0.5
n_cell_pool0 = [1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500]
n_cell_pool1 = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500]
lda_trial_type = None

# lambda functions
rmv_nan_elements = lambda y: [[np.array(xx)[~np.isnan(xx)] for xx in yy] for yy in y]

########################################################################################################################
########################################################################################################################

class EmptyAnalysisData(object):
    def __init__(self):
        # field initialisation
        self._cluster = []
        self.cluster = None
        self.rotation = None

        # exclusion filter fields
        self.exc_gen_filt = None
        self.exc_rot_filt = None
        self.exc_ud_filt = None

        # other flags
        self.req_update = True
        self.force_calc = True
        self.files = None

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


def calc_ks2_stat(hist_1, hist_2):
    '''

    :param hist_1:
    :param hist_2:
    :return:
    '''

    _, ks_stat = stats.ks_2samp(hist_1, hist_2)
    return ks_stat


def calc_wasserstein(hist_1, hist_2):
    '''

    :param hist_1:
    :param hist_2:
    :return:
    '''

    return stats.wasserstein_distance(hist_1, hist_2)

###############################################
####    ART STATS CALCULATION FUNCTIONS    ####
###############################################

def calc_art_stats(x1, x2, y, c_type='X1'):
    '''

    :param ttype:
    :param x:
    :param y:
    :return:
    '''

    def get_art_test_values(s_data):
        '''

        :param s_data:
        :return:
        '''

        # parameters
        p_tol = 0.05

        # REMOVE ME LATER
        p_str, c_grp = None, None

        # runs the stats based on the type
        if isinstance(s_data, RS4):
            # retrieves the summary dataframe
            summ_df = pandas2ri.ri2py(robjects.r['summary'](s_data))

            # retrieves the category/p-values based on the test type
            if 'x1_pairwise' in summ_df.columns:
                # case is the interaction analysis
                pass

            else:
                # retrieves the category/p-value values
                c_str, p_value = summ_df['contrast'], summ_df['p.value']

                # determines the unique category groupings
                c_str_sp = np.vstack([x.split(' - ') for x in c_str])
                c_str_uniq = list(np.unique(c_str_sp))

                # retrieves the group indices and sorted category groups
                i_grp, n_grp = [[c_str_uniq.index(y) for y in x] for x in c_str_sp], len(c_str_uniq)
                c_grp = [c_str_sp[0, 0]] + list(c_str_sp[:(n_grp - 1), 1])

                # sets the p-value strings for each groups
                p_str, k = np.empty((n_grp, n_grp), dtype=object), 0
                for i in range(n_grp):
                    for j in range(i, n_grp):
                        if i == j:
                            # case is the symmetric case
                            p_str[i, j] = 'N/A'
                        else:
                            # case is the non-symmetric case
                            if p_value[k] < 1e-20:
                                # case is the p-value is <1e-10. so use a fixed value instead
                                p_str_nw = '{:5.3e}*'.format(1e-20)
                            elif p_value[k] < 1e-3:
                                # case is very small p-values, so use compact form
                                p_str_nw = '{:5.3e}*'.format(p_value[k])
                            else:
                                # otherwise, use normal form
                                p_str_nw = '{:5.3f}{}'.format(p_value[k], cf.stats_suffix(p_value[k], p_tol))

                            # sets the final string and increments the counter
                            p_str[i_grp[k][0], i_grp[k][1]] = p_str[i_grp[k][1], i_grp[k][0]] = p_str_nw
                            k += 1

        else:
            # case is the ANOVA analysis
            pass

        # returns the comparison group/statistics string
        return c_grp, p_str

    # calculates the ART anova
    calc_art_anova = robjects.r(
        """
            function(x1, x2, y, s_type){
                library("ARTool")
                library(emmeans)
                library(phia)

                df <- data.frame(x1, x2, y)
                m <- art(y ~ x1 * x2, data = df)
                m_anova <- anova(m)

                if (s_type == 0) {
                out <- contrast(emmeans(artlm(m, "x1"), ~ x1), method="pairwise")                
                } else if (s_type == 1) {
                out <- contrast(emmeans(artlm(m, "x2"), ~ x2), method="pairwise")                
                } else if (s_type == 2) {
                out <- contrast(emmeans(artlm(m, "x1:x2"), ~ x1:x2), method="pairwise", interaction=TRUE)
                } else {
                out <- m_anova
                }
            }
        """
    )

    # returns the stats category groupings/p-values (for the desired stats type)
    ind = ['X1', 'X2', 'X1_X2', 'ANOVA'].index(c_type)
    return get_art_test_values(calc_art_anova(x1, x2, y, ind))

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
    return np.nansum(metric_weighted_mean, axis=2)


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
            elif exc_gen == 'lesion':
                cl_inc = np.logical_and(cl_inc, c['expInfo']['lesion'] != ex_gt)
            elif exc_gen == 'record_state':
                cl_inc = np.logical_and(cl_inc, c['expInfo']['record_state'] != ex_gt)

            # if there are no valid cells, then exit the function
            if not np.any(cl_inc):
                return cl_inc

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

    def calc_pw_noise_correl(n_spt0, is_3d=False):
        '''

        :param n_spt:
        :return:
        '''

        # array dimensioning and parameters
        z_tol = 3.
        n_t0, n_c = int(np.size(n_spt0, axis=0) / 2), np.size(n_spt0, axis=1)

        # array dimensioning
        if len(np.shape(n_spt0)) == 2:
            # if a 2D array, then convert to a 3D array by including a redundant 3rd axis
            n_spt = np.reshape(n_spt0[:n_t0, :] + n_spt0[n_t0:, :], (n_t0, n_c, 1))
        else:
            n_spt = n_spt0[:n_t0, :, :] + n_spt0[n_t0:, :, :]

        # memory allocation
        r_pair = np.nan * np.ones((n_c, n_c))
        z_sp_pair = np.empty((n_c, n_c), dtype=object)

        # calculates the pair-wise pearson correlations between each cell pair (over all trials)
        for i_c0 in range(n_c):
            for i_c1 in range(i_c0 + 1, n_c):
                # sets up the pairwise array and from this calculates the mean/std dev
                n_sp_pair = n_spt[:, np.array([i_c0, i_c1]), :]
                x, y = n_sp_pair[:, 0, :].flatten(), n_sp_pair[:, 1, :].flatten()

                # calculates the x/y z-scored values
                x_z, y_z = (x - np.mean(x)) / np.std(x), (y - np.mean(y)) / np.std(y)
                is_acc = np.logical_and(np.abs(x_z) < z_tol, np.abs(y_z) < z_tol)

                # calculates the pair-wise pearson correlations
                r_pair[i_c0, i_c1] = r_pair[i_c1, i_c0] = pr(x_z[is_acc], y_z[is_acc])[0]
                z_sp_pair[i_c0, i_c1] = np.vstack((x_z[is_acc], y_z[is_acc])).T

        # returns the pairwise correlation array
        if is_3d:
            return r_pair
        else:
            return r_pair, z_sp_pair

    # array dimensioning
    n_cond = len(d_data.ttype)
    n_ex, n_t, n_dim = len(n_sp), int(np.size(n_sp[0], axis=0) / (2 * n_cond)), len(np.shape(n_sp[0]))

    # memory allocation
    A = np.empty(n_ex, dtype=object)
    d_data.pw_corr = dcopy(A)

    for i_ex in range(n_ex):
        if n_dim == 2:
            # memory allocation (first iteration only)
            if i_ex == 0:
                d_data.z_corr = dcopy(A)

            # case is analysing non-shuffled data
            d_data.pw_corr[i_ex], d_data.z_corr[i_ex] = zip(*[
                calc_pw_noise_correl(n_sp[i_ex][np.arange(i * n_t, (i + 2) * n_t), :]) for i in range(n_cond)
            ])
        else:
            # case is analysing shuffled data
            d_data.pw_corr[i_ex] = [
                calc_pw_noise_correl(n_sp[i_ex][np.arange(i * n_t, (i + 2) * n_t), :, :], True) for i in range(n_cond)
            ]


def set_def_para(para, p_str, def_val):
    '''

    :param rot_para:
    :param p_str:
    :param def_val:
    :return:
    '''

    # returns the parameter value if present in the para dictionary, otherwise return the default value
    return dcopy(para[p_str]) if p_str in para else dcopy(def_val)


########################################################################################################################
####                                           LDA CALCULATION FUNCTIONS                                            ####
########################################################################################################################

######################################
####    ROTATION LDA FUNCTIONS    ####
######################################


def setup_lda_spike_counts(r_obj, i_cell, i_ex, n_t, return_all=True):
    '''

    :param r_obj:
    :param i_cell:
    :param i_ex:
    :param n_t:
    :return:
    '''

    # memory allocation
    n_sp, t_sp, i_grp = [], [], []
    ind_c = [np.where(i_expt == i_ex)[0][i_cell] for i_expt in r_obj.i_expt]
    n_cell, ind_t = len(ind_c[0]), np.array(range(n_t))

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
    n_sp, i_grp = np.hstack(n_sp).T, np.hstack(i_grp)

    # returns the important arrays
    if return_all:
        return n_sp, t_sp, i_grp, n_cell
    else:
        return n_sp, i_grp


def norm_spike_counts(n_sp, N, is_norm):
    '''

    :param n_sp:
    :return:
    '''

    # makes a copy of the spike count array
    n_sp_calc = dcopy(n_sp)

    # normalises the spike counts (if required)
    if is_norm:
        n_sp_mn, n_sp_sd = np.mean(n_sp_calc, axis=0), np.std(n_sp_calc, axis=0)
        n_sp_calc = np.divide(n_sp_calc - repmat(n_sp_mn, N, 1), repmat(n_sp_sd, N, 1))

        # any cells where the std. deviation is zero are set to zero (to remove any NaNs)
        n_sp_calc[:, n_sp_sd == 0] = 0

    # returns the final values
    return n_sp_calc


def run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max, d_data=None, is_shuffle=False, w_prog=None,
                pW0=0., pW=100., n_sp0=None):
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

    def run_lda_predictions(w_prog, r_obj, lda_para, n_sp, i_cell, i_ex, is_shuffle, pW0, pW):
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
            n_cond = int(np.size(n_sp, axis=0) / (2 * n_t))

            # shuffles the trials for each of the cells
            for i_cell in range(n_cell):
                # sets the permutation array ensures the following:
                #  * CW/CCW trials are shuffled the same between conditions
                #  * Trials are shuffled independently between conditions
                ind_perm = [np.random.permutation(n_t) for _ in range(n_cond)]
                ind_c = np.hstack([np.hstack((x + 2 * i * n_t, x + (2 * i + 1) * n_t)) for i, x in enumerate(ind_perm)])

                # shuffles the trials for the current cell
                n_sp[:, i_cell] = n_sp[ind_c, i_cell]

            # returns the array
            return n_sp

        # memory allocation and initialisations
        is_OTO = get_glob_para('lda_trial_type') == 'One-Trial Out'
        n_t, n_grp, N = n_trial_max, 2 * r_obj.n_filt, 2 * r_obj.n_filt * n_trial_max
        pWS = 1 / r_obj.n_expt

        ####################################
        ####    LDA DATA ARRAY SETUP    ####
        ####################################

        if n_sp is None:
            # case is the spike counts haven't been provided
            n_sp, t_sp, i_grp, n_cell = setup_lda_spike_counts(r_obj, i_cell, i_ex, n_trial_max)

        else:
            # case is the spike counts have been provided

            # retrieves the cell count and group indices
            n_cell = np.size(n_sp, axis=1)
            i_grp = np.array(cf.flat_list([list(x * np.ones(n_trial_max, dtype=int)) for x in range(n_grp)]))

        # shuffles the trials (if required)
        if is_shuffle:
            n_sp = shuffle_trial_counts(n_sp, n_trial_max)

        # normalises the spike count array (if required)
        n_sp_calc = norm_spike_counts(n_sp, N, lda_para['is_norm'])

        ###########################################
        ####    LDA PREDICTION CALCULATIONS    ####
        ###########################################

        # sets the LDA solver type
        lda = setup_lda_solver(lda_para)

        # memory allocation
        lda_pred, c_mat = np.zeros(N, dtype=int), np.zeros((n_grp, n_grp), dtype=int)
        lda_pred_chance, c_mat_chance = np.zeros(N, dtype=int), np.zeros((n_grp, n_grp), dtype=int)
        p_mat, is_keep = np.zeros((N, n_grp), dtype=float), np.ones(N, dtype=bool)

        # sets the total number of iterations (based on trial setup type)
        if is_OTO:
            # case is "one-trial out" setup
            NN, xi_rmv = n_trial_max, np.arange(0, N, n_trial_max)
        else:
            # case is "one-phase out" setup
            NN = len(i_grp)

        # fits the LDA model and calculates the prediction for each
        for i_trial in range(NN):
            # updates the progress bar
            if w_prog is not None:
                w_str = 'Running LDA Predictions (Expt {0} of {1})'.format(i_ex + 1, r_obj.n_expt)
                w_prog.emit(w_str, pW0 + pW * pWS * (i_ex + i_trial / NN))

            # fits the one-out-trial lda model
            if is_OTO:
                is_keep[xi_rmv + i_trial] = False
            else:
                is_keep[i_trial] = False

            # fits the one-out-trial lda model
            try:
                lda.fit(n_sp_calc[is_keep, :], i_grp[is_keep])
            except:
                if w_prog is not None:
                    e_str = 'There was an error running the LDA analysis with the current solver parameters. ' \
                            'Either choose a different solver or alter the solver parameters before retrying'
                    w_prog.emit(e_str, 'LDA Analysis Error')
                return None, False

            # resets the acceptance array
            if is_OTO:
                # calculates the model prediction from the remaining trial and increments the confusion matrix
                for i in (xi_rmv + i_trial):
                    lda_pred[i] = lda.predict(n_sp_calc[i, :].reshape(1, -1))
                    p_mat[i, :] = lda.predict_proba(n_sp_calc[i, :].reshape(1, -1))
                    c_mat[i_grp[i], lda_pred[i]] += 1

                    # re-adds the removed trial
                    is_keep[i] = True

            else:
                # calculates the model prediction from the remaining trial and increments the confusion matrix
                lda_pred[i_trial] = lda.predict(n_sp_calc[i_trial, :].reshape(1, -1))
                p_mat[i_trial, :] = lda.predict_proba(n_sp_calc[i_trial, :].reshape(1, -1))
                c_mat[i_grp[i_trial], lda_pred[i_trial]] += 1

                # re-adds the removed trial
                is_keep[i_trial] = True

            # # fits the one-out-trial shuffled lda model
            # ind_chance = np.random.permutation(len(i_grp) - 1)
            # lda.fit(n_sp_calc[ii, :], i_grp[ii][ind_chance])
            #
            # # calculates the chance model prediction from the remaining trial and increments the confusion matrix
            # lda_pred_chance[i_pred] = lda.predict(np.reshape(n_sp_calc[i_pred, :], (1, n_cell)))
            # c_mat_chance[i_grp[i_pred], lda_pred_chance[i_pred]] += 1


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
    n_ex = len(i_expt)

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
        lda[i_ex], n_sp[i_ex], ok = run_lda_predictions(w_prog, r_obj, lda_para, n_sp0, i_cell[i_ex],
                                                        i_ex, is_shuffle, pW0, pW)
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
        d_data.i_expt = i_expt
        d_data.i_cell = i_cell
        d_data.y_acc = y_acc
        d_data.exp_name = exp_name
        d_data.lda_trial_type = get_glob_para('lda_trial_type')

        # sets the solver parameters
        set_lda_para(d_data, lda_para, r_filt, n_trial_max)

        # sets the phase duration/offset parameters
        d_data.tofs = t_ofs
        d_data.tphase = t_phase
        d_data.usefull = calc_para['use_full_rot']

        # updates the progress bar (if provided)
        if w_prog is not None:
            w_prog.emit('Calculating Pairwise Correlations...', pW0 + 0.99 * pW)

        # calculates the noise correlation (if required)
        calc_noise_correl(d_data, n_sp)

        # returns a true value
        return True

    elif is_shuffle:
        # otherwise, return the calculated values
        return [lda, y_acc, exp_name, n_sp]

    else:
        # otherwise, return the calculated values
        return [lda, y_acc, exp_name]


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


def run_reducing_cell_lda(w_prog, lda, lda_para, n_sp, i_grp, p_w0, p_w, w_str, is_dec=False):
    '''

    :param w_prog:
    :param lda:
    :param calc_para:
    :param n_sp:
    :param j_grp:
    :param n_t:
    :param is_dec:
    :return:
    '''

    def run_lda_predictions(lda, n_sp, i_grp):
        '''

        :param w_prog:
        :param lda:
        :param n_sp:
        :param i_grp:
        :return:
        '''

        # memory allocation
        n_tt, n_cell = np.shape(n_sp)
        c_mat, is_keep, n_grp = np.zeros((2, 2), dtype=int), np.ones(n_tt, dtype=bool), i_grp[-1] + 1
        is_OTO, n_t = get_glob_para('lda_trial_type') == 'One-Trial Out', int(n_tt / n_grp)

        # sets the total number of iterations (based on trial setup type)
        if is_OTO:
            # case is "one-trial out" setup
            NN, xi_rmv = n_t, np.arange(0, n_tt, n_t)
        else:
            # case is "one-phase out" setup
            NN = len(i_grp)

        # fits the LDA model and calculates the prediction for each
        for i_trial in range(NN):
            # fits the one-out-trial lda model
            if is_OTO:
                is_keep[xi_rmv + i_trial] = False
            else:
                is_keep[i_trial] = False

            # fits the one-out-trial lda model
            try:
                lda.fit(n_sp[is_keep, :], i_grp[is_keep])
            except:
                if w_prog is not None:
                    e_str = 'There was an error running the LDA analysis with the current solver parameters. ' \
                            'Either choose a different solver or alter the solver parameters before retrying'
                    w_prog.emit(e_str, 'LDA Analysis Error')
                return np.nan

            if is_OTO:
                # calculates the model prediction from the remaining trial and increments the confusion matrix
                for i in (xi_rmv + i_trial):
                    lda_pred = lda.predict(n_sp[i, :].reshape(1, -1))
                    c_mat[i_grp[i], lda_pred] += 1

                    # re-adds the removed trial
                    is_keep[i] = True
            else:
                # calculates the model prediction from the remaining trial and increments the confusion matrix
                lda_pred = lda.predict(n_sp[i_trial, :].reshape(1, -1))
                c_mat[i_grp[i_trial], lda_pred] += 1

                # re-adds the removed trial
                is_keep[i_trial] = True

        # returns the decoding accuracy
        return (c_mat[0, 0] + c_mat[1, 1]) / n_tt

    # array indexing and memory allocation
    n_tf, n_cell = np.shape(n_sp)
    use_cell, y_acc = np.ones(n_cell, dtype=bool), np.zeros(n_cell)

    for i_cell in range(n_cell):
        # updates the progressbar
        w_str_nw = '{0}, Cell {1}/{2})'.format(w_str, i_cell + 1, n_cell)
        w_prog.emit(w_str_nw, 100 * (p_w0  + p_w * (i_cell / n_cell)))

        # normalises the spike counts (based on the remaining cells)
        n_sp_norm = norm_spike_counts(n_sp[:, use_cell], n_tf, lda_para['is_norm'])

        # runs the lda predictions for the current configuration
        y_acc[i_cell] = run_lda_predictions(lda, n_sp_norm, i_grp)

        # removes the next cell from the analysis
        if is_dec:
            # case is the top rated coefficients are being removed
            use_cell[i_cell] = False
        else:
            # case is the lowest rated coefficients are being removed
            use_cell[-(i_cell + 1)] = False

    # returns the accuracy values
    return y_acc

#######################################
####    KINEMATIC LDA FUNCTIONS    ####
#######################################

def calc_binned_kinemetic_spike_freq(data, plot_para, calc_para, w_prog, roc_calc=True, replace_ttype=True, r_data=None):
    '''

    :param calc_para:
    :return:
    '''

    # initialises the RotationData class object (if not provided)
    if r_data is None:
        r_data = data.rotation

    # parameters and initialisations
    vel_bin, equal_time = float(calc_para['vel_bin']), calc_para['equal_time']

    # sets the condition types (ensures that the black phase is always included)
    r_filt_base = cf.init_rotation_filter_data(False)
    if plot_para['rot_filt'] is not None:
        if replace_ttype:
            r_filt_base['t_type'] = plot_para['rot_filt']['t_type']
        else:
            r_filt_base['t_type'] = list(np.unique(r_filt_base['t_type'] + plot_para['rot_filt']['t_type']))

    # sets up the black phase data filter and returns the time spikes
    r_data.r_obj_kine = RotationFilteredData(data, r_filt_base, 0, None, True, 'Whole Experiment', False)

    # memory allocation
    if equal_time:
        if r_data.vel_sf_rs is None:
            r_data.vel_sf_rs, r_data.spd_sf_rs = {}, {}
    else:
        if r_data.vel_sf is None:
            r_data.vel_sf, r_data.spd_sf = {}, {}

    # sets the speed/velocity time counts
    if equal_time:
        # case is using resampling from equal time bin sizes
        n_rs = calc_para['n_sample']
        vel_f, xi_bin, dt = rot.calc_resampled_vel_spike_freq(data, w_prog, r_data.r_obj_kine, [vel_bin], n_rs)
    else:
        # calculates the velocity kinematic frequencies
        w_prog.emit('Calculating Fixed Spiking Frequencies', 50.)
        vel_f, xi_bin, dt = rot.calc_kinemetic_spike_freq(data, r_data.r_obj_kine, [10, vel_bin], calc_type=1)

    # resets the frequencies based on the types
    for i_filt in range(len(vel_f)):
        if len(np.shape(vel_f[i_filt])) == 4:
            if calc_para['freq_type'] == 'All':
                # case is considering mean frequency types (take mean of the decreasing/increasing velocity frequencies)
                vel_f[i_filt] = np.mean(vel_f[i_filt], axis=3)
            elif calc_para['freq_type'] == 'Decreasing':
                # case is only considering decreasing velocity frequencies
                vel_f[i_filt] = vel_f[i_filt][:, :, :, 0]
            elif calc_para['freq_type'] == 'Increasing':
                # case is only considering increasing velocity frequencies
                vel_f[i_filt] = vel_f[i_filt][:, :, :, 1]
            elif calc_para['freq_type'] == 'Both':
                # case is only considering both velocity frequency types (stack both types on top of each other)
                vel_f[i_filt] = np.vstack((vel_f[i_filt][:, :, :, 0], vel_f[i_filt][:, :, :, 1]))

    # sets the comparison bin for the velocity/speed arrays
    n_filt, i_bin0 = r_data.r_obj_kine.n_filt, np.where(xi_bin[:, 0] == 0)[0][0]
    r_data.is_equal_time = equal_time
    r_data.vel_xi, r_data.spd_xi = xi_bin, xi_bin[i_bin0:, :]

    if 'spd_x_rng' in calc_para:
        x_rng = calc_para['spd_x_rng'].split()
        r_data.i_bin_spd = list(xi_bin[i_bin0:, 0]).index(int(x_rng[0]))
        r_data.i_bin_vel = [list(xi_bin[:, 0]).index(-int(x_rng[2])), list(xi_bin[:, 0]).index(int(x_rng[0]))]

    # calculates the velocity/speed binned spiking frequencies
    for i_filt, rr in enumerate(r_data.r_obj_kine.rot_filt_tot):
        # memory allocation
        tt = rr['t_type'][0]

        # if the values have already been calculated, then continue
        if equal_time:
            if tt in r_data.vel_sf_rs:
                continue
        else:
            if tt in r_data.vel_sf:
                continue

        # sets the speed frequencies into a single array
        spd_f = np.vstack((np.flip(vel_f[i_filt][:, :i_bin0, :], 1), vel_f[i_filt][:, i_bin0:, :]))

        if equal_time:
            # case is using equally spaced time bins
            r_data.vel_sf_rs[tt] = dcopy(vel_f[i_filt])
            r_data.spd_sf_rs[tt] = dcopy(spd_f)

        else:
            # case is using the normal time bins
            r_data.vel_sf[tt], r_data.spd_sf[tt] = dcopy(vel_f[i_filt]), dcopy(spd_f)

    # outputs a message to screen
    w_prog.emit('Spiking Frequency Calculation Complete!', 100.)


def calc_shuffled_kinematic_spike_freq(data, calc_para, w_prog):
    '''

    :param data:
    :param plot_para:
    :param calc_para:
    :param w_prog:
    :return:
    '''

    def shuffle_cond_spike_freq(r_data, calc_para, vel_sf, w_prog, pW, i_filt, tt):
        '''

        :param calc_para:
        :param vel_sf:
        :return:
        '''

        def shuffle_cell_spike_freq(calc_para, vel_sf_grp):
            '''

            :param calc_para:
            :param n_sp_cell:
            :return:
            '''

            # initialisations
            n_shuffle, n_sm = calc_para['n_shuffle'], calc_para['n_smooth'] * calc_para['is_smooth']
            n_trial, n_bin = np.shape(vel_sf_grp)

            # memory allocation
            v_sf_shuff = np.zeros((n_shuffle, n_bin, n_trial))

            # sets up the shuffled spike counts over all trials/shuffles
            for i_trial in range(n_trial):
                for i_shuffle in range(n_shuffle):
                    ind_shuffle = np.random.permutation(n_bin)
                    v_sf_shuff[i_shuffle, :, i_trial] = vel_sf_grp[i_trial, ind_shuffle]

            # returns the mean spiking frequencies over all trials
            return smooth_signal(np.mean(v_sf_shuff, axis=2), n_sm)

        def calc_spike_freq_corr(v_bin, v_sf, mlt):
            '''

            :param v_bin:
            :param v_sf:
            :param mlt:
            :return:
            '''

            # memory allocation
            n_sf = np.size(v_sf, axis=0)
            v_corr = np.zeros(n_sf)

            # calculate the correlation coefficients for each shuffled value
            for i_sf in range(n_sf):
                v_corr[i_sf] = mlt * np.corrcoef(v_sf[i_sf, :], v_bin)[0, 1]

            # returns the correlation array
            return v_corr

        # parameters
        p_value = 2.5           # SET THIS TO EITHER 2.5 or 5

        # initialisations
        pW0 = i_filt * pW
        n_trial, n_bin, n_cell = np.shape(vel_sf)
        n_sm = calc_para['n_smooth'] * calc_para['is_smooth']

        # sets the indices of the groups (based on whether the velocity is being split)
        if calc_para['split_vel']:
            ind_grp = [np.arange(n_bin / 2).astype(int), np.arange(n_bin / 2, n_bin).astype(int)]
        else:
            ind_grp = [np.arange(n_bin).astype(int)]

        # sets the negative/positive velocity indices
        v_bin_grp = [np.mean(r_data.vel_xi[i_b, :], axis=1) for i_b in ind_grp]
        mlt = [-1, 1] if len(ind_grp) == 2 else [1]

        # memory allocation
        n_grp, n_shuffle = len(ind_grp), calc_para['n_shuffle']
        A = np.empty((n_cell, n_grp), dtype=object)
        v_sf_sh, v_sf_mu, is_sig = dcopy(A), dcopy(A), np.zeros((n_cell, n_grp), dtype=bool)
        v_corr, v_corr_sh = np.ones((n_cell, n_grp)), np.ones((n_shuffle, n_cell, n_grp))

        # loops through each cell calculating the shuffled trials/correlations
        for i_cell in range(n_cell):
            # updates the progressbar
            w_str = 'Shuffling Spike Frequecies ({0} - {1}/{2})'.format(tt, i_cell + 1, n_cell)
            w_prog.emit(w_str, 100 * (pW0 + pW * i_cell / n_cell))

            for i_grp in range(n_grp):
                # sets up the binned spike counts for the current cell (NB - spike counts may not necessarily be integer
                # because the velocity spiking averages are the average of both the increasing/decreasing velocities)
                vel_sf_grp = vel_sf[:, :, i_cell][:, ind_grp[i_grp]]
                vel_sf_grp = vel_sf_grp[np.logical_not(np.isnan(vel_sf_grp[:, 0])), :]

                # calculates the shuffled spiking frequencies
                v_sf_sh[i_cell, i_grp] = shuffle_cell_spike_freq(calc_para, vel_sf_grp)
                v_corr_sh[:, i_cell, i_grp] = calc_spike_freq_corr(v_bin_grp[i_grp], v_sf_sh[i_cell, i_grp], mlt[i_grp])

                # calculates the mean
                v_sf_mu[i_cell, i_grp] = smooth_signal(np.mean(vel_sf_grp, axis=0), n_sm)
                v_corr[i_cell, i_grp] = mlt[i_grp] * np.corrcoef(v_sf_mu[i_cell, i_grp], v_bin_grp[i_grp])[0, 1]

                # calculates the cell correlation significance
                p_tile = np.percentile(v_corr_sh[:, i_cell, i_grp], [p_value, (100 - p_value)])
                is_sig[i_cell, i_grp] = (v_corr[i_cell, i_grp] < p_tile[0]) or (v_corr[i_cell, i_grp] > p_tile[1])

        # returns the shuffled spiking frequencies, correlation arrays
        return v_sf_mu, v_sf_sh, v_corr_sh, v_corr, is_sig

    # if the shuffled spiking frequencies have been calculated already, then exit the function
    if data.rotation.vel_shuffle_calc:
        return

    # initialisations
    r_data, equal_time = data.rotation, calc_para['equal_time']
    r_obj_k, vel_sf = r_data.r_obj_kine, dcopy(r_data.vel_sf_rs) if equal_time else dcopy(r_data.vel_sf)
    n_filt = len(r_obj_k.rot_filt_tot)
    pW = 1 / n_filt

    # initialises the correlation data fields
    if r_data.vel_sf_corr is None:
        r_data.vel_sf_mean, r_data.vel_sf_sig = {}, {}
        r_data.vel_sf_corr_mn, r_data.vel_sf_corr, r_data.vel_sf_shuffle = {}, {}, {}

    # sets the calculation parameters
    r_data.vel_sf_nsm = calc_para['n_smooth'] * calc_para['is_smooth']
    r_data.vel_bin_corr = float(calc_para['vel_bin'])
    r_data.n_shuffle_corr = float(calc_para['n_shuffle'])
    r_data.split_vel = calc_para['split_vel']
    r_data.vel_sf_eqlt = equal_time

    # calculates the velocity/speed binned spiking frequencies
    for i_filt, rr in enumerate(r_obj_k.rot_filt_tot):
        # retrieves
        tt = rr['t_type'][0]
        if tt in r_data.vel_sf_corr:
            # if the values have already been calculated, then continue
            continue

        # calculates the shuffled spike frequencies
        vel_sf_mean, vel_sf_shuffle, vel_sf_corr, vel_sf_corr_mn, vel_sf_sig = \
                    shuffle_cond_spike_freq(r_data, calc_para, dcopy(vel_sf[tt]), w_prog, pW, i_filt, tt)

        # sets the final arrays into the class object
        r_data.vel_sf_mean[tt], r_data.vel_sf_shuffle[tt] = vel_sf_mean, vel_sf_shuffle
        r_data.vel_sf_corr[tt], r_data.vel_sf_corr_mn[tt] = vel_sf_corr, vel_sf_corr_mn
        r_data.vel_sf_sig[tt] = vel_sf_sig

    # updates the shuffled spiking frequency calculation flag
    r_data.vel_shuffle_calc = True


def calc_shuffled_sf_corr(f_corr, i_file, calc_para, i_prog, w_prog):
    '''

    :param f_corr:
    :param i_file:
    :param calc_para:
    :param i_prog:
    :param w_prog:
    :return:
    '''

    def calc_linregress_para(sf_fix, sf_free):
        '''

        :param sf_fix:
        :param sf_free:
        :return:
        '''

        sf_fix_fit, sf_free_fit = sf_fix.reshape(-1, 1), sf_free.reshape(-1, 1)
        model = LinearRegression(fit_intercept=True).fit(sf_fix_fit, sf_free_fit)

        return np.array([model.coef_[0][0],model.intercept_[0]])

    def calc_cell_shuffled_corr(sf_fix, sf_free, n_sh):
        '''

        :param sf_fix:
        :param sf_free:
        :param n_shuff:
        :return:
        '''

        # memory allocation
        n_bin = len(sf_fix)
        sf_corr = np.zeros(n_sh)

        # calculate the correlation coefficients for each shuffled value
        for i_sh in range(n_sh):
            ind_sh = np.random.permutation(n_bin)
            sf_corr[i_sh] = np.corrcoef(sf_fix, sf_free[ind_sh])[0, 1]

        # returns the correlation array
        return sf_corr

    # parameters
    p_value = 2.5       # sets this to either 2.5 or 5
    p_value_rng = [p_value, (100 - p_value)]

    # initialisations
    n_shuff, n_cond = calc_para['n_shuffle'], np.size(f_corr.sf_fix, axis=1)
    sf_fix, sf_free, sf_grad = f_corr.sf_fix[i_file, :], f_corr.sf_free[i_file, :], f_corr.sf_grad[i_file, :]
    sf_corr, sf_corr_sh, is_sig = f_corr.sf_corr[i_file, :], f_corr.sf_corr_sh[i_file, :], f_corr.sf_corr_sig[i_file, :]

    # memory allocation
    n_cell, n_bin = np.shape(sf_fix[0])
    if calc_para['split_vel']:
        ind_grp = [np.arange(n_bin/2).astype(int), np.arange(n_bin/2, n_bin).astype(int)]
    else:
        ind_grp = [np.arange(n_bin).astype(int)]

    n_grp = len(ind_grp)
    for i_cond in range(n_cond):
        sf_grad[i_cond] = np.zeros((n_cell, 2, n_grp))
        sf_corr[i_cond] = np.zeros((n_cell, n_grp))
        sf_corr_sh[i_cond] = np.zeros((n_cell, n_shuff, n_grp))
        is_sig[i_cond] = np.zeros((n_cell, n_grp), dtype=int)

    #
    for i_cell in range(n_cell):
        # updates the progressbar
        i_cell_tot = (i_cell + 1) + i_prog[0]
        w_str = 'Correlation Calculations (Cell #{0}/{1})'.format(i_cell_tot, i_prog[1])
        w_prog.emit(w_str, 100. * (i_cell_tot / i_prog[1]))

        # if there is no spiking frequency data then continue
        if np.isnan(sf_free[0][i_cell, 0]):
            continue

        # calculates the correlation/shuffled correlations over all conditions/velocity polarities
        for i_cond in range(n_cond):
            # retrieves the free/fixed spiking frequencies for the velocity range
            sf_fix_cond = sf_fix[i_cond][i_cell, :]
            sf_free_cond = sf_free[i_cond][i_cell, :]

            for i_grp in range(n_grp):
                # sets the values for the current velocity bin grouping
                sf_fix_nw, sf_free_nw = sf_fix_cond[ind_grp[i_grp]], sf_free_cond[ind_grp[i_grp]]

                # sets the linear regression values
                sf_grad[i_cond][i_cell, :, i_grp] = calc_linregress_para(sf_fix_nw, sf_free_nw)

                # calculates the cell spiking frequency correlations
                sf_corr[i_cond][i_cell, i_grp] = np.corrcoef(sf_fix_nw, sf_free_nw)[0, 1]
                sf_corr_sh[i_cond][i_cell, :, i_grp] = calc_cell_shuffled_corr(sf_fix_nw, sf_free_nw, n_shuff)

                # calculates the cell's shuffled spiking frequency correlations and statistical significance
                p_tile = np.percentile(sf_corr_sh[i_cond][i_cell, :, i_grp], p_value_rng)
                is_sig[i_cond][i_cell, i_grp] = int(sf_corr[i_cond][i_cell, i_grp] > p_tile[1]) - \
                                                int(sf_corr[i_cond][i_cell, i_grp] < p_tile[0])


def setup_kinematic_lda_sf(data, r_filt, calc_para, i_cell, n_trial_max, w_prog,
                           is_pooled=False, use_spd=True, r_data=None):
    '''

    :param data:
    :param r_filt:
    :param calc_para:
    :param i_cell:
    :param n_trial_max:
    :param w_prog:
    :return:
    '''

    # initialises the RotationData class object (if not provided)
    if r_data is None:
        r_data = data.rotation

    # initialisations
    tt = r_filt['t_type']

    # calculates the binned kinematic spike frequencies
    _plot_para = {'rot_filt': {'t_type': calc_para['lda_para']['comp_cond']}}

    # splits up the spiking frequencies into separate experiments (removing any non-valid cells)
    ind = np.concatenate(([0], np.cumsum([len(x) for x in i_cell])))
    ind_ex = [np.arange(ind[i], ind[i + 1]) for i in range(len(i_cell))]

    # averages the spiking frequencies (between the positive/negative velocities) and sets the reqd max trials
    if use_spd:
        # calculates the binned spiking frequencies
        calc_binned_kinemetic_spike_freq(data, _plot_para, calc_para, w_prog, roc_calc=False, replace_ttype=True)

        # retrieves the spiking frequencies based on the calculation type
        if calc_para['equal_time']:
            # case is the equal timebin (resampled) spiking frequencies
            spd_sf0 = dcopy(r_data.spd_sf_rs)
        else:
            # case is the non-equal timebin spiking frequencies
            spd_sf0 = dcopy(r_data.spd_sf)

            # reduces the arrays to only include the first n_trial_max trials (averages pos/neg phases)
        n_t = int(np.size(spd_sf0[tt[0]], axis=0) / 2)
        sf = [0.5 * (spd_sf0[ttype][:n_t, :, :] + spd_sf0[ttype][n_t:, :, :])[:n_trial_max, :, :] for ttype in tt]
    else:
        # initialisations and memory allocation
        sf, _calc_para = np.empty(2, dtype=object), dcopy(calc_para)

        # calculates the binned spiking frequencies
        for i_ft, ft in enumerate(['Decreasing', 'Increasing']):
            # resets the velocity spiking freqencies
            if calc_para['equal_time']:
                r_data.vel_sf_rs = None
            else:
                r_data.vel_sf = None

            # calculates the binned frequecies
            _calc_para['freq_type'] = ft
            calc_binned_kinemetic_spike_freq(data, _plot_para, _calc_para, w_prog, roc_calc=False, replace_ttype=True)

            # retrieves the spiking frequencies based on the calculation type
            if calc_para['equal_time']:
                # case is the equal timebin (resampled) spiking frequencies
                vel_sf0 = dcopy(r_data.vel_sf_rs)
            else:
                # case is the non-equal timebin spiking frequencies
                vel_sf0 = dcopy(r_data.vel_sf)

            # reduces the arrays to only include the first n_trial_max trials
            sf[i_ft] = [vel_sf0[ttype][:n_trial_max, :, :] for ttype in tt]

    # sets up the final speed spiking frequency based on the analysis type
    if is_pooled:
        # case is for the pooled analysis
        sf_ex = [np.dstack([_sf[:, :, i_ex][:, :, i_c] for i_ex, i_c in zip(ind_ex, i_cell)]) for _sf in sf]
    else:
        # case is for the non-pooled analysis
        if use_spd:
            sf_ex = [[_sf[:, :, i_ex][:, :, i_c] for _sf in sf] for i_ex, i_c in zip(ind_ex, i_cell)]
        else:
            sf_ex = [[[_sf[:, :, i_ex][:, :, i_c] for _sf in sf[i]] for i in range(2)]
                                                  for i_ex, i_c in zip(ind_ex, i_cell)]

    # retrieves the rotation kinematic trial types (as they could be altered from the original list)
    _r_filt = dcopy(r_filt)
    _r_filt['t_type'] = r_data.r_obj_kine.rot_filt['t_type']

    # returns the final array
    return sf_ex, _r_filt


def run_full_kinematic_lda(data, spd_sf, calc_para, r_filt, n_trial,
                           w_prog=None, d_data=None, r_data=None):
    '''

    :param data:
    :param spd_sf:
    :param calc_para:
    :param r_filt:
    :param n_trial:
    :param w_prog:
    :param d_data:
    :return:
    '''

    def run_lda_predictions(w_prog, w_str0, pw_0, n_ex, lda, spd_sf, i_grp):
        '''

        :param spd_sf:
        :param lda:
        :return:
        '''

        # array dimensioning
        N, n_grp = np.size(spd_sf, axis=0), i_grp[-1] + 1
        is_OTO = get_glob_para('lda_trial_type') == 'One-Trial Out'
        is_keep, n_t = np.ones(N, dtype=bool), int(N / n_grp)

        # memory allocation
        lda_pred, c_mat = np.zeros(N, dtype=int), np.zeros((n_grp, n_grp), dtype=int)
        lda_pred_chance, c_mat_chance = np.zeros(N, dtype=int), np.zeros((n_grp, n_grp), dtype=int)
        p_mat = np.zeros((N, n_grp), dtype=float)

        ###########################################
        ####    LDA PREDICTION CALCULATIONS    ####
        ###########################################

        if is_OTO:
            # case is "one-trial out" setup
            NN, xi_rmv = n_t, np.arange(0, N, n_t)
        else:
            # case is "one-phase out" setup
            NN = len(i_grp)

        # fits the LDA model and calculates the prediction for each
        for i_trial in range(NN):
            # updates the progressbar (if provided)
            if w_prog is not None:
                w_str = '{0}, Group {1}/{2})'.format(w_str0, i_grp[i_trial] + 1, n_grp)
                w_prog.emit(w_str, 100. * (pw_0 + (i_trial / NN) / n_ex))

            # removes the trial/phase from the training dataset
            if is_OTO:
                # case is "one-trial out" setup
                is_keep[xi_rmv + i_trial] = False
            else:
                # case is "one-phase out" setup
                is_keep[i_trial] = False

            # fits the one-out-trial lda model
            try:
                lda.fit(spd_sf[is_keep, :], i_grp[is_keep])
            except:
                e_str = 'There was an error running the LDA analysis with the current solver parameters. ' \
                        'Either choose a different solver or alter the solver parameters before retrying'
                return None, False, e_str

            # calculates the model prediction from the remaining trial and increments the confusion matrix
            if is_OTO:
                for i in xi_rmv + i_trial:
                    lda_pred[i] = lda.predict(spd_sf[i, :].reshape(1, -1))
                    p_mat[i, :] = lda.predict_proba(spd_sf[i, :].reshape(1, -1))
                    c_mat[i_grp[i], lda_pred[i]] += 1

                    # re-adds the removed trials
                    is_keep[i] = True
            else:
                lda_pred[i_trial] = lda.predict(spd_sf[i_trial, :].reshape(1, -1))
                p_mat[i_trial, :] = lda.predict_proba(spd_sf[i_trial, :].reshape(1, -1))
                c_mat[i_grp[i_trial], lda_pred[i_trial]] += 1

                # re-adds the removed trial
                is_keep[i_trial] = True

        # returns the final values in a dictionary object
        return {
            'c_mat': c_mat, 'p_mat': p_mat, 'lda_pred': lda_pred,
        }, True, None

    def set_binary_mask(i_bin, n_c, n_bin):
        '''

        :param i_bin:
        :param n_c:
        :param n_bin:
        :return:
        '''

        # memory allocation
        B = np.zeros((1, n_c * n_bin), dtype=bool)

        # sets the masks values
        for i_c in range(n_c):
            B[0, i_c * n_bin + i_bin] = True

        # returns the array
        return B

    # initialises the RotationData class object (if not provided)
    if r_data is None:
        r_data = data.rotation

    # initialisations
    tt = r_filt['t_type']
    lda_para, xi_bin = calc_para['lda_para'], r_data.spd_xi
    n_c, n_ex, n_bin = len(tt), len(spd_sf), np.size(xi_bin, axis=0)

    #########################
    ####    LDA SETUP    ####
    #########################

    # memory allocation for accuracy binary mask calculations
    BD = np.zeros((n_bin, n_bin * n_c), dtype=int)
    y_acc = np.nan * np.ones((n_ex, n_bin, n_c), dtype=float)

    # sets up the group indices
    i_grp0 = np.array(cf.flat_list([list(x * np.ones(n_trial, dtype=int)) for x in range(n_bin)]))
    i_grp = np.array(cf.flat_list([list(i_c * n_bin + i_grp0) for i_c in range(n_c)]))

    # sets up the binary masks for the group/direction types
    for i_c in range(n_c):
        for i_bin in range(n_bin):
            BD[i_bin, (i_c * n_bin) + i_bin] = True

    # sets the LDA solver type
    lda_0 = setup_lda_solver(lda_para)

    ################################
    ####    LDA CALCULATIONS    ####
    ################################

    # loops through each of the experiments performing the lda calculations
    for i_ex in range(n_ex):
        # sets the progress strings (if progress bar handle is provided)
        if w_prog is not None:
            w_str0, pw_0 = 'Kinematic LDA (Expt {0}/{1}'.format(i_ex + 1, n_ex), i_ex / n_ex

        # combines the spiking frequencies into a single array (for the current experiment)
        spd_sf_ex = np.hstack([np.hstack([sf[:, i_bin, :].T for i_bin in range(n_bin)]) for sf in spd_sf[i_ex]]).T

        # normalises the spike count array (if required)
        if lda_para['is_norm']:
            N = np.size(spd_sf_ex, axis=1)
            spd_sf_mn = repmat(np.mean(spd_sf_ex, axis=1).reshape(-1, 1), 1, N)
            spd_sf_sd = repmat(np.std(spd_sf_ex, axis=1).reshape(-1, 1), 1, N)
            spd_sf_ex = np.divide(spd_sf_ex - spd_sf_mn, spd_sf_sd)

            # any cells where the std. deviation is zero are set to zero (to remove any NaNs)
            spd_sf_ex[spd_sf_sd == 0] = 0

        # runs the LDA predictions on the current spiking frequency array
        lda, ok, e_str = run_lda_predictions(w_prog, w_str0, pw_0, n_ex, lda_0, spd_sf_ex, i_grp)
        if not ok:
            # if there was an error, then exit with a false flag
            if w_prog is not None:
                w_prog.emit(e_str, 'LDA Analysis Error')
            return False

        # calculates the grouping accuracy values
        c_mat = lda['c_mat'] / n_trial

        # calculates the direction accuracy values (over each condition)
        for i_c in range(n_c):
            c_mat_sub = c_mat[(i_c * n_bin):((i_c + 1) * n_bin), :]
            for i_bin in range(n_bin):
                BD_sub = set_binary_mask(i_bin, n_c, n_bin)
                y_acc[i_ex, i_bin, i_c] = np.sum(np.multiply(c_mat_sub[i_bin, :], BD_sub))

    #######################################
    ####    HOUSE-KEEPING EXERCISES    ####
    #######################################

    # sets the lda values
    d_data.lda = 1
    d_data.y_acc = y_acc
    d_data.lda_trial_type = get_glob_para('lda_trial_type')

    # sets the rotation values
    d_data.spd_xi = r_data.spd_xi

    # sets a copy of the lda parameters and updates the comparison conditions
    _lda_para = dcopy(lda_para)
    _lda_para['comp_cond'] = r_data.r_obj_kine.rot_filt['t_type']

    # sets the solver parameters
    set_lda_para(d_data, _lda_para, r_filt, n_trial)

    # sets the phase duration/offset parameters
    d_data.vel_bin = calc_para['vel_bin']
    d_data.n_sample = calc_para['n_sample']
    d_data.equal_time = calc_para['equal_time']

    # returns a true value indicating success
    return True


def run_kinematic_lda(data, spd_sf, calc_para, r_filt, n_trial, w_prog=None, d_data=None, r_data=None):
    '''

    :param data:
    :param spd_sf:
    :param calc_para:
    :param r_filt:
    :param i_cell:
    :param n_trial:
    :param w_prog:
    :param d_data:
    :param i_shuff:
    :return:
    '''

    # initialises the RotationData class object (if not provided)
    if r_data is None:
        r_data = data.rotation

    # initialisations
    tt = r_filt['t_type']
    lda_para = calc_para['lda_para']
    n_c, n_ex = len(tt), len(spd_sf)

    ################################
    ####    LDA CALCULATIONS    ####
    ################################

    # memory allocation and other initialisations
    i_bin_spd = r_data.i_bin_spd
    ind_t, xi_bin = np.array(range(n_trial)), r_data.spd_xi
    n_bin = np.size(xi_bin, axis=0)

    # memory allocation for accuracy binary mask calculations
    BD = np.zeros((2, 2 * n_c), dtype=bool)
    y_acc = np.nan * np.ones((n_ex, n_bin, n_c), dtype=float)

    # sets up the binary masks for the group/direction types
    for i_c in range(n_c):
        BD[0, 2 * i_c], BD[1, 2 * i_c + 1] = True, True

    # loops through each of the experiments performing the lda calculations
    for i_ex in range(n_ex):
        # sets the progress strings (if progress bar handle is provided)
        if w_prog is not None:
            w_str0 = 'Kinematic LDA (Expt {0}/{1}, Bin'.format(i_ex + 1, n_ex)

        # sets the experiment name and runs the LDA prediction calculations
        for i_bin in range(n_bin):
            # updates the progressbar (if provided)
            if w_prog is not None:
                w_prog.emit('{0} {1}/{2})'.format(w_str0, i_bin + 1, n_bin), 100. * (i_ex + (i_bin / n_bin)) / n_ex)

            if i_bin != i_bin_spd:
                # stacks the speed spiking frequency values into a single array
                spd_sf_bin = np.hstack([np.hstack((sf[:, i_bin_spd, :].T, sf[:, i_bin, :].T)) for sf in spd_sf[i_ex]])

                # runs the LDA predictions on the current spiking frequency array
                lda, ok, e_str = run_kinematic_lda_predictions(spd_sf_bin, lda_para, n_c, n_trial)
                if not ok:
                    # if there was an error, then exit with a false flag
                    if w_prog is not None:
                        w_prog.emit(e_str, 'LDA Analysis Error')
                    return False

                # calculates the grouping accuracy values
                c_mat = lda['c_mat'] / n_trial

                # calculates the direction accuracy values (over each condition)
                for i_c in range(n_c):
                    y_acc[i_ex, i_bin, i_c] = np.sum(np.multiply(BD, c_mat[(2 * i_c):(2 * (i_c + 1)), :])) / 2

    #######################################
    ####    HOUSE-KEEPING EXERCISES    ####
    #######################################

    if d_data is not None:
        # sets the lda values
        d_data.lda = 1
        d_data.y_acc = y_acc
        d_data.exp_name = [os.path.splitext(os.path.basename(x['expFile']))[0] for x in data.cluster]
        d_data.lda_trial_type = get_glob_para('lda_trial_type')

        # sets the rotation values
        d_data.spd_xi = r_data.spd_xi
        d_data.i_bin_spd = r_data.i_bin_spd

        # sets a copy of the lda parameters and updates the comparison conditions
        _lda_para = dcopy(lda_para)
        _lda_para['comp_cond'] = r_data.r_obj_kine.rot_filt['t_type']

        # sets the solver parameters
        set_lda_para(d_data, _lda_para, r_filt, n_trial)

        # sets the phase duration/offset parameters
        d_data.spd_xrng = calc_para['spd_x_rng']
        d_data.vel_bin = calc_para['vel_bin']
        d_data.n_sample = calc_para['n_sample']
        d_data.equal_time = calc_para['equal_time']

        # calculates the psychometric curves
        y_acc_mn, d_vel, vel_mx = np.mean(y_acc, axis=0), float(calc_para['vel_bin']), 80.
        xi_fit = np.arange(d_vel, vel_mx + 0.01, d_vel)
        d_data.y_acc_fit, _, _, _ = calc_psychometric_curves(y_acc_mn, xi_fit, n_c, i_bin_spd)

        # returns a true value indicating success
        return True
    else:
        # otherwise, return the accuracy/pyschometric fit values
        return [y_acc]


def run_vel_dir_lda(data, vel_sf, calc_para, r_filt, n_trial, w_prog, d_data, r_data=None):
    '''

    :param data:
    :param vel_sf:
    :param calc_para:
    :param _r_filt:
    :param n_trial:
    :param w_prog:
    :param d_data:
    :return:
    '''

    # initialises the RotationData class object (if not provided)
    if r_data is None:
        r_data = data.rotation

    # initialisations
    tt = r_filt['t_type']
    lda_para, vel_xi = calc_para['lda_para'], r_data.vel_xi
    n_c, n_ex, n_bin = len(tt), len(vel_sf), np.size(vel_xi, axis=0)
    n_bin_h = int(n_bin / 2)

    ################################
    ####    LDA CALCULATIONS    ####
    ################################

    # memory allocation for accuracy binary mask calculations
    BD = np.zeros((2, 2 * n_c), dtype=bool)
    y_acc = np.nan * np.ones((n_ex, n_bin_h, n_c, 2), dtype=float)

    # sets up the binary masks for the group/direction types
    for i_c in range(n_c):
        BD[0, 2 * i_c], BD[1, 2 * i_c + 1] = True, True

    # loops through each of the experiments performing the lda calculations
    for i_ex in range(n_ex):
        # sets the progress strings (if progress bar handle is provided)
        w_str0 = 'Discrimination LDA (Expt {0}/{1}, Bin'.format(i_ex + 1, n_ex)

        # sets the experiment name and runs the LDA prediction calculations
        for i_bin in range(n_bin_h):
            # updates the progressbar (if provided)
            if w_prog is not None:
                w_prog.emit('{0} {1}/{2})'.format(w_str0, i_bin + 1, n_bin_h), 100. * (i_ex + (i_bin / n_bin_h)) / n_ex)

            # sets the positive/negative velocity indices
            i_neg, i_pos = i_bin, n_bin - (i_bin + 1)

            for i_dir in range(2):
                # stacks the speed spiking frequency values into a single array
                j, k, vel_sf_bin0 = 1 - i_dir, i_dir, [[] for _ in range(n_c)]
                for i_c in range(n_c):
                    vel_sf_bin0[i_c] = np.hstack((vel_sf[i_ex][j][i_c][:, i_pos, :].T,
                                                  vel_sf[i_ex][k][i_c][:, i_neg, :].T))

                # runs the LDA predictions on the current spiking frequency array
                vel_sf_bin = np.hstack(vel_sf_bin0)
                lda, ok, e_str = run_kinematic_lda_predictions(vel_sf_bin, lda_para, n_c, n_trial)
                if not ok:
                    # if there was an error, then exit with a false flag
                    if w_prog is not None:
                        w_prog.emit(e_str, 'LDA Analysis Error')
                    return False

                # calculates the grouping accuracy values
                c_mat = lda['c_mat'] / n_trial

                # calculates the direction accuracy values (over each condition)
                for i_c in range(n_c):
                    y_acc[i_ex, i_pos - n_bin_h, i_c, i_dir] = \
                                                np.sum(np.multiply(BD, c_mat[(2 * i_c):(2 * (i_c + 1)), :])) / 2

    #######################################
    ####    HOUSE-KEEPING EXERCISES    ####
    #######################################

    # sets the lda values
    d_data.lda = 1
    d_data.y_acc = np.mean(y_acc, axis=3)
    d_data.lda_trial_type = get_glob_para('lda_trial_type')

    # sets the rotation values
    d_data.spd_xi = r_data.spd_xi

    # sets a copy of the lda parameters and updates the comparison conditions
    _lda_para = dcopy(lda_para)
    _lda_para['comp_cond'] = r_data.r_obj_kine.rot_filt['t_type']

    # sets the solver parameters
    set_lda_para(d_data, _lda_para, r_filt, n_trial)

    # sets the phase duration/offset parameters
    d_data.vel_bin = calc_para['vel_bin']
    d_data.n_sample = calc_para['n_sample']
    d_data.equal_time = calc_para['equal_time']

    # returns a true value indicating success
    return True


def run_kinematic_lda_predictions(sf, lda_para, n_c, n_t):
    '''

    :param sf:
    :param lda_para:
    :param n_c:
    :param n_t:
    :return:
    '''

    # array dimensioning and memory allocation
    i_grp, n_grp = [], 2 * n_c
    N, n_cell = 2 * n_t * n_c, np.size(sf, axis=0)
    is_OTO = get_glob_para('lda_trial_type') == 'One-Trial Out'
    is_keep = np.ones(N, dtype=bool)

    ####################################
    ####    LDA DATA ARRAY SETUP    ####
    ####################################

    # sets up the LDA data/group index arrays across each condition
    for i_filt in range(n_c):
        # sets the grouping indices
        ind_g = [2 * i_filt, 2 * i_filt + 1]
        i_grp.append(np.hstack((ind_g[0] * np.ones(n_t), ind_g[1] * np.ones(n_t))).astype(int))

    # combines the spike counts/group indices into the final combined arrays
    i_grp = np.hstack(i_grp)

    # normalises the spike count array (if required)
    sf_calc = dcopy(sf)
    if lda_para['is_norm']:
        sf_mn = repmat(np.mean(sf_calc, axis=1).reshape(-1, 1), 1, N)
        sf_sd = repmat(np.std(sf_calc, axis=1).reshape(-1, 1), 1, N)
        sf_calc = np.divide(sf_calc - sf_mn, sf_sd)

        # any cells where the std. deviation is zero are set to zero (to remove any NaNs)
        sf_calc[sf_sd == 0] = 0

    # transposes the array
    sf_calc = sf_calc.T

    ###########################################
    ####    LDA PREDICTION CALCULATIONS    ####
    ###########################################

    # creates the lda solver object
    lda = setup_lda_solver(lda_para)

    # memory allocation
    lda_pred, c_mat = np.zeros(N, dtype=int), np.zeros((n_grp, n_grp), dtype=int)
    lda_pred_chance, c_mat_chance = np.zeros(N, dtype=int), np.zeros((n_grp, n_grp), dtype=int)
    p_mat = np.zeros((N, n_grp), dtype=float)

    if is_OTO:
        # case is "one-trial out" setup
        NN, xi_rmv = n_t, np.arange(0, N, n_t)
    else:
        # case is "one-phase out" setup
        NN = len(i_grp)

    # fits the LDA model and calculates the prediction for each
    for i_trial in range(NN):
        # fits the one-out-trial lda model
        if is_OTO:
            is_keep[xi_rmv + i_trial] = False
        else:
            is_keep[i_trial] = False

        try:
            lda.fit(sf_calc[is_keep, :], i_grp[is_keep])
        except:
            e_str = 'There was an error running the LDA analysis with the current solver parameters. ' \
                    'Either choose a different solver or alter the solver parameters before retrying'
            return None, False, e_str

        # calculates the model prediction from the remaining trial and increments the confusion matrix
        if is_OTO:
            for i in xi_rmv + i_trial:
                lda_pred[i] = lda.predict(sf_calc[i, :].reshape(1, -1))
                p_mat[i, :] = lda.predict_proba(sf_calc[i, :].reshape(1, -1))
                c_mat[i_grp[i], lda_pred[i]] += 1

                # re-adds the removed trial
                is_keep[i] = True

        else:
            lda_pred[i_trial] = lda.predict(sf_calc[i_trial, :].reshape(1, -1))
            p_mat[i_trial, :] = lda.predict_proba(sf_calc[i_trial, :].reshape(1, -1))
            c_mat[i_grp[i_trial], lda_pred[i_trial]] += 1

            # re-adds the removed trial
            is_keep[i_trial] = True

        # # fits the one-out-trial shuffled lda model
        # ind_chance = np.random.permutation(len(i_grp) - 1)
        # lda.fit(sf_calc[ii, :], i_grp[ii][ind_chance])

        # # calculates the chance model prediction from the remaining trial and increments the confusion matrix
        # lda_pred_chance[i_pred] = lda.predict(sf_calc[i_pred, :].reshape(1, -1))
        # c_mat_chance[i_grp[i_pred], lda_pred_chance[i_pred]] += 1

    # calculates the LDA transform values (uses svd solver to accomplish this)
    if lda_para['solver_type'] != 'lsqr':
        # memory allocation
        lda_X, lda_X0 = np.empty(n_grp, dtype=object), lda.fit(sf_calc, i_grp)

        # calculates the variance explained
        if len(lda_X0.explained_variance_ratio_) == 2:
            lda_var_exp = np.round(100 * lda_X0.explained_variance_ratio_.sum(), 2)
        else:
            lda_var_sum = lda_X0.explained_variance_ratio_.sum()
            lda_var_exp = np.round(100 * np.sum(lda_X0.explained_variance_ratio_[:2] / lda_var_sum), 2)

        # separates the transform values into the individual groups
        lda_X0T = lda_X0.transform(sf_calc)
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
    }, True, None

############################################
####    PSYCHOMETRIC CURVE FUNCTIONS    ####
############################################


def calc_all_psychometric_curves(d_data, d_vel):
    '''

    :param d_data:
    :param d_vel:
    :return:
    '''

    # parameters and array indexing
    vel_mx = 80.
    nC, n_tt, n_xi = len(d_data.n_cell), len(d_data.ttype), len(d_data.spd_xi)

    # memory allocation
    y_acc_fit, A = np.zeros((n_xi, nC, n_tt)), np.empty(n_tt, dtype=object)
    p_acc, p_acc_lo, p_acc_hi = dcopy(A), dcopy(A), dcopy(A)

    # other initialisations
    y_acc = dcopy(d_data.y_acc)
    i_bin_spd = d_data.i_bin_spd
    xi_fit = np.arange(d_vel, vel_mx + 0.01, d_vel)

    # calculates the psychometric fits for each condition trial type
    for i_tt in range(n_tt):
        # sets the mean accuracy values (across all cell counts)
        y_acc_mn = np.hstack((np.mean(100. * y_acc[i_tt][:, :, :-1], axis=0),
                              100. * y_acc[i_tt][0, :, -1].reshape(-1, 1)))

        # calculates/sets the psychometric fit values
        y_acc_fit[:, :, i_tt], p_acc[i_tt], p_acc_lo[i_tt], p_acc_hi[i_tt] = \
                            calc_psychometric_curves(y_acc_mn, xi_fit, nC, i_bin_spd)

    # updates the class fields
    d_data.y_acc_fit, d_data.p_acc, d_data.p_acc_lo, d_data.p_acc_hi = y_acc_fit, p_acc, p_acc_lo, p_acc_hi


def calc_psychometric_curves(y_acc_mn, xi, n_cond, i_bin_spd):
    '''

    :param y_acc_mn:
    :param d_vel:
    :param n_cond:
    :param i_bin_spd:
    :return:
    '''

    from lmfit import Model

    def fit_func(x, y0, yA, k, xH):
        '''

        :param x:
        :param p0:
        :param p1:
        :param p2:
        :param p3:
        :return:
        '''

        # checks if the sum of the scale/steady state value is infeasible
        if y0 + yA > 100:
            # if infeasible, then return a high value
            return 1e6 * np.ones(len(x))
        else:
            # calculates the function values for the current parameters
            F = y0 + (yA / (1. + np.exp(-k * (x - xH))))
            if F[0] < 0 or F[-1] > 100:
                # if the function values are infeasible, then return a high value
                return 1e6 * np.ones(len(x))
            else:
                # otherwise, return the function values
                return F

    def init_fit_para(x, y, n_para):
        '''

        :param y:
        :return:
        '''

        # memory allocation
        x0 = np.zeros(n_para)

        # calculates the estimated initial/steady state values
        y_min, y_max = np.min(y), np.max(y)
        x0[0], x0[1] = y_min, y_max - y_min

        # calculates the estimated half activation point
        x0[3] = xi[np.argmin(np.abs(((y - y_min) / (y_max - y_min)) - 0.5))]

        # calculates the inverse exponent values
        z = np.divide(-np.log(np.divide(x0[1], y - x0[0]) - 1), x - x0[3])

        # calculates the estimated exponential rate value
        is_feas = np.logical_and(~np.isinf(z), z > 0)
        if not np.any(is_feas):
            # if there are no feasible values, then set a default value
            x0[2] = 0.05
        else:
            # otherwise, calculate the mean of the feasible values
            x0[2] = np.mean(z[is_feas])

        # returns the initial parameter estimate
        return x0

    # student-t value for the dof and confidence level
    n_para, alpha = 4, 0.05
    tval = t.ppf(1. - alpha / 2., max(0, len(xi) - n_para))
    bounds = ((0., 0., 0., 0.), (100., 100., 0.5, 200.))

    # memory allocation
    y_acc_fit, A = np.empty(n_cond, dtype=object), np.zeros((n_cond, n_para))
    p_acc, p_acc_lo, p_acc_hi = dcopy(A), dcopy(A), dcopy(A)

    # sets the indices of the values to be fit
    ii = np.ones(len(xi), dtype=bool)
    ii[i_bin_spd] = False

    # gmodel = Model(fit_func)

    # loops through each of the conditions calculating the psychometric fits
    for i_c in range(n_cond):
        # initialisations
        maxfev = 10000

        # sets up the initial parameters
        p0 = init_fit_para(xi[ii], y_acc_mn[ii, i_c], n_para)
        # params = gmodel.make_params(y0=p0[0], yA=p0[1], k=p0[2], xH=p0[3])

        # keep attempting to find the psychometric fit until a valid solution is found
        while 1:
            try:
                # runs the fit function
                p_acc[i_c, :], pcov = curve_fit(fit_func, xi[ii], y_acc_mn[ii, i_c], p0=p0, maxfev=maxfev,
                                                bounds=bounds)
                # result = gmodel.fit(y_acc_mn[ii, i_c], params, x=xi[ii])

                # if successful, calculation the fit values and exits the inner loop
                y_acc_fit[i_c] = fit_func(xi, *p_acc[i_c, :])
                for i_p, pp, pc in zip(range(n_para), p_acc[i_c, :], np.diag(pcov)):
                    sigma = pc ** 0.5
                    if i_p in [0, 1]:
                        p_acc_lo[i_c, i_p], p_acc_hi[i_c, i_p] = max(0, pp - sigma * tval), min(100., pp + sigma * tval)
                    else:
                        p_acc_lo[i_c, i_p], p_acc_hi[i_c, i_p] = max(0, pp - sigma * tval), pp + sigma * tval

                break
            except:
                # if there was an error, then increment the max function evaluation parameter
                maxfev *= 2

    # returns the fit values
    return np.vstack(y_acc_fit).T, p_acc, p_acc_lo, p_acc_hi

####################################################
####    LDA SOLVER PARAMETER/SETUP FUNCTIONS    ####
####################################################


def setup_lda_solver(lda_para):
    '''

    :param lda_para:
    :return:
    '''

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

    # returns the lda solver object
    return lda


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
        'y_acc_min': 0,
        'y_auc_max': 100,
        'y_auc_min': 50,
    }


def init_def_class_para(d_data_0, d_data_f=None, d_data_def=None):
    '''

    :param d_data_0:
    :param d_data_f:
    :param d_data_def:
    :return:
    '''

    def set_def_class_para(d_data, p_str):
        '''

        :param d_data:
        :param p_str:
        :return:
        '''

        # initialisations
        def_para = {}

        # retrieves the values from the data class
        for ps in p_str:
            if hasattr(d_data, ps):
                p_val = d_data.__getattribute__(ps)
                if p_val is None:
                    continue
                elif p_val != -1:
                    def_para[ps] = p_val

        # returns the lda default parameter dictionary
        return def_para

    if d_data_f is None:
        # if the field name is not provided, set the default data class
        d_data = d_data_0
    elif hasattr(d_data_0, d_data_f):
        # if the class does have the sub-field, then retrieves the sub-class
        d_data = getattr(d_data_0, d_data_f)
    else:
        # otherwise, use the default sub-class and add it to the parent class
        d_data = d_data_def
        setattr(d_data_0, d_data_f, d_data_def)

    # set the default parameter object (based on type)
    if d_data_f == 'spikedf':
        # case is the spiking frequency dataframe
        def_para = set_def_class_para(d_data, ['rot_filt', 'bin_sz', 't_over'])

    # returns the default parameter object
    return def_para


def init_corr_para(r_data):
    '''

    :param r_data:
    :return:
    '''

    def set_corr_para(para_def, para_curr):
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
            return para_def if (para_curr <= 0) else para_curr

    # initialisations
    dv_0 = 5
    r_para = {}

    # sets the parameter fields
    para_flds = ['n_shuffle', 'vel_bin', 'n_smooth', 'is_smooth', 'n_sample', 'equal_time']

    #
    for pf in para_flds:
        # sets the parameter type
        p_type = 'Number'

        # sets the default value and class field name
        if pf == 'n_shuffle':
            def_val, fld_name = 100, 'n_shuffle_corr'

        elif pf == 'vel_bin':
            def_val, fld_name, p_type = str(dv_0), 'vel_bin_corr', 'String'

        elif pf == 'n_smooth':
            def_val, fld_name = 5, 'vel_sf_nsm'

        elif pf == 'is_smooth':
            def_val, fld_name, p_type = False, 'vel_sf_nsm', 'Boolean'

        elif pf == 'n_sample':
            def_val, fld_name = 100, 'n_rs'

        elif pf == 'equal_time':
            def_val, fld_name = False, 'vel_sf_eqlt'

        # sets the roc parameter values into the parameter dictionary
        if hasattr(r_data, fld_name):
            if p_type == 'Number':
                r_para[pf] = int(set_corr_para(def_val, getattr(r_data, fld_name)))
            elif p_type == 'String':
                r_para[pf] = set_corr_para(def_val, str(int(getattr(r_data, fld_name))))
            elif p_type == 'Boolean':
                r_para[pf] = set_corr_para(def_val, getattr(r_data, fld_name)) > 0
        else:
            r_para[pf] = def_val

    # returns the parameter dictionary
    return r_para

def init_clust_para(c_comp, free_exp):
    '''

    :param c_comp:
    :param free_exp:
    :return:
    '''

    def set_clust_para(para_def, para_curr):
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

    # memory allocation and parameters
    c_para = {}

    # sets the parameter fields to be retrieved (based on type)
    para_flds = ['d_max', 'r_max', 'sig_corr_min', 'isi_corr_min', 'sig_diff_max',
                 'sig_feat_min', 'w_sig_feat', 'w_sig_comp', 'w_isi']

    # sets the comparison data object belonging to the current experiment
    if 'No Fixed/Free Data Loaded' in free_exp:
        # case is there is no loaded freely moving data files
        c_data = object
    else:
        # case is there are freely moving data files
        i_expt = cf.det_likely_filename_match([x.free_name for x in c_comp.data], free_exp)
        c_data = c_comp.data[i_expt]

    #
    for pf in para_flds:
        # sets the new parameter field name (could be altered below...)
        def_val = float(get_glob_para(pf))
        if pf in ['d_max']:
            def_val = int(def_val)

        # sets the roc parameter values into the parameter dictionary
        if hasattr(c_data, pf):
            c_para[pf] = set_clust_para(def_val, getattr(c_data, pf))
        else:
            c_para[pf] = def_val

    # returns the parameter dictionary
    return c_para


def init_roc_para(r_data_0, f_type, r_data_f=None, r_data_def=None):
    '''

    :param r_data_0:
    :param r_data_f:
    :param r_data_def:
    :return:
    '''

    def set_roc_para(para_def, para_curr):
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

    if r_data_f is None:
        # case is there are no additional field information to be added
        r_data = r_data_0
    elif hasattr(r_data_0, r_data_f):
        # case is there is an additional field to be added. if the field exists, then overwrite it
        r_data = getattr(r_data_0, r_data_f)
    else:
        # otherwise, simply add the new field to the class
        r_data = r_data_def
        setattr(r_data_0, r_data_f, r_data_def)

    # memory allocation and parameters
    dv_0 = 5
    r_para, para_flds = {}, []

    # sets the parameter fields to be retrieved (based on type)
    if f_type == 'vel_roc_sig':
        para_flds = ['n_boot_kine_ci', 'kine_auc_stats_type', 'vel_bin', 'vel_x_rng', 'equal_time', 'n_sample']

    #
    for pf in para_flds:
        # sets the new parameter field name (could be altered below...)
        pf_nw = pf

        #############################
        ####    SPECIAL CASES    ####
        #############################

        if pf == 'vel_x_rng':
            pf_nw = 'spd_x_rng'
            if hasattr(r_data, 'i_bin_vel'):
                if isinstance(r_data.i_bin_vel, list):
                    vc_rng = get_kinematic_range_strings(float(r_para['vel_bin']), True)
                    r_para[pf_nw] = vc_rng[r_data.i_bin_vel[1]]
                else:
                    r_para[pf_nw] = '0 to {0}'.format(r_para['vel_bin'])
            else:
                r_para[pf_nw] = '0 to {0}'.format(r_para['vel_bin'])

        ############################
        ####    NORMAL CASES    ####
        ############################

        else:
            p_type = 'Number'

            if pf == 'n_boot_kine_ci':
                def_val, fld_name, pf_nw = 100, 'n_boot_kine_ci', 'n_boot'

            elif pf == 'kine_auc_stats_type':
                def_val, fld_name, pf_nw = 'Delong', 'kine_auc_stats_type', 'auc_stype'

            elif pf == 'equal_time':
                def_val, fld_name = False, 'is_equal_time'

            elif pf == 'n_sample':
                def_val, fld_name = 100, 'n_rs'

            elif pf == 'vel_bin':
                def_val, fld_name, p_type = str(dv_0), 'vel_bin', 'String'

            # sets the roc parameter values into the parameter dictionary
            if hasattr(r_data, fld_name):
                if p_type == 'Number':
                    r_para[pf_nw] = set_roc_para(def_val, getattr(r_data, fld_name))
                elif p_type == 'String':
                    r_para[pf_nw] = set_roc_para(def_val, str(int(getattr(r_data, fld_name))))
            else:
                r_para[pf_nw] = def_val

    # returns the parameter dictionary
    return r_para


def init_lda_para(d_data_0, d_data_f=None, d_data_def=None):
    '''

    :param d_data_0:
    :param d_data_f:
    :param d_data_def:
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
            if hasattr(d_data, ps):
                p_val = d_data.__getattribute__(ps)
                if p_val is None:
                    continue
                elif p_val != -1:
                    def_para[ps] = p_val

        # returns the lda default parameter dictionary
        return def_para

    if d_data_f is None:
        # case is there are no additional field information to be added
        d_data = d_data_0
    elif hasattr(d_data_0, d_data_f):
        # case is there is an additional field to be added. if the field exists, then overwrite it
        d_data = getattr(d_data_0, d_data_f)
    else:
        # otherwise, simply add the new field to the class
        d_data = d_data_def
        setattr(d_data_0, d_data_f, d_data_def)

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

        if hasattr(d_data, 'yaccmn'):
            lda_para['y_acc_min'] = set_lda_para(lda_para['y_acc_min'], d_data.yaccmn)
        else:
            lda_para['y_acc_min'] = 0

        if hasattr(d_data, 'yaucmx'):
            lda_para['y_auc_max'] = set_lda_para(lda_para['y_auc_max'], d_data.yaucmx)
        else:
            lda_para['y_auc_max'] = 0

        if hasattr(d_data, 'yaucmn'):
            lda_para['y_auc_min'] = set_lda_para(lda_para['y_auc_min'], d_data.yaucmn)
        else:
            lda_para['y_auc_min'] = 0

    # sets the default parameters based on the type
    if d_data.type in ['Direction', 'Individual', 'LDAWeight']:
        # case is the default rotational LDA
        def_para = set_def_lda_para(d_data, ['tofs', 'tphase', 'usefull'])

    elif d_data.type in ['Temporal']:
        # case is the temporal LDA
        def_para = set_def_lda_para(d_data, ['dt_phs', 'dt_ofs', 'phs_const'])

    elif d_data.type in ['TrialShuffle']:
        # case is the shuffled trial LDA
        def_para = set_def_lda_para(d_data, ['tofs', 'tphase', 'usefull', 'nshuffle'])

    elif d_data.type in ['Partial']:
        # case is the partial LDA
        def_para = set_def_lda_para(d_data, ['tofs', 'tphase', 'usefull', 'nshuffle', 'poolexpt'])

    # sets the default parameters based on the type
    if d_data.type in ['IndivFilt']:
        # case is the default rotational LDA
        def_para = set_def_lda_para(d_data, ['tofs', 'tphase', 'usefull', 'yaccmn', 'yaccmx'])

    elif d_data.type in ['SpdAcc', 'SpdCompDir']:
        # case is the velocity comparison LDA
        def_para = set_def_lda_para(d_data, ['vel_bin', 'n_sample', 'equal_time'])

    elif d_data.type in ['SpdComp']:
        # case is the velocity comparison LDA
        def_para = set_def_lda_para(d_data, ['spd_xrng', 'vel_bin', 'n_sample', 'equal_time'])

    elif d_data.type in ['SpdCompPool']:
        # case is the velocity comparison LDA
        def_para = set_def_lda_para(d_data, ['spd_xrng', 'vel_bin', 'n_sample', 'equal_time', 'nshuffle', 'poolexpt'])

    # returns the lda solver/default parameter dictionaries
    return lda_para, def_para


def get_eyetrack_para(et_data):
    '''

    :param et_data:
    :return:
    '''

    # memory allocation and parameters
    c_para = {}

    # sets the parameter fields to be retrieved (based on type)
    para_flds = ['use_med_filt', 'rmv_baseline', 'dp_max', 'n_sd', 'n_pre', 'n_post']

    # retrieves the parameter values
    for pf in para_flds:
        c_para[pf] = getattr(et_data, pf)

    # returns the parameter dictionary
    return c_para

def set_lda_para(d_data, lda_para, r_filt, n_trial_max, ignore_list=None):
    '''

    :param d_data:
    :param lda_para:
    :param r_filt:
    :param n_trial_max:
    :return:
    '''

    #
    if ignore_list is None:
        ignore_list = []

    # sets the parameter to class conversion strings
    conv_str = {
        'n_cell_min': 'cellmin',
        'n_trial_min': 'trialmin',
        'solver_type': 'solver',
        'use_shrinkage': 'shrinkage',
        'is_norm': 'norm',
        'cell_types': 'ctype',
        'y_acc_max': 'yaccmx',
        'y_acc_min': 'yaccmn',
        'y_auc_max': 'yaucmx',
        'y_auc_min': 'yaucmn',
        'comp_cond': 'ttype',
    }

    # sets the trial count and trial types
    _lda_para = dcopy(lda_para)
    d_data.ntrial = n_trial_max

    # sets the LDA solver parameters
    for ldp in lda_para:
        if ldp not in ignore_list:
            setattr(d_data, conv_str[ldp], _lda_para[ldp])

#####################################
####    GENERAL LDA FUNCTIONS    ####
#####################################

def reduce_cluster_data(data, i_expt):
    '''

    :param data:
    :param i_expt:
    :param cell_ok:
    :return:
    '''

    # creates a copy of the data and removes any
    data_tmp = EmptyAnalysisData()

    # copies the sub-fields
    data_tmp.rotation = scopy(data.rotation)
    data_tmp.classify = scopy(data.classify)
    data_tmp.comp = scopy(data.comp)

    # reduces down the number of
    if data.cluster is not None:
        data_tmp.cluster = [scopy(data.cluster[i_ex]) for i_ex in i_expt]

    if data._cluster is not None:
        data_tmp._cluster = [scopy(data._cluster[i_ex]) for i_ex in i_expt]

    # returns the reduced data class object
    return data_tmp


def setup_lda(data, calc_para, d_data=None, w_prog=None, return_reqd_arr=False, r_data=None, w_err=None):
    '''

    :param data:
    :param calc_para:
    :return:
    '''

    def det_valid_cells(data, ind, lda_para, r_data=None):
        '''

        :param cluster:
        :param lda_para:
        :return:
        '''

        # sets the exclusion field name to cluster field key
        f_key = {
            'region_name': 'chRegion',
            'record_layer': 'chLayer',
        }

        # initialises the RotationData class object (if not provided)
        if r_data is None:
            r_data = data.rotation

        # determines the cells that are in the valid regions (RSPg and RSPd)
        exc_filt = data.exc_rot_filt
        if cf.use_raw_clust(data):
            cluster = data._cluster[ind]
            ind_m = np.arange(len(cluster['clustID']))
        else:
            cluster = data.cluster[ind]
            if len(cluster['clustID']):
                ind_m = np.array([data._cluster[ind]['clustID'].index(x) for x in cluster['clustID']])
            else:
                return np.zeros(1, dtype=bool)

        # sets the boolean flags for the valid cells
        # is_valid = np.logical_or(cluster['chRegion'] == 'RSPg', cluster['chRegion'] == 'RSPd') #### COMMENT ME OUT FOR RETROSPLANIAL ONLY CELLS
        is_valid = np.ones(len(cluster['chRegion']),dtype=bool)
        if len(is_valid) == 0:
            return np.zeros(1, dtype=bool)

        # removes any values that correspond to the fields in the exclusion filter
        for ex_key in ['region_name', 'record_layer']:
            if len(exc_filt[ex_key]):
                for f_exc in exc_filt[ex_key]:
                    is_valid[cluster[f_key[ex_key]] == f_exc] = False

        # if the cell types have been set, then remove the cells that are not the selected type
        if lda_para['cell_types'] == 'Narrow Spike Cells':
            # case is narrow spikes have been selected
            is_valid[np.logical_not(data.classify.grp_str[ind][ind_m] == 'Nar')] = False
        elif lda_para['cell_types'] == 'Wide Spike Cells':
            # case is wide spikes have been selected
            is_valid[np.logical_not(data.classify.grp_str[ind][ind_m] == 'Wid')] = False

        # determines if the individual LDA has been calculated
        d_data_i = data.discrim.indiv
        if d_data_i.lda is not None:
            # if so, determines the trial type corresponding to the black direction decoding type
            if ind in d_data_i.i_expt:
                # if the black decoding type is present, remove the cells which have a decoding accuracy above max
                ind_g = np.where(d_data_i.i_expt == ind)[0][0]
                ii = np.where(d_data_i.i_cell[ind_g])[0]
                is_valid[ii[np.any(100. * d_data_i.y_acc[ind_g][:, 1:] > lda_para['y_acc_max'],axis=1)]] = False
                is_valid[ii[np.any(100. * d_data_i.y_acc[ind_g][:, 1:] < lda_para['y_acc_min'],axis=1)]] = False

        #
        if hasattr(r_data, 'phase_roc_auc'):
            if (r_data.phase_roc_auc is not None):
                if len(r_data.phase_roc_auc):
                    # sets the indices
                    ind_0 = np.cumsum([0] + [x['nC'] for x in data._cluster])
                    ind_ex = [np.arange(ind_0[i], ind_0[i + 1]) for i in range(len(ind_0) - 1)]

                    # retrieves the black phase roc auc values (ensures the compliment is calculated)
                    auc_ex = r_data.phase_roc_auc[ind_ex[ind], :]
                    ii = auc_ex < 0.5
                    auc_ex[ii] = 1 - auc_ex[ii]

                    # determines which values meet the criteria
                    is_valid[np.any(100. * auc_ex > lda_para['y_auc_max'], axis=1)] = False
                    is_valid[np.any(100. * auc_ex < lda_para['y_auc_min'], axis=1)] = False

        # if the number of valid cells is less than the reqd count, then set all cells to being invalid
        if np.sum(is_valid) < lda_para['n_cell_min']:
            is_valid[:] = False

        # returns the valid index array
        return is_valid

    # initialisations
    lda_para, s_flag = calc_para['lda_para'], 2
    if len(lda_para['comp_cond']) < 1:
        # if no trial conditions are selected then output an error to screen
        if w_err is not None:
            e_str = 'At least 1 trial condition must be selected before running this function.'
            w_err.emit(e_str, 'Invalid LDA Analysis Parameters')

        # returns a false flag
        return None, None, None, None, 0

    # sets up the black phase data filter and returns the time spikes
    r_filt = cf.init_rotation_filter_data(False)
    r_filt['t_type'] = lda_para['comp_cond']
    r_obj0 = RotationFilteredData(data, r_filt, None, None, True, 'Whole Experiment', False)

    # retrieves the trial counts from each of the filter types/experiments
    n_trial = np.zeros((r_obj0.n_filt, r_obj0.n_expt), dtype=int)
    for i_filt in range(r_obj0.n_filt):
        # sets the trial counts for each experiment for the current filter option
        i_expt_uniq, ind = np.unique(r_obj0.i_expt[i_filt], return_index=True)
        n_trial[i_filt, i_expt_uniq] = r_obj0.n_trial[i_filt][ind]

    # removes any trials less than the minimum and from this determines the overall minimum trial count
    n_trial[n_trial < lda_para['n_trial_min']] = -1
    if np.all(n_trial < 0):
        # if there are no valid experiments, then output an error message to screen
        if w_prog is not None:
            e_str = 'There are no experiments which meet the LDA solver parameters/criteria.' \
                    'Either load more experiments or alter the solver/calculation parameters.'
            w_err.emit(e_str, 'No Valid Experiments Found')

        # returns a false flag
        return None, None, None, None, 0
    else:
        # otherwise, reduce down the experiments to those which are feasible
        n_trial_max = np.min(n_trial[n_trial > 0])

    # determines if the number of trials has changed (and if the lda calculation values have been set)
    if d_data is not None:
        if (n_trial_max == d_data.ntrial) and (d_data.lda is not None):
            # if there is no change and the values are set, then exit with a true flag
            s_flag = 1
            if not return_reqd_arr:
                return None, None, None, None, s_flag

    # determines the valid cells from each of the loaded experiments
    n_clust = len(data._cluster) if cf.use_raw_clust(data) else len(data.cluster)
    i_cell = np.array([det_valid_cells(data, ic, lda_para) for ic in range(n_clust)])

    # determines if there are any valid loaded experiments
    i_expt = np.where([(np.any(x) and np.min(n_trial[:, i_ex]) >= lda_para['n_trial_min'])
                        for i_ex, x in enumerate(i_cell)])[0]
    if len(i_expt) == 0:
        # if there are no valid experiments, then output an error message to screen
        if w_prog is not None:
            e_str = 'There are no experiments which meet the LDA solver parameters/criteria.' \
                    'Either load more experiments or alter the solver/calculation parameters.'
            w_err.emit(e_str, 'No Valid Experiments Found')

        # returns a false flag
        return None, None, None, None, 0

    # returns the import values for the LDA calculations
    return r_filt, i_expt, i_cell[i_expt], n_trial_max, s_flag


def get_pool_cell_counts(data, lda_para, type=0):
    '''

    :param data:
    :param lda_para:
    :return:
    '''

    if type == 0:
        n_cell = n_cell_pool0
    else:
        n_cell = n_cell_pool1

    # determine the valid cell counts for each experiment
    _, _, i_cell, _, _ = setup_lda(data, {'lda_para': lda_para}, None)

    # returns the final cell count
    if i_cell is None:
        return []
    else:
        n_cell_tot = np.sum([sum(x) for x in i_cell])
        return [x for x in n_cell if x <= n_cell_tot] + ([n_cell_tot] if n_cell_tot not in n_cell else [])


def det_uniq_channel_layers(data, lda_para):
    '''

    :param data:
    :param wght_lda_para:
    :return:
    '''

    # retrieves the valid experiment/cell and reduces down the cluster data array
    _, i_expt, i_cell, _, _ = setup_lda(data, {'lda_para': lda_para})

    # returns the unique layer types
    if i_expt is None:
        return None
    else:
        return list(np.unique(cf.flat_list([data._cluster[i_ex]['chLayer'] for i_ex in i_expt])))


def get_glob_para(gp_type):
    '''

    :param trial_type:
    :return:
    '''

    # loads the default file
    with open(cf.default_dir_file, 'rb') as fp:
        def_data = _p.load(fp)

    # sets the trial type value
    return def_data['g_para'][gp_type]


def get_dir_para(dir_type):
    '''

    :param trial_type:
    :return:
    '''

    # loads the default file
    with open(cf.default_dir_file, 'rb') as fp:
        def_data = _p.load(fp)

    # sets the trial type value
    return def_data['dir'][dir_type]

########################################################################################################################
####                                      MISCELLANEOUS CALCULATION FUNCTIONS                                       ####
########################################################################################################################

def smooth_signal(y_sig, n_sm):
    '''

    :param y_sig:
    :param n_sm:
    :return:
    '''

    if n_sm > 0:
        # smooths the signal (if required)
        if np.ndim(y_sig) == 1:
            # case is there is only one signal
            return medfilt(y_sig, n_sm)
        else:
            # case is multiple signals
            y_sig_sm = np.zeros(np.shape(y_sig))

            # calculates the median filter for each signal
            for i_sig in range(np.size(y_sig, axis=0)):
                y_sig_sm[i_sig, :] = medfilt(y_sig[i_sig, :], n_sm)

            # returns the final array
            return y_sig_sm
    else:
        return y_sig


def get_rsp_reduced_clusters(data):
    '''

    :param data:
    :return:
    '''

    # memory allocation
    _data = EmptyAnalysisData()

    # reduces the data for each cluster
    _data._cluster = [[None] for _ in range(len(data._cluster))]
    for i_cc, cc in enumerate(data._cluster):
        # determines the cells that are in the valid regions (RSPg and RSPd)
        c = scopy(cc)
        i_cell = np.logical_or(c['chRegion'] == 'RSPg', c['chRegion'] == 'RSPd')

        # removes the non-valid cells from the depth, region and layer arrays
        c['clustID'] = list(np.array(c['clustID'])[i_cell])
        c['chDepth'] = c['chDepth'][i_cell]
        c['chRegion'] = c['chRegion'][i_cell]
        c['chLayer'] = c['chLayer'][i_cell]

        # removes the non-valid cell from the time spikes
        for tt in c['rotInfo']['trial_type']:
            c['rotInfo']['t_spike'][tt] = c['rotInfo']['t_spike'][tt][i_cell, :, :]

        # reduces the data for each cluster
        if _data.cluster is not None:
            for c in _data.cluster:
                # determines the cells that are in the valid regions (RSPg and RSPd)
                i_cell = np.logical_or(c['chRegion'] == 'RSPg', c['chRegion'] == 'RSPd')

                # removes the non-valid cells from the depth, region and layer arrays
                c['clustID'] = list(np.array(c['clustID'])[i_cell])
                c['chDepth'] = c['chDepth'][i_cell]
                c['chRegion'] = c['chRegion'][i_cell]
                c['chLayer'] = c['chLayer'][i_cell]

                # removes the non-valid cell from the time spikes
                for tt in c['rotInfo']['trial_type']:
                    c['rotInfo']['t_spike'][tt] = c['rotInfo']['t_spike'][tt][i_cell, :, :]

        # sets the cluster values
        _data._cluster[i_cc] = c

    # returns the valid cells array
    return _data


def get_channel_depths_tt(cluster, tt_type):
    '''

    :param cluster:
    :param comp_cond:
    :return:
    '''

    # memory allocation
    ch_depth, ch_region, ch_layer = {}, {}, {}

    #
    for tt in tt_type:
        # memory allocation
        depth_nw, region_nw, layer_nw = [], [], []

        for c in cluster:
            if tt in c['rotInfo']['trial_type']:
                # calculates the depth of the cell in the current experiment
                chMap, depth_max = c['expInfo']['channel_map'], c['expInfo']['probe_depth']
                depth_nw.append(depth_max - chMap[c['chDepth'].astype(int), 3])
                region_nw.append(list(c['chRegion']))
                layer_nw.append(list(c['chLayer']))

        #
        ch_depth[tt] = np.array(cf.flat_list(depth_nw))
        ch_region[tt] = np.array(cf.flat_list(region_nw))
        ch_layer[tt] = np.array(cf.flat_list(layer_nw))

    # returns the final array
    return ch_depth, ch_region, ch_layer

def get_plot_canvas_pos(plot_left, dY, fig_hght):
    '''

    :param fig_hght:
    :param plot_left:
    :param dY:
    :return:
    '''

    # calculates the figure width
    fig_wid = int(fig_hght * float(get_glob_para('w_ratio')))

    # returns the figure canvas position
    return QRect(plot_left, dY, fig_wid, fig_hght)


def det_missing_data_fields(fld_vals, f_name, chk_flds):
    '''

    :param fld_vals:
    :param is_missing:
    :return:
    '''

    for i_cf, cfld in enumerate(chk_flds):
        if cfld == 'probe_depth':
            # retrieves all the configuration file names
            f_cfg, cfg_dir = [], get_dir_para('configDir')
            for f in os.listdir(cfg_dir):
                if f.endswith('.cfig'):
                    f_cfg.append(f.replace('.cfig', ''))

            for i_c in np.where(fld_vals[:, i_cf] == None)[0]:
                f_match, f_score = cf.det_closest_file_match(f_cfg, f_name[i_c])
                if f_score > 95:
                    # retrieves the depth marker values from the config file (if the match is good enough)
                    cfig_new = os.path.join(cfg_dir, '{0}.cfig'.format(f_match))
                    f_str = cf.get_cfig_line(cfig_new, 'depthHi')

                    f_val = eval(f_str[1:-1])
                    if isinstance(f_val, tuple):
                        fld_vals[i_c, i_cf] = int(f_val[0])
                    elif isinstance(f_val, str):
                        fld_vals[i_c, i_cf] = int(f_val)

    # returns the final values array
    return fld_vals


def det_matching_ttype_expt(r_obj, cluster):
    '''

    :param r_obj:
    :param cluster:
    :return:
    '''

    # returns the unique trial types from the rotation object
    t_type = set(np.unique(cf.flat_list([x['t_type'] for x in r_obj.rot_filt_tot])))

    # returns the indices of the experiments which don't have the correct trial type counts
    return np.where([len(t_type.intersection(set(c['rotInfo']['trial_type']))) != len(t_type) for c in cluster])[0]

# def normalise_spike_freq(spd_sf_calc, N, i_ax=1):
#     '''
#
#     :param spd_sf_calc:
#     :param N:
#     :return:
#     '''
#
#     # normalises the spiking frequencies
#     if i_ax == 0:
#         spd_sf_mn = repmat(np.mean(spd_sf_calc, axis=0).reshape(-1, 1), N, 1)
#         spd_sf_sd = repmat(np.std(spd_sf_calc, axis=0).reshape(-1, 1), N, 1)
#     else:
#         spd_sf_mn = repmat(np.mean(spd_sf_calc, axis=1).reshape(-1, 1), 1, N)
#         spd_sf_sd = repmat(np.std(spd_sf_calc, axis=1).reshape(-1, 1), 1, N)
#
#     # any cells where the std. deviation is zero are set to zero (to remove any NaNs)
#     spd_sf_calc = np.divide(spd_sf_calc - spd_sf_mn, spd_sf_sd)
#     spd_sf_calc[spd_sf_sd == 0] = 0
#
#     # returns the normalised array
#     return spd_sf_calc

def calc_expt_roc_sig(r_obj, roc_sig, c_ofs, i_filt, calc_mean=False, i_expt=None):
    '''

    :param roc_sig:
    :param i_expt_robj:
    :param i_expt:
    :return:
    '''

    # sets the
    if i_expt is None:
        i_expt = np.arange(len(c_ofs))

    # memory allocation
    n_expt, cl_ind = len(i_expt), r_obj.clust_ind[i_filt]
    roc_sig_expt = np.empty(n_expt, dtype=object)

    # retrieves the roc significance values for each of the rotation filter types
    for i_ex in range(n_expt):
        # retrieves the roc significance values belonging to the current experiment
        i_cl = cl_ind[i_expt[i_ex]] + c_ofs[i_expt[i_ex]]
        roc_sig_expt[i_ex] = roc_sig[i_cl, :]

        # calculates the mean (if required)
        if calc_mean:
            roc_sig_expt[i_ex] = 100. * np.mean(roc_sig_expt[i_ex], axis=0)

    # returns the final array
    return roc_sig_expt


def get_all_match_cond_cells(data, t_type):
    '''

    :param r_obj:
    :return:
    '''

    def det_cell_index_interset(ind, tt, i_cell_nw):
        '''

        :param ind:
        :param i_cell_nw:
        :return:
        '''

        if tt in ind:
            # if the field does exist, then determine the intersection of current/new index arrays
            return list(set(ind[tt]).intersection(i_cell_nw))
        else:
            # if the field doesn't exist, then use the whole index array
            return i_cell_nw

    # initalises the filter data
    rot_filt = cf.init_rotation_filter_data(False)
    rot_filt['t_type'] = t_type

    # retrieves the rotation filter object
    r_obj = RotationFilteredData(data, rot_filt, None, None, True, 'Whole Experiment', False)

    #
    if len(t_type) == 1:
        # if only only trial condition, then use all indices
        i_cell_i, _ = cf.det_cell_match_indices(r_obj, [0, 0])
        ind_all = {t_type[0]: i_cell_i}

    else:
        # initialisations
        n_filt, ind_all = r_obj.n_filt, {}

        # determines the matching cell indices between all trial conditions
        for i in range(n_filt - 1):
            for j in range(i + 1, n_filt):
                # determines the matching indices between the 2 conditions
                i_cell_i, i_cell_j = cf.det_cell_match_indices(r_obj, [i, j])
                tt_i, tt_j = r_obj.rot_filt_tot[i]['t_type'][0], r_obj.rot_filt_tot[j]['t_type'][0]

                # updates the intersection of the current indices with the new ones
                ind_all[tt_i] = det_cell_index_interset(ind_all, tt_i, i_cell_i)
                ind_all[tt_j] = det_cell_index_interset(ind_all, tt_j, i_cell_j)

    #
    return ind_all


def get_kinematic_range_strings(dv, is_vel, v_rng=80):
    '''

    :param dv:
    :param is_vel:
    :param v_rng:
    :return:
    '''

    # returns the range strings based on the kinematic type
    if is_vel:
        # case is velocity
        return ['{0} to {1}'.format(int(i * dv - v_rng), int((i + 1) * dv - v_rng)) for i in range(int(2 * v_rng / dv))]
    else:
        # case is speed
        return ['{0} to {1}'.format(int(i * dv), int((i + 1) * dv)) for i in range(int(v_rng / dv))]


def get_cond_filt_data(data, r_obj):
    '''

    :param r_obj:
    :param plot_exp_name:
    :return:
    '''

    # memory allocation
    A = np.empty(r_obj.n_filt, dtype=object)
    i_cell_b, r_obj_tt = dcopy(A), dcopy(A)

    # determine the matching cell indices between the current and black filter
    for i_filt in range(r_obj.n_filt):
        # sets up a base filter with each of the filter types
        r_filt_base = cf.init_rotation_filter_data(False)
        r_filt_base['t_type'] = r_obj.rot_filt_tot[i_filt]['t_type']
        r_obj_tt[i_filt] = RotationFilteredData(data, r_filt_base, None, None, True, 'Whole Experiment', False)

        # finds the corresponding cell types between the overall and user-specified filters
        i_cell_b[i_filt], _ = cf.det_cell_match_indices(r_obj_tt[i_filt], [0, i_filt], r_obj)

    # returns the array
    return i_cell_b, r_obj_tt


def get_common_filtered_cell_indices(data, r_obj, tt_filt, det_intersect, ind_cond=None):
    '''

    :param data:
    :param r_obj:
    :param t_type:
    :param i_cell_b:
    :return:
    '''

    # retrieves the condition filtered rotation data
    i_cell_b, r_obj_indiv = get_cond_filt_data(data, r_obj)

    # only return the matching indices/filters that intersect with the trial type filter array
    i_match = np.array([i for i, x in enumerate(r_obj_indiv) if x.rot_filt_tot[0]['t_type'][0] in tt_filt])
    i_cell_b, r_obj_indiv = i_cell_b[i_match], r_obj_indiv[i_match]

    if det_intersect:
        # if the common cell index array is not provided, then initialise it here
        if ind_cond is None:
            ind_cond = get_all_match_cond_cells(data, r_obj.rot_filt['t_type'])

        # collapses the cell index arrays to only those that are present for all selected trial conditions
        i_cell_b = [np.intersect1d(ind_cond[tt], i_c) for tt, i_c in zip(tt_filt, i_cell_b)]

    # returns the cell indices and the individually filtered objects
    return i_cell_b, r_obj_indiv


def check_existing_compare(comp_data, fix_name, free_name):
    '''

    :param comp_data:
    :param fix_name:
    :param free_name:
    :return:
    '''

    for i_comp in range(len(comp_data)):
        # determines if there is a match for either the fixed/free files
        file_match = (comp_data[i_comp].fix_name == fix_name) + 2 * (comp_data[i_comp].free_name == free_name)
        if file_match > 0:
            # if there is, then return their aggregate score
            return file_match, i_comp

    # if no match, then return a zero value
    return 0, 0


def get_matching_fix_free_strings(data, exp_name):
    '''

    :param data:
    :param exp_name:
    :param i_sel:
    :return:
    '''

    # retrieves the experiment index and mapping values
    i_expt, f2f_map = cf.det_matching_fix_free_cells(data, exp_name=[exp_name])
    is_ok = f2f_map[0][:, 1] > 0

    # retrieves the cluster indices
    i_ex = cf.get_global_expt_index(data, data.comp.data[i_expt[0]])
    clust_id = np.array(data._cluster[i_ex]['clustID'])

    # retrieves the fixed cluster ID#'s
    r_filt = cf.init_rotation_filter_data(False)
    r_filt['t_type'] += ['Uniform']
    r_obj = RotationFilteredData(data, r_filt, None, None, True, 'Whole Experiment', False)
    clust_id_fix = clust_id[r_obj.clust_ind[0][i_expt[0]][is_ok]]

    # retrieves the free cluster ID#'s
    clust_id_free = np.array(data.externd.free_data.cell_id[0])[f2f_map[0][is_ok, 1]]

    # returns the combined fixed/free cluster ID strings
    return ['Fixed #{0}/Free #{1}'.format(id_fix, id_free) for id_fix, id_free in zip(clust_id_fix, clust_id_free)]


def calc_posthoc_stats(y_orig, p_value=0.05, c_ofs=0):
    '''

    :param y:
    :param p_value:
    :return:
    '''

    def calc_kw_stats(y):
        '''

        :param y:
        :return:
        '''

        # sets the indices for the elements within each grouping
        i_grp = cf.flat_list([[i] * len(x) for i, x in enumerate(y)])

        # calculates and returns the kruskal wallis test p-value
        kw_stats = r_stats.kruskal_test(FloatVector(cf.flat_list(y)), FloatVector(i_grp))
        return kw_stats[kw_stats.names.index('p.value')][0]

    def calc_dunn_stats(y, p_kw_stats, p_value):
        '''

        :param y:
        :param is_sig:
        :return:
        '''

        # memory allocation
        n_grp = len(y)
        d_stats = np.empty((n_grp, n_grp), dtype=object)
        d_stats[:] = 'N/A'

        # calculates the dunn posthoc test values
        p_dunn = sp.posthoc_dunn(y, p_adjust='bonferroni')

        # sets the statistics strings for each grouping
        for i_grp in range(n_grp):
            for j_grp in range(i_grp + 1, n_grp):
                p_val = p_dunn[i_grp + 1][j_grp + 1]
                p_str = '{:.3f}{}'.format(p_val, cf.sig_str_fcn(p_val, p_value))
                d_stats[i_grp, j_grp] = d_stats[j_grp, i_grp] = p_str

        # returns the stats array
        return d_stats

    # initialisations and memory allocation
    p_within, p_btwn = None, None
    n_grp, n_filt = len(y_orig) - c_ofs, np.shape(y_orig[0])[1]

    # determines if the between filter type statistics need to be calculated (n_filt > 1)
    if n_grp > 1:
        # memory allocation
        p_within = np.empty((n_filt, 2), dtype=object)
        y_within = rmv_nan_elements([[list(x[:, i_f]) for x in y_orig[c_ofs:]] for i_f in range(n_filt)])

        # calculates the within filter type statistics for each group type
        for i_filt in range(n_filt):
            p_within[i_filt, 0] = calc_kw_stats(y_within[i_filt])
            p_within[i_filt, 1] = calc_dunn_stats(y_within[i_filt], p_within[i_filt, 0], p_value)

    # determines if the between filter type statistics need to be calculated (n_filt > 1)
    if n_filt > 1:
        # memory allocation
        p_btwn = np.empty((n_grp, 2), dtype=object)
        y_btwn = rmv_nan_elements([[list(y_orig[i + c_ofs][:, j]) for j in range(n_filt)] for i in range(n_grp)])

        # calculates the between filter type statistics for each group type
        for i_grp in range(n_grp):
            p_btwn[i_grp, 0] = calc_kw_stats(y_btwn[i_grp])
            p_btwn[i_grp, 1] = calc_dunn_stats(y_btwn[i_grp], p_btwn[i_grp, 0], p_value)

    # returns the
    return [p_within, p_btwn]