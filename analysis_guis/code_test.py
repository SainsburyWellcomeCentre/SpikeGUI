# -*- coding: utf-8 -*-
"""
Simple example using BarGraphItem
"""
# import initExample ## Add path to library (just for examples; you do not need this)

import numpy as np
import pickle as p
import pandas as pd

from analysis_guis.dialogs.rotation_filter import RotationFilter
from analysis_guis.dialogs import config_dialog
from analysis_guis.dialogs.info_dialog import InfoDialog
from rotation_analysis.analysis.probe.probe_io.probe_io import TriggerTraceIo, BonsaiIo, IgorIo
from PyQt5.QtWidgets import QApplication

from datetime import datetime
from dateutil import parser
import analysis_guis.calc_functions as cfcn
import analysis_guis.rotational_analysis as rot
import matplotlib.pyplot as plt

from pyphys.pyphys.pyphys import PxpParser
from collections import OrderedDict
import analysis_guis.common_func as cf

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

date2sec = lambda t: np.sum([3600 * t.hour, 60 * t.minute, t.second])
trig_count = lambda data, cond: len(np.where(np.diff(data[cond]['cpg_ttlStim']) > 1)[0]) + 1

get_bin_index = lambda x, y: next((i for i in range(len(y)) if x < y[i]), len(y)) - 1

def setup_polar_spike_freq(r_obj, sFreq, b_sz, is_pos):
    '''

    :param wvPara:
    :param tSpike:
    :param sFreq:
    :param b_sz:
    :return:
    '''

    # memory allocation
    wvPara, tSpike = r_obj.wvm_para[i_filt], r_obj.t_spike[i_filt],
    ind_inv, xi_bin_tot = np.empty(2, dtype=object), np.empty(2, dtype=object)

    # calculates the bin times
    xi_bin_tot[0], t_bin, t_phase = rot.calc_wave_kinematic_times(wvPara[0][0], b_sz, sFreq, is_pos, yDir=-1)
    xi_bin_tot[1], dt_bin = -xi_bin_tot[0], np.diff(t_bin)

    # determines the bin indices
    for i in range(2):
        xi_mid, ind_inv[i] = np.unique(0.5 * (xi_bin_tot[i][:-1] + xi_bin_tot[i][1:]), return_inverse=True)

    # memory allocation
    yDir = wvPara[0]['yDir']
    n_trial, n_bin = len(yDir), len(xi_mid)
    tSp_bin = np.zeros((n_bin, n_trial))

    #
    for i_trial in range(n_trial):
        # combines the time spikes in the order that the CW/CCW phases occur
        ii = int(yDir[i_trial] == 1)
        tSp = np.hstack((tSpike[1 + ii][i_trial], tSpike[2 - ii][i_trial] + t_phase))

        # appends the times
        t_hist = np.histogram(tSp, bins=t_bin)
        for j in range(len(t_hist[0])):
            i_bin = ind_inv[ii][j]
            tSp_bin[i_bin, i_trial] += t_hist[0][j] / (2.0 * dt_bin[j])

    # returns the final bin
    return xi_mid, tSp_bin

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    # loads the data for testing
    with open('C:\\Work\\EPhys\\Code\\Sepi\\wvPara.p', 'rb') as fp:
        wvPara = p.load(fp)
        tSpike = p.load(fp)

    #
    sFreq = 30000
    kb_sz = 10
    title_str = ['Displacement', 'Velocity']
    lg_str = ['Type 1', 'Type 2', 'Type 3']

    # memory allocation
    n_filt = len(wvPara)
    c = cf.get_plot_col(n_filt)

    #
    fig = plt.figure()
    ax = np.empty(2, dtype=object)

    #
    for i_type in range(2):
        # sets up the spiking frequency arrays
        tSp_bin = np.empty(n_filt, dtype=object)
        for i_filt in range(n_filt):
            xi_mid, tSp_bin[i_filt] = setup_polar_spike_freq(wvPara[i_filt], tSpike[i_filt], sFreq, kb_sz, i_type==0)

        #
        xi_min = xi_mid[0] - np.diff(xi_mid[0:2])[0]/2
        theta = np.pi * (1 - (xi_mid - xi_min) / np.abs(2 * xi_min))
        x_tick = np.linspace(xi_min, -xi_min, 7 + 2 * i_type)

        # creates the subplot
        ax[i_type] = plt.subplot(1, 2, i_type + 1, projection='polar')
        ax[i_type].set_thetamin(0)
        ax[i_type].set_thetamax(180)

        # creates the radial plots for each of the filter types
        h_plt = []
        for i_filt in range(n_filt):
            # creates the plot and resets the labels
            tSp_mn = np.mean(tSp_bin[i_filt], axis=1)
            h_plt.append(ax[i_type].plot(theta, tSp_mn, 'o-', c=c[i_filt]))

            # sets the axis properties (first filter only)
            if i_filt == 0:
                ax[i_type].set_title(title_str[i_type])
                ax[i_type].set_xticks(np.pi * (x_tick - xi_min) / np.abs(2 * xi_min))
                ax[i_type].set_xticklabels([str(int(np.round(-x))) for x in x_tick])

        # sets the legend (first subplot only)
        if i_type == 0:
            ax[i_type].legend(lg_str, loc=1)

    # determines the overall radial maximum (over all subplots) and resets the radial ticks
    y_max = [max(x.get_ylim()) for x in ax]
    i_max = np.argmax(y_max)
    dy = np.diff(ax[i_max].get_yticks())[0]
    y_max_tot = dy * (np.floor(y_max[i_max] / dy) + 1)

    # resets the axis radial limits
    for x in ax:
        x.set_ylim(0, y_max_tot)

    # shows the plot
    plt.show()
    a = 1

    # app = QApplication([])
    # h_obj = RotationFilter(data)
    # h_obj = InfoDialog(data)
    # a = 1

    # #
    # igor_waveforms_path = 'G:\\Seagate\\Work\\EPhys\\Data\\CA326_C_day3\\Igor\\CA326_C_day3'
    # bonsai_metadata_path = 'G:\\Seagate\\Work\\EPhys\\Data\\CA326_C_day3\\Bonsai\\CA326_C_day3_all.csv'
    #
    # #
    # file_time_key = 'FileTime'
    # bonsai_io = BonsaiIo(bonsai_metadata_path)
    #
    #
    # # determines the indices of the experiment condition triel group
    # t_bonsai = [parser.parse(x) for x in bonsai_io.data['Timestamp']]
    # t_bonsai_sec = np.array([date2sec(x) for x in t_bonsai])
    # d2t_bonsai = np.diff(t_bonsai_sec, 2)
    # grp_lim = grp_lim = [-1] + list(np.where(d2t_bonsai > 60)[0] + 1) + [len(d2t_bonsai) + 1]
    # ind_grp = [np.arange(grp_lim[x] + 1, grp_lim[x + 1] + 1) for x in range(len(grp_lim) - 1)]
    #
    # # sets the time, name and trigger count from each of these groups
    # t_bonsai_grp = [t_bonsai_sec[x[0]] for x in ind_grp]
    # c_bonsai_grp = [bonsai_io.data['Condition'][x[0]] for x in ind_grp]
    # n_trig_bonsai = [len(x) for x in ind_grp]
    #
    # # determines the feasible variables from the igor data file
    # igor_data = PxpParser(igor_waveforms_path)
    # var_keys = list(igor_data.data.keys())
    # is_ok = ['command' in igor_data.data[x].keys() if isinstance(igor_data.data[x], OrderedDict) else False for x in var_keys]
    #
    # # sets the name, time and trigger count from each of the igor trial groups
    # c_igor_grp = [y for x, y in zip(is_ok, var_keys) if x]
    # t_igor_grp, t_igor_str, n_trig_igor = [], [], [trig_count(igor_data.data, x) for x in c_igor_grp]
    # for ck in c_igor_grp:
    #     t_igor_str_nw = igor_data.data[ck]['vars'][file_time_key][0]
    #     t_igor_str.append(t_igor_str_nw)
    #     t_igor_grp.append(date2sec(datetime.strptime(t_igor_str_nw, '%H:%M:%S').time()))
    #
    # # calculates the point-wise differences between the trial timer and trigger count
    # dt_grp = cfcn.calc_pointwise_diff(t_igor_grp, t_bonsai_grp)
    # dn_grp = cfcn.calc_pointwise_diff(n_trig_igor, n_trig_bonsai)
    #
    # # ensures that only groups that have equal trigger counts are matched
    # dt_max = np.max(dt_grp) + 1
    # dt_grp[dn_grp > 0] = dt_max
    #
    # #
    # iter = 0
    # while 1:
    #     i2b = np.argmin(dt_grp, axis=1)
    #     i2b_uniq, ni2b = np.unique(i2b, return_counts=True)
    #
    #     ind_multi = np.where(ni2b > 1)[0]
    #     if len(ind_multi):
    #         if iter == 0:
    #             for ii in ind_multi:
    #                 jj = np.where(i2b == i2b_uniq[ii])[0]
    #
    #                 imn = np.argmin(dt_grp[jj, i2b[ii]])
    #                 for kk in jj[jj != jj[imn]]:
    #                     dt_grp[kk, i2b[ii]] = dt_max
    #         else:
    #             pass
    #     else:
    #         break
    #
    # # sets the igor-to-bonsai name groupings
    # i2b_key, x = {}, np.array(c_igor_grp)[i2b]
    # for cc in c_bonsai_grp:
    #     if cc not in i2b_key:
    #         jj = np.where([x == cc for x in c_bonsai_grp])[0]
    #         i2b_key[cc] = x[jj]
