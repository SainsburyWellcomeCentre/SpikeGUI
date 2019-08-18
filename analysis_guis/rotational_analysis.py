# module import
import copy
import peakutils
import math as m
import numpy as np
import pandas as pd
from random import random as rnd
from datetime import datetime
from dateutil import parser
from collections import OrderedDict

# custom module import
import analysis_guis.common_func as cf
from pyphys.pyphys.pyphys import PxpParser
from analysis_guis.dialogs import config_dialog
from rotation_analysis.analysis.probe.probe_io.probe_io import TriggerTraceIo, BonsaiIo, IgorIo


#
import analysis_guis.testing.diagnostic_plots as diag_plot
import _pickle as cp

# pyqt5 module import
from PyQt5.QtWidgets import (QMessageBox)

# other function declarations
dcopy = copy.deepcopy
date2sec = lambda t: np.sum([3600 * t.hour, 60 * t.minute, t.second])
trig_count = lambda data, cond: len(np.where(np.diff(data[cond]['cpg_ttlStim']) > 1)[0]) + 1
ss_scale = lambda x: np.min([np.max([x, -1.0]), 1.0])
valid_ind_func = lambda c, tt: [False if x['rotInfo'] is None else (tt in x['rotInfo']['trial_type']) for x in c]

# fixed parameters
DEGREES_PER_VOLT = 20
T_VISUAL_STIM = 2

########################################################################################################################
########################################################################################################################

####################################
####    DIAGNOSTIC FUNCTIONS    ####
####################################

def check_rot_analysis_files(hParent, exp_info, dlg_info):
    '''

    :param exp_info:
    :return:
    '''

    # initialisations
    ra_str, e_str = ['bonsaiFile', 'igorFile', 'probeTrigFile', 'stimOnsetFile', 'photoTrigFile'], None

    # determines
    if exp_info['expCond'] == 'Free':
        if np.any([len(exp_info[x]) for x in ra_str]):
            # if the experiment is free preparation, then all rotational analysis files must be removed
            e_str = 'All rotational analysis data files must be removed for free preparation experiments.\n' \
                    'Please ensure all these files are removed before continuing.'
    else:
        ra_file_len = np.array([len(exp_info[x]) for x in ra_str])
        ind, ind_f = [np.array([0, 1]), np.array([2]), np.array([3, 4])], np.arange(4, 9)

        if np.all(ra_file_len == 0):
            # if the experiment is free preparation, then all rotational analysis files must be removed
            e_str = 'No rotational analysis data files have been set for the fixed preparation experiment. ' \
                    'This means you will not be able to access any of the rotational analysis functions.\n\n' \
                    'Do you still wish to continue?'
            u_choice = QMessageBox.question(hParent, 'Output Images To Sub-Directory?', e_str,
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if u_choice == QMessageBox.No:
                e_str = 0

        else:
            # determines if the bonsai/igor files have all been set
            if np.any(ra_file_len[ind[0]] == 0):
                # if not, then set up the error string for each of the missing files
                e_str = 'The following the files have not been set:\n\n'
                for ff in np.where(ra_file_len[ind[0]] == 0)[0]:
                    e_str = '{0} * {1}\n'.format(e_str, dlg_info[ind_f[ind[0][ff]]][0])

            # determines if the other stimuli files have all been set
            if (ra_file_len[ind[1]] == 0):
                # if not, then add to the error message all of the missing files
                if np.any(ra_file_len[ind[2]] == 0):
                    if e_str is None:
                        # initialises the error string (if not done so already)
                        e_str = 'The following the files have not been set:\n\n'

                    # appends all the missing files to the error string. note that if either of the stimuli
                    # onset or photodiode triggering files have been set, then use this (instead of listing
                    # the probe trigger data file)
                    for i in range(1 + np.any(ra_file_len[ind[2]]), 3):
                        for ff in np.where(ra_file_len[ind[i]] == 0)[0]:
                            e_str = '{0} * {1}\n'.format(e_str, dlg_info[ind_f[ind[i][ff]]][0])

            # finialises the string
            if e_str is not None:
                e_str = '{0}\nPlease ensure these files are set before continuing ' \
                        'data file initialisation'.format(e_str)

    # returns the error string
    return e_str


def det_igor_to_bonsai_pairing(bonsai_io, igor_data):
    '''

    :return:
    '''

    #
    file_time_key = 'FileTime'

    # determines the indices of the experiment condition triel group
    t_bonsai = [parser.parse(x) for x in bonsai_io.data['Timestamp']]
    t_bonsai_sec = np.array([date2sec(x) for x in t_bonsai])
    d2t_bonsai = np.diff(t_bonsai_sec, 2)
    grp_lim = grp_lim = [-1] + list(np.where(d2t_bonsai > 60)[0] + 1) + [len(d2t_bonsai) + 1]
    ind_grp = [np.arange(grp_lim[x] + 1, grp_lim[x + 1] + 1) for x in range(len(grp_lim) - 1)]

    # sets the time, name and trigger count from each of these groups
    t_bonsai_grp = [t_bonsai_sec[x[0]] for x in ind_grp]
    c_bonsai_grp = [bonsai_io.data['Condition'][x[0]] for x in ind_grp]
    n_trig_bonsai = [len(x) for x in ind_grp]

    # determines the feasible variables from the igor data file
    var_keys = list(igor_data.data.keys())
    is_ok = ['command' in igor_data.data[x].keys() if isinstance(igor_data.data[x], OrderedDict) else False for x in var_keys]

    # sets the name, time and trigger count from each of the igor trial groups
    c_igor_grp0 = [y for x, y in zip(is_ok, var_keys) if x]
    t_igor_grp, t_igor_str, n_trig_igor = [], [], [trig_count(igor_data.data, x) for x in c_igor_grp0]
    has_ft = np.zeros(len(c_igor_grp0), dtype=bool)

    for ick, ck in enumerate(c_igor_grp0):
        if file_time_key in igor_data.data[ck]['vars']:
            has_ft[ick] = True
            t_igor_str_nw = igor_data.data[ck]['vars'][file_time_key][0]
            t_igor_str.append(t_igor_str_nw)
            t_igor_grp.append(date2sec(datetime.strptime(t_igor_str_nw, '%H:%M:%S').time()))

    # calculates the point-wise differences between the trial timer and trigger count
    n_trig_igor = [np.array(n_trig_igor)[has_ft]]
    c_igor_grp = np.array(c_igor_grp0)[has_ft]
    dt_grp = cf.calc_pointwise_diff(t_igor_grp, t_bonsai_grp)
    dn_grp = cf.calc_pointwise_diff(n_trig_igor, n_trig_bonsai)

    # ensures that only groups that have equal trigger counts are matched
    dt_max = np.max(dt_grp) + 1
    dt_grp[dn_grp > 0] = dt_max

    #
    i2b = -np.ones(len(c_bonsai_grp), dtype=int)
    for ii in range(len(c_bonsai_grp)):
        #
        imn = np.unravel_index(dt_grp.argmin(), dt_grp.shape)
        i2b[imn[0]] = imn[1]
        dt_grp[:, imn[1]] = dt_grp[imn[0], :] = dt_max

    # sets the igor-to-bonsai name groupings
    i2b_key, x = {}, np.array(c_igor_grp)[i2b]
    for ii, jj in enumerate(i2b):
        # only include the keys if there was a valid match
        if jj >= 0:
            cc = c_bonsai_grp[ii]
            if cc in i2b_key:
                i2b_key[cc].append(x[ii])
            else:
                i2b_key[cc] = [x[ii]]

    # adds in any missing entries
    key_match = [x in i2b_key for x in c_bonsai_grp]
    if not np.all(key_match):
        for ii in np.where(np.logical_not(key_match))[0]:
            # for all missing entries, determine the names that match between igor and bonsai
            cc = c_bonsai_grp[ii]
            if cc in c_igor_grp0:
                # if there is a direct match, then set that as being the match
                i2b_key[cc] = [cc]
            else:
                # WILL NEED TO CODE HERE TO FIX!
                a = 1

    # returns the keys
    return i2b_key


def get_rot_condition_key(exp_info):
    '''

    :param exp_info:
    :return:
    '''

    # loads the igor and bonsai data
    title, init_data = 'Experiment Condition Match', None
    bonsai_io = BonsaiIo(exp_info['bonsaiFile'])

    # retrieves the trigger trace I/O data
    if len(exp_info['stimOnsetFile']):
        # case is the stimuli onset/photo-diode trigger files have been provided
        tt_io = TriggerTraceIo(None,
                               stimulus_onset_trigger_path=exp_info['stimOnsetFile'],
                               photodiode_trigger_path=exp_info['photoTrigFile'])
    else:
        # case is the probe trigger file has been provided
        tt_io = TriggerTraceIo(exp_info['probeTrigFile'])

    # checks to see if the probe/bonsai trigger counts match
    if tt_io.n_triggers != bonsai_io.n_triggers:
        # if the bonaai/probe trigger counts do not match, then create an error string
        e_str = 'The trigger counts do not match between the Bonsai and Probe Trigger data files:\n\n' \
                ' * Bonsai File Trigger Count = {0}\n * Probe Trigger Count = {1}\n\n' \
                'Re-run the data file initialisation with the correct data files.'.format(
            bonsai_io.n_triggers, tt_io.n_triggers
        )

        # outputs to screen an error and exits with an error flag
        cf.show_error(e_str, 'Mismatching Rotational Analysis Data Files')
        return None, False

    # determines the feasible variables from the igor data file
    igor_data = PxpParser(exp_info['igorFile'])
    var_keys = list(igor_data.data.keys())
    is_ok = ['command' in igor_data.data[x].keys() if isinstance(igor_data.data[x], OrderedDict) else False for x in var_keys]
    feas_var = [y for x, y in zip(is_ok, var_keys) if x]

    # retrieves the conditions from the bonsai file
    i2b_key = det_igor_to_bonsai_pairing(bonsai_io, igor_data)

    # creates the dialog info data struct
    dlg_info = [[] for _ in i2b_key]
    for ik, ck in enumerate(i2b_key):
        dlg_info[ik] = ['Experiment Trial Condition = "{0}"'.format(ck), ck, 'CheckCombo', feas_var, True, False, ik]

    # opens up the config dialog box and retrieves the final file information
    while True:
        # opens up the config dialog box and retrieves the final file information
        cfig_dlg = config_dialog.ConfigDialog(dlg_info=dlg_info, title=title, width=500,
                                              init_data=i2b_key, has_reset=False)

        # retrieves the information from the dialog
        trig_key_dlg = cfig_dlg.get_info()
        if trig_key_dlg is None:
            # if the user cancelled, then exit the loop with a false flag
            return None, False
        else:
            # retrieves the user's selections
            trig_key = cfig_dlg.get_info()

            # determines if there have been any multiple condition selections made
            ck, nck = np.unique(cf.flat_list(list(trig_key.values())), return_counts=True)
            if len(np.where(nck > 1)[0]):
                # if there are multiple condition selections, then output an error to screen
                e_str = 'The following conditions have repeat selections:\n'
                for ick in np.where(nck > 1)[0]:
                    # sets the name of the ignr condition that was repeated
                    e_str = '{0}\n * "{1}" in'.format(e_str, ck[ick])
                    mlt_match = [k for k in trig_key if ck[ick] in trig_key[k]]

                    # lists the bonsai conditions where the repetition was made
                    for i_mm, mm in enumerate(mlt_match):
                        e_str = '{0} {1}{2}'.format(e_str, mlt_match[i_mm], ',' if (i_mm+1) < len(mlt_match) else '')

                # outputs the final error message to screen
                e_str = '{0}\n\nRemove these repeated selections before ' \
                        'continuing the data file initialisation.'.format(e_str)
                cf.show_error(e_str, 'Incorrect Match Selections')

                # sets the initial output to the current selections
                i2b_key = trig_key
            else:
                # calculates the number of triggers for each specified trigger type
                n_trig = dict([(tk, [trig_count(igor_data.data, x) for x in trig_key[tk]]) for tk in trig_key])
                n_trig_tot = sum(cf.flat_list(list(n_trig.values())))
                if n_trig_tot != tt_io.n_triggers:
                    # if the total number of triggers doesn't match the probe trigger count, then output an error
                    e_str = 'The trigger counts do not match between the Bonsai/Probe Trigger data files and the ' \
                            'Igor experiment condition group selections:\n\n' \
                            ' * Bonsai/Probe Trigger Count = {0}\n' \
                            ' * Igor Experiment Condition Group Selections Count = {1}\n'.format(
                        tt_io.n_triggers, n_trig_tot
                    )

                    #
                    for tk in trig_key:
                        e_str = '{0}\n   -> "{1}" Trigger Counts = {2}\n'.format(e_str, tk, sum(n_trig[tk]))
                        for tk_sub, ntk_sub in zip(trig_key[tk], n_trig[tk]):
                            e_str = '{0}       * "{1}" = {2}\n'.format(e_str, tk_sub, ntk_sub)

                    #
                    e_str = '{0}\nYou will need to ensure the trigger counts match before continuing'.format(e_str)
                    cf.show_error(e_str, 'Incorrect Experiment Type Groupings')

                    # sets the initial output to the current selections
                    i2b_key = trig_key
                else:
                    # otherwise, return the match selections
                    return trig_key, True

################################################
####    ROTATIONAL ANALYSIS DATA LOADING    ####
################################################

def load_rot_analysis_data(A, exp_info, sp_io, w_prog=None, pW0=None, is_diagnostic=False):
    '''

    :return:
    '''

    # if there is no rotational data files provided, then exit
    if len(exp_info['bonsaiFile']) == 0:
        return None

    # retrieves the experiment parameters
    n_clust, s_freq, clust_id = A['nC'], A['sFreq'], A['clustID']

    # updates the progess-bar (if provided)
    if w_prog is not None:
        w_prog.emit('Loading Rotational Analysis Files...', 5.0)

    # retrieves the trigger trace I/O data
    if len(exp_info['stimOnsetFile']):
        # case is the stimuli onset/photo-diode trigger files have been provided
        tt_io = TriggerTraceIo(None,
                               stimulus_onset_trigger_path=exp_info['stimOnsetFile'],
                               photodiode_trigger_path=exp_info['photoTrigFile'])
    else:
        # case is the probe trigger file has been provided
        tt_io = TriggerTraceIo(exp_info['probeTrigFile'])

    # creates the bonsai I/O class object from the provided file
    bonsai_io, i2b_key = BonsaiIo(exp_info['bonsaiFile']), exp_info['i2b_key']
    trial_type = np.array(list(i2b_key.keys()))

    # creates the igor I/O class object from the provided file and associated trial name keys
    igor_io = IgorIo(exp_info['igorFile'], trial_type)

    # retrieves the upscaled waveforms and the global trial index array
    tt_igor = np.array(cf.flat_list([list(x) for x in i2b_key.values()]))
    s_rate = [igor_io.get_sampling_rate_from_condition(x)[0] for x in tt_igor]
    ind_trial = [np.where(bonsai_io.data['Condition'] == ck)[0] for ck in i2b_key]
    i_grp = [np.array([list(tt_igor).index(y) for y in i2b_key[x]]) for x in trial_type]

    #
    w_form0 = np.array([upsample_waveform(igor_io.get_waveforms_in_condition_group([x]),
                                                        s_freq / y) for x, y in zip(tt_igor, s_rate)])
    if isinstance(w_form0[0], list):
        w_form = [sum(list(w_form0[ig]), []) for ig in i_grp]
    else:
        w_form = [cf.flat_list([w_form0[y] for y in ig]) for ig in i_grp]

    # updates the progess-bar (if provided)
    if w_prog is not None:
        w_prog.emit('Determines Trial Time Points...', 10.0)

    # retrieves the trigger start/end time points for each of the trials
    i_trig_loc, i_trig_start, i_trig_end = get_trial_time_points(tt_io, w_form, ind_trial, s_freq)
    t_spike_trial = get_trial_spike_times(trial_type, clust_id, sp_io, i_trig_start, i_trig_end)


    # updates the progess-bar (if provided)
    if w_prog is not None:
        w_prog.emit('Splitting Phase Spike Times...', 15.0)

    # calculates the parameters for each of the stimuli waveforms
    wfm_para, t_spike = {}, {}
    for i in range(len(trial_type)):
        # calculates the waveform parameters
        ind, tt = ind_trial[i], trial_type[i]
        wfm_para_tmp = [
            det_waveform_para(x, s_freq, bonsai_io, y, z) for x, y, z in zip(w_form[i], ind_trial[i], i_trig_loc[i])
        ]

        # splits the spikes within the trial into the separate phases
        wfm_para[tt] = pd.DataFrame(wfm_para_tmp)[list(wfm_para_tmp[0].keys())].to_records(False)
        t_spike[tt] = set_stimuli_phase_spikes(wfm_para[tt], t_spike_trial[tt], trial_type[i])

    # sets the final rotational analysis information dictionary
    rot_info = {'t_spike': t_spike, 'wfm_para': wfm_para, 'trial_type': trial_type, 'ind_trial': ind_trial}

    # returns the rotational analysis data dictionary
    return rot_info


def set_stimuli_phase_spikes(wfm_para, t_spike_trial, t_type):
    '''

    :param wfm_para:
    :param t_spike_trial:
    :param n_clust:
    :return:
    '''

    # memory allocation and other initialisations
    n_clust, n_trial = np.size(t_spike_trial, axis=0), np.size(t_spike_trial, axis=1)

    # sets the phase spike times based on the trial stimuli type
    if np.any(wfm_para['yAmp'] > 0):
        # case is the trial involves a rotation stimuli

        # memory allocation
        t_spike = np.empty((n_clust, n_trial, 3), dtype=object)

        # determines which stimuli is
        if t_type == 'MotorDrifting':
            is_P1_CW = (wfm_para['yDir'] > 0).astype(int)
        else:
            is_P1_CW = (wfm_para['yDir'] < 0).astype(int)

        # sets the start/end points for each of the phases
        t_half = wfm_para['tPeriod'] / 2.0
        ind_BLS = np.vstack([wfm_para['tBLF'] - np.floor(t_half), wfm_para['tBLF']]).T
        ind_SS1 = np.vstack([wfm_para['tSS0'], wfm_para['tSS0'] + np.floor(t_half)]).T
        ind_SS2 = np.vstack([wfm_para['tSS0'] + np.ceil(t_half), wfm_para['tSS0'] + wfm_para['tPeriod']]).T

        # sets the time spikes for each phase within each trial
        for i_trial in range(n_trial):
            # sets the indices of the storage indices for the stimuli 1/2 phases (links CC/CCW to phases)
            #  IMPORTANT FLAG - THIS COULD BE INCORRECT?! MATCH UP WITH STEVE'S CODE
            iSS1, iSS2 = 2 - is_P1_CW[i_trial], 1 + is_P1_CW[i_trial]

            # sets the spike times for each phase (the baseline, clockwise and counter-clockwise) for each cluster
            for i_clust in range(n_clust):
                t_spike[i_clust, i_trial, 0] = set_phase_events(t_spike_trial[i_clust, i_trial], ind_BLS[i_trial, :])
                t_spike[i_clust, i_trial, iSS1] = set_phase_events(t_spike_trial[i_clust, i_trial], ind_SS1[i_trial, :])
                t_spike[i_clust, i_trial, iSS2] = set_phase_events(t_spike_trial[i_clust, i_trial], ind_SS2[i_trial, :])
    else:
        # case is the trial involves a visual stimuli

        # memory allocation
        t_spike = np.empty((n_clust, n_trial, 2), dtype=object)

        # index arrays for the baseline/stimulus experiment phases
        nPts = int(wfm_para['nPts'][0] / 2)
        ind_BLS, ind_STIM = [0, nPts], [nPts, 2*nPts]

        # sets the time spikes for each phase within each trial
        for i_trial in range(n_trial):
            for i_clust in range(n_clust):
                t_spike[i_clust, i_trial, 0] = set_phase_events(t_spike_trial[i_clust, i_trial], ind_BLS)
                t_spike[i_clust, i_trial, 1] = set_phase_events(t_spike_trial[i_clust, i_trial], ind_STIM)

    # returns the spike times
    return t_spike


def set_phase_events(t_spike, ind_phase):
    '''

    :param t_spike:
    :param ind_phase:
    :return:
    '''

    # returns the spike times within the phase
    return t_spike[np.logical_and(t_spike > ind_phase[0], t_spike < ind_phase[1])] - ind_phase[0]


def get_trial_spike_times(cond_key, clust_id, sp_io, i_trig_start, i_trig_end):
    '''

    :param clust_id:
    :param sp_io:
    :param i_trig_start:
    :param i_trig_end:
    :return:
    '''

    # memory allocation
    n_clust, t_spike = len(clust_id), {}

    for ik, ck in enumerate(cond_key):
        # memory allocation
        n_trial = len(i_trig_start[ik])
        t_spike[ck] = np.empty((n_clust, n_trial), dtype=object)

        # retrieves the cluster spike times for each cluster over each trial
        for i_clust in range(n_clust):
            for i_trial in range(n_trial):
                t_spike0 = sp_io.cluster_spike_times_in_interval(clust_id[i_clust],
                                                                 i_trig_start[ik][i_trial],
                                                                 i_trig_end[ik][i_trial])
                t_spike[ck][i_clust, i_trial] = t_spike0 - i_trig_start[ik][i_trial]

    # returns the trial spike times
    return t_spike


def get_trial_time_points(tt_io, w_form, ind_trial, s_freq):
    '''

    :param tt_io:
    :param w_form:
    :return:
    '''

    def get_trigger_ofs(w_form, s_freq):
        '''

        :param w_form:
        :return:
        '''

        # offset only occurs for flat-waveforms
        #  --- CHANGE THIS SO THAT TIME OFFSET FOR UNIFORMDRIFTING IS A MAX OF 2 SECONDS (not the duration of the waveform)
        return [T_VISUAL_STIM * s_freq if (np.max(np.abs(x)) < 1.0) else 0 for x in w_form]

    # calculates the time offsets
    t_visual_bl = [get_trigger_ofs(x, s_freq) for x in w_form]
    n_pts = [z if (z[0] > 0) else [len(y) for y in x] for x, z in zip(w_form, t_visual_bl)]

    # determines the trigger location and start/end index points
    i_trig_loc = [[tt_io.get_corrected_trigger(idx, verbose=False) for idx in idt] for idt in ind_trial]
    i_trig_start = [[(loc - dt) for loc, dt in zip(x, y)] for x, y in zip(i_trig_loc, t_visual_bl)]
    i_trig_end = [[(loc + np) for loc, np in zip(x,y)] for x, y in zip(i_trig_loc, n_pts)]

    # returns the index arrays
    return i_trig_loc, i_trig_start, i_trig_end


def upsample_waveform(w_form, s_factor):
    '''

    :return:
    '''

    # memory allocation
    w_form_us = [[] for _ in range(len(w_form))]

    # upscales the wave forms for by the required scale factor
    for i in range(len(w_form)):
        # sets the interpolation x/y points
        n_pts = len(w_form[i])
        x, y = np.linspace(0, n_pts * s_factor, n_pts), DEGREES_PER_VOLT * w_form[i].flatten()

        # upscales the waveform
        w_form_us[i] = np.interp(np.linspace(0, n_pts * s_factor, n_pts * s_factor), x, y)

    # returns the upscaled waveforms
    return w_form_us


def det_waveform_para(y_sig, s_freq, bonsai_io, idx, ind0):
    '''

    :param y_sig:
    :return:
    '''

    # calculates the amplitude of the signal
    y_amp = np.round((np.max(y_sig) - np.min(y_sig)) / 2.0)

    if (np.abs(y_amp) > 1e-6):
        # otherwise, determine the minimum/maximum points
        s_para = {'yAmp': y_amp, 'ind0': ind0, 'yDir': 1.0, 'tBLF': None,
                  'tSS0': 0.0, 'tPeriod': None,'nPts': len(y_sig)}
        i_min, i_max = peakutils.indexes(-y_sig), peakutils.indexes(y_sig)

        # calculates the period of the signal
        if len(i_min) == 2:
            # case is there is 2 minimum points
            s_para['yDir'] = -1.0
            s_para['tPeriod'], s_para['tSS0'] = np.diff(i_min)[0], i_min[0]
        else:
            # case is there is 2 maximum points
            s_para['tPeriod'], s_para['tSS0'] = np.diff(i_max)[0], i_max[0]

        # calculates the base-line time
        dy = s_para['yDir'] * (y_sig - recreate_waveform(s_para))
        s_para['tBLF'] = next(i for i in range(s_para['nPts']) if np.abs(dy[i]) > 1e-16) - 1
    else:
        #
        t_freq = bonsai_io.data['TemporalFrequency'][idx]
        t_cycles = bonsai_io.data['TotalCycles'][idx]

        #
        s_para = {'yAmp': 0, 'ind0': ind0, 'yDir': np.sign(t_freq), 'tFreq': np.abs(t_freq),
                  'tCycle': t_cycles, 'nPts': 2 * T_VISUAL_STIM * s_freq}

    # returns the parameters of the signal
    return s_para


def recreate_waveform(s_para, calc_deriv=False):
    '''

    :param s_para:
    :return:
    '''

    # memory allocation
    y_sig = np.zeros(s_para['nPts'])

    # calculates the sinusoidal component of the signal (if the parameters are set)
    if s_para['tPeriod'] is not None:
        # sets up the sinusoidal component of the signal
        t_rng = np.arange(np.round(s_para['tSS0']) - s_para['tPeriod'] / 4.0,
                          np.round(s_para['tSS0']) + s_para['tPeriod'] * (5.0 / 4.0))
        i_rng = t_rng.astype(int)

        # sets the signal period
        freq = 2.0 * np.pi / s_para['tPeriod']
        if calc_deriv:
            y_sig[i_rng] = -freq * s_para['yAmp'] * s_para['yDir'] * np.sin(freq * (t_rng - s_para['tSS0']))
        else:
            y_sig[i_rng] = s_para['yAmp'] * s_para['yDir'] * np.cos(freq * (t_rng - s_para['tSS0']))

            # # calculates the other components of the signal (if required - only for position values)
            # if (not sinusoid_only) and (s_para['ppSig'] is not None):
            #     y_sig += s_para['yDir'] * s_para['ppSig'](np.arange(s_para['nPts']))

    # returns the final signal
    return y_sig


def setup_filter_permutations(d_clust, rot_filt):
    '''

    :param rot_filt:
    :return:
    '''

    # ensures all trial types are included (if running a rotational analysis expt with 'all' trial types)
    if ('All' in rot_filt['t_type']) and (not rot_filt['is_ud'][0]):
        t_type = np.unique(cf.flat_list([x['rotInfo']['trial_type'] if x['rotInfo'] is not None else [] for x in d_clust]))
        rot_filt['t_type'] = t_type[t_type != 'UniformDrifting']

    # determines which fields have multiple selections
    f_perm, f_key = None, []
    rf_key = np.array(list(rot_filt.keys()), dtype=object)
    rf_val = np.array(list(rot_filt.values()), dtype=object)

    # determines the multi-selected filter fields (removes for the key field)
    n_filt = np.array([len(x) for x in rf_val])
    n_filt[list(rf_key).index('t_key')] = 1

    # determines which values are to be included in the filtering
    is_specific = [False if ccf in ['t_type', 't_key'] else
                                (isinstance(val[0], str) and(val[0] != 'All')) for ccf, val in zip(rf_key, rf_val)]
    is_multi = np.logical_or(n_filt > 1, is_specific)

    # memory allocation
    if not np.any(is_multi):
        # no fields have any multiple selections
        rot_filt_ex = [rot_filt]
    else:
        # sets up the key/permutation arrays
        for rf_k, rf_v in zip(rf_key[is_multi], rf_val[is_multi]):
            if f_perm is None:
                #
                f_perm, f_key = np.reshape(rf_v, (-1, len(rf_v))).T, [rf_k]
            else:
                #
                f_key = [rf_k] + f_key
                f_perm = np.vstack(
                    [np.concatenate(
                        (np.reshape([v] * np.size(f_perm, axis=0), (-1, np.size(f_perm, axis=0))).T, f_perm), axis=1
                    ) for v in rf_v]
                )

        # creates each different
        rot_filt_ex = [copy.deepcopy(rot_filt) for _ in range(np.size(f_perm, axis=0))]
        for i in range(np.size(f_perm, axis=0)):
            for ifk, fk in enumerate(f_key):
                rot_filt_ex[i][fk] = [f_perm[i, ifk]]

    # returns the parameter permutation list
    return rot_filt_ex, f_perm, f_key


def apply_rot_filter(data, rot_filt, expt_filter_lvl, exp_name):
    '''

    :param data:
    :param rot_filt:
    :return:
    '''

    # retrieves the data clusters for each of the valid rotation experiments
    is_rot_expt = cf.det_valid_rotation_expt(data)

    # sets the experiment
    if exp_name is None:
        # case is filtering multiple experiments
        i_expt_match = np.where(is_rot_expt)[0]
        if data.cluster is None:
            d_clust = [data._cluster[x] for x in i_expt_match]
        else:
            d_clust = [data.cluster[x] for x in i_expt_match]
    else:
        # case is filtering on a single experiment level
        if data.cluster is None:
            i_expt_match = [cf.get_expt_index(exp_name, data._cluster, cf.det_valid_rotation_expt(data))]
            d_clust = [data._cluster[i_expt_match[0]]]
        else:
            i_expt_match = [cf.get_expt_index(exp_name, data.cluster, cf.det_valid_rotation_expt(data))]
            d_clust = [data.cluster[i_expt_match[0]]]

    # sets up the filter permutation array
    rot_filt_p, f_perm, f_key = setup_filter_permutations(d_clust, copy.deepcopy(rot_filt))

    # memory allocation
    n_filt, d_copy = len(rot_filt_p), copy.deepcopy
    A = np.empty(n_filt, dtype=object)
    t_spike, wvm_para, trial_ind, clust_ind, i_expt = d_copy(A), d_copy(A), d_copy(A), d_copy(A), d_copy(A)

    # applies the rotation filter for each filter permutation
    is_ok = np.zeros(n_filt, dtype=bool)
    for i_filt in range(n_filt):
        t_spike[i_filt], wvm_para[i_filt], trial_ind[i_filt], clust_ind[i_filt], i_expt[i_filt] = \
                        apply_single_rot_filter(data, d_clust, rot_filt_p[i_filt], expt_filter_lvl, i_expt_match)
        is_ok[i_filt] = not np.all([x is None for x in t_spike[i_filt]])

    # determines if any of the filters failed to turn up a match, then output an error message to screen
    if np.any(np.logical_not(is_ok)):
        # # if so, then create the warning message
        # e_str = 'The following filters did not turn up any matches for the experiment "{0}":\n\n'.format(exp_name)
        # for i in np.where(np.logical_not(is_ok))[0]:
        #     e_str = '{0} {1} {2}\n'.format(e_str, cf._bullet_point, ', '.join(f_perm[i, :]))
        # e_str = '{0}\nTo prevent this message in future then de-select these filter options.'.format(e_str)
        #
        # # outputs the message to screen
        # cf.show_error(e_str, 'Missing Filter Matches')

        # removes any of the non-feasible entries
        t_spike, wvm_para, trial_ind = t_spike[is_ok], wvm_para[is_ok], trial_ind[is_ok]
        clust_ind, i_expt, f_perm = clust_ind[is_ok], i_expt[is_ok], f_perm[is_ok, :]
        rot_filt_p = [x for x, y in zip(rot_filt_p, is_ok) if y]

    # returns the spike time/waveform parameter/filter parameter arrays
    return t_spike, wvm_para, trial_ind, clust_ind, i_expt, f_perm, f_key, rot_filt_p

def apply_single_rot_filter(data, d_clust, rot_filt, expt_filter_lvl, i_expt_match):
    '''

    :param data:
    :param rot_filt:
    :return:
    '''

    # sets the trial types
    if rot_filt['is_ud'][0]:
        t_type, is_ud = 'UniformDrifting', True
        exc_filt = data.exc_ud_filt
        if exc_filt is None:
            exc_filt = cf.init_rotation_filter_data(True, is_empty=True)
    else:
        t_type, is_ud = rot_filt['t_type'][0], False
        exc_filt = data.exc_rot_filt
        if exc_filt is None:
            exc_filt = cf.init_rotation_filter_data(False, is_empty=True)

    # memory allocation
    n_expt, d_copy = len(d_clust), copy.deepcopy
    is_ok, A = np.zeros(n_expt, dtype=bool), np.empty(n_expt, dtype=object)
    t_spike, wfm_para, trial_ind, clust_ind = d_copy(A), d_copy(A), d_copy(A), d_copy(A)

    ##########################################
    ####    EXPERIMENT-BASED FILTERING    ####
    ##########################################

    # experiment-wide conditions
    chk_str = ['record_coord']

    # ensures that the experiments meet the conditions in the list
    for i_expt in range(n_expt):
        # determines if the trial type exists in the current experiment
        if 'rotInfo' in d_clust[i_expt]:
            is_ok[i_expt] = t_type in d_clust[i_expt]['rotInfo']['trial_type']

        if not is_ok[i_expt]:
            # if not, then continue
            continue

        # determines if the check string exists in the current experiments experimental info check array (only perform
        # the check if the user is searching for a specific filter (i.e., not "All"))
        for c_type in chk_str:
            if rot_filt[c_type][0] != 'All':
                is_ok[i_expt] = d_clust[i_expt]['expInfo'][c_type] in rot_filt[c_type]

    # if there are no valid experiments which meet the trial/experiment filter conditions then exit with None values
    if not np.any(is_ok):
        return [None] * n_expt, [None] * n_expt, [None] * n_expt, [None] * n_expt, np.array([])

    ####################################
    ####    TRIAL-TYPE FILTERING    ####
    ####################################

    #### retrieves the rotational data for the experiment trial types

    # splits the spike times/waveform parameters by the trial type
    for i_expt in range(n_expt):
        # only store values if the experiment is a valid rotation analysis data file
        if is_ok[i_expt]:
            # retrieves the spike counts and waveform parameters for the current experiment
            t_spike[i_expt] = d_clust[i_expt]['rotInfo']['t_spike'][t_type]
            wfm_para[i_expt] = d_clust[i_expt]['rotInfo']['wfm_para'][t_type]

    ############################################
    ####    CELL-CLUSTER BASED FILTERING    ####
    ############################################

    # sets the filter strings
    cc_filt_str = ['sig_type', 'match_type', 'region_name', 'record_layer']
    is_check = [(data.classify.class_set and (expt_filter_lvl > 0)),    # signal type must be calculated and not single cell
                (data.comp.is_set and (expt_filter_lvl > 0)),           # comparison type must be calculated and not single cell
                (expt_filter_lvl > 0),                                  # not single cell
                (expt_filter_lvl > 0)]                                  # not single cell

    # if uniform drifting, add on the visual stimuli parameters to the filter conditions
    if rot_filt['is_ud'][0]:
        cc_filt_str += ['t_freq', 't_freq_dir', 't_cycle']
        is_check += [True] * 3

    # applies the field filters to each experiment
    for i_expt in range(n_expt):
        if is_ok[i_expt]:
            # sets the indices of the values that are to be kept
            ind_cl = np.ones(np.size(t_spike[i_expt], axis=0), dtype=bool)
            ind_tr = np.ones(np.size(t_spike[i_expt], axis=1), dtype=bool)

            # goes through the filter fields removing entries that don't meet the criteria
            for iccf, ccf in enumerate(cc_filt_str):
                ind_cl_nw, ind_tr_nw = None, None

                if (rot_filt[ccf][0] == 'All'):
                    if len(exc_filt[ccf]):
                        # initialisations
                        is_cl, is_tr = False, False

                        if ccf == 'sig_type':
                            # case is the signal type (wide or narrow spikes)
                            grp_str = data.classify.grp_str[i_expt][data.classify.grp_str[i_expt] != 'N/A']
                            cv, is_cl = ['{0} Spikes'.format(x) for x in grp_str], True

                        elif ccf == 'match_type':
                            # case is the match type (either Matched or Unmatched)
                            is_cl = True
                            cv = ['Matched Clusters' if x else 'Unmatched Clusters' for x in data.comp.is_accept]

                        elif ccf == 'region_name':
                            # case is the region name
                            cv, is_cl = d_clust[i_expt]['chRegion'], True

                        elif ccf == 'record_layer':
                            # case is the recording layer
                            cv, is_cl = d_clust[i_expt]['chLayer'], True

                        elif ccf == 't_freq':
                            # case is the temporal frequency
                            cv, is_tr = [str(x) for x in wfm_para[i_expt]['tFreq']], True

                        elif ccf == 't_freq_dir':
                            # case is the temporal direction (-1/CW or 1/CCW)
                            cv, is_tr = [str(int(x)) for x in wfm_para[i_expt]['yDir']], True

                        elif ccf == 't_cycle':
                            # case is the temporal cycle frequency
                            cv, is_tr = [str(int(x)) for x in wfm_para[i_expt]['tCycle']], True

                        # sets the new cluster/trial acceptance flags (based on type)
                        if is_cl:
                            # case is the filter type is cluster based
                            ind_cl_nw = np.logical_and.reduce([np.array([yy != x for yy in cv]) for x in exc_filt[ccf]])
                        else:
                            # case is the filter type is trial based
                            ind_tr_nw = np.logical_and.reduce([np.array([yy != x for yy in cv]) for x in exc_filt[ccf]])

                elif is_check[iccf] and (rot_filt[ccf][0] != 'All'):
                    if ccf == 'sig_type':
                        grp_str = data.classify.grp_str[i_expt][data.classify.grp_str[i_expt] != 'N/A']
                        ind_cl_nw = [np.any([x in y for y in rot_filt[ccf]]) for x in grp_str]

                    elif ccf == 'match_type':
                        m_flag = (rot_filt[ccf][0].split(' ')[0] != 'Matched')
                        ind_cl_nw = list(np.where(data.comp.is_accept == m_flag)[0])

                    elif ccf == 'region_name':
                        ind_cl_nw = [x == rot_filt[ccf][0] for x in d_clust[i_expt]['chRegion']]

                    elif ccf == 'record_layer':
                        ind_cl_nw = [x == rot_filt[ccf][0] for x in d_clust[i_expt]['chLayer']]

                    elif ccf == 't_freq':
                        ind_tr_nw = np.abs(wfm_para[i_expt]['tFreq'] - float(rot_filt[ccf][0])) < 1e-6

                    elif ccf == 't_freq_dir':
                        ind_tr_nw = np.abs(wfm_para[i_expt]['yDir'] - float(rot_filt[ccf][0])) < 1e-6

                    elif ccf == 't_cycle':
                        ind_tr_nw = np.abs(wfm_para[i_expt]['tCycle'] - float(rot_filt[ccf][0])) < 1e-6

                # removes any infeasible trials
                if ind_tr_nw is not None:
                    ind_tr = np.logical_and(ind_tr, np.array(ind_tr_nw))

                # removes any infeasible clusters
                if ind_cl_nw is not None:
                    ind_cl = np.logical_and(ind_cl, ind_cl_nw)


            # removes the infeasible cluster rows/trial columns
            t_spike[i_expt] = t_spike[i_expt][ind_cl, :, :]
            t_spike[i_expt] = t_spike[i_expt][:, ind_tr, :]

            # removes the infeasible trial/cluster indices from the other arrays
            wfm_para[i_expt] = wfm_para[i_expt][ind_tr]
            trial_ind[i_expt] = np.where(ind_tr)[0]
            clust_ind[i_expt] = np.where(ind_cl)[0]

    # returns the spike times/waveform parameters
    return t_spike[is_ok], wfm_para[is_ok], trial_ind[is_ok], clust_ind[is_ok], np.array(i_expt_match)[is_ok]


def calc_waveform_values(A, w, t):
    '''

    :param A:
    :param w:
    :param t:
    :return:
    '''

    # returns the position/velocity values
    return A * np.cos(w * t), -A * w * np.sin(w * t)


def calc_kinematic_bin_times(b_sz, k_rng, w, calc_type=2):
    '''

    :param b_sz:
    :param k_rng:
    :param w:
    :return:
    '''

    # lambda function declarations
    rev_func = lambda x: np.concatenate((x, 2 * x[-1] - np.flip(x[:-1], axis=0)))
    rev_func_2 = lambda x: np.concatenate(
        (np.concatenate((x, np.flip(x[:-1], axis=0))), -np.concatenate((x, np.flip(x[:-1], axis=0)))[1:]))

    # memory allocation
    n_bin = 2
    A = np.empty(n_bin, dtype=object)
    xi_bin, xi_bin0, t_bin, i_bin = dcopy(A), dcopy(A), dcopy(A), dcopy(A)

    # sets up the positional time/location arrays
    xi_binT = np.arange(k_rng[0], -(k_rng[0] + 1e-6), -b_sz[0])
    xi_bin0[0] = np.concatenate((xi_binT, np.flip(xi_binT[:-1], axis=0)))
    t_bin[0] = rev_func(np.unique([m.acos(ss_scale(x / k_rng[0])) for x in xi_bin0[0]]) / w)

    # sets up the velocity time/location arrays
    xi_bin0[1] = rev_func_2(np.arange(0, k_rng[1] + 1e-6, b_sz[1]))
    t_binT = np.unique(np.abs([m.asin(ss_scale(-v / k_rng[1])) for v in xi_bin0[1]])) / w
    t_bin[1], xi_bin0[1][0] = rev_func(rev_func(t_binT)), 0

    # determines the groupings for each time bin
    for i in range(n_bin):
        i_grp = np.vstack([np.sort([xi_bin0[i][k:(k+2)]])[0] for k in range(len(xi_bin0[i])-1)])
        xi_bin[i], i_bin[i] = np.unique(i_grp, axis=0, return_inverse=True)

    # returns the time-bin array (depending on the calculation type)
    if calc_type == 0:
        # case is only position is being considered
        return xi_bin0[0], xi_bin[0], t_bin[0], i_bin[0]
    elif calc_type == 1:
        # case is only velocity is being considered
        return xi_bin0[1], xi_bin[1], t_bin[1], i_bin[1]
    else:
        # case is both position/velocity is being considered
        return xi_bin0, xi_bin, t_bin, i_bin


def calc_resampled_vel_spike_freq(data, w_prog, r_obj, b_sz, n_sample, indD=None, r_data=None):
    '''

    :param data:
    :param r_obj:
    :param b_sz:
    :param n_sample:
    :return:
    '''

    def calc_resampled_counts(t_sp, t_bin, dt_min, n_sample):
        '''

        :param t_sp:
        :param t_bin:
        :param t_bin_min:
        :param n_sample:
        :return:
        '''

        # calculates the residual gap between the time bin duration and the min time bin duration
        dt_gap = np.diff(t_bin)[0] - dt_min

        # determines if there are any spikes not completely covered by the min time duration
        if (t_sp[0] < (t_bin[0] + dt_gap)) or (t_sp[-1] > (t_bin[1] - dt_gap)):
            # if so, then calculate the resampled frequency
            t_count, t_bin_min = np.zeros(n_sample), np.array([0, dt_min])
            for i_sample in range(n_sample):
                # sets the new randomly sample time bins and determines the spikes within this reduced bin
                t_bin_rs = (t_bin[0] + rnd() * dt_gap) + t_bin_min
                t_count[i_sample] = sum(np.logical_and(t_sp > t_bin_rs[0], t_sp <= t_bin_rs[1]))

            # returns the average resampled time bin counts
            return np.mean(t_count)
        else:
            return len(t_sp)

    # initialises the RotationData class object (if not provided)
    if r_data is None:
        r_data = data.rotation

    # initialisations
    v_rng, w, is_full_rs = 80, np.pi / r_obj.t_phase[0][0], indD is None
    xi_bin0, xi_bin, t_bin, i_grp = calc_kinematic_bin_times([10, b_sz[0]], [90, v_rng], w, calc_type=1)
    n_vbin, sd_vel = np.size(xi_bin, axis=0), np.sign(np.diff(xi_bin0))

    # # sets up the
    # A, B = np.arange(0, v_rng + 1e-6, b_sz[0]), np.arange(v_rng, -(v_rng + 1e-6), -b_sz[0])
    # xi_bin = np.array(list(A) + list(B[1:-1]) + list(np.flip(-A, axis=0))).astype(int)
    # t_bin = np.array(list(np.arcsin(A / v_rng)) + list(np.pi - np.arcsin(B / v_rng)[1:-1]) + \
    #                  list(2 * np.pi - np.arcsin(np.flip(A, axis=0) / v_rng))) / w

    # sets the filter indices and the size of the comparison/smallest time bin
    if is_full_rs:
        # case is the full resampling calculations
        dt_min = np.min(np.diff(t_bin))
        ind_bin = range(len(t_bin) - 1)
        ind_filt = range(len(r_data.r_obj_kine.rot_filt_tot))
    else:
        # case is the single bin calculations
        _xi, ind_bin = -B[indD['ind_bin']:(indD['ind_bin'] + 2)].astype(int), []
        for i_m in np.where(xi_bin == _xi[0])[0]:
            if i_m > 0:
                if xi_bin[i_m - 1] == _xi[1]:
                    ind_bin.append(i_m - 1)

            if (i_m + 1) < len(xi_bin):
                if xi_bin[i_m + 1] == _xi[1]:
                    ind_bin.append(i_m)

        # sets the filter index and the bin resample duration
        ind_filt, dt_min = [indD['ind_filt']], (t_bin[ind_bin[0] + 1] - t_bin[ind_bin[0]] ) / 2.

    # memory allocation
    n_filt, f_keys = len(ind_filt), list(r_data.vel_sf_rs.keys())
    vel_f = np.empty(n_filt, dtype=object)

    # calculates the position/velocity for each filter type
    for i_filt in ind_filt:
        rr = r_obj.rot_filt_tot[i_filt]
        if is_full_rs:
            # if the values have already been calculated, then continue (total resampling only)
            if rr['t_type'][0] in r_data.vel_sf_rs:
                vel_f[i_filt] = r_data.vel_sf_rs[rr['t_type'][0]]
                continue

        # memory allocation for the current filter
        n_cell = np.size(r_obj.t_spike[i_filt], axis=0)
        n_trial_max = np.max([np.size(x, axis=0) for x in r_obj.wvm_para[i_filt]])

        # sets the experiment indices that belong to the current trial type
        tt = r_obj.rot_filt_tot[i_filt]['t_type'][0]
        valid_ind = np.where(valid_ind_func(data.cluster, tt))[0]
        is_md_expt = tt == 'MotorDrifting'

        # sets the cell indices to be analysed
        if is_full_rs:
            ind_cell = range(n_cell)
            w_str, w0 = 'Resampling Calculations ({0})'.format(tt), i_filt / n_filt
        else:
            ind_cell = [indD['ind_cell']]

        # calculates the position/velocity for each cell
        for ii, i_cell in enumerate(ind_cell):
            # updates the progress bar
            if is_full_rs:
                w_prog.emit(w_str, 100. * (w0 + (1. / n_filt) * (i_cell / n_cell)))

            # retrieves the experiment index, sampling frequency and phase duration
            i_expt = np.where(r_obj.i_expt[i_filt][i_cell] == valid_ind)[0][0]
            wvm_p = r_obj.wvm_para[i_filt][i_expt]

            # calculates the position/velocity for each of the trials
            t_phase = r_obj.t_phase[i_filt][i_expt]
            t_sp, n_trial_c, y_dir = r_obj.t_spike[i_filt][i_cell, :, :], np.size(wvm_p, axis=0), wvm_p['yDir']

            # memory allocation for the position/velocity bins (first cell only)
            if ii == 0:
                if is_full_rs:
                    # case is calculating the resampling for all cells
                    vel_f[i_filt] = np.empty((n_trial_max, n_vbin, n_cell, 2))
                else:
                    # case is calculating for a single cell
                    vel_f[i_filt] = np.empty(n_trial_max)

                # sets all values to NaNs
                vel_f[i_filt][:] = np.nan

            # memory allocation
            vel_bin = np.zeros((n_trial_c, n_vbin, 2))
            for i_trial in range(n_trial_c):
                # sets the spike times in the correct order for the current trial
                if t_sp[i_trial, 1] is None:
                    continue

                if (y_dir[i_trial] == -1 and not is_md_expt) or (y_dir[i_trial] == 1 and is_md_expt):
                    # case is a CW => CCW trial
                    t_sp_trial, m = np.concatenate((t_sp[i_trial, 1], t_sp[i_trial, 2] + t_phase)), -1
                else:
                    # case is a CCW => CW trial
                    t_sp_trial, m = np.concatenate((t_sp[i_trial, 2], t_sp[i_trial, 1] + t_phase)), 1

                # if there are no spikes for the current trial, then continue
                if len(t_sp_trial) == 0:
                    continue

                # sets the bin indices and allocates memory for the bin spike counts
                n_sp_bin = np.zeros(len(t_bin))
                for i_bin in ind_bin:
                    # determines the indices of the time spikes within the current time bin
                    i_sp_bin = np.logical_and(t_sp_trial > t_bin[i_bin], t_sp_trial <= t_bin[i_bin + 1])
                    if np.any(i_sp_bin):
                        # if there are any
                        n_sp_bin[i_bin] = calc_resampled_counts(
                            t_sp_trial[i_sp_bin], t_bin[i_bin:i_bin + 2], dt_min, n_sample
                        )

                # sets the position/velocity values
                vel_bin_tmp = reorder_array(n_sp_bin, i_grp, sd_vel, m=-m, dtype=float)
                for k in range(2):
                    vel_bin[i_trial, :, k] = vel_bin_tmp[k, :]

                # # sets the groupings for each time bins
                # i_grp, ind_inv = np.unique(
                #     np.sort(np.vstack([-y_dir[i_trial] * xi_bin[i:i + 2] for i in range(len(xi_bin) - 1)]),
                #     axis=1), axis=0, return_inverse=True
                # )

            # calculates the position/velocity spiking frequencies over all trials
            for k in range(2):
                if is_full_rs:
                    # case is setting all spiking frequencies

                    # sets the trial indices
                    if k == 0:
                        ind_t = np.array(range(n_trial_c))

                    # sets the full velocity/position spiking rates
                    vel_f[i_filt][ind_t, :, i_cell, k] = vel_bin[:, :, k] / dt_min
                else:
                    # case is calculating the average spiking frequency
                    a = 1
                    # pos_f[i_filt][i_cell, :, k] = np.mean(pos_bin[:, :, k], axis=0) / pos_dt
                    # vel_f[i_filt][i_cell, :, k] = np.mean(vel_bin[:, :, k], axis=0) / vel_dt

            # #
            # if is_full_rs:
            #     for i_bin in range(np.size(vel_n[i_filt], axis=1)):
            #         vel_n[i_filt][i_trial, :, i_cell] = np.array(
            #             [np.sum(n_sp_bin[ind_inv == i]) for i in range(np.size(vel_n[i_filt], axis=1))])
            # else:
            #     vel_n[i_filt][i_trial] = np.mean(n_sp_bin[np.array(ind_bin)])

    # returns the values
    if is_full_rs:
        return vel_f, xi_bin
    else:
        return vel_f


def calc_kinemetic_spike_freq(data, r_obj, b_sz, calc_type=2):
    '''

    :param wvm_para:
    :param s_freq:
    :param i_expt:
    :return:
    '''

    # initialisations and memory allocation
    k_rng, n_filt, calc_avg_sf = [90, 80], len(r_obj.t_spike), calc_type == 2
    t_bin = np.zeros(2, dtype=object)
    pos_f, vel_f = np.empty(n_filt, dtype=object), np.empty(n_filt, dtype=object)

    # calculates the position/velocity for each filter type
    for i_filt in range(n_filt):
        # memory allocation for the current filter
        n_cell = np.size(r_obj.t_spike[i_filt], axis=0)
        n_trial_max = np.max([np.size(x, axis=0) for x in r_obj.wvm_para[i_filt]])

        # determines the experiments which contain the trial type
        tt = r_obj.rot_filt_tot[i_filt]['t_type'][0]
        is_md_expt = tt == 'MotorDrifting'
        valid_ind = np.where(valid_ind_func(data.cluster, tt))[0]

        # calculates the position/velocity for each cell
        for i_cell in range(n_cell):
            # retrieves the experiment index, sampling frequency and phase duration
            if r_obj.is_single_cell:
                i_expt = 0
            else:
                i_expt = np.where(r_obj.i_expt[i_filt][i_cell] == valid_ind)[0][0]

            # calculates the position/velocity for each of the trials
            t_phase, wvm_p = r_obj.t_phase[i_filt][i_expt], r_obj.wvm_para[i_filt][i_expt]
            y_dir, y_amp = wvm_p['yDir'], wvm_p['yAmp'][0]
            t_sp, w, n_trial_c = r_obj.t_spike[i_filt][i_cell, :, :], np.pi / t_phase, np.size(wvm_p, axis=0)

            # calculates the time-bins (only required for first iteration pass)
            if (i_filt == 0) and (i_cell == 0):
                # retrieves the time/value bins and index groupings
                xi_bin0, xi_bin, t_bin, i_grp = calc_kinematic_bin_times(b_sz, k_rng, w)

                # memory allocation
                n_pbin, n_vbin = np.size(xi_bin[0], axis=0), np.size(xi_bin[1], axis=0)
                sd_pos, sd_vel = np.sign(np.diff(xi_bin0[0])), np.sign(np.diff(xi_bin0[1]))

                # calculates the position/velocity time bin durations
                pos_dt = reorder_array(np.diff(t_bin[0]), i_grp[0], sd_pos, dtype=float)[0, :]
                vel_dt = reorder_array(np.diff(t_bin[1]), i_grp[1], sd_vel, dtype=float)[0, :]

            # memory allocation for the position/velocity bins (first cell only)
            if i_cell == 0:
                # allocates memory for the inner arrays (dependent on type)
                if calc_avg_sf:
                    # case is calculating the average spiking frequency
                    pos_f[i_filt] = np.zeros((n_cell, n_pbin, 2))
                    vel_f[i_filt] = np.zeros((n_cell, n_vbin, 2))
                else:
                    # case is calculating individual spiking frequencies
                    pos_f[i_filt] = np.empty((n_trial_max, n_pbin, n_cell, 2))
                    vel_f[i_filt] = np.empty((n_trial_max, n_vbin, n_cell, 2))
                    vel_f[i_filt][:], pos_f[i_filt][:] = np.nan, np.nan

            # memory allocation for the position/velocity bins
            pos_bin = np.zeros((n_trial_c, n_pbin, 2))
            vel_bin = np.zeros((n_trial_c, n_vbin, 2))

            for i_trial in range(n_trial_c):
                # sets the spike times in the correct order for the current trial
                if t_sp[i_trial, 1] is None:
                    continue

                if (y_dir[i_trial] == -1 and not is_md_expt) or (y_dir[i_trial] == 1 and is_md_expt):
                    # case is a CW => CCW trial
                    t_sp_trial, m = np.concatenate((t_sp[i_trial, 1], t_sp[i_trial, 2] + t_phase)), -1
                else:
                    # case is a CCW => CW trial
                    t_sp_trial, m = np.concatenate((t_sp[i_trial, 2], t_sp[i_trial, 1] + t_phase)), 1

                # calculates the counts for each of the time bins
                if len(t_sp_trial):
                    # calculates the position/velocity histogram bin values
                    pos_bin_tmp = reorder_array(np.histogram(t_sp_trial, bins=t_bin[0])[0], i_grp[0], sd_pos, m=m)
                    vel_bin_tmp = reorder_array(np.histogram(t_sp_trial, bins=t_bin[1])[0], i_grp[1], sd_vel, m=-m)

                    # h_vel = np.arange(64)
                    # vel_bin_tmp = reorder_array(h_vel, i_grp[1], sd_vel)

                    # sets the position/velocity values
                    for k in range(2):
                        pos_bin[i_trial, :, k] = pos_bin_tmp[k, :]
                        vel_bin[i_trial, :, k] = vel_bin_tmp[k, :]

            # calculates the position/velocity spiking frequencies over all trials
            for k in range(2):
                if calc_avg_sf:
                    # case is calculating the average spiking frequency
                    pos_f[i_filt][i_cell, :, k] = np.mean(pos_bin[:, :, k], axis=0) / pos_dt
                    vel_f[i_filt][i_cell, :, k] = np.mean(vel_bin[:, :, k], axis=0) / vel_dt
                else:
                    # case is setting all spiking frequencies

                    # sets the trial indices
                    if k == 0:
                        ind_t = np.array(range(n_trial_c))
                        _pos_dt = np.matlib.repmat(dcopy(pos_dt), len(ind_t), 1)
                        _vel_dt = np.matlib.repmat(dcopy(vel_dt), len(ind_t), 1)

                    # sets the full velocity/position spiking rates
                    pos_f[i_filt][ind_t, :, i_cell, k] = pos_bin[:, :, k] / _pos_dt
                    vel_f[i_filt][ind_t, :, i_cell, k] = vel_bin[:, :, k] / _vel_dt

    # returns the position/velocity spiking frequency arrays
    if calc_type == 0:
        # calculation type is position data only
        return pos_f, xi_bin[0]
    elif calc_type == 1:
        # calculation type is velocity data only
        return vel_f, xi_bin[1]
    elif calc_type == 2:
        # calculation type is both kinematic types
        return [pos_f, vel_f], xi_bin
    else:
        # calculation type is both kinematic types (but non-averaged values)
        return [pos_f, vel_f], xi_bin


def reorder_array(h, i_grp, sd, dtype=int, m=1):
    '''

    :param y0:
    :param j_grp:
    :return:
    '''

    # memory allocation
    nH, i_row = int(len(i_grp) / 2), ((sd + 1) / 2).astype(int)
    y_arr = -np.ones((2, nH), dtype=dtype)

    # sets the ordering of the index array
    if m == 1:
        # index array is in the correct order
        j_grp = dcopy(i_grp)
    else:
        # index array needs to be reversed
        j_grp = max(i_grp) - dcopy(i_grp)
        i_row = 1 - i_row

    # sets the values into the array
    for i in range(len(j_grp)):
        y_arr[i_row[i], j_grp[i]] = h[i]

    # sets the ordered values and return the final array
    return y_arr


# def calc_wave_kinematic_times(wvm_para, s_freq, i_expt, kb_sz, is_pos, yDir=1):
#     '''
#
#     :param wvm_para:
#     :param kb_sz:
#     :return:
#     '''
#
#     # initialisations
#     yAmp, tPeriod = wvm_para[int(i_expt)]['yAmp'], wvm_para[int(i_expt)]['tPeriod']
#     w, A = 2 * np.pi * s_freq / tPeriod, yAmp * yDir
#
#     # if velocity, then convert the amplitude from position to speed
#     if not is_pos:
#         yAmp = np.ceil(yAmp * w)
#
#     # calculates the number of bins for half the full sinusoidal wave
#     n_bin = yAmp / kb_sz
#     # if (n_bin - np.round(n_bin)) > 1e-6:
#     #     # if a non-integer number of bins are being created, then exit with an error output to screen
#     #     e_str = 'The discretisation bin size does not produce an even number of bins:\n\n' \
#     #             '  => Wave Amplitude = {0}\n  => Bin Size = {1}\n'.format(yAmp, kb_sz)
#     #     e_str = e_str + '  => Bin Count = {:4.1f}\n\nAlter the discretisation bin size until an ' \
#     #                     'even number of bins is achieved.'.format(2 * n_bin)
#     #     cf.show_error(e_str, 'Incorrect Discretisation Bin Size')
#     #
#     #     # returns none values
#     #     return None, None, None
#     # else:
#
#     # otherwise, initialise the angle bin array (repeats for the full sinusoid period)
#     xi_bin = np.linspace(-yAmp, yAmp, 1 + np.ceil(2 * n_bin))
#     xi_bin_tot = np.hstack((xi_bin, xi_bin))
#
#     # calculates the time offsets for each of the bins
#     if is_pos:
#         # case is position is being considered
#         phi = np.array([m.acos(ss_scale(x / A)) for x in xi_bin])
#         phi_tot = np.hstack((phi, 2 * np.pi - phi))
#     else:
#         # case is speed is being considered
#         phi = np.array([m.acos(ss_scale(-v / (w * A))) for v in xi_bin])
#
#         # converts the angles to the arc-sin range (from the arc-cosine range) and ensures all angles are positive
#         phi_tot = np.hstack((np.pi / 2 - phi, phi - 3 * np.pi / 2))
#         phi_tot[phi_tot < 0] = 2 * np.pi + phi_tot[phi_tot< 0]
#
#     # sorts the angle bins in chronological order
#     i_sort = np.argsort(phi_tot)
#     xi_bin_tot, phi_tot = xi_bin_tot[i_sort], phi_tot[i_sort]
#
#     # ensures the start/end of the discretisation bins are the same
#     if np.abs(xi_bin_tot[0] - xi_bin_tot[-1]) > 1e-6:
#         xi_bin_tot = np.append(xi_bin_tot, xi_bin_tot[0])
#         phi_tot = np.append(phi_tot, 2 * np.pi)
#
#     # removes any repeats
#     is_ok = np.array([True] + list(np.diff(xi_bin_tot) != 0))
#     xi_bin_tot, t_bin = xi_bin_tot[is_ok], phi_tot[is_ok] / w
#
#     # returns the bin discretisation, bin time points, and the the duration of the CC/CCW phases
#     return xi_bin_tot, t_bin, tPeriod / (s_freq * 2.0)