# module import
import copy
import peakutils
import math as m
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil import parser
from collections import OrderedDict

# custom module import
import analysis_guis.calc_functions as cfcn
import analysis_guis.common_func as cf
from pyphys.pyphys.pyphys import PxpParser
from analysis_guis.dialogs import config_dialog
from rotation_analysis.analysis.probe.probe_io.probe_io import TriggerTraceIo, BonsaiIo, IgorIo

#
import analysis_guis.testing.diagnostic_plots as diag_plot
import _pickle as cp

# pyqt5 module import
from PyQt5.QtWidgets import (QMessageBox)

# lambda function declarations
date2sec = lambda t: np.sum([3600 * t.hour, 60 * t.minute, t.second])
trig_count = lambda data, cond: len(np.where(np.diff(data[cond]['cpg_ttlStim']) > 1)[0]) + 1
ss_scale = lambda x: np.min([np.max([x, -1.0]), 1.0])

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
    grp_lim = [-1] + list(np.where(d2t_bonsai > 60)[0] + 1) + [len(d2t_bonsai) + 1]
    ind_grp = [np.arange(grp_lim[x] + 1, grp_lim[x + 1] + 1) for x in range(len(grp_lim) - 1)]

    # sets the time, name and trigger count from each of these groups
    t_bonsai_grp = [t_bonsai_sec[x[0]] for x in ind_grp]
    c_bonsai_grp = [bonsai_io.data['Condition'][x[0]] for x in ind_grp]
    n_trig_bonsai = [len(x) for x in ind_grp]

    # determines the feasible variables from the igor data file
    var_keys = list(igor_data.data.keys())
    is_ok = ['command' in igor_data.data[x].keys() if isinstance(igor_data.data[x], OrderedDict) else False for x in var_keys]

    # sets the name, time and trigger count from each of the igor trial groups
    c_igor_grp = [y for x, y in zip(is_ok, var_keys) if x]
    t_igor_grp, t_igor_str, n_trig_igor = [], [], [trig_count(igor_data.data, x) for x in c_igor_grp]
    has_ft = np.zeros(len(c_igor_grp), dtype=bool)

    for ick, ck in enumerate(c_igor_grp):
        if file_time_key in igor_data.data[ck]['vars']:
            has_ft[ick] = True
            t_igor_str_nw = igor_data.data[ck]['vars'][file_time_key][0]
            t_igor_str.append(t_igor_str_nw)
            t_igor_grp.append(date2sec(datetime.strptime(t_igor_str_nw, '%H:%M:%S').time()))

    # calculates the point-wise differences between the trial timer and trigger count
    n_trig_igor = [np.array(n_trig_igor)[has_ft]]
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
    kk = np.where(has_ft)[0]
    i2b_key, x = {}, np.array(c_igor_grp)[i2b]
    for cc in c_bonsai_grp:
        if cc not in i2b_key:
            jj = np.where([x == cc for x in c_bonsai_grp])[0]
            i2b_key[cc] = x[jj]

    #
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
            tt_io.n_triggers, bonsai_io.n_triggers
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
        cfig_dlg = config_dialog.ConfigDialog(dlg_info=dlg_info, title=title, width=400,
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

    #
    w_form0 = np.array([upsample_waveform(igor_io.get_waveforms_in_condition_group([x]), s_freq / y) for x, y in zip(tt_igor, s_rate)])
    i_grp = [np.array([list(tt_igor).index(y) for y in i2b_key[x]]) for x in trial_type]
    w_form = [sum(list(w_form0[ig]), []) for ig in i_grp]

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

    # with open('TestStimTiming.p', 'wb') as fw:
    #     cp.dump(rot_info, fw)
    #     cp.dump(w_form, fw)
    #     cp.dump(igor_io, fw)
    #     cp.dump(tt_io, fw)

    # if is_diagnostic:
    #     diag_plot.plot_stim_phase_timing(rot_info, w_form, t_type='Black', p_type='Full')

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
        t_type = np.unique(cf.flat_list([x['rotInfo']['trial_type'] for x in d_clust]))
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


def apply_rot_filter(data, rot_filt, is_multi_cell, exp_name):
    '''

    :param data:
    :param rot_filt:
    :return:
    '''

    # retrieves the data clusters for each of the valid rotation experiments
    is_rot_expt = cf.det_valid_rotation_expt(data)
    d_clust = [x for x, y in zip(data.cluster, is_rot_expt) if y]

    #
    if exp_name is None:
        i_expt_match = None
    else:
        i_expt_match = cf.get_expt_index(exp_name, data.cluster, cf.det_valid_rotation_expt(data))

    # sets up the filter permutation array
    rot_filt_p, f_perm, f_key = setup_filter_permutations(d_clust, copy.deepcopy(rot_filt))

    # memory allocation
    n_filt, d_copy = len(rot_filt_p), copy.deepcopy
    A = np.empty(n_filt, dtype=object)
    t_spike, wvm_para, trial_ind, clust_ind, i_expt = d_copy(A), d_copy(A), d_copy(A), d_copy(A), d_copy(A)

    #
    for i_filt in range(n_filt):
        t_spike[i_filt], wvm_para[i_filt], trial_ind[i_filt], clust_ind[i_filt], i_expt[i_filt] = \
                        apply_single_rot_filter(data, d_clust, rot_filt_p[i_filt], is_multi_cell, i_expt_match)

    # returns the spike time/waveform parameter/filter parameter arrays
    return t_spike, wvm_para, trial_ind, clust_ind, i_expt, f_perm, f_key, rot_filt_p


def apply_single_rot_filter(data, d_clust, rot_filt, is_multi_cell, i_expt_match):
    '''

    :param data:
    :param rot_filt:
    :return:
    '''

    # sets the trial types
    if rot_filt['is_ud'][0]:
        t_type, is_ud = ['UniformDrifting'], True
    else:
        t_type, is_ud = rot_filt['t_type'], False

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
        if (d_clust[i_expt]['rotInfo'] is not None) and (i_expt == i_expt_match if i_expt_match is not None else True):
            # case is the rotational information has been set
            is_ok[i_expt] = any([x in d_clust[i_expt]['rotInfo']['trial_type'] for x in rot_filt['t_type']])
            if not is_ok[i_expt]:
                continue

            for c_type in chk_str:
                # if there is a specific case being searched, then search all the entries for a match
                if rot_filt[c_type][0] != 'All':
                    is_ok[i_expt] = d_clust[i_expt]['expInfo'][c_type] in rot_filt[c_type]

    ####################################
    ####    TRIAL-TYPE FILTERING    ####
    ####################################

    #### retrieves the rotational data for the experiment trial types

    # splits the spike times/waveform parameters by the trial type
    for i_expt in range(n_expt):
        # only store values if the experiment is a valid rotation analysis data file
        if is_ok[i_expt]:
            # retrieves the spike counts for the
            t_spike_expt = d_clust[i_expt]['rotInfo']['t_spike']
            wfm_para_expt = d_clust[i_expt]['rotInfo']['wfm_para']

            # stores the spike count/waveform parameters for each trial type that matches the condition strings
            tt_expt = d_clust[i_expt]['rotInfo']['trial_type']
            if t_type[0] in tt_expt:
                t_spike[i_expt] = t_spike_expt[t_type[0]]
                wfm_para[i_expt] = wfm_para_expt[t_type[0]]
            else:
                is_ok[i_expt] = False

    ############################################
    ####    CELL-CLUSTER BASED FILTERING    ####
    ############################################

    # sets the filter strings
    cc_filt_str = ['sig_type', 'match_type', 'region_name', 'record_layer', 'lesion_type']
    is_check = [(data.classify.class_set and is_multi_cell), (data.comp.is_set and is_multi_cell),
                is_multi_cell, is_multi_cell, is_multi_cell]

    # if uniform drifting, add on the ended parameter search
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
                if is_check[iccf] and (rot_filt[ccf][0] != 'All'):
                    ind_cl_nw, ind_tr_nw = None, None
                    if ccf == 'sig_type':
                        ind_cl_nw = [np.any([x in y for y in rot_filt[ccf]]) for x in data.classify.grp_str[i_expt]]

                    elif ccf == 'match_type':
                        m_flag = (rot_filt[ccf][0].split(' ')[0] != 'Matched')
                        ind_cl_nw = list(np.where(data.comp.is_accept == m_flag)[0])

                    elif ccf == 'region_name':
                        ind_cl_nw = [x == rot_filt[ccf][0] for x in d_clust[i_expt]['chRegion']]

                    elif ccf == 'record_layer':
                        ind_cl_nw = [x == rot_filt[ccf][0] for x in d_clust[i_expt]['chLayer']]

                    elif ccf == 'lesion_type':
                        ind_cl_nw = [x == rot_filt[ccf][0] for x in d_clust[i_expt]['chLayer']]

                    elif ccf == 't_freq':
                        ind_tr_nw = np.abs(wfm_para[i_expt]['tFreq'] - float(rot_filt[ccf][0])) < 1e-6

                    elif ccf == 't_freq_dir':
                        ind_tr_nw = np.abs(wfm_para[i_expt]['yDir'] - float(rot_filt[ccf][0])) < 1e-6

                    elif ccf == 't_cycle':
                        ind_tr_nw = np.abs(wfm_para[i_expt]['tCycle'] - float(rot_filt[ccf][0])) < 1e-6

                    # removes any infeasible clusters
                    if ind_cl_nw is not None:
                        ind_cl = np.logical_and(ind_cl, np.array(ind_cl_nw))

                    # removes any infeasible trials
                    if ind_tr_nw is not None:
                        ind_tr = np.logical_and(ind_tr, np.array(ind_tr_nw))

            # removes the infeasible cluster rows/trial columns
            t_spike[i_expt] = t_spike[i_expt][ind_cl, :, :]
            t_spike[i_expt] = t_spike[i_expt][:, ind_tr, :]

            # removes the infeasible trial/cluster indices from the other arrays
            wfm_para[i_expt] = wfm_para[i_expt][ind_tr]
            trial_ind[i_expt] = np.where(ind_tr)[0]
            clust_ind[i_expt] = np.where(ind_cl)[0]

    # returns the spike times/waveform parameters
    return t_spike[is_ok], wfm_para[is_ok], trial_ind[is_ok], clust_ind[is_ok], np.where(is_ok)[0]


def calc_wave_kinematic_times(wvm_para, s_freq, i_expt, kb_sz, is_pos, yDir=1):
    '''

    :param wvm_para:
    :param kb_sz:
    :return:
    '''

    # initialisations
    yAmp, tPeriod = wvm_para[int(i_expt)]['yAmp'], wvm_para[int(i_expt)]['tPeriod']
    w, A = 2 * np.pi * s_freq / tPeriod, yAmp * yDir

    # if velocity, then convert the amplitude from position to speed
    if not is_pos:
        yAmp = np.ceil(yAmp * w)

    # calculates the number of bins for half the full sinusoidal wave
    n_bin = yAmp / kb_sz
    if (n_bin - np.round(n_bin)) > 1e-6:
        # if a non-integer number of bins are being created, then exit with an error output to screen
        e_str = 'The discretisation bin size does not produce an even number of bins:\n\n' \
                '  => Wave Amplitude = {0}\n  => Bin Size = {1}\n'.format(yAmp, kb_sz)
        e_str = e_str + '  => Bin Count = {:4.1f}\n\nAlter the discretisation bin size until an ' \
                        'even number of bins is achieved.'.format(2 * n_bin)
        cf.show_error(e_str, 'Incorrect Discretisation Bin Size')

        # returns none values
        return None, None, None
    else:
        # otherwise, initialise the angle bin array (repeats for the full sinusoid period)
        xi_bin = np.linspace(-yAmp, yAmp, 1 + np.ceil(2 * n_bin))
        xi_bin_tot = np.hstack((xi_bin, xi_bin))

    # calculates the time offsets for each of the bins
    if is_pos:
        # case is position is being considered
        phi = np.array([m.acos(ss_scale(x / A)) for x in xi_bin])
        phi_tot = np.hstack((phi, 2 * np.pi - phi))
    else:
        # case is speed is being considered
        phi = np.array([m.acos(ss_scale(-v / (w * A))) for v in xi_bin])

        # converts the angles to the arc-sin range (from the arc-cosine range) and ensures all angles are positive
        phi_tot = np.hstack((np.pi / 2 - phi, phi - 3 * np.pi / 2))
        phi_tot[phi_tot < 0] = 2 * np.pi + phi_tot[phi_tot< 0]

    # sorts the angle bins in chronological order
    i_sort = np.argsort(phi_tot)
    xi_bin_tot, phi_tot = xi_bin_tot[i_sort], phi_tot[i_sort]

    # ensures the start/end of the discretisation bins are the same
    if np.abs(xi_bin_tot[0] - xi_bin_tot[-1]) > 1e-6:
        xi_bin_tot = np.append(xi_bin_tot, xi_bin_tot[0])
        phi_tot = np.append(phi_tot, 2 * np.pi)

    # removes any repeats
    is_ok = np.array([True] + list(np.diff(xi_bin_tot) != 0))
    xi_bin_tot, t_bin = xi_bin_tot[is_ok], phi_tot[is_ok] / w

    # returns the bin discretisation, bin time points, and the the duration of the CC/CCW phases
    return xi_bin_tot, t_bin, tPeriod / (s_freq * 2.0)