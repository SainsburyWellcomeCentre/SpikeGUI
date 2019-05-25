# module import
import gc
import os
import copy
import random
import numpy as np
import pickle as p
import pandas as pd
import multiprocessing as mp
from numpy.matlib import repmat

# scipy module imports
from scipy.spatial.distance import *
from scipy.interpolate import PchipInterpolator as pchip
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

# pyqt5 module import
from PyQt5.QtCore import QThread, pyqtSignal

# custom module imports
import analysis_guis.common_func as cf
import analysis_guis.calc_functions as cfcn
import analysis_guis.rotational_analysis as rot
from analysis_guis.dialogs.rotation_filter import RotationFilteredData
from analysis_guis.cluster_read import ClusterRead
from probez.spike_handling import spike_io

# other parameters
dcopy = copy.deepcopy
default_dir_file = os.path.join(os.getcwd(), 'default_dir.p')

########################################################################################################################
########################################################################################################################

class WorkerThread(QThread):
    # creates the signal object
    work_started = pyqtSignal()
    work_progress = pyqtSignal(str, float)
    work_finished = pyqtSignal(object)
    work_error = pyqtSignal(str, str)
    work_plot = pyqtSignal(object)

    def __init__(self, parent=None, main_gui=None):
        # creates the worker object
        super(WorkerThread, self).__init__(parent)

        self.update_pbar = True
        self.is_running = False
        self.forced_quit = False
        self.sub_job = None
        self.is_ok = True
        self.data = None

        # other initialisations
        self.main_gui = main_gui
        self.thread_job_primary = None
        self.thread_job_secondary = None
        self.thread_job_para = None

    def set_worker_func_type(self, thread_job_primary, thread_job_secondary=None, thread_job_para=None):
        '''

        :param func_type:
        :return:
        '''

        # updates the worker primary/secondary job type and parameters
        self.thread_job_primary = thread_job_primary
        self.thread_job_secondary = thread_job_secondary
        self.thread_job_para = thread_job_para

    def run(self):
        '''

        :return:
        '''

        # updates the running/forced quit flagsv
        self.is_running = True
        self.forced_quit = False
        self.is_ok = True

        # updates the running parameter and enables the progress group parameters
        self.work_started.emit()

        # runs the job based on the type
        thread_data = None
        if self.thread_job_primary == 'init_data_file':
            # case is initialising the data file
            self.init_cluster_data()

        elif self.thread_job_primary == 'init_pool_object':
            # case is initialising the pool worker object
            thread_data = self.init_pool_worker()

        elif self.thread_job_primary == 'load_data_files':
            # case is loading the data files
            thread_data = self.load_data_file()

        elif self.thread_job_primary == 'cluster_matches':
            # case is determining the cluster matches
            thread_data = self.det_cluster_matches()

        elif self.thread_job_primary == 'run_calc_func':
            # case is the calculation functions
            calc_para, plot_para = self.thread_job_para[0], self.thread_job_para[1]
            data, pool = self.thread_job_para[2], self.thread_job_para[3]

            if self.thread_job_secondary == 'Cluster Cross-Correlogram':
                # case is the cc-gram type determinations
                thread_data = self.calc_ccgram_types(calc_para, data.cluster)

            elif self.thread_job_secondary == 'Shuffled Cluster Distances':
                # case is the shuffled cluster distances
                thread_data = self.calc_shuffled_cluster_dist(calc_para, data.cluster)

            elif self.thread_job_secondary == 'Direction ROC Curves (Single Cell)':
                # case is the shuffled cluster distances
                self.calc_cond_roc_curves(data, pool, calc_para, plot_para, False, 100.)

            elif self.thread_job_secondary == 'Direction ROC Curves (Whole Experiment)':
                if plot_para['use_resp_grp_type']:
                    if not self.calc_dirsel_group_types(data, calc_para, plot_para):
                        self.is_ok = False

                # calculates the phase roc-curves for each cell
                self.calc_cond_roc_curves(data, pool, calc_para, plot_para, False, 33.)
                self.calc_phase_roc_curves(data, 33.)
                self.calc_phase_roc_significance(calc_para, data, pool, 66.)

            elif self.thread_job_secondary == 'Velocity ROC Curves (Single Cell)':
                # calculates the binned kinematic spike frequencies
                self.calc_binned_kinemetic_spike_freq(data, plot_para, calc_para)
                self.calc_kinematic_roc_curves(data, pool, calc_para, 50.)

            elif self.thread_job_secondary == 'Velocity ROC Curves (Whole Experiment)':
                # calculates the binned kinematic spike frequencies
                self.calc_binned_kinemetic_spike_freq(data, plot_para, calc_para)
                self.calc_kinematic_roc_curves(data, pool, calc_para, 50.)

            elif self.thread_job_secondary == 'Velocity ROC Curves (Pos/Neg Comparison)':
                # calculates the binned kinematic spike frequencies
                self.calc_binned_kinemetic_spike_freq(data, plot_para, calc_para)
                self.calc_kinematic_roc_curves(data, pool, calc_para, 50.)

            elif self.thread_job_secondary == 'Condition ROC Curve Comparison':
                # calculates the phase roc-curves for each cell
                self.calc_cond_roc_curves(data, pool, calc_para, plot_para, True, 33.)
                self.calc_phase_roc_curves(data, 33.)
                self.calc_phase_roc_significance(calc_para, data, pool, 66.)

            elif self.thread_job_secondary == 'Motion/Direction Selectivity Cell Grouping Scatterplot':
                if plot_para['use_resp_grp_type']:
                    if not self.calc_dirsel_group_types(data, calc_para, plot_para):
                        self.is_ok = False

                # calculates the phase roc-curves for each cell
                self.calc_cond_roc_curves(data, pool, calc_para, plot_para, True, 33.)
                self.calc_phase_roc_curves(data, 33.)
                self.calc_phase_roc_significance(calc_para, data, pool, 66.)

            elif self.thread_job_secondary == 'Rotation/Visual Stimuli Response Statistics':
                # calculates the direction/selection group types
                if not self.calc_dirsel_group_types(data, calc_para, plot_para):
                    self.is_ok = False

            # elif self.thread_job_secondary == 'Kinematic Spiking Frequency':
            #     # calculates the binned kinematic spike frequencies
            #     self.calc_binned_kinemetic_spike_freq(data, plot_para, calc_para, False)

        elif self.thread_job_primary == 'update_plot':
            pass

        # emits the finished work signal
        self.work_finished.emit(thread_data)

    ############################################
    ####    THREAD CALCULATION FUNCTIONS    ####
    ############################################

    def load_data_file(self):
        '''

        :param exp_file:
        :return:
        '''

        # retrieves the job parameters
        load_dlg, loaded_exp, is_multi = self.thread_job_para[0], self.thread_job_para[1], self.thread_job_para[2]
        if not np.any([not x in loaded_exp for x in load_dlg.exp_name]):
            # if there are no new experiments to load, then exit the function
            return None
        else:
            n_file = len(load_dlg.exp_files)
            dpw, p_rlx, data = 1.0 / n_file, 0.05, []

        #
        for i_file in range(n_file):
            if not self.is_running:
                # if the user cancelled, then exit
                return None
            else:
                # updates the progress bar string
                p_str, pw0 = 'Loading File {0} of {1}'.format(i_file+1, n_file), i_file / n_file
                self.work_progress.emit(p_str, 100.0 * pw0)

            # sets the experiment file and name
            if load_dlg.exp_name[i_file] not in loaded_exp:
                # loads the data from the data file
                with open(load_dlg.exp_files[i_file], 'rb') as fp:
                    data_nw = p.load(fp)

                # setting of other fields
                data_nw['expFile'] = load_dlg.exp_files[i_file]

                # re-calculates the signal features (single experiment only)
                if not is_multi:
                    # memory allocation for the signal features
                    xi = np.array(range(data_nw['nPts']))
                    sFeat = np.zeros((data_nw['nC'], 2))

                    for i in range(data_nw['nC']):
                        # creates the piecewise-polynomial of the mean signal
                        pp, t_max = pchip(xi, data_nw['vMu'][:, i]), data_nw['sigFeat'][i, 2]
                        t_min = np.argmin(data_nw['vMu'][int(t_max):, i]) + t_max
                        v_max_2 = data_nw['vMu'][int(t_max), i] / 2.0
                        v_min = np.min(data_nw['vMu'][int(t_max):, i])
                        v_half = data_nw['vMu'][int(data_nw['sigFeat'][i, 1]), i] / 2.0

                        ##################################################
                        ####    POST-STIMULI SPIKE HALF-WIDTH TIME    ####
                        ##################################################

                        # determines the point/voltage of the pmaximum proceding the minimum
                        bnd_1 = [(data_nw['sigFeat'][i, 0], data_nw['sigFeat'][i, 1])]
                        bnd_2 = [(data_nw['sigFeat'][i, 1], data_nw['sigFeat'][i, 2])]
                        bnd_3 = [(data_nw['sigFeat'][i, 2], t_min)]

                        # determines the location of the half-width points
                        t_hw1_lo = cfcn.opt_time_to_y0((pp, v_half), bnd_1)
                        t_hw1_hi = cfcn.opt_time_to_y0((pp, v_half), bnd_2)
                        t_hw2_lo = cfcn.opt_time_to_y0((pp, v_max_2), bnd_2)
                        t_hw2_hi = cfcn.opt_time_to_y0((pp, v_max_2), bnd_3)
                        t_rlx = cfcn.opt_time_to_y0((pp, v_min + p_rlx * (v_max_2 - v_min)), bnd_3)

                        # determine if it is feasible to find the 2nd peak half-width point
                        if (t_hw2_hi is None) or (t_rlx is None):
                            # if not, then linearly extrapolate past the end point of the signal
                            xi2 = np.array(range(2*xi[-1]))
                            ppL = IUS(xi, data_nw['vMu'][:, i], k=1)

                            # determines the half-width/relaxtion time from the extrapolated signal
                            bnd_4 = [(data_nw['sigFeat'][i, 2], xi2[-1])]
                            t_hw2_hi = cfcn.opt_time_to_y0((ppL, v_max_2), bnd_4)
                            t_rlx = cfcn.opt_time_to_y0((ppL, v_min + p_rlx * (v_max_2 - v_min)), bnd_4)

                        # calculates the new signal features
                        data_nw['sigFeat'][i, 3] = t_hw1_lo
                        data_nw['sigFeat'][i, 4] = t_hw1_hi
                        sFeat[i, 0] = t_hw2_hi - t_hw2_lo
                        sFeat[i, 1] = t_rlx - t_max

                    # concatenates the new signal feature date
                    data_nw['sigFeat'] = np.concatenate((data_nw['sigFeat'], sFeat), axis=1)
                    data_nw['expInfo']['clInclude'] = np.ones(data_nw['nC'], dtype=bool)

                # appends the new data dictionary to the overall data list
                data.append(data_nw)

        # appends the current filename to the data dictionary and returns the object
        return data

    def init_pool_worker(self):
        '''

        :return:
        '''

        # creates the pool worker object
        p = mp.Pool(int(np.floor(1.5 * mp.cpu_count())))

        # returns the object
        return p

    def init_cluster_data(self):
        '''

        :return:
        '''

        def map_cluster_depths():
            '''

            :param cluster_depth:
            :return:
            '''

            # retrieves the cluster depths from the spike I/O class object
            cluster_depth = sp_io.get_cluster_depths(cluster_ids)

            # sets the mapped cluster depths based on the file type
            if (exp_info['dmapFile'] is None) or (len(exp_info['dmapFile']) == 0):
                # no map is given so return the original depth values
                return cluster_depth, None
            else:
                # otherwise, map the cluster depth values from the probe to actual values
                data = np.array(pd.read_csv(exp_info['dmapFile']))
                if np.size(data, axis=1) < 4:
                    # if the mapping file is not correct, then output an error to screen
                    e_str = 'Channel mapping file does not have the correct format.\n\n' \
                            'Re-select a valid file before attempting to initialise the combined data files.'
                    self.work_error.emit(e_str, 'Invalid Channel Mapping File')

                    # return none values indicating the error
                    return None, None
                else:
                    # otherwise, return the mapped channel depths and the other mapping values
                    return np.array([data[data[:, 1] == x, 0][0] for x in cluster_depth]), data[:, :4]

        # retrieves the job parameters
        exp_info, out_name, g_para = self.thread_job_para[0], self.thread_job_para[1], self.thread_job_para[2]

        # sets the global parameters
        n_hist = int(g_para['n_hist'])
        n_spike = int(g_para['n_spike'])

        # retrieves the spike I/O data and sets the cluster IDs based on the cluster type
        sp_io = spike_io.SpikeIo(exp_info['srcDir'], exp_info['traceFile'], int(exp_info['nChan']))
        if exp_info['clusterType'] == 'Good':
            # case is the good clusters
            cluster_ids = sp_io.good_cluster_ids
        elif exp_info['clusterType'] == 'MUA':
            # case is the multi-unit clusters
            cluster_ids = sp_io.MUA_cluster_ids

        # retrieves the clusters spike data and channel depths
        self.work_progress.emit('Reshaping Cluster Data...', 0.0)
        clusters = [ClusterRead(sp_io, cid) for cid in cluster_ids]

        # determines the channel depths mapping
        depth, channel_map_data = map_cluster_depths()
        if depth is None:
            # if the file has an incorrect format, then exit the function
            return

        # determines if the mapping values were set correctly
        if channel_map_data is not None:
            # if so, then determine the region/recording layers
            y_coords = channel_map_data[:, 3]
            depthLo, depthHi = np.array(exp_info['depthLo']).astype(int), np.array(exp_info['depthHi']).astype(int)
            indD = np.array([next((i for i in range(len(depthHi)) if x <= depthHi[i]), len(depthHi)-1) for x in y_coords])
            chRegion = np.array(exp_info['regionName'])[indD][depth.astype(int)]
            chLayer = np.array(exp_info['recordLayer'])[indD][depth.astype(int)]
        else:
            # otherwise, return N/A for the region/recording layers
            chRegion, chLayer = ['N/A'] * len(clusters), ['N/A'] * len(clusters)

        # sets the signal point-wise/ISI bin vectors
        xi_pts_H = np.linspace(-200, 100, n_hist + 1)
        xi_isi_H = np.linspace(0, 1000, n_hist + 1)

        # creates the recording/experimental information sub-dictionaries
        expInfo = {'name': exp_info['expName'], 'date': exp_info['expDate'], 'cond': exp_info['expCond'],
                   'type': exp_info['expType'], 'sex': exp_info['expSex'], 'age': exp_info['expAge'],
                   'probe': exp_info['expProbe'], 'lesion': exp_info['lesionType'], 'channel_map': channel_map_data,
                   'cluster_type': exp_info['clusterType'], 'other_info': exp_info['otherInfo'],
                   'record_state': exp_info['recordState'], 'record_coord': exp_info['recordCoord']}

        # memory allocation
        pW0, pW1, nFeat = 20.0, 60.0, 5
        nC, nSample = len(clusters), np.size(sp_io.traces, axis=0)
        sFreq, vGain = float(exp_info['sFreq']), float(exp_info['vGain'])

        # sets the data file dictionary object
        A = {
            'vSpike': np.empty(nC, dtype=object), 'tSpike': np.empty(nC, dtype=object),
            'vMu': None, 'vSD': None, 'ccGram': None, 'ccGramXi': None, 'sigFeat': np.zeros((nC, nFeat)),
            'clustID': cluster_ids, 'expInfo': expInfo, 'chDepth': depth, 'chRegion': chRegion, 'chLayer': chLayer,
            'sFreq': sFreq, 'nC': nC,  'nPts': None, 'tExp': nSample / sFreq, 'vGain': vGain,
            'isiHist': np.empty(nC, dtype=object), 'isiHistX': xi_isi_H,
            'ptsHist': np.empty(nC, dtype=object), 'ptsHistX': xi_pts_H,
            'rotInfo': None,
        }

        # sets up the rotation analysis data dictionary
        A['rotInfo'] = rot.load_rot_analysis_data(A, exp_info, sp_io, w_prog=self.work_progress, pW0=pW0)

        # sets up the sub-job flags
        self.sub_job = np.zeros(nC, dtype=bool)

        # retrieves the cluster data
        for i, c in enumerate(clusters):
            if not self.is_running:
                # if the user cancelled, then exit the function
                return
            else:
                # updates the main gui progressnbar
                pW = pW0 + pW1 * (i + 1) / nC
                self.work_progress.emit('Processing Cluster {0} of {1}'.format(i + 1, nC), pW)

            ###################################################
            ####    DATA RETRIEVAL & MEMORY ALLOCATIONS    ####
            ###################################################

            # retrieves the spike voltage/timing
            v_spike = c.channel_waveforms
            t_spike = 1000.0 * sp_io.get_spike_times_in_cluster(cluster_ids[i]) / sFreq

            # memory allocation (only for the first cluster)
            if i == 0:
                A['nPts'] = np.size(v_spike, axis=0)
                A['vMu'] = np.zeros((A['nPts'], nC), dtype=float)
                A['vSD'] = np.zeros((A['nPts'], nC), dtype=float)
                xi = np.array(range(A['nPts']))

            ###############################################
            ####    MAIN METRIC CALCULATION/STORAGE    ####
            ###############################################

            # sets the values into the final array
            A['vSpike'][i] = v_spike[:, :n_spike] * vGain
            A['tSpike'][i] = t_spike[:np.size(v_spike, axis=1)]

            # calculates the mean/standard deviation of the voltage spikes
            A['vMu'][:, i] = np.mean(v_spike, axis=1) * vGain
            A['vSD'][:, i] = np.std(v_spike, axis=1) * vGain

            ######################################
            ####    HISTOGRAM CALCULATIONS    ####
            ######################################

            # calculates the point-wise histograms
            A['ptsHist'][i] = np.zeros((A['nPts'], n_hist), dtype=int)
            for iPts in range(A['nPts']):
                H = np.histogram(v_spike[iPts, :], bins=xi_pts_H)
                A['ptsHist'][i][iPts, :] = H[0]

            # calculates the ISI histograms
            dT = np.diff(A['tSpike'][i])
            dT = dT[dT <= xi_isi_H[-1]]
            H_isi = np.histogram(dT, bins=xi_isi_H, range=(xi_isi_H[0], xi_isi_H[-1]))
            A['isiHist'][i] = H_isi[0]

            ###########################################
            ####    SIGNAL FEATURE CALCULATIONS    ####
            ###########################################

            # creates the piecewise-polynomial of the mean signal
            pp = pchip(xi, A['vMu'][:, i])

            # determines the point/voltage of the pmaximum proceding the minimum
            i_min = np.argmin(A['vMu'][:, i])
            i_max1 = np.argmax(A['vMu'][:i_min, i])
            i_max2 = np.argmax(A['vMu'][i_min:, i]) + i_min

            # determines the location of the half-width points
            v_half = (min(pp(i_max1), pp(i_max2)) + pp(i_min)) / 2.0
            t_lo = cfcn.opt_time_to_y0((pp, v_half), [(i_max1, i_min)])
            t_hi = cfcn.opt_time_to_y0((pp, v_half), [(i_min, i_max2)])

            # sets the signal features into the final array
            A['sigFeat'][i, :] = [i_max1, i_min, i_max2, t_lo, t_hi]

            # memory garbage collection
            gc.collect()

        ######################################################
        ####    CLUSTER CROSS-CORRELOGRAM CALCULATIONS    ####
        ######################################################

        # memory allocation
        win_size = 50

        # calculates the cross-correlation between each signal from each cluster
        for i_row in range(nC):
            if not self.is_running:
                # if the user cancelled, then exit the function
                return
            else:
                # updates the main gui progressbar
                pW = (pW0 + pW1) + (100.0 - (pW0 + pW1)) * (i_row + 1) / (nC + 1)
                self.work_progress.emit('Calculating CC-Grams...', pW)

            # calculates the cross-correlograms between each of the other clusters
            for j_row in range(nC):
                if (i_row == 0) and (j_row == 0):
                    # case is the first cluster so allocate memory and set the time bin array
                    ccGram, A['ccGramXi'] = cfcn.calc_ccgram(A['tSpike'][i_row], A['tSpike'][j_row], win_size)
                    A['ccGram'] = np.zeros((nC, nC, len(ccGram)))
                    A['ccGram'][i_row, j_row, :] = ccGram
                else:
                    # otherwise, set the new values directly into the array
                    A['ccGram'][i_row, j_row, :], _ = cfcn.calc_ccgram(A['tSpike'][i_row], A['tSpike'][j_row], win_size)

        #################################
        ####    FINAL DATA OUTPUT    ####
        #################################

        # dumps the cluster data to file
        self.work_progress.emit('Outputting Data To File...', 99.0)
        with open(out_name, 'wb') as fw:
            p.dump(A, fw)

    def det_cluster_matches(self):
        '''

        :param exp_name:
        :param comp_dlg:
        :return:
        '''

        # retrieves the thread job parameters
        comp, data_fix = self.thread_job_para[0], self.thread_job_para[1]
        data_free, g_para = self.thread_job_para[2], self.thread_job_para[3]

        def det_overall_cluster_matches(is_feas, D):
            '''

            :param data_fix:
            :param data_free:
            :param D:
            :return:
            '''

            # calculates the pair-wise SS distances between each the fixed/free mean signals
            iDsort, n_rows = np.argsort(D.T, axis=None), np.size(D, axis=0)

            # memory allocation
            isFix = np.zeros(data_fix['nC'], dtype=bool)
            isFree = np.zeros(data_free['nC'], dtype=bool)
            i_match = -np.ones(data_fix['nC'], dtype=int)

            # determines the overall unique
            for i in range(len(iDsort)):
                # determines the indices of the next best match
                iR, iC = cfcn.ind2sub(n_rows, iDsort[i])
                if not (isFix[iR] or isFree[iC]) and is_feas[iR, iC]:
                    # if there is not already a match, then update the match arrays
                    i_match[iR] = iC
                    isFix[iR], isFree[iC] = True, True
                    if all(isFix) or all(isFree):
                        # if all matches are found, then exit the loop
                        break

            # returns the final match array
            return i_match

        def det_cluster_matches_old(comp, is_feas, d_depth, g_para):
            '''

            :param data_fix:
            :param data_free:
            :return:
            '''

            # parameters
            z_max = float(g_para['z_max'])
            sig_corr_min = float(g_para['sig_corr_min'])

            # calculates the inter-signal euclidean distances
            DD = cdist(data_fix['vMu'].T, data_free['vMu'].T)

            # determines the matches based on the signal euclidean distances
            comp.i_match_old = det_overall_cluster_matches(is_feas, DD)

            # calculates the correlation coefficients between the best matching signals
            for i in range(data_fix['nC']):
                # calculation of the z-scores
                i_match = comp.i_match_old[i]
                if i_match >= 0:
                    # z-score calculations
                    dW = data_fix['vMu'][:, i] - data_free['vMu'][:, i_match]
                    comp.z_score[:, i] = np.divide(dW, data_fix['vSD'][:, i])

                    # calculates the correlation coefficient
                    CC = np.corrcoef(data_fix['vMu'][:, i], data_free['vMu'][:, i_match])
                    comp.sig_corr_old[i] = CC[0, 1]
                    comp.sig_diff_old[i] = DD[i, i_match]
                    comp.d_depth_old[i] = d_depth[i, i_match]

                    # sets the acceptance flag. for a cluster to be accepted, the following must be true:
                    #   * the maximum absolute z-score must be < z_max
                    #   * the correlation coefficient between the fixed/free signals must be > sig_corr_min
                    comp.is_accept_old[i] = np.max(np.abs(comp.z_score[:, i])) < z_max and \
                                                   comp.sig_corr_old[i] > sig_corr_min
                else:
                    # sets NaN values for all the single value metrics
                    comp.sig_corr[i] = np.nan
                    comp.d_depth_old[i] = np.nan

                    # ensures the group is rejected
                    comp.is_accept_old[i] = False

            # returns the comparison data class object
            return comp

        def det_cluster_matches_new(comp, is_feas, d_depth, r_spike, g_para):
            '''

            :param data_fix:
            :param data_free:
            :return:
            '''

            # parameters
            pW = 100.0 / 7.0
            sig_corr_min = float(g_para['sig_corr_min'])
            isi_corr_min = float(g_para['isi_corr_min'])
            sig_diff_max = float(g_para['sig_diff_max'])
            sig_feat_min = float(g_para['sig_feat_min'])
            w_sig_feat = float(g_para['w_sig_feat'])
            w_sig_comp = float(g_para['w_sig_comp'])
            w_isi = float(g_para['w_isi'])

            # memory allocation
            signal_metrics = np.zeros((data_fix['nC'], data_free['nC'], 4))
            isi_metrics = np.zeros((data_fix['nC'], data_free['nC'], 3))
            isi_metrics_norm = np.zeros((data_fix['nC'], data_free['nC'], 3))
            total_metrics = np.zeros((data_fix['nC'], data_free['nC'], 3))

            # initialises the comparison data object
            self.work_progress.emit('Calculating Signal DTW Indices', pW)
            comp = cfcn.calc_dtw_indices(comp, data_fix, data_free, is_feas)

            # calculates the signal feature metrics
            self.work_progress.emit('Calculating Signal Feature Metrics', 2.0 * pW)
            signal_feat = cfcn.calc_signal_feature_diff(data_fix, data_free, is_feas)

            # calculates the signal direct matching metrics
            self.work_progress.emit('Calculating Signal Comparison Metrics', 3.0 * pW)
            cc_dtw, dd_dtw, dtw_scale = \
                cfcn.calc_signal_corr(comp.i_dtw, data_fix, data_free, is_feas)

            signal_metrics[:, :, 0] = cc_dtw
            signal_metrics[:, :, 1] = 1.0 - dd_dtw
            signal_metrics[:, :, 2] = dtw_scale
            signal_metrics[:, :, 3] = \
                cfcn.calc_signal_hist_metrics(data_fix, data_free, is_feas, cfcn.calc_hist_intersect, max_norm=True)

            # calculates the ISI histogram metrics
            self.work_progress.emit('Calculating ISI Histogram Comparison Metrics', 4.0 * pW)
            isi_metrics[:, :, 0], isi_metrics_norm[:, :, 0] = \
                cfcn.calc_isi_corr(data_fix, data_free, is_feas)
            isi_metrics[:, :, 1], isi_metrics_norm[:, :, 1] = \
                cfcn.calc_isi_hist_metrics(data_fix, data_free, is_feas, cfcn.calc_hist_intersect, max_norm=True)
            # isi_metrics[:, :, 2], isi_metrics_norm[:, :, 2] = \
            #     cfcn.calc_isi_hist_metrics(data_fix, data_free, is_feas, cfcn.calc_wasserstein, max_norm=False)
            # isi_metrics[:, :, 3], isi_metrics_norm[:, :, 3] = \
            #     cfcn.calc_isi_hist_metrics(data_fix, data_free, is_feas, cfcn.calc_bhattacharyya, max_norm=True)

            # sets the isi relative spiking rate metrics
            isi_metrics[:, :, 2] = np.nan
            for i_row in range(np.size(r_spike, axis=0)):
                isi_metrics[i_row, is_feas[i_row, :], 2] = r_spike[i_row, is_feas[i_row, :]]
            isi_metrics_norm[:, :, 2] = cfcn.norm_array_rows(isi_metrics[:, :, 2], max_norm=False)

            # calculates the array euclidean distances (over all measures/clusters)
            total_metrics[:, :, 0] = cfcn.calc_array_euclidean(signal_feat)
            total_metrics[:, :, 1] = cfcn.calc_array_euclidean(signal_metrics)
            total_metrics[:, :, 2] = cfcn.calc_array_euclidean(isi_metrics_norm)
            total_metrics_mean = cfcn.calc_weighted_mean(total_metrics, W=[w_sig_feat, w_sig_comp, w_isi])

            # determines the unique overall cluster matches
            self.work_progress.emit('Determining Overall Cluster Matches', 5.0 * pW)
            comp.i_match = det_overall_cluster_matches(is_feas, -total_metrics_mean)

            # calculates the correlation coefficients between the best matching signals
            self.work_progress.emit('Setting Final Match Metrics', 6.0 * pW)
            for i in range(data_fix['nC']):
                # calculation of the z-scores
                i_match = comp.i_match[i]
                if i_match >= 0:
                    # sets the signal feature metrics
                    comp.match_intersect[:, i] = cfcn.calc_single_hist_metric(data_fix, data_free, i, i_match,
                                                                              True, cfcn.calc_hist_intersect)
                    comp.match_wasserstain[:, i] = cfcn.calc_single_hist_metric(data_fix, data_free, i,
                                                                                i_match, True, cfcn.calc_wasserstein)
                    comp.match_bhattacharyya[:, i] = cfcn.calc_single_hist_metric(data_fix, data_free, i,
                                                                                  i_match, True, cfcn.calc_bhattacharyya)

                    # sets the signal difference metrics
                    comp.d_depth[i] = d_depth[i, i_match]
                    comp.dtw_scale[i] = dtw_scale[i, i_match]
                    comp.sig_corr[i] = cc_dtw[i, i_match]
                    comp.sig_diff[i] = max(0.0, 1 - dd_dtw[i, i_match])
                    comp.sig_intersect[i] = signal_metrics[i, i_match, 2]

                    # sets the isi metrics
                    comp.isi_corr[i] = isi_metrics[i, i_match, 0]
                    comp.isi_intersect[i] = isi_metrics[i, i_match, 1]
                    # comp.isi_wasserstein[i] = isi_metrics[i, i_match, 2]
                    # comp.isi_bhattacharyya[i] = isi_metrics[i, i_match, 3]

                    # sets the total match metrics
                    comp.signal_feat[i, :] = signal_feat[i, i_match, :]
                    comp.total_metrics[i, :] = total_metrics[i, i_match, :]
                    comp.total_metrics_mean[i] = total_metrics_mean[i, i_match]


                    # sets the acceptance flag. for a cluster to be accepted, the following must be true:
                    #   * the ISI correlation coefficient must be > isi_corr_min
                    #   * the signal correlation coefficient must be > sig_corr_min
                    #   * the inter-signal euclidean distance must be < sig_diff_max
                    comp.is_accept[i] = (comp.isi_corr[i] > isi_corr_min) and \
                                        (comp.sig_corr[i] > sig_corr_min) and \
                                        (comp.sig_diff[i] > (1 - sig_diff_max)) and \
                                        (np.all(comp.signal_feat[i, :] > sig_feat_min))
                else:
                    # sets NaN values for all the single value metrics
                    comp.d_depth[i] = np.nan
                    comp.dtw_scale[i] = np.nan
                    comp.sig_corr[i] = np.nan
                    comp.sig_diff[i] = np.nan
                    comp.sig_intersect[i] = np.nan
                    comp.isi_corr[i] = np.nan
                    comp.isi_intersect[i] = np.nan
                    # comp.isi_wasserstein[i] = np.nan
                    # comp.isi_bhattacharyya[i] = np.nan
                    comp.signal_feat[i, :] = np.nan
                    comp.total_metrics[i, :] = np.nan
                    comp.total_metrics_mean[i] = np.nan

                    # ensures the group is rejected
                    comp.is_accept[i] = False

            # returns the comparison data class object
            return comp

        # parameters
        d_max = int(g_para['d_max'])
        r_max = float(g_para['r_max'])

        # determines the number of spikes
        n_spike_fix = [len(x) / data_fix['tExp'] for x in data_fix['tSpike']]
        n_spike_free = [len(x) / data_free['tExp'] for x in data_free['tSpike']]

        # calculates the relative spiking rates (note - ratios are coverted so that they are all > 1)
        r_spike = np.divide(repmat(n_spike_fix, data_free['nC'], 1).T,
                            repmat(n_spike_free, data_fix['nC'], 1))
        r_spike[r_spike < 1] = 1 / r_spike[r_spike < 1]

        # calculates the pair-wise distances between the fixed/free probe depths
        d_depth = np.abs(np.subtract(repmat(data_fix['chDepth'], data_free['nC'], 1).T,
                                     repmat(data_free['chDepth'], data_fix['nC'], 1)))

        # determines the feasible fixed/free cluster groupings such that:
        #  1) the channel depth has to be <= d_max
        #  2) the relative spiking rates between clusters is <= r_max
        is_feas = np.logical_and(r_spike < r_max, d_depth < d_max)

        # determines the cluster matches from the old/new methods
        comp = det_cluster_matches_old(comp, is_feas, d_depth, g_para)
        comp = det_cluster_matches_new(comp, is_feas, d_depth, r_spike, g_para)

        # returns the comparison data class object
        return comp

    def calc_ccgram_types(self, calc_para, data):
        '''

        :return:
        '''

        # determines the indices of the experiment to be analysed
        if calc_para['calc_all_expt']:
            # case is all experiments are to be analysed
            i_expt = list(range(len(data)))
        else:
            # case is a single experiment is being analysed
            i_expt = [cf.get_expt_index(calc_para['calc_exp_name'], data)]

        # memory allocation
        d_copy = copy.deepcopy
        A, B, C = np.empty(len(i_expt), dtype=object), [[] for _ in range(5)], [[] for _ in range(4)]
        c_type, t_dur, t_event, ci_lo, ci_hi, ccG_T = d_copy(A), d_copy(A), d_copy(A), d_copy(A), d_copy(A), d_copy(A)

        #
        for i_ex in i_expt:
            # sets the experiment ID info based on the number of experiments being analysed
            if len(i_expt) == 1:
                # only one experiment is being analysed
                expt_id = None
            else:
                # multiple experiments are being analysed
                expt_id = [(i_ex+1), len(i_expt)]

            # retrieves the cluster information
            t_dur[i_ex], t_event[i_ex] = d_copy(C), d_copy(C)
            c_type[i_ex], ci_lo[i_ex], ci_hi[i_ex], ccG_T[i_ex] = d_copy(B), d_copy(B), d_copy(B), d_copy(B)
            ccG, ccG_xi, t_spike = data[i_ex]['ccGram'], data[i_ex]['ccGramXi'], data[i_ex]['tSpike']

            c_id = data[i_ex]['clustID']

            # runs the cc-gram type calculation function
            c_type0, t_dur[i_ex], t_event[i_ex], ci_hi0, ci_lo0, ccG_T0 = cfcn.calc_ccgram_types(
                            ccG, ccG_xi, t_spike, calc_para=calc_para, expt_id=expt_id, w_prog=self.work_progress, c_id=c_id)

            # sets the final values into their respective groupings
            for i in range(5):
                # sets the final type values and lower/upper bound confidence interval signals
                if len(c_type0[i]):
                    #
                    c_type[i_ex][i] = np.vstack(c_type0[i])

                    # sorts the values by the reference cluster index
                    i_sort = np.lexsort((c_type[i_ex][i][:, 1], c_type[i_ex][i][:, 0]))
                    c_type[i_ex][i] = c_type[i_ex][i][i_sort, :]

                    # reorders the duration/timing of the events (if they exist)
                    if i < len(t_dur[i_ex]):
                        t_dur[i_ex][i] = np.array(t_dur[i_ex][i])[i_sort]
                        t_event[i_ex][i] = np.array(t_event[i_ex][i])[i_sort]

                        ci_lo[i_ex][i] = (np.vstack(ci_lo0[i]).T)[:, i_sort]
                        ci_hi[i_ex][i] = (np.vstack(ci_hi0[i]).T)[:, i_sort]
                        ccG_T[i_ex][i] = (np.vstack(ccG_T0[i]).T)[:, i_sort]

        # returns the data as a dictionary
        return {'c_type': c_type, 't_dur': t_dur, 't_event': t_event,
                'ci_lo': ci_lo, 'ci_hi': ci_hi, 'ccG_T': ccG_T, 'calc_para': calc_para}

    def calc_shuffled_cluster_dist(self, calc_para, data):
        '''

        :return:
        '''

        # FINISH ME!
        pass

    ######################################
    ####    ROC CURVE CALCULATIONS    ####
    ######################################

    def calc_phase_roc_curves(self, data, pW):
        '''

        :param calc_para:
        :param plot_para:
        :param data:
        :param pool:
        :return:
        '''

        # parameters and initialisations
        phase_str, r_data = ['CW/BL', 'CCW/BL', 'CCW/CW'], data.rotation

        # if the roc curves have already been calculated then exit with the current array
        if r_data.phase_roc is not None:
            return

        # sets up the black phase data filter and returns the time spikes
        r_filt_black = cf.init_rotation_filter_data(False)
        # r_filt_black['t_type'] = ['Uniform']                        # CHANGE THIS TO THE TRIAL CONDITION YOU WANT
        r_data.r_obj_black = RotationFilteredData(data, r_filt_black, 0, None, True, 'Whole Experiment', False)
        t_spike = r_data.r_obj_black.t_spike[0]

        # memory allocation
        n_cell = np.size(r_data.r_obj_black.t_spike[0], axis=0)
        r_data.phase_roc = np.empty((n_cell, len(phase_str)), dtype=object)
        r_data.phase_roc_xy = np.empty(n_cell, dtype=object)
        r_data.phase_roc_auc = np.ones((n_cell, len(phase_str)))

        # calculates the roc curves/integrals for all cells over each phase
        for i_phs, p_str in enumerate(phase_str):
            # updates the progress bar string
            w_str = 'ROC Curve Calculations ({0})...'.format(p_str)
            self.work_progress.emit(w_str, pW * i_phs / len(phase_str))

            # calculates the bootstrapped confidence intervals for each cell
            ind = np.array([1 * (i_phs > 1), 1 + (i_phs > 0)])
            for i_cell in range(n_cell):
                # calculates the roc curve/auc integral
                r_data.phase_roc[i_cell, i_phs] = cf.calc_roc_curves(t_spike[i_cell, :, :], ind=ind)
                r_data.phase_roc_auc[i_cell, i_phs] = cf.get_roc_auc_value(r_data.phase_roc[i_cell, i_phs])

                # if the CW/CCW phase interaction, then set the roc curve x/y coordinates
                if (i_phs + 1) == len(phase_str):
                    r_data.phase_roc_xy[i_cell] = cf.get_roc_xy_values(r_data.phase_roc[i_cell, i_phs])

    def calc_cond_roc_curves(self, data, pool, calc_para, plot_para, calc_cell_grp, pW):
        '''

        :param calc_para:
        :param plot_para:
        :param data:
        :param pool:
        :return:
        '''

        # parameters and initialisations
        r_obj_sig, plot_scope = None, 'Whole Experiment'
        phase_str, r_data = ['CW/BL', 'CCW/BL', 'CCW/CW'], data.rotation

        # initisalises the rotational filter (if not initialised already)
        if plot_para['rot_filt'] is None:
            plot_para['rot_filt'] = cf.init_rotation_filter_data(False)

        # sets the condition types (ensures that the black phase is always included)
        t_type = plot_para['rot_filt']['t_type']
        if 'Black' not in t_type:
            t_type = ['Black'] + t_type

        # sets up a base filter with only the
        r_filt_base = cf.init_rotation_filter_data(False)
        r_filt_base['t_type'] = [x for x in t_type if x != 'UniformDrifting']

        # sets up the black phase data filter and returns the time spikes
        r_obj = RotationFilteredData(data, r_filt_base, None, plot_para['plot_exp_name'], True, plot_scope, False)

        # memory allocation (if the conditions have not been set)
        if r_data.cond_roc is None:
            r_data.cond_roc, r_data.cond_roc_xy, r_data.cond_roc_auc = {}, {}, {}
            r_data.cond_gtype, r_data.cond_auc_sig, r_data.cond_i_expt, r_data.cond_cl_id = {}, {}, {}, {}
            r_data.cond_ci_lo, r_data.cond_ci_hi, r_data.r_obj_cond = {}, {}, {}

        for i_rr, rr in enumerate(r_obj.rot_filt_tot):
            tt = rr['t_type'][0]
            if tt not in r_data.cond_roc:
                # array dimensions
                t_spike = r_obj.t_spike[i_rr]
                n_cell = np.size(t_spike, axis=0)

                # memory allocation and initialisations
                r_data.cond_roc[tt] = np.empty((n_cell, 3), dtype=object)
                r_data.cond_roc_xy[tt] = np.empty(n_cell, dtype=object)
                r_data.cond_roc_auc[tt] = np.zeros((n_cell, 3))
                r_data.cond_gtype[tt] = -np.ones((n_cell, 3))
                r_data.cond_auc_sig[tt] = np.zeros((n_cell, 3), dtype=bool)
                r_data.cond_i_expt[tt] = r_obj.i_expt[i_rr]
                r_data.cond_cl_id[tt] = r_obj.cl_id[i_rr]
                r_data.cond_ci_lo[tt] = -np.ones((n_cell, 2))
                r_data.cond_ci_hi[tt] = -np.ones((n_cell, 2))
                r_data.r_obj_cond[tt] = dcopy(r_obj)

                # calculates the roc curves/integrals for all cells over each phase
                w_str = 'ROC Curve Calculations ({0})...'.format(tt)
                for i_phs, p_str in enumerate(phase_str):
                    # updates the progress bar string
                    self.work_progress.emit(w_str, pW * ((i_rr / r_obj.n_filt) + (i_phs / len(phase_str))))

                    # calculates the roc curve values for each phase
                    ind = np.array([1 * (i_phs > 1), 1 + (i_phs > 0)])
                    for ic in range(n_cell):
                        r_data.cond_roc[tt][ic, i_phs] = cf.calc_roc_curves(t_spike[ic, :, :], ind=ind)
                        r_data.cond_roc_auc[tt][ic, i_phs] = cf.get_roc_auc_value(r_data.cond_roc[tt][ic, i_phs])

                        if (i_phs + 1) == len(phase_str):
                            r_data.cond_roc_xy[tt][ic] = cf.get_roc_xy_values(r_data.cond_roc[tt][ic, i_phs])

            # calculates the confidence intervals for the current (only if bootstrapping count has changed or the
            # confidence intervals has not already been calculated)
            if 'auc_stype' in calc_para:
                # updates the auc statistics calculation type
                r_data.cond_auc_stats_type = calc_para['auc_stype']

                # determine if the auc confidence intervals need calculation
                is_boot = int(calc_para['auc_stype'] == 'Bootstrapping')
                if is_boot:
                    # if bootstrapping, then determine if the
                    if r_data.n_boot_cond_ci != calc_para['n_boot']:
                        # if the bootstrapping count has changed, flag that the confidence intervals needs updating
                        r_data.n_boot_cond_ci, calc_ci = calc_para['n_boot'], True
                    else:
                        # otherwise, recalculate the confidence intervals if they have not been set
                        calc_ci = np.any(r_data.cond_ci_lo[tt][:, 1] < 0)
                else:
                    # otherwise, recalculate the confidence intervals if they have not been set
                    calc_ci = np.any(r_data.cond_ci_lo[tt][:, 0] < 0)

                # calculates the confidence intervals (if required)
                if calc_ci:
                    conf_int = self.calc_roc_conf_intervals(pool, r_data.cond_roc[tt][:, 2],
                                                            calc_para['auc_stype'], calc_para['n_boot'])
                    r_data.cond_ci_lo[tt][:, is_boot] = conf_int[:, 0]
                    r_data.cond_ci_hi[tt][:, is_boot] = conf_int[:, 1]

            # if not calculating the cell group indices, or the condition type is Black (the phase statistics for
            # this condition are already calculated in "calc_phase_roc_significance"), then continue
            if (not calc_cell_grp) or (tt == 'Black'):
                continue

            # sets the rotation object filter (if using wilcoxon paired test for the cell group stats type)
            if calc_para['grp_stype'] == 'Wilcoxon Paired Test':
                if np.all(r_data.cond_gtype[tt][:, 0] >= 0):
                    # if all the values have been calculated, then exit the function
                    return

                # sets the rotation object for the current condition
                r_obj_sig = RotationFilteredData(data, r_obj.rot_filt_tot[i_rr], None, plot_para['plot_exp_name'],
                                                 True, plot_scope, False)

            # calculates the condition cell group types
            self.calc_phase_roc_significance(calc_para, data, pool, None, c_type='cond',
                                             roc=r_data.cond_roc[tt], auc=r_data.cond_roc_auc[tt],
                                             g_type=r_data.cond_gtype[tt], auc_sig=r_data.cond_auc_sig[tt],
                                             r_obj=r_obj_sig)

    def calc_phase_roc_significance(self, calc_para, data, pool, pW, c_type='phase',
                                    roc=None, auc=None, g_type=None, auc_sig=None, r_obj=None):
        '''

        :param calc_data:
        :param data:
        :param pool:
        :return:
        '''

        # sets the roc objects/integrals (if not provided)
        r_data = data.rotation
        if c_type == 'phase':
            # case is the significance tests are being calculated for the phase
            r_data.phase_grp_stats_type = calc_para['grp_stype']
            roc, auc, r_obj = r_data.phase_roc, r_data.phase_roc_auc, r_data.r_obj_black
        else:
            # case is the significance tests are being calculated for the conditions
            r_data.cond_grp_stats_type = calc_para['grp_stype']

        # parameters and initialisations
        phase_str, i_col = ['CW/BL', 'CCW/BL', 'CCW/CW'], 0
        p_value, n_cell = 0.05, np.size(roc, axis=0)

        # allocates memory for the group-types (if not already calculated)
        if c_type == 'phase':
            # case is for the phase type
            n_boot = r_data.n_boot_phase_grp
            if r_data.phase_gtype is None:
                # group type has not been set, so initialise the array
                r_data.phase_gtype = g_type = -np.ones((n_cell, 3))
                r_data.phase_auc_sig = auc_sig = np.zeros((n_cell, 3), dtype=bool)
            else:
                # otherwise, retrieve the currently stored array
                g_type, auc_sig = r_data.phase_gtype, r_data.phase_auc_sig
        else:
            # case is for the condition type
            n_boot = r_data.n_boot_cond_grp

        #########################################
        ####    WILCOXON STATISTICAL TEST    ####
        #########################################

        if calc_para['grp_stype'] == 'Wilcoxon Paired Test':
            # if the statistics have already been calculated, then exit the function
            if np.all(g_type[:, 0] >= 0):
                return

            # updates the progress bar string
            if pW is not None:
                self.work_progress.emit('Calculating Wilcoxon Stats...', pW + 25.)

            # calculates the statistical significance between the phases
            sp_f0, sp_f = cf.calc_phase_spike_freq(r_obj)
            _, _, sf_stats, _ = cf.setup_spike_freq_plot_arrays(r_obj, sp_f0, sp_f, None)

            # determines which cells are motion/direction sensitive
            for i_phs in range(len(sf_stats)):
                auc_sig[:, i_phs] = sf_stats[i_phs] < p_value

        ##########################################
        ####    ROC-BASED STATISTICAL TEST    ####
        ##########################################

        else:
            # determines what kind of statistics are to be calculated
            is_boot = calc_para['grp_stype'] == 'Bootstrapping'
            i_col, phase_stype = 1 + is_boot, calc_para['grp_stype']

            # if the statistics have been calculated for the selected type, then exit the function
            if is_boot:
                if np.all(g_type[:, 2] >= 0) and (calc_para['n_boot'] == n_boot):
                    # if bootstrapping is selected, but all values have been calculated and the bootstrapping values
                    # has not changed, then exit the function
                    return
                else:
                    # otherwise, update the bootstrapping count
                    if c_type == 'phase':
                        r_data.n_boot_phase_grp = dcopy(calc_para['n_boot'])
                    else:
                        r_data.n_boot_cond_grp = dcopy(calc_para['n_boot'])

            elif np.all(g_type[:, 1] >= 0):
                # if delong significance is selected, and all values have been calculated, then exit the function
                return

            # calculates the significance for each phase
            for i_phs, p_str in enumerate(phase_str):
                # updates the progress bar string
                if pW is not None:
                    w_str = 'ROC Curve Calculations ({0})...'.format(p_str)
                    self.work_progress.emit(w_str, pW * (1. + i_phs / len(phase_str)))

                # calculates the confidence intervals for the current
                conf_int = self.calc_roc_conf_intervals(pool, roc[:, i_phs], phase_stype, n_boot)

                # determines the significance for each cell in the phase
                auc_ci_lo = (auc[:, i_phs] + conf_int[:, 1]) < 0.5
                auc_ci_hi = (auc[:, i_phs] - conf_int[:, 0]) > 0.5
                auc_sig[:, i_phs] = np.logical_or(auc_ci_lo, auc_ci_hi)

        # calculates the cell group types
        g_type[:, i_col] = cf.calc_cell_group_types(auc_sig, calc_para['grp_stype'])

    def calc_roc_conf_intervals(self, pool, roc, phase_stype, n_boot):
        '''

        :param r_data:
        :return:
        '''

        # sets the parameters for the multi-processing pool
        p_data = []
        for i_cell in range(len(roc)):
            p_data.append([roc[i_cell], phase_stype, n_boot])

        # returns the rotation data class object
        return np.array(pool.map(cf.calc_roc_conf_intervals, p_data))

    def calc_dirsel_group_types(self, data, calc_para, plot_para):
        '''

        :param data:
        :param plot_para:
        :return:
        '''

        def calc_combined_spiking_stats(r_obj, p_value, ind_type=None):
            '''

            :param r_obj:
            :param ind_type:
            :return:
            '''

            # calculates the individual trial/mean spiking rates and sets up the plot/stats arrays
            sp_f0, sp_f = cf.calc_phase_spike_freq(r_obj)
            s_plt, _, sf_stats, i_grp = cf.setup_spike_freq_plot_arrays(r_obj, sp_f0, sp_f, ind_type)

            # returns the direction selectivity scores
            return cf.calc_dirsel_scores(s_plt, sf_stats, p_value), i_grp

        # initialises the rotation filter (if not set)
        rot_filt = plot_para['rot_filt']
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # sets the p-value
        if 'p_value' in calc_para:
            p_value = calc_para['p_value']
        else:
            p_value = 0.05

        # initialisations and memory allocation
        p_scope, n_grp, r_data = 'Whole Experiment', 4, data.rotation
        r_filt_rot, r_filt_vis = dcopy(rot_filt), dcopy(rot_filt)
        plot_exp_name, plot_all_expt = plot_para['plot_exp_name'], plot_para['plot_all_expt']
        r_data.ds_p_value = dcopy(p_value)

        # sets up the black-only rotation filter object
        r_filt_black = cf.init_rotation_filter_data(False)
        r_obj_black = RotationFilteredData(data, r_filt_black, None, plot_exp_name, plot_all_expt, p_scope, False)

        # retrieves the rotational filtered data
        r_filt_rot['t_type'], r_filt_rot['is_ud'] = ['Black'], [False]
        r_data.r_obj_rot_ds = RotationFilteredData(data, r_filt_rot, None, plot_exp_name, plot_all_expt, p_scope,
                                                   False)

        # retrieves the uniform-drifting filtered data (split into CW/CCW phases)
        r_filt_vis['t_type'], r_filt_vis['is_ud'], r_filt_vis['t_cycle'] = ['UniformDrifting'], [True], ['15']
        r_obj_vis, ind_type = cf.split_unidrift_phases(data, r_filt_vis, None, plot_exp_name, plot_all_expt,
                                                       p_scope, calc_para['t_phase'], calc_para['t_ofs'])
        if r_obj_vis is None:
            # output an error to screen
            e_str = 'The entered analysis duration and offset is greater than the experimental phase duration:\n\n' \
                    '  * Analysis Duration + Offset = {0}\n s. * Experiment Phase Duration = {1} s.\n\n' \
                    'Enter a correct analysis duration/offset combination before re-running ' \
                    'the function.'.format(calc_para['t_phase'] + calc_para['t_ofs'], 2.0)
            self.work_error.emit(e_str, 'Incorrect Analysis Function Parameters')

            # return a false value indicating the calculation is invalid
            return False

        # calculate the visual/rotation stats scores
        sf_score_rot, i_grp_rot = calc_combined_spiking_stats(r_data.r_obj_rot_ds, p_value)
        sf_score_vis, i_grp_vis = calc_combined_spiking_stats(r_obj_vis, p_value, ind_type)

        # memory allocation
        pr_type_tmp, pd_type_tmp, r_data.ds_gtype_N, r_data.pd_type_N = [], [], [], []
        A = -np.ones(np.size(r_obj_black.t_spike[0], axis=0), dtype=int)
        r_data.ds_gtype, r_data.pd_type = dcopy(A), dcopy(A)

        # reduces the arrays to the matching cells
        for i in range(len(i_grp_rot)):
            # retrieves the matching rotation/visual indices
            ind_rot, ind_vis = cf.det_cell_match_indices(r_data.r_obj_rot_ds, i, r_obj_vis)

            # retrieves the CW vs BL/CCW vs BL significance scores
            _sf_score_rot = sf_score_rot[i_grp_rot[i][ind_rot]][:, :-1]
            _sf_score_vis = sf_score_vis[i_grp_vis[i][ind_vis]][:, :-1]

            # from this, determine the reaction type from the score phase types (append proportion/N-value arrays)
            #   0 = None
            #   1 = Rotation Only
            #   2 = Visual Only
            #   3 = Both
            ds_gtype = (np.sum(_sf_score_rot, axis=1) > 0) + 2 * (np.sum(_sf_score_vis, axis=1) > 0)
            pr_type_tmp.append(cf.calc_rel_prop(ds_gtype, 4))
            r_data.ds_gtype_N.append(len(ind_rot))

            # determines which cells have significance for both rotation/visual stimuli. from this determine the
            # preferred direction from the CW vs CCW spiking rates
            is_both_ds = ds_gtype == 3
            pref_rot_dir = sf_score_rot[i_grp_rot[i][ind_rot]][is_both_ds, -1]
            pref_vis_dir = sf_score_vis[i_grp_vis[i][ind_vis]][is_both_ds, -1]
            pref_dir_comb = np.vstack((pref_rot_dir, pref_vis_dir)).T

            # # determines the preferred directions
            # is_both_ds = ds_gtype == 3
            # _sf_score_rot = sf_score_rot[i_grp_rot[i][ind_rot]][is_both_ds, :]
            # _sf_score_vis = sf_score_vis[i_grp_vis[i][ind_vis]][is_both_ds, :]
            # pref_dir_comb = np.vstack((pref_rot_dir, pref_vis_dir)).T

            # make the calculations simpler (look at what is significantly direction selective)
            #   => Look at relative ratio of CCW/CW to check congruency
            #   => Have this an option (to exclude None, Rotation/Visual only) to have Congruency)

            # determines the preferred direction type (for clusters which have BOTH rotation and visual significance)
            #   0 = None
            #   1 = Rotation preferred only
            #   2 = Visual preferred only
            #   3 = Congruent (preferred direction is different)
            #   4 = Incongruent (preferred direction is the same)
            pd_type = np.zeros(len(pref_rot_dir), dtype=int)
            pd_type_rng, pd_type_min = cfcn.arr_range(pref_dir_comb, 1), np.min(pref_dir_comb, axis=1)
            pd_type[np.logical_and(pref_dir_comb[:, 0] > 0, pref_dir_comb[:, 1] == 0)] = 1
            pd_type[np.logical_and(pref_dir_comb[:, 0] == 0, pref_dir_comb[:, 1] > 0)] = 2
            pd_type[np.logical_and(pd_type_min > 0, pd_type_rng != 0)] = 3
            pd_type[np.logical_and(pd_type_min > 0, pd_type_rng == 0)] = 4

            # calculates the preferred direction type count/proportions
            r_data.pd_type_N.append(cf.calc_rel_count(pd_type, 5))
            pd_type_tmp.append(cf.calc_rel_prop(pd_type, 5))

            # sets the indices of the temporary group type into the total array
            ind_bl, ind_bl_rot = cf.det_cell_match_indices(r_obj_black, [0, i], r_data.r_obj_rot_ds)
            ind_comb = ind_bl[np.searchsorted(ind_bl_rot, ind_rot)]
            r_data.ds_gtype[ind_comb] = ds_gtype
            r_data.pd_type[ind_comb[is_both_ds]] = pd_type

        # combines the relative proportion lists into a single array (direction selectivity/preferred direction)
        r_data.ds_gtype_pr = np.vstack(pr_type_tmp).T
        r_data.pd_type_pr = np.vstack(pd_type_tmp).T

        # return a true flag to indicate the analysis was valid
        return True

    def calc_binned_kinemetic_spike_freq(self, data, plot_para, calc_para, roc_calc=True):
        '''

        :param calc_para:
        :return:
        '''

        # parameters and initialisations
        vel_bin, r_data, equal_time = float(calc_para['vel_bin']), data.rotation, calc_para['equal_time']

        # if the velocity bin size has changed or isn't initialised, then reset velocity roc values
        if roc_calc:
            if vel_bin != r_data.vel_bin:
                r_data.vel_sf_rs, r_data.spd_sf_rs = None, None
                r_data.vel_sf, r_data.spd_sf = None, None
                r_data.vel_bin, r_data.vel_roc = vel_bin, None

            if r_data.is_equal_time != equal_time:
                r_data.vel_roc = None

        # if using equal time bins, then check to see if the sample size has changed (if so then recalculate)
        if equal_time:
            if r_data.n_rs != calc_para['n_sample']:
                r_data.vel_sf_rs, r_data.spd_sf_rs = None, None
                r_data.n_rs, r_data.vel_roc = calc_para['n_sample'], None

        # sets the condition types (ensures that the black phase is always included)
        r_filt_base = cf.init_rotation_filter_data(False)
        if plot_para['rot_filt'] is not None:
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
        if calc_para['equal_time']:
            # case is using resampling from equal time bin sizes
            n_rs, w_prog = calc_para['n_sample'], self.work_progress
            vel_n, xi_bin, t_bin = rot.calc_resampled_vel_spike_freq(data, w_prog, r_data.r_obj_kine, [vel_bin], n_rs)
        else:
            vel_n, xi_bin, t_bin = rot.calc_kinemetic_spike_freq(data, r_data.r_obj_kine, [10, vel_bin], True, False)

        # sets the bin duration/limits
        n_filt, i_bin0 = r_data.r_obj_kine.n_filt, np.where(xi_bin == 0)[0][0]
        if not equal_time:
            vel_dt = 2 * np.abs(np.diff(t_bin))
            spd_dt = vel_dt[i_bin0:]

        # sets the comparison bin for the velocity/speed arrays
        r_data.is_equal_time = equal_time
        r_data.vel_xi, r_data.spd_xi = xi_bin, xi_bin[i_bin0:]

        if 'spd_x_rng' in calc_para:
            x_rng = calc_para['spd_x_rng'].split()
            r_data.i_bin_spd = list(xi_bin[i_bin0:]).index(int(x_rng[0]))
            r_data.i_bin_vel = [list(xi_bin).index(-int(x_rng[2])), list(xi_bin).index(int(x_rng[0]))]

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

            # sets the speed counts
            spd_n = np.vstack((np.flip(vel_n[i_filt][:, :i_bin0, :], 1), vel_n[i_filt][:, i_bin0:, :]))

            if equal_time:
                # case is using equally spaced time bins
                r_data.vel_sf_rs[tt] = dcopy(vel_n[i_filt]) / t_bin
                r_data.spd_sf_rs[tt] = dcopy(spd_n) / t_bin
            else:
                # case is using the normal time bins
                r_data.vel_sf[tt], r_data.spd_sf[tt] = dcopy(vel_n[i_filt]), dcopy(spd_n)

                # memory allocation
                n_trial = np.size(vel_n[i_filt], axis=0)
                dt_bin_vel = np.matlib.repmat(dcopy(vel_dt), n_trial, 1)
                dt_bin_spd = np.matlib.repmat(dcopy(spd_dt), 2 * n_trial, 1)

                # normalises the binned spike counts by the duration of the time bin
                for i_cell in range(np.size(vel_n[i_filt], axis=2)):
                    r_data.vel_sf[tt][:, :, i_cell] /= dt_bin_vel
                    r_data.spd_sf[tt][:, :, i_cell] /= dt_bin_spd

    def calc_kinematic_roc_curves(self, data, pool, calc_para, pW0):
        '''

        :param calc_para:
        :return:
        '''

        # def resample_spike_freq(data, r_data, rr, ind, n_rs=100):
        #     '''
        #
        #     :param sf:
        #     :param n_shuffle:
        #     :return:
        #     '''
        #
        #     # sets the index dictionary
        #     indD = {'ind_filt': ind[0], 'ind_bin': ind[1], 'ind_cell': ind[2]}
        #
        #     # resamples the spike frequencies twice for the dependent/independent values
        #     sf_x = rot.calc_resampled_vel_spike_freq(data, None, r_data.r_obj_kine, [r_data.vel_bin], n_rs, indD)
        #     sf_y = rot.calc_resampled_vel_spike_freq(data, None, r_data.r_obj_kine, [r_data.vel_bin], n_rs, indD)
        #
        #
        #     #
        #     return sf_x, sf_y

        def resample_spike_freq(sf, n_rs=1000):
            '''

            :param data:
            :param r_data:
            :param rr:
            :param ind:
            :param n_rs:
            :return:
            '''

            # initialisations and memory allocation
            n_trial, A = len(sf), np.empty(n_rs, dtype=object)
            sf_x, sf_y, n_trial_h = dcopy(A), dcopy(A), int(np.floor(n_trial / 2))

            # returns the shuffled spike frequency arrays
            for i_rs in range(n_rs):
                ind0 = np.random.permutation(n_trial)
                sf_x[i_rs], sf_y[i_rs] = sf[ind0[:n_trial_h]], sf[ind0[n_trial_h:(2 * n_trial_h)]]

            _roc = [cf.calc_roc_curves(None, None, x_grp=x, y_grp=y) for x, y in zip(sf_x, sf_y)]
            _roc_xy = cfcn.calc_avg_roc_curve([cf.get_roc_xy_values(x) for x in _roc])

            # returns the arrays
            return _roc_xy[:, 0], _roc_xy[:, 1]

        # initialisations
        r_data, pW1 = data.rotation, 50.

        # checks to see if the dependent speed has changed
        if 'spd_x_rng' in calc_para:
            # case is a single speed bin range comparison

            # if the dependent speed range has changed then reset the roc curve calculations
            if r_data.comp_spd != calc_para['spd_x_rng']:
                r_data.vel_roc = None

            if r_data.pn_comp is True:
                r_data.vel_roc, r_data.pn_comp = None, False

            # updates the speed comparison flag
            r_data.comp_spd = calc_para['spd_x_rng']

        else:
            # case is the positive/negative speed comparison

            # if the positive/negative comparison flag is not set to true, then reset the roc curve calculations
            if r_data.pn_comp is False:
                r_data.vel_roc, r_data.pn_comp = None, True


        # memory allocation (if the conditions have not been set)
        if r_data.vel_roc is None:
            r_data.vel_roc, r_data.vel_roc_xy, r_data.vel_roc_auc = {}, {}, {}
            r_data.spd_roc, r_data.spd_roc_xy, r_data.spd_roc_auc = {}, {}, {}
            r_data.vel_ci_lo, r_data.vel_ci_hi, r_data.spd_ci_lo, r_data.spd_ci_hi = {}, {}, {}, {}

        for i_rr, rr in enumerate(r_data.r_obj_kine.rot_filt_tot):
            tt, _pW1 = rr['t_type'][0], pW1 * (i_rr / r_data.r_obj_kine.n_filt)
            if tt not in r_data.vel_roc:
                # array dimensions
                calc_ci = None
                if r_data.is_equal_time:
                    vel_sf = dcopy(r_data.vel_sf_rs[tt])
                    if not r_data.pn_comp:
                        spd_sf = dcopy(r_data.spd_sf_rs[tt])
                else:
                    vel_sf = dcopy(r_data.vel_sf[tt])
                    if not r_data.pn_comp:
                        spd_sf = dcopy(r_data.spd_sf[tt])

                # array indexing
                n_trial, n_bin_vel, n_cell = np.shape(vel_sf)
                if r_data.pn_comp:
                    n_bin_vel = int(n_bin_vel / 2)

                # velocity roc memory allocation and initialisations
                r_data.vel_roc[tt] = np.empty((n_cell, n_bin_vel), dtype=object)
                r_data.vel_roc_xy[tt] = np.empty((n_cell, n_bin_vel), dtype=object)
                r_data.vel_roc_auc[tt] = np.zeros((n_cell, n_bin_vel))
                r_data.vel_ci_lo[tt] = -np.ones((n_cell, n_bin_vel, 2))
                r_data.vel_ci_hi[tt] = -np.ones((n_cell, n_bin_vel, 2))

                # speed roc memory allocation and initialisations (non pos/neg comparison only
                if not r_data.pn_comp:
                    n_bin_spd = np.size(spd_sf, axis=1)
                    r_data.spd_roc[tt] = np.empty((n_cell, n_bin_spd), dtype=object)
                    r_data.spd_roc_xy[tt] = np.empty((n_cell, n_bin_spd), dtype=object)
                    r_data.spd_roc_auc[tt] = np.zeros((n_cell, n_bin_spd))
                    r_data.spd_ci_lo[tt] = -np.ones((n_cell, n_bin_spd, 2))
                    r_data.spd_ci_hi[tt] = -np.ones((n_cell, n_bin_spd, 2))

                # calculates the roc curves/integrals for all cells over each phase
                w_str = 'ROC Curve Calculations ({0})...'.format(tt)
                for ic in range(n_cell):
                    # updates the progress bar string
                    self.work_progress.emit(w_str, pW0 + _pW1 + (pW1 / r_data.r_obj_kine.n_filt) * ( + (ic/ n_cell)))

                    # calculates the velocity roc curves values for each velocity bin
                    ii_v = ~np.isnan(vel_sf[:, 0, ic])
                    for i_bin in range(n_bin_vel):
                        if r_data.pn_comp:
                            vel_sf_x = vel_sf[ii_v, n_bin_vel + i_bin, ic]
                            vel_sf_y = vel_sf[ii_v, n_bin_vel - (i_bin + 1), ic]
                        else:
                            # case is single bin comparison
                            if (i_bin == r_data.i_bin_vel[0]) or (i_bin == r_data.i_bin_vel[1]):
                                vel_sf_x, vel_sf_y = resample_spike_freq(vel_sf[ii_v, i_bin, ic])
                            else:
                                vel_sf_x = vel_sf[ii_v, i_bin, ic]
                                if r_data.vel_xi[i_bin] < 0:
                                    vel_sf_y = vel_sf[ii_v, r_data.i_bin_vel[0], ic]
                                else:
                                    vel_sf_y = vel_sf[ii_v, r_data.i_bin_vel[1], ic]

                        r_data.vel_roc[tt][ic, i_bin] = cf.calc_roc_curves(None, None, x_grp=vel_sf_x, y_grp=vel_sf_y)
                        r_data.vel_roc_auc[tt][ic, i_bin] = cf.get_roc_auc_value(r_data.vel_roc[tt][ic, i_bin])
                        r_data.vel_roc_xy[tt][ic, i_bin] = cf.get_roc_xy_values(r_data.vel_roc[tt][ic, i_bin])

                    # calculates the speed roc curves values for each speed bin
                    if not r_data.pn_comp:
                        ii_s = ~np.isnan(spd_sf[:, 0, ic])
                        for i_bin in range(n_bin_spd):
                            calc_roc = True
                            if (i_bin == r_data.i_bin_spd):
                                # spd_sf_x, spd_sf_y = resample_spike_freq(data, r_data, rr, [i_rr, i_bin, ic])
                                spd_sf_x, spd_sf_y = resample_spike_freq(spd_sf[ii_s, i_bin, ic])
                            else:
                                spd_sf_x, spd_sf_y = spd_sf[ii_s, r_data.i_bin_spd, ic], spd_sf[ii_s, i_bin, ic]

                            r_data.spd_roc[tt][ic, i_bin] = cf.calc_roc_curves(None, None, x_grp=spd_sf_x, y_grp=spd_sf_y)
                            r_data.spd_roc_auc[tt][ic, i_bin] = cf.get_roc_auc_value(r_data.spd_roc[tt][ic, i_bin])
                            r_data.spd_roc_xy[tt][ic, i_bin] = cf.get_roc_xy_values(r_data.spd_roc[tt][ic, i_bin])

                    # calculates the confidence intervals for the current (only if bootstrapping count has changed or
                    # the confidence intervals has not already been calculated)
                    if calc_ci is None:
                        if ('auc_stype' in calc_para):
                            # updates the auc statistics calculation type
                            r_data.kine_auc_stats_type = calc_para['auc_stype']

                            # determine if the auc confidence intervals need calculation
                            is_boot = int(calc_para['auc_stype'] == 'Bootstrapping')
                            if is_boot:
                                # if bootstrapping, then determine if the
                                if r_data.n_boot_kine_ci != calc_para['n_boot']:
                                    # if the count has changed, flag the confidence intervals needs updating
                                    r_data.n_boot_kine_ci, calc_ci = calc_para['n_boot'], True
                                else:
                                    # otherwise, recalculate the confidence intervals if they have not been set
                                    calc_ci = np.any(r_data.vel_ci_lo[tt][ic, :, 1] < 0)
                            else:
                                # otherwise, recalculate the confidence intervals if they have not been set
                                calc_ci = np.any(r_data.vel_ci_lo[tt][ic, :, 0] < 0)

                    # calculates the confidence intervals (if required)
                    if calc_ci:
                        # calculates the velocity confidence intervals
                        conf_int_vel = self.calc_roc_conf_intervals(pool, r_data.vel_roc[tt][ic, :],
                                                                    calc_para['auc_stype'], calc_para['n_boot'])
                        r_data.vel_ci_lo[tt][ic, :, is_boot] = conf_int_vel[:, 0]
                        r_data.vel_ci_hi[tt][ic, :, is_boot] = conf_int_vel[:, 1]

                        # calculates the speed confidence intervals
                        if not r_data.pn_comp:
                            conf_int_spd = self.calc_roc_conf_intervals(pool, r_data.spd_roc[tt][ic, :],
                                                                        calc_para['auc_stype'], calc_para['n_boot'])
                            r_data.spd_ci_lo[tt][ic, :, is_boot] = conf_int_spd[:, 0]
                            r_data.spd_ci_hi[tt][ic, :, is_boot] = conf_int_spd[:, 1]

    # def calc_roc_direction_stats(self, calc_para, plot_para, data, pool):
    #     '''
    #
    #     :param calc_para:
    #     :param plot_para:
    #     :param data:
    #     :return:
    #     '''
    #
    #     # parameters and initialisations
    #     p_value, n_boot = 0.05, calc_para['n_boot']
    #     c_data, r_data = data.cluster, data.rotation
    #     is_single_cell = plot_para['plot_scope'] == 'Individual Cell'
    #     p_sig, boot_ci_lo, boot_ci_hi = r_data.p_sig, r_data.boot_ci_lo, r_data.boot_ci_hi
    #     boot_ci_single, is_sig = r_data.boot_ci_single, r_data.is_sig
    #
    #     # retrieves the black condition rotation filtered data
    #     r_filt_black = cf.init_rotation_filter_data(False)
    #     r_obj_black = RotationFilteredData(data, r_filt_black, 0, None, True, 'Whole Experiment', False)
    #     t_spike = r_obj_black.t_spike[0]
    #
    #     # memory allocation
    #     n_cell = np.size(r_obj_black.t_spike[0], axis=0)
    #     A, B = -np.ones((n_cell, 3)), np.zeros((n_cell, 3), dtype=bool)
    #
    #     # if the statistical arrays have been set, but not
    #     if p_sig is None:
    #         # if the MS/DS arrays are not set, then initialise them
    #         p_sig, boot_ci_lo, boot_ci_hi = dcopy(A), dcopy(A), dcopy(A)
    #         is_sig = [dcopy(B) for _ in range(2)]
    #     else:
    #         if n_cell != np.size(p_sig, axis=0):
    #             p_sig, boot_ci_lo, boot_ci_hi = dcopy(A), dcopy(A), dcopy(A)
    #             is_sig = [dcopy(B) for _ in range(2)]
    #
    #     #########################################
    #     ####    WILCOXON STATISTICAL TEST    ####
    #     #########################################
    #
    #     # updates the progress bar string
    #     self.work_progress.emit('Calculating Wilcoxon Stats...', 100. / 3.)
    #
    #     # only calculate the statistics if they haven't already done so
    #     if p_sig[0, 0] == -1:
    #         # calculates the statistical significance between the phases
    #         sp_f0, sp_f = cf.calc_phase_spike_freq(r_obj_black)
    #         _, _, sf_stats, _ = cf.setup_spike_freq_plot_arrays(r_obj_black, sp_f0, sp_f, None)
    #
    #         # determines which cells are motion/direction sensitive
    #         for i in range(len(sf_stats)):
    #             p_sig[:, i] = sf_stats[i]
    #             is_sig[0][:, i] = p_sig[:, i] < p_value
    #
    #     ##########################################
    #     ####    ROC INTEGRAL BOOTSTRAPPING    ####
    #     ##########################################
    #
    #     # if the bootstrapping count is different from the current, then reset the bootstrapping confidence intervals
    #     if r_data.n_boot != calc_para['n_boot']:
    #         boot_ci_lo, boot_ci_hi = dcopy(A), dcopy(A)
    #
    #     # sets the indices of the values that need to be bootstrapped
    #     if is_single_cell:
    #         # updates the progress bar string
    #         self.work_progress.emit('C/I Bootstrapping...', 200. / 3.)
    #
    #         # determines the experiment index and cell count for each experiment
    #         i_cluster = plot_para['i_cluster']
    #         i_expt = cf.get_expt_index(plot_para['plot_exp_name'], c_data)
    #         n_cell_expt = [len(x['tSpike']) for x in c_data]
    #
    #         # determines if the cluster index is valid
    #         if i_cluster > n_cell_expt[i_expt]:
    #             # if the cluster index is not valid, then exit
    #             return None
    #
    #         # retrieves the single cell filtered options
    #         r_obj_sing = RotationFilteredData(data, plot_para['rot_filt'], plot_para['i_cluster'],
    #                                           plot_para['plot_exp_name'], False, 'Individual Cell', False)
    #
    #         # initialisations and memory allocation
    #         p_data = []
    #         for i_filt in range(r_obj_sing.n_filt):
    #             p_data.append([r_obj_sing.t_spike[0][0, :, :], n_boot, [1, 2]])
    #
    #         # calculates the bootstrapped confidence intervals
    #         boot_ci_single = np.array(pool.map(cf.calc_cell_roc_bootstrap_wrapper, p_data))
    #
    #         # # calculates the roc curve integral
    #         # _, roc_int = cf.calc_roc_curves(t_spike[ind, :, :], ind=[1, 2])
    #         # # if roc_int < 0.5:
    #         # #     roc_int = 1 - roc_int
    #         #
    #         # # determines if cw/ccw phase difference is significance
    #         # is_sig[1][ind, 2] = (roc_int < boot_ci_lo[ind, 2]) or (roc_int > boot_ci_hi[ind, 2])
    #     else:
    #         # updates the progress bar string
    #         if boot_ci_lo[0, 1] == -1:
    #             phase_str, pW0 = ['CW/BL', 'CCW/BL', 'CCW/CW'], 100. / 3.
    #             n_phs, roc_int = len(phase_str), dcopy(A)
    #
    #             #
    #             for i_phs, p_str in enumerate(phase_str):
    #                 # updates the progress bar string
    #                 w_str = 'C/I Bootstrapping ({0})...'.format(p_str)
    #                 self.work_progress.emit(w_str, pW0 * (1 + i_phs / n_phs))
    #
    #                 # calculates the bootstrapped confidence intervals for each cell
    #                 ind = np.array([1 * (i_phs > 1), 1 + (i_phs > 0)])
    #                 boot_ci_lo[:, i_phs], boot_ci_hi[:, i_phs], roc_int[:, i_phs] = \
    #                                         self.calc_roc_significance(pool, t_spike, s_type, n_boot, ind)
    #
    #             # sets the significance flags
    #             is_sig[1] = np.logical_or(roc_int < boot_ci_lo, roc_int > boot_ci_hi)
    #
    #     # returns the wilcoxon significance values and bootstrapping confidence intervals
    #     return [p_sig, boot_ci_lo, boot_ci_hi, boot_ci_single, is_sig, r_obj_black]

    # def calc_roc_significance_old(self, pool, t_spike, s_type, n_boot, ind):
    #     '''
    #
    #     :param t_spike:
    #     :return:
    #     '''
    #
    #     # initialisations and memory allocation
    #     n_cell, p_data = np.size(t_spike, axis=0), []
    #     roc_int, roc_xy = np.zeros(n_cell), np.empty(n_cell, dtype=object)
    #
    #     #
    #     for i_cell in range(n_cell):
    #         p_data.append([t_spike[i_cell,:, :], n_boot, ind])
    #
    #         # calculates the
    #         _, roc_int[i_cell] = cf.calc_roc_curves(t_spike[i_cell, :, :], ind=ind)
    #         # if roc_int[i_cell] < 0.5:
    #         #     # ensures the values are greater then 0.5
    #         #     roc_int[i_cell] = 1 - roc_int[i_cell]
    #
    #     # calculates the boot-strapped confidence intervals for
    #     boot_ci = np.array(pool.map(cf.calc_cell_roc_bootstrap_wrapper, p_data))
    #     return boot_ci[:, 0], boot_ci[:, 1], roc_int


    # plot_combined_stimuli_stats(self, rot_filt, plot_exp_name, plot_all_expt, p_value, plot_grid, plot_scope)
