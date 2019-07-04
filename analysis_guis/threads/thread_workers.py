# module import
import gc
import os
import copy
import random
import platform
import numpy as np
import pickle as p
import pandas as pd
import multiprocessing as mp
from numpy.matlib import repmat

# scipy module imports
from scipy.stats import norm
from scipy.spatial.distance import *
from scipy.interpolate import PchipInterpolator as pchip
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

# sklearn module imports
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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

        elif self.thread_job_primary == 'save_data_files':
            # retrieves the parameters
            data, out_info = self.thread_job_para[0], self.thread_job_para[1]

            # case is loading the data files
            thread_data = self.save_data_file(data, out_info)

        elif self.thread_job_primary == 'cluster_matches':
            # case is determining the cluster matches
            thread_data = self.det_cluster_matches()

        elif self.thread_job_primary == 'run_calc_func':
            # case is the calculation functions
            calc_para, plot_para = self.thread_job_para[0], self.thread_job_para[1]
            data, pool, g_para = self.thread_job_para[2], self.thread_job_para[3], self.thread_job_para[4]

            if self.thread_job_secondary == 'Cluster Cross-Correlogram':
                # case is the cc-gram type determinations
                thread_data = self.calc_ccgram_types(calc_para, data.cluster)

            elif self.thread_job_secondary == 'Shuffled Cluster Distances':
                # case is the shuffled cluster distances
                thread_data = self.calc_shuffled_cluster_dist(calc_para, data.cluster)

            elif self.thread_job_secondary == 'Direction ROC Curves (Single Cell)':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, g_para, ['condition'])

                # case is the shuffled cluster distances
                if not self.calc_cond_roc_curves(data, pool, calc_para, plot_para, g_para, False, 100.):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

            elif self.thread_job_secondary == 'Direction ROC Curves (Whole Experiment)':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, g_para, ['condition', 'phase'])

                # calculates the phase roc-curves for each cell
                if not self.calc_cond_roc_curves(data, pool, calc_para, plot_para, g_para, False, 33.):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                # calculates the phase roc curve/significance values
                self.calc_phase_roc_curves(data, calc_para, 66.)
                self.calc_phase_roc_significance(calc_para, g_para, data, pool, 100.)

            elif self.thread_job_secondary == 'Direction ROC AUC Histograms':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, g_para, ['condition'])

                # calculates the phase roc-curves for each cell
                if not self.calc_cond_roc_curves(data, pool, calc_para, plot_para, g_para, True, 100., True):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

            elif self.thread_job_secondary == 'Velocity ROC Curves (Single Cell)':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, g_para, ['vel'], other_para=True)

                # calculates the binned kinematic spike frequencies
                self.calc_binned_kinemetic_spike_freq(data, plot_para, calc_para)
                self.calc_kinematic_roc_curves(data, pool, calc_para, g_para, 50.)

            elif self.thread_job_secondary == 'Velocity ROC Curves (Whole Experiment)':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, g_para, ['vel'], other_para=True)

                # calculates the binned kinematic spike frequencies
                self.calc_binned_kinemetic_spike_freq(data, plot_para, calc_para)
                self.calc_kinematic_roc_curves(data, pool, calc_para, g_para, 50.)

            elif self.thread_job_secondary == 'Velocity ROC Curves (Pos/Neg Comparison)':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, g_para, ['vel'], other_para=True)

                # calculates the binned kinematic spike frequencies
                self.calc_binned_kinemetic_spike_freq(data, plot_para, calc_para)
                self.calc_kinematic_roc_curves(data, pool, calc_para, g_para, 50.)

            elif self.thread_job_secondary == 'Condition ROC Curve Comparison':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, g_para, ['phase'])

                # calculates the phase roc-curves for each cell
                if not self.calc_cond_roc_curves(data, pool, calc_para, plot_para, g_para, True, 33.):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                # calculates the phase roc curve/significance values
                self.calc_phase_roc_curves(data, calc_para, 66.)
                self.calc_phase_roc_significance(calc_para, g_para, data, pool, 100.)

            elif self.thread_job_secondary == 'Motion/Direction Selectivity Cell Grouping Scatterplot':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, g_para, ['condition', 'phase'])

                # calculates the phase roc-curves for each cell
                if not self.calc_cond_roc_curves(data, pool, calc_para, plot_para, g_para, True, 33.):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                # calculates the phase roc curve/significance values
                self.calc_phase_roc_curves(data, calc_para, 66.)
                self.calc_phase_roc_significance(calc_para, g_para, data, pool, 100.)

                if cf.det_valid_vis_expt(data, True):
                    if not self.calc_dirsel_group_types(data, pool, calc_para, plot_para, g_para):
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

            elif self.thread_job_secondary == 'Rotation/Visual Stimuli Response Statistics':
                # calculates the phase roc curve/significance values
                self.calc_phase_roc_curves(data, calc_para, 50.)

                # calculates the direction/selection group types
                if not self.calc_dirsel_group_types(data, pool, calc_para, plot_para, g_para):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)

            elif self.thread_job_secondary == 'Combined Direction ROC Curves (Whole Experiment)':
                # checks that the conditions are correct for running the function
                if not self.check_combined_conditions(calc_para, plot_para):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, g_para, ['condition', 'phase', 'visual'])

                # initisalises the rotational filter (if not initialised already)
                if plot_para['rot_filt'] is None:
                    plot_para['rot_filt'] = cf.init_rotation_filter_data(False)

                # creates a copy of the plotting/calculation parameters
                _plot_para, _calc_para = dcopy(plot_para), dcopy(calc_para)

                # adds motordrifting (if the visual expt type)
                if calc_para['vis_expt_type'] == 'MotorDrifting':
                    _plot_para['rot_filt']['t_type'].append('MotorDrifting')

                # resets the flags to use the full rotation/visual phases
                _calc_para['use_full_rot'], _calc_para['use_full_vis'] = True, True

                # calculates the phase roc-curves for each cell
                if not self.calc_cond_roc_curves(data, pool, _calc_para, _plot_para, g_para, False, 33.):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                # calculates the phase roc curve/significance values
                self.calc_phase_roc_curves(data, _calc_para, 66.)
                if (calc_para['vis_expt_type'] == 'UniformDrifting') and \
                                                (calc_para['grp_stype'] != 'Wilcoxon Paired Test'):
                    # sets up the visual rotation filter
                    r_filt_v = cf.init_rotation_filter_data(False)
                    r_filt_v['t_type'], r_filt_v['is_ud'], r_filt_v['t_cycle'] = ['UniformDrifting'], [True], ['15']

                    # retrieves the visual filter object
                    plot_exp_name, plot_all_expt = plot_para['plot_exp_name'], plot_para['plot_all_expt']
                    r_obj_vis, ind_type = cf.split_unidrift_phases(data, r_filt_v, None, plot_exp_name, plot_all_expt,
                                                                   'Whole Experiment', 2.)

                    # calculates the full uniform-drifting curves
                    self.calc_ud_roc_curves(data, r_obj_vis, ind_type, 66.)

                # calculates the direction selection types
                if not self.calc_dirsel_group_types(data, pool, _calc_para, _plot_para, g_para):
                    self.is_ok = False

                # calculates the partial roc curves
                self.calc_partial_roc_curves(data, calc_para, plot_para, 66.)

            # elif self.thread_job_secondary == 'Kinematic Spiking Frequency':
            #     # calculates the binned kinematic spike frequencies
            #     self.calc_binned_kinemetic_spike_freq(data, plot_para, calc_para, False)

            elif self.thread_job_secondary == 'Rotation Direction LDA':
                # if the solver parameter have not been set, then initalise them
                d_data = data.discrim.dir

                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, g_para, ['lda'], other_para=d_data)

                # sets up the lda values
                r_filt, i_expt, i_cell, n_trial_max, status = self.setup_lda(data, calc_para, d_data)
                if status == 0:
                    # if there was an error in the calculations, then return an error flag
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return
                elif status == 2:
                    # if an update in the calculations is required, then run the rotation LDA analysis
                    if not cfcn.run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max,
                                            d_data=d_data, w_prog=self.work_progress):
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

            elif self.thread_job_secondary == 'Temporal Duration/Offset LDA Analysis':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, g_para, ['lda'], other_para=data.discrim.temp)

                # if the temporal data parameters have changed/has not been initialised then calculate the values
                if data.discrim.temp.lda is None:
                    # checks to see if any base LDA calculation parameters have been altered
                    self.check_altered_para(data, calc_para, g_para, ['lda'], other_para=data.discrim.dir)

                    # sets up the important arrays for the LDA
                    r_filt, i_expt, i_cell, n_trial_max, status = self.setup_lda(data, calc_para, data.discrim.dir)
                    if status == 0:
                        # if there was an error in the calculations, then return an error flag
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

                    # if an update in the calculations is required, then run the temporal LDA analysis
                    if status == 2:
                        if not self.run_temporal_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
                            # if there was an error in the calculations, then return an error flag
                            self.is_ok = False
                            self.work_finished.emit(thread_data)
                            return

            elif self.thread_job_secondary == 'Individual LDA Analysis':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, g_para, ['lda'], other_para=data.discrim.indiv)
                self.check_altered_para(data, calc_para, g_para, ['lda'], other_para=data.discrim.dir)

                # sets up the important arrays for the LDA
                r_filt, i_expt, i_cell, n_trial_max, status = self.setup_lda(data, calc_para, data.discrim.dir, True)
                if status == 0:
                    # if there was an error in the calculations, then return an error flag
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return
                elif status == 2:
                    # if an update in the calculations is required, then run the rotation LDA analysis
                    if not cfcn.run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max,
                                            d_data=data.discrim.dir, w_prog=self.work_progress):
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

                # if the individual data parameters have changed/has not been initialised then calculate the values
                if data.discrim.indiv.lda is None:
                    # runs the individual LDA
                    if not self.run_individual_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
                        # if there was an error in the calculations, then return an error flag
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

            elif self.thread_job_secondary == 'Shuffled LDA Analysis':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, g_para, ['lda'], other_para=data.discrim.shuffle)
                self.check_altered_para(data, calc_para, g_para, ['lda'], other_para=data.discrim.dir)

                # sets up the important arrays for the LDA
                r_filt, i_expt, i_cell, n_trial_max, status = self.setup_lda(data, calc_para, data.discrim.dir, True)
                if status == 0:
                    # if there was an error in the calculations, then return an error flag
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return
                elif status == 2:
                    # if an update in the calculations is required, then run the rotation LDA analysis
                    if not cfcn.run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max,
                                            d_data=data.discrim.dir, w_prog=self.work_progress):
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

                # runs the shuffled LDA
                if not self.run_shuffled_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
                    # if there was an error in the calculations, then return an error flag
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

            elif self.thread_job_secondary == 'Pooling LDA Analysis':
                # resets the minimum cell count and checks if the pooled parameters have been altered
                calc_para['lda_para']['n_cell_min'] = calc_para['n_cell_min']
                self.check_altered_para(data, calc_para, g_para, ['lda'], other_para=data.discrim.part)

                # if the pooled data parameters have changed/has not been initialised then calculate the values
                if data.discrim.part.lda is None:
                    # checks to see if any base LDA calculation parameters have been altered
                    self.check_altered_para(data, calc_para, g_para, ['lda'], other_para=data.discrim.dir)

                    # sets up the important arrays for the LDA
                    r_filt, i_expt, i_cell, n_trial_max, status = self.setup_lda(data, calc_para, data.discrim.dir, True)
                    if status == 0:
                        # if there was an error in the calculations, then return an error flag
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return
                    elif status == 2:
                        # if an update in the calculations is required, then run the rotation LDA analysis
                        if not cfcn.run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max,
                                                d_data=data.discrim.dir, w_prog=self.work_progress):
                            self.is_ok = False
                            self.work_finished.emit(thread_data)
                            return

                    # runs the partial LDA
                    if not self.run_partial_lda(pool, data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
                        # if there was an error in the calculations, then return an error flag
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

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
                if isinstance(data_nw, dict):
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

    def save_data_file(self, data, out_info):
        '''

        :return:
        '''

        # updates the progressbar
        self.work_progress.emit('Saving Data To File...', 50.0)

        # sets the output file name
        out_file = os.path.join(out_info['inputDir'], '{0}.mdata'.format(out_info['dataName']))

        # outputs the data to file
        with open(out_file, 'wb') as fw:
            p.dump(data, fw)

        # updates the progressbar
        self.work_progress.emit('Data Save Complete!', 100.0)

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
            z_max = 1.0
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

    ################################
    ####    LDA CALCULATIONS    ####
    ################################

    def setup_lda(self, data, calc_para, d_data, return_reqd_arr=False):
        '''

        :param data:
        :param calc_para:
        :return:
        '''

        def det_valid_cells(data, ind, lda_para):
            '''

            :param cluster:
            :param lda_para:
            :return:
            '''

            # determines the cells that are in the valid regions (RSPg and RSPd)
            cluster = data.cluster[ind]
            is_valid = np.logical_or(cluster['chRegion'] == 'RSPg', cluster['chRegion'] == 'RSPd')

            # if the cell types have been set, then remove the cells that are not the selected type
            if lda_para['cell_types'] == 'Narrow Spike Cells':
                # case is narrow spikes have been selected
                is_valid[data.classify.grp_str[ind] == 'Wid'] = False
            elif lda_para['cell_types'] == 'Wide Spike Cells':
                # case is wide spikes have been selected
                is_valid[data.classify.grp_str[ind] == 'Nar'] = False

            # determines if the individual LDA has been calculated
            d_data_i = data.discrim.indiv
            if d_data_i.lda is not None:
                # if so, determines the trial type corresponding to the black direction decoding type
                i_type = next((i for i in range(len(d_data_i.ttype)) if d_data_i.ttype[i] == 'Black'), None)
                if (i_type is not None) and (ind in d_data_i.i_expt):
                    # if the black decoding type is present, remove the cells which have a decoding accuracy above max
                    ind_g = np.where(d_data_i.i_expt == ind)[0][0]
                    ii = np.where(d_data_i.i_cell[ind_g])[0]
                    is_valid[ii[100. * d_data_i.y_acc[ind_g][:, i_type + 1] > lda_para['y_acc_max']]] = False
                    is_valid[ii[100. * d_data_i.y_acc[ind_g][:, i_type + 1] < lda_para['y_acc_min']]] = False

            # if the number of valid cells is less than the reqd count, then set all cells to being invalid
            if np.sum(is_valid) < lda_para['n_cell_min']:
                is_valid[:] = False

            # returns the valid index array
            return is_valid

        # initialisations
        lda_para, s_flag = calc_para['lda_para'], 2
        if len(lda_para['comp_cond']) < 2:
            # if less than 2 trial conditions are selected then output an error to screen
            e_str = 'At least 2 trial conditions must be selected before running this function.'
            self.work_error.emit(e_str, 'Invalid LDA Analysis Parameters')

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
        n_trial_max = np.min(n_trial[n_trial > 0])

        # determines if the number of trials has changed (and if the lda calculation values have been set)
        if (n_trial_max == d_data.ntrial) and (d_data.lda is not None):
            # if there is no change and the values are set, then exit with a true flag
            s_flag = 1
            if not return_reqd_arr:
                return None, None, None, None, s_flag

        # determines the valid cells from each of the loaded experiments
        i_cell = np.array([det_valid_cells(data, ic, lda_para) for ic in range(len(data.cluster))])

        # determines if there are any valid loaded experiments
        i_expt = np.where([(np.any(x) and np.min(n_trial[:, i_ex]) >= lda_para['n_trial_min'])
                            for i_ex, x in enumerate(i_cell)])[0]
        if len(i_expt) == 0:
            # if there are no valid experiments, then output an error message to screen
            e_str = 'The LDA function can''t be run using the currently loaded experiments/parameter configuration. ' \
                    'Either load more experiments or alter the calculation parameters.'
            self.work_error.emit(e_str, 'Invalid LDA Experiments Loaded')

            # returns a false flag
            return None, None, None, None, 0

        # returns the import values for the LDA calculations
        return r_filt, i_expt, i_cell[i_expt], n_trial_max, s_flag

    def run_temporal_lda(self, data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
        '''

        :param data:
        :param calc_para:
        :param r_filt:
        :param i_expt:
        :param i_cell:
        :param n_trial_max:
        :return:
        '''

        # initialisations and memory allocation
        d_data, w_prog = data.discrim.temp, self.work_progress
        d_data.lda, d_data.y_acc = np.empty(2, dtype=object), np.empty(2, dtype=object)

        # retrieves the rotation phase duration
        r_obj = RotationFilteredData(data, r_filt, None, None, True, 'Whole Experiment', False)
        t_phase = r_obj.t_phase[0][0]

        ################################################
        ####    DIFFERING PHASE LDA CALCULATIONS    ####
        ################################################

        # creates a copy of the calculation parameters for the differing phase duration LDA calculations
        calc_para_phs = dcopy(calc_para)
        calc_para_phs['t_ofs_rot'] = 0

        # memory allocation
        dt_phs = np.arange(calc_para['dt_phase'], t_phase, calc_para['dt_phase'])
        d_data.lda[0], d_data.y_acc[0] = np.empty(len(dt_phs), dtype=object), np.empty(len(dt_phs), dtype=object)

        # loops through each of the phase discretisations calculating the LDA calculations
        n_phs = len(dt_phs)
        for i_phs in range(n_phs):
            # updates the progress bar
            w_str = 'Duration LDA Calculations (Group {0} of {1})'.format(i_phs + 1, n_phs)
            w_prog.emit(w_str, 50. * ((i_phs + 1)/ n_phs))

            # updates the phase duration parameter
            calc_para_phs['t_phase_rot'] = dt_phs[i_phs]

            # runs the rotation analysis for the current configuration
            result = cfcn.run_rot_lda(data, calc_para_phs, r_filt, i_expt, i_cell, n_trial_max)
            if isinstance(result, bool):
                # if there was an error, then return a false flag value
                return False
            else:
                # otherwise, store the lda/accuracy values
                d_data.lda[0][i_phs], d_data.y_acc[0][i_phs] = result[0], result[1]

        #################################################
        ####    DIFFERING OFFSET LDA CALCULATIONS    ####
        #################################################

        # creates a copy of the calculation parameters for the differing offset LDA calculations
        calc_para_ofs = dcopy(calc_para)
        calc_para_ofs['t_phase_rot'] = calc_para['t_phase_const']

        # sets the differing phase/offset value arrays
        dt_ofs = np.arange(0., t_phase - calc_para['t_phase_const'], calc_para['t_phase_const'])
        d_data.lda[1], d_data.y_acc[1] = np.empty(len(dt_ofs), dtype=object), np.empty(len(dt_ofs), dtype=object)

        # loops through each of the phase discretisations calculating the LDA calculations
        n_ofs = len(dt_ofs)
        for i_ofs in range(n_ofs):
            # updates the progress bar
            w_str = 'Offset LDA Calculations (Group {0} of {1})'.format(i_ofs + 1, n_ofs)
            w_prog.emit(w_str, 50. * (1 + ((i_ofs + 1) / n_ofs)))

            # updates the phase duration parameter
            calc_para_ofs['t_ofs_rot'] = dt_ofs[i_ofs]

            # runs the rotation analysis for the current configuration
            result = cfcn.run_rot_lda(data, calc_para_ofs, r_filt, i_expt, i_cell, n_trial_max)
            if isinstance(result, bool):
                # if there was an error, then return a false flag value
                return False
            else:
                # otherwise, store the lda/accuracy values
                d_data.lda[1][i_ofs], d_data.y_acc[1][i_ofs] = result[0], result[1]

        #######################################
        ####    HOUSE KEEPING EXERCISES    ####
        #######################################

        # retrieves the LDA solver parameter fields
        lda_para = calc_para['lda_para']

        # sets the solver parameters
        d_data.exp_name = result[2]
        d_data.i_expt = i_expt
        d_data.i_cell = i_cell
        cfcn.set_lda_para(d_data, lda_para, r_filt, n_trial_max)

        # sets the other calculation parameters
        d_data.dt_phs = calc_para['dt_phase']
        d_data.dt_ofs = calc_para['dt_ofs']
        d_data.phs_const = calc_para['t_phase_const']

        # sets the other variables/parameters of interest
        d_data.xi_phs = dt_phs
        d_data.xi_ofs = dt_ofs

        # returns a true value indicating the calculations were successful
        return True

    def run_shuffled_lda(self, data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
        '''

        :param data:
        :param calc_para:
        :param r_filt:00
        :param i_expt:
        :param i_cell:
        :param n_trial_max:
        :return:
        '''

        # initialisations and memory allocation
        d_data, w_prog = data.discrim.shuffle, self.work_progress
        if d_data.lda is not None:
            return True

        # retrieves the phase duration/offset values
        t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)
        if t_ofs is None:
            t_ofs, t_phase = 0, 3.5346

        ###############################################
        ####    SHUFFLED TRIAL LDA CALCULATIONS    ####
        ###############################################

        # creates a reduce data object and creates the rotation filter object
        n_ex, n_sh, n_cond = len(i_expt), calc_para['n_shuffle'], len(r_filt['t_type'])
        d_data.y_acc = np.empty((n_ex, n_cond + 1, n_sh), dtype=object)
        n_sp = np.empty((n_ex, n_sh), dtype=object)

        # runs the LDA for each of the shuffles
        for i_sh in range(n_sh):
            # updates the progressbar
            w_str = 'Shuffled Trial LDA (Shuffle #{0} of {1})'.format(i_sh + 1, n_sh)
            w_prog.emit(w_str, 100. * (i_sh / n_sh))

            # runs the rotation analysis for the current configuration
            result = cfcn.run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max, is_shuffle=True)
            if isinstance(result, bool):
                # if there was an error, then return a false flag value
                return False
            else:
                # otherwise, store the lda/accuracy values
                d_data.y_acc[:, :, i_sh], n_sp[:, i_sh] = result[1], result[3]
                if i_sh == 0:
                    # sets the experiment names (for the first shuffle only)
                    d_data.exp_name == result[2]

        #######################################
        ####    HOUSE KEEPING EXERCISES    ####
        #######################################

        # retrieves the LDA solver parameter fields
        lda_para = calc_para['lda_para']

        # sets the solver parameters
        d_data.lda = 1
        d_data.i_expt = i_expt
        d_data.i_cell = i_cell
        cfcn.set_lda_para(d_data, lda_para, r_filt, n_trial_max)

        # sets the phase offset/duration parameters
        d_data.tofs = t_ofs
        d_data.tphase = t_phase
        d_data.usefull = calc_para['use_full_rot']

        # sets the other parameters
        d_data.nshuffle = n_sh
        # d_data.bsz = calc_para['b_sz']

        # calculates the correlations
        n_sp_tot = [np.dstack(x) for x in n_sp]
        cfcn.calc_noise_correl(d_data, n_sp_tot)

        # returns a true value indicating the calculations were successful
        return True

    def run_individual_lda(self, data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
        '''
        
        :param data: 
        :param calc_para: 
        :param r_filt: 
        :param i_expt: 
        :param i_cell: 
        :param n_trial_max: 
        :return: 
        '''

        # initialisations and memory allocation
        d_data, w_prog = data.discrim.indiv, self.work_progress

        ################################################
        ####    INDIVIDUAL CELL LDA CALCULATIONS    ####
        ################################################

        # creates a reduce data object and creates the rotation filter object
        n_ex = len(i_expt)
        A = np.empty(n_ex, dtype=object)
        d_data.y_acc, d_data.exp_name = dcopy(A), dcopy(A)
        n_cell = [len(i_c) for i_c in i_cell]

        #
        for i_ex in range(n_ex):
            # creates a copy a copy of the accepted cell array for the analysis
            _i_cell = np.zeros(n_cell[i_ex], dtype=bool)
            _n_cell = np.sum(i_cell[i_ex])
            d_data.y_acc[i_ex] = np.zeros((_n_cell, 1 + len(calc_para['lda_para']['comp_cond'])))

            # runs the LDA analysis for each of the cells
            for i, i_c in enumerate(np.where(i_cell[i_ex])[0]):
                # updates the progressbar
                w_str = 'Single Cell LDA (Cell {0}/{1}, Expt {2}/{3})'.format(i + 1, _n_cell, i_ex + 1, n_ex)
                w_prog.emit(w_str, 100. * (i_ex + i / _n_cell) / n_ex)

                # sets the cell for analysis and runs the LDA
                _i_cell[i_c] = True
                results = cfcn.run_rot_lda(data, calc_para, r_filt, [i_expt[i_ex]], [_i_cell], n_trial_max)
                if isinstance(results, bool):
                    # if there was an error, then return a false flag value
                    return False
                else:
                    # otherwise, reset the cell boolear flag
                    _i_cell[i_c] = False

                # stores the results from the single cell LDA
                d_data.y_acc[i_ex][i, :] = results[1]
                if i == 0:
                    # if the first iteration, then store the experiment name
                    d_data.exp_name[i_ex] = results[2]

        #######################################
        ####    HOUSE KEEPING EXERCISES    ####
        #######################################

        # retrieves the LDA solver parameter fields
        lda_para = calc_para['lda_para']
        t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)

        # sets the solver parameters
        d_data.lda = 1
        d_data.i_expt = i_expt
        d_data.i_cell = i_cell
        cfcn.set_lda_para(d_data, lda_para, r_filt, n_trial_max)

        # sets the phase offset/duration
        d_data.tofs = t_ofs
        d_data.tphase = t_phase
        d_data.usefull = calc_para['use_full_rot']

        # returns a true value indicating the calculations were successful
        return True

    def run_partial_lda(self, pool, data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
        '''

        :param data:
        :param calc_para:
        :param r_filt:
        :param i_expt:
        :param i_cell:
        :param n_trial_max:
        :return:
        '''

        def run_partial_lda_expt(data, calc_para, r_filt, i_expt, i_cell, n_trial_max, n_cell):
            '''

            :param data:
            :param calc_para:
            :param r_filt:
            :param i_expt:
            :param i_cell:
            :param n_trial_max:
            :param xi:
            :return:
            '''

            # sets the required number of cells for the LDA analysis
            for i_ex in range(len(i_expt)):
                # determines the original valid cells for the current experiment
                ii = np.where(i_cell[i_ex])[0]

                # from these cells, set n_cell cells as being valid (for analysis purposes)
                i_cell[i_ex][:] = False
                i_cell[i_ex][ii[np.random.permutation(len(ii))][:n_cell]] = True

            # runs the LDA
            results = cfcn.run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max)

            # returns the decoding accuracy values
            return results[1]

        # initialisations
        d_data = data.discrim.part

        #############################################
        ####    PARTIAL CELL LDA CALCULATIONS    ####
        #############################################

        # initialisations
        xi = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
        y_acc_d, w_prog = data.discrim.dir.y_acc, self.work_progress
        n_expt, n_cond, n_xi, n_sh = min([3, len(i_expt)]), len(r_filt['t_type']), len(xi), calc_para['n_shuffle']

        # retrieves the top n_expt experiments based on the base decoding accuracy
        ii = np.sort(np.argsort(-np.prod(y_acc_d, axis=1))[:n_expt])
        i_expt, i_cell = i_expt[ii], i_cell[ii]

        # memory allocation
        d_data.y_acc = np.zeros((n_expt, n_cond + 1, n_xi, n_sh))

        # loops through each of the cell counts calculating the partial LDA
        for i_sh in range(n_sh):
            # updates the progressbar
            w_str = 'Pooling LDA Calculations (Shuffle {0} of {1})'.format(i_sh + 1, n_sh)
            w_prog.emit(w_str, 100. * (i_sh / n_sh))

            # runs the analysis based on the operating system
            if 'Windows' in platform.platform():
                # case is Richard's local computer

                # initialisations and memory allocation
                p_data = [[] for _ in range(n_xi)]
                for i_xi in range(n_xi):
                    p_data[i_xi].append(data)
                    p_data[i_xi].append(calc_para)
                    p_data[i_xi].append(r_filt)
                    p_data[i_xi].append(i_expt)
                    p_data[i_xi].append(i_cell)
                    p_data[i_xi].append(n_trial_max)
                    p_data[i_xi].append(xi[i_xi])

                # runs the pool object to run the partial LDA
                p_results = pool.map(cfcn.run_part_lda_pool, p_data)
                for i_xi in range(n_xi):
                    j_xi = xi.index(p_results[i_xi][0])
                    d_data.y_acc[:, :, j_xi, i_sh] = p_results[i_xi][1]
            else:
                # case is Subiculum

                # initialisations and memory allocation
                for i_xi in range(n_xi):
                    d_data.y_acc[:, :, i_xi, i_sh] = run_partial_lda_expt(
                        data, calc_para, r_filt, i_expt, dcopy(i_cell), n_trial_max, xi[i_xi]
                    )

        #######################################
        ####    HOUSE KEEPING EXERCISES    ####
        #######################################

        # retrieves the LDA solver parameter fields
        lda_para = calc_para['lda_para']
        t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)

        # sets the solver parameters
        d_data.lda = 1
        d_data.i_expt = i_expt
        d_data.i_cell = i_cell
        cfcn.set_lda_para(d_data, lda_para, r_filt, n_trial_max, ignore_list=['n_cell_min'])

        # sets the phase offset/duration parametrs
        d_data.tofs = t_ofs
        d_data.tphase = t_phase
        d_data.usefull = calc_para['use_full_rot']

        # sets the other parameters/arrays
        d_data.nshuffle = n_sh
        d_data.cellminpart = calc_para['n_cell_min']
        d_data.xi = xi

        # returns a true value indicating the calculations were successful
        return True

    ######################################
    ####    ROC CURVE CALCULATIONS    ####
    ######################################

    def calc_partial_roc_curves(self, data, calc_para, plot_para, pW):
        '''

        :param data:
        :param calc_para:
        :param plot_para:
        :param pW:
        :return:
        '''

        # memory allocation
        r_data = data.rotation
        r_data.part_roc, r_data.part_roc_xy, r_data.part_roc_auc = {}, {}, {}

        # initisalises the rotational filter (if not initialised already)
        if plot_para['rot_filt'] is None:
            plot_para['rot_filt'] = cf.init_rotation_filter_data(False)

        # calculates the partial roc curves for each of the trial conditions
        for tt in plot_para['rot_filt']['t_type']:
            # if tt not in r_data.part_roc:
            r_data.part_roc[tt], r_data.part_roc_xy[tt], r_data.part_roc_auc[tt] = \
                                        self.calc_phase_roc_curves(data, calc_para, pW, t_type=tt)

    def calc_phase_roc_curves(self, data, calc_para, pW, t_type=None):
        '''

        :param calc_para:
        :param plot_para:
        :param data:
        :param pool:
        :return:
        '''

        # parameters and initialisations
        phase_str, r_data = ['CW/BL', 'CCW/BL', 'CCW/CW'], data.rotation

        # if the black phase is calculated already, then exit the function
        if (r_data.phase_roc is not None) and (t_type is None):
            return

        # retrieves the offset parameters
        t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)

        # sets up the black phase data filter and returns the time spikes
        r_filt = cf.init_rotation_filter_data(False)

        if t_type is None:
            r_data.r_obj_black = r_obj = RotationFilteredData(data, r_filt, 0, None, True, 'Whole Experiment', False,
                                                              t_phase=t_phase, t_ofs=t_ofs)
        else:
            r_filt['t_type'] = [t_type]
            r_obj = RotationFilteredData(data, r_filt, 0, None, True, 'Whole Experiment', False,
                                         t_phase=t_phase, t_ofs=t_ofs)

        # retrieves the time spikes and sets the roc class fields for update
        t_spike = r_obj.t_spike[0]

        # memory allocation
        n_cell = np.size(t_spike, axis=0)
        roc = np.empty((n_cell, len(phase_str)), dtype=object)
        roc_xy = np.empty(n_cell, dtype=object)
        roc_auc = np.ones((n_cell, len(phase_str)))

        # calculates the roc curves/integrals for all cells over each phase
        for i_phs, p_str in enumerate(phase_str):
            # updates the progress bar string
            w_str = 'ROC Curve Calculations ({0})...'.format(p_str)
            self.work_progress.emit(w_str, pW * i_phs / len(phase_str))

            # calculates the bootstrapped confidence intervals for each cell
            ind = np.array([1 * (i_phs > 1), 1 + (i_phs > 0)])
            for i_cell in range(n_cell):
                # calculates the roc curve/auc integral
                roc[i_cell, i_phs] = cf.calc_roc_curves(t_spike[i_cell, :, :], ind=ind)
                roc_auc[i_cell, i_phs] = cf.get_roc_auc_value(roc[i_cell, i_phs])

                # if the CW/CCW phase interaction, then set the roc curve x/y coordinates
                if (i_phs + 1) == len(phase_str):
                    roc_xy[i_cell] = cf.get_roc_xy_values(roc[i_cell, i_phs])

        # case is the rotation (black) condition
        if t_type is None:
            r_data.phase_roc, r_data.phase_roc_xy, r_data.phase_roc_auc = roc, roc_xy, roc_auc
        else:
            return roc, roc_xy, roc_auc

    def calc_ud_roc_curves(self, data, r_obj_vis, ind_type, pW):
        '''

        :param data:
        :param r_obj_vis:
        :param calc_para:
        :param pW:
        :return:
        '''

        # parameters and initialisations
        phase_str, r_data, ind = ['CW/BL', 'CCW/BL', 'CCW/CW'], data.rotation, np.array([0, 1])
        ind_CC, ind_CCW = ind_type[0][0], ind_type[1][0]

        # if the uniformdrifting phase is calculated already, then exit the function
        if r_data.phase_roc_ud is not None:
            return

        # sets the time spike array
        t_spike = r_obj_vis.t_spike
        n_trial = np.min([np.size(t_spike[0], axis=1), np.size(t_spike[1], axis=1)])

        # memory allocation
        n_cell = np.size(r_obj_vis.t_spike[0], axis=0)
        roc = np.empty((n_cell, len(phase_str)), dtype=object)
        roc_xy = np.empty(n_cell, dtype=object)
        roc_auc = np.ones((n_cell, len(phase_str)))

        # calculates the roc curves/integrals for all cells over each phase
        for i_phs, p_str in enumerate(phase_str):
            # updates the progress bar string
            w_str = 'ROC Curve Calculations ({0})...'.format(p_str)
            self.work_progress.emit(w_str, 100 * pW * i_phs / len(phase_str))

            # loops through each of the cells calculating the roc curves (and associated values)
            for i_cell in range(n_cell):
                # sets the time spike arrays depending on the phase type
                if (i_phs + 1) == len(phase_str):
                    t_spike_phs = np.vstack((t_spike[ind_CC][i_cell, :n_trial, 1],
                                             t_spike[ind_CCW][i_cell, :n_trial, 1])).T
                else:
                    t_spike_phs = t_spike[i_phs][i_cell, :, :]

                # calculates the roc curve/auc integral
                roc[i_cell, i_phs] = cf.calc_roc_curves(t_spike_phs, ind=ind)
                roc_auc[i_cell, i_phs] = cf.get_roc_auc_value(roc[i_cell, i_phs])

                # if the CW/CCW phase interaction, then set the roc curve x/y coordinates
                if (i_phs + 1) == len(phase_str):
                    roc_xy[i_cell] = cf.get_roc_xy_values(roc[i_cell, i_phs])

        # sets the final
        r_data.phase_roc_ud, r_data.phase_roc_xy_ud, r_data.phase_roc_auc_ud = roc, roc_xy, roc_auc

    def calc_cond_roc_curves(self, data, pool, calc_para, plot_para, g_para, calc_cell_grp, pW,
                             force_black_calc=False):
        '''

        :param calc_para:
        :param plot_para:
        :param data:
        :param pool:
        :return:
        '''

        # parameters and initialisations
        t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)
        r_obj_sig, plot_scope, c_lvl = None, 'Whole Experiment', float(g_para['roc_clvl'])
        phase_str, r_data = ['CW/BL', 'CCW/BL', 'CCW/CW'], data.rotation

        # initisalises the rotational filter (if not initialised already)
        if plot_para['rot_filt'] is None:
            plot_para['rot_filt'] = cf.init_rotation_filter_data(False)

        # sets the condition types (ensures that the black phase is always included)
        t_type = dcopy(plot_para['rot_filt']['t_type'])
        if 'Black' not in t_type:
            t_type = ['Black'] + t_type

        if 'vis_expt_type' in calc_para:
            if calc_para['vis_expt_type'] == 'MotorDrifting':
                t_type += ['MotorDrifting']

        # retrieves the rotation phase offset time/duration
        if t_ofs is not None:
            # if the values are not none, and do not match previous values, then reset the stored roc array
            if (r_data.t_ofs_rot != t_ofs) or (r_data.t_phase_rot != t_phase):
                r_data.t_ofs_rot, r_data.t_phase_rot, r_data.cond_roc = t_ofs, t_phase, None
        elif 'use_full_rot' in calc_para:
            # if using the full rotation, and the previous calculations were made using non-full rotation phases,
            # the reset the stored roc array
            if (r_data.t_ofs_rot > 0):
                r_data.t_ofs_rot, r_data.t_phase_rot, r_data.cond_roc = -1, -1, None

        # sets up a base filter with only the
        r_filt_base = cf.init_rotation_filter_data(False)
        r_filt_base['t_type'] = [x for x in t_type if x != 'UniformDrifting']

        # sets up the black phase data filter and returns the time spikes
        r_obj = RotationFilteredData(data, r_filt_base, None, plot_para['plot_exp_name'], True, plot_scope, False,
                                     t_ofs=t_ofs, t_phase=t_phase)
        if not r_obj.is_ok:
            # if there was an error, then output an error to screen
            self.work_error.emit(r_obj.e_str, 'Incorrect Analysis Function Parameters')
            return False

        # memory allocation (if the conditions have not been set)
        if r_data.cond_roc is None:
            r_data.cond_roc, r_data.cond_roc_xy, r_data.cond_roc_auc = {}, {}, {}
            r_data.cond_gtype, r_data.cond_auc_sig, r_data.cond_i_expt, r_data.cond_cl_id = {}, {}, {}, {}
            r_data.cond_ci_lo, r_data.cond_ci_hi, r_data.r_obj_cond = {}, {}, {}
            r_data.phase_gtype, r_data.phase_auc_sig, r_data.phase_roc = None, None, None

        for i_rr, rr in enumerate(r_obj.rot_filt_tot):
            # sets the trial type
            tt = rr['t_type'][0]

            # updates the progress bar string
            w_str = 'ROC Curve Calculations ({0})...'.format(tt)
            self.work_progress.emit(w_str, pW * (i_rr / r_obj.n_filt))

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
                                                            calc_para['auc_stype'], calc_para['n_boot'], c_lvl)
                    r_data.cond_ci_lo[tt][:, is_boot] = conf_int[:, 0]
                    r_data.cond_ci_hi[tt][:, is_boot] = conf_int[:, 1]

            # if not calculating the cell group indices, or the condition type is Black (the phase statistics for
            # this condition are already calculated in "calc_phase_roc_significance"), then continue
            if (not calc_cell_grp) or ((tt == 'Black') and (not force_black_calc)):
                continue

            # sets the rotation object filter (if using wilcoxon paired test for the cell group stats type)
            if calc_para['grp_stype'] == 'Wilcoxon Paired Test':
                if np.all(r_data.cond_gtype[tt][:, 0] >= 0):
                    # if all the values have been calculated, then exit the function
                    return True

                # sets the rotation object for the current condition
                r_obj_sig = RotationFilteredData(data, r_obj.rot_filt_tot[i_rr], None, plot_para['plot_exp_name'],
                                                 True, plot_scope, False, t_ofs=t_ofs, t_phase=t_phase)
                if not r_obj_sig.is_ok:
                    # if there was an error, then output an error to screen
                    self.work_error.emit(r_obj_sig.e_str, 'Incorrect Analysis Function Parameters')
                    return False

            # calculates the condition cell group types
            self.calc_phase_roc_significance(calc_para, g_para, data, pool, None, c_type='cond',
                                             roc=r_data.cond_roc[tt], auc=r_data.cond_roc_auc[tt],
                                             g_type=r_data.cond_gtype[tt], auc_sig=r_data.cond_auc_sig[tt],
                                             r_obj=r_obj_sig)

        # returns a true value
        return True

    def calc_phase_roc_significance(self, calc_para, g_para, data, pool, pW, c_type='phase',
                                    roc=None, auc=None, g_type=None, auc_sig=None, r_obj=None):
        '''

        :param calc_data:
        :param data:
        :param pool:
        :return:
        '''

        # sets the roc objects/integrals (if not provided)
        r_data, c_lvl = data.rotation, float(g_para['roc_clvl'])
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
                conf_int = self.calc_roc_conf_intervals(pool, roc[:, i_phs], phase_stype, n_boot, c_lvl)

                # determines the significance for each cell in the phase
                auc_ci_lo = (auc[:, i_phs] + conf_int[:, 1]) < 0.5
                auc_ci_hi = (auc[:, i_phs] - conf_int[:, 0]) > 0.5
                auc_sig[:, i_phs] = np.logical_or(auc_ci_lo, auc_ci_hi)

        # calculates the cell group types
        g_type[:, i_col] = cf.calc_cell_group_types(auc_sig, calc_para['grp_stype'])

    def calc_roc_conf_intervals(self, pool, roc, phase_stype, n_boot, c_lvl):
        '''

        :param r_data:
        :return:
        '''

        # sets the parameters for the multi-processing pool
        p_data = []
        for i_cell in range(len(roc)):
            p_data.append([roc[i_cell], phase_stype, n_boot, c_lvl])

        # returns the rotation data class object
        return np.array(pool.map(cf.calc_roc_conf_intervals, p_data))

    def calc_dirsel_group_types(self, data, pool, calc_para, plot_para, g_para):
        '''

        :param data:
        :param plot_para:
        :return:
        '''

        def calc_combined_spiking_stats(r_data, r_obj, pool, calc_para, g_para, p_value, ind_type=None,
                                        t_type='Black'):
            '''

            :param r_obj:
            :param ind_type:
            :return:
            '''

            # calculates the individual trial/mean spiking rates and sets up the plot/stats arrays
            sp_f0, sp_f = cf.calc_phase_spike_freq(r_obj)
            s_plt, _, sf_stats, i_grp = cf.setup_spike_freq_plot_arrays(r_obj, sp_f0, sp_f, ind_type)

            # calculates the CW/CCW spiking frequency ratio
            r_CCW_CW = np.array(s_plt[2][1]) / np.array(s_plt[2][0])

            #########################################
            ####    WILCOXON STATISTICAL TEST    ####
            #########################################

            if calc_para['grp_stype'] == 'Wilcoxon Paired Test':
                # case is the wilcoxon paired test
                sf_scores = cf.calc_ms_scores(s_plt, sf_stats, p_value)

            ##########################################
            ####    ROC-BASED STATISTICAL TEST    ####
            ##########################################

            else:
                # determines what kind of statistics are to be calculated
                phase_stype = calc_para['grp_stype']
                is_boot, n_boot = calc_para['grp_stype'] == 'Bootstrapping', calc_para['n_boot']
                phase_str, c_lvl, pW = ['CW/BL', 'CCW/BL', 'CCW/CW'], float(g_para['roc_clvl']), 100.

                # retrieves the roc/auc fields (depending on the type)
                if t_type == 'Black':
                    # case is the black (rotation) condition
                    roc, auc = r_data.phase_roc, r_data.phase_roc_auc
                elif t_type == 'UniformDrifting':
                    # case is the uniformdrifting (visual) condition
                    roc, auc = r_data.phase_roc_ud, r_data.phase_roc_auc_ud
                else:
                    # case is the motordrifting (visual) condition
                    roc, auc = r_data.cond_roc['MotorDrifting'], r_data.cond_roc_auc['MotorDrifting']

                # REMOVE ME LATER?
                c_lvl = 0.95

                # if the statistics have been calculated for the selected type, then exit the function
                if is_boot:
                    # otherwise, update the bootstrapping count
                    r_data.n_boot_comb_grp = dcopy(calc_para['n_boot'])

                # calculates the significance for each phase
                auc_sig = np.zeros((np.size(roc, axis=0), 3), dtype=bool)
                for i_phs, p_str in enumerate(phase_str):
                    # updates the progress bar string
                    if pW is not None:
                        w_str = 'ROC Curve Calculations ({0})...'.format(p_str)
                        self.work_progress.emit(w_str, pW * (i_phs / len(phase_str)))

                    # calculates the confidence intervals for the current
                    conf_int = self.calc_roc_conf_intervals(pool, roc[:, i_phs], phase_stype, n_boot, c_lvl)

                    # determines the significance for each cell in the phase
                    auc_ci_lo = (auc[:, i_phs] + conf_int[:, 1]) < 0.5
                    auc_ci_hi = (auc[:, i_phs] - conf_int[:, 0]) > 0.5
                    auc_sig[:, i_phs] = np.logical_or(auc_ci_lo, auc_ci_hi)

                # case is the wilcoxon paired test
                sf_scores = np.zeros((np.size(roc, axis=0), 3), dtype=int)
                sf_scores[i_grp[0], :] = cf.calc_ms_scores(auc, auc_sig[i_grp[0], :], None)

            # returns the direction selectivity scores
            return sf_scores, i_grp, r_CCW_CW

        def det_dirsel_cells(sf_score, grp_stype):
            '''

            :param sf_score:
            :return:
            '''

            # calculates the minimum/sum scores
            if grp_stype == 'Wilcoxon Paired Test':
                score_min, score_sum = np.min(sf_score[:, :2], axis=1), np.sum(sf_score[:, :2], axis=1)

                # determines the direction selective cells, which must meet the following conditions:
                #  1) one direction only produces a significant result, OR
                #  2) both directions are significant AND the CW/CCW comparison is significant
                one_dir_sig = np.logical_and(score_min == 0, score_sum > 0)     # cells where one direction is significant
                both_dir_sig = np.min(sf_score[:, :2], axis=1) > 0              # cells where both CW/CCW is significant
                comb_dir_sig = sf_score[:, -1] > 0                              # cells where CW/CCW difference is significant

                # determines which cells are direction selective (removes non-motion sensitive cells)
                return np.logical_or(one_dir_sig, np.logical_and(both_dir_sig, comb_dir_sig)).astype(int)
            else:
                # case is the roc analysis statistics (only consider the CW/CCW comparison for ds)
                return sf_score[:, 2] > 0

        # initialises the rotation filter (if not set)
        rot_filt = plot_para['rot_filt']
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # sets the p-value
        if 'p_value' in calc_para:
            p_val = calc_para['p_value']
        else:
            p_val = 0.05

        # initialisations and memory allocation
        p_scope, n_grp, r_data, grp_stype = 'Whole Experiment', 4, data.rotation, calc_para['grp_stype']
        r_filt_rot, r_filt_vis = dcopy(rot_filt), dcopy(rot_filt)
        plot_exp_name, plot_all_expt = plot_para['plot_exp_name'], plot_para['plot_all_expt']
        r_data.ds_p_value = dcopy(p_val)

        t_ofs_rot, t_phase_rot = cfcn.get_rot_phase_offsets(calc_para)
        t_ofs_vis, t_phase_vis = cfcn.get_rot_phase_offsets(calc_para, True)

        # determines what type of visual experiment is being used for comparison (if provided)
        if 'vis_expt_type' in calc_para:
            # case is a calculation parameter is set
            ud_rot_expt = calc_para['vis_expt_type'] == 'UniformDrifting'
        else:
            # case is no calculation parameter is set, so use uniform drifting
            ud_rot_expt = True

        # sets up the black-only rotation filter object
        r_filt_black = cf.init_rotation_filter_data(False)
        r_obj_black = RotationFilteredData(data, r_filt_black, None, plot_exp_name, plot_all_expt, p_scope, False,
                                           t_ofs=t_ofs_rot, t_phase=t_phase_rot)

        # retrieves the rotational filtered data (black conditions only)
        r_filt_rot['t_type'], r_filt_rot['is_ud'] = ['Black'], [False]
        r_data.r_obj_rot_ds = RotationFilteredData(data, r_filt_rot, None, plot_exp_name, plot_all_expt,
                                                   p_scope, False)

        # retrieves the visual filtered data
        if ud_rot_expt:
            # sets the visual phase/offset
            if t_phase_vis is None:
                # if the phase duration is not set
                t_phase_vis, t_ofs_vis = 2., 0.
            elif (t_phase_vis + t_ofs_vis) > 2:
                # output an error to screen
                e_str = 'The entered analysis duration and offset is greater than the experimental phase duration:\n\n' \
                        '  * Analysis Duration + Offset = {0}\n s. * Experiment Phase Duration = {1} s.\n\n' \
                        'Enter a correct analysis duration/offset combination before re-running ' \
                        'the function.'.format(t_phase_vis + t_ofs_vis, 2.0)
                self.work_error.emit(e_str, 'Incorrect Analysis Function Parameters')

                # return a false value indicating the calculation is invalid
                return False

            # case is uniform-drifting experiments (split into CW/CCW phases)
            r_filt_vis['t_type'], r_filt_vis['is_ud'], r_filt_vis['t_cycle'] = ['UniformDrifting'], [True], ['15']
            r_obj_vis, ind_type = cf.split_unidrift_phases(data, r_filt_vis, None, plot_exp_name, plot_all_expt,
                                                           p_scope, t_phase_vis, t_ofs_vis)

            if (r_data.phase_roc_ud is None) and ('Wilcoxon' not in calc_para['grp_stype']):
                self.calc_ud_roc_curves(data, r_obj_vis, ind_type, 66.)

        else:
            # case is motor-drifting experiments

            # retrieves the filtered data from the loaded datasets
            r_filt_vis['t_type'], r_filt_vis['is_ud'], ind_type = ['MotorDrifting'], [False], None
            t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para, is_vis=True)

            # runs the rotation filter
            r_obj_vis = RotationFilteredData(data, r_filt_vis, None, plot_exp_name, plot_all_expt,
                                                    p_scope, False, t_ofs=t_ofs, t_phase=t_phase)
            if not r_obj_vis.is_ok:
                # if there was an error, then output an error to screen
                self.work_error.emit(r_obj_vis.e_str, 'Incorrect Analysis Function Parameters')
                return False

        # calculate the visual/rotation stats scores
        sf_score_rot, i_grp_rot, r_CCW_CW_rot = calc_combined_spiking_stats(r_data, r_data.r_obj_rot_ds, pool,
                                                                            calc_para, g_para, p_val)
        sf_score_vis, i_grp_vis, r_CCW_CW_vis = calc_combined_spiking_stats(r_data, r_obj_vis, pool,
                                                                            calc_para, g_para, p_val, ind_type,
                                                                            r_filt_vis['t_type'][0])

        # memory allocation
        ds_type_tmp, ms_type_tmp, pd_type_tmp = [], [], []
        r_data.ms_gtype_N, r_data.ds_gtype_N, r_data.pd_type_N = [], [], []
        A = -np.ones(np.size(r_obj_black.t_spike[0], axis=0), dtype=int)
        r_data.ds_gtype, r_data.ms_gtype, r_data.pd_type = dcopy(A), dcopy(A), dcopy(A)

        # reduces the arrays to the matching cells
        for i in range(len(i_grp_rot)):
            if len(i_grp_rot[i]):
                # retrieves the matching rotation/visual indices
                ind_rot, ind_vis = cf.det_cell_match_indices(r_data.r_obj_rot_ds, i, r_obj_vis)

                # determines the motion sensitivity from the score phase types (append proportion/N-value arrays)
                #   0 = None
                #   1 = Rotation Only
                #   2 = Visual Only
                #   3 = Both
                _sf_score_rot = sf_score_rot[i_grp_rot[i][ind_rot]][:, :-1]
                _sf_score_vis = sf_score_vis[i_grp_vis[i][ind_vis]][:, :-1]
                ms_gtype = (np.sum(_sf_score_rot, axis=1) > 0) + 2 * (np.sum(_sf_score_vis, axis=1) > 0)
                ms_type_tmp.append(cf.calc_rel_prop(ms_gtype, 4))
                r_data.ms_gtype_N.append(len(ind_rot))

                # determines the direction selectivity type from the score phase types (append proportion/N-value arrays)
                #   0 = None
                #   1 = Rotation Only
                #   2 = Visual Only
                #   3 = Both
                is_ds_rot = det_dirsel_cells(sf_score_rot[i_grp_rot[i][ind_rot]], calc_para['grp_stype'])
                is_ds_vis = det_dirsel_cells(sf_score_vis[i_grp_vis[i][ind_vis]], calc_para['grp_stype'])
                ds_gtype = is_ds_rot.astype(int) + 2 * is_ds_vis.astype(int)
                ds_type_tmp.append(cf.calc_rel_prop(ds_gtype, 4))
                r_data.ds_gtype_N.append(len(ind_rot))

                # determines which cells have significance for both rotation/visual stimuli. from this determine the
                # preferred direction from the CW vs CCW spiking rates
                is_both_ds = ds_gtype == 3
                r_CCW_CW_comb = np.vstack((r_CCW_CW_rot[i_grp_rot[i][ind_rot]][is_both_ds],
                                           r_CCW_CW_vis[i_grp_vis[i][ind_vis]][is_both_ds])).T

                # determines the preferred direction type (for clusters which have BOTH rotation and visual significance)
                #   0 = Congruent (preferred direction is different)
                #   1 = Incongruent (preferred direction is the same)
                pd_type = np.ones(sum(is_both_ds), dtype=int)
                pd_type[np.sum(r_CCW_CW_comb > 1, axis=1) == 1] = 0

                # calculates the preferred direction type count/proportions
                r_data.pd_type_N.append(cf.calc_rel_count(pd_type, 2))
                pd_type_tmp.append(cf.calc_rel_prop(pd_type, 2))

                # sets the indices of the temporary group type into the total array
                ind_bl, ind_bl_rot = cf.det_cell_match_indices(r_obj_black, [0, i], r_data.r_obj_rot_ds)
                ind_comb = ind_bl[np.searchsorted(ind_bl_rot, ind_rot)]

                # sets the final motion sensitivity, direction selectivity and congruency values
                r_data.ms_gtype[ind_comb] = ms_gtype
                r_data.ds_gtype[ind_comb] = ds_gtype
                r_data.pd_type[ind_comb[is_both_ds]] = pd_type
            else:
                # appends the counts to the motion sensitive/direction selectivity arrays
                r_data.ms_gtype_N.append(0)
                r_data.ds_gtype_N.append(0)

                # appends NaN arrays to the temporary arrays
                ms_type_tmp.append(np.array([np.nan] * 4))
                ds_type_tmp.append(np.array([np.nan] * 4))
                pd_type_tmp.append(np.array([np.nan] * 2))

        # combines the relative proportion lists into a single array ()
        r_data.ms_gtype_pr = np.vstack(ms_type_tmp).T
        r_data.ds_gtype_pr = np.vstack(ds_type_tmp).T
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
        if equal_time:
            # case is using resampling from equal time bin sizes
            n_rs, w_prog = calc_para['n_sample'], self.work_progress
            vel_f, xi_bin = rot.calc_resampled_vel_spike_freq(data, w_prog, r_data.r_obj_kine, [vel_bin], n_rs)
        else:
            # calculates the velocity kinematic frequencies
            vel_f, xi_bin = rot.calc_kinemetic_spike_freq(data, r_data.r_obj_kine, [10, vel_bin], calc_type=1)

        # resets the frequencies based on the types
        for i_filt in range(len(vel_f)):
            if len(np.shape(vel_f[i_filt])) == 4:
                if calc_para['freq_type'] == 'All':
                    # case is considering all frequency types (stack frequencies on top of each other)
                    vel_f[i_filt] = np.mean(vel_f[i_filt], axis=3)
                elif calc_para['freq_type'] == 'Decreasing':
                    # case is only considering decreasing velocity frequencies
                    vel_f[i_filt] = vel_f[i_filt][:, :, :, 0]
                else:
                    # case is only considering increasing velocity frequencies
                    vel_f[i_filt] = vel_f[i_filt][:, :, :, 1]

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

    def calc_kinematic_roc_curves(self, data, pool, calc_para, g_para, pW0):
        '''

        :param calc_para:
        :return:
        '''

        def resample_spike_freq(pool, sf, c_lvl, n_rs=100):
            '''

            :param data:
            :param r_data:
            :param rr:
            :param ind:
            :param n_rs:
            :return:
            '''

            # array dimensioning and other initialisations
            n_trial = len(sf)
            pz = norm.ppf(1 - (1 - c_lvl) / 2)
            n_trial_h = int(np.floor(n_trial / 2))

            # if the spiking frequency values are all identical, then return the fixed values
            if cfcn.arr_range(sf) == 0.:
                return sf[0] * np.ones(n_trial_h), sf[0] * np.ones(n_trial_h), 0.5, np.zeros(2)

            # initialisations and memory allocation
            p_data = [[] for _ in range(n_rs)]

            # returns the shuffled spike frequency arrays
            for i_rs in range(n_rs):
                ind0 = np.random.permutation(n_trial)
                p_data[i_rs].append(np.sort(sf[ind0[:n_trial_h]]))
                p_data[i_rs].append(np.sort(sf[ind0[n_trial_h:(2 * n_trial_h)]]))

            # calculates the roc curves and the x/y coordinates
            _roc = pool.map(cfcn.calc_roc_curves_pool, p_data)
            _roc_xy = cfcn.calc_avg_roc_curve([cf.get_roc_xy_values(x) for x in _roc])

            # calculate the roc auc values (ensures that they are > 0.5)
            _roc_auc = [cf.get_roc_auc_value(x) for x in _roc]
            _roc_auc = [(1. - x) if x < 0.5 else x for x in _roc_auc]

            # calculates the roc auc mean/confidence interval
            roc_auc_mn = np.mean(_roc_auc)
            roc_auc_ci = pz * np.ones(2) * (np.std(_roc_auc) / (n_rs ** 0.5))

            # returns the arrays and auc mean/confidence intervals
            return _roc_xy[:, 0], _roc_xy[:, 1], roc_auc_mn, roc_auc_ci

        # initialisations
        r_data, pW1, c_lvl = data.rotation, 50., float(g_para['roc_clvl'])

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

                    # memory allocations
                    vel_auc_ci, ii_v = [], ~np.isnan(vel_sf[:, 0, ic])

                    # calculates the velocity roc curves values for each velocity bin
                    for i_bin in range(n_bin_vel):
                        if r_data.pn_comp:
                            is_resampled = False
                            vel_sf_x = vel_sf[ii_v, n_bin_vel + i_bin, ic]
                            vel_sf_y = vel_sf[ii_v, n_bin_vel - (i_bin + 1), ic]
                        else:
                            # case is single bin comparison
                            if (i_bin == r_data.i_bin_vel[0]) or (i_bin == r_data.i_bin_vel[1]):
                                is_resampled = True
                                vel_sf_x, vel_sf_y, vel_auc_roc, _auc_ci = \
                                                    resample_spike_freq(pool, vel_sf[ii_v, i_bin, ic], c_lvl)
                                vel_auc_ci.append(_auc_ci)
                            else:
                                is_resampled = False
                                vel_sf_x = vel_sf[ii_v, i_bin, ic]
                                if r_data.vel_xi[i_bin, 0] < 0:
                                    vel_sf_y = vel_sf[ii_v, r_data.i_bin_vel[0], ic]
                                else:
                                    vel_sf_y = vel_sf[ii_v, r_data.i_bin_vel[1], ic]

                        # calculates the roc curves/coordinates from the spiking frequencies
                        r_data.vel_roc[tt][ic, i_bin] = cf.calc_roc_curves(None, None,
                                                                           x_grp=vel_sf_x, y_grp=vel_sf_y)
                        r_data.vel_roc_xy[tt][ic, i_bin] = cf.get_roc_xy_values(r_data.vel_roc[tt][ic, i_bin])

                        # sets the roc auc values
                        if is_resampled:
                            # case is the resampled frequencies
                            r_data.vel_roc_auc[tt][ic, i_bin] = vel_auc_roc
                        else:
                            # other cases
                            r_data.vel_roc_auc[tt][ic, i_bin] = cf.get_roc_auc_value(r_data.vel_roc[tt][ic, i_bin])

                    # calculates the speed roc curves values for each speed bin
                    if not r_data.pn_comp:
                        ii_s = ~np.isnan(spd_sf[:, 0, ic])
                        for i_bin in range(n_bin_spd):
                            calc_roc = True
                            if (i_bin == r_data.i_bin_spd):
                                # spd_sf_x, spd_sf_y = resample_spike_freq(data, r_data, rr, [i_rr, i_bin, ic])
                                is_resampled = True
                                spd_sf_x, spd_sf_y, spd_auc_roc, spd_auc_ci = \
                                                resample_spike_freq(pool, spd_sf[ii_s, i_bin, ic], c_lvl)
                            else:
                                is_resampled = False
                                spd_sf_x, spd_sf_y = spd_sf[ii_s, r_data.i_bin_spd, ic], spd_sf[ii_s, i_bin, ic]

                            # calculates the roc curves/coordinates from the spiking frequencies
                            r_data.spd_roc[tt][ic, i_bin] = cf.calc_roc_curves(None, None, x_grp=spd_sf_x, y_grp=spd_sf_y)
                            r_data.spd_roc_xy[tt][ic, i_bin] = cf.get_roc_xy_values(r_data.spd_roc[tt][ic, i_bin])

                            # sets the roc auc values
                            if is_resampled:
                                # case is the resampled frequencies
                                r_data.spd_roc_auc[tt][ic, i_bin] = spd_auc_roc
                            else:
                                # other cases
                                r_data.spd_roc_auc[tt][ic, i_bin] = cf.get_roc_auc_value(r_data.spd_roc[tt][ic, i_bin])

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
                        auc_type, n_boot = calc_para['auc_stype'], calc_para['n_boot']
                        conf_int_vel = self.calc_roc_conf_intervals(pool, r_data.vel_roc[tt][ic, :],
                                                                    auc_type, n_boot, c_lvl)

                        # resets the resampled confidence interval values
                        if not r_data.pn_comp:
                            conf_int_vel[r_data.i_bin_vel[0], :] = vel_auc_ci[0]
                            conf_int_vel[r_data.i_bin_vel[1], :] = vel_auc_ci[1]

                        # sets the upper and lower velocity confidence intervals
                        r_data.vel_ci_lo[tt][ic, :, is_boot] = conf_int_vel[:, 0]
                        r_data.vel_ci_hi[tt][ic, :, is_boot] = conf_int_vel[:, 1]

                        # calculates the speed confidence intervals
                        if not r_data.pn_comp:
                            # calculates the speed confidence intervals
                            conf_int_spd = self.calc_roc_conf_intervals(pool, r_data.spd_roc[tt][ic, :],
                                                                        auc_type, n_boot, c_lvl)

                            # resets the resampled confidence interval values
                            conf_int_spd[r_data.i_bin_spd] = spd_auc_ci

                            # sets the upper and lower speed confidence intervals
                            r_data.spd_ci_lo[tt][ic, :, is_boot] = conf_int_spd[:, 0]
                            r_data.spd_ci_hi[tt][ic, :, is_boot] = conf_int_spd[:, 1]

    def check_combined_conditions(self, calc_para, plot_para):
        '''

        :param calc_para:
        :param plot_para:
        :return:
        '''

        if plot_para['rot_filt'] is not None:
            if 'MotorDrifting' in plot_para['rot_filt']['t_type']:
                # if the mapping file is not correct, then output an error to screen
                e_str = 'MotorDrifting is not a valid filter option when running this function.\n\n' \
                        'De-select this filter option before re-running this function.'
                self.work_error.emit(e_str, 'Invalid Filter Options')

                # returns a false value
                return False

        # if everything is correct, then return a true value
        return True

    def check_altered_para(self, data, calc_para, g_para, chk_type, other_para=None):
        '''

        :param calc_para:
        :param g_para:
        :param chk_type:
        :return:
        '''

        # initialisations
        r_data = data.rotation
        t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)

        # loops through each of the check types determining if any parameters changed
        for ct in chk_type:
            # initialises the change flag
            is_change = False

            if ct == 'condition':
                # case is the roc condition parameters

                # retrieves the rotation phase offset time/duration
                if t_ofs is not None:
                    # if the values are not none, and do not match previous values, then reset the stored roc array
                    if (r_data.t_ofs_rot != t_ofs) or (r_data.t_phase_rot != t_phase):
                        r_data.t_ofs_rot, r_data.t_phase_rot, is_change = t_ofs, t_phase, True
                elif 'use_full_rot' in calc_para:
                    # if using the full rotation, and the previous calculations were made using non-full rotation
                    # phases, the reset the stored roc array
                    if (r_data.t_ofs_rot > 0):
                        r_data.t_ofs_rot, r_data.t_phase_rot, is_change = -1, -1, True

                # if there was a change, then re-initialise the roc condition fields
                if is_change:
                    # memory allocation (if the conditions have not been set)
                    r_data.phase_roc, r_data.phase_roc_auc, r_data.phase_roc_xy = {}, {}, {}
                    r_data.phase_ci_lo, self.phase_ci_hi, self.phase_gtype = None, None, None
                    r_data.phase_auc_sig, r_data.phase_grp_stats_type = None, None

                    r_data.cond_roc, r_data.cond_roc_xy, r_data.cond_roc_auc = {}, {}, {}
                    r_data.cond_gtype, r_data.cond_auc_sig, r_data.cond_i_expt, r_data.cond_cl_id = {}, {}, {}, {}
                    r_data.cond_ci_lo, r_data.cond_ci_hi, r_data.r_obj_cond = {}, {}, {}
                    r_data.phase_gtype, r_data.phase_auc_sig, r_data.phase_roc = None, None, None

                    r_data.part_roc, r_data.part_roc_xy, r_data.part_roc_auc = {}, {}, {}

            elif ct == 'phase':
                # case is the phase ROC calculations
                pass

                # # if there was a change, then re-initialise the roc phase fields
                # if is_change:
                #     a = 1

            elif ct == 'visual':
                # retrieves the visual phase time offset/duration
                t_ofs_vis, t_phase_vis = cfcn.get_rot_phase_offsets(calc_para, True)

                # if the values are not none, and do not match previous values, then reset the stored roc array
                if (r_data.t_ofs_vis != t_ofs_vis) or (r_data.t_phase_vis != t_phase_vis):
                    r_data.t_ofs_vis, r_data.t_phase_vis, is_change = t_ofs_vis, t_phase_vis, True

                # if there was a change, then re-initialise the fields
                if is_change:
                    r_data.phase_roc_ud, r_data.phase_roc_auc_ud, r_data.phase_roc_xy_ud = None, None, None

            elif ct == 'vel':
                # case is the kinematic calculations

                # initialisations
                roc_calc, vel_bin = other_para, float(calc_para['vel_bin'])

                # checks to see if the dependent speed has changed
                if 'spd_x_rng' in calc_para:
                    # case is a single speed bin range comparison

                    # if the dependent speed range has changed then reset the roc curve calculations
                    if r_data.comp_spd != calc_para['spd_x_rng']:
                        is_change = True

                    if r_data.pn_comp is True:
                        r_data.pn_comp, is_change = False, True

                    # updates the speed comparison flag
                    r_data.comp_spd = calc_para['spd_x_rng']

                else:
                    # case is the positive/negative speed comparison

                    # if the positive/negative comparison flag is not set to true, then reset the roc curve calculations
                    if r_data.pn_comp is False:
                        r_data.pn_comp, is_change = True, True

                # if the velocity bin size has changed or isn't initialised, then reset velocity roc values
                if roc_calc:
                    if (vel_bin != r_data.vel_bin) or (calc_para['freq_type'] != r_data.freq_type):
                        r_data.vel_sf_rs, r_data.spd_sf_rs = None, None
                        r_data.vel_sf, r_data.spd_sf = None, None
                        r_data.vel_bin, r_data.vel_roc = vel_bin, None
                        r_data.freq_type = calc_para['freq_type']

                    if r_data.is_equal_time != calc_para['equal_time']:
                        r_data.vel_roc = None

                # if using equal time bins, then check to see if the sample size has changed (if so then recalculate)
                if calc_para['equal_time']:
                    if r_data.n_rs != calc_para['n_sample']:
                        r_data.vel_sf_rs, r_data.spd_sf_rs = None, None
                        r_data.n_rs, r_data.vel_roc = calc_para['n_sample'], None

                # if there was a change, then re-initialise the roc phase fields
                if is_change:
                    r_data.vel_roc = None

            elif ct == 'lda':
                # case is the LDA calculations

                # if initialising the LDA then continue (as nothing has been set)
                d_data, lda_para = other_para, calc_para['lda_para']
                if d_data.lda is None:
                    continue

                # otherwise, determine if there are any changes in the parameters
                is_equal = [
                    d_data.solver == lda_para['solver_type'],
                    d_data.shrinkage == lda_para['use_shrinkage'],
                    d_data.norm == lda_para['is_norm'],
                    d_data.cellmin == lda_para['n_cell_min'],
                    d_data.trialmin == lda_para['n_trial_min'],
                    d_data.yaccmx == lda_para['y_acc_max'],
                    d_data.yaccmn == lda_para['y_acc_min'],
                    set(d_data.ttype) == set(lda_para['comp_cond']),
                ]

                #
                if d_data.type in ['Direction', 'Individual', 'TrialShuffle', 'Partial']:
                    if 'use_full_rot' in calc_para:
                        if d_data.usefull:
                            is_equal += [
                                d_data.usefull == calc_para['use_full_rot']
                            ]
                        else:
                            is_equal += [
                                d_data.tofs == t_ofs,
                                d_data.tphase == t_phase,
                            ]

                    if d_data.type == 'TrialShuffle':
                        is_equal += [
                            d_data.nshuffle == calc_para['n_shuffle'],
                            # d_data.bsz == calc_para['b_sz']
                        ]
                    elif d_data.type == 'Partial':
                        is_equal[3] = True

                        is_equal += [
                            d_data.nshuffle == calc_para['n_shuffle'],
                            d_data.cellminpart  == calc_para['n_cell_min']
                        ]

                elif d_data.type == 'Temporal':
                    is_equal += [
                        d_data.dt_phs == calc_para['dt_phase'],
                        d_data.dt_ofs == calc_para['dt_ofs'],
                        d_data.phs_const == calc_para['t_phase_const'],
                     ]

                # if there was a change in any of the parameters, then reset the LDA data field
                if not np.all(is_equal):
                    d_data.lda = None

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


