# module import
import copy
import datetime
import functools
import math as m
import os
import pickle as p
import re
import time
import warnings
import numpy as np
import pandas as pd
import xlsxwriter
from random import sample

import seaborn as sns
sns.set()
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)
# sns.set_context('talk', font_scale=1.2)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba_array
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import matplotlib as mpl
from matplotlib.pyplot import rc
import matplotlib.style

#
import mplcursors
from mpldatacursor import datacursor

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()
r_stats = importr("stats")
r_pROC = importr("pROC")

# try:
#     r_pHOC = importr("PMCMRplus")
# except:
#     pass

from scipy.signal import medfilt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator as pchip

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.mixture import BayesianGaussianMixture as GMM

# pyqt5 module import
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import (QMainWindow, QWidget, QGroupBox, QLineEdit, QComboBox,
                             QCheckBox, QDialog,  QFormLayout, QMessageBox)

# custom module import
import analysis_guis.common_func as cf
import analysis_guis.calc_functions as cfcn
import analysis_guis.rotational_analysis as rot
from analysis_guis.dialogs.file_dialog import FileDialogModal
from analysis_guis.dialogs import load_expt, config_dialog, expt_compare
from analysis_guis.dialogs.rotation_filter import RotationFilter, RotationFilteredData
from analysis_guis.dialogs.info_dialog import InfoDialog
from analysis_guis.threads import thread_workers

# other parameters
dX = 10
dY = 10
dY_obj = 25
grp_Y0 = 10
n_plot_max = 25
table_fsize = 12
default_dir_file = os.path.join(os.getcwd(), 'default_dir.p')

# font objects
txt_font = cf.create_font_obj()
table_font = cf.create_font_obj(size=12)
table_font_small = cf.create_font_obj(size=10)
txt_font_bold = cf.create_font_obj(is_bold=True, font_weight=QFont.Bold)
grp_font_sub = cf.create_font_obj(size=10, is_bold=True, font_weight=QFont.Bold)
grp_font_sub2 = cf.create_font_obj(size=9, is_bold=True, font_weight=QFont.Bold)
grp_font_main = cf.create_font_obj(size=12, is_bold=True, font_weight=QFont.Bold)

# general group width sizes
grp_wid = 401
grp_inner = grp_wid - 2 * dX
grp_inner2 = grp_inner - 2 * dX

# lambda function declarations
lin_func = lambda x, a: a * x
ebar_col = lambda x: 'r' if x else 'k'
get_list_fields = lambda comp, c_field: np.concatenate([getattr(x, c_field) for x in comp])
formatter = lambda **kwargs: ', '.join(kwargs['point_label'])
setup_heatmap_bins = lambda t_stim, dt: np.arange(t_stim + dt / 1000, step=dt / 1000.0)
sig_str_fcn = lambda x, p_value: '*' if x < p_value else ''
convert_rgb_col = lambda col: to_rgba_array(np.array(col) / 255, 1)

# other initialisations
dcopy = copy.deepcopy
func_types = np.array(['Cluster Matching','Cluster Classification','Rotation Analysis',
                       'UniformDrift Analysis', 'ROC Analysis', 'Combined Analysis', 'Single Experiment Analysis'])
_red, _black, _green = [140, 0, 0], [0, 0, 0], [47, 150, 0]
_blue, _gray, _light_gray, _orange = [0, 30, 150], [90, 90, 50], [200, 200, 200], [255, 110, 0]
_bright_red, _bright_cyan, _bright_purple = (249, 2, 2), (2, 241, 249), (245, 2, 249)
_bright_yellow = (249, 221, 2)

########################################################################################################################
########################################################################################################################

class AnalysisGUI(QMainWindow):
    def __init__(self, parent=None, loaded_data=[]):
        # creates the object
        super(AnalysisGUI, self).__init__(parent)

        # data field initialisations
        self.data = AnalysisData()

        # other initialisations
        self.is_multi = False
        self.calc_ok = True
        self.can_close = False
        self.initialising = True
        self.analysis_scope = 'Unprocessed'

        # determines if the default data file has been set
        if os.path.isfile(default_dir_file):
            # if so, then the data from the file
            with open(default_dir_file, 'rb') as fp:
                self.def_data = p.load(fp)
        else:
            # otherwise, set the initial data to None
            self.def_data = self.init_def_data()

        # initialises each part of the analysis GUI
        self.init_main_window()
        self.init_expt_info()
        self.init_func_group()
        self.init_plot_group()
        self.init_menu_items()
        self.init_progress_group()
        self.init_other_objects()

        # sets the central widget and shows the gui window
        self.initialising = False
        self.setCentralWidget(self.centralwidget)

        # ensures the gui has a fixed size
        cf.set_obj_fixed_size(self, self.width(), self.height())
        self.show()

    ####################################################################################################################
    ####                                        GUI INITIALISATION FUNCTIONS                                        ####
    ####################################################################################################################

    def init_main_window(self):
        '''

        :return:
        '''

        # sets the main window properties
        self.resize(1681, 931)
        self.setObjectName("AnalysisMain")
        self.setWindowTitle("EPhys Analysis GUI")
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        # creates the central widget object
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        # turns off all warnings
        warnings.filterwarnings("ignore")

    def init_expt_info(self):
        '''

        :return:
        '''

        # creates the experiment information groupbox
        self.grp_info = cf.create_groupbox(self.centralwidget, QRect(10, 10, grp_wid, 136), grp_font_main,
                                           "Experiment Information", "grp_info")

        # creates the header label objects
        self.lbl_expt_count_h = cf.create_label(self.grp_info, txt_font_bold, "Loaded Experiment Count: ",
                                                QRect(10, 30, 181, 20), "lbl_expt_count_h", 'right')
        self.lbl_analy_scope_h = cf.create_label(self.grp_info, txt_font_bold, "Analysis Scope: ",
                                                 QRect(10, 50, 181, 20), "lbl_analy_scope_h", 'right')
        self.lbl_comp_set_h = cf.create_label(self.grp_info, txt_font_bold, "Comparison Experiments Set?: ",
                                              QRect(10, 70, 181, 20), "lbl_comp_set_h", 'right')
        self.lbl_comp_fix_h = cf.create_label(self.grp_info, txt_font_bold, "Fixed Preparation Experiment: ",
                                              QRect(10, 90, 191, 20), "lbl_comp_set_h", 'right')
        self.lbl_comp_free_h = cf.create_label(self.grp_info, txt_font_bold, "Free Preparation Experiment: ",
                                              QRect(10, 110, 191, 20), "lbl_comp_set_h", 'right')

        # creates the text field label objects
        self.lbl_expt_count = cf.create_label(self.grp_info, txt_font, "0", QRect(195, 30, 40, 20), "lbl_expt_count")
        self.lbl_analy_scope = cf.create_label(self.grp_info, txt_font, "N/A", QRect(195, 50, 150, 20), "lbl_analy_scope")
        self.lbl_comp_set = cf.create_label(self.grp_info, txt_font, "No", QRect(195, 70, 40, 20), "lbl_comp_set")
        self.lbl_comp_fix = cf.create_label(self.grp_info, txt_font, "N/A", QRect(205, 90, 150, 20), "lbl_comp_fix")
        self.lbl_comp_free = cf.create_label(self.grp_info, txt_font, "N/A", QRect(205, 110, 150, 20), "lbl_comp_free")

        # disables the objects within the group
        self.set_group_enabled_props(self.grp_info, False)

    def init_func_group(self):
        '''

        :return:
        '''

        # creates the groupbox objects
        self.grp_func = cf.create_groupbox(self.centralwidget, QRect(10, 155, grp_wid, 656), grp_font_main,
                                           "Analysis Functions", "grp_func")
        self.grp_scope = cf.create_groupbox(self.grp_func, QRect(10, 30, grp_inner, 55), grp_font_sub,
                                           "Analysis Type", "grp_scope")
        self.grp_funcsel = cf.create_groupbox(self.grp_func, QRect(10, 95, grp_inner, 191), grp_font_sub,
                                           "Function Select", "grp_funcsel")
        self.grp_para = cf.create_groupbox(self.grp_func, QRect(10, 295, grp_inner, 351), grp_font_sub,
                                           "Function Parameters", "grp_para")

        # creates the combobox objects
        self.combo_scope = cf.create_combobox(self.grp_scope, txt_font, func_types, QRect(10, 25, grp_inner2, 20),
                                              "combo_scope", self.change_scope)

        # creates the listbox objects
        self.list_funcsel = cf.create_listbox(self.grp_funcsel, QRect(10, 20, grp_inner2, 161), txt_font,
                                              None, "list_funcsel", cb_fcn=self.func_select)

        # creates the pushbutton objects
        self.push_update = cf.create_button(self.grp_para, QRect(10, 25, grp_inner2, 22), txt_font_bold,
                                            "Update Plot Figure", 'push_update', cb_fcn=self.update_click)

        # other initialisations and setting object properties
        self.fcn_data = AnalysisFunctions(self.grp_para, self)
        self.set_group_enabled_props(self.grp_func, False)

    def init_progress_group(self):
        '''

        :return:
        '''

        # creates the groupbox object
        self.grp_prog = cf.create_groupbox(self.centralwidget, QRect(10, 821, grp_wid, 81), grp_font_sub,
                                           "Progress", "grp_prog")

        # creates the progress groupbox objects
        self.text_prog = cf.create_label(self.grp_prog, txt_font, "Waiting For Process...",
                                         QRect(10, 23, 241, 18), "text_prog", align='right')
        self.pbar_prog = cf.create_progressbar(self.grp_prog, QRect(260, 20, 131, 20), txt_font,
                                               name="pbar_prog")
        self.push_cancel = cf.create_button(self.grp_prog, QRect(10, 50, grp_inner, 22), txt_font_bold,
                                            "Cancel Process", 'push_cancel', cb_fcn=self.cancel_progress)

        # creates the worker thread and initialises the parameters
        self.worker = [thread_workers.WorkerThread(main_gui=self) for _ in range(2)]

        # sets the slot connection values
        for i in range(2):
            self.worker[i].work_started.connect(self.start_thread_job)
            self.worker[i].work_progress.connect(self.update_thread_job)
            self.worker[i].work_finished.connect(functools.partial(self.finished_thread_job, i))
            self.worker[i].work_error.connect(self.error_thread_job)
            self.worker[i].work_plot.connect(self.plot_values)
            self.worker[i].setTerminationEnabled(True)

        # disables the objects in the groupbox
        self.set_group_enabled_props(self.grp_prog, False)

    def init_other_objects(self):
        '''

        :return:
        '''

        # determines the next available worker
        iw = self.det_avail_thread_worker()

        # starts the worker thread
        self.worker[iw].update_pbar = False
        self.worker[iw].set_worker_func_type('init_pool_object', thread_job_para=[])
        self.worker[iw].start()

    def init_plot_group(self):
        '''

        :return:
        '''

        # creates the groupbox objects
        plot_left = grp_wid + 2 * dX
        self.grp_plot = cf.create_groupbox(self.centralwidget, QRect(plot_left,10,1251,891), None, "", "frm_plot")

        # creates the plot figure object
        self.plot_fig = None

    def init_menu_items(self):
        '''

        :return:
        '''

        # creates the main menu items
        self.menubar = cf.create_menubar(self, QRect(0, 0, 1521, 21), "menubar")
        self.menu_file = cf.create_menu(self.menubar, "File", "menu_file")
        self.menu_data = cf.create_menu(self.menubar, "Data", "menu_data")

        #############################
        ###    FILE MENU ITEMS    ###
        #############################

        # creates the file menu/menu-items
        self.menu_cluster_data = cf.create_menu(self.menu_file, "Cluster Datasets", "cluster_data")
        self.menu_output_data = cf.create_menu(self.menu_file, "Output Data", "save_data")
        self.menu_default = cf.create_menuitem(self, "Set Default Directories", "menu_default", self.set_default,
                                               s_cut='Ctrl+D')
        self.menu_global_para = cf.create_menuitem(self, "Global Parameters", "global_para", self.update_glob_para,
                                                   s_cut='Ctrl+G')
        self.menu_exit = cf.create_menuitem(self, "Exit Program", "menu_exit", self.exit_program, s_cut='Ctrl+X')

        # adds the menu items to the file menu
        self.menu_file.addAction(self.menu_cluster_data.menuAction())
        self.menu_file.addAction(self.menu_output_data.menuAction())
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.menu_default)
        self.menu_file.addAction(self.menu_global_para)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.menu_exit)

        # disables the output data menu item
        self.menu_output_data.setEnabled(False)

        ########################################
        ###    CLUSTER DATASET MENU ITEMS    ###
        ########################################

        # creates the menu items
        self.menu_init_data = cf.create_menuitem(self, 'Initialise Cluster Dataset', 'init_data', self.init_data,
                                                 s_cut='Ctrl+L')
        self.menu_load_data = cf.create_menuitem(self, 'Load Cluster Datasets', 'load_data', self.load_data,
                                                 s_cut='Ctrl+O')
        self.menu_set_compare = cf.create_menuitem(self, 'Set Comparison Datasets', 'set_compare', self.set_compare,
                                                   s_cut='Ctrl+C')

        # adds the menu items to the cluster dataset menu
        self.menu_cluster_data.addAction(self.menu_init_data)
        self.menu_cluster_data.addSeparator()
        self.menu_cluster_data.addAction(self.menu_load_data)
        self.menu_cluster_data.addAction(self.menu_set_compare)

        ####################################
        ###    DATA OUTPUT MENU ITEMS    ###
        ####################################

        # creates the menu items
        self.menu_save_figure = cf.create_menuitem(self, 'Plot Figure', 'save_figure', self.save_figure,
                                                   s_cut='Ctrl+P')
        self.menu_save_data = cf.create_menuitem(self, 'Data File', 'save_figure', self.save_data, s_cut='Ctrl+S')
        self.menu_save_file = cf.create_menuitem(self, 'Multi-Experiment Data File', 'save_file', self.save_file,
                                                 s_cut='Ctrl+F')

        # adds the menu items to the data output menu
        self.menu_output_data.addAction(self.menu_save_figure)
        self.menu_output_data.addAction(self.menu_save_data)
        self.menu_output_data.addSeparator()
        self.menu_output_data.addAction(self.menu_save_file)

        #############################
        ###    DATA MENU ITEMS    ###
        #############################

        # creates the menu items
        self.menu_show_info = cf.create_menuitem(self, 'Show Dataset Information', 'show_info', self.show_info,
                                                 s_cut='Ctrl+I')
        self.menu_init_filt = cf.create_menu(self, "Set Exclusion Filter Fields", "init_rotdata")

        # adds the menu items to the file menu
        self.menu_data.addAction(self.menu_show_info)
        self.menu_data.addSeparator()
        self.menu_data.addAction(self.menu_init_filt.menuAction())

        # disables the output data menu item
        self.menu_data.setEnabled(False)
        self.menu_show_info.setEnabled(False)

        ##########################################
        ###    ROTATIONAL FILTER MENU ITEMS    ###
        ##########################################

        # creates the menu items
        self.menu_gen_filt = cf.create_menuitem(self, 'General Filter', 'init_genfilt', self.init_genfilt,
                                                s_cut='Ctrl+G')
        self.menu_rot_filt = cf.create_menuitem(self, 'Rotational Stimuli Filter', 'init_rotfilt', self.init_rotfilt,
                                                s_cut='Ctrl+R')
        self.menu_ud_filt = cf.create_menuitem(self, 'Uniform Drifting Filter', 'init_udfilt', self.init_udfilt,
                                               s_cut='Ctrl+U')

        # adds the menu items to the file menu
        self.menu_init_filt.addAction(self.menu_gen_filt)
        self.menu_init_filt.addAction(self.menu_rot_filt)
        self.menu_init_filt.addAction(self.menu_ud_filt)

        # disables the output data menu item
        self.menu_gen_filt.setEnabled(False)
        self.menu_rot_filt.setEnabled(False)
        self.menu_ud_filt.setEnabled(False)

        ##############################
        ###    FINAL MENU SETUP    ###
        ##############################

        # adds the main menus to the menubar
        self.menubar.addAction(self.menu_file.menuAction())
        self.menubar.addAction(self.menu_data.menuAction())
        self.setMenuBar(self.menubar)

        # disables the output data menu item
        self.menu_save_figure.setEnabled(False)

    def init_def_data(self):
        '''

        :return:
        '''

        # sets the default data file directory path
        data_dir = os.path.join(os.getcwd(), 'data_files')

        # global parameters
        g_para = {'n_hist': '100', 'n_spike': '1000', 'd_max': '2', 'r_max': '100.0',
                  'sig_corr_min': '0.95', 'sig_diff_max': '0.30', 'isi_corr_min': '0.65', 'sig_feat_min': '0.50',
                  'w_sig_feat': '0.25', 'w_sig_comp': '1.00', 'w_isi': '0.25', 'z_max': '1.00'}
        def_dir = {'configDir': data_dir, 'inputDir': data_dir, 'dataDir': data_dir, 'figDir': data_dir}

        # sets the final default data dictionary
        def_data = {'dir': def_dir, 'g_para': g_para}

        # writes the data to file
        with open(default_dir_file, 'wb') as fw:
            p.dump(def_data, fw)

        # returns the default data dictionary
        return def_data

    def get_data(self):
        '''

        :return:
        '''

        return self.data

    def get_plot_group(self):
        '''

        :return:
        '''

        return self.combo_scope.currentText()

    def get_plot_func(self):
        '''

        :return:
        '''

        return self.list_funcsel.selectedItems()[0].text()

    ####################################################################################################################
    ####                                          WORKER THREAD FUNCTIONS                                           ####
    ####################################################################################################################

    def start_thread_job(self):
        '''

        :return:
        '''

        # enables the progress groupbox
        self.set_group_enabled_props(self.grp_prog, True)

    def update_thread_job(self, text_prog, pbar_prog):
        '''

        :param update_list:
        :return:
        '''

        # updates text label and progress bar value
        self.text_prog.setText(text_prog)
        self.pbar_prog.setValue(pbar_prog)

    def finished_thread_job(self, iw, worker_data=None, manual_exit=False, is_plot=False):
        '''

        :return:
        '''

        # enables the progress groupbox
        self.set_group_enabled_props(self.grp_prog, False)
        if iw is None:
            self.text_prog.setText('Process Complete!')
            self.pbar_prog.setValue(100.0)
        else:
            # resets the thread running flag
            self.worker[iw].is_running = False

            # final update of the progress text and bar
            if self.worker[iw].update_pbar:
                if self.worker[iw].forced_quit:
                    self.text_prog.setText('User Cancelled Process...')
                    self.pbar_prog.setValue(100.0)
                else:
                    self.text_prog.setText('Process Complete!')
                    self.pbar_prog.setValue(100.0)
            else:
                self.worker[iw].update_pbar = True

        # resets the text label and the progress bar
        time.sleep(0.5)
        self.text_prog.setText('Waiting For Process...')
        self.pbar_prog.setValue(0)

        # if only plotting, then exit the function
        if is_plot:
            return

        # performs any post thread functions (based on the thread job type)
        if not self.worker[iw].forced_quit:
            if self.worker[iw].thread_job_primary == 'load_data_files':
                # case is loading the data files

                # loops through all the new file names and loads
                if (worker_data is not None) and len(worker_data):
                    for loaded_data in worker_data:
                        self.data._cluster.append(loaded_data)

                # sets up the analysis functions
                self.data.rotation.init_rot_fields()
                self.fcn_data.init_all_func()

                # sets the analysis function groupbox properties
                self.set_group_enabled_props(self.grp_info, len(self.data._cluster) > 0)
                self.set_group_enabled_props(self.grp_func, len(self.data._cluster) > 0)
                self.lbl_expt_count.setText(str(len(self.data._cluster)))
                self.fcn_data.set_exp_name([cf.extract_file_name(x['expFile']) for x in self.data._cluster],
                                           self.is_multi)

                # sets the enabled properties of the menu items
                self.menu_output_data.setEnabled(True)
                self.menu_save_file.setEnabled(len(self.data._cluster) > 1 and (not self.is_multi))

                # initialises the classification data fields
                exp_name = [cf.extract_file_name(x['expFile']) for x in self.data._cluster]
                clust_id = [x['clustID'] for x in self.data._cluster]
                self.data.classify.init_classify_fields(exp_name, clust_id)

                # enables the menu item
                self.menu_data.setEnabled(True)
                self.menu_show_info.setEnabled(True)

                #
                if self.is_multi:
                    if 'Cluster Matching' in [x['type'] for x in self.data._cluster]:
                        self.data.comp.is_set = True
                        new_func_types = func_types
                    else:
                        self.data.comp.is_set = False
                        new_func_types = func_types[1:]

                    # disnables the cluster matching comparison menu item
                    self.menu_set_compare.setEnabled(False)
                else:
                    #
                    has_rot_expt = any(cf.det_valid_rotation_expt(self.data))
                    has_ud_expt = any(cf.det_valid_rotation_expt(self.data, True))
                    has_both = has_ud_expt and has_rot_expt
                    is_keep = [True, True, has_rot_expt, has_ud_expt, has_rot_expt, has_both, True]
                    new_func_types = func_types[np.array(is_keep)]

                    # otherwise, enable the cluster matching comparison menu item
                    self.menu_set_compare.setEnabled(True)
                    self.menu_init_filt.setEnabled(has_rot_expt)
                    self.menu_gen_filt.setEnabled(True)
                    self.menu_rot_filt.setEnabled(has_rot_expt)
                    self.menu_ud_filt.setEnabled(has_ud_expt)

                    # updates the general exclusion filter
                    self.data.update_gen_filter()

                    # updates the exclusion rotation filter (if any experiments contain rotational data)
                    if has_rot_expt:
                        self.data.rotation.update_rot_filter()

                    # updates the exclusion uniformdrifting filter (if any experiments contain uniformdrifting data)
                    if has_ud_expt:
                        self.data.rotation.update_ud_filter()

                if self.combo_scope.count() != len(new_func_types):
                    # sets the flag which disables the function type callback function
                    self.initialising = True

                    #
                    for _ in reversed(range(self.combo_scope.count())):
                        self.combo_scope.removeItem(0)

                    #
                    for new_ft in new_func_types:
                        self.combo_scope.addItem(new_ft)

                    # resets the flag
                    self.initialising = False

                # initialises the function selection listbox
                if len(self.data._cluster):
                    self.change_scope()

            elif self.worker[iw].thread_job_primary == 'init_pool_object':
                # case is the pool worker object
                pool = worker_data
                self.fcn_data.set_pool_worker(pool)

            elif self.worker[iw].thread_job_primary == 'cluster_matches':
                # case is calculating the cluster matches

                # updates the comparison data struct
                self.data.comp = worker_data

                # sets the information label properties
                if self.data.comp.is_set:
                    self.lbl_comp_set.setText("Yes")
                    self.menu_output_data.setEnabled(True)
                else:
                    self.lbl_comp_set.setText("No")

            elif self.worker[iw].thread_job_primary == 'run_calc_func':
                # case is the calculation functions

                # if the calculation failed, then exit without updating
                if not self.worker[iw].is_ok:
                    return

                # retrieves the data from the worker thread
                calc_data = worker_data

                # sets the data based on the calculation function that was run
                if self.worker[iw].thread_job_secondary == 'Cluster Cross-Correlogram':
                    # case is calculating the cross-correlogram

                    # sets the individual components of the calculated data
                    c_type, t_dur = calc_data['c_type'], calc_data['t_dur']
                    t_event, calc_para = calc_data['t_event'], calc_data['calc_para']
                    ci_lo, ci_hi, ccG_T = calc_data['ci_lo'], calc_data['ci_hi'], calc_data['ccG_T']

                    # determines the indices of the experiment to be analysed
                    if calc_para['calc_all_expt']:
                        # case is all experiments are to be analysed
                        i_expt = list(range(len(c_type)))
                    else:
                        # case is a single experiment is being analysed
                        i_expt = [cf.get_expt_index(calc_para['calc_exp_name'], self.data._cluster)]

                    # sets the action type indices for each combination (over each type)
                    act_type = [[] for _ in range(len(c_type))]
                    for i in range(len(c_type)):
                        # memory allocation
                        j = i_expt[i]
                        act_type[i] = np.zeros(self.data._cluster[j]['nC'], dtype=int)
                        _cl_ind = cfcn.get_inclusion_filt_indices(self.data._cluster[j], self.data.exc_gen_filt)
                        cl_ind, act_type[i][np.logical_not(_cl_ind)] = np.where(_cl_ind)[0], -1

                        # removes any cross-over points between the excitatory/inhibitory lists
                        if len(c_type[i][0]) and len(c_type[i][1]):
                            if np.any(np.array([x in c_type[i][0][:, 0] for x in c_type[i][1][:, 0]])):
                                # FINISH ME!
                                a = 1

                            # sets the inhibitory/excitatory flags
                            act_type[i][cl_ind[np.unique(c_type[i][0][:, 0])]] = 1
                            act_type[i][cl_ind[np.unique(c_type[i][1][:, 0])]] = 2

                    # sets the action data into the classification class object
                    self.data.classify.set_action_data(calc_para, c_type, t_dur, t_event,
                                                       ci_lo, ci_hi, ccG_T, i_expt, act_type)

                elif self.worker[iw].thread_job_secondary == 'Shuffled Cluster Distances':
                    # case is calculating the shuffled distances
                    a = 1

                elif self.worker[iw].thread_job_secondary == 'Direction ROC Curves':
                    # updates the rotation set flag
                    self.data.rotation.is_set = True

                # re-runs the plotting function
                self.update_click()

    def error_thread_job(self, e_str, title):
        '''

        :param e_str:
        :param title:
        :return:
        '''

        cf.show_error(e_str, title)

    def plot_values(self, Y):
        '''

        :param Y:
        :return:
        '''

        plt.figure()
        plt.plot(Y)
        plt.show()

    def is_thread_running(self):
        '''

        :return:
        '''

        is_running = [x.is_running for x in self.worker]

        if np.all(is_running):
            err_txt = 'Unable to continue as initialisation is currently in progress.\n' \
                      'Either wait until the current process is complete or press the "Cancel Process" button.'
            cf.show_error(err_txt,'Data File Initialisation Currently In Progress!')
            return True
        else:
            return False

    ####################################################################################################################
    ####                                          MENU CALLBACK FUNCTIONS                                           ####
    ####################################################################################################################

    def init_data(self):
        '''

        :return:
        '''

        # if data output is currently in progress then output an error an exit the function
        if self.is_thread_running():
            return

        # initialisations
        now, init_data = datetime.datetime.now().strftime("%Y-%m-%d"), None
        title = 'Experiment Information Profile'

        # table header and combo options
        table_para = ['depthLo', 'depthHi', 'regionName', 'recordLayer']
        table_hdr = ['Lower Limit ({0}m)'.format(cf._mu), 'Upper Limit ({0}m)'.format(cf._mu),
                     'Region Name', 'Recording Layer']
        table_opt = {2: ['', 'RSPd', 'RSPg', 'SC', 'SUB', 'V1', 'ProS', 'Post', 'pV2L', 'V2M', 'Hip', 'ATN', 'N/A'],
                     3: ['', 'Layer 2/3', 'Layer 4', 'Layer 5', 'Layer 6', 'SG', 'OP',
                         'IG', 'DG', 'CA1', 'DG', 'LD', 'AD', 'N/A']}
        table_info = [table_hdr, ['Number', 'Number', 'List', 'List'], table_opt, 'Layer', 10]

        # initialises the dialog info fields
        dlg_info = [
            ['Configuration File', 'configFile', 'File', 'Config Files (*.cfig)', False, True, 0, _black],
            ['Source Data Directory', 'srcDir', 'Directory', '', True, False, 1, _black],
            ['Raw Trace File', 'traceFile', 'File', 'Binary File (*.bin)', True, False, 2, _black],
            ['Channel Depth Mapping File', 'dmapFile', 'File', 'CSV File (*.csv)', False, False, 3, _black],

            ['Bonsai Stimuli Data File', 'bonsaiFile', 'File', 'Binary File (*.csv)', False, False, 4, _orange],
            ['Igor Stimuli Data File', 'igorFile', 'File', 'All Files (*)', False, False, 5, _orange],
            ['Probe Trigger Trace File', 'probeTrigFile', 'File', 'Numpy File (*.npy)', False, False, 6, _orange],
            ['Stimulus Onset Trigger File', 'stimOnsetFile', 'File', 'Numpy File (*.npy)', False, False, 7, _orange],
            ['Photodiode Trigger File', 'photoTrigFile', 'File', 'Numpy File (*.npy)', False, False, 8, _orange],

            ['Experiment Name', 'expName', 'String', '', True, False, 9, _red],
            ['Experiment Date', 'expDate', 'String', now, True, False, 9, _red],
            ['Experiment Type', 'expType', 'List', ['Acute', 'Chronic'], True, False, 9, _red],
            ['Experiment Condition', 'expCond', 'List', ['Fixed', 'Free'], True, False, 9, _red],
            ['Mouse Age', 'expAge', 'Number', '0', True, False, 9, _red],
            ['Mouse Sex', 'expSex', 'List', ['Male', 'Female'], True, False, 10, _red],

            ['Channel Count', 'nChan', 'Number', '32', True, False, 10, _green],
            ['Sampling Frequency (Hz)', 'sFreq', 'Number', '25000', True, False, 10, _green],
            ['Voltage Gain ({0}V)'.format(cf._mu), 'vGain', 'Number', '1', True, False, 10, _green],
            ['Additional Info', 'otherInfo', 'String', '', False, False, 10, _green],

            ['Recording Coordinate', 'recordCoord', 'List', ['Rostral', 'Middle', 'Caudal'], True, False, 11, _blue],
            ['Recording State', 'recordState', 'List', ['Awake', 'Anaesthetised'], True, False, 11, _blue],
            ['Probe Type', 'expProbe', 'List', ['Neuronexus', 'Neuropixels', 'Other'], True, False, 11, _blue],
            ['Lesion Type', 'lesionType', 'List', ['None', 'Vestibular', 'RHP', 'ATN'], True, False, 11, _blue],
            ['Cluster Type', 'clusterType', 'List', ['Good', 'MUA'], True, False, 11, _blue],

            ['Probe Location Details', table_para, 'TableCombo', table_info, True, False, 12, _gray]
        ]

        while True:
            # opens up the config dialog box and retrieves the final file information
            cfig_dlg = config_dialog.ConfigDialog(dlg_info=dlg_info, title=title,
                                                  init_data=init_data, def_dir=self.def_data['dir'])

            # retrieves the information from the dialog
            exp_info = cfig_dlg.get_info()
            if exp_info is None:
                return
            else:
                # determines if everything has been properly set for the data file initialisation
                e_str = rot.check_rot_analysis_files(self, exp_info, dlg_info)
                if e_str is None:
                    # if everything is correct, then exit the loop
                    break
                else:
                    # otherwise, output an error to screen and reset the initial data file
                    init_data = exp_info
                    if isinstance(e_str, str):
                        cf.show_error(e_str, 'Incorrect Data File Initialisation')

        # sets the output file name
        if self.def_data is None:
            def_file_name = os.path.join(os.getcwd(),'{0}.cdata'.format(exp_info['expName']))
        else:
            def_file_name = os.path.join(self.def_data['dir']['inputDir'], '{0}.cdata'.format(exp_info['expName']))

        # determines if rotational analysis is being performed
        if len(exp_info['bonsaiFile']):
            # prompts the user for the matching experiment condition keys
            exp_info['i2b_key'], is_ok = rot.get_rot_condition_key(exp_info)
            if not is_ok:
                # flag that the user cancelled and exit
                self.ok = False
                return
        else:
            exp_info['i2b_key'] = None

        file_dlg = FileDialogModal(caption='Set Cluster Data File',
                                   filter='Cluster Data File (*.cdata)',
                                   directory=def_file_name,
                                   is_save=True)

        if (file_dlg.exec() == QDialog.Accepted):
            # otherwise, set the output file name
            out_file = file_dlg.selectedFiles()
            out_name = cf.set_file_name(out_file[0], '(*.cdata)')
            g_para = self.def_data['g_para']
        else:
            # flag that the user cancelled and exit
            self.ok = False
            return

        # # REMOVE ME LATER
        # tcode.test_init_data(exp_info, out_name, g_para)

        # starts the worker thread
        iw = self.det_avail_thread_worker()
        self.worker[iw].set_worker_func_type('init_data_file', thread_job_para=[exp_info, out_name, g_para])
        self.worker[iw].start()

    def load_data(self):
        '''

        :return:
        '''

        # if data output is currently in progress then output an error an exit the function
        if self.is_thread_running():
            return

        # if the loaded data is not
        load_dlg = load_expt.LoadExpt(loaded_data=self.data._cluster,def_dir=self.def_data['dir'])
        if load_dlg.is_ok:
            # clears the plot axes
            try:
                self.plot_fig.hide()
                time.sleep(0.1)
            except:
                pass

            # retrieves the names of the loaded files
            loaded_exp = [cf.extract_file_name(x['expFile']) for x in self.data._cluster]

            # sets the analysis scope label string
            if load_dlg.is_multi:
                self.is_multi = True
                self.lbl_analy_scope.setText('Multi-Experiment')
            else:
                self.is_multi = False
                self.lbl_analy_scope.setText('Single Experiment')

            # removes any loaded data not in the final selection
            for i in reversed(range(len(self.data._cluster))):
                if cf.extract_file_name(self.data._cluster[i]['expFile']) not in load_dlg.exp_name:
                    self.data._cluster.pop(i)

            # determines if the comparison datasets have been set
            if self.data.comp.is_set:
                # if so, determine either the fixed or free files have been unloaded
                if self.data.comp.fix_name not in load_dlg.exp_name or \
                    self.data.comp.free_name not in load_dlg.exp_name:
                        # if so, then reset the comparison flag and the comparison label strings
                        self.data.comp.is_set = False
                        self.lbl_comp_set.setText('No')
                        self.lbl_comp_fix.setText('N/A')
                        self.lbl_comp_free.setText('N/A')

            # starts the worker thread
            iw = self.det_avail_thread_worker()
            self.worker[iw].set_worker_func_type('load_data_files', thread_job_para=[load_dlg, loaded_exp, self.is_multi])
            self.worker[iw].start()

    def set_compare(self):
        '''

        :return:
        '''

        # if data output is currently in progress then output an error an exit the function
        if self.is_thread_running():
            return

        # initialisations
        e_str, exp_type = None, None

        # determines if any data files have been loaded
        if len(self.data._cluster):
            # determines if both fixed and free experiments have been loaded
            exp_type = [x['expInfo']['cond'] for x in self.data._cluster]
            if not 'Fixed' in exp_type:
                e_str = 'No Fixed experiment types have been loaded.\n\n' \
                        'Load at least 1 free experiment type data file and retry.'
            elif not 'Free' in exp_type:
                e_str = 'No Free experiment types have been loaded.\n\n' \
                        'Load at least 1 free experiment type data file and retry.'
        else:
            # if not, then create the error message
            e_str = 'No experimental data files have been loaded.\n\n' \
                    'Load at least 1 fixed/free experiment type data file and retry.'

        # determines if it is feasible to load the experiment comparison dialog
        if e_str is None:
            # if so, load the experimental comparsion dialog and wait for the users choice
            exp_name = [cf.extract_file_name(x['expFile']) for x in self.data._cluster]
            comp_dlg = expt_compare.ExptCompare(exp_name=exp_name, exp_type=exp_type, comp_data=self.data.comp)
            if comp_dlg.is_ok:
                # if the user chose to continue, then update the comparison indices
                ind = [exp_name.index(comp_dlg.list_fixed.selectedItems()[0].text()),
                       exp_name.index(comp_dlg.list_free.selectedItems()[0].text())]
                data_fix, data_free = self.get_comp_datasets(ind, is_full=True)

                # sets the fixed/free experiment text labels
                fix_name = cf.extract_file_name(data_fix['expFile'])
                free_name = cf.extract_file_name(data_free['expFile'])
                self.lbl_comp_fix.setText(fix_name)
                self.lbl_comp_free.setText(free_name)

                # initialises the comparison data struct
                self.data.comp.init_comparison_data(ind, n_fix=data_fix['nC'],
                                                    n_free=data_free['nC'],
                                                    n_pts=data_fix['nPts'],
                                                    fix_name=fix_name,
                                                    free_name=free_name)

                # starts the worker thread
                iw = self.det_avail_thread_worker()
                thread_job_para = [self.data.comp, data_fix, data_free, self.def_data['g_para']]
                self.worker[iw].set_worker_func_type('cluster_matches', thread_job_para=thread_job_para)
                self.worker[iw].start()

        else:
            # otherwise, create the error message
            cf.show_error(e_str, 'Comparison Experiment Set Error!')

            # sets the information label properties
            if self.data.comp.is_set:
                self.lbl_comp_set.setText("Yes")
                self.menu_output_data.setEnabled(True)
            else:
                self.lbl_comp_set.setText("No")

    def save_figure(self):
        '''

        :return:
        '''

        # initialisations
        file_name = 'Figure'

        # sets the figure output directory
        if self.def_data is None:
            fig_dir = os.getcwd()
        else:
            fig_dir = self.def_data['dir']['figDir']

        # creates the figure file data dictionary
        img_types = ['Encapsulated Postscript (.eps)', 'Postscript (.ps)', 'Portable Document File (.pdf)',
                     'Portable Network Graphic (.png)', 'Scalable Vector Graph (.svg)']
        fig_data = {'figDir': fig_dir, 'figName': file_name, 'figDPI': 200, 'fColour': 'White', 'eColour': 'White',
                    'figOrient': 'Portrait', 'figFmt': img_types[0], 'outType': 'Output Current Figure Only'}

        # sets the dialog info list
        dlg_info = [
            ['Output Directory', 'figDir', 'Directory', '', True, False, 0],
            ['Figure Name', 'figName', 'String', fig_data['figName'], True, False, 1],
            ['Figure DPI', 'figDPI', 'Number', str(fig_data['figDPI']), True, False, 1],
            ['Face Colour', 'fColour', 'List', ['White', 'Black'], True, False, 2],
            ['Edge Colour', 'eColour', 'List', ['White', 'Black'], True, False, 2],
            ['Orientation', 'figOrient', 'List', ['Portrait', 'Landscape'], True, False, 3],
            ['Figure Format', 'figFmt', 'List', img_types, True, False, 3],
        ]

        # determines the index of the currently selected file
        current_fcn = self.list_funcsel.selectedItems()[0].text()
        grp_details = self.fcn_data.details[self.combo_scope.currentText()]
        i_func = next(i for i, x in enumerate(grp_details) if x['name'] == current_fcn)

        # if the function has multi-figure capabilities, then include a radio button for selection
        if 'multi_fig' in grp_details[i_func]:
            # includes the radio options
            var_name = grp_details[i_func]['multi_fig'][0]
            para_name = grp_details[i_func]['para'][var_name]['text']
            radio_options = [fig_data['outType'], 'Output Figures For Each {0}'.format(para_name)]

            # appends the new option to the dialog information list
            dlg_info.append(['Figure Output Type', 'outType', 'Radio', radio_options, True, False, 4])

        # opens up the config dialog box and retrieves the final file information
        title = 'Output Figure Information'
        cfig_dlg = config_dialog.ConfigDialog(dlg_info=dlg_info, title=title, width=550, init_data=fig_data)

        # retrieves the information from the dialog
        fig_info = cfig_dlg.get_info()
        if fig_info is not None:
            # sets the default output type (if not provided)
            if 'outType' not in fig_info:
                fig_info['outType'] = 'Output Current Figure Only'

            if fig_info['outType'] == 'Output Current Figure Only':
                # case is outputting single image
                self.output_single_figure(fig_info)
            else:
                # if the file already exists, prompt the user if they wish to overwrite the file
                prompt_text = "Do you want to output the images to a separate sub-directory?"
                u_choice = QMessageBox.question(self, 'Output Images To Sub-Directory?', prompt_text,
                                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if u_choice == QMessageBox.Yes:
                    # if the user rejected then exit the function
                    nw_dir = os.path.join(fig_info['figDir'], '{0} (Image Group)'.format(fig_info['figName']))
                    fig_info['figDir'] = nw_dir

                    # if the new directory does not exist then create it
                    if not os.path.isdir(nw_dir):
                        os.mkdir(nw_dir)

                # retrieves the current parameter struct
                fcn_para = grp_details[i_func]['para']
                curr_para0 = dcopy(self.fcn_data.curr_para)

                if self.is_multi:
                    data_dummy = self.data._cluster[0]['data'][0]
                else:
                    data_dummy = self.data._cluster[0]

                # alters the values of any link parameters (to the variable) so that the variable is unaffected
                for p in fcn_para:
                    if fcn_para[p]['link_para'] is not None:
                        if fcn_para[p]['link_para'][0] == var_name:
                            if isinstance(fcn_para[p]['link_para'][1], bool):
                                # if the link parameter is boolean, then set the opposite value
                                self.fcn_data.curr_para[p] = not fcn_para[p]['link_para'][1]
                            else:
                                # FINISH ME!
                                pass

                # sets the number of plots (for multi-plot output)
                n_plot = {'i_cluster': data_dummy['nC']}

                # case is outputting multiple images
                for i_plot in range(n_plot[var_name]):
                    self.output_single_figure(fig_info, var_name=var_name, para_name=para_name, i_plot=i_plot+1)

                # updates the plot axes to the original
                self.fcn_data.curr_para = curr_para0
                self.update_click()

    def save_data(self):
        '''

        :return:
        '''

        # determines which flags have been set
        is_set = np.array([self.data.comp.is_set, self.data.classify.is_set, self.data.rotation.is_set])
        if not np.any(is_set):
            return

        # sets the initial output data dictionary
        radio_options = np.array(func_types[:-1])[is_set]
        file_type = ['Comma Separated Value (.csv)', 'Excel Spreadsheet (.xlsx)']
        out_data = {'dataDir': self.def_data['dir']['dataDir'], 'dataName': '',
                    'fileType': file_type[0], 'outData': radio_options[0]}

        # initialisations
        dlg_info = [
            ['Data File Output Directory', 'dataDir', 'Directory', self.def_data['dir']['dataDir'], True, False, 0],
            ['Data File Name', 'dataName', 'String', '', True, False, 1],
            ['Data File Type', 'fileType', 'List', file_type, True, False, 1],
            ['Experimental Data To Output', 'outData', 'Radio', radio_options, True, False, 2],
        ]

        # opens up the config dialog box and retrieves the final file information
        cfig_dlg = config_dialog.ConfigDialog(dlg_info=dlg_info,
                                              title='Data Output Options',
                                              width=500,
                                              init_data=out_data)

        # retrieves the information from the dialog
        out_info = cfig_dlg.get_info()
        if out_info is not None:
            # sets the output file name
            out_tmp = os.path.join(out_info['dataDir'], out_info['dataName'])
            out_file = cf.set_file_name(out_tmp, out_info['fileType'])
            is_csv = out_file[-3:] == 'csv'

            # outputs the data depending on the type
            if out_info['outData'] == 'Cluster Matching':
                # case is the cluster matching
                self.output_cluster_matching_data(out_file, is_csv)
            elif out_info['outData'] == 'Cluster Classification':
                # case is the cell classification (FINISH ME!)
                self.output_cell_classification_data(out_file, is_csv)
            elif out_info['outData'] == 'Rotation Analysis':
                # case is the rotation analysis (FINISH ME!)
                self.output_rotation_analysis_data(out_file, is_csv)

    def save_file(self):
        '''

        :return:
        '''

        # determines which flags have been set
        if not self.data.comp.is_set:
            return

        # sets the initial output data dictionary
        out_data = {'inputDir': self.def_data['dir']['inputDir'], 'dataName': ''}

        # initialisations
        dlg_info = [
            ['Combined Cluster Matching Data File Output Directory', 'inputDir', 'Directory',
             self.def_data['dir']['inputDir'], True, False, 0],
            ['Combined Cluster Matching Data File Name', 'dataName', 'String', '', True, False, 1],
        ]

        # opens up the config dialog box and retrieves the final file information
        cfig_dlg = config_dialog.ConfigDialog(dlg_info=dlg_info,
                                              title='Combined Cluster Matching Data File Output Options',
                                              width=500,
                                              init_data=out_data)

        # retrieves the information from the dialog
        out_info = cfig_dlg.get_info()
        if out_info is not None:
            # sets the output file name
            out_file = os.path.join(out_info['inputDir'], '{0}.mdata'.format(out_info['dataName']))

            # creates the multi-experiment data file based on the type
            data_out = {'type': out_info['outType'], 'data': None}
            data_out['data'] = [[] for _ in range(2)]
            data_out['data'][0], data_out['data'][1] = self.get_comp_datasets()
            data_out['comp_data'] = self.data.comp

            # outputs the data to file
            with open(out_file, 'wb') as fw:
                p.dump(data_out, fw)

    def set_default(self):
        '''

        :return:
        '''

        # initialisations
        dlg_info = [
            ['Configuration File Directory', 'configDir', 'Directory', '', True, False, 0],
            ['Input Data File Directory', 'inputDir', 'Directory', '', True, False, 1],
            ['Analysis Data File Directory', 'dataDir', 'Directory', '', True, False, 2],
            ['Analysis Figure Directory', 'figDir', 'Directory', '', True, False, 3],
        ]

        # opens up the config dialog box and retrieves the final file information
        title = 'Default Data Directories'
        cfig_dlg = config_dialog.ConfigDialog(dlg_info=dlg_info,
                                              title=title,
                                              width=600,
                                              init_data=self.def_data['dir'])

        # retrieves the information from the dialog
        def_dir_new = cfig_dlg.get_info()
        if def_dir_new is not None:
            # if the user set the default directories, then update the default directory data file
            self.def_data['dir'] = def_dir_new
            with open(default_dir_file, 'wb') as fw:
                p.dump(self.def_data, fw)

    def update_glob_para(self):
        '''

        :return:
        '''

        # initialisations
        dlg_info = [
            ['ISI/Signal Histogram Bin Count', 'n_hist', 'Number', '', True, False, 0, _black],
            ['Stored Spike Count', 'n_spike', 'Number', '', True, False, 0, _black],
            ['Max Channel Depth Difference', 'd_max', 'Number', '', True, False, 1, _red],
            ['Max Relative Spike Frequency Rate', 'r_max', 'Number', '', True, False, 1, _red],

            ['Signal Correlation Minimum', 'sig_corr_min', 'Number', '', True, False, 2, _green],
            ['ISI Correlation Minimum', 'isi_corr_min', 'Number', '', True, False, 2, _green],
            ['Maximum Proportional Signal Diff', 'sig_diff_max', 'Number', '', True, False, 3, _green],
            ['Signal Feature Difference Minimum', 'sig_feat_min', 'Number', '', True, False, 3, _green],

            ['Signal Feature Score Weight', 'w_sig_feat', 'Number', '', True, False, 4, _blue],
            ['Signal Comparison Score Weight', 'w_sig_comp', 'Number', '', True, False, 4, _blue],
            ['ISI Score Weight', 'w_isi', 'Number', '', True, False, 5, _blue],
            ['Maximum Z-Score Tolerance', 'z_max', 'Number', '', True, False, 5, _gray],
        ]

        # opens up the config dialog box and retrieves the final file information
        title = 'Default Global Parameters'
        cfig_dlg = config_dialog.ConfigDialog(dlg_info=dlg_info,
                                              title=title,
                                              width=600,
                                              init_data=self.def_data['g_para'],
                                              has_reset=False)

        # retrieves the information from the dialog
        g_para_new = cfig_dlg.get_info()
        if g_para_new is not None:
            # if the user set the default directories, then update the default directory data file
            self.def_data['g_para'] = g_para_new
            with open(default_dir_file, 'wb') as fw:
                p.dump(self.def_data, fw)

    def show_info(self):
        '''

        :return:
        '''

        # opens the data file information dialog
        InfoDialog(self)

    def init_genfilt(self):
        '''

        :return:
        '''

        # runs the general filter in exclusion mode
        r_filt = RotationFilter(self.fcn_data, init_data=self.data.exc_gen_filt, is_gen=True, is_exc=True)

        # determines if the gui was updated correctly
        if r_filt.is_ok:
            # updates the current parameter value
            self.data.exc_gen_filt = r_filt.get_info()
            self.data.req_update = True

    def init_rotfilt(self):
        '''

        :return:
        '''

        # runs the rotation filter in exclustion mode
        r_filt = RotationFilter(self.fcn_data, init_data=self.data.rotation.exc_rot_filt, is_exc=True)

        # determines if the gui was updated correctly
        if r_filt.is_ok:
            # updates the current parameter value
            self.data.rotation.exc_rot_filt = r_filt.get_info()
            self.data.req_update = True

    def init_udfilt(self):
        '''

        :return:
        '''

        # runs the uniformdrifting filter in exclustion mode
        r_filt = RotationFilter(self.fcn_data, init_data=self.data.rotation.exc_ud_filt, is_exc=True)

        # determines if the gui was updated correctly
        if r_filt.is_ok:
            # updates the current parameter value
            self.data.rotation.exc_ud_filt = r_filt.get_info()
            self.data.req_update = True

    def exit_program(self):
        '''

        :return:
        '''

        #
        if np.any([x.is_running for x in self.worker]):
            # if the file already exists, prompt the user if they wish to overwrite the file
            prompt_text = "Thread process is currently running. Are you sure you wish to exit the program? " \
                          "(This action will terminate the currently running process)."
            u_choice = QMessageBox.question(self, 'Exit Program?', prompt_text,
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if u_choice == QMessageBox.No:
                # if the user rejected then exit the function
                return
            else:
                # otherwise, force terminate the process and exit the program
                self.cancel_progress()

        # closes the analysis GUI
        self.can_close = True
        self.close()

    ##########################################
    ####     OTHER CALLBACK FUNCTIONS     ####
    ##########################################

    def change_scope(self):
        '''

        :return:
        '''

        # if initialising then exit the function
        if self.initialising:
            return

        # clears all of the current items from the
        self.list_funcsel.clear()

        # adds the function names for the analysis type/scope
        for fcn in self.fcn_data.get_func_names(self.combo_scope.currentText()):
            self.list_funcsel.addItem(fcn)

        # sets the current select row to the first item
        self.list_funcsel.setCurrentRow(0)

    def func_select(self):
        '''

        :return:
        '''

        # determines the currently selected item and sets the push button enabled properties
        sel_item = self.list_funcsel.selectedItems()
        self.push_update.setEnabled(len(sel_item)>0)

        # sets the parameters based on selection type
        if len(sel_item):
            self.fcn_data.create_para_objects(sel_item[0].text())

    def cancel_progress(self):
        '''

        :return:
        '''

        #
        for iw in np.where([x.is_running for x in self.worker])[0]:
            # updates the worker flags
            self.worker[iw].forced_quit = True

            # terminates the worker object and updates the GUI properties
            self.worker[iw].terminate()
            self.finished_thread_job(iw)
            self.worker[iw].wait()

            # updates the worker flags
            self.worker[iw].forced_quit = False

    def update_click(self):
        '''

        :return:
        '''

        # if data output is currently in progress then output an error an exit the function
        if self.is_thread_running():
            return

        # determines if the current analysis function/loaded data configuration is feasible
        elif self.combo_scope.currentText() == "Cluster Matching" and not self.data.comp.is_set:
            # if not, then output an error and exit the function
            e_str = 'The cluster comparison datasets have not been set!\n\n' \
                    'Run "File => Cluster Dataset => Set Comparison Dataset" and set the ' \
                    'comparison datasets before retrying.'
            cf.show_error(e_str, 'Analysis Function Error!')
            return

        # creates the data plot
        self.plot_data(**self.fcn_data.curr_para)

    def closeEvent(self, evnt):
        '''

        :param evnt:
        :return:
        '''

        if self.can_close:
            super(AnalysisGUI, self).closeEvent(evnt)
        else:
            evnt.ignore()

    ####################################################################################################################
    ####                                        ANALYSIS PLOTTING FUNCTIONS                                         ####
    ####################################################################################################################

    def plot_data(self, **all_para):
        '''

        :param all_para:
        :return:
        '''

        # initialisations
        self.calc_ok = True
        calc_para, plot_para = self.split_func_para()

        # retrieves the currently selected function
        current_fcn = self.list_funcsel.selectedItems()[0].text()
        plot_scope = self.combo_scope.currentText()
        grp_details = self.fcn_data.details[plot_scope]
        func_name = next(x['func'] for x in grp_details if x['name'] == current_fcn)

        # closes the current figure
        if self.plot_fig is not None:
            self.plot_fig.close()

        # reduces down the dataset to remove the rejected clusters
        self.remove_rejected_clusters(plot_scope, current_fcn)

        # determines the calculation parameters
        if len(calc_para):
            # if there are any calculation parameters, then determine if any of them have changed (or have been set)
            if self.det_calc_para_change(calc_para, plot_para, current_fcn):
                # if so, then run the calculation thread
                self.fcn_data.prev_calc_para = dcopy(calc_para)
                self.fcn_data.prev_plot_para = dcopy(plot_para)
                self.set_group_enabled_props(self.grp_prog, True)

                # runs the worker thread
                fcn_para = [calc_para, plot_para, self.data, self.fcn_data.pool]

                # runs the calculation on the next available thread
                iw = self.det_avail_thread_worker()
                self.worker[iw].set_worker_func_type('run_calc_func', current_fcn, fcn_para)
                self.worker[iw].start()
                return
        else:
            # if there aren't any calculation parameters, then reset the previous calculation parameter object
            self.fcn_data.prev_calc_para = None
            self.fcn_data.prev_plot_para = None

        # creates the new plot canvas
        self.plot_fig = PlotCanvas(self.grp_plot, width=8, height=8)
        self.plot_fig.move(10, 10)
        self.plot_fig.show()

        try:
            self.update_thread_job('Creating Plot Figure', 100.0 / 2.0)
            eval('self.{0}(**plot_para)'.format(func_name))
        except:
            a = 1

        # if the calculation/plotting when successfully, then show the figure
        if self.calc_ok:
            self.plot_fig.draw()
            self.plot_fig.fig.canvas.update()
            self.menu_save_figure.setEnabled(True)
            self.update_thread_job('Plot Figure Complete!', 100.0)
        else:
            self.plot_fig.hide()
            self.menu_save_figure.setEnabled(False)
            self.update_thread_job('Plot Figure Error!', 100.0)

        # updates te progress bar to indicate the plot has finished being created
        time.sleep(0.1)
        self.finished_thread_job(None, is_plot=True)

    ###################################################
    ####    CLUSTER MATCHING ANALYSIS FUNCTIONS    ####
    ###################################################

    def plot_multi_match_means(self, i_cluster, plot_all, m_type, exp_name=None, plot_grid=True):
        '''

        :return:
        '''

        # sets the acceptance flags based on the method
        comp = dcopy(self.data.comp)
        if self.is_multi:
            # case is multi-experiment files have been loaded
            data_fix, data_free, comp = self.get_multi_comp_datasets(False, exp_name, is_list=False)
        else:
            # retrieves the fixed/free datasets
            data_fix, _ = self.get_comp_datasets()
            _data_fix, data_free = self.get_comp_datasets(is_full=True)

            # retrieves the fixed/free cluster inclusion indices
            cl_inc_fix = cfcn.get_inclusion_filt_indices(_data_fix, self.data.exc_gen_filt)
            cl_inc_free = cfcn.get_inclusion_filt_indices(data_free, self.data.exc_gen_filt)

            # removes any excluded cells from the free dataset
            ii = np.where(comp.i_match >= 0)[0]
            jj = comp.i_match[ii]
            comp.i_match[ii[np.logical_not(cl_inc_free[jj])]] = -1
            comp.i_match_old[ii[np.logical_not(cl_inc_free[jj])]] = -1

            # reduces down the match indices to only include the feasible fixed dataset indices
            comp.i_match, comp.i_match_old = comp.i_match[cl_inc_fix], comp.i_match_old[cl_inc_fix]
            comp.is_accept, comp.is_accept_old = comp.is_accept[cl_inc_fix], comp.is_accept_old[cl_inc_fix]

        # sets the match/acceptance flags
        if m_type == 'New Method':
            i_match = comp.i_match
            is_acc = comp.is_accept
        else:
            i_match = comp.i_match_old
            is_acc = comp.is_accept_old

        # checks cluster index if plotting all clusters
        i_cluster, e_str = self.check_cluster_index_input(i_cluster, plot_all, data_fix['nC'])
        if e_str is not None:
            cf.show_error(e_str,'Infeasible Cluster Indices')
            self.calc_ok = False
            return

        # sets the indices of the clusters to plot and creates the figure/axis objects
        i_cluster = self.set_cluster_indices(i_cluster)
        T = self.setup_time_vector(data_fix['sFreq'], np.size(data_fix['vMu'], axis=0))

        # sets up the figure/axis
        n_plot, col = len(i_cluster), 'kg'
        self.init_plot_axes(n_plot=n_plot)

        # plots the values over all subplots
        for i_plot in range(n_plot):
            # sets the actual fixed/free plot indices
            j_plot = i_cluster[i_plot] - 1
            i_match_new = i_match[j_plot]
            id_fix = data_fix['clustID'][j_plot]

            # plots the fixed signal
            self.plot_fig.ax[i_plot].plot(T, data_fix['vMu'][:, j_plot], linewidth=3.0)
            if i_match_new >= 0:
                # if there was a match, then plot the mean matches
                self.plot_fig.ax[i_plot].plot(T, data_free['vMu'][:, i_match_new], 'r--', linewidth=2.0)

                # set the title match/colour
                t_str = 'Fixed #{0}/Free #{1}'.format(id_fix, data_free['clustID'][i_match_new])
                t_col = col[int(is_acc[j_plot])]
            else:
                # otherwise, there was no feasible match (set reduced title which is to be black)
                t_str, t_col = 'Fixed #{0}'.format(id_fix), 'k'

            # sets the plot values
            self.plot_fig.ax[i_plot].set_xlim(T[0], T[-1])
            self.plot_fig.ax[i_plot].set_title(t_str, color=t_col)
            self.plot_fig.ax[i_plot].set_xlabel('Time (ms)')
            self.plot_fig.ax[i_plot].set_ylabel('Voltage ({0}V)'.format(cf._mu))
            self.plot_fig.ax[i_plot].grid(plot_grid)

            # creates the legend (first plot only)
            if i_plot == 0:
                self.plot_fig.ax[i_plot].legend(['Fixed', 'Free'], loc=0)

    def plot_single_match_mean(self, i_cluster, n_trace, is_horz, rej_outlier, exp_name=None, plot_grid=True):
        '''

        :return:
        '''

        # sets the acceptance flags based on the method
        if self.is_multi:
            # case is multi-experiment files have been loaded
            data_fix, data_free, comp = self.get_multi_comp_datasets(False, exp_name, is_list=False)
        else:
            # retrieves the fixed/free datasets
            comp = self.data.comp
            data_fix, data_free = self.get_comp_datasets(is_full=True)

        # check to see if the cluster index is feasible
        i_cluster, e_str = self.check_cluster_index_input(i_cluster, False, data_fix['nC'])
        if e_str is not None:
            cf.show_error(e_str, 'Infeasible Cluster Indices')
            self.calc_ok = False
            return
        elif len(i_cluster) > 1:
            e_str = 'Not possible to view multiple cluster indices.'
            cf.show_error(e_str, 'Infeasible Cluster Indices')
            self.calc_ok = False
            return

        # intiialisations
        i_cluster, p_lim = i_cluster[0] - 1, 0.4
        i_match = comp.i_match[i_cluster]

        # determines if there was a feasible cluster match
        if i_match < 0:
            # sets up the error string
            e_str = 'Unable to create single cluster match plot for cluster index {0} (Free Cluster ID# {1}) ' \
                    'because there is not a feasible match.'.format(i_cluster+1, data_fix['clustID'][i_cluster])
            cf.show_error(e_str, 'Missing Feasible Fixed Cluster Match')

            # resets the ok flag and exits
            self.calc_ok = False
            return

        # sets the indices of the fixed/free clusters to be plotted
        ind_fix = np.random.permutation(np.size(data_fix['vSpike'][i_cluster],axis=1))[:n_trace]
        ind_free = np.random.permutation(np.size(data_free['vSpike'][i_match],axis=1))[:n_trace]
        fix_ID, free_ID = data_fix['clustID'][i_cluster], data_free['clustID'][comp.i_match[i_cluster]]

        # sets the free/fixed spikes
        spike_fix = data_fix['vSpike'][i_cluster][:, ind_fix]
        spike_free = data_free['vSpike'][i_match][:, ind_free]

        # sets up the figure/axis
        T = self.setup_time_vector(data_fix['sFreq'], np.size(data_fix['vMu'], axis=0))
        n_row, n_col = 3 - 2 * int(is_horz), 1 + 2 * int(is_horz)
        self.init_plot_axes(n_plot = 3, n_row=n_row, n_col=n_col)

        # creates the combined subplot
        self.plot_fig.ax[2].plot(T, data_fix['vMu'][:, i_cluster], linewidth=3.0)
        self.plot_fig.ax[2].plot(T, data_free['vMu'][:, i_match],'r', linewidth=3.0)
        self.plot_fig.ax[2].set_title('Combined Fixed/Free Cluster Plot')
        self.plot_fig.ax[2].set_xlabel('Time (ms)')
        self.plot_fig.ax[2].set_ylabel('Voltage ({0}V)'.format(cf._mu))
        self.plot_fig.ax[2].legend(['Fixed', 'Free'], loc=3)

        # determines the overall
        yL0 = self.plot_fig.ax[2].get_ylim()
        yL = np.array(yL0) + p_lim*np.diff(yL0)*np.array([-1, 1])

        # removes any outlier signals (if selected as a parameter)
        if rej_outlier:
            ok_fix = np.logical_and(np.min(spike_fix, axis=0) >= yL[0], np.max(spike_fix, axis=0) < yL[1])
            ok_free = np.logical_and(np.min(spike_free, axis=0) >= yL[0], np.max(spike_free, axis=0) < yL[1])
            spike_fix, spike_free = spike_fix[:, ok_fix], spike_free[:, ok_free]

        # creates the fixed subplot
        self.plot_fig.ax[0].plot(T, spike_fix,'b')
        self.plot_fig.ax[0].plot(T, data_fix['vMu'][:, i_cluster], 'k', linewidth=4.0)
        self.plot_fig.ax[0].set_title('Fixed Cluster #{0}'.format(fix_ID))
        self.plot_fig.ax[0].set_xlabel('Time (ms)')
        self.plot_fig.ax[0].set_ylabel('Voltage ({0}V)'.format(cf._mu))

        # creates the free subplot
        self.plot_fig.ax[1].plot(T, spike_free, 'r')
        self.plot_fig.ax[1].plot(T, data_free['vMu'][:, i_match], 'k', linewidth=4.0)
        self.plot_fig.ax[1].set_title('Free Cluster #{0}'.format(free_ID))
        self.plot_fig.ax[1].set_xlabel('Time (ms)')
        self.plot_fig.ax[1].set_ylabel('Voltage ({0}V)'.format(cf._mu))

        #
        self.plot_fig.ax[0].grid(plot_grid)
        self.plot_fig.ax[1].grid(plot_grid)
        self.plot_fig.ax[2].grid(plot_grid)

        # resets the x-axis limits
        for ax in self.plot_fig.ax:
            ax.set_xlim(T[0], T[-1])
            ax.set_ylim(yL)

    def plot_signal_metrics(self, is_3d, m_type, all_expt=None, exp_name=None, plot_grid=True):
        '''

        :return:
        '''

        # initialisations
        pWL, pWU = 0.90, 1.10
        ex_name = cf.extract_file_name

        # retrieves the fixed/free data sets based on the type
        if self.is_multi:
            # case is multi-experiment files have been loaded
            data_fix, data_free, comp_data = self.get_multi_comp_datasets(all_expt, exp_name)

            # sets the experiment names (if multi-experiment)
            if all_expt:
                exp_files = np.array(cf.flat_list([[ex_name(x['expFile'])] * x['data'][0]['nC']
                                                    for x in self.data.cluster]))
        else:
            # case is only single experiment files have been loaded
            data_fix, data_free = self.get_comp_datasets(is_full=True)

        # ensures the data files are stored in lists
        if not isinstance(data_fix, list):
            data_fix, data_free = [data_fix], [data_free]

        # sets up the figure/axis
        self.init_plot_axes(is_3d=is_3d)

        # sets the z-scores to be plotted based on type
        if m_type == 'New Method':
            # sets the match indices and acceptance flags
            if self.is_multi:
                # sets the match/acceptance flags
                i_match = np.array(cf.flat_list([list(x.i_match) for x in comp_data]))
                is_accept = np.array(cf.flat_list([list(x.is_accept) for x in comp_data]))

                # sets the x, y and z-axis plot values
                x_plot = np.array(cf.flat_list([list(x.sig_diff) for x in comp_data]))
                y_plot = np.array(cf.flat_list([list(np.min(x.signal_feat, axis=1)) for x in comp_data]))
                z_plot = np.array(cf.flat_list([list(x.isi_corr) for x in comp_data]))

                #
                id_free = np.array(cf.flat_list(
                    [self.get_free_cluster_match_ids(x['clustID'], y.i_match) for x, y in zip(data_free, comp_data)]
                ))

            else:
                # sets the x, y and z-axis plot values
                comp = self.data.comp
                i_match, is_accept = comp.i_match, comp.is_accept
                x_plot, y_plot, z_plot = comp.sig_diff, np.min(comp.signal_feat, axis=1), comp.isi_corr
                id_free = np.array(self.get_free_cluster_match_ids(data_free[0]['clustID'], i_match))

            # sets the axis labels
            x_label = 'Max Total Distance'
            y_label = 'Signal Feature Difference'
            z_label = 'ISI Cross-Correlation'
            x_lim = [pWL * np.nanmin(x_plot), 1.0]
            y_lim = [pWL * np.nanmin(y_plot), 1.0]
            z_lim = [pWL * np.nanmin(z_plot), 1.0]
        else:
            # sets the match indices and acceptance flags
            if self.is_multi:
                # sets the match/acceptance flags
                i_match = np.array(cf.flat_list([list(x.i_match_old) for x in comp_data]))
                is_accept = np.array(cf.flat_list([list(x.is_accept_old) for x in comp_data]))

                # sets the x, y and z-axis plot values
                x_plot = np.array(cf.flat_list([list(x.sig_corr) for x in comp_data]))
                y_plot = np.array(cf.flat_list([list(x.sig_diff_old) for x in comp_data]))
                z_plot = np.array(cf.flat_list([list(np.nanmax(np.abs(x.z_score), axis=0)) for x in comp_data]))

                #
                id_free = np.array(cf.flat_list(
                    [self.get_free_cluster_match_ids(x['clustID'], y.i_match) for x, y in zip(data_free, comp_data)]
                ))
            else:
                # sets the x, y and z-axis plot values
                comp = self.data.comp
                i_match, is_accept = comp.i_match_old, comp.is_accept_old
                x_plot, y_plot, z_plot = comp.sig_corr, comp.sig_diff_old, np.nanmax(np.abs(comp.z_score), axis=0)
                id_free = np.array(self.get_free_cluster_match_ids(data_free[0]['clustID'], i_match))

            # sets the axis labels
            x_label, y_label, z_label = 'Correlation Coefficient', 'L2 Norm', 'Max Z-Score'
            x_lim = [pWL * np.nanmin(x_plot), 1.0]
            y_lim = [pWL * np.nanmin(y_plot), pWU * np.nanmax(y_plot)]
            z_lim = [pWL * np.nanmin(z_plot), pWU * np.nanmax(z_plot)]

        # sets the rejection flags
        is_plot = i_match >= 0
        type_str = ['Rejected', 'Accepted']

        # sets the accepted/reject cluster ID labels
        cm = ['g' if is_accept[x] else 'r' for x in np.where(is_plot)[0]]
        id_fix = np.array(cf.flat_list([x['clustID'] for x in data_fix]))

        # sets the scatterplot label strings
        lbl = ['Fixed #{0}\nFree #{1}\n{2}\n=========='.format(x, y, type_str[z]) for x, y, z in
               zip(id_fix[is_plot], id_free[is_plot], is_accept[is_plot])]
        lbl = ['{}\nX = {:5.3f}\nY = {:5.3f}'.format(x, y, z) for x, y, z in zip(lbl, x_plot[is_plot], y_plot[is_plot])]
        if is_3d:
            # adds in the z-values if plotting in 3D
            lbl = ['{}\nZ = {:5.3f}'.format(x, y) for x, y in zip(lbl, z_plot[is_plot])]

        if self.is_multi and all_expt:
            # appends the experiment names to the labels (if plotting multiple experiment data)
            lbl = ['{0}\n==========\n{1}'.format(x, y) for x, y in zip(exp_files[is_plot], lbl)]

        # creates the scatterplot
        if is_3d:
            # case is a 3D plot
            h = self.plot_fig.ax[0].scatter(x_plot[is_plot], y_plot[is_plot], z_plot[is_plot], marker='o', c=cm)
            self.remove_scatterplot_spines(self.plot_fig.ax[0])
            self.plot_fig.ax[0].view_init(20, -45)
        else:
            # case is a 2D plot
            h = self.plot_fig.ax[0].scatter(x_plot[is_plot], y_plot[is_plot], marker='o', c=cm)

        # creates the cursor object
        datacursor(h, formatter=formatter, point_labels=lbl, hover=True)

        # # creates the cursor object
        # cursor = mplcursors.cursor(h, hover=True)
        # cursor.connect("add", lambda sel: sel.annotation.set_text(lbl[sel.target.index]))

        # sets the scatterplot properties
        # self.plot_fig.ax[0].legend(['Accepted', 'Rejected'], loc=3)
        self.plot_fig.ax[0].set_xlabel('{0} (X)'.format(x_label))
        self.plot_fig.ax[0].set_ylabel('{0} (Y)'.format(y_label))
        self.plot_fig.ax[0].set_xlim(x_lim)
        self.plot_fig.ax[0].set_ylim(y_lim)
        self.plot_fig.ax[0].grid(plot_grid)
        self.plot_fig.draw()

        # adds the z-axis label (if a 3D plot)
        if is_3d:
            self.plot_fig.ax[0].set_zlabel('{0} (Z)'.format(z_label))
            self.plot_fig.ax[0].set_zlim(z_lim)

    def plot_old_cluster_signals(self, i_cluster, plot_all, exp_name=None, plot_grid=True):
        '''

        :return:
        '''

        # sets the acceptance flags based on the method
        if self.is_multi:
            # case is multi-experiment files have been loaded
            data_fix, data_free, comp = self.get_multi_comp_datasets(False, exp_name, is_list=False)
            n_pts = np.size(data_fix['vMu'], axis=0)
        else:
            # retrieves the fixed/free datasets
            comp, n_pts = self.data.comp, self.data.comp.n_pts
            data_fix, data_free = self.get_comp_datasets(is_full=True)

        # resets the cluster index if plotting all clusters
        i_cluster, e_str = self.check_cluster_index_input(i_cluster, plot_all, data_fix['nC'])
        if e_str is not None:
            cf.show_error(e_str,'Infeasible Cluster Indices')
            self.calc_ok = False
            return

        # sets the indices of the clusters to plot and creates the figure/axis objects
        i_cluster = self.set_cluster_indices(i_cluster)
        T, col = self.setup_time_vector(data_fix['sFreq'], n_pts), 'rg'

        # sets up the figure/axis
        n_plot = len(i_cluster)
        self.init_plot_axes(n_plot=n_plot)

        # plots the values over all subplots
        for i_plot in range(n_plot):
            # creates the new subplot
            j_plot = i_cluster[i_plot] - 1
            i_match = comp.i_match[j_plot]
            id_fix = data_fix['clustID'][j_plot]

            # only plot data values if there was a match
            if i_match >= 0:
                # plots the z-scores and the upper/lower limits
                self.plot_fig.ax[i_plot].plot(T,comp.z_score[:, j_plot], 'b')
                self.plot_fig.ax[i_plot].plot([0, n_pts], [1, 1], 'r--')
                self.plot_fig.ax[i_plot].plot([0, n_pts], [-1, -1], 'r--')

                # sets the title properties
                id_free = data_free['clustID'][i_match]
                t_str = 'Fixed #{0}/Free #{1}'.format(id_fix, id_free)
                t_col = col[int(comp.is_accept[j_plot])]
            else:
                # otherwise, set reduced title properties
                t_str, t_col = 'Fixed #{0}'.format(id_fix), 'k'

            # sets the subplot properties
            self.plot_fig.ax[i_plot].set_title(t_str, color=t_col)
            self.plot_fig.ax[i_plot].set_ylim(-4, 4)
            self.plot_fig.ax[i_plot].set_xlim(T[0], T[-1])
            self.plot_fig.ax[i_plot].set_ylabel('Z-Score')
            self.plot_fig.ax[i_plot].set_xlabel('Time (ms)')
            self.plot_fig.ax[i_plot].grid(plot_grid)

        # # sets the final figure layout
        # self.plot_fig.fig.tight_layout(h_pad=-0.9)

    def plot_new_cluster_signals(self, i_cluster, plot_all, sig_type, exp_name=None, plot_grid=True):
        '''

        :return:
        '''

        # initialisation
        reset_ylim, y_lim = True, [0, 0]

        # sets the acceptance flags based on the method
        if self.is_multi:
            # case is multi-experiment files have been loaded
            data_fix, data_free, comp = self.get_multi_comp_datasets(False, exp_name, is_list=False)
        else:
            # retrieves the fixed/free datasets
            comp = self.data.comp
            data_fix, data_free = self.get_comp_datasets(is_full=True)

        # resets the cluster index if plotting all clusters
        i_cluster, e_str = self.check_cluster_index_input(i_cluster, plot_all, data_fix['nC'])
        if e_str is not None:
            cf.show_error(e_str,'Infeasible Cluster Indices')
            self.calc_ok = False
            return

        # sets the indices of the clusters to plot and creates the figure/axis objects
        i_cluster = self.set_cluster_indices(i_cluster)
        T, col = self.setup_time_vector(data_fix['sFreq'], np.size(data_fix['vMu'], axis=0)), 'rg'

        # sets the plot data based on the signal type
        if sig_type == 'Intersection':
            # case is the signal intersection
            y_data = comp.match_intersect
            y_label = 'Intersection'
            y_lim, reset_ylim = [0, 1], False
        elif sig_type == 'Wasserstein Distance':
            # case is the wasserstain (earth-mover) distance
            y_data = comp.match_wasserstain
            y_label = 'Wasser. Distance'
        else:
            # case is the bhattacharyya distance
            y_data = comp.match_bhattacharyya
            y_label = 'BHA Distance'

        # sets up the figure/axis
        n_plot = len(i_cluster)
        self.init_plot_axes(n_plot=n_plot)

        # plots the values over all subplots
        for i_plot in range(n_plot):
            # creates the new subplot
            j_plot = i_cluster[i_plot] - 1
            i_match = comp.i_match[j_plot]
            id_fix = data_fix['clustID'][j_plot]

            # only plot data values if there was a match
            if i_match >= 0:
                # plots the z-scores and the upper/lower limits
                self.plot_fig.ax[i_plot].plot(T,y_data[:, j_plot], 'b')

                # sets the title properties
                id_free = data_free['clustID'][i_match]
                t_str = 'Fixed #{0}/Free #{1}'.format(id_fix, id_free)
                t_col = col[int(comp.is_accept[j_plot])]
            else:
                # otherwise, set reduced title properties
                t_str, t_col = 'Fixed #{0}'.format(id_fix), 'k'

            # sets the subplot properties
            self.plot_fig.ax[i_plot].set_title(t_str, color=t_col)
            self.plot_fig.ax[i_plot].set_xlim(T[0], T[-1])
            self.plot_fig.ax[i_plot].set_ylabel(y_label)
            self.plot_fig.ax[i_plot].set_xlabel('Time (ms)')
            self.plot_fig.ax[i_plot].grid(plot_grid)

            # updates the axis y-limits (if required)
            if not reset_ylim:
                # fixed limit has been set
                self.plot_fig.ax[i_plot].set_ylim(y_lim)
            elif i_match >= 0:
                # limit is based over all sub-plots
                y_lim_ax = self.plot_fig.ax[i_plot].get_ylim()
                y_lim[1] = max([y_lim[1],y_lim_ax[1]])

        if reset_ylim:
            for ax in self.plot_fig.ax:
                ax.set_ylim(y_lim)

    def plot_cluster_distances(self, n_shuffle, n_spikes, i_cluster, plot_all, p_type, exp_name=None, plot_grid=True):
        '''

        :return:
        '''

        # sets the acceptance flags based on the method
        if self.is_multi:
            # case is multi-experiment files have been loaded
            data_fix, data_free, comp = self.get_multi_comp_datasets(False, exp_name, is_list=False)
        else:
            # retrieves the fixed/free datasets
            comp = self.data.comp
            data_fix, data_free = self.get_comp_datasets()

        # resets the cluster index if plotting all clusters
        i_cluster, e_str = self.check_cluster_index_input(i_cluster, plot_all, data_fix['nC'])
        if e_str is not None:
            cf.show_error(e_str,'Infeasible Cluster Indices')
            self.calc_ok = False
            return

        # sets the indices of the clusters to plot and creates the figure/axis objects
        i_cluster = self.set_cluster_indices(i_cluster)

        # calculates the shuffled distances
        cfcn.cluster_distance(data_fix, data_free, n_shuffle, n_spikes, i_cluster)

        # initialisations
        xi = np.array(range(data_free['nC'] + 1))
        n_plot = len(i_cluster)
        x_lbl = [str(x + 1) for x in data_free['clustID']] + ['Mu']

        # sets the indices of the clusters to plot and creates the figure/axis objects
        self.init_plot_axes(n_plot=n_plot)

        # plots the values over all subplots
        for i_plot in range(n_plot):
            # creates the new subplot
            j_plot = i_cluster[i_plot] - 1
            x_plot = self.data.comp.mu_dist[i_plot, :, :].T

            # creates the final plot based on the type
            if p_type == 'boxplot':
                self.plot_fig.ax[i_plot].boxplot(x_plot)
            else:
                # calculates the mean/error values
                d_mu = np.mean(x_plot, axis=0)
                d_sem = np.mean(x_plot, axis=0)/m.sqrt(n_shuffle)

                # determines the subplot column count (for determining the capsize width)
                if i_plot == 0:
                    n_col, _ = cf.det_subplot_dim(n_plot)

                # creates the boxplot
                self.plot_fig.ax[i_plot].bar(xi, d_mu)
                self.plot_fig.ax[i_plot].errorbar(xi, d_mu, yerr=d_sem, ecolor='r', fmt='.', capsize=10.0 / n_col)

            # sets the subplot properties
            self.plot_fig.ax[i_plot].set_title('Fixed Cluster #{0}'.format(data_fix['clustID'][j_plot]))
            self.plot_fig.ax[i_plot].set_xticks(xi+int(p_type=='boxplot'))
            self.plot_fig.ax[i_plot].set_xticklabels(x_lbl)
            self.plot_fig.ax[i_plot].set_ylabel('L2 Norm')
            self.plot_fig.ax[i_plot].set_xlabel('Free Cluster ID#')
            self.plot_fig.ax[i_plot].grid(plot_grid)

            # rotates the xtick labels
            for xt in self.plot_fig.ax[i_plot].get_xticklabels():
                xt.set_rotation(90)

        # # shows the final plot
        # if p_type == 'boxplot':
        #     self.plot_fig.fig.tight_layout(h_pad=-0.8)
        # else:
        #     self.plot_fig.fig.tight_layout(h_pad=-0.9)

    def plot_cluster_isi(self, i_cluster, plot_all, t_lim, plot_all_bin, is_norm, equal_ax, exp_name, plot_grid=True):
        '''

        :param i_cluster:
        :param plot_all:
        :param is_norm:
        :return:
        '''

        # sets the acceptance flags based on the method
        if self.is_multi:
            # case is multi-experiment files have been loaded
            data_fix, data_free, comp = self.get_multi_comp_datasets(False, exp_name, is_list=False)
        else:
            # retrieves the fixed/free datasets
            comp = self.data.comp
            data_fix, data_free = self.get_comp_datasets()

        # resets the cluster index if plotting all clusters
        i_cluster, e_str = self.check_cluster_index_input(i_cluster, plot_all, data_fix['nC'])
        if e_str is not None:
            cf.show_error(e_str,'Infeasible Cluster Indices')
            self.calc_ok = False
            return

        # sets the indices of the clusters to plot and creates the figure/axis objects
        i_cluster = self.set_cluster_indices(i_cluster)

        # sets the y-axis label (based on whether the distributions are normalised)
        if is_norm:
            y_label = 'Probability'
        else:
            y_label = 'Count'

        # sets up the figure/axis
        n_plot, col, y_lim = len(i_cluster), 'kg', [0, 0]
        self.init_plot_axes(n_plot=n_plot)

        # sets the time bin coordinates
        dxi = 0.5 * np.diff(data_fix['isiHistX'][:2])
        xi = data_fix['isiHistX'][:-1] + dxi

        # sets the plot indices based on the type
        if plot_all_bin:
            # case is plotting all time bins
            ii = np.array(range(len(xi)))
        else:
            # case is plotting for a given upper limit
            ii = xi <= t_lim
            xi = xi[ii]

        # plots the values over all subplots
        for i_plot in range(n_plot):
            # creates the new subplot
            j_plot = i_cluster[i_plot] - 1
            i_match = comp.i_match[j_plot]
            id_fix = data_fix['clustID'][j_plot]

            # plots the fixed cluster isi histogram
            hist_fix = data_fix['isiHist'][j_plot]
            self.plot_fig.ax[i_plot].plot(xi, hist_fix[ii] / (1.0 + (sum(hist_fix) - 1.0)*int(is_norm)), 'b')

            # plots the fixed
            if i_match >= 0:
                # plots the z-scores and the upper/lower limits
                hist_free = data_free['isiHist'][i_match]
                self.plot_fig.ax[i_plot].plot(xi, hist_free[ii] / (1.0 + (sum(hist_free) - 1.0) * int(is_norm)), 'r')

                # sets the title properties
                id_free = data_free['clustID'][i_match]
                t_str = 'Fixed #{0}/Free #{1}'.format(id_fix, id_free)
                t_col = col[int(comp.is_accept[j_plot])]
            else:
                # otherwise, set reduced title properties
                t_str, t_col = 'Fixed #{0}'.format(id_fix), 'k'

            # sets the subplot properties
            self.plot_fig.ax[i_plot].set_title(t_str, color=t_col)
            self.plot_fig.ax[i_plot].set_ylabel(y_label)
            self.plot_fig.ax[i_plot].set_xlabel('ISI Time (ms)')
            self.plot_fig.ax[i_plot].set_xlim(xi[0] - dxi, xi[-1] + dxi)
            self.plot_fig.ax[i_plot].grid(plot_grid)

            # updates the axis y-limits (if required)
            if equal_ax and (i_match >= 0):
                y_lim_ax = self.plot_fig.ax[i_plot].get_ylim()
                y_lim[1] = max([y_lim[1],y_lim_ax[1]])

        if equal_ax:
            for ax in self.plot_fig.ax:
                ax.set_ylim(y_lim)

    ######################################################
    ####    CELL CLASSIFICATION ANALYSIS FUNCTIONS    ####
    ######################################################

    def plot_classification_metrics(self, exp_name, all_expt, c_met1, c_met2, c_met3, use_3met, use_pca,
                                    class_type, plot_grid=True):
        '''

        :return:
        '''

        ###################################################
        ####    INITIALISATIONS & MEMORY ALLOCATION    ####
        ###################################################

        # initialisations
        cluster = self.data.cluster
        col = ['b', 'r', convert_rgb_col(_bright_cyan), convert_rgb_col(_bright_yellow)]
        t_lim, n_met, p_size = [-1e10, 1e10], 7, 2.5
        lg_str = ['Narrow Spikes', 'Wide Spikes']

        # initialises the subplot axes
        self.clear_plot_axes()
        self.plot_fig.ax = np.empty(4, dtype=object)

        # sets up the classification scatterplot
        if use_3met or use_pca:
            m_size = 30
            self.plot_fig.ax[0] = self.plot_fig.figure.add_subplot(1, 2, 1, projection='3d')
        else:
            m_size = 60
            self.plot_fig.ax[0] = self.plot_fig.figure.add_subplot(1, 2, 1)

        # sets up the signal subplots
        for i in range(3):
            self.plot_fig.ax[i+1] = self.plot_fig.figure.add_subplot(3, 2, 2*(i+1))

        # initialisations
        c_metric = ['Trough to 2nd Peak Time (ms)', '2nd Peak to Trough Ratio', 'Trough Half-Width (ms)',
                    'Peak Ratio'] #, '2nd Peak Half-Width (ms)', '2nd Peak Relaxation Time (ms)',
                    # 'Firing Rate (Hz)']

        # determines if there the metric selections are valid (only if not using pca)
        if use_pca:
            # sets the metric indices to all
            m_ind = np.array(range(len(c_metric)))
        else:
            # initialisations
            e_str = None
            if use_3met and ((c_met1 in [c_met2, c_met3]) or (c_met2 == c_met3)):
                    e_str = 'Classification function can only be run with unique metrics.'
            elif (c_met1 == c_met2):
                e_str = 'Classification function can only be run with unique metrics.'

            # if there was an error then exit the function
            if e_str is not None:
                cf.show_error(e_str, 'Invalid Metric Selection')
                self.calc_ok = False
                return
            else:
                # otherwise set the metric indices
                if use_3met:
                    m_ind = np.array([c_metric.index(c_met1), c_metric.index(c_met2), c_metric.index(c_met3)])
                else:
                    m_ind = np.array([c_metric.index(c_met1), c_metric.index(c_met2)])

        # sets up the data array
        if all_expt:
            i_expt = np.array(range(len(cluster)))
        else:
            # otherwise, set the index of the
            i_expt = np.array([cf.get_expt_index(exp_name, cluster)])

        # combines the information from each of the selected experiments
        expt_sf = np.concatenate([cluster[i]['sigFeat'] for i in i_expt], axis=0)
        expt_cl = np.array(cf.flat_list([cluster[i]['clustID'] for i in i_expt]))
        expt_id = np.array(cf.flat_list([[cf.extract_file_name(cluster[i]['expFile'])]*cluster[i]['nC'] for i in i_expt]))
        expt_vm = np.concatenate([cluster[i]['vMu'] for i in i_expt], axis=1)
        s_freq = np.array(cf.flat_list([[cluster[i]['sFreq']] * cluster[i]['nC'] for i in i_expt]))
        t_scale = 1000.0 / s_freq

        # retrieves the action types
        if self.data.classify.action_set:
            lg_str += ['Inhibitory', 'Excitatory']
            act_str, _act_type = ['None', 'Inhibitory', 'Excitatory'], dcopy(self.data.classify.act_type)
            act_type = np.array(cf.flat_list([_act_type[i][_act_type[i] >= 0] for i in i_expt]))

        ##############################################
        ####    CLUSTERING METRIC CALCULATIONS    ####
        ##############################################

        # creates the cluster array
        x_clust = np.zeros((len(expt_id), n_met))

        # case is the trough to 2nd peak time
        x_clust[:, 0] = (expt_sf[:, 2] - expt_sf[:, 1]) * t_scale

        # case is the trough to 2nd peak time
        B = np.array([expt_vm[int(x), i] for i, x in enumerate(expt_sf[:, 2])])
        D = np.array([expt_vm[int(x), i] for i, x in enumerate(expt_sf[:, 1])])
        x_clust[:, 1] = -B / D

        # case is the trough half-width
        x_clust[:, 2] = (expt_sf[:, 4] - expt_sf[:, 3]) * t_scale

        # case is the maximum spike ratio
        A = np.array([expt_vm[int(x), i] for i, x in enumerate(expt_sf[:, 0])])
        B = np.array([expt_vm[int(x), i] for i, x in enumerate(expt_sf[:, 2])])
        x_clust[:, 3] = (B - A) / (B + A)

        # case is the 2nd peak half-width
        x_clust[:, 4] = expt_sf[:, 5] * t_scale

        # case is the 2nd peak relaxation time
        x_clust[:, 5] = expt_sf[:, 6] * t_scale

        # case is the mean firing rate
        x_clust[:, 6] = np.array(cf.flat_list([
                            [len(x) / cluster[i]['tExp'] for x in cluster[i]['tSpike']] for i in i_expt]))

        #######################################
        ####    CLUSTER CLASSIFICATION     ####
        #######################################

        # sets the plot data based on whether PCA is used
        # x_clust_scaled = scale(x_clust, with_std=False, with_mean=False)
        x_clust_scaled = scale(x_clust[:, m_ind], with_std=False)
        if use_pca:
            # transforms the data using PCA
            pca = PCA(n_components=np.size(x_clust_scaled, axis=1))
            x_clust_scaled = pca.fit_transform(x_clust_scaled)

        # runs the k-mean clustering on the data
        if class_type == 'K-Means':
            kmeans = KMeans(n_clusters=2, init='random').fit(x_clust_scaled)
            grp_idx = kmeans.labels_
        else:
            # sets up the gaussian mixture model
            n_grp, p_tol = 15, 0.05
            g_mod = np.empty(n_grp, dtype=object)
            grp_idx0 = np.zeros((np.size(x_clust_scaled, axis=0), n_grp))
            grp_prob = np.zeros((np.size(x_clust_scaled, axis=0), n_grp, 2))
            grp_means = np.zeros((n_grp, len(m_ind), 2))

            # calculates the predicted values for the set number of iterations
            for i in range(n_grp):
                # sets up the gaussian mixture model
                g_mod[i] = GMM(n_components=2, covariance_type='full', init_params='random')

                # keep looping until a valid grouping has been found
                while 1:
                    # calclulates the gaussian mixture model and predicts the groupings
                    g_mod[i].fit(x_clust_scaled)
                    grp_means[i, :, :] = g_mod[i].means_.T
                    grp_idx0[:, i] = g_mod[i].predict(x_clust_scaled)

                    # if not all values are from the same group then exit the loop
                    dgrp_idx0 = np.diff(grp_idx0[:, i])
                    if (np.max(dgrp_idx0) - np.min(dgrp_idx0)) > 0:
                        grp_prob[:, i, :] = g_mod[i].predict_proba(x_clust_scaled)
                        break

            # determines
            while 1:
                A = np.logical_and(grp_prob[:, :, 1] > p_tol, grp_prob[:, :, 1] < (1.0 - p_tol))
                all_ok = np.where(np.all(np.logical_not(A), axis=0))[0]

                if np.any(all_ok):
                    break
                else:
                    p_tol += 0.01

            # sets the final cluster index array
            grp_idx0 = grp_idx0[:, all_ok]
            i_grp, i_ind = cfcn.det_gmm_cluster_groups(grp_means[all_ok, :, :])
            grp_idx = grp_idx0[:, i_ind[np.argmax([len(x) for x in i_ind])][0]]

        ###########################
        ####    SPIKE SETUP    ####
        ###########################

        # normalises the signal by the minimum points
        pp_VmN = np.empty(np.size(expt_vm, axis=1), dtype=object)
        for i in range(np.size(expt_vm, axis=1)):
            # sets up the time vector
            n_pts, i_min = len(expt_vm[:, i]), np.argmin(expt_vm[:, i])
            T = self.setup_time_vector(s_freq[i], n_pts)

            # creates the piece-wise polynomial objects
            t_lim = [max(t_lim[0], T[0] - T[i_min]), min(t_lim[1], T[-1] - T[i_min])]
            pp_VmN[i] = pchip(T - T[i_min], -expt_vm[:, i] / np.min(expt_vm[i_min, i]))

        # sets the final
        T_plot = np.arange(t_lim[0], t_lim[1], 0.01)
        vm_plot = np.zeros((len(T_plot), np.size(expt_vm, axis=1)))
        for i in range(np.size(vm_plot, axis=1)):
            vm_plot[:, i] = pp_VmN[i](T_plot)

        # sets the signals for the
        is_grp0 = grp_idx == 0
        Vm_1, Vm_2 = vm_plot[:, is_grp0], vm_plot[:, np.logical_not(is_grp0)]

        ##########################################
        ####    FINAL GROUP CLASSIFICATION    ####
        ##########################################

        # determines the final classification of the cluster groups
        Vm_1mn, Vm_2mn = np.mean(Vm_1, axis=1), np.mean(Vm_2, axis=1)

        # determines which spike group is wider
        if np.argmax(Vm_1mn[np.argmin(Vm_1mn):]) > np.argmax(Vm_2mn[np.argmin(Vm_2mn):]):
            # group 1 is wider
            type_str, Vm_W, Vm_N = ['Wide', 'Narrow'], Vm_1, Vm_2
        else:
            # group 2 is wider
            type_str, Vm_W, Vm_N = ['Narrow', 'Wide'], Vm_2, Vm_1

        # sets the final group strings
        grp_str = np.array([type_str[0] if x else type_str[1] for x in is_grp0])

        # sets the data into the classification class object
        class_para = {'c_met1': c_met1, 'c_met2': c_met2, 'c_met3': c_met3, 'class_type': class_type}
        self.data.classify.set_classification_data(self.data, class_para, expt_id, x_clust, grp_str, c_metric)

        ###################################
        ####    K-MEANS SCATTERPLOT    ####
        ###################################

        #
        cm, x_clust_plt = np.array([col[0] if gs == 'Narrow' else col[1] for gs in grp_str]), x_clust[:, m_ind]

        # sets the scatterplot labels
        if use_pca:
            # case is using PCA
            x_label, y_label, z_label = 'PCA Coordinate #1', 'PCA Coordinate #2', 'PCA Coordinate #3'
        else:
            # case is using the metrics directly
            x_label, y_label, z_label = c_met1, c_met2, c_met3

        # sets the scatterplot tooltip strings
        lbl = ['Expt = {0}\n==========\nID# = {1}\nGroup = {2}\n=========='.format(
            x, y, z+1) for x, y, z in zip(expt_id, expt_cl, grp_idx)
        ]
        lbl = ['{}\nX = {:5.3f}\nY = {:5.3f}'.format(x, y, z) for x, y, z in zip(lbl, x_clust[:, 0], x_clust[:, 1])]

        #
        if use_3met or use_pca:
            lbl = ['{}\nZ = {:5.3f}'.format(x, y) for x, y in zip(lbl, x_clust[:, 2])]

        #
        if self.data.classify.action_set:
            lbl = ['{0}\n==========\n{1}'.format(x, act_str[y]) for x, y in zip(lbl, act_type)]
            i1, i2 = act_type == 1, act_type == 2

        # creates the scatterplot
        if use_3met or use_pca:
            if self.data.classify.action_set:
                self.plot_fig.ax[0].scatter(x_clust_plt[i1, 0], x_clust_plt[i1, 1], x_clust_plt[i1, 2],
                                            marker='o', c=col[2], s=p_size*m_size)
                self.plot_fig.ax[0].scatter(x_clust_plt[i2, 0], x_clust_plt[i2, 1], x_clust_plt[i2, 2],
                                            marker='o', c=col[3], s=p_size*m_size)

            # case is a 3D plotf
            h = self.plot_fig.ax[0].scatter(x_clust_plt[:, 0], x_clust_plt[:, 1], x_clust_plt[:, 2],
                                            marker='o', c=cm, s=m_size)
            self.remove_scatterplot_spines(self.plot_fig.ax[0])
            self.plot_fig.ax[0].view_init(20, -45)
        else:
            if self.data.classify.action_set:
                self.plot_fig.ax[0].scatter(x_clust_plt[i1, 0], x_clust_plt[i1, 1], marker='o', c=col[2], s=p_size*m_size)
                self.plot_fig.ax[0].scatter(x_clust_plt[i2, 0], x_clust_plt[i2, 1], marker='o', c=col[3], s=p_size*m_size)

            # case is a 2D plot
            h = self.plot_fig.ax[0].scatter(x_clust_plt[:, 0], x_clust_plt[:, 1], marker='o', c=cm, s=m_size)
            self.plot_fig.ax[0].grid(plot_grid)

        # retrieves the axis limits
        x_lim = self.plot_fig.ax[0].get_xlim()
        y_lim = self.plot_fig.ax[0].get_ylim()

        if use_3met or use_pca:
            # creates the legend plots
            z_lim = self.plot_fig.ax[0].get_zlim()
            h1 = self.plot_fig.ax[0].scatter(x_lim[0] - 1, y_lim[0] - 1, z_lim[0] - 1, marker='o', c=col[0])
            h2 = self.plot_fig.ax[0].scatter(x_lim[0] - 1, y_lim[0] - 1, z_lim[0] - 1, marker='o', c=col[1])

            if self.data.classify.action_set:
                h3 = self.plot_fig.ax[0].scatter(x_lim[0] - 1, y_lim[0] - 1, z_lim[0] - 1, marker='o', c=col[2])
                h4 = self.plot_fig.ax[0].scatter(x_lim[0] - 1, y_lim[0] - 1, z_lim[0] - 1, marker='o', c=col[3])
        else:
            h1 = self.plot_fig.ax[0].scatter(x_lim[0] - 1, y_lim[0] - 1, marker='o', c=col[0])
            h2 = self.plot_fig.ax[0].scatter(x_lim[0] - 1, y_lim[0] - 1, marker='o', c=col[1])

            if self.data.classify.action_set:
                h3 = self.plot_fig.ax[0].scatter(x_lim[0] - 1, y_lim[0] - 1, marker='o', c=col[2])
                h4 = self.plot_fig.ax[0].scatter(x_lim[0] - 1, y_lim[0] - 1, marker='o', c=col[3])

        # creates the legend object
        if self.data.classify.action_set:
            self.plot_fig.ax[0].legend([h1, h2, h3, h4], lg_str)
        else:
            self.plot_fig.ax[0].legend([h1, h2], lg_str)

        # resets the access limit
        self.plot_fig.ax[0].set_xlim(x_lim)
        self.plot_fig.ax[0].set_ylim(y_lim)

        # sets the subplot axes properties
        self.plot_fig.ax[0].set_title('Classification Metrics')
        self.plot_fig.ax[0].set_xlabel('(X) {0}'.format(x_label))
        self.plot_fig.ax[0].set_ylabel('(Y) {0}'.format(y_label))

        if use_3met or use_pca:
            self.plot_fig.ax[0].set_zlim(z_lim)
            self.plot_fig.ax[0].set_zlabel('\n(Z) {0}'.format(z_label))

        # creates the cursor object
        datacursor(h, formatter=formatter, point_labels=lbl, hover=True)

        # resizes the axis position
        ax_pos = self.plot_fig.ax[0].get_position()
        self.plot_fig.ax[0].set_position([ax_pos.x0, ax_pos.y0, 1.05 * ax_pos.width, ax_pos.height])

        ###################################
        ####    MEAN SIGNAL SUBPLOT    ####
        ###################################

        # sets up the time vector
        n_grp_t = len(grp_idx)
        n_grp_N = sum([x == 'Narrow' for x in grp_str])
        p1, p2 = '{:4.1f}'.format(100*(n_grp_N/n_grp_t)), '{:4.1f}'.format(100*((n_grp_t - n_grp_N)/n_grp_t))
        t_str = ['Narrow Spikes ({0}/{1}) = {2}%'.format(n_grp_N, n_grp_t, p1),
                 'Wide Spikes ({0}/{1}) = {2}%'.format(n_grp_t - n_grp_N, n_grp_t, p2),
                 'Combined Spikes']

        # creates the spike signal grouping subplots
        for i in range(3):
            # case is the narrow spikes
            if i in [0, 2]:
                self.plot_fig.ax[i+1].plot(T_plot, Vm_N, col[0])
                if (i == 0):
                    self.plot_fig.ax[i + 1].plot(T_plot, np.mean(Vm_N, axis=1), 'k', linewidth=3)

            # case is the wide spikes
            if i in [1, 2]:
                self.plot_fig.ax[i+1].plot(T_plot, Vm_W, col[1])
                if (i == 1):
                    self.plot_fig.ax[i + 1].plot(T_plot, np.mean(Vm_W, axis=1), 'k', linewidth=3)

            # plots the spike line
            self.plot_fig.ax[i + 1].plot([0, 0], self.plot_fig.ax[i+1].get_ylim(), 'k--')

            # sets the plot
            self.plot_fig.ax[i+1].set_xlim(T_plot[0], T_plot[-1])
            self.plot_fig.ax[i+1].set_title(t_str[i])
            self.plot_fig.ax[i+1].set_xlabel('Time (ms)')
            self.plot_fig.ax[i+1].set_ylabel('Normalised Voltage')
            self.plot_fig.ax[i + 1].grid(plot_grid)

    def plot_classification_ccgram(self, plot_exp_name, action_type, plot_type, i_plot, plot_all,
                                   window_size, plot_grid=True):
        '''

        :return:
        '''

        # initialisations
        cl_data = self.data.classify
        i_expt = cf.get_expt_index(plot_exp_name, self.data.cluster)
        at_type = {'Excitatory': [1, 1], 'Inhibitory': [0, 1], 'Excited': [1, -1], 'Inhibited': [0, -1],
                   'Rejected Excitatory': [3, 1], 'Rejected Inhibitory': [2, 1], 'Abnormal': [4, 1]}[action_type]

        # resets the cluster index if plotting all clusters
        data_plot = self.data.cluster[i_expt]
        i_plot, e_str = self.check_cluster_index_input(i_plot, plot_all, len(cl_data.c_type[i_expt][at_type[0]]))
        if e_str is not None:
            cf.show_error(e_str, 'Infeasible Cluster Indices')
            self.calc_ok = False
            return

        # fixed/calculation parameters
        i_plot = np.array(i_plot) - 1
        t_min, t_max = 1.5, 4.0
        pW_lo, pW_hi = 0.85, 1.15
        c_type, n_plot = self.data.classify.c_type[i_expt][at_type[0]], len(i_plot)

        # sets the indices of the clusters to plot and creates the figure/axis objects
        xi = self.data.cluster[i_expt]['ccGramXi']
        ind = np.where(abs(xi) <= window_size)[0]
        x_lim, bin_sz = [xi[ind[0]], xi[ind[-1]]], xi[1] - xi[0]

        # sets the activity search region
        if at_type[1] == 1:
            act_reg = np.logical_and(xi[ind] >= t_min, xi[ind] <= t_max)
        else:
            act_reg = np.logical_and(xi[ind] <= -t_min, xi[ind] >= -t_max)

        # retrieves the plot data
        t_spike = self.data.cluster[i_expt]['tSpike']
        cl_ind = cl_data.c_type[i_expt][at_type[0]][i_plot]
        ccG = cl_data.ccG_T[i_expt][at_type[0]][:, i_plot]
        ci_lo = cl_data.ci_lo[i_expt][at_type[0]][:, i_plot]
        ci_hi = cl_data.ci_hi[i_expt][at_type[0]][:, i_plot]

        # reverses the arrays if inhibited/excited
        if at_type[1] == -1:
            cl_ind, ccG, ci_lo, ci_hi = cl_ind[:, ::-1], ccG[::-1, :], ci_lo[::-1, :], ci_hi[::-1, :]

        # sets the cluster id tags
        cl_id = np.vstack([np.array(cl_data.clust_id[i_expt])[x] for x in cl_ind])

        # plots the values over all subplots
        self.init_plot_axes(n_plot=n_plot)
        for i_plot in range(n_plot):
            # calculates the scale factor
            search_lim = []
            f_scale = len(t_spike[cl_ind[i_plot, 0]]) / 1000.0

            # calculates the confidence intervals
            n_hist = ccG[ind, i_plot] / f_scale
            ci_lo_new = ci_lo[ind, i_plot] / f_scale
            ci_hi_new = ci_hi[ind, i_plot] / f_scale

            # calculates the overal limits of the axis
            y_lim = [pW_lo * min(np.min(n_hist), np.min(ci_lo_new)),
                     pW_hi * max(np.max(n_hist), np.max(ci_hi_new))]

            #
            if at_type[0] in [0, 2]:
                ii = cf.det_largest_index_group(np.logical_and(n_hist <= ci_lo_new, act_reg))
            elif at_type[0] in [1, 3]:
                ii = cf.det_largest_index_group(np.logical_and(n_hist >= ci_hi_new, act_reg))
            else:
                ii = np.logical_or(np.logical_and(n_hist <= ci_lo_new, act_reg),
                                    np.logical_and(n_hist >= ci_hi_new, act_reg))

            # creates the search limit region patches
            if at_type[1] == 1:
                search_lim.append(Rectangle(((t_min - bin_sz / 2.0), y_lim[0]), ((t_max + bin_sz) - t_min), y_lim[1]))
            else:
                search_lim.append(Rectangle((-(t_max + bin_sz / 2.0), y_lim[0]), ((t_max + bin_sz) - t_min), y_lim[1]))
            pc = PatchCollection(search_lim, facecolor='r', alpha=0.25, edgecolor='k')
            self.plot_fig.ax[i_plot].add_collection(pc)

            # plots the auto-correlogram and confidence interval limits
            if plot_type == 'bar':
                self.plot_fig.ax[i_plot].bar(xi[ind], height=n_hist, width=bin_sz)
                self.plot_fig.ax[i_plot].bar(xi[ind[ii]], height=n_hist[ii], width=bin_sz, color='r')
            else:
                self.plot_fig.ax[i_plot].scatter(xi[ind], n_hist)
                self.plot_fig.ax[i_plot].scatter(xi[ind[ii]], n_hist[ii], color='r')

            # plots the lower/upper limits
            self.plot_fig.ax[i_plot].plot(xi[ind], ci_lo_new, 'k--')
            self.plot_fig.ax[i_plot].plot(xi[ind], ci_hi_new, 'k--')

            # sets the zero time-lag marker
            self.plot_fig.ax[i_plot].plot(np.zeros(2), y_lim, 'k--')

            # sets the axis properties
            self.plot_fig.ax[i_plot].set_title('#{0} vs #{1}'.format(cl_id[i_plot, 0], cl_id[i_plot, 1]))
            self.plot_fig.ax[i_plot].set_ylabel('Frequency (Hz)')
            self.plot_fig.ax[i_plot].set_xlabel('Time Lag (ms)')
            self.plot_fig.ax[i_plot].set_xlim(x_lim)
            self.plot_fig.ax[i_plot].set_ylim(y_lim)
            self.plot_fig.ax[i_plot].grid(plot_grid)

    #############################################
    ####    ROTATIONAL ANALYSIS FUNCTIONS    ####
    #############################################

    def plot_rotation_trial_spikes(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, n_bin,
                                   plot_grid):
        '''

        :param plot_scope:
        :return:
        '''

        # applies the rotation filter to the dataset
        r_obj = RotationFilteredData(self.data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, False)
        self.create_raster_hist(r_obj, n_bin, plot_grid)

    def plot_phase_spike_freq(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, p_value,
                                  stats_type, plot_scope, plot_trend, plot_grid, is_3d):
        '''

        :param rot_filt:
        :param i_cluster:
        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_scope:
        :param plot_grid:
        :return:
        '''

        # applies the rotation filter to the dataset
        r_obj = RotationFilteredData(self.data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, False)

        # creates the spike frequency plot/statistics tables
        self.create_spike_freq_plot(r_obj, plot_grid, plot_trend, p_value, stats_type, is_3d)

    def plot_spike_freq_heatmap(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, dt):
        '''

        :param rot_filt:
        :param i_cluster:
        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_scope:
        :param plot_grid:
        :return:
        '''

        # applies the rotation filter to the dataset
        r_obj = RotationFilteredData(self.data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, False)
        self.create_spike_heatmap(r_obj, dt)

    def plot_motion_direction_selectivity(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope,
                                          plot_cond, plot_trend, plot_even_axis, p_type, plot_grid):
        '''

        :param rot_filt:
        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_grid:
        :return:
        '''

        def create_single_selectivity_plot(r_obj, plot_grid, t_type_exp, plot_even_axis):
            '''

            :param r_obj:
            :param plot_grid:
            :param t_type_exp:
            :param plot_even_axis:
            :return:
            '''

            # parameters
            p_value = 0.05
            plt_type = ['CW Motion Sensitivity', 'CCW Motion Sensitivity', 'Direction Selectivity']
            props = {'ha': 'center', 'va': 'center', 'size': 15, 'weight': 'bold'}
            is_scatter = p_type == 'scatterplot'

            #
            ind_type = [np.where([tt in r_filt['t_type'] for r_filt in r_obj.rot_filt_tot])[0] for tt in t_type_exp]
            is_ok = np.array([(len(x) > 0) for x in ind_type])
            t_type_exp, ind_type = list(np.array(t_type_exp)[is_ok]), list(np.array(ind_type)[is_ok])
            n_filt, n_type = len(ind_type[0]), 3

            # determines the filter indices pertaining to the black trial conditions, and also the indices of the
            # filters that belong to each unique trial condition
            i_black = t_type_exp.index('Black')
            i_other = np.where(np.array(range(len(t_type_exp))) != i_black)[0]

            #
            h_ylbl = np.empty(n_type, dtype=object)
            t_str = ['{0} vs {1}'.format(t_type_exp[i], t_type_exp[0]) for i in i_other]
            can_plt = np.array([np.size(x, axis=0) for x in r_obj.t_spike]) > 0
            sp_f0, sp_f = cf.calc_phase_spike_freq(r_obj)
            n_trial = [np.size(x, axis=0) for x in sp_f]

            # calculates the CW/CCW Motion Selectivity
            plt_data = np.empty(3, dtype=object)
            plt_data[0] = [np.divide(x[:, 1] - x[:, 0], x[:, 0]) if (x is not None) else None for x in sp_f]      # CW Motion Selectivity
            plt_data[1] = [np.divide(x[:, 2] - x[:, 0], x[:, 0]) if (x is not None) else None for x in sp_f]      # CCW Motion Selectivity

            # calculates the Direction Selectivity
            dCW = [(x[:, 1] - x[:, 0]) if (x is not None) else None for x in sp_f]
            dCCW = [(x[:, 2] - x[:, 0]) if (x is not None) else None for x in sp_f]
            plt_data[2] = [np.array([np.abs((z[zz, 1]-z[zz, 2])/(z[zz, 1]+z[zz, 2])) if (np.sign(xx*yy) == 1) else 1
                            for xx, yy, zz in zip(x,y,range(np.size(z,axis=0)))]) if (x is not None) else None
                            for x, y, z in zip(dCW, dCCW, sp_f)]

            # creates the plot axes
            c = cf.get_plot_col(n_filt)
            if is_scatter:
                n_cond = len(t_type_exp) - 1
                self.init_plot_axes(n_row=n_type, n_col=n_cond)
                sp_corr = np.zeros((n_cond * n_type, n_filt), dtype=float)

                for i_type in range(n_type):
                    for i_cond in range(n_cond):
                        # sets the current plot index
                        i_plt = i_type * n_cond + i_cond

                        # updates the grid properties
                        self.plot_fig.ax[i_plt].grid(plot_grid)

                        # plots the values for each filter type
                        h_plt = []
                        for i_filt in range(n_filt):
                            # set the black/other plot indices, and the cell matches between the 2 trial types
                            ii_black, ii_other = ind_type[i_black][i_filt], ind_type[i_other[i_cond]][i_filt]
                            if (not can_plt[ii_other]) or (not can_plt[ii_black]):
                                # if the current configuration is infeasible, then continue
                                continue
                            else:
                                # otherwise, determine the matching cell indices
                                n_plt = min([n_trial[ii_black], n_trial[ii_other]])
                                i_xy_plt = np.array(range(n_plt))

                            # sets the x/y plot values
                            x, y = plt_data[i_type][ii_black], plt_data[i_type][ii_other][i_xy_plt]
                            xx, yy = x[i_xy_plt], y[i_xy_plt]

                            # creates the scatter plot
                            h_plt.append(self.plot_fig.ax[i_plt].scatter(xx, yy, c=c[i_filt]))
                            if plot_trend:
                                ii = np.logical_not(np.logical_or(np.isnan(x), np.isinf(x)))
                                sp_corr_new, _ = curve_fit(cf.lin_func, np.array(x)[ii], np.array(y)[ii])
                                sp_corr[i_plt, i_filt] = sp_corr_new[0]

                            # sets the subplot properties
                            h_ylbl[i_type] = self.plot_fig.ax[i_plt].set_ylabel(t_type_exp[i_other[i_cond]])

                            # calculates the non-paired wilcoxon test between the 2 conditions
                            results = r_stats.wilcox_test(FloatVector(x), FloatVector(y), paired=False, exact=True)
                            pv_str = '{:5.3f}'.format(results[results.names.index('p.value')][0])

                            # sets the title (first row only)
                            if i_type == 0:
                                self.plot_fig.ax[i_plt].set_title('{0}\n(P-Value = {1})'.format(t_str[i_cond], pv_str))
                            else:
                                self.plot_fig.ax[i_plt].set_title('(P-Value = {0})'.format(pv_str))

                            # creates the legend (first subplot only)
                            if (i_plt == 0) and (r_obj.n_filt != len(t_type_exp)):
                                lg_str = [r_obj.lg_str[i].replace('\nBlack', '') for i in ind_type[i_black]]
                                self.plot_fig.ax[i_plt].legend(h_plt, lg_str, loc=0)

                            # sets the x-axis label (final row only)
                            if (i_type + 1) == n_type:
                                self.plot_fig.ax[i_plt].set_xlabel('Black')

                # resets the x/y of the motion/direction selectivity subplots so that they are even
                if plot_even_axis:
                    cf.set_equal_axis_limits(self.plot_fig.ax, np.array(range(2*n_cond)))
                    cf.set_equal_axis_limits(self.plot_fig.ax, np.array(range(2*n_cond, 3*n_cond)))

                # plots unitary lines for each subplot
                for i_plt, ax in enumerate(self.plot_fig.ax):
                    xL, yL = ax.get_xlim(), ax.get_ylim()
                    xx = [min(xL[0], yL[0]), max(xL[1], yL[1])]
                    ax.plot(xx, xx, 'k--')

                    if plot_trend:
                        cf.set_axis_limits(ax, xL, yL)
                        for i_filt in range(n_filt):
                            ax.plot(5 * np.array(xL), 5 * sp_corr[i_plt, i_filt] * np.array(xL), '--', c=c[i_filt])

                p_ofs = 0.2 if (n_cond == 3) else 0.2 / (1.75 * (3 - n_cond))
                for i_type in range(n_type):
                    i_plt = i_type * n_cond
                    xL, yL = self.plot_fig.ax[i_plt].get_xlim(), self.plot_fig.ax[i_plt].get_ylim()
                    self.plot_fig.ax[i_plt].text(xL[0] - p_ofs * (xL[1] - xL[0]), np.mean(yL), plt_type[i_type],
                                                 props, rotation=90)
            else:
                # initialisations
                n_cond, n_all = len(plt_type), len(r_obj.lg_str)
                self.init_plot_axes(n_row=n_cond, n_col=1)

                for i_cond in range(n_cond):
                    plt_data_nw = plt_data[i_cond]
                    # if (i_cond + 1) == n_cond:
                    #     a = 1
                    # else:
                    #     plt_data_nw = plt_data[i_cond]

                    # creates the bubble-boxplot
                    cf.create_bubble_boxplot(self.plot_fig.ax[i_cond], plt_data_nw)

                    # sets the axis properties
                    self.plot_fig.ax[i_cond].set_title(plt_type[i_cond])
                    self.plot_fig.ax[i_cond].set_ylabel(plt_type[i_cond].replace('CCW ', '').replace('CW ', ''))
                    self.plot_fig.ax[i_cond].set_xticklabels(r_obj.lg_str)
                    self.plot_fig.ax[i_cond].grid(plot_grid)

            #
            self.plot_fig.fig.set_tight_layout(False)
            self.plot_fig.fig.tight_layout(rect=[0.0, 0.0, 1, 0.945])
            self.plot_fig.fig.suptitle('Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0], r_obj.ch_id[0][0]),
                                       fontsize=16, fontweight='bold')

        def create_wc_selectivity_plot(r_obj, plot_grid, t_type_exp, plot_even_axis):
            '''

            :param r_obj:
            :param plot_grid:
            :return:
            '''

            # parameters
            p_value = 0.05
            plt_type = ['CW Motion Selectivity', 'CCW Motion Selectivity', 'Direction Selectivity']
            props = {'ha': 'center', 'va': 'center', 'size': 15, 'weight': 'bold'}
            is_scatter = p_type == 'scatterplot'

            #
            ind_type = [np.where([tt in r_filt['t_type'] for r_filt in r_obj.rot_filt_tot])[0] for tt in t_type_exp]
            is_ok = np.array([(len(x) > 0) for x in ind_type])
            t_type_exp, ind_type = list(np.array(t_type_exp)[is_ok]), list(np.array(ind_type)[is_ok])

            # determines the filter indices pertaining to the black trial conditions, and also the indices of the
            # filters that belong to each unique trial condition
            i_black = t_type_exp.index('Black')
            i_other = np.where(np.array(range(len(t_type_exp))) != i_black)[0]

            # memory allocation and other initialisations
            n_filt, n_type = len(ind_type[0]), 3
            h_ylbl = np.empty(n_type, dtype=object)
            t_str = ['{0} vs {1}'.format(t_type_exp[i], t_type_exp[0]) for i in i_other]

            #
            can_plt = np.array([np.size(x, axis=0) for x in r_obj.t_spike]) > 0

            # calculates the mean spiking rate for each cell across each phase
            sp_f0 = [cf.sp_freq_fcn(x, y[0]) if np.size(x, axis=0) > 0 else None
                                                for x, y in zip(r_obj.t_spike, np.array(r_obj.t_phase))]
            sp_mn = [np.mean(x, axis=1) if x is not None else None for x in sp_f0]

            # combines the all the data from each phase type
            n_stats = 1 + int(not r_obj.is_ud)
            sf_stats = np.empty(n_stats, dtype=object)
            for i_sub in range(n_stats):
                if r_obj.is_ud:
                    # calculates the wilcoxon signed rank test between the baseline/stimuli phases
                    if not r_obj.is_single_cell:
                        sf_stats[i_sub] = cf.calc_spike_freq_stats(sp_f0, [0, 1], concat_results=False)
                else:
                    # calculates the wilcoxon signed rank test between the stimuli phases
                    i1, i2 = 1 * (i_sub > 1), 1 + (i_sub > 0)
                    if not r_obj.is_single_cell:
                        sf_stats[i_sub] = cf.calc_spike_freq_stats(sp_f0, [i1, i2], concat_results=False)

            # calculates the CW/CCW Motion Selectivity
            plt_data = np.empty(3, dtype=object)
            plt_data[0] = [np.divide(x[:, 1] - x[:, 0], x[:, 0]) if (x is not None) else None for x in sp_mn]      # CW Motion Selectivity
            plt_data[1] = [np.divide(x[:, 2] - x[:, 0], x[:, 0]) if (x is not None) else None for x in sp_mn]      # CCW Motion Selectivity

            # calculates the Direction Selectivity
            dCW = [(x[:, 1] - x[:, 0]) if (x is not None) else None for x in sp_mn]
            dCCW = [(x[:, 2] - x[:, 0]) if (x is not None) else None for x in sp_mn]
            plt_data[2] = [np.array([np.abs((z[zz, 1]-z[zz, 2])/(z[zz, 1]+z[zz, 2])) if (np.sign(xx*yy) == 1) else 1
                            for xx, yy, zz in zip(x,y,range(np.size(z,axis=0)))]) if (x is not None) else None
                            for x, y, z in zip(dCW, dCCW, sp_mn)]

            # determines which cells from the black phase responded to the stimuli (CHECK WITH SEPI!)
            if r_obj.is_ud:
                # case is uniform-drifting, so determine if there was a significant response to the stimuli phase
                is_sig = [(sf_stats[0][x] < p_value) if can_plt[x] else None for x in ind_type[i_black]]
            else:
                # otherwise, determine if there was a significant response to the CW or CCW phases
                is_sig = [np.logical_or(sf_stats[0][x] < p_value, sf_stats[1][x] < p_value)
                          if can_plt[x] else None for x in ind_type[i_black]]

            # creates the plot axes
            c = cf.get_plot_col(n_filt)
            if is_scatter:
                n_cond = len(t_type_exp) - 1
                self.init_plot_axes(n_row=n_type, n_col=n_cond)
                sp_corr = np.empty((n_cond * n_type, n_filt), dtype=float)

                for i_type in range(n_type):
                    for i_cond in range(n_cond):
                        # sets the current plot index
                        i_plt = i_type * n_cond + i_cond

                        # updates the grid properties
                        self.plot_fig.ax[i_plt].grid(plot_grid)

                        # plots the values for each filter type
                        h_plt = []
                        for i_filt in range(n_filt):
                            # set the black/other plot indices, and the cell matches between the 2 trial types
                            ii_black, ii_other = ind_type[i_black][i_filt], ind_type[i_other[i_cond]][i_filt]
                            if (not can_plt[ii_other]) or (not can_plt[ii_black]):
                                # if the current configuration is infeasible, then continue
                                continue
                            else:
                                # otherwise, determine the matching cell indices
                                i_cell_black, i_cell_other = cf.det_cell_match_indices(r_obj, [ii_black, ii_other])

                            # sets the x/y plot values
                            x, y = plt_data[i_type][ii_black][i_cell_black], plt_data[i_type][ii_other][i_cell_other]
                            if (i_type + 1) == n_type:
                                # if plotting motion sensitivity, then only plot the significant points
                                x, y = x[is_sig[i_filt][i_cell_black]], y[is_sig[i_filt][i_cell_black]]

                            # creates the scatter plot
                            h_plt.append(self.plot_fig.ax[i_plt].scatter(x, y, c=c[i_filt]))
                            if plot_trend:
                                ii = np.logical_not(np.logical_or(np.isnan(x), np.isinf(x)))
                                sp_corr_new, _ = curve_fit(cf.lin_func, np.array(x)[ii], np.array(y)[ii])
                                sp_corr[i_plt, i_filt] = sp_corr_new[0]

                            # sets the subplot properties
                            h_ylbl[i_type] = self.plot_fig.ax[i_plt].set_ylabel(t_type_exp[i_other[i_cond]])

                            # sets the title (first row only)
                            if i_type == 0:
                                self.plot_fig.ax[i_plt].set_title(t_str[i_cond])

                            # creates the legend (first subplot only)
                            if (i_plt == 0) and (r_obj.n_filt != len(t_type_exp)):
                                lg_str = [r_obj.lg_str[i].replace('\nBlack', '') for i in ind_type[i_black]]
                                self.plot_fig.ax[i_plt].legend(h_plt, lg_str, loc=0)

                            # sets the x-axis label (final row only)
                            if (i_type + 1) == n_type:
                                self.plot_fig.ax[i_plt].set_xlabel('Black')

                # resets the x/y of the motion/direction selectivity subplots so that they are even
                if plot_even_axis:
                    cf.set_equal_axis_limits(self.plot_fig.ax, np.array(range(2*n_cond)))
                    cf.set_equal_axis_limits(self.plot_fig.ax, np.array(range(2*n_cond, 3*n_cond)))

                # plots unitary lines for each subplot
                for i_plt, ax in enumerate(self.plot_fig.ax):
                    xL, yL = ax.get_xlim(), ax.get_ylim()
                    xx = [min(xL[0], yL[0]), max(xL[1], yL[1])]
                    ax.plot(xx, xx, 'k--')

                    if plot_trend:
                        cf.set_axis_limits(ax, xL, yL)
                        for i_filt in range(n_filt):
                            ax.plot(5 * np.array(xL), 5 * sp_corr[i_plt, i_filt] * np.array(xL), '--', c=c[i_filt])

                p_ofs = 0.2 if (n_cond == 3) else 0.2 / (1.75 * (3 - n_cond))
                for i_type in range(n_type):
                    i_plt = i_type * n_cond
                    xL, yL = self.plot_fig.ax[i_plt].get_xlim(), self.plot_fig.ax[i_plt].get_ylim()
                    self.plot_fig.ax[i_plt].text(xL[0] - p_ofs * (xL[1] - xL[0]), np.mean(yL), plt_type[i_type],
                                                 props, rotation=90)
            else:
                # initialisations
                n_cond, n_all = len(plt_type), len(r_obj.lg_str)
                self.init_plot_axes(n_row=n_cond, n_col=1)

                for i_cond in range(n_cond):
                    plt_data_nw = plt_data[i_cond]
                    # if (i_cond + 1) == n_cond:
                    #     a = 1
                    # else:
                    #     plt_data_nw = plt_data[i_cond]

                    # creates the bubble-boxplot
                    cf.create_bubble_boxplot(self.plot_fig.ax[i_cond], plt_data_nw)

                    # sets the axis properties
                    self.plot_fig.ax[i_cond].set_title(plt_type[i_cond])
                    self.plot_fig.ax[i_cond].set_ylabel(plt_type[i_cond].replace('CCW ', '').replace('CW ', ''))
                    self.plot_fig.ax[i_cond].set_xticklabels(r_obj.lg_str)
                    self.plot_fig.ax[i_cond].grid(plot_grid)

        ################################################################################################################
        ################################################################################################################

        # sets up the rotational filter (for the specified trial condition given in plot_cond)
        rot_filt, e_str, et_str = cf.setup_trial_condition_filter(rot_filt, plot_cond)
        if e_str is not None:
            cf.show_error(e_str, et_str)
            self.ok = False
            return

        # filters the rotational data and runs the analysis function
        r_obj = RotationFilteredData(self.data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, False)
        if plot_scope == 'Individual Cell':
            create_single_selectivity_plot(r_obj, plot_grid, rot_filt['t_type'], plot_even_axis)
        else:
            create_wc_selectivity_plot(r_obj, plot_grid, rot_filt['t_type'], plot_even_axis)

    def plot_spike_freq_kinematics(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope,
                                   pos_bin, vel_bin, n_smooth, is_smooth, plot_grid):
        '''

        :param rot_filt:
        :param i_cluster:
        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_scope:
        :param b_sz:
        :param plot_grid:
        :return:
        '''

        def create_kinematic_plots(r_obj, b_sz, n_smooth, is_smooth, plot_grid):
            '''

            :param b_sz:
            :param plot_grid:
            :return:
            '''

            # if there was an error setting up the rotation calculation object, then exit the function with an error
            if not r_obj.is_ok:
                self.calc_ok = False
                return

            # initialisations
            c, k_rng = cf.get_plot_col(r_obj.n_filt), [90, 80]
            proj_type = ['polar', None, None, None]
            title_str = ['Position', 'Velocity', 'Displacement', 'Speed']

            # calculates the position/velocity values over all trials/cells
            k_sf, xi_bin, _ = rot.calc_kinemetic_spike_freq(self.data, r_obj, b_sz, False)

            # creates the plot outlay and titles
            self.init_plot_axes(n_row=2, n_col=2, n_plot=2, proj_type=proj_type)

            for i_type in range(2):
                # sets the bin values
                xi_mid = 0.5 * (xi_bin[i_type][1:] + xi_bin[i_type][:-1])
                if i_type == 0:
                    # sets the tick-labels for the polar subplot
                    xi_min = xi_mid[0] - np.diff(xi_mid[0:2])[0] / 2
                    xi_mid = np.pi * (1 - (xi_mid - xi_min) / np.abs(2 * xi_min))

                # sets up the tickmarks
                x_tick = np.linspace(-k_rng[i_type], k_rng[i_type], 7 + 2 * i_type)

                # creates the radial plots for each of the filter types
                h_plt = []
                for i_filt in range(r_obj.n_filt):
                    # calculates the
                    k_sf_mn = np.mean(k_sf[i_type][i_filt], axis=0)

                    # creates the plot and resets the labels
                    if proj_type[i_type] is None:
                        # smooths the signal (if required)
                        if is_smooth:
                            k_sf_mn = medfilt(k_sf_mn, n_smooth)

                        h_plt.append(self.plot_fig.ax[i_type].plot(xi_mid, k_sf_mn, 'o-', color=c[i_filt]))
                    else:
                        #
                        d_xi = 0.5 * (xi_mid[0] - xi_mid[1]) * (((2 * i_filt + 1) / r_obj.n_filt) - 1)
                        h_plt.append(self.plot_fig.ax[i_type].bar(xi_mid - d_xi, height=k_sf_mn, color=c[i_filt],
                                                                  width=np.deg2rad(b_sz[i_type]) / r_obj.n_filt,
                                                                  linewidth=0))

                    if i_filt == 0:
                        self.plot_fig.ax[i_type].set_title(title_str[i_type])
                        mlt = 1 if proj_type[i_type] is None else -1

                        if proj_type[i_type] is 'polar':
                            self.plot_fig.ax[i_type].set_xticks(np.pi * (x_tick - xi_min) / np.abs(2 * xi_min))
                        else:
                            self.plot_fig.ax[i_type].set_xticks(x_tick)

                        self.plot_fig.ax[i_type].set_xticklabels([str(int(np.round(mlt * x))) for x in x_tick])

                # sets the legend (first subplot only)
                if i_type == 0:
                    self.plot_fig.ax[i_type].legend(r_obj.lg_str, loc=2)

            # adds in the bin lines for the polar plot
            yL = self.plot_fig.ax[0].get_ylim()
            self.plot_fig.ax[0].set_ylim(yL)
            for xi in (np.pi / 2) * (1 + xi_bin[0] / k_rng[0]):
                self.plot_fig.ax[0].plot([xi, xi], 2 * np.array(yL), 'k--')

            # resets the subplot layout
            self.plot_fig.fig.set_tight_layout(False)
            if r_obj.is_single_cell:
                # sets the layout size
                self.plot_fig.fig.tight_layout(rect=[0, 0.01, 1, 0.955])

                # sets the cell cluster ID
                self.plot_fig.fig.suptitle('Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0], r_obj.ch_id[0][0]),
                                           fontsize=16, fontweight='bold')
            else:
                # sets the layout size
                self.plot_fig.fig.tight_layout(rect=[0, 0, 1, 1])

            # resets the axis properties
            for i in range(len(self.plot_fig.ax)):
                # resets the axis limits
                ax = self.plot_fig.ax[i]
                ax.grid(plot_grid)
                h_title = ax.set_title(title_str[i])

                #
                if proj_type[i] is not None:
                    ax.set_thetamin(0)
                    ax.set_thetamax(180)

                    # resets the subplot position
                    h_ofs = 0.060 + 0.015 * int(r_obj.is_single_cell)
                    i_col, i_row = i % 2, np.floor(i / 2)
                    ax.set_position([i_col * 0.5, 0.5 * (1 - i_row) - h_ofs, 0.5, 0.5])

                    # adds and resets the position of the title
                    h_title.set_position([0.5, 0.85])

        ################################################################################################################
        ################################################################################################################

        # ensures the smoothing window is an odd integer
        if is_smooth:
            if n_smooth % 2 != 1:
                # if not, then exit with an error
                e_str = 'The median smoothing filter window span must be an odd integer.'
                cf.show_error(e_str, 'Incorrect Smoothing Window Span')
                self.calc_ok = False
                return

        # applies the rotation filter to the dataset
        r_obj = RotationFilteredData(self.data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, False)
        create_kinematic_plots(r_obj, [float(pos_bin), float(vel_bin)], n_smooth, is_smooth, plot_grid)

    def plot_overall_direction_bias(self, rot_filt, plot_exp_name, plot_all_expt, plot_grid, plot_scope):
        '''

        :param rot_filt:
        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_grid:
        :return:
        '''

        # initialises the rotation filter (if not set)
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        r_obj = RotationFilteredData(self.data, rot_filt, None, plot_exp_name, plot_all_expt, 'Whole Experiment', False)
        if not r_obj.is_ok:
            self.calc_ok = False
            return

        # memory allocation and parameters
        s_plt = np.empty(r_obj.n_filt, dtype=object)
        f_stats = np.zeros((r_obj.n_filt, 4), dtype=object)

        # calculates the individual trial/mean spiking rates and sets up the plot/stats arrays
        _, sp_f = cf.calc_phase_spike_freq(r_obj)

        # sets up the plot values and statistics table strings
        for i_filt in range(r_obj.n_filt):
            # sets the CW/CCW values and calculates the gradient/sets the plot values
            CW, CCW = sp_f[i_filt][:, 1], sp_f[i_filt][:, 2]
            m_CCW_CW, s_plt[i_filt] = CCW / CW, np.vstack((CW, CCW)).T
            n_CW, n_CCW = np.sum(m_CCW_CW < 1), np.sum(m_CCW_CW >= 1)

            # calculates the wilcoxon text between the avg. CW/CCW spiking frequencies
            results = r_stats.wilcox_test(FloatVector(CW), FloatVector(CCW), paired=True, exact=True)
            p_value = results[results.names.index('p.value')][0]

            # determines the number of CW/CCW preferred cells and sets the p-value string
            f_stats[i_filt, 0], f_stats[i_filt, 1] = str(n_CW), str(n_CCW)
            f_stats[i_filt, 2] = '{:5.3f}{}'.format(p_value, sig_str_fcn(p_value, 0.05))

            # sets the overall bias type
            if (n_CW == n_CCW) or (p_value > p_lim):
                # if the preferred direction counts are equal, or the p-value is not significant then set None
                f_stats[i_filt, 3] = 'None'
            else:
                # otherwise, set the direction with the higher preferred count
                f_stats[i_filt, 3] = 'CW' if (n_CW > n_CCW) else 'CCW'


        # sets the plot/table properties
        c, h_plt = cf.get_plot_col(r_obj.n_filt), []
        col_hdr = ['CW Pref', 'CCW Pref', 'P-Value', 'Bias']
        row_hdr = ['#{0}'.format(i + 1) for i in range(r_obj.n_filt)]
        c_col = cf.get_plot_col(len(col_hdr), r_obj.n_filt)

        self.plot_fig.fig.set_tight_layout(False)
        self.plot_fig.fig.tight_layout(rect=[0.01, 0.02, 0.98, 0.97])
        self.init_plot_axes(n_row=1, n_col=2)

        # re-sizes the plot axis
        ax_pos, d_wid, d_x0 = [x.get_position() for x in self.plot_fig.ax], 0.2, 0.02
        self.plot_fig.ax[0].set_position([ax_pos[0].x0 - d_x0, ax_pos[0].y0, ax_pos[0].width + d_wid, ax_pos[0].height])
        self.plot_fig.ax[1].set_position([ax_pos[1].x0 + d_wid, ax_pos[1].y0, ax_pos[1].width - d_wid, ax_pos[1].height])
        self.plot_fig.ax[1].axis('off')

        # creates the scatter plot
        for i_filt in range(r_obj.n_filt):
            h_plt.append(self.plot_fig.ax[0].plot(s_plt[i_filt][:, 0], s_plt[i_filt][:, 1], 'o', c=c[i_filt]))

        # sets the plot labels
        self.plot_fig.ax[0].set_ylabel('Mean CCW Spiking Freq. (Hz)')
        self.plot_fig.ax[0].set_xlabel('Mean CW Spiking Freq. (Hz)')
        self.plot_fig.ax[0].grid(plot_grid)

        # resets the plot axis limits so that the plot axis is square
        ax_lim = np.ceil(max(self.plot_fig.ax[0].get_xlim()[1], self.plot_fig.ax[0].get_ylim()[1]))
        cf.set_axis_limits(self.plot_fig.ax[0], [0, ax_lim], [0, ax_lim])
        self.plot_fig.ax[0].plot([-1, ax_lim + 1], [-1, ax_lim + 1], 'k--')

        # adds the plot legend
        lg_str = ['#{0} - {1}'.format(i + 1, x) for i, x in enumerate(r_obj.lg_str)]
        self.plot_fig.ax[0].legend([x[0] for x in h_plt], lg_str, loc=0)

        # # creates the title text object
        # t_str = 'Direction Bias Statistics'
        # h_title = self.plot_fig.ax[0].text(0.5, 1, t_str, fontsize=15, horizontalalignment='center')

        # sets up the n-value table
        cf.add_plot_table(self.plot_fig, 1, table_font, f_stats, row_hdr, col_hdr, c, c_col,
                          None, n_col=3, p_wid=1.5)

    def plot_depth_direction_selectivity(self, rot_filt, plot_exp_name, plot_all_expt, plot_grid):
        '''

        :param rot_filt:
        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_grid:
        :return:
        '''

        # applies the rotation filter to the dataset
        r_obj = RotationFilteredData(self.data, rot_filt, None, plot_exp_name, plot_all_expt, 'Whole Experiment', False)
        self.create_depth_dirsel_plot(r_obj, plot_grid)

    ##################################/##############
    ####    UNIFORM DRIFT ANALYSIS FUNCTIONS    ####
    ################################################

    def plot_unidrift_trial_spikes(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, n_bin,
                                   plot_grid, rmv_median):
        '''

        :param rot_filter:
        :param plot_scope:
        :return:
        '''

        # applies the rotation filter to the dataset
        r_obj = RotationFilteredData(self.data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, True)
        self.create_raster_hist(r_obj, n_bin, plot_grid)

    def plot_unidrift_spike_freq(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, p_value,
                                 stats_type, plot_scope, plot_trend, plot_grid):
        '''

        :param rot_filt:
        :param i_cluster:
        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_scope:
        :param plot_grid:
        :return:
        '''

        # initialises the rotation filter (if not set)
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(True)

        # splits up the uniform drifting into CW/CCW phases
        t_phase, t_ofs = self.fcn_data.curr_para['t_phase'], self.fcn_data.curr_para['t_ofs']
        r_obj, ind_type = cf.split_unidrift_phases(self.data, rot_filt, i_cluster, plot_exp_name,
                                                   plot_all_expt, plot_scope, t_phase, t_ofs)
        if r_obj is None:
            e_str = 'The entered analysis duration and offset is greater than the experimental phase duration:\n\n' \
                    '  * Analysis Duration + Offset = {0} s.\n  * Experiment Phase Duration = {1} s.\n\n' \
                    'Enter a correct analysis duration/offset combination before re-running ' \
                    'the function.'.format(t_phase + t_ofs, 2.0)
            cf.show_error(e_str, 'Incorrect Analysis Function Parameters')

            self.calc_ok = False
            return

        # applies the rotation filter to the dataset
        self.create_spike_freq_plot(r_obj, plot_grid, plot_trend, p_value, stats_type, ind_type=ind_type)

    def plot_unidrift_spike_heatmap(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, dt):
        '''

        :param rot_filt:
        :param i_cluster:
        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_scope:
        :param plot_grid:
        :return:
        '''

        # applies the rotation filter to the dataset
        r_obj = RotationFilteredData(self.data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, True)
        self.create_spike_heatmap(r_obj, dt)

    ######################################
    ####    ROC ANALYSIS FUNCTIONS    ####
    ######################################

    def plot_direction_roc_curves_single(self, rot_filt, i_cluster, plot_exp_name, plot_grid,
                                         plot_all_expt, plot_scope):
        '''

        :param rot_filt:
        :param i_cluster:
        :param plot_exp_name:
        :param plot_grid:
        :return:
        '''

        def calc_cond_significance(roc, r_data):
            '''

            :param roc:
            :return:
            '''

            # memory allocation
            n_filt = len(roc)
            auc_stats = np.empty((n_filt, n_filt), dtype=object)

            # calculates the p-values over all filter conditions
            for i_filt in range(n_filt):
                for j_filt in range(i_filt, n_filt):
                    # calculates the p-value for unique indices only
                    if i_filt != j_filt:
                        p_value = cf.calc_inter_roc_significance(roc[i_filt][0], roc[j_filt][0],
                                                               r_data.cond_auc_stats_type, r_data.n_boot_cond_grp)

                        p_value_str = '{:5.3f}{}'.format(p_value, sig_str_fcn(p_value, 0.05))
                        auc_stats[i_filt, j_filt] = auc_stats[j_filt, i_filt] = p_value_str
                    else:
                        auc_stats[i_filt, j_filt] = 'N/A'

            # returns the p-value array
            return auc_stats

        # checks cluster index if plotting all clusters
        i_expt = cf.get_expt_index(plot_exp_name, self.data.cluster)
        i_cluster, e_str = self.check_cluster_index_input(i_cluster, False, self.data.cluster[i_expt]['nC'])
        if e_str is not None:
            cf.show_error(e_str,'Infeasible Cluster Indices')
            self.calc_ok = False
            return

        # initialises the rotation filter (if not set)
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        r_obj = RotationFilteredData(self.data, rot_filt, i_cluster[0], plot_exp_name, False, 'Individual Cell', False)
        if not r_obj.is_ok:
            self.calc_ok = False
            return

        # memory allocation and other initialisations
        n_filt, r_data = r_obj.n_filt, self.data.rotation
        t_type, c = [x['t_type'][0] for x in r_obj.rot_filt_tot], cf.get_plot_col(n_filt)
        is_boot = int(r_data.cond_auc_stats_type == 'Bootstrapping')
        lg_str = ['({0}) - {1}'.format(i+1, x) for i, x in enumerate(r_obj.lg_str)]

        # determines the indices of the cell in the overall array
        cell_ind = [np.where(np.logical_and(np.array(r_data.cond_cl_id[x]) == r_obj.cl_id[0],
                                                     r_data.cond_i_expt[x] == r_obj.i_expt[0]))[0][0] for x in t_type]

        # retrieves the necessary plot values
        roc = [r_data.cond_roc[x][y] for x, y in zip(t_type, cell_ind)]
        roc_xy = [r_data.cond_roc_xy[x][y] for x, y in zip(t_type, cell_ind)]
        roc_auc = [r_data.cond_roc_auc[x][y, 2] for x, y in zip(t_type, cell_ind)]
        y_err = [[r_data.cond_ci_lo[x][y, is_boot], r_data.cond_ci_hi[x][y, is_boot]] for x, y in zip(t_type, cell_ind)]

        ######################################
        ####    SUBPLOT/TABLE CREATION    ####
        ######################################

        # initialises the plot axes
        self.init_plot_axes(n_row=1, n_col=2)

        # creates the roc curve/auc subplots
        self.create_roc_curves(self.plot_fig.ax[0], roc_xy, lg_str, plot_grid)
        self.create_single_auc_plot(self.plot_fig.ax[1], roc_auc, plot_grid, r_obj.lg_str, y_err)

        # sets the subplot titles
        self.plot_fig.ax[0].set_title('ROC Curves')
        self.plot_fig.ax[1].set_title('ROC AUC')

        # resizes the figure to include the super-title
        self.plot_fig.fig.set_tight_layout(False)
        self.plot_fig.fig.suptitle('Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0], r_obj.ch_id[0][0]),
                                   fontsize=16, fontweight='bold')
        self.plot_fig.fig.tight_layout(rect=[0, 0.01, 1, 0.955])

        if r_obj.n_filt > 1:
            # sets the column/row headers
            auc_stats = calc_cond_significance(roc, r_data)
            row_hdr = col_hdr = ['#{0}'.format(str(x+1)) for x in range(n_filt)]

            # calculates the table dimensions
            cf.add_plot_table(self.plot_fig, 1, table_font, auc_stats, row_hdr, col_hdr, c, c,
                              'bottom', pfig_sz=0.955)

    def plot_direction_roc_curves_whole(self, rot_filt, plot_exp_name, plot_all_expt, use_avg, connect_lines, md_grp_type,
                              resp_grp_type, use_resp_grp_type, plot_grid, plot_scope):
        '''

        :param rot_filt:
        :param plot_exp_name:
        :param plot_all_expt:
        :param use_table:
        :param use_avg:
        :param grp_type:
        :param connect_lines:
        :param plot_grid:
        :return:
        '''

        # initialises the rotation filter (if not set)
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        r_obj = RotationFilteredData(self.data, rot_filt, None, plot_exp_name, plot_all_expt, 'Whole Experiment', False)
        if not r_obj.is_ok:
            self.calc_ok = False
            return
        elif connect_lines:
            # if (r_obj.n_filt != 2):
            #     e_str = 'The connect AUC points option requires exactly 2 filter options. Either de-select ' \
            #             'the option or alter the filter options.'
            #     cf.show_error(e_str, 'AUC Connection Line Error')
            #     self.calc_ok = False
            #     return
            #
            # el
            if len(np.unique([x['t_type'][0] for x in r_obj.rot_filt_tot])) != 2:
                e_str = 'The connect AUC point option requires 2 trial condition filter options. Either de-select ' \
                        'the option or alter the filter options.'
                cf.show_error(e_str, 'AUC Connection Line Error')
                self.calc_ok = False
                return

        if use_resp_grp_type:
            g_type = ['None', 'Rotation', 'Visual', 'Both']
            ig_type = g_type.index(resp_grp_type)
        else:
            g_type = ['MS/DS', 'MS/Not DS', 'Not MS', 'All Cells']
            ig_type = g_type.index(md_grp_type)

        # initialisations
        n_filt, r_data = r_obj.n_filt, self.data.rotation
        lg_str = np.array(['({0}) - {1}'.format(i + 1, x) for i, x in enumerate(r_obj.lg_str)])
        has_data, c = np.ones(n_filt, dtype=bool), cf.get_plot_col(n_filt)

        # memory allocation
        A, B = np.empty(n_filt, dtype=object), np.zeros(n_filt, dtype=object)
        roc_xy = dcopy(A)
        if use_avg and (not r_obj.is_single_cell):
            roc_xy_avg, roc_auc_avg = dcopy(A), dcopy(B)

        # determine the matching cell indices between the current and black filter
        i_cell_b = dcopy(A)
        for i_filt in range(r_obj.n_filt):
            # finds the corresponding cell types between the overall and user-specified filters
            r_obj_tt = r_data.r_obj_cond[r_obj.rot_filt_tot[i_filt]['t_type'][0]]
            i_cell_b[i_filt], _ = cf.det_cell_match_indices(r_obj_tt, [0, i_filt], r_obj)

        # sets the significance flags (based on the type)
        if use_resp_grp_type:
            g_type_data = dcopy(r_data.ds_gtype)
            is_sig = g_type_data == ig_type
        else:
            st_type = ['Wilcoxon Paired Test', 'Delong', 'Bootstrapping'].index(r_data.phase_grp_stats_type)
            g_type_data = dcopy(r_data.phase_gtype)[:, st_type]

            if (ig_type + 1) == len(g_type):
                is_sig = np.ones(np.size(g_type_data, axis=0), dtype=bool)
            else:
                is_sig = g_type_data == ig_type

        # sets the significance criteria for each filter type
        i_cell_sig = [is_sig[z] for z in i_cell_b]

        # if there are no valid selections, then exit
        if np.all(np.array([len(x) for x in i_cell_sig]) == 0):
            e_str = 'The Rotational Analysis filter configuration does not produce any matching results.\n' \
                    'Re-run the function by selecting a different filter configuration.'
            cf.show_error(e_str, 'No Matching Results!')
            self.calc_ok = False
            return

        for i_filt in range(n_filt):
            # memory allocation
            if i_filt == 0:
                roc_auc, pref_cw_dir = dcopy(A), dcopy(A)

            #
            if sum(i_cell_sig[i_filt]) == 0:
                has_data[i_filt] = False
                continue

            # other initialisations and memory allocations
            n_cell = sum(i_cell_sig[i_filt])
            if n_cell == 0:
                continue

            pref_cw_dir[i_filt] = np.zeros(n_cell, dtype=bool)
            tt_filt = r_obj.rot_filt_tot[i_filt]['t_type'][0]

            # roc_xy[i_filt] = r_data.cond_roc_xy[tt_filt][i_cell_sig[i_filt]]
            # roc_auc[i_filt] = r_data.cond_roc_auc[tt_filt][i_cell_sig[i_filt], 2]
            roc_xy[i_filt] = r_data.cond_roc_xy[tt_filt][i_cell_b[i_filt]][i_cell_sig[i_filt]]
            roc_auc[i_filt] = r_data.cond_roc_auc[tt_filt][i_cell_b[i_filt]][i_cell_sig[i_filt], 2]

            # calculates the roc curves overall trials (for each cell)
            for i_cell in range(n_cell):
                # if integral is less than 0.5, then set the complementary values
                if roc_auc[i_filt][i_cell] < 0.5:
                    roc_xy[i_filt][i_cell] = roc_xy[i_filt][i_cell][:, ::-1]
                    roc_auc[i_filt][i_cell] = 1 - roc_auc[i_filt][i_cell]
                    pref_cw_dir[i_filt][i_cell] = True

            # if using the pooled cells, then
            if use_avg:
                # calculates the average roc curve/integrals
                if sum(i_cell_sig[i_filt]) > 0:
                    roc_xy_avg[i_filt] = cfcn.calc_avg_roc_curve(roc_xy[i_filt])
                    roc_auc_avg[i_filt] = np.trapz(roc_xy_avg[i_filt][:, 1], roc_xy_avg[i_filt][:, 0])

        #
        roc_xy, roc_auc = roc_xy[has_data], roc_auc[has_data]
        pref_cw_dir, lg_str0, n_filt = pref_cw_dir[has_data], np.array(r_obj.lg_str)[has_data], np.sum(has_data)
        if use_avg:
            roc_xy_avg, roc_auc_avg = roc_xy_avg[has_data], roc_auc_avg[has_data]

        #################################
        ####    ROC CURVE SUBPLOT    ####
        #################################

        # initialises the plot axes
        self.init_plot_axes(n_row=1, n_col=2)

        # creates
        if use_avg:
            self.create_roc_curves(self.plot_fig.ax[0], roc_xy_avg, lg_str, plot_grid)
        else:
            self.create_roc_curves(self.plot_fig.ax[0], roc_xy, lg_str, plot_grid)

        # creates the multi-cell auc plot
        self.create_multi_auc_plot(self.plot_fig.ax[1], roc_auc, plot_grid, connect_lines, lg_str0)

        #
        self.plot_fig.ax[0].set_title('ROC Curves ({0})'.format(g_type[ig_type]))
        self.plot_fig.ax[1].set_title('ROC AUC ({0})'.format(g_type[ig_type]))

        # resets the subplot dimensions
        self.plot_fig.fig.set_tight_layout(False)
        self.plot_fig.fig.tight_layout(rect=[0, 0.01, 1, 0.980])

        ############################################
        ####    FILTER CONDITION STATS TABLE    ####
        ############################################

        # resets the figure layout (single cell only)
        if (n_filt > 1):
            # memory allocation
            n_cell, cc = [len(x) for x in roc_auc], cf.get_plot_col(n_filt)
            is_paired = True if (np.max(n_cell) - np.min(n_cell)) == 0 else False
            auc_stats = np.empty((n_filt, n_filt), dtype=object)

            # calculates the auc p-values
            for i_row in range(n_filt):
                for i_col in range(i_row, n_filt):
                    if i_col != i_row:
                        results = r_stats.wilcox_test(FloatVector(roc_auc[i_row]), FloatVector(roc_auc[i_col]),
                                                      paired=is_paired, exact=True)
                        p_value = results[results.names.index('p.value')][0]

                        p_value_str = '{:5.3f}{}'.format(p_value, sig_str_fcn(p_value, 0.05))
                        auc_stats[i_row, i_col] = auc_stats[i_col, i_row] = p_value_str
                    else:
                        auc_stats[i_row, i_col] = 'N/A'

            # sets the column/row headers
            row_hdr = col_hdr = ['#{0}'.format(str(x+1)) for x in range(n_filt)]

            # calculates the table dimensions
            cf.add_plot_table(self.plot_fig, 0, table_font, auc_stats, row_hdr,
                              col_hdr, cc, cc, t_loc='bottom')

        ######################################
        ####    CELL GROUP COUNT TABLE    ####
        ######################################

        # table dimensioning values
        cc, has_non_cond = cf.get_plot_col(2), True
        col_hdr = dcopy(g_type)

        # other initialisations
        n_ttype = len(np.unique([x['t_type'][0] for x in np.array(r_obj.rot_filt_tot)[has_data]]))
        if (n_ttype == r_obj.n_filt):
            # no
            has_non_cond = False
        elif (n_ttype == 1):
            #
            lg_str = r_obj.lg_str
        else:
            #
            lg_str = [x.split('\n')[:-1] for x in r_obj.lg_str]

        n_grp = 3 + use_resp_grp_type
        if has_non_cond:
            # sets the counts for each grouping
            n_cell_grp_filt = [[np.sum(g_type_data[y] == x) for x in range(n_grp)] for y in i_cell_b]
            n_cell_grp_cond, ind = np.unique(np.vstack(n_cell_grp_filt), axis=0, return_index=True)

            n_cell_grp_cond = n_cell_grp_cond[np.argsort(ind), :]
            n_sig_grp = np.vstack((n_cell_grp_cond, np.sum(n_cell_grp_cond, axis=0)))
            n_sig_grp = np.hstack((n_sig_grp, np.sum(n_sig_grp, axis=1).reshape(-1, 1)))

            # case is there as there is at least one non-condition based grouping
            row_hdr0, i_row = np.unique(cf.flat_list(lg_str), return_inverse=True)
            row_hdr = ['#{0}'.format(i + 1) for i in range(len(row_hdr0))] + ['Total']
            row_cols = [cc[1]] * len(row_hdr0) + [[0.75, 0.75, 0.75]]
        else:
            # case is there are no non-condition based groupings
            row_hdr, row_cols = ['Total'], [[0.75, 0.75, 0.75]]
            n_sig_grp = np.array([np.sum(g_type_data[i_cell_b[0]] == x) for x in range(n_grp)])
            n_sig_grp = np.append(n_sig_grp, np.sum(n_sig_grp)).reshape(1, -1)

        #
        if use_resp_grp_type:
            col_hdr = col_hdr + ['Total']

        # calculates the table dimensions
        cf.add_plot_table(self.plot_fig, 1, table_font, n_sig_grp, row_hdr,
                          col_hdr, row_cols, [cc[0]] * len(col_hdr), t_loc='bottom')

    def plot_velocity_roc_curves_single(self, rot_filt, i_cluster, plot_exp_name, vel_y_rng,
                                         spd_y_rng, use_vel, plot_grid, plot_all_expt, plot_scope):
        '''

        :param rot_filt:
        :param i_cluster:
        :param plot_exp_name:
        :param vel_y_rng:
        :param spd_y_rng:
        :param use_vel:
        :param plot_grid:
        :param plot_all_expt:
        :param plot_scope:
        :return:
        '''

        def setup_plot_axes():
            '''

            :return:
            '''

            # sets up the axes dimensions
            top, bottom, pH, wspace, hspace = 0.9, 0.06, 0.01, 0.2, 0.2

            # creates the gridspec object
            gs = gridspec.GridSpec(2, 2, width_ratios=[1 / 2] * 2, height_ratios=[1 / 2] * 2, figure=self.plot_fig.fig,
                                   wspace=wspace, hspace=hspace, left=0.05, right=0.98, bottom=bottom, top=top)

            # creates the subplots
            self.plot_fig.ax = np.empty(3, dtype=object)
            self.plot_fig.ax[0] = self.plot_fig.figure.add_subplot(gs[0, 0])
            self.plot_fig.ax[1] = self.plot_fig.figure.add_subplot(gs[0, 1])
            self.plot_fig.ax[2] = self.plot_fig.figure.add_subplot(gs[1, :])

        # checks cluster index if plotting all clusters
        i_expt = cf.get_expt_index(plot_exp_name, self.data.cluster)
        i_cluster, e_str = self.check_cluster_index_input(i_cluster, False, self.data.cluster[i_expt]['nC'])
        if e_str is not None:
            cf.show_error(e_str,'Infeasible Cluster Indices')
            self.calc_ok = False
            return

        # initialises the rotation filter (if not set)
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        r_obj = RotationFilteredData(self.data, rot_filt, i_cluster[0], plot_exp_name, False, 'Individual Cell', False)
        if not r_obj.is_ok:
            self.calc_ok = False
            return

        # initialisations and other array indexing
        n_filt, r_data = r_obj.n_filt, self.data.rotation
        lg_str, c = r_obj.lg_str, cf.get_plot_col(n_filt)
        is_boot = int(r_data.kine_auc_stats_type == 'Bootstrapping')

        # memory allocation
        A, roc_auc, y_err = np.empty(n_filt, dtype=object), np.zeros(n_filt), []
        roc_xy, roc_auc_full, is_sig, ci_lo, ci_hi = dcopy(A), dcopy(A), dcopy(A), dcopy(A), dcopy(A)

        # sets the comparison bin for the velocity/speed arrays
        if use_vel:
            i_bin = list(r_data.vel_xi).index(int(vel_y_rng.split()[0]))
            _roc_xy, _roc_auc = dcopy(r_data.vel_roc_xy), dcopy(r_data.vel_roc_auc)
            _ci_lo, _ci_hi = dcopy(r_data.vel_ci_lo), dcopy(r_data.vel_ci_hi)
            xi_bin = dcopy(0.5 * (r_data.vel_xi[:-1] + r_data.vel_xi[1:]))
        else:
            i_bin = list(r_data.spd_xi).index(int(spd_y_rng.split()[0]))
            _roc_xy, _roc_auc = dcopy(r_data.spd_roc_xy), dcopy(r_data.spd_roc_auc)
            _ci_lo, _ci_hi = dcopy(r_data.spd_ci_lo), dcopy(r_data.spd_ci_hi)
            xi_bin = dcopy(0.5 * (r_data.spd_xi[:-1] + r_data.spd_xi[1:]))

        # determines the indices of the cell in the overall array
        B = np.zeros(len(xi_bin))
        t_type_base = list(r_data.spd_sf_rs.keys()) if r_data.is_equal_time else list(r_data.spd_sf.keys())
        for i_filt in range(n_filt):
            # determins the match condition with the currently calculated roc values
            tt = r_obj.rot_filt_tot[i_filt]['t_type'][0]
            i_match = t_type_base.index(tt)

            # sets up the comparison rotational object
            r_filt_k = cf.init_rotation_filter_data(False)
            r_filt_k['t_type'] = [t_type_base[i_match]]
            r_obj_k = RotationFilteredData(self.data, r_filt_k, 0, None, True, 'Whole Experiment', False)

            # finds the corresponding cell types between the overall and user-specified filters
            i_cell_b, _ = cf.det_cell_match_indices(r_obj_k, [0, i_filt], r_obj)
            i_cell = int(i_cell_b[i_cluster] - 1)

            # sets the roc plot coordinates and auc values
            roc_xy[i_filt] = _roc_xy[t_type_base[i_match]][i_cell, i_bin]
            roc_auc[i_filt] = _roc_auc[t_type_base[i_match]][i_cell, i_bin]
            roc_auc_full[i_filt] = _roc_auc[t_type_base[i_match]][i_cell, :]

            # determines which errorbars are significant
            ci_lo_tmp, ci_hi_tmp = _ci_lo[tt][i_cell, :, is_boot], _ci_hi[tt][i_cell, :, is_boot]
            is_sig[i_filt] = np.logical_or((roc_auc_full[i_filt] - ci_lo_tmp) > 0.5,
                                           (roc_auc_full[i_filt] + ci_hi_tmp) < 0.5)

            # # sets the lower/upper confidence intervals
            # ci_lo[i_filt], ci_hi[i_filt] = ci_lo_tmp, ci_hi_tmp

            # calculates the complimentary roc values if any are below 0.5
            ii = roc_auc_full[i_filt] < 0.5
            roc_auc_full[i_filt][ii] = 1 - roc_auc_full[i_filt][ii]

            # swaps the lower/upper confidence intervals
            ci_lo[i_filt], ci_hi[i_filt] = dcopy(B), dcopy(B)
            ci_lo[i_filt][ii], ci_lo[i_filt][np.logical_not(ii)] = ci_lo_tmp[ii], ci_hi_tmp[np.logical_not(ii)]
            ci_hi[i_filt][ii], ci_hi[i_filt][np.logical_not(ii)] = ci_hi_tmp[ii], ci_lo_tmp[np.logical_not(ii)]

            # sets up the error bars for the current filter
            y_err.append([_ci_lo[tt][i_cell, i_bin, is_boot], _ci_hi[tt][i_cell, i_bin, is_boot]])

        ################################
        ####    SUBPLOT CREATION    ####
        ################################

        # initialises the plot axes
        setup_plot_axes()

        # creates the roc curve/auc subplots
        self.create_roc_curves(self.plot_fig.ax[0], roc_xy, lg_str, plot_grid)
        self.create_single_auc_plot(self.plot_fig.ax[1], roc_auc, plot_grid, r_obj.lg_str, y_err=y_err)

        # sets the subplot titles
        self.plot_fig.ax[0].set_title('ROC Curves')
        self.plot_fig.ax[1].set_title('ROC AUC')

        # plots the full auROC curves for each filter type
        for i_filt in range(n_filt):
            # plots the auc values
            self.plot_fig.ax[2].plot(xi_bin, roc_auc_full[i_filt], 'o-', c=c[i_filt])

            # adds the error bars to the running
            e_col = ['r' if x else 'k' for x in is_sig[i_filt]]
            for i_bin in range(len(xi_bin)):
                y_err_nw = np.vstack((ci_lo[i_filt][i_bin], ci_lo[i_filt][i_bin]))
                self.plot_fig.ax[2].errorbar(xi_bin[i_bin], roc_auc_full[i_filt][i_bin], yerr=y_err_nw,
                                             capsize=100 / len(xi_bin), color=e_col[i_bin])

        self.plot_fig.ax[2].plot([-90 * use_vel, 90], [0.5, 0.5], 'k--', linewidth=2)

        cf.set_axis_limits(self.plot_fig.ax[2], [-80 * use_vel, 80], [0, 1])
        self.plot_fig.ax[2].grid(plot_grid)
        self.plot_fig.ax[2].set_ylabel('auROC')
        if use_vel:
            self.plot_fig.ax[2].plot([0, 1], [0., 1.], 'k--', linewidth=2)
            self.plot_fig.ax[2].set_xlabel('Velocity (deg/s)')
        else:
            self.plot_fig.ax[2].set_xlabel('Speed (deg/s)')

        # resizes the figure to include the super-title
        self.plot_fig.fig.set_tight_layout(False)
        self.plot_fig.fig.suptitle('Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0], r_obj.ch_id[0][0]),
                                   fontsize=16, fontweight='bold')
        self.plot_fig.fig.tight_layout(rect=[0, 0.01, 1, 0.955])

    def plot_velocity_roc_curves_whole(self, rot_filt, plot_exp_name, plot_all_expt, use_vel, lo_freq_lim,
                                        hi_freq_lim, exc_type, use_comp, plot_err, plot_grid, plot_scope):
        '''

        :param rot_filt:
        :param plot_exp_name:
        :param plot_all_expt:
        :param use_vel:
        :param plot_grid:
        :param plot_scope:
        :return:
        '''

        # initialises the rotation filter (if not set)
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        r_obj = RotationFilteredData(self.data, rot_filt, None, plot_exp_name, plot_all_expt, 'Whole Experiment', False)
        if not r_obj.is_ok:
            self.calc_ok = False
            return

        # check to see the upper/lower frequency limits are feasible
        if hi_freq_lim <= lo_freq_lim:
            # if not output an error to screen and exit the function
            e_str = 'Lower frequency limit cannot exceed upper frequency limit'
            cf.show_error(e_str, 'Frequency Limit Error')
            self.calc_ok = False
            return

        # plots the whole experiment kinematic roc curves
        freq_lim = [lo_freq_lim, hi_freq_lim]
        self.plot_kine_whole_roc(r_obj, freq_lim, exc_type, use_comp, plot_err, plot_grid, use_vel)

    def plot_velocity_roc_pos_neg(self, rot_filt, plot_exp_name, plot_all_expt, lo_freq_lim, hi_freq_lim,
                                   exc_type, use_comp, plot_err, plot_grid, plot_scope):
        '''

        :param rot_filt:
        :param plot_exp_name:
        :param plot_all_expt:
        :param use_vel:
        :param use_comp:
        :param mean_type:
        :param plot_err:
        :param plot_grid:
        :param plot_scope:
        :return:
        '''

        # initialises the rotation filter (if not set)
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        r_obj = RotationFilteredData(self.data, rot_filt, None, plot_exp_name, plot_all_expt, 'Whole Experiment', False)
        if not r_obj.is_ok:
            self.calc_ok = False
            return

        # check to see the upper/lower frequency limits are feasible
        if hi_freq_lim <= lo_freq_lim:
            # if not output an error to screen and exit the function
            e_str = 'Lower frequency limit cannot exceed upper frequency limit'
            cf.show_error(e_str, 'Frequency Limit Error')
            self.calc_ok = False
            return

        # plots the whole experiment kinematic roc curves
        freq_lim = [lo_freq_lim, hi_freq_lim]
        self.plot_kine_whole_roc(r_obj, freq_lim, exc_type, use_comp, plot_err, plot_grid)

    def plot_cond_grouping_scatter(self, rot_filt, plot_exp_name, plot_all_expt, plot_cond,
                                   use_resp_grp_type, show_sig_markers, plot_trend, plot_grid, plot_scope):
        '''

        :param rot_filt:
        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_cond:
        :param plot_grid:
        :param plot_scope:
        :return:
        '''

        # sets up the rotational filter (for the specified trial condition given in plot_cond)
        _rot_filt = dcopy(rot_filt)
        _rot_filt, e_str, et_str = cf.setup_trial_condition_filter(_rot_filt, plot_cond)
        if e_str is not None:
            cf.show_error(e_str, et_str)
            self.ok = False
            return

        # filters the rotational data and runs the analysis function
        r_obj = RotationFilteredData(self.data, _rot_filt, 0, plot_exp_name, plot_all_expt, 'Whole Experiment', False)
        if not r_obj.is_ok:
            self.calc_ok = False
            return

        # memory allocation and other initialisations
        x_trend = np.arange(0, 10, 0.02)
        n_filt, r_data = r_obj.n_filt, self.data.rotation
        A = np.empty(n_filt, dtype=object)

        #
        if use_resp_grp_type:
            grp_type, g_type = ['None', 'Rotation', 'Visual', 'Both'], r_data.ds_gtype
        else:
            st_type = ['Wilcoxon Paired Test', 'Delong', 'Bootstrapping'].index(r_data.phase_grp_stats_type)
            grp_type, g_type = ['MS/DS', 'MS/Not DS', 'Not MS'], r_data.phase_gtype[:, st_type]

        # determine the matching cell indices between the current and black filter
        i_cell_b, r_obj_tt = dcopy(A), dcopy(A)
        for i_filt in range(n_filt):
            # sets up a base filter with only the
            r_filt_base = cf.init_rotation_filter_data(False)
            r_filt_base['t_type'] = r_obj.rot_filt_tot[i_filt]['t_type']
            r_obj_tt[i_filt] = RotationFilteredData(self.data, r_filt_base, None, plot_exp_name,
                                            True, 'Whole Experiment', False)

            # finds the corresponding cell types between the overall and user-specified filters
            # r_obj_tt = r_data.r_obj_cond[r_obj.rot_filt_tot[i_filt]['t_type'][0]]
            i_cell_b[i_filt], _ = cf.det_cell_match_indices(r_obj_tt[i_filt], [0, i_filt], r_obj)

        #
        ind_black = np.where(['Black' in x['t_type'] for x in r_obj.rot_filt_tot])[0]
        ind_match = [cf.det_matching_filters(r_obj, i) for i in ind_black]
        m, m_size, c = ['o', 'x', '^', 's', 'D', 'H', '*'], 50, cf.get_plot_col(len(grp_type))
        sig_col = [np.array(x) / 255 for x in [0, _bright_red, _bright_cyan, _bright_purple]]

        ####################################
        ####    SCATTERPLOT CREATION    ####
        ####################################

        # legend properties initialisations
        h_plt, lg_str = [], ['Black Sig,', '{0} Sig.'.format(plot_cond), 'Both Sig.'] if show_sig_markers else []

        # initialises the plot axes
        self.init_plot_axes(n_plot=1)

        # initialises the plot axes region
        self.plot_fig.ax[0].plot([0, 1], [0, 1], 'k--', linewidth=2)
        self.plot_fig.ax[0].grid(plot_grid)

        #
        for i, im in enumerate(ind_match):
            # sets the black/comparison trial condition cell group type values (for the current match)
            if len(i_cell_b[im[1]]):
                # if there is a mis-match in cell count, then find the matching cells between conditions
                if len(i_cell_b[im[0]]) != len(i_cell_b[im[1]]):
                    i_cell_b[im[0]], _ = cf.det_cell_match_indices(r_obj_tt[im[0]], [0, 0], r_obj_tt[im[1]])

                # retrieves the auc values and cell grouping indices for the current filter combination
                g_type_m = g_type[i_cell_b[im[0]]]
                x_auc = r_data.cond_roc_auc['Black'][i_cell_b[im[0]], 2]
                y_auc = r_data.cond_roc_auc[plot_cond][i_cell_b[im[1]], 2]

                # removes any cells where the group type was not calculated
                is_ok = g_type_m >= 0
                if not np.all(is_ok):
                    g_type_m, x_auc, y_auc = g_type_m[is_ok], x_auc[is_ok], y_auc[is_ok]
                    i_cell_b[im[0]], i_cell_b[im[1]] = i_cell_b[im[0]][is_ok], i_cell_b[im[1]][is_ok]

                # calculates the compliment of any auc values < 0.5
                ix_c, iy_c = x_auc < 0.5, y_auc < 0.5
                x_auc[ix_c], y_auc[iy_c] = 1 - x_auc[ix_c], 1 - y_auc[iy_c]

                if show_sig_markers:
                    # sets the x/y significance points
                    x_sig = r_data.phase_auc_sig[i_cell_b[im[0]], 2]
                    y_sig = r_data.cond_auc_sig[plot_cond][i_cell_b[im[1]], 2]
                    xy_sig = x_sig + 2 * y_sig
                    sig_col_plt = np.array([sig_col[x] if x > 0 else None for x in xy_sig])

                    # creates the significance markers
                    jj = xy_sig > 0
                    self.plot_fig.ax[0].scatter(x_auc[jj], y_auc[jj], c=sig_col_plt[jj], marker=m[i], s=3*m_size)

                    # creates the significance legend plot markers
                    if i == 0:
                        for j in range(1,4):
                            h_plt.append(self.plot_fig.ax[0].scatter(-1, -1, c=to_rgba_array(sig_col[j]), marker=m[i]))

                # creates the markers for each of the phases
                for igt, gt in enumerate(grp_type):
                    ii = g_type_m == igt
                    if np.any(ii):
                        self.plot_fig.ax[0].scatter(x_auc[ii], y_auc[ii], c=c[igt], marker=m[i], s=m_size, alpha=1)

                        # h_plt[i, igt] = self.plot_fig.ax[0].plot(-1, -1, c=c[igt], marker=m[igt])
                        h_plt.append(self.plot_fig.ax[0].scatter(-1, -1, c=c[igt], marker=m[i]))
                        if len(ind_match) > 1:
                            lg_str.append('{0} ({1})'.format(gt, r_obj.lg_str[im[0]].replace('Black\n', '')))
                        else:
                            lg_str.append(gt)

                        # adds the trend-line (if selected)
                        if plot_trend:
                            auc_trend, _ = curve_fit(cf.lin_func, x_auc[ii] - 0.5, y_auc[ii] - 0.5)
                            self.plot_fig.ax[0].plot(x_trend + 0.5, auc_trend * x_trend + 0.5, '-{0}'.format(m[i]),
                                                     color=c[igt], markersize=8, linewidth=2, linestyle='dashed')

        # updates the axis limits
        cf.set_axis_limits(self.plot_fig.ax[0], [0.5, 1], [0.5, 1])
        self.plot_fig.ax[0].set_xlabel('Black auROC Scores')
        self.plot_fig.ax[0].set_ylabel('{0} auROC Scores'.format(plot_cond))
        self.plot_fig.ax[0].legend(h_plt, lg_str, loc=0, ncol=1+len(ind_match))

    def plot_roc_cond_comparison(self, rot_filt, plot_exp_name, plot_all_expt, plot_cond, plot_grid, plot_scope):
        '''

        :param rot_filt:
        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_cond:
        :param plot_grid:
        :param plot_scope:
        :return:
        '''

        def setup_plot_axes():
            '''

            :return:
            '''

            # sets up the axes dimensions
            top, bottom, pH, wspace = 0.96, 0.06, 0.01, 0.1

            # creates the gridspec object
            gs = gridspec.GridSpec(2, 3, width_ratios=[1/3] * 3, height_ratios=[1/2] * 2, figure=self.plot_fig.fig,
                                   wspace=wspace, left=0.03, right=0.98, bottom=bottom, top=top, hspace=0.1)

            # creates the subplots
            self.plot_fig.ax = np.empty(3, dtype=object)
            self.plot_fig.ax[0] = self.plot_fig.figure.add_subplot(gs[:, :2])
            self.plot_fig.ax[1] = self.plot_fig.figure.add_subplot(gs[0, -1])
            self.plot_fig.ax[2] = self.plot_fig.figure.add_subplot(gs[1, -1])

        # sets up the rotational filter (for the specified trial condition given in plot_cond)
        rot_filt, e_str, et_str = cf.setup_trial_condition_filter(rot_filt, plot_cond)
        if e_str is not None:
            cf.show_error(e_str, et_str)
            self.ok = False
            return

        # filters the rotational data and runs the analysis function
        r_obj = RotationFilteredData(self.data, rot_filt, None, plot_exp_name, plot_all_expt, 'Whole Experiment', False)
        if not r_obj.is_ok:
            self.calc_ok = False
            return

        # memory allocation and other initialisations
        n_filt, r_data = r_obj.n_filt, self.data.rotation
        A = np.empty(n_filt, dtype=object)

        #
        grp_type = ['MS/DS', 'MS/Not DS', 'Not MS']
        st_type = ['Wilcoxon Paired Test', 'Delong', 'Bootstrapping'].index(r_data.phase_grp_stats_type)
        b_gtype, c_gtype, n_grp = r_data.phase_gtype[:, st_type], r_data.cond_gtype[plot_cond][:, st_type], len(grp_type)


        # determine the matching cell indices between the current and black filter
        i_cell_b, r_obj_tt = dcopy(A), dcopy(A)
        for i_filt in range(n_filt):
            # sets up a base filter with only the
            r_filt_base = cf.init_rotation_filter_data(False)
            r_filt_base['t_type'] = r_obj.rot_filt_tot[i_filt]['t_type']
            r_obj_tt[i_filt] = RotationFilteredData(self.data, r_filt_base, None, plot_exp_name,
                                            True, 'Whole Experiment', False)

            # finds the corresponding cell types between the overall and user-specified filters
            i_cell_b[i_filt], _ = cf.det_cell_match_indices(r_obj_tt[i_filt], [0, i_filt], r_obj)

        # determines the black and matching phases
        ind_black = np.where(['Black' in x['t_type'] for x in r_obj.rot_filt_tot])[0]
        ind_match = [cf.det_matching_filters(r_obj, i) for i in ind_black]

        # memory allocation
        N = [np.zeros((2, 3), dtype=int) for _ in range(len(ind_match))]
        Pr = np.zeros((len(ind_match), 3), dtype=float)

        # for each match type, calculate the N-values for each group
        for i, im in enumerate(ind_match):
            # sets the black/condition cell group types
            b_gtype_grp, c_gtype_grp = b_gtype[i_cell_b[im[0]]], c_gtype[i_cell_b[im[1]]]

            # calculates the cell group types for each type
            for i_grp in range(n_grp):
                N[i][0, i_grp], N[i][1, i_grp] = sum(b_gtype_grp == i_grp), sum(c_gtype_grp == i_grp)

            # calculates the relative proportion
            Pr[i, :] = 100 * N[i][1, :] / N[i][0, :]

        # m, m_size, c = ['o', 'x', '^', 's', 'D', 'H', '*'], 50, cf.get_plot_col(len(grp_type))
        # sig_col = [np.array(x) / 255 for x in [0, _dark_red, _dark_purple, _black]]

        ####################################
        ####    SCATTERPLOT CREATION    ####
        ####################################

        # initialises the plot axes
        setup_plot_axes()

        #
        self.plot_fig.ax[0].plot([1, 10], [1, 10])
        self.plot_fig.ax[1].plot([1, 10], [1, 10])
        self.plot_fig.ax[2].plot([1, 10], [1, 10])

        # FINISH ME!
        a = 1

    def setup_pooled_neuron(r_obj, i_filt, pref_cw_dir, mtrial_type='Min Match', i_pool=None, p_sz=None):
        '''

        :param r_obj:
        :param pref_cw_dir:
        :param p_sz:
        :return:
        '''

        # memory allocation and initialisations
        n_cell, n_trial, _ = np.shape(r_obj.t_spike[i_filt])
        n_spike_pool = np.zeros((n_trial, 2), dtype=int)

        # create a random sample from all the cells of size p_sz (if pooling indices not provided)
        if i_pool is None:
            i_pool = np.sort(sample(range(n_cell), p_sz))

        # index array of the cells to be pooled
        t_spike = r_obj.t_spike[i_filt][i_pool, :, :]
        valid_trial = np.ones(n_trial, dtype=bool)

        # retrieves the spike times ensuring they are in the preferred direction
        for ip_sz in range(len(i_pool)):
            # determines the valid (non-None) trials
            if mtrial_type == 'Min Match':
                # if min match then only consider the
                valid_trial = np.logical_and(valid_trial, np.array([x is not None for x in t_spike[ip_sz, :, 0]]))
            else:
                valid_trial = np.array([x is not None for x in t_spike[ip_sz, :, 0]])

            # calculates the preferred/non-preferred
            if pref_cw_dir[i_pool[ip_sz]]:
                # case is cw is preferred
                n_spike_pref = cf.spike_count_fcn(t_spike[ip_sz, valid_trial, 1])
                n_spike_non_pref = cf.spike_count_fcn(t_spike[ip_sz, valid_trial, 2])
            else:
                # case is ccw is preferred
                n_spike_pref = cf.spike_count_fcn(t_spike[ip_sz, valid_trial, 2])
                n_spike_non_pref = cf.spike_count_fcn(t_spike[ip_sz, valid_trial, 1])

            # adds the preferred/non-preferred direction spike counts to the final arrays
            n_spike_pool[valid_trial, 0] += n_spike_non_pref
            n_spike_pool[valid_trial, 1] += n_spike_pref

            # adds in any missing trial values (if selected)
            if np.any(~valid_trial) and (mtrial_type == 'Add Missing'):
                a = 1

        # removes any missing trials (if using minimum trial matches)
        if np.any(~valid_trial) and (mtrial_type == 'Min Match'):
            n_spike_pool = n_spike_pool[valid_trial, :]

        # returns the pooled neuron spike counts
        return n_spike_pool

    #############################################
    ####    COMMON ROC ANALYSIS FUNCTIONS    ####
    #############################################

    def create_roc_curves(self, ax, roc_xy, lg_str, plot_grid, show_ax_lbl=True):
        '''

        :param ax:
        :param roc_xy:
        :return:
        '''

        # initialisations
        n_filt, h_plt = len(roc_xy), []
        c = cf.get_plot_col(n_filt)

        # plots the roc curves
        for i_filt in range(n_filt):
            if len(np.shape(roc_xy[i_filt])) == 1:
                for i_cell in range(len(roc_xy[i_filt])):
                    ax.plot(roc_xy[i_filt][i_cell][:, 0], roc_xy[i_filt][i_cell][:, 1], 'o-', c=c[i_filt])
            else:
                ax.plot(roc_xy[i_filt][:, 0], roc_xy[i_filt][:, 1], 'o-', c=c[i_filt])

            if n_filt > 1:
                # appends to the legend (if more than one filter)
                h_plt.append(ax.plot([2, 3], [2, 3], 'o-', c=c[i_filt]))

        # creates the legend (if more than one filter)
        if len(h_plt):
            # create the legend
            ax.legend([x[0] for x in h_plt], lg_str, loc=0)

        # sets the other axis properties
        cf.set_axis_limits(ax, [0, 1], [0, 1])

        # sets the other axes properties
        ax.plot([0, 1], [0, 1], 'k--')
        ax.grid(plot_grid)

        # sets the axis labels (if required)
        if show_ax_lbl:
            ax.set_xlabel('p(Non-Pref > Crit)')
            ax.set_ylabel('p(Pref > Crit)')

    def create_single_auc_plot(self, ax, roc_auc, plot_grid, lg_str, y_err=None):
        '''

        :param roc_auc:
        :param y_err:
        :return:
        '''

        # sets the x-indices and title string
        t_str, n_filt = 'ROC Integrals', len(roc_auc)
        c, xi = cf.get_plot_col(n_filt), np.array(range(n_filt)) + 1

        if y_err is not None:
            cc = [ebar_col((x - y[0]) > 0.5) if x > 0.5 else ebar_col((x + y[1]) < 0.5) for x, y in
                  zip(roc_auc, y_err)]

        # plots the roc curves and integrals
        for i_filt in range(n_filt):
            # creates the plot marker
            x, y = xi[i_filt], roc_auc[i_filt]
            ax.plot(x, y, 'o', c=c[i_filt])

            # creates the errorbars
            if y_err is not None:
                y_err_nw = np.array(y_err[i_filt]).reshape(-1, 1)
                ax.errorbar(x, y, yerr=y_err_nw, capsize=80 / n_filt, color=cc[i_filt])

        # resets the axis limits
        cf.set_axis_limits(ax, [0.5, n_filt + 0.5], [0, 1])

        # sets the axis properties
        ax.plot([0, n_filt + 1], [0.5, 0.5], 'k--')
        ax.set_xticks(xi)
        ax.set_xticklabels(lg_str)
        ax.set_xlabel('Filter Groupings')
        ax.set_ylabel('auROC')
        ax.grid(plot_grid)

    def create_multi_auc_plot(self, ax, roc_auc, plot_grid, connect_lines, lg_str):
        '''

        :param ax:
        :param roc_auc:
        :param plot_grid:
        :param lg_str:
        :return:
        '''

        # sets the x-indices and title string
        n_filt = len(roc_auc)
        xi = np.array(range(n_filt)) + 1

        # creates the bubble plot and the decision line
        if connect_lines:
            cf.create_connected_line_plot(ax, roc_auc)
        else:
            cf.create_bubble_boxplot(ax, roc_auc)

        # resets the axis limits
        ax.plot([-1, xi[-1]+1], [0.5, 0.5], 'k--')
        cf.set_axis_limits(ax, [xi[0] - 0.5, xi[-1] + 0.5], [0, 1])

        # sets the other axis properties
        ax.set_xticks(xi)
        ax.set_xticklabels(lg_str)
        ax.set_xlabel('Filter Groupings')
        ax.grid(plot_grid)

    def plot_kine_whole_roc(self, r_obj, freq_lim, exc_type, use_comp, plot_err, plot_grid, use_vel=True):
        '''

        :param r_obj:
        :param use_comp:
        :param mean_type:
        :param plot_err:
        :param plot_grid:
        :param use_vel:
        :return:
        '''

        def calc_auc_kwstats(roc_auc):
            '''

            :param roc_auc_mn:
            :return:
            '''

            def calc_stats_pvalue(y_grp, i_grp):
                '''

                :param y_grp:
                :param i_grp:
                :return:
                '''

                # calculates the kruskal-wallis
                # try:
                #     ph_stats = r_pHOC.kwAllPairsDunnTest(FloatVector(y_grp), FloatVector(i_grp),
                #                                          p_adjust_method="bonferroni")
                #     return ph_stats[ph_stats.names.index('p.value')][0]
                # except:
                kw_stats = r_stats.kruskal_test(FloatVector(y_grp), FloatVector(i_grp))
                return kw_stats[kw_stats.names.index('p.value')][0]

            # initialisations
            roc_auc_b = dcopy(roc_auc[0])
            n_comp, n_bin = len(roc_auc) - 1, np.size(roc_auc[0], axis=1)
            p_stats = np.empty((n_comp, n_bin), dtype=object)

            #
            for i_comp in range(n_comp):
                #
                roc_auc_c = dcopy(roc_auc[i_comp + 1])
                i_grp = [0] * np.size(roc_auc_b, axis=0) + [1] * np.size(roc_auc_c, axis=0)


                for i_bin in range(n_bin):
                    # sets the base/comparison values into a single array
                    y_grp = np.hstack((dcopy(roc_auc_b[:, i_bin]), dcopy(roc_auc_c[:, i_bin])))
                    if cfcn.arr_range(y_grp) < 1e-6:
                        # hack => if the range of y-values is small then kw test gives weird results? This fixes it...
                        p_value = np.nan
                    else:
                        # otherwise, calculate the test as per normal
                        p_value = calc_stats_pvalue(dcopy(y_grp), dcopy(i_grp))

                    if np.isnan(p_value):
                        p_stats[i_comp, i_bin] = 'NaN'
                    else:
                        p_stats[i_comp, i_bin] = '{:5.3f}{}'.format(p_value, sig_str_fcn(p_value, 0.05))

            # returns the stats array
            return p_stats

        # initialisations and other array indexing
        n_filt, r_data = r_obj.n_filt, self.data.rotation
        c, h_plt = cf.get_plot_col(n_filt), []

        # memory allocation
        A = np.empty(n_filt, dtype=object)
        roc_auc, roc_auc_mn, roc_auc_sem = dcopy(A), dcopy(A), dcopy(A)

        # sets the comparison bin for the velocity/speed arrays
        if use_vel:
            _roc_auc = dcopy(r_data.vel_roc_auc)
            xi_bin = dcopy(0.5 * (r_data.vel_xi[:-1] + r_data.vel_xi[1:]))
            if r_data.pn_comp:
                xi_bin = xi_bin[int(len(xi_bin) / 2):]
        else:
            _roc_auc = dcopy(r_data.spd_roc_auc)
            xi_bin = dcopy(0.5 * (r_data.spd_xi[:-1] + r_data.spd_xi[1:]))

        # determines the indices of the cell in the overall array
        t_type_base = list(r_data.vel_sf_rs.keys()) if r_data.is_equal_time else list(r_data.vel_sf.keys())
        for i_filt in range(n_filt):
            # determines the match condition with the currently calculated roc values
            tt = r_obj.rot_filt_tot[i_filt]['t_type'][0]
            i_match = t_type_base.index(tt)

            # sets up the comparison rotational object
            r_filt_k = cf.init_rotation_filter_data(False)
            r_filt_k['t_type'] = [t_type_base[i_match]]
            r_obj_k = RotationFilteredData(self.data, r_filt_k, 0, None, True, 'Whole Experiment', False)

            # finds the corresponding cell types between the overall and user-specified filters
            i_cell_b, _ = cf.det_cell_match_indices(r_obj_k, [0, i_filt], r_obj)

            # if using the frequency limit, then remove all cells with a low firing rate
            if exc_type != 'Use All Cells':
                # retrieves the spiking frequency arrays based on the calculation/plotting type
                if use_vel:
                    # case is velocity is being used for plotting
                    if r_data.is_equal_time:
                        # case is equal spacing was used for the calculation type
                        sf = r_data.vel_sf_rs[tt][:, :, i_cell_b]
                    else:
                        sf = r_data.vel_sf[tt][:, :, i_cell_b]
                else:
                    # case is speed is being used for plotting
                    if r_data.is_equal_time:
                        # case is equal spacing was used for the calculation type
                        sf = r_data.spd_sf_rs[tt][:, :, i_cell_b]
                    else:
                        sf = r_data.spd_sf[tt][:, :, i_cell_b]

                    # exc_type = ['Use All Cells', 'Low Firing Cells', 'High Firing Cells', 'Band Pass']

                # removes the cells which have the mean firing rate less than the limit
                sf_mn = np.array([np.mean(np.mean(sf[:, :, ic], axis=0)) for ic in range(np.size(sf, axis=2))])
                if exc_type == 'Low Firing Cells':
                    i_cell_b = i_cell_b[sf_mn > freq_lim[0]]
                elif exc_type == 'High Firing Cells':
                    i_cell_b = i_cell_b[sf_mn < freq_lim[1]]
                else:
                    i_cell_b = i_cell_b[np.logical_and(sf_mn > freq_lim[0], sf_mn < freq_lim[1])]

            # sets the roc auc values
            roc_auc[i_filt] = dcopy(_roc_auc[t_type_base[i_match]][i_cell_b, :])

            # if enforcing complimentary values, then ensure all values are above 0.5
            if use_comp:
                i_comp = roc_auc[i_filt] < 0.5
                roc_auc[i_filt][i_comp] = 1. - roc_auc[i_filt][i_comp]

            # sets the full auc binned values (calculates the complimentary values if < 0.5)
            # if mean_type == 'Mean':
            roc_auc_mn[i_filt] = np.mean(roc_auc[i_filt], axis=0)

            # elif mean_type == 'Median':
            #     roc_auc_mn[i_filt] = np.median(roc_auc[i_filt], axis=0)

            # calculates the standard error mean
            roc_auc_sem[i_filt] = np.std(roc_auc[i_filt], axis=0) / (np.sqrt(len(i_cell_b)) - 1)

        ################################
        ####    SUBPLOT CREATION    ####
        ################################

        # initialises the plot axes
        self.init_plot_axes()

        # plots the auROC curves for each filter type
        for i_filt in range(n_filt):
            # plots the mean auc values
            h_plt.append(self.plot_fig.ax[0].plot(xi_bin, roc_auc_mn[i_filt], 'o-', c=c[i_filt]))

            # creates the errorbars (if required)
            if plot_err:
                for i_bin in range(len(xi_bin)):
                    y_err_nw = np.vstack((roc_auc_sem[i_filt][i_bin], roc_auc_sem[i_filt][i_bin]))
                    self.plot_fig.ax[0].errorbar(xi_bin[i_bin], roc_auc_mn[i_filt][i_bin], yerr=y_err_nw,
                                                 capsize=100 / len(xi_bin), color=c[i_filt])

        # sets the axis legend and other axis properties
        lg_str = ['#{0} - {1}'.format(i + 1, x) for i, x in enumerate(r_obj.lg_str)]
        self.plot_fig.ax[0].legend([x[0] for x in h_plt], lg_str, loc=0)
        self.plot_fig.ax[0].set_ylabel('auROC')
        self.plot_fig.ax[0].grid(plot_grid)

        # sets the x-axis label (based on type)
        if use_vel:
            self.plot_fig.ax[0].set_xlabel('Velocity (deg/s)')
        else:
            self.plot_fig.ax[0].set_xlabel('Speed (deg/s)')

        self.plot_fig.fig.set_tight_layout(False)
        self.plot_fig.fig.tight_layout(rect=[0.01, 0.02, 0.98, 0.97])

        if cf.is_linux:
            p_wid = 1.25 - 0.10 * r_data.pn_comp
        else:
            p_wid = 1.15 - 0.10 * r_data.pn_comp

        # displays the pairwise comparison stats (if more than one filter)
        if n_filt > 1:
            # calculates the stats
            p_stats = calc_auc_kwstats(roc_auc)

            # sets up the n-value table
            if use_vel:
                xi_rng = dcopy(r_data.vel_xi)
            else:
                xi_rng = dcopy(r_data.spd_xi)

            ax = self.plot_fig.ax[0]
            row_hdr = ['#{0} vs #1'.format(x + 2) for x in range(np.size(p_stats, axis=0))]
            if r_data.pn_comp:
                col_hdr = ['{0}/-{0}'.format(int(x)) for x in xi_rng[int(len(xi_rng) / 2 + 1):]]
            else:
                col_hdr = ['{0}/{1}'.format(int(xi_rng[i]), int(xi_rng[i + 1])) for i in range(len(xi_rng) - 1)]

            if use_vel and (not r_data.pn_comp):
                n_mid = int(len(xi_rng) / 2)
                col_hdr_1, p_stats_1 = col_hdr[:n_mid], p_stats[:, :n_mid]
                col_hdr_2, p_stats_2 = col_hdr[n_mid:], p_stats[:, n_mid:]
                c_col = [np.array(_light_gray) / 255] * len(col_hdr_1)

                # sets up the positive velocity range
                t_props_1 = cf.add_plot_table(self.plot_fig, ax, table_font_small, p_stats_1, row_hdr, col_hdr_1,
                                              c[1:], c_col, 'bottom', n_row=1, n_col=1, p_wid=p_wid)
                t_props_2 = cf.add_plot_table(self.plot_fig, ax, table_font_small, p_stats_2, row_hdr, col_hdr_2,
                                              c[1:], c_col, 'bottom', n_row=1, n_col=1, p_wid=p_wid)

                # resizes the plot axes position (to account for the second table)
                ax_p, fig_hght = ax.get_position(), self.plot_fig.height()
                d_hght = t_props_1[2] * t_props_1[0]._bbox[3] * (1 + .5 / (np.size(p_stats_1, axis=0) + 1))
                ax_pos_nw = [ax_p.x0, ax_p.y0 + d_hght / fig_hght, ax_p.width, ax_p.height - d_hght / fig_hght]
                ax.set_position(ax_pos_nw)

                # resets the bottom location of the upper table
                c_hght = t_props_1[0]._bbox[3] / (np.size(p_stats_1, axis=0) + 1)
                t_props_1[0]._bbox[1] = t_props_1[0]._bbox[1] * ax_p.height / ax_pos_nw[3]
                t_props_2[0]._bbox[1] = t_props_1[0]._bbox[1] - (t_props_1[0]._bbox[3] + c_hght)

            else:
                c_col = [np.array(_light_gray) / 255] * len(col_hdr)
                cf.add_plot_table(self.plot_fig, ax, table_font_small, p_stats, row_hdr, col_hdr, c[1:],
                                  c_col, 'bottom', n_row=1, n_col=1, p_wid=p_wid)

    ###########################################
    ####    COMBINED ANALYSIS FUNCTIONS    ####
    ###########################################

    def plot_combined_stimuli_stats(self, rot_filt, plot_exp_name, plot_all_expt, plot_grid, plot_scope, plot_type):
        '''

        :return:
        '''

        # initialisations and memory allocation
        n_grp, r_data = [4, 5], self.data.rotation
        stat_type, c = 'Combined Stimuli Response', [cf.get_plot_col(x) for x in n_grp]

        #
        if plot_type == 'Direction Selectivity':
            plt_vals, i_grp = dcopy(r_data.ds_gtype_pr), 0
            lg_str = ['None', 'Rotation', 'Visual', 'Both']
        else:
            plt_vals, i_grp = dcopy(r_data.pd_type_pr), 1
            lg_str = ['None', 'Rotation', 'Visual', 'Incongruent', 'Congruent']

        # creates the plot outlay and titles
        self.init_plot_axes(n_row=1, n_col=2)
        x_ticklbl = ['#{0} - {1}'.format(i+1, x) for i, x in enumerate(dcopy(r_data.r_obj_rot_ds.lg_str))]

        # creates the bar graph
        h_bar = cf.create_stacked_bar(self.plot_fig.ax[0], plt_vals, c[i_grp])
        self.plot_fig.ax[0].set_xticklabels(x_ticklbl)
        self.plot_fig.ax[0].grid(plot_grid)

        # updates the y-axis limits/labels and creates the legend
        self.plot_fig.ax[0].set_ylim([0, 100])
        self.plot_fig.ax[0].set_ylabel('Population %')
        cf.reset_axes_dim(self.plot_fig.ax[0], 'bottom', 0.0375, True)

        # creates the bar graph
        self.plot_fig.ax[0].legend([x[0] for x in h_bar], lg_str, ncol=n_grp[i_grp], loc='upper center',
                                   columnspacing=0.125, bbox_to_anchor=(0.5, 1.05))

        # calculates the number of direction sensitive/insensitive cells (over all conditions)
        self.plot_fig.ax[1].axis('off')
        self.plot_fig.ax[1].axis([0, 1, 0, 1])

        # determines
        n_DS0 = np.vstack([r_data.ds_gtype_N] * 4) * r_data.ds_gtype_pr / 100
        n_DS = np.vstack((n_DS0[0, :], np.sum(n_DS0[1:, ], axis=0)))

        #
        n_PD0 = np.vstack(r_data.pd_type_N)
        n_PD = np.vstack((n_PD0, np.sum(n_PD0, axis=0)))

        #
        self.create_spike_freq_stats_table(self.plot_fig.ax[1], n_DS, r_data.r_obj_rot_ds.n_filt,
                                           stat_type, n_row=1, n_col=2, n_PD=n_PD)

    ####################################################
    ####    SINGLE EXPERIMENT ANALYSIS FUNCTIONS    ####
    ####################################################

    def plot_signal_means(self, i_cluster, plot_all, exp_name, plot_grid=True):
        '''

        :param i_cluster:
        :param plot_all:
        :param exp_name:
        :return:
        '''

        # retrieves the data dictionary corresponding to the selected experiment
        i_expt = cf.get_expt_index(exp_name, self.data.cluster)
        data_plot = self.data.cluster[i_expt]
        i_cluster, e_str = self.check_cluster_index_input(i_cluster, plot_all, data_plot['nC'])

        # resets the cluster index if plotting all clusters
        if e_str is not None:
            cf.show_error(e_str, 'Infeasible Cluster Indices')
            self.calc_ok = False
            return

        # sets the indices of the clusters to plot and creates the figure/axis objects
        i_cluster = self.set_cluster_indices(i_cluster, n_max=data_plot['nC'])
        T = self.setup_time_vector(data_plot['sFreq'], n_pts=np.size(data_plot['vMu'], axis=0))

        # sets up the figure/axis
        n_plot = len(i_cluster)
        self.init_plot_axes(n_plot=n_plot)

        # plots the values over all subplots
        for i_plot in range(n_plot):
            # sets the actual fixed/free plot indices
            j_plot = i_cluster[i_plot] - 1
            id_clust = data_plot['clustID'][j_plot]

            # plots the mean signal
            self.plot_fig.ax[i_plot].plot(T, data_plot['vMu'][:, j_plot], linewidth=3.0)

            # sets the plot values
            self.plot_fig.ax[i_plot].grid(plot_grid)
            self.plot_fig.ax[i_plot].set_xlim(T[0], T[-1])
            self.plot_fig.ax[i_plot].set_title('Cluster #{0}'.format(id_clust))
            self.plot_fig.ax[i_plot].set_xlabel('Time (ms)')
            self.plot_fig.ax[i_plot].set_ylabel('Voltage ({0}V)'.format(cf._mu))

    def plot_cluster_auto_ccgram(self, i_cluster, plot_all, window_size, exp_name):
        '''

        :param i_cluster:
        :param plot_all:
        :param c_type:
        :return:
        '''

        # initialisations
        ex_name, cluster = cf.extract_file_name, self.data.cluster

        # retrieves the data dictionary corresponding to the selected experiment
        i_expt = cf.get_expt_index(exp_name, self.data.cluster)
        data_plot = self.data.cluster[i_expt]

        # resets the cluster index if plotting all clusters
        i_cluster, e_str = self.check_cluster_index_input(i_cluster, plot_all, data_plot['nC'])
        if e_str is not None:
            cf.show_error(e_str, 'Infeasible Cluster Indices')
            self.calc_ok = False
            return

        # sets the indices of the clusters to plot and creates the figure/axis objects
        n_bin_tot = int(len(data_plot['ccGramXi']) / 2)
        ind = np.arange(n_bin_tot - window_size, n_bin_tot + window_size)

        # sets up the figure/axis
        n_plot = len(i_cluster)
        self.init_plot_axes(n_plot=n_plot)
        xi_hist = data_plot['ccGramXi'][ind]

        # sets the x-axis limits
        bin_sz = xi_hist[1] - xi_hist[0]
        x_lim = [xi_hist[0] - bin_sz / 2.0, xi_hist[-1] + bin_sz / 2.0]

        # plots the values over all subplots
        for i_plot in range(n_plot):
            # sets the actual fixed/free plot indices
            j_plot = i_cluster[i_plot] - 1
            id_clust = data_plot['clustID'][j_plot]

            # creates the bar-graph and the centre marker
            n_hist = data_plot['ccGram'][j_plot, j_plot, ind]
            self.plot_fig.ax[i_plot].bar(xi_hist, height=n_hist, width=1)
            self.plot_fig.ax[i_plot].plot(np.zeros(2), self.plot_fig.ax[i_plot].get_ylim(), 'r--')

            # sets the axis properties
            self.plot_fig.ax[i_plot].set_title('Cluster #{0}'.format(id_clust))
            self.plot_fig.ax[i_plot].set_ylabel('Frequency (Hz)')
            self.plot_fig.ax[i_plot].set_xlabel('Time Lag (ms)')
            self.plot_fig.ax[i_plot].set_xlim(x_lim)

    def plot_cluster_cross_ccgram(self, exp_name, i_ref, i_comp, plot_all, plot_type, window_size, p_lim, f_cutoff):
        '''

        :param i_cluster:
        :param plot_all:
        :param c_type:
        :return:
        '''

        # retrieves the data dictionary corresponding to the selected experiment
        i_expt = cf.get_expt_index(exp_name, self.data.cluster)
        data_plot = self.data.cluster[i_expt]

        # resets the cluster index if plotting all clusters
        i_ref, e_str1 = self.check_cluster_index_input(i_ref, False, data_plot['nC'])
        if e_str1 is not None:
            cf.show_error(e_str1, 'Infeasible Cluster Indices')
            self.calc_ok = False
            return
        elif len(i_ref) > 1:
            e_str = 'You have entered multiple indices for the reference cluster index.'
            cf.show_error(e_str, 'Infeasible Cluster Indices')
            self.calc_ok = False
            return
        else:
            i_comp, e_str2 = self.check_cluster_index_input(i_comp, plot_all, data_plot['nC'])
            if e_str2 is not None:
                cf.show_error(e_str2, 'Infeasible Cluster Indices')
                self.calc_ok = False
                return

        # sets the indices of the clusters to plot and creates the figure/axis objects
        ind = np.where(abs(data_plot['ccGramXi']) <= window_size)[0]

        # sets up the figure/axis
        t_min, t_max, pW_lo, pW_hi, p_lim = 1.5, 4.0, 0.85, 1.15, p_lim / 100.0
        n_plot, j_plot1 = len(i_comp), i_ref[0] - 1
        id_clust1 = data_plot['clustID'][j_plot1]
        xi_hist = data_plot['ccGramXi'][ind]
        self.init_plot_axes(n_plot=n_plot)

        # sets the x-axis limits
        bin_sz = xi_hist[1] - xi_hist[0]
        x_lim = [xi_hist[0], xi_hist[-1]]

        # sets up the gaussian signal filter
        f_scale_ref = len(data_plot['tSpike'][i_ref[0] - 1]) / 1000.0
        freq_range = np.arange(0, len(xi_hist)) * (1 / (xi_hist[1] - xi_hist[0]))
        freq = next((i for i in range(len(xi_hist)) if freq_range[i] > f_cutoff), len(xi_hist))

        # plots the values over all subplots
        for i_plot in range(n_plot):
            # sets the actual fixed/free plot indices
            search_lim = []
            j_plot2 = i_comp[i_plot] - 1
            id_clust2 = data_plot['clustID'][j_plot2]

            # calculates the confidence intervals
            n_hist = data_plot['ccGram'][j_plot1, j_plot2, ind]
            ciN_lo, ciN_hi, _ = cfcn.calc_ccgram_prob(data_plot['ccGram'][j_plot1, j_plot2, :] * f_scale_ref, freq, p_lim)

            # creates the search limit region
            y_lim = [pW_lo * min(np.min(n_hist), np.min(ciN_lo[ind] / f_scale_ref)),
                     pW_hi * max(np.max(n_hist), np.max(ciN_hi[ind] / f_scale_ref))]
            search_lim.append(Rectangle((-t_max, y_lim[0]), (t_max - t_min), y_lim[1]))
            search_lim.append(Rectangle((t_min, y_lim[0]), (t_max - t_min), y_lim[1]))
            pc = PatchCollection(search_lim, facecolor='r', alpha=0.25, edgecolor='k')
            self.plot_fig.ax[i_plot].add_collection(pc)

            if plot_type == 'bar':
                self.plot_fig.ax[i_plot].bar(xi_hist, height=n_hist, width=1)
            else:
                self.plot_fig.ax[i_plot].scatter(xi_hist, n_hist, marker='o', c='b')

            # plots the auto-correlogram and confidence interval limits
            self.plot_fig.ax[i_plot].plot(xi_hist, ciN_lo[ind] / f_scale_ref, 'k--')
            self.plot_fig.ax[i_plot].plot(xi_hist, ciN_hi[ind] / f_scale_ref, 'k--')

            # sets the zero time-lag marker
            self.plot_fig.ax[i_plot].plot(np.zeros(2), y_lim, 'k--')

            # sets the axis properties
            self.plot_fig.ax[i_plot].set_title('Cluster #{0} vs #{1}'.format(id_clust1, id_clust2))
            self.plot_fig.ax[i_plot].set_ylabel('Frequency (Hz)')
            self.plot_fig.ax[i_plot].set_xlabel('Time Lag (ms)')
            self.plot_fig.ax[i_plot].set_xlim(x_lim)
            self.plot_fig.ax[i_plot].set_ylim(y_lim)

    #########################################
    ####    COMMON ANALYSIS FUNCTIONS    ####
    #########################################

    def create_raster_hist(self, r_obj, n_bin, plot_grid, rmv_median=False):
        '''

        :param t_spike:
        :param clust_ind:
        :param f_perm:
        :param f_key:
        :param rot_filt_tot:
        :return:
        '''

        def setup_spike_time_hist(t_spike, n_trial, xi, is_single_cell):
            '''

            :param tSp:
            :param xi:
            :param dxi:
            :return:
            '''

            # memory allocation
            n_hist = np.empty(np.size(t_spike, axis=0), dtype=object)
            dxi = xi[1] - xi[0]

            # calculates the histograms for each of the trial over all cells
            for i_hist in range(len(n_hist)):
                ind_trial = np.array([(t_spike[i_hist, i] is not None) for i in range(n_trial)])
                n_hist[i_hist] = np.vstack([np.histogram(x, bins=xi)[0] / dxi for x in t_spike[i_hist, ind_trial]])

            # returns the array
            if is_single_cell:
                return n_hist[0]
            else:
                return np.vstack([np.mean(x, axis=0) for x in n_hist])

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        if not r_obj.is_ok:
            self.calc_ok = False
            return

        # initialisations
        yL_mn, yL_mx, n_hist_min = 0, -1e6, 1e6
        c = cf.get_plot_col(r_obj.n_filt)

        # sets the firing rate string
        if rmv_median:
            fr_str = 'Relative Mean Firing Rate (Hz)'
        else:
            fr_str = 'Mean Firing Rate (Hz)'

        # creates the plot outlay and titles
        self.init_plot_axes(n_row=2, n_col=r_obj.n_phase, n_plot=2 * r_obj.n_phase)

        # plots the histogram/rasterplot for each phase
        for i_phase in range(r_obj.n_phase):
            # memory allocation
            c_raster, t_raster, h_plot, y_tick, y_tick_lbl = [], [], [], [], []

            for i_filt in range(r_obj.n_filt):
                # calculates the histogram
                t_phase = r_obj.t_phase[i_filt][0]
                xi = np.linspace(0, t_phase, n_bin + 1)
                xi_hist = (xi[:-1] + xi[1:]) / 2.0

                # sets the histogram counts for each of the
                t_sp = r_obj.t_spike[i_filt][:, :, i_phase]
                n_hist = setup_spike_time_hist(t_sp, np.shape(t_sp)[1], xi, r_obj.is_single_cell)

                # appends the new spike times/colours to the overall rasterplot arrays
                n_trial = np.size(n_hist, axis=0)
                c_raster.append([c[i_filt]] * n_trial)
                if r_obj.is_single_cell:
                    t_raster.append([list(x) for x in cf.flat_list(t_sp)])
                else:
                    t_raster_new = [list(np.hstack(t_sp[i_row, np.array(range(x))])) for i_row, x
                                    in zip(range(n_trial), r_obj.n_trial[i_filt])]
                    t_raster.append(t_raster_new)

                # calculates the mean histogram count (across all trials/cells)
                n_hist_mn = np.mean(n_hist, axis=0)
                if rmv_median:
                    # if the baseline phase, then calculate the baseline median
                    if i_phase == 0:
                        bl_median = np.median(n_hist_mn)

                    # removes the baseline median
                    n_hist_mn -= bl_median

                # determines the overall minimum histogram value
                n_hist_min = min(n_hist_min, np.min(n_hist_mn))

                # creates the new plot object
                h_plot.append(self.plot_fig.ax[r_obj.n_phase + i_phase].step(xi_hist, n_hist_mn, color=c[i_filt]))
                if i_filt == 0:
                    self.plot_fig.ax[i_phase].set_xlim(0, t_phase)
                    self.plot_fig.ax[i_phase + r_obj.n_phase].set_xlim(0, t_phase)
                    i_ofs = n_trial
                else:
                    self.plot_fig.ax[i_phase].plot([0, t_phase], (i_ofs - 0.5) * np.ones(2), 'k-', linewidth=1)
                    i_ofs += n_trial

                # appends the tick-marks to the y-tick array
                y_tick.append([i_ofs - n_trial / 2])

            #
            self.plot_fig.ax[i_phase].eventplot(positions=cf.flat_list(t_raster), orientation='horizontal',
                                                colors=cf.flat_list(c_raster))
            self.plot_fig.ax[i_phase].set_ylim(-0.5, i_ofs - 0.5)
            self.plot_fig.ax[i_phase].set_title('{0}'.format(r_obj.phase_lbl[i_phase]))
            self.plot_fig.ax[r_obj.n_phase + i_phase].set_xlabel('Time (s)')

            if i_phase == 0:
                # sets the y-axis properties for the raster plot
                self.plot_fig.ax[i_phase].set_yticks(y_tick)
                self.plot_fig.ax[i_phase].get_yaxis().set_visible(True)
                self.plot_fig.ax[i_phase].set_yticklabels(r_obj.lg_str, rotation=90, va='bottom',
                                                          ha='center', rotation_mode='anchor')

                # sets the x/y-axis titles for the histogram subplot
                self.plot_fig.ax[i_phase].set_ylabel('Trial Type')
                self.plot_fig.ax[r_obj.n_phase + i_phase].set_ylabel(fr_str)
            else:
                self.plot_fig.ax[i_phase].get_yaxis().set_visible(False)

            # sets the grid properties
            self.plot_fig.ax[i_phase].grid(plot_grid)
            self.plot_fig.ax[r_obj.n_phase + i_phase].grid(plot_grid)

            # determines the rolling overall maximum y-axis value
            y_lim = self.plot_fig.ax[r_obj.n_phase + i_phase].get_ylim()
            yL_mx = max([y_lim[1], yL_mx])

            # if the first phase, then append the legend
            if i_phase == 0 and r_obj.f_perm is not None:
                n_col = min(3, len(r_obj.lg_str)) if (r_obj.n_phase == 3) else min(4, len(r_obj.lg_str))
                self.plot_fig.ax[r_obj.n_phase + i_phase].legend(
                    r_obj.lg_str, loc=2, ncol=n_col, handletextpad=0.5, mode='expand')

        # resets the overall heights of the histograms so that they are the same height
        n_hist_min = max(-0.05, n_hist_min - 0.05 * (yL_mx - n_hist_min))
        for i_phase in range(r_obj.n_phase):
            self.plot_fig.ax[r_obj.n_phase + i_phase].set_ylim(n_hist_min, yL_mx)

        # resets the figure layout (single cell only)
        if r_obj.is_single_cell:
            self.plot_fig.fig.set_tight_layout(False)
            self.plot_fig.fig.suptitle('Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0], r_obj.ch_id[0][0]),
                                       fontsize=16, fontweight='bold')
            self.plot_fig.fig.tight_layout(rect=[0, 0.01, 1, 0.955])

    def create_spike_freq_plot(self, r_obj, plot_grid, plot_trend, p_value, stats_type, is_3d=False, ind_type=None):
        '''

        :param r_obj:
        :param plot_grid:
        :param is_3d:
        :return:
        '''

        def set_stim_phase_str(r_obj, sp_f, sp_stats, plt_ind, lbl_ind=None, p_value=None):
            '''

            :param r_obj:
            :param sp_f:
            :param plt_ind:
            :return:
            '''

            #
            if lbl_ind is None:
                lbl_ind = plt_ind

            # initialisations
            plt_ind, sep_str = np.array(plt_ind), '--------------------------'
            sp_f = np.array(sp_f)[plt_ind]
            if r_obj.is_ud:
                p_lbl = np.array(['BL', 'Stim'])
            else:
                p_lbl = np.array(['BL', 'CW', 'CCW'])[np.array(lbl_ind)]

            #
            has_stats = sp_stats is not None
            n_trial, i_ofs = len(sp_f[0]), 2 + int(not r_obj.is_single_cell) * (
                1 + int(r_obj.plot_all_expt + 3 * has_stats))
            lbl_str = np.empty((n_trial, i_ofs + len(plt_ind)), dtype=object)

            #
            if r_obj.is_single_cell:
                #
                lbl_str[:, 0] = np.array(['Trial #{0}'.format(i + 1) for i in range(n_trial)])
                lbl_str[:, 1] = np.array([sep_str] * n_trial)
            else:
                #
                j_ofs = int(r_obj.plot_all_expt)
                k_ofs = j_ofs + 3

                #
                if r_obj.plot_all_expt:
                    exp_name = [cf.extract_file_name(x['expFile']) for x in self.data.cluster]
                    i_expt = cf.flat_list([list(x) for x in r_obj.i_expt])
                    lbl_str[:, 0] = np.array(['Expt = {0}'.format(exp_name[i_ex]) for i_ex in i_expt])

                #
                lbl_str[:, j_ofs] = np.array(
                    ['Cluster #{0}'.format(cl_id) for cl_id in cf.flat_list(r_obj.cl_id)])
                lbl_str[:, 1 + j_ofs] = np.array(
                    ['Channel #{0}'.format(ch_id) for ch_id in cf.flat_list(r_obj.ch_id)])
                lbl_str[:, 2 + j_ofs] = np.array([sep_str] * n_trial)

                #
                if sp_stats is not None:
                    sig_str = ['No', 'Yes']
                    lbl_str[:, k_ofs] = np.array(['P-Value = {:5.3f}'.format(x) for x in sp_stats])
                    lbl_str[:, k_ofs + 1] = np.array(
                        ['Significant = {0}'.format(sig_str[x < p_value]) for x in sp_stats])
                    lbl_str[:, k_ofs + 2] = np.array([sep_str] * n_trial)

            # appends the
            for i in range(len(plt_ind)):
                lbl_str[:, i_ofs + i] = np.array([p_lbl[i] + ' Freq = {:6.2f}'.format(x) for x in sp_f[i]])

            # sets the final combined label strings
            return ['\n'.join(list(lbl_str[i_row, :])) for i_row in range(np.size(lbl_str, axis=0))]

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        if not r_obj.is_ok:
            self.calc_ok = False
            return

        # initialisations
        p_alpha, m_sz = 0.8, 30
        n_filt = int(r_obj.n_filt / 2) if r_obj.is_ud else r_obj.n_filt
        c = cf.get_plot_col(n_filt)
        c2 = cf.get_plot_col(4, n_filt)
        n_sub = 1 + 2 * (not is_3d)

        # memory allocation
        h_plt, i_grp = [], np.empty(2, dtype=object)
        h, lbl = np.empty(n_sub, dtype=object), np.empty(n_sub, dtype=object)

        # calculates the individual trial/mean spiking rates and sets up the plot/stats arrays
        sp_f0, sp_f = cf.calc_phase_spike_freq(r_obj)
        s_plt, sf_trend, sf_stats, i_grp[0] = cf.setup_spike_freq_plot_arrays(r_obj, sp_f0, sp_f, ind_type, n_sub,
                                                                           plot_trend=plot_trend, is_3d=is_3d)

        #########################################
        ####    SCATTERPLOT SUBPLOT SETUP    ####
        #########################################

        # sets the scatter plot object colour array
        c_scatter = cf.flat_list([[c[i_filt]] * np.size(sp_f[i_filt], axis=0) for i_filt in range(n_filt)])

        # creates the plot outlay and titles
        n_row = 1 + int((not is_3d) and (not r_obj.is_single_cell))

        if r_obj.is_single_cell:
            if is_3d:
                self.init_plot_axes(n_row=n_row, n_col=n_sub, is_3d=is_3d)
            else:
                self.init_plot_axes(n_row=2, n_col=2, is_3d=is_3d)
                self.plot_fig.ax[3].axis('off')
        else:
            self.init_plot_axes(n_row=n_row, n_col=n_sub, is_3d=is_3d)

        # setups up the scatterplots for each subplot phase
        for i_sub in range(n_sub):
            # loop initialisations
            sp, i1, i2 = s_plt[i_sub], 0, 1
            if is_3d:
                # case is plotting data in 3 dimensions

                # creates the plot and row label strings
                h[0] = self.plot_fig.ax[0].scatter(sp[0], sp[1], sp[2], marker='o', c=c_scatter)
                lbl[0] = set_stim_phase_str(r_obj, s_plt[0], sf_stats[i_sub], [0, 1, 2])

                # creates the legend markers
                h_plt = [self.plot_fig.ax[0].scatter(-1, -1, -1, marker='o', c=cc) for cc in c]

                # sets the title/z-axis strings
                t_str = '{0} vs {1} vs {2}'.format(r_obj.phase_lbl[2], r_obj.phase_lbl[1], r_obj.phase_lbl[0])
                self.plot_fig.ax[0].set_title(t_str)
                self.plot_fig.ax[0].set_zlabel('{0} Frequency (Hz)'.format(r_obj.phase_lbl[2]))

                # resets the z-axis so the lower limit is 0
                cf.reset_plot_axes_limits(self.plot_fig.ax[i_sub], 0, 'z', False)
            else:
                # case is the default graph types

                # sets the scatterplot alpha values
                if not r_obj.is_single_cell:
                    # c_scatter = [to_rgba_array(x, max(0.1, 1 - y)) for x, y in zip(c_scatter, sf_stats[i_sub])]
                    c_scatter = [to_rgba_array(x, 1 - p_alpha * (y > p_value)) for x, y in
                                 zip(c_scatter, sf_stats[i_sub])]
                    c_scatter = np.squeeze(c_scatter)

                # creates the plot and row label strings
                i1, i2 = 1 * (i_sub > 1), 1 + (i_sub > 0)
                h[i_sub] = self.plot_fig.ax[i_sub].scatter(sp[0], sp[1], marker='o', c=c_scatter, s=m_sz)
                lbl[i_sub] = set_stim_phase_str(r_obj, sp, sf_stats[i_sub], [0, 1], [i1, i2], p_value=p_value)

                # creates the legend markers (first subplot only)
                if i_sub == 0:
                    h_plt = [self.plot_fig.ax[0].scatter(-1, -1, marker='o', c=cc) for cc in c]

            # sets the title string (non-3d plot only)
            if not is_3d:
                t_str = '{0} vs {1}'.format(r_obj.phase_lbl[i2], r_obj.phase_lbl[i1])
                self.plot_fig.ax[i_sub].set_title(t_str)

            # sets the x/y-label and the grid
            self.plot_fig.ax[i_sub].set_xlabel('{0} Frequency (Hz)'.format(r_obj.phase_lbl[i1]))
            self.plot_fig.ax[i_sub].set_ylabel('{0} Frequency (Hz)'.format(r_obj.phase_lbl[i2]))
            self.plot_fig.ax[i_sub].grid(plot_grid)

        # # creates the cursor objects for each subplot
        # for i_sub in range(n_sub):
        #     datacursor(h[i_sub], formatter=formatter, point_labels=lbl[i_sub], hover=True)

        # creates the legend (if more than one filter type)
        if n_filt > 1 or (not r_obj.is_single_cell):
            if r_obj.is_ud and r_obj.n_filt == 2:
                lg_str = ['All Cells']
            else:
                lg_str = ['({0}) - {1}'.format(i + 1, x) for i, x in enumerate(r_obj.lg_str)]

            self.plot_fig.ax[0].legend(h_plt, lg_str, loc=0)

        # resets the axis limits based on the plot type
        if is_3d:
            # alters the initial orientation
            self.plot_fig.ax[0].view_init(20, -45)

            # determines the overall x/y/z axis limit maximum
            xLnw = self.plot_fig.ax[0].get_xlim()
            yLnw = self.plot_fig.ax[0].get_ylim()
            zLnw = self.plot_fig.ax[0].get_zlim()
            axL = max(xLnw[1], yLnw[1], zLnw[1])
        else:
            # otherwise, determine the overall x/y axis limit maximum
            axL = -1e6
            for i_sub in range(n_sub):
                xLnw = self.plot_fig.ax[i_sub].get_xlim()
                yLnw = self.plot_fig.ax[i_sub].get_ylim()
                axL = max(xLnw[1], yLnw[1], axL)

                # adds the trend-line (if selected)
                if plot_trend:
                    x = np.array([0, 100 * axL])
                    for i_filt in range(n_filt):
                        y = sf_trend[i_sub][i_filt, 0] * x
                        self.plot_fig.ax[i_sub].plot(x, y, '--', c=c[i_filt])

        if r_obj.is_single_cell:
            #
            self.plot_fig.fig.set_tight_layout(False)
            self.plot_fig.fig.tight_layout(rect=[0.0, 0.01, 1, 0.955])
            self.plot_fig.fig.suptitle(
                'Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0], r_obj.ch_id[0][0]),
                fontsize=16, fontweight='bold')

            # self.plot_fig.ax[-1].plot([0,1],[0,1])
            # self.plot_fig.ax[-1].axis('on')
            # cf.set_axis_limits(self.plot_fig.ax[-1], [0, 1], [0, 1])

            # table initialisations
            # r_hght, table_gap, title_gap, t_wid, dy_ofs = 0.065, 0.075, 0.025, 0.8, 0.05
            # t_hght, t_str = r_hght * (r_obj.n_filt + 1), 'Condition P-Values'

            col_hdr = ['CW vs BL', 'CCW vs BL', 'CCW vs CW']
            t_data = np.vstack([['{:5.3f}{}'.format(y, sig_str_fcn(y, p_value)) for y in x] for x in sf_stats]).T

            # calculates the table dimensions
            cf.add_plot_table(self.plot_fig, len(self.plot_fig.ax)-1, table_font, t_data, r_obj.lg_str,
                              col_hdr, c, cf.get_plot_col(3, len(c)), None, n_row=2, pfig_sz=0.955)

        for i_sub in range(n_sub):
            # updates the the x/y axis limits
            self.plot_fig.ax[i_sub].set_xlim(0, axL)
            self.plot_fig.ax[i_sub].set_ylim(0, axL)
            if (is_3d):
                # if 3d, then update the z-axis limits
                self.plot_fig.ax[i_sub].set_zlim(0, axL)
            else:
                # otherwise, draw the unity line through the data
                self.plot_fig.ax[i_sub].plot([0, axL], [0, axL], 'k--')

        ###############################################
        ####    STATISTICS CALCULATIONS/DISPLAY    ####
        ###############################################

        if (not is_3d) and (not r_obj.is_single_cell):
            # memory allocation
            sf_type_pr = np.empty(n_sub - 1, dtype=object)
            sf_score = cf.calc_dirsel_scores(s_plt, sf_stats, p_value=p_value)
            score_min, score_sum = np.min(sf_score[:, :2], axis=1), np.sum(sf_score[:, :2], axis=1)

            # determines the reaction type from the score phase types
            #   0 = None
            #   1 = Inhibited
            #   2 = Excited
            #   3 = Mixed
            sf_type = np.max(sf_score[:, :2], axis=1) + (np.sum(sf_score[:, :2], axis=1) == 3).astype(int)
            sf_type_pr[0] = np.vstack([cf.calc_rel_prop(sf_type[x], 4) for x in i_grp[0]]).T

            # determines all motion sensitive cells (sf_type > 0)
            is_mot_sens = sf_type > 0

            # determines the direction selective cells, which must meet the following conditions:
            #  1) one direction only produces a significant result, OR
            #  2) both directions are significant AND the CW/CCW comparison is significant
            one_dir_sig = np.logical_and(score_min == 0, score_sum > 0)     # cells where one direction is significant
            both_dir_sig = np.min(sf_score[:, :2], axis=1) > 0              # cells where both CW/CCW is significant
            comb_dir_sig = sf_score[:, -1] > 0                              # cells where CW/CCW difference is significant

            # determines which cells are direction selective (removes non-motion sensitive cells)
            is_dir_sel = np.logical_or(one_dir_sig, np.logical_and(both_dir_sig, comb_dir_sig)).astype(int)
            i_grp[1] = [x[is_mot_sens[x]] for x in i_grp[0]]
            sf_type_pr[1] = np.vstack([cf.calc_rel_prop(is_dir_sel[x], 2) for x in i_grp[1]]).T

            for i in range(2):
                # creates the bar graph
                h_bar = cf.create_stacked_bar(self.plot_fig.ax[i + n_sub], dcopy(sf_type_pr[i]), c2)
                if r_obj.is_ud and r_obj.n_filt == 2:
                    self.plot_fig.ax[i + n_sub].set_xticklabels(['All Cells'])
                else:
                    self.plot_fig.ax[i + n_sub].set_xticklabels(r_obj.lg_str)
                self.plot_fig.ax[i + n_sub].grid(plot_grid)

                # sets the legend strings based on the type
                if i == 0:
                    lg_str = ['None', 'Inhibited', 'Excited', 'Mixed']
                else:
                    lg_str = ['Direction Insensitive', 'Direction Sensitive']

                # updates the y-axis limits/labels and creates the legend
                self.plot_fig.ax[i + n_sub].set_ylim([0, 100])
                self.plot_fig.ax[i + n_sub].set_ylabel('Population %')
                cf.reset_axes_dim(self.plot_fig.ax[i + n_sub], 'bottom', 0.075, True)
                self.plot_fig.ax[i + n_sub].legend([x[0] for x in h_bar], lg_str, ncol=len(lg_str),
                                                    loc='upper center', columnspacing=0.125, bbox_to_anchor=(0.5, 1.15))

            # calculates the number of direction sensitive/insensitive cells (over all conditions)
            self.plot_fig.ax[2 + n_sub].axis('off')
            self.plot_fig.ax[2 + n_sub].axis([0, 1, 0, 1])

            # creates the spiking frequency statstics table
            n_DS = self.setup_stats_nvalue_array(sf_type, sf_type_pr, i_grp, stats_type)
            self.create_spike_freq_stats_table(self.plot_fig.ax[2 + n_sub], n_DS, n_filt, stats_type, c, c2)

        # for ax in self.plot_fig.ax:
        #     self.remove_scatterplot_spines(ax)

    def create_spike_heatmap(self, r_obj, dt):
        '''

        :param plot_grid:
        :param dt:
        :return:
        '''

        def init_heatmap_plot_axes(r_obj):
            '''

            :return:
            '''

            # sets up the axes dimensions
            c_ofs = int(not r_obj.is_single_cell)
            top, bottom, pH = 0.96 - 0.04 * r_obj.is_single_cell, 0.06, 0.01
            cb_wid, lcm = 0.03, cf.lcm(r_obj.n_phase, 2)
            ax_dim = [2 + r_obj.n_filt, lcm + c_ofs]
            n_col_hm, n_col_cb = int(lcm / r_obj.n_phase), int(lcm / 2)

            # sets the axes height/width ratios
            cb_hght, gap_hght = 0.025, pH * r_obj.n_filt
            ax_hght = (1 - (cb_hght + gap_hght)) / r_obj.n_filt
            h_ratio = [ax_hght] * r_obj.n_filt + [gap_hght] + [cb_hght]
            if c_ofs == 1:
                w_ratio = [(1. - cb_wid) / ax_dim[1]] * (ax_dim[1] - 1) + [cb_wid]
            else:
                w_ratio = [1. / ax_dim[1]] * ax_dim[1]

            # creates the plot axes
            wspace = 2 / 10 if (n_col_hm == 3) else 4 / 50
            gs = gridspec.GridSpec(ax_dim[0], ax_dim[1], width_ratios=w_ratio, height_ratios=h_ratio,
                                   figure=self.plot_fig.fig, wspace=wspace, left=0.03, right=0.98,
                                   bottom=bottom, top=top, hspace=0.1)

            # sets each of the plot axes
            self.plot_fig.ax = np.empty(r_obj.n_filt * (r_obj.n_phase + 1) + (1 + c_ofs), dtype=object)

            # sets up the axes for each of the heatmap types
            for i_filt in range(r_obj.n_filt):
                for i_phase in range(r_obj.n_phase + c_ofs):
                    # sets the plot index
                    i_plot = i_filt * (r_obj.n_phase + c_ofs) + i_phase

                    # creates the subplot
                    if i_phase == r_obj.n_phase:
                        # case is the depth heatmap axes
                        self.plot_fig.ax[i_plot] = self.plot_fig.figure.add_subplot(gs[i_filt, -1])
                    else:
                        # case is the spiking heatmap axes
                        i_col = [i_phase * n_col_hm, (i_phase + 1) * n_col_hm]
                        self.plot_fig.ax[i_plot] = self.plot_fig.figure.add_subplot(gs[i_filt, i_col[0]:i_col[1]])

            # sets the colour bar axes (depending on type)
            if c_ofs:
                # creates the heatmap/depth colourbars
                self.plot_fig.ax[-2] = self.plot_fig.figure.add_subplot(gs[-1, :n_col_cb])
                self.plot_fig.ax[-1] = self.plot_fig.figure.add_subplot(gs[-1, n_col_cb:2 * n_col_cb])
            else:
                # creates the heatmap colourbar only
                self.plot_fig.ax[-1] = self.plot_fig.figure.add_subplot(gs[-1, :])

        def setup_heatmap_labels(r_obj, I_hm, i_filt, D):
            '''

            :param r_obj:
            :param I_hm:
            :return:
            '''

            # initialisations
            n_row = np.size(I_hm, axis=0)
            n_lbl = 1 + int(not r_obj.is_single_cell) * (2 + int(r_obj.plot_all_expt))
            lbl_str = np.empty((n_row, n_lbl), dtype=object)

            # sets the label strings based on the analysis type
            if r_obj.is_single_cell:
                # case is single cell analysis
                lbl_str[:, 0] = np.array(['Trial #{0}'.format(x + 1) for x in range(n_row)])
            else:
                # case is whole cell analysis
                j_ofs = int(r_obj.plot_all_expt)
                if r_obj.plot_all_expt:
                    # case is all experiments are being used, so set the experiment names
                    i_expt = list(r_obj.i_expt[i_filt])
                    exp_name = [cf.extract_file_name(x['expFile']) for x in self.data.cluster]
                    lbl_str[:, 0] = np.array(['Expt = {0}'.format(exp_name[i_ex]) for i_ex in i_expt])

                # sets the other label strings
                lbl_str[:, j_ofs] = np.array(
                    ['Cluster #{0}'.format(cl_id) for cl_id in cf.flat_list(r_obj.cl_id[i_filt])])
                lbl_str[:, 1 + j_ofs] = np.array(
                    ['Channel #{0}'.format(ch_id) for ch_id in cf.flat_list(r_obj.ch_id[i_filt])])
                lbl_str[:, 2 + j_ofs] = np.array(['Depth = {0}{1}m'.format(d, cf._mu) for d in np.argsort(D)])

            # returns the final label strings
            return ['\n'.join(list(lbl_str[i, :])) for i in range(n_row)]

        def create_heatmap_markers(ax, lbl):
            '''

            :param ax:
            :param lbl_str:
            :return:
            '''

            # search_lim.append(Rectangle((t_min, y_lim[0]), (t_max - t_min), y_lim[1]))
            # pc = PatchCollection(search_lim, facecolor='r', alpha=0.25, edgecolor='k')
            # self.plot_fig.ax[i_plot].add_collection(pc)

            # initialisations
            dY = 0.15
            xL, yL, h = ax.get_xlim(), ax.get_ylim(), []

            #
            for i_patch in range(len(lbl)):
                h.append(Rectangle((xL[0], yL[0] + i_patch + dY), xL[1] - xL[0], 1 - 2 * dY))

            # creates the cursor object
            pc = PatchCollection(h, facecolor='g', alpha=0.0, zorder=10)
            ax.add_collection(pc)
            datacursor(pc, formatter=formatter, point_labels=lbl, hover=True)

        def get_channel_depths(cluster):
            '''

            :param cluster:
            :return:
            '''

            # memory allocation
            depth = []

            # retrieves the depths from each experiment
            for c in cluster:
                chMap = c['expInfo']['channel_map']
                depth.append([chMap[chMap[:, 1] == x, 3][0] for x in c['chDepth']])

            # returns the
            return depth

        def setup_spiking_heatmap(t_spike, xi_h, is_single_cell, D):
            '''

            :param t_spike:
            :param xi_h:
            :param row_norm:
            :return:
            '''

            # sets the number of
            if is_single_cell:
                n_row = np.size(t_spike, axis=1)
            else:
                n_row = np.size(t_spike, axis=0)

            # memory allocation
            n_phase, n_bin = np.size(t_spike, axis=2), np.size(xi_h, axis=1)
            I_hm = np.zeros((n_row, n_bin - 1, n_phase))

            #
            for i_phase in range(n_phase):
                for i_trial in range(n_row):
                    if is_single_cell:
                        t_sp_flat = cf.flat_list(
                            [list(x) if x is not None else [] for x in t_spike[:, i_trial, i_phase]])
                        h_gram = np.histogram(t_sp_flat, bins=xi_h[0, :])
                        dt = np.diff(xi_h[0, :])
                    else:
                        t_sp_flat = cf.flat_list(
                            [list(x) if x is not None else [] for x in t_spike[i_trial, :, i_phase]])
                        h_gram = np.histogram(t_sp_flat, bins=xi_h[i_trial, :])
                        dt = np.diff(xi_h[i_trial, :])

                    # normalises the histograms by the duration of the bins
                    I_hm[i_trial, :, i_phase] = h_gram[0] / dt

            # sorts the clusters by depth (whole experiments only)
            if not is_single_cell:
                I_hm = I_hm[np.argsort(D), :, :]

            # returns the final heatmap
            return I_hm

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        if not r_obj.is_ok:
            self.calc_ok = False
            return

        # retrieves the depths from each of the experiments (based on type)
        if not r_obj.is_single_cell:
            # case is the whole experiments
            data = self.get_data().cluster
            ch_depth = np.array(get_channel_depths(data))

            D = [[ch_depth[x][y] for x, y in zip(i_ex, cf.flat_list([list(x) for x in cl_id]))] for i_ex, cl_id in
                 zip(r_obj.i_expt, r_obj.clust_ind)]
        else:
            # case is single cell (no need to sort by depth)
            D = [None] * r_obj.n_filt

        # creates the plot outlay and titles
        init_heatmap_plot_axes(r_obj)

        # creates the heatmaps for each filter/phase
        I_hm = np.empty(r_obj.n_filt, dtype=object)
        for i_filt in range(r_obj.n_filt):
            # determines the stimuli times (converts from indices to actual times)
            wvm_p = r_obj.wvm_para[i_filt]
            _, ind = np.unique(r_obj.i_expt[i_filt], return_inverse=True)
            t_stim = [wvm_p[i][0]['nPts'] / (2 * sf) if r_obj.is_ud else wvm_p[i][0]['tPeriod'] / (2 * sf)
                      for i, sf in zip(ind, r_obj.s_freq[i_filt])]

            # sets up the histogram bins
            xi_h0 = [setup_heatmap_bins(x, dt) for x in t_stim]
            for i in range(len(xi_h0)):
                xi_h0[i][-1] = t_stim[i]

            # calculates the spiking frequency histograms
            xi_h = np.vstack(xi_h0)
            I_hm[i_filt] = setup_spiking_heatmap(r_obj.t_spike[i_filt], xi_h, r_obj.is_single_cell, D[i_filt])

        # sorts the clusters by depth
        if not r_obj.is_single_cell:
            # normalises each trial across each phase/filter type
            # I_hm_norm = np.max(np.vstack([np.max(np.max(x, axis=1), axis=1) for x in I_hm]), axis=0)
            for i_filt in range(len(I_hm)):
                I_hm_norm = np.max(np.max(I_hm[i_filt], axis=1), axis=1)
                for i_trial in range(len(I_hm_norm)):
                    if I_hm_norm[i_trial] > 0:
                        I_hm[i_filt][i_trial, :, :] /= I_hm_norm[i_trial]

        # creates the heatmaps for each filter/phase
        for i_filt in range(r_obj.n_filt):
            i_ofs = i_filt * (r_obj.n_phase + int(not r_obj.is_single_cell))
            for i_phase in range(r_obj.n_phase):
                # creates the
                i_plot = i_ofs + i_phase
                self.plot_fig.ax[i_plot].grid(False)

                # sets up the heatmap markers
                cf.set_axis_limits(self.plot_fig.ax[i_plot],
                                   [-0.5, np.size(I_hm[i_filt], axis=1) - 0.5],
                                   [-0.5, np.size(I_hm[i_filt], axis=0) - 0.5])

                # # creates the heatmap row labels (if being set)
                # if show_labels:
                #     lbl = setup_heatmap_labels(r_obj, I_hm[i_filt], i_filt, D[i_filt])
                #     create_heatmap_markers(self.plot_fig.ax[i_plot], lbl)

                # displays the heatmap
                im = self.plot_fig.ax[i_plot].imshow(I_hm[i_filt][:, :, i_phase], aspect='auto',
                                                     cmap='viridis', origin='lower')

                # # IS THIS NECESSARY?!
                # self.plot_fig.ax[i_plot].invert_yaxis()

                # sets the subplot title (first row only)
                if i_filt == 0:
                    self.plot_fig.ax[i_plot].set_title(r_obj.phase_lbl[i_phase])

                # sets the x-axis label (last row only)
                if (i_filt + 1) == r_obj.n_filt:
                    self.plot_fig.ax[i_plot].set_xlabel('Time (s)')

                #
                if ((i_phase + 1) == r_obj.n_phase) and (not r_obj.is_single_cell):
                    im_d = self.plot_fig.ax[i_plot + 1].imshow(np.sort(D[i_filt]).reshape(-1, 1), aspect='auto',
                                                               cmap='Reds', origin='lower')
                    self.plot_fig.ax[i_plot + 1].grid(False)
                    self.plot_fig.ax[i_plot + 1].get_xaxis().set_visible(False)
                    self.plot_fig.ax[i_plot + 1].get_yaxis().set_visible(False)
                    cbar = self.plot_fig.figure.colorbar(im_d, cax=self.plot_fig.ax[-2], orientation='horizontal')
                    self.plot_fig.ax[-2].set_xlabel('Depth ({0}m)'.format(cf._mu))

                #
                if i_plot == 0:
                    self.plot_fig.figure.colorbar(im, cax=self.plot_fig.ax[-1], orientation='horizontal')
                    self.plot_fig.ax[-1].set_xlabel(
                        '{0}Firing Rate'.format('' if r_obj.is_single_cell else 'Normalised '))

                # sets the y-axis labels (based on phase index)
                if i_phase == 0:
                    # if the first phase, then add the filter type as the y-axis string
                    yL = self.plot_fig.ax[i_plot].get_ylim()
                    self.plot_fig.ax[i_plot].set_yticks([np.mean(yL)])
                    self.plot_fig.ax[i_plot].set_yticklabels([r_obj.lg_str[i_filt]], rotation=90, va='bottom',
                                                             ha='center', rotation_mode='anchor')
                else:
                    # otherwise, remove all tickmarks from the y-axis
                    self.plot_fig.ax[i_plot].get_yaxis().set_visible(False)

                # converts the time points to actual time
                if (i_filt + 1) < r_obj.n_filt:
                    self.plot_fig.ax[i_plot].get_xaxis().set_visible(False)
                else:
                    x_ticks = self.plot_fig.ax[i_plot].get_xticks().astype(float)[1:-1]
                    x_ticks_new = ['{:4.2f}'.format(x) for x in x_ticks * (dt / 1000)]
                    self.plot_fig.ax[i_plot].set_xticks(x_ticks)
                    self.plot_fig.ax[i_plot].set_xticklabels(x_ticks_new)

        # resets the figure layout (single cell only)
        if r_obj.is_single_cell:
            self.plot_fig.fig.set_tight_layout(False)
            self.plot_fig.fig.suptitle('Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0], r_obj.ch_id[0][0]),
                                       fontsize=16, fontweight='bold')
            self.plot_fig.fig.tight_layout(rect=[0, 0.01, 1, 0.955])

    def create_depth_dirsel_plot(self, r_obj, plot_grid):
        '''

        :param r_obj:
        :param plot_grid:
        :return:
        '''

        # FINISH ME
        a = 1

    ####################################################
    ####    SPIKING FREQUENCY ANALYSIS FUNCTIONS    ####
    ####################################################

    def setup_stats_nvalue_array(self, sf_type, sf_type_pr, i_grp, stats_type):
        '''

        :param sf_type_pr:
        :param i_grp:
        :param stats_type:
        :return:
        '''

        # calculates the n-values for each type
        n_DS = (np.vstack([[sum(sf_type[x] > 0) for x in i_grp[1]]] * 2) * sf_type_pr[1] / 100)
        if stats_type == 'Motion Sensitivity':
            n_DS_Tot, n_Tot = np.sum(n_DS, axis=0), np.array([len(x) for x in i_grp[0]])
            n_DS = np.vstack([n_Tot - n_DS_Tot, n_DS_Tot])

        # returns the N-value array
        return n_DS

    def create_spike_freq_stats_table(self, ax, n_DS, n_filt, stats_type, c=None, c2=None, n_row=2, n_col=3, n_PD=None):
        '''

        :param n_DS:
        :return:
        '''

        # enforces tight layout format
        self.plot_fig.fig.tight_layout()

        # initialisations
        n_DS_Full = cf.add_rowcol_sum(n_DS).T

        # creates the colours (if not provided)
        if c is None:
            c = cf.get_plot_col(np.size(n_DS, axis=1))
            c2 = cf.get_plot_col(2, np.size(n_DS, axis=1))

        # creates the title text object
        t_str = '{0} N-Values'.format(stats_type)
        row_hdr = ['#{0}'.format(x + 1) for x in range(n_filt)] + ['Total']
        h_title = ax.text(0.5, 1, t_str, fontsize=15, horizontalalignment='center')

        # sets up the n-value table
        t_props_1 = cf.add_plot_table(self.plot_fig, ax, table_font, n_DS_Full.astype(int), row_hdr,
                                    ['Insensitive', 'Sensitive', 'Total'], c + [(0.75, 0.75, 0.75)],
                                    c2[0:2]  + [(0.75, 0.75, 0.75)], None, n_row=n_row, n_col=n_col, h_title=h_title)

        # calculates the height between the title and the top of the table
        dh_title = h_title.get_position()[1] - (t_props_1[0]._bbox[1] + t_props_1[0]._bbox[3])
        c_hght = t_props_1[0]._bbox[3] / (np.size(n_DS_Full, axis=0) + 1)

        #
        if n_PD is not None:
            # sets the prederred direction parameters
            cPD = cf.get_plot_col(5, np.size(n_PD, axis=1))
            col_hdr = ['None', 'Rotation', 'Visual', 'Incongruent', 'Congruent']

            # creates the title text object
            t_str_pd = 'Preferred Direction N-Values'.format(stats_type)
            h_title_pd = ax.text(0.5, 1, t_str_pd, fontsize=15, horizontalalignment='center')

            # sets up the n-value table
            t_props_pd = cf.add_plot_table(self.plot_fig, ax, table_font, n_PD.astype(int), row_hdr, col_hdr,
                                        c + [(0.75, 0.75, 0.75)], cPD, None, n_row=n_row, n_col=n_col,
                                        h_title=h_title_pd)

            # resets the bottom location of the upper table
            t_props_pd[0]._bbox[1] = t_props_1[0]._bbox[1] - (t_props_1[0]._bbox[3] + 2 * c_hght)

            # resets the titla position
            t_pos = list(h_title_pd.get_position())
            t_pos[1] = t_props_pd[0]._bbox[1] + t_props_pd[0]._bbox[3] + dh_title
            h_title_pd.set_position(tuple(t_pos))

            #
            t_props = t_props_pd
        else:
            #
            t_props = t_props_1

        # creates the statistics table
        if n_filt > 1:
            # parameters
            p_value_sig = 0.05  # does this need to be scaled?

            # calculates the chi-squared test p-values between each filter type group
            chi_stats = np.zeros((n_filt, n_filt), dtype='U10')
            for i_row in range(n_filt):
                for i_col in range(n_filt):
                    if i_row == i_col:
                        # row and column indices are equal
                        chi_stats[i_row, i_col] = 'N/A'
                    else:
                        # otherwise, calculate the statistics
                        Br = ro.r.matrix(n_DS[:, np.array([i_col, i_row])], nrow=2, ncol=2)
                        result = r_stats.chisq_test(Br, correct=False)
                        chi_pval = result[result.names.index('p.value')][0]

                        # sets the p-value significance string for the table cell
                        if np.isnan(chi_pval):
                            # case is p-value is a NaN-value
                            sig_str = '*****'
                        else:
                            # case is a legitimate p-value
                            sig_str = '{:5.3f}{}'.format(chi_pval, '*' if chi_pval < p_value_sig else '')

                        chi_stats[i_row, i_col] = chi_stats[i_col, i_row] = sig_str

            # creates the title text object
            t_str_2 = '{0} Statistics'.format(stats_type)
            h_title_2 = ax.text(0.5, 1, t_str_2, fontsize=15, horizontalalignment='center')

            # sets up the table other object properties
            t_ofs = (1 - t_props[1]) * t_props[2]
            row_hdr = col_hdr = ['#{0}'.format(str(x + 1)) for x in range(n_filt)]

            # sets up the n-value table
            t_props_2 = cf.add_plot_table(self.plot_fig, ax, table_font, chi_stats, row_hdr, col_hdr,
                                          c, c, None, n_row=n_row, n_col=n_col, h_title=h_title_2)

            # resets the bottom location of the upper table
            t_props_2[0]._bbox[1] = t_props[0]._bbox[1] - (t_props[0]._bbox[3] + c_hght)

            # resets the titla position
            t_pos = list(h_title_2.get_position())
            t_pos[1] = t_props_2[0]._bbox[1] + t_props_2[0]._bbox[3] + dh_title
            h_title_2.set_position(tuple(t_pos))

    ###############################################
    ####    GENERAL ANALYSIS PLOT FUNCTIONS    ####
    ###############################################

    def init_plot_axes(self, n_plot=None, n_row=None, n_col=None, is_3d=False, is_polar=False, proj_type=None):
        '''

        :return:
        '''

        # creates the new subplot sonfiguration
        self.clear_plot_axes()
        self.plot_fig.setup_plot_axis(n_plot=n_plot, n_row=n_row, n_col=n_col, is_3d=is_3d,
                                      is_polar=is_polar, proj_type=proj_type)

    def clear_plot_axes(self):
        '''

        :return:
        '''

        # removes any previous axis objects
        try:
            if self.plot_fig.ax is not None:
                for ax in self.plot_fig.ax:
                    ax.remove()
        except:
            pass

    def remove_scatterplot_spines(self, ax):
        '''

        :return:
        '''

        for spine in ax.spines.values():
            spine.set_visible(False)

    def det_calc_para_change(self, calc_para, plot_para, current_fcn):
        '''

        :param calc_para:
        :param current_fcn:
        :return:
        '''

        # mandatory update function list
        func_plot_chk = ['Direction ROC Curves',
                         'Direction ROC Curves (Single Cell)',
                         'Direction ROC Curves (Whole Experiment)',
                         'Motion/Direction Selectivity Cell Grouping Scatterplot',
                         'Rotation/Visual Stimuli Response Statistics',
                         'Velocity ROC Curves (Single Cell)',
                         'Velocity ROC Curves (Whole Experiment)',
                         'Velocity ROC Curves (Pos/Neg Comparison)']

        if self.fcn_data.prev_calc_para is None:
            # otherwise, if there were no previous calculation parameters then update
            return True

        else:
            # otherwise, determine if any of the calculation parameters have changed
            for cp in calc_para:
                if cp in self.fcn_data.prev_calc_para:
                    if (cp != 'exp_name') and (calc_para[cp] != self.fcn_data.prev_calc_para[cp]):
                        return True
                else:
                    # otherwise, return a true value
                    return True

            # if in the analysis list, determine if any of the plotting parameters have changed
            if current_fcn in func_plot_chk:
                for pp in plot_para:
                    if pp in self.fcn_data.prev_plot_para:
                        if (pp != 'exp_name') and (plot_para[pp] != self.fcn_data.prev_plot_para[pp]):
                            return True
                    else:
                        # otherwise, return a true value
                        return True

            # flag that no change has taken place
            return False

    def split_func_para(self):
        '''

        :return:
        '''

        # retrieves the currently selected parameters
        sel_item = self.list_funcsel.selectedItems()
        fcn_para = next(x['para'] for x in self.fcn_data.details[self.fcn_data.type] if x['name'] == sel_item[0].text())

        # returns the
        return dict([(x, self.fcn_data.curr_para[x]) for x in cf.get_para_dict(fcn_para, 'C')]), \
               dict([(x, self.fcn_data.curr_para[x]) for x in cf.get_para_dict(fcn_para, 'P')])

    def get_rotation_names(self, f_perm, f_key, t_key):
        '''

        :param key:
        :return:
        '''

        #
        return [y if (t_key[x] is None) else t_key[x][y] for x, y in zip(f_key, f_perm)]

    ####################################################################################################################
    ####                                           DATA OUTPUT FUNCTIONS                                            ####
    ####################################################################################################################

    def output_cluster_matching_data(self, out_file, is_csv):
        '''

        :return:
        '''

        # sets the greek characters (depending on file type)
        if is_csv:
            # no greek characters for csv files
            d_str, mu_str = 'd', 'u'
        else:
            # otherwise, set the codes for the greek characters
            d_str, mu_str = cf._delta, cf._mu

        # column header strings
        h_str = np.array(['Expt Name', 'Fixed ID#', 'Cell Match?', 'Free ID#', '',
                 '{0}Depth ({1}m)'.format(d_str, mu_str), 'Signal Correlation',
                 'Total Distance', 'Signal Intersection', '',
                 'ISI Correlation', 'ISI Intersection', '',
                 '%{0}(B)'.format(d_str), '%{0}(C)'.format(d_str),
                 '%{0}(A-D)'.format(d_str), '%{0}(B-D)'.format(d_str)])

        # column header key
        h_key = np.array(['Experiment Name'
                 'Cluster ID# from fixed preperation', 'Indicates whether a feasible match was made',
                 'Matching Cluster ID# from free preparation', '', 'Difference in channel depth between matches',
                 'Cross-correlation between matching signals', 'Total distance between matching signals',
                 'Intersection between matching signal point histograms', '',
                 'ISI histogram cross-correlation between matching signals',
                 'ISI histogram intersection between matching signals', '',
                 '% difference between the 2nd peak amplitude between matching signals',
                 '% difference between overall minimum between matching signals',
                 '% difference between the (1st peak amplitude - overall minimum) between matching signals',
                 '% difference between the (2nd peak amplitude - overall minimum) between matching signals'])

        # sets the acceptance flags based on the method
        if self.is_multi:
            # case is multi-experiment files have been loaded
            exp_name = [cf.extract_file_name(x['expFile']) for x in self.data.cluster]
            data_fix, data_free, comp = self.get_multi_comp_datasets(True, exp_name)
        else:
            # retrieves the fixed/free datasets
            comp, h_str, h_key = [self.data.comp], h_str[1:], h_key[1:]
            data_fix, data_free = self.get_comp_datasets()

        # retrieves the acceptance flags/match indices
        is_acc, i_match = get_list_fields(comp,'is_accept'), get_list_fields(comp,'i_match')

        # memory allocation
        c_ofs = int(self.is_multi)
        n_clust, n_col = len(is_acc), len(h_str) + 2
        n_row, ind_row = max(n_col, n_clust + 1), np.array(range(n_clust)) + 1
        data = np.empty((n_row, n_col), dtype='U200')

        # sets the fixed/free ID flags
        if self.is_multi:
            # case is a multi-experiment
            fix_id = cf.flat_list([x['clustID'] for x in data_fix])
            free_id = np.array(cf.flat_list([
                self.get_free_cluster_match_ids(x['clustID'], y.i_match, y.is_accept) for x, y in zip(data_free, comp)])
            )

            # sets the experiment names
            data[ind_row, 0] = np.array(cf.flat_list([[x] * y['nC'] for x, y in zip(exp_name, data_fix)]))
        else:
            # case is a single experiment
            fix_id = data_fix['clustID']
            free_id = self.get_free_cluster_match_ids(data_free['clustID'], comp[0].i_match, comp[0].is_accept)

        # sets up the data output array
        data[0, :len(h_str)] = h_str
        data[ind_row, 0 + c_ofs] = np.array(fix_id).astype(str)
        data[ind_row, 1 + c_ofs] = np.array(['Yes' if x else 'No' for x in is_acc])
        data[ind_row, 2 + c_ofs] = free_id

        # signal match values
        data[ind_row, 4 + c_ofs] = self.set_match_values(get_list_fields(comp,'d_depth'), is_acc)
        data[ind_row, 5 + c_ofs] = self.set_match_values(get_list_fields(comp,'sig_corr'), is_acc)
        data[ind_row, 6 + c_ofs] = self.set_match_values(get_list_fields(comp,'sig_diff'), is_acc)
        data[ind_row, 7 + c_ofs] = self.set_match_values(get_list_fields(comp,'sig_intersect'), is_acc)

        # ISI histogram match values
        data[ind_row, 9 + c_ofs] = self.set_match_values(get_list_fields(comp,'isi_corr'), is_acc)
        data[ind_row, 10 + c_ofs] = self.set_match_values(get_list_fields(comp,'isi_intersect'), is_acc)

        # signal feature differences values
        signal_feat = get_list_fields(comp,'signal_feat')
        for i_col in range(np.size(signal_feat, axis=1)):
            data[ind_row, 12 + (i_col + c_ofs)] = self.set_match_values(signal_feat[:, i_col], is_acc)

        # sets the header key fields
        data = self.set_header_key_fields(data, h_str, h_key, len(h_str) + 1)

        # outputs the data to file
        self.create_data_file(out_file, data, is_csv)

    def output_cell_classification_data(self, out_file, is_csv):
        '''

        :return:
        '''

        # retrieves the classification data
        cl_data = self.data.classify
        expt_id = self.data.classify.expt_id
        grp_idx = self.data.classify.grp_idx
        clust_id = self.data.classify.clust_id

        # column header strings
        h_str = np.array(['Expt Name', 'Cluster ID#', 'Group #', '',
                          'C (ms)', 'B/D', 'Trough HW (ms)', '(B-A)/(B+A)', '2nd Peak HW (ms)',
                          '2nd Peak Tau (ms)', 'Firing Rate (Hz)'])

        # column header key
        h_key = np.array(['Experiment Name', 'Cell Cluster ID#', 'Cell Type Clustering Group Index', '',
                          'Trough to 2nd Peak Time', '2nd Peak to Trough Ratio', 'Trough Half-Width Duration',
                          'Peak Ratio', '2nd Peak Half-Width Duration', '2nd Peak Relaxation Time', 'Firing Rate'])

        # memory allocation
        n_clust, n_col, n_met = np.size(cl_data.x_clust, axis=0), len(h_str) + 2, np.size(cl_data.x_clust, axis=1)
        n_row, ind_row = max(n_col, n_clust + 1), np.array(range(n_clust)) + 1
        data = np.empty((n_row, n_col), dtype='U100')

        # sets up the data output array
        data[0, :len(h_str)] = h_str
        data[ind_row, 0] = expt_id
        data[ind_row, 1] = clust_id.astype(str)
        data[ind_row, 2] = (grp_idx + 1).astype(str)

        #
        for i in range(n_met):
            data[ind_row, i + 4] = cl_data.x_clust[:, i].astype(str)

        # sets the header key fields
        data = self.set_header_key_fields(data, h_str, h_key, len(h_str) + 1)

        # outputs the data to file
        self.create_data_file(out_file, data, is_csv)

    def output_rotation_analysis_data(self, out_file, is_csv):
        '''

        :return:
        '''

        # FINISH ME!
        a = 1

        # outputs the data to file
        self.create_data_file(out_file, data, is_csv)

    @staticmethod
    def set_header_key_fields(data, h_str, h_key, key_col):
        '''

        :param data:
        :param h_str:
        :param h_key:
        :param key_col:
        :return:
        '''

        # sets the header string
        data[0, key_col] = 'Column Header Key'

        # sets the data header key strings
        for i_row in range(len(h_key)):
            if len(h_str[i_row]):
                data[i_row + 2, key_col] = '{0} - {1}'.format(h_str[i_row], h_key[i_row])

        # returns the data array
        return data

    def create_data_file(self, out_file, data, is_csv):
        '''

        :param out_file:
        :param data:
        :return:
        '''

        # retrieves the output file name
        f_name = cf.extract_file_name(out_file)

        if is_csv:
            # case is a csv file
            while 1:
                try:
                    # creates a dataframe and outputs the data to file
                    df = pd.DataFrame(data)
                    df.to_csv(out_file, index=False, header=False)
                    break
                except PermissionError:
                    # if the file is open, prompt the user to close the file before continuing
                    e_str = 'The file "{0}.csv" is currently open. ' \
                            'Close the file and click OK to continue.'.format(f_name)
                    cf.show_error(e_str, 'Data File Output Error')
        else:
            # case is an xlsx spreadsheet
            while 1:
                try:
                    # opens the workboof and prepares it for writing data
                    wbook = xlsxwriter.Workbook(out_file)
                    wsheet = wbook.add_worksheet()

                    # output the data to the work sheet
                    for i_row in range(np.size(data, axis=0)):
                        for i_col in range(np.size(data, axis=1)):
                            wsheet.write(i_row, i_col, data[i_row, i_col])

                    # closes the workbook
                    wbook.close()
                    break
                except PermissionError:
                    # if the file is open, prompt the user to close the file before continuing
                    e_str = 'The file "{0}.xlsx" is currently open. ' \
                            'Close the file and click OK to continue.'.format(f_name)
                    cf.show_error(e_str, 'Data File Output Error')

    def output_single_figure(self, fig_info, var_name=None, para_name=None, i_plot=None):

        # sets the base image name
        img_name = os.path.join(fig_info['figDir'], fig_info['figName'])
        if i_plot is not None:
            # if outputting all figures, then set the new output image
            img_name =  '{0} ({1} #{2})'.format(img_name, para_name, i_plot)

            # updates the plot axes
            self.fcn_data.curr_para[var_name] = i_plot
            self.update_click()
            time.sleep(0.1)

        # sets the full image file name
        if os.path.isfile(img_name) and (i_plot is None):
            # if the file already exists, prompt the user if they wish to overwrite the file
            prompt_text = "File already exists. Are you sure you want to overwrite?"
            u_choice = QMessageBox.question(self, 'Overwrite File?', prompt_text,
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if u_choice == QMessageBox.No:
                # if the user rejected then exit the function
                return

        # saves the figure to file
        self.plot_fig.fig.savefig(cf.set_file_name(img_name, fig_info['figFmt']),
                                  dpi=int(fig_info['figDPI']),
                                  facecolor=fig_info['fColour'][0].lower(),
                                  edgecolor=fig_info['eColour'][0].lower(),
                                  orientation=fig_info['figOrient'].lower())

    #########################################
    ####     MISCELLANEOUS FUNCTIONS     ####
    #########################################

    @staticmethod
    def opt_minmax(x, pp, is_max=True):

        if is_max:
            return -pp(x[0])
        else:
            return pp(x[0])

    def get_comp_datasets(self, ind=None, is_full=False):
        '''

        :return:
        '''

        if ind is None:
            if (self.data.cluster is None) or is_full:
                return self.data._cluster[self.data.comp.ind[0]], self.data._cluster[self.data.comp.ind[1]]
            else:
                return self.data.cluster[self.data.comp.ind[0]], self.data.cluster[self.data.comp.ind[1]]
        else:
            if (self.data.cluster is None) or is_full:
                return self.data._cluster[ind[0]], self.data._cluster[ind[1]]
            else:
                return self.data.cluster[ind[0]], self.data.cluster[ind[1]]

    def get_multi_comp_datasets(self, all_expt, exp_name, is_list=True):

        if all_expt:
            return [x['data'][0] for x in self.data.cluster], \
                   [x['data'][1] for x in self.data.cluster], \
                   [x['comp_data'] for x in self.data.cluster]
        else:
            # retrieves the experiment
            i_expt = cf.get_expt_index(exp_name, self.data.cluster)
            if is_list:
                return self.data.cluster[i_expt]['data'][0], \
                       self.data.cluster[i_expt]['data'][1], \
                       [self.data.cluster[i_expt]['comp_data']]
            else:
                return self.data.cluster[i_expt]['data'][0], \
                       self.data.cluster[i_expt]['data'][1], \
                       self.data.cluster[i_expt]['comp_data']

    def check_cluster_index_input(self, i_cluster, plot_all, n_max):
        '''

        :param i_cluster:
        :param n_fix:
        :return:
        '''

        # initialisations
        e_str = None
        if plot_all:
            i_cluster = np.arange(n_max) + 1

        # checks to see the cluster indices are correct (based on their type)
        if isinstance(i_cluster, int):
            if i_cluster > n_max:
                e_str = 'You have specified an index greater than the number of clusters ({0}). ' \
                        'Reset the cluster index and re-run the function.'.format(n_max)
        else:
            n_plot = len(i_cluster)
            if any([x > n_max for x in i_cluster]):
                e_str = 'You have specified one or more indices that are greater than the number of ' \
                        'clusters ({0}). Reset the cluster indices and re-run the function.'.format(n_max)
            elif n_plot > n_plot_max:
                e_str = 'You have specified {0} subplots which is greater than the maximum number of ' \
                        'subplots ({1}). Reset the cluster indices and re-run the function.'.format(n_plot, n_plot_max)

        # ensures the cluster index array has the correct form
        if e_str is None:
            i_cluster = self.set_cluster_indices(i_cluster, n_max=n_max)

        # returns the error string
        return i_cluster, e_str

    def get_free_cluster_match_ids(self, clustID, i_match, is_accept=None):
        '''

        :param clustID:
        :param i_match:
        :return:
        '''

        if is_accept is None:
            return [clustID[x] for x in i_match]
        else:
            return [str(clustID[x]) if y else 'N/A' for x, y in zip(i_match, is_accept)]

    def setup_time_vector(self, sFreq, n_pts):
        '''

        :return:
        '''

        return 1000.0 * np.arange(n_pts) / sFreq

    def set_cluster_indices(self, i_cluster, n_max=None):
        '''

        :param self:
        :param i_cluster:
        :return:
        '''

        # sets the cluster indices based on the type
        if i_cluster is None:
            # indices not specified, so plot all fixed clusters
            if n_max is None:
                data_fix, _ = self.get_comp_datasets()
                return np.array(range(data_fix['nC'])) + 1
            else:
                return np.array(range(n_max)) + 1
        elif isinstance(i_cluster, int):
            # index is a single value, so convert to an array
            return [i_cluster]
        else:
            # otherwise, use the original values
            return i_cluster

    def set_group_enabled_props(self, h_groupbox, is_enabled=True):
        '''

        :param h_groupbox:
        :return:
        '''

        for h_obj in h_groupbox.findChildren(QWidget):
            h_obj.setEnabled(is_enabled)
            if isinstance(h_obj, QGroupBox):
                self.set_group_enabled_props(h_obj, is_enabled)

    def set_match_values(self, X, is_accept):
        '''

        :param X:
        :param is_acc:
        :return:
        '''

        return np.array(['{:5.4f}'.format(x) if y else 'N/A' for x, y in zip(X, is_accept)])

    def det_avail_thread_worker(self):
        '''

        :return:
        '''

        # returns the index of the next available worker
        return next(i for i in range(2) if not self.worker[i].is_running)

    def remove_rejected_clusters(self, plot_scope, current_fcn):
        '''

        :return:
        '''

        # determines if an update is required
        if not self.data.req_update:
            # if not, then exit the function
            return
        else:
            # otherwise, reset the update flag
            self.data.req_update = False

        # sets a copy of the data class
        self.update_thread_job('Removing Excluded Cells...', 10.)
        self.data.cluster = dcopy(self.data._cluster)

        # if the function is not
        for i, c in enumerate(self.data.cluster):
            # retrieves the clusters that are to be included
            cl_inc = cfcn.get_inclusion_filt_indices(c, self.data.exc_gen_filt)

            # removes/keeps the voltage spikes depending on the function type
            if current_fcn != 'plot_single_match_mean':
                # if the voltage spikes are not required, then remove them
                del c['vSpike']
            else:
                # otherwise, include only the accepted voltage spike traces
                c['vSpike'] = c['vSpike'][cl_inc]

            # if all of the clusters are to be included, then continue
            if np.all(cl_inc):
                continue

            # reduces down the cluster arrays
            c['tSpike'], c['vMu'], c['vSD'] = c['tSpike'][cl_inc], c['vMu'][:, cl_inc], c['vSD'][:, cl_inc]
            c['ccGram'], c['sigFeat'] = c['ccGram'][cl_inc, :][:, cl_inc], c['sigFeat'][cl_inc, :]
            c['clustID'], c['chDepth'] = list(np.array(c['clustID'])[cl_inc]), c['chDepth'][cl_inc]
            c['chRegion'], c['chLayer'] = c['chRegion'][cl_inc], c['chLayer'][cl_inc]
            c['isiHist'], c['ptsHist'], c['nC'] = c['isiHist'][cl_inc], c['ptsHist'][cl_inc], np.sum(cl_inc)

            # reduces down the rotational analysis information (if present in current data file)
            if c['rotInfo'] is not None:
                rI = c['rotInfo']
                for tt in rI['t_spike']:
                    rI['t_spike'][tt] = rI['t_spike'][tt][cl_inc, :, :]

########################################################################################################################
########################################################################################################################

class AnalysisFunctions(object):
    def __init__(self, h_para_grp, main_obj):
        # initialisations
        self.get_data_fcn = main_obj.get_data
        self.get_plot_grp_fcn = main_obj.get_plot_group
        self.get_plot_fcn = main_obj.get_plot_func
        self.update_plot = main_obj.update_click
        self.is_multi = False
        self.type = None
        self.details = {}
        self.curr_fcn = None
        self.curr_para = None
        self.prev_calc_para = None
        self.prev_plot_para = None
        self.is_updating = False
        self.exp_name = None
        self.pool = None

        # initialises the calculation/plotting parameter groupboxes
        self.init_para_groupbox(h_para_grp)

    def set_pool_worker(self, pool):
        '''

        :param pool:
        :return:
        '''

        # sets the pool worker
        self.pool = pool

    def init_all_func(self):

        # initialisations
        m_type = ['New Method', 'Old Method']

        #########################################
        ####    CLUSTER MATCHING FUNCTIONS   ####
        #########################################

        # initialisations
        plt_list = ['Intersection', 'Wasserstein Distance', 'Bhattacharyya Distance']

        # ====> Signal Comparison (All Clusters)
        para = {
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1, 'is_list':True},
            'plot_all': {'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para':['i_cluster',True]},
            'm_type': {'type': 'L', 'text': 'Matching Type', 'def_val': m_type[0], 'list': m_type},
            'exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'Experiments', 'is_multi': True},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Cluster Matching',
                      name='Signal Comparison (All Clusters)',
                      func='plot_multi_match_means',
                      multi_fig=['i_cluster'],
                      para=para)

        # ====> Signal Comparison (Single Cluster)
        para = {
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'n_trace': {'text': 'Raw Trace Count', 'def_val': 500},
            'is_horz': {'type': 'B', 'text': 'Plot Subplots Horizontally', 'def_val': False},
            'rej_outlier': {'type': 'B', 'text': 'Reject Outlier Traces', 'def_val': True},
            'exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'Experiments', 'is_multi': True},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Cluster Matching',
                      name='Signal Comparison (Single Cluster)',
                      func='plot_single_match_mean',
                      multi_fig=['i_cluster'],
                      para=para)

        # ====> Match Metrics
        para = {
            'is_3d': {'type': 'B', 'text': 'Plot 3D Data', 'def_val': True},
            'm_type': {'type': 'L', 'text': 'Matching Type', 'def_val': m_type[0], 'list': m_type},
            'all_expt': {'type': 'B', 'text': 'Plot All Experiments',
                         'def_val': False, 'link_para': ['exp_name', True], 'is_multi': True},
            'exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'Experiments', 'is_multi': True},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Cluster Matching',
                      name='Cluster Match Metrics',
                      func='plot_signal_metrics',
                      para=para)

        # ====> Old Cluster Matched Signals
        para = {
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1, 'is_list':True},
            'plot_all': {'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para':['i_cluster', True]},
            'exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'Experiments', 'is_multi': True},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Cluster Matching',
                      name='Old Cluster Matched Signals',
                      func='plot_old_cluster_signals',
                      para=para)

        # ====> New Cluster Matched Signals
        para = {
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1, 'is_list':True},
            'plot_all': {'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para':['i_cluster', True]},
            'sig_type': {'type': 'L', 'text': 'Plot Type', 'def_val': 'Intersection', 'list': plt_list},
            'exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'Experiments', 'is_multi': True},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Cluster Matching',
                      name='New Cluster Matched Signals',
                      func='plot_new_cluster_signals',
                      multi_fig=['i_cluster'],
                      para=para)

        # # ====> Shuffled Cluster Distances
        # para = {
        #     # calculation parameters
        #     'n_shuffle': {'gtype': 'C', 'text': 'Number of shuffles', 'def_val': 100},
        #     'n_spikes': {'gtype': 'C', 'text': 'Spikes per shuffle', 'def_val': 10},
        #
        #     # plotting parameters
        #     'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1, 'is_list':True},
        #     'plot_all': {'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para':['i_cluster', True]},
        #     'p_type': {'type': 'L', 'text': 'Plot Type', 'def_val':'bar', 'list':['bar', 'boxplot']},
        #     'exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'Experiments', 'is_multi': True},
        #     'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        # }
        # self.add_func(type='Cluster Matching',
        #               name='Shuffled Cluster Distances',
        #               func='plot_cluster_distances',
        #               multi_fig=['i_cluster'],
        #               para=para)

        # ====> Inter-Spike Interval Distributions
        para = {
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1, 'is_list':True},
            'plot_all': {'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para':['i_cluster', True]},
            't_lim': {'text': 'Upper Time Limit', 'def_val': 500, 'min_val': 10},
            'plot_all_bin': {'type': 'B', 'text': 'Plot All Histogram Time Bins', 
                             'def_val': False, 'link_para': ['t_lim', True]},
            'is_norm': {'type': 'B', 'text': 'Use ISI Probabilities', 'def_val': True},
            'equal_ax': {'type': 'B', 'text': 'Use Equal Y-Axis Limits', 'def_val': False},
            'exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'Experiments', 'is_multi': True},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Cluster Matching',
                      name='Inter-Spike Interval Distributions',
                      func='plot_cluster_isi',
                      multi_fig=['i_cluster'],
                      para=para)

        ###############################################
        ####    CLUSTER CLASSIFICATION FUNCTIONS   ####
        ###############################################

        # combobox lists
        c_metric = ['Trough to 2nd Peak Time (ms)', '2nd Peak to Trough Ratio', 'Trough Half-Width (ms)',
                    'Peak Ratio'] #, '2nd Peak Half-Width (ms)', '2nd Peak Relaxation Time (ms)',
                    # 'Firing Rate (Hz)']
        act_type = ['Excitatory', 'Inhibitory', 'Excited', 'Inhibited',
                    'Rejected Excitatory', 'Rejected Inhibitory', 'Abnormal']
        class_type = ['K-Means','Gaussian Mixture Model']

        # ====> Cluster Classification
        para = {
            'exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'Experiments'},
            'all_expt': {'type': 'B', 'text': 'Analyse All Experiments',
                         'def_val': True, 'link_para': ['exp_name', True]},
            'c_met1': {'type': 'L', 'text': 'Metric #1', 'def_val': c_metric[0], 'list': c_metric},
            'c_met2': {'type': 'L', 'text': 'Metric #2', 'def_val': c_metric[1], 'list': c_metric},
            'c_met3': {'type': 'L', 'text': 'Metric #3', 'def_val': c_metric[2], 'list': c_metric},
            'use_3met': {'type': 'B', 'text': 'Use 3 Metrics For Classification', 'def_val': False,
                      'link_para': ['c_met3', False]},
            'use_pca': {'type': 'B', 'text': 'Use PCA For Classification', 'def_val': False,
                        'link_para': [['c_met1', True], ['c_met2', True], ['c_met3', True], ['use_3met', True]]},
            'class_type': {'type': 'L', 'text': 'Classification Method', 'def_val': class_type[0], 'list': class_type},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Cluster Classification',
                      name='Classification Metrics',
                      func='plot_classification_metrics',
                      para=para)

        # ====> Cluster Cross-Correlogram
        para = {
            # calculation parameters
            'calc_exp_name': {'gtype': 'C', 'type': 'L', 'text': 'Experiment',
                              'def_val': None, 'list': 'Experiments'},
            'calc_all_expt': {'gtype': 'C', 'type': 'B', 'text': 'Analyse All Experiments',
                         'def_val': True, 'link_para': ['calc_exp_name', True]},
            'f_cutoff': {'gtype': 'C', 'text': 'Frequency Cutoff (kHz)', 'def_val': 5, 'min_val': 1},
            'p_lim': {'gtype': 'C', 'text': 'Confidence Interval (%)', 'def_val': 99.9999,
                      'min_val': 90, 'max_val': 100.0 - 1e-6},
            'n_min_lo': {'gtype': 'C', 'text': 'Lower Contifguous Points', 'def_val': 3,
                         'min_val': 1, 'max_val': 4},
            'n_min_hi': {'gtype': 'C', 'text': 'Upper Contiguous Points', 'def_val': 2,
                         'min_val': 1, 'max_val': 4},

            # plotting parameters
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'Experiments'},
            'action_type': {'type': 'L', 'text': 'Classification Type', 'def_val': act_type[0], 'list': act_type},
            'plot_type': {'type': 'L', 'text': 'Plot Type', 'def_val': 'bar', 'list': ['bar', 'scatterplot']},
            'i_plot': {'text': 'Plot Indices', 'def_val': 1, 'min_val': 1, 'is_list': True},
            'plot_all': {'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para': ['i_plot', True]},
            'window_size': {'text': 'Window Size (ms)', 'def_val': 10, 'min_val': 5},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Cluster Classification',
                      name='Cluster Cross-Correlogram',
                      func='plot_classification_ccgram',
                      para=para)

        ##########################################
        ####    ROTATION ANALYSIS FUNCTIONS   ####
        ##########################################

        # parameters
        scope_txt = ['Individual Cell', 'Whole Experiment']
        s_type = ['Direction Selectivity', 'Motion Sensitivity']
        p_cond = list(np.unique(cf.flat_list(cf.det_reqd_cond_types(
                            self.get_data_fcn(), ['Uniform', 'LandmarkLeft', 'LandmarkRight']))))
        pos_bin = [str(x) for x in [3, 4, 5, 6, 10, 15, 20, 30, 45, 60]]
        vel_bin = [str(x) for x in [4, 5, 8, 10, 16, 20, 40]]
        t_phase, t_ofs = 1.0, 0.2

        #
        rot_filt_all = cf.init_rotation_filter_data(False)
        rot_filt_all['t_type'] = ['Black'] + p_cond

        # ====> Rotation Trial Spike Rate Rasterplot/Histograms
        para = {
            # calculation parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True]
            },
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[0],
                'link_para': [['i_cluster', 'Whole Experiment'],
                              ['plot_all_expt', 'Individual Cell']]
            },

            'n_bin': {'text': 'Histogram Bin Count', 'def_val': 20, 'min_val': 10},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Rotation Analysis',
                      name='Rotation Trial Spiking Rates',
                      func='plot_rotation_trial_spikes',
                      para=para)

        # ====> Rotation Trial Spike Rate Comparison
        para = {
            # calculation parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True]
            },
            'p_value': {'text': 'Significance Level', 'def_val': 0.05, 'min_val': 0.00, 'max_val': 0.05},
            'stats_type': {'type': 'L', 'text': 'Statistics Type', 'list': s_type, 'def_val': s_type[0]},
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[0],
                'link_para': [['i_cluster', 'Whole Experiment'],
                              ['plot_all_expt', 'Individual Cell'],
                              ['p_value', 'Individual Cell'],
                              ['stats_type', 'Individual Cell']]
            },

            'plot_trend': {'type': 'B', 'text': 'Plot Group Trendlines', 'def_val': False},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            'is_3d': {'type': 'B', 'text': 'Plot 3D Scatterplot', 'def_val': False, 'link_para': ['plot_trend', True]},
        }
        self.add_func(type='Rotation Analysis',
                      name='Phase Spiking Rate Comparison',
                      func='plot_phase_spike_freq',
                      para=para)

        # ====> Rotation Trial Spike Rate Heatmap
        para = {
            # calculation parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {'type': 'B', 'text': 'Analyse All Experiments',
                              'def_val': True, 'link_para': ['plot_exp_name', True]},
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[0],
                'link_para': [['i_cluster', 'Whole Experiment'], ['plot_exp_name', 'Individual Cell'],
                              ['plot_all_expt', 'Individual Cell']]
            },

            'dt': {'text': 'Heatmap Resolution (ms)', 'def_val': 100, 'min_val': 10},
        }
        self.add_func(type='Rotation Analysis',
                      name='Spiking Rate Heatmap',
                      func='plot_spike_freq_heatmap',
                      para=para)

        # ====> Rotation Trial Motion/Direction Selectivity
        para = {
            # calculation parameters
            't_phase': {'gtype': 'C', 'text': 'UniformDrifting Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10},
            't_ofs': {'gtype': 'C', 'text': 'UniformDriftings Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00},

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter,
                'para_gui_var': {'rmv_fields': ['t_type']}, 'def_val': None
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None,
                              'list': 'RotationExperimentMD'},
            'plot_all_expt': {'type': 'B', 'text': 'Analyse All Experiments',
                              'def_val': True, 'link_para': ['plot_exp_name', True]},

            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1],
                'link_para': [['i_cluster', 'Whole Experiment'], ['plot_exp_name', 'Individual Cell'],
                              ['plot_all_expt', 'Individual Cell']]
            },

            'plot_cond': {'type': 'CL', 'text': 'Plot Conditions', 'list': p_cond,
                          'def_val': np.ones(len(p_cond), dtype=bool)},
            'plot_trend': {'type': 'B', 'text': 'Plot Group Trendlines', 'def_val': False},
            'plot_even_axis': {'type': 'B', 'text': 'Set Equal Axis Limits', 'def_val': True},
            'p_type': {'type': 'L', 'text': 'Plot Type', 'def_val': 'scatterplot',
                       'list': ['scatterplot', 'bubble'], 'link_para': [['plot_trend', 'bubble'],
                       ['plot_even_axis', 'bubble']]},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Rotation Analysis',
                      name='Motion/Direction Selectivity',
                      func='plot_motion_direction_selectivity',
                      para=para)

        # ====> Rotation Trial Spike Rate Kinematics (Polar Plots)
        para = {
            # calculation parameters
            'n_sample': {'gtype': 'C', 'text': 'Equal Timebin Resampling Count', 'def_val': 100},
            'equal_time': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Equal Timebins', 'def_val': False,
                'link_para': ['n_sample', False]
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {'type': 'B', 'text': 'Analyse All Experiments',
                              'def_val': True, 'link_para': ['plot_exp_name', True]},
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[0],
                'link_para': [['i_cluster', 'Whole Experiment'], ['plot_exp_name', 'Individual Cell'],
                              ['plot_all_expt', 'Individual Cell']]
            },

            'pos_bin': {'type': 'L', 'text': 'Position Bin Size (deg)', 'list': pos_bin, 'def_val': '10'},
            'vel_bin': {'type': 'L', 'text': 'Velocity Bin Size (deg/s)', 'list': vel_bin, 'def_val': '5'},

            'n_smooth': {'text': 'Smoothing Window', 'def_val': 5, 'min_val': 3},
            'is_smooth': {'type': 'B', 'text': 'Smooth Velocity Trace', 'def_val': True,
                           'link_para': ['n_smooth', False]},

            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Rotation Analysis',
                      name='Kinematic Spiking Frequency',
                      func='plot_spike_freq_kinematics',
                      para=para)

        # ====> Rotation Trial Cell Depth Direction Selectivity
        para = {
            # calculation parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True]
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
        }
        self.add_func(type='Rotation Analysis',
                      name='Overall Direction Bias',
                      func='plot_overall_direction_bias',
                      para=para)

        # ====> Rotation Trial Cell Depth Direction Selectivity
        para = {
            # calculation parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True]
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Rotation Analysis',
                      name='Cell Depth Direction Selectivity',
                      func='plot_depth_direction_selectivity',
                      para=para)

        ###############################################
        ####    UNIFORM DRIFT ANALYSIS FUNCTIONS   ####
        ###############################################

        # ====> Trial Spike Rate Rasterplot/Histograms
        para = {
            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'UniformDrifting Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True]
            },
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[0],
                'link_para': [['i_cluster', 'Whole Experiment'], ['plot_exp_name', 'Individual Cell'],
                              ['plot_all_expt', 'Individual Cell']]
            },

            'n_bin': {'text': 'Histogram Bin Count', 'def_val': 100, 'min_val': 10},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            'rmv_median': {'type': 'B', 'text': 'Remove Baseline Median', 'def_val': True},
        }
        self.add_func(type='UniformDrift Analysis',
                      name='UniformDrift Trial Spiking Rates',
                      func='plot_unidrift_trial_spikes',
                      para=para)

        # ====> Trial Spike Rate Comparison
        para = {
            # calculation parameters
            't_phase': {'gtype': 'C', 'text': 'Analysis Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10},
            't_ofs': {'gtype': 'C', 'text': 'Analysis Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00},

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'UniformDrifting Filter Parameters', 'para_gui': RotationFilter,
                'para_gui_var': {'rmv_fields': ['t_freq_dir']}, 'def_val': None
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True]
            },
            'p_value': {'text': 'Significance Level', 'def_val': 0.05, 'min_val': 0.00, 'max_val': 0.05},
            'stats_type': {'type': 'L', 'text': 'Statistics Type', 'list': s_type, 'def_val': s_type[0]},
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[0],
                'link_para': [['i_cluster', 'Whole Experiment'], ['plot_exp_name', 'Individual Cell'],
                              ['plot_all_expt', 'Individual Cell']]
            },
            'plot_trend': {'type': 'B', 'text': 'Plot Group Trendlines', 'def_val': False},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='UniformDrift Analysis',
                      name='UniformDrift Phase Spiking Rate Comparison',
                      func='plot_unidrift_spike_freq',
                      para=para)

        # ====> Trial Spike Rate Heatmap
        para = {
            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'UniformDrifting Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True]
            },
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[0],
                'link_para': [['i_cluster', 'Whole Experiment'], ['plot_exp_name', 'Individual Cell'],
                              ['plot_all_expt', 'Individual Cell']]
            },

            'dt': {'text': 'Heatmap Resolution (ms)', 'def_val': 100, 'min_val': 10},
        }
        self.add_func(type='UniformDrift Analysis',
                      name='UniformDrift Spiking Rate Heatmap',
                      func='plot_unidrift_spike_heatmap',
                      para=para)

        #####################################
        ####    ROC ANALYSIS FUNCTIONS   ####
        #####################################

        #
        n_boot_def = 100
        md_grp_type = ['MS/DS', 'MS/Not DS', 'Not MS', 'All Cells']
        resp_grp_type = ['None', 'Rotation', 'Visual', 'Both']
        grp_stype = ['Wilcoxon Paired Test', 'Delong', 'Bootstrapping']
        auc_stype = ['Delong', 'Bootstrapping']
        mean_type = ['Mean', 'Median']
        # k_grp_type = ['Discriminating Cells', 'Non-Discriminating Cells', 'All Cells']
        exc_type = ['Use All Cells', 'Low Firing Cells', 'High Firing Cells', 'Band Pass']
        has_ud_expt = any(cf.det_valid_rotation_expt(self.get_data_fcn(), True))
        roc_vel_bin = ['5', '10', '20', '40']

        # velocity/speed ranges
        dv, v_rng = 5, 80
        vc_rng = ['{0} to {1}'.format(i * dv - v_rng, (i + 1) * dv - v_rng) for i in range(int(2 * v_rng / dv))]
        sc_rng = ['{0} to {1}'.format(i * dv, (i + 1) * dv) for i in range(int(v_rng / dv))]

        # ====> Direction ROC Curves (Single Cell)
        para = {
            # calculation parameters
            'n_boot': {'gtype': 'C', 'text': 'Number bootstrapping shuffles', 'def_val': n_boot_def, 'min_val': 100},
            'auc_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'AUC Significance Test', 'list': auc_stype,
                'def_val': auc_stype[0], 'link_para': ['n_boot', 'Delong']
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_all_expt': {'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'is_visible': False},
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[0], 'is_visible': False
            },
        }
        self.add_func(type='ROC Analysis',
                      name='Direction ROC Curves (Single Cell)',
                      func='plot_direction_roc_curves_single',
                      para=para)

        # ====> Direction ROC Curves (Whole Experiment)
        para = {
            # calculation parameters
            'n_boot': {'gtype': 'C', 'text': 'Number bootstrapping shuffles', 'def_val': n_boot_def, 'min_val': 100},
            'grp_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'Cell Grouping Significance Test', 'list': grp_stype,
                'def_val': grp_stype[0], 'link_para': [['n_boot', 'Delong'], ['n_boot', 'Wilcoxon Paired Test']]
            },
            'auc_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'AUC Significance Test', 'list': auc_stype, 'def_val': auc_stype[0],
                'link_para': ['n_boot', 'Delong']
            },
            't_phase': {
                'gtype': 'C', 'text': 'UniformDrifting Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10,
                'is_enabled': has_ud_expt
            },
            't_ofs': {
                'gtype': 'C', 'text': 'UniformDriftings Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00,
                'is_enabled': has_ud_expt
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True]
            },
            'use_avg': {'type': 'B', 'text': 'Plot Cell Grouping Average', 'def_val': True},
            'connect_lines': {'type': 'B', 'text': 'Connect AUC Values', 'def_val': False},
            'md_grp_type': {
                'type': 'L', 'text': 'MD/DS Cell Grouping Type', 'list': md_grp_type, 'def_val': md_grp_type[0]
            },
            'resp_grp_type': {
                'type': 'L', 'text': 'Response Cell Grouping Type', 'list': resp_grp_type, 'def_val': resp_grp_type[0],
                'is_enabled': has_ud_expt
            },
            'use_resp_grp_type': {
                'type': 'B', 'text': 'Response Cell Grouping', 'def_val': False,
                'link_para': [['resp_grp_type', False], ['md_grp_type', True]], 'is_enabled': has_ud_expt
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
        }
        self.add_func(type='ROC Analysis',
                      name='Direction ROC Curves (Whole Experiment)',
                      func='plot_direction_roc_curves_whole',
                      para=para)

        # ====> Velocity ROC Curves (Single Cell)
        para = {
            # calculation parameters
            'n_boot': {'gtype': 'C', 'text': 'Number bootstrapping shuffles', 'def_val': n_boot_def, 'min_val': 100},
            'auc_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'AUC Significance Test', 'list': auc_stype, 'def_val': auc_stype[0],
                'link_para': ['n_boot', 'Delong']
            },
            'spd_x_rng': {
                'gtype': 'C', 'type': 'L', 'text': 'Comparison Speed Range', 'list': sc_rng, 'def_val': '0 to 10'
            },
            'vel_bin': {
                'gtype': 'C', 'type': 'L', 'text': 'Velocity Bin Size (deg/sec)', 'list': roc_vel_bin,
                'def_val': str(dv), 'para_reset': [['spd_x_rng', self.reset_spd_rng]]
            },
            'n_sample': {'gtype': 'C', 'text': 'Equal Timebin Resampling Count', 'def_val': 100},
            'equal_time': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Equal Timebins', 'def_val': False,
                'link_para': ['n_sample', False]
            },
            'pn_calc': {'gtype': 'C', 'text': 'Use Pos/Neg', 'def_val': False, 'is_visible': False},

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},

            'vel_y_rng': {'type': 'L', 'text': 'Free Velocity Range', 'list': vc_rng, 'def_val': sc_rng[0]},
            'spd_y_rng': {'type': 'L', 'text': 'Free Speed Range', 'list': sc_rng, 'def_val': sc_rng[0]},
            'use_vel': {
                'type': 'B', 'text': 'Plot Velocity ROC Values', 'def_val': True,
                'link_para': [['spd_y_rng', True], ['vel_y_rng', False]]
            },

            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_all_expt': {'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'is_visible': False},
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[0], 'is_visible': False
            },
        }
        self.add_func(type='ROC Analysis',
                      name='Velocity ROC Curves (Single Cell)',
                      func='plot_velocity_roc_curves_single',
                      para=para)

        # ====> Velocity ROC Curves (Whole Experiment)
        para = {
            # calculation parameters
            'n_boot': {'gtype': 'C', 'text': 'Number bootstrapping shuffles', 'def_val': n_boot_def, 'min_val': 100},
            'auc_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'AUC Significance Test', 'list': auc_stype, 'def_val': auc_stype[0],
                'link_para': ['n_boot', 'Delong']
            },
            'spd_x_rng': {
                'gtype': 'C', 'type': 'L', 'text': 'Dependent Speed Range', 'list': sc_rng, 'def_val': '0 to 10'
            },
            'vel_bin': {
                'gtype': 'C', 'type': 'L', 'text': 'Velocity Bin Size (deg/sec)', 'list': roc_vel_bin, 'def_val': '5',
                'para_reset': [['spd_x_rng', self.reset_spd_rng]]
            },
            'n_sample': {'gtype': 'C', 'text': 'Equal Timebin Resampling Count', 'def_val': 100},
            'equal_time': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Equal Timebins', 'def_val': False,
                'link_para': ['n_sample', False]
            },
            'pn_calc': {'gtype': 'C', 'text': 'Use Pos/Neg', 'def_val': False, 'is_visible': False},


            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True]
            },
            'use_vel': {'type': 'B', 'text': 'Plot Velocity ROC Values', 'def_val': True},

            'lo_freq_lim': {'text': 'Low Frequency Limit (Hz)', 'def_val': 1, 'min_val': 0},
            'hi_freq_lim': {'text': 'High Frequency Limit (Hz)', 'def_val': 20, 'min_val': 0},
            'exc_type': {
                'type': 'L', 'text': 'Cell Exclusion Type', 'list': exc_type, 'def_val': exc_type[0],
                'link_para': [['lo_freq_lim', [exc_type[0], exc_type[2]]], ['hi_freq_lim', [exc_type[0], exc_type[1]]]]
            },

            'use_comp': {'type': 'B', 'text': 'Enforce auROC Complimentary Values', 'def_val': False},
            # 'mean_type': {'type': 'L', 'text': 'Signal Mean Type', 'list': mean_type, 'def_val': mean_type[0]},
            # 'k_grp_type': {'type': 'L', 'text': 'Cell Group Type', 'list': k_grp_type, 'def_val': k_grp_type[0]},
            'plot_err': {'type': 'B', 'text': 'Plot auROC Errorbars', 'def_val': True},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
        }
        self.add_func(type='ROC Analysis',
                      name='Velocity ROC Curves (Whole Experiment)',
                      func='plot_velocity_roc_curves_whole',
                      para=para)

        # ====> Velocity ROC Curves (Pos/Neg Comparison)
        para = {
            # calculation parameters
            'n_boot': {'gtype': 'C', 'text': 'Number bootstrapping shuffles', 'def_val': n_boot_def, 'min_val': 100},
            'auc_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'AUC Significance Test', 'list': auc_stype, 'def_val': auc_stype[0],
                'link_para': ['n_boot', 'Delong']
            },
            'vel_bin': {
                'gtype': 'C', 'type': 'L', 'text': 'Velocity Bin Size (deg/sec)', 'list': roc_vel_bin, 'def_val': '5',
            },
            'n_sample': {'gtype': 'C', 'text': 'Equal Timebin Resampling Count', 'def_val': 100},
            'equal_time': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Equal Timebins', 'def_val': False,
                'link_para': ['n_sample', False]
            },
            'pn_calc': {'gtype': 'C', 'text': 'Use Pos/Neg', 'def_val': True, 'is_visible': False},

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True]
            },

            'lo_freq_lim': {'text': 'Low Frequency Limit (Hz)', 'def_val': 1, 'min_val': 0},
            'hi_freq_lim': {'text': 'High Frequency Limit (Hz)', 'def_val': 20, 'min_val': 0},
            'exc_type': {
                'type': 'L', 'text': 'Cell Exclusion Type', 'list': exc_type, 'def_val': exc_type[0],
                'link_para': [['lo_freq_lim', [exc_type[0], exc_type[2]]], ['hi_freq_lim', [exc_type[0], exc_type[1]]]]
            },

            'use_comp': {'type': 'B', 'text': 'Enforce auROC Complimentary Values', 'def_val': False},
            # 'mean_type': {'type': 'L', 'text': 'Signal Mean Type', 'list': mean_type, 'def_val': mean_type[0]},
            'plot_err': {'type': 'B', 'text': 'Plot auROC Errorbars', 'def_val': True},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
        }
        self.add_func(type='ROC Analysis',
                      name='Velocity ROC Curves (Pos/Neg Comparison)',
                      func='plot_velocity_roc_pos_neg',
                      para=para)

        # ====> Condition ROC Curve Comparison
        para = {
            # calculation parameters
            'n_boot': {'gtype': 'C', 'text': 'Number bootstrapping shuffles', 'def_val': n_boot_def, 'min_val': 100},
            'grp_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'Cell Grouping Significance Test', 'list': grp_stype,
                'def_val': grp_stype[0], 'link_para': [['n_boot', 'Delong'], ['n_boot', 'Wilcoxon Paired Test']]
            },
            'auc_stype': {'gtype': 'C', 'type': 'L', 'text': 'AUC Significance Test', 'list': auc_stype,
                         'def_val': auc_stype[0], 'link_para': ['n_boot', 'Delong']},

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter,
                'para_gui_var': {'rmv_fields': ['t_type']}, 'def_val': rot_filt_all
            },
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {'type': 'B', 'text': 'Analyse All Experiments',
                              'def_val': True, 'link_para': ['plot_exp_name', True]},
            'plot_cond': {'type': 'L', 'text': 'Plot Conditions', 'list': p_cond, 'def_val': 'Uniform'},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1],
                'is_visible': False
            },
        }
        self.add_func(type='ROC Analysis',
                      name='Condition ROC Curve Comparison',
                      func='plot_roc_cond_comparison',
                      para=para)

        # ====> Motion/Direction Selectivity Cell Grouping Scatterplot
        para = {
            # calculation parameters
            'n_boot': {'gtype': 'C', 'text': 'Number bootstrapping shuffles', 'def_val': n_boot_def, 'min_val': 100},
            'grp_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'Cell Grouping Significance Test', 'list': grp_stype,
                'def_val': grp_stype[0], 'link_para': [['n_boot', 'Delong'], ['n_boot', 'Wilcoxon Paired Test']]
            },
            'auc_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'AUC Significance Test', 'list': auc_stype, 'def_val': auc_stype[0],
                'link_para': ['n_boot', 'Delong']
            },
            't_phase': {
                'gtype': 'C', 'text': 'UniformDrifting Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10,
                'is_enabled': has_ud_expt
            },
            't_ofs': {
                'gtype': 'C', 'text': 'UniformDriftings Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00,
                'is_enabled': has_ud_expt
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter,
                'para_gui_var': {'rmv_fields': ['t_type']}, 'def_val': rot_filt_all
            },
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True]
            },
            'plot_cond': {'type': 'L', 'text': 'Plot Conditions', 'list': p_cond, 'def_val': 'Uniform'},
            'show_sig_markers': {'type': 'B', 'text': 'Show Significance Markers', 'def_val': True},
            'use_resp_grp_type': {
                'type': 'B', 'text': 'Use Response Cell Grouping', 'def_val': False, 'is_enabled': has_ud_expt
            },
            'plot_trend': {'type': 'B', 'text': 'Plot Group Trendlines', 'def_val': True},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1],
                'is_visible': False
            },
        }
        self.add_func(type='ROC Analysis',
                      name='Motion/Direction Selectivity Cell Grouping Scatterplot',
                      func='plot_cond_grouping_scatter',
                      para=para)

        ##########################################
        ####    COMBINED ANALYSIS FUNCTIONS   ####
        ##########################################

        # initialisations
        comb_type = ['Direction Selectivity', 'Preferred Direction']

        # ====> Combined Stimuli Statistics
        para = {
            # calculation parameters
            't_phase': {'gtype': 'C', 'text': 'UniformDrifting Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10},
            't_ofs': {'gtype': 'C', 'text': 'UniformDriftings Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00},
            'p_value': {'gtype': 'C', 'text': 'Significance Level', 'def_val': 0.05, 'min_val': 0.00, 'max_val': 0.05},

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Filter Parameters', 'para_gui': RotationFilter, 'def_val': None,
                'para_gui_var': {'rmv_fields': ['t_freq_dir', 't_type', 't_cycle']}
            },
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True]
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
            'plot_type': {'type': 'L', 'text': 'Plot Type', 'list': comb_type, 'def_val': comb_type[0]}
        }
        self.add_func(type='Combined Analysis',
                      name='Rotation/Visual Stimuli Response Statistics',
                      func='plot_combined_stimuli_stats',
                      para=para)



        ##########################################
        ####    SINGLE EXPERIMENT FUNCTIONS   ####
        ##########################################

        # ====> Mean Spike Signal (All Clusters)
        para = {
            # plotting parameters
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1, 'is_list':True},
            'plot_all': {'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para':['i_cluster',True]},
            'exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'Experiments'},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Single Experiment Analysis',
                      name='Mean Spike Signals',
                      func='plot_signal_means',
                      multi_fig=['i_cluster'],
                      para=para)

        # ====> Auto Cross-Correlogram
        para = {
            # plotting parameters
            'exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'Experiments'},
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1, 'is_list':True},
            'plot_all': {'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para':['i_cluster', True]},
            'window_size': {'text': 'Window Size (ms)', 'def_val': 10, 'min_val': 5, 'max_val': 50},
        }
        self.add_func(type='Single Experiment Analysis',
                      name='Auto-Correlogram',
                      func='plot_cluster_auto_ccgram',
                      multi_fig=['i_cluster'],
                      para=para)

        # ====> Cross Cross-Correlogram
        para = {
            # plotting parameters
            'exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'Experiments'},
            'i_ref': {'text': 'Reference Cluster Index', 'def_val': 1, 'min_val': 1},
            'i_comp': {'text': 'Comparison Cluster Indices', 'def_val': 1, 'min_val': 1, 'is_list': True},
            'plot_all': {'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para':['i_comp',True]},
            'plot_type': {'type': 'L', 'text': 'Plot Type', 'def_val': 'bar', 'list': ['bar', 'scatterplot']},
            'window_size': {'text': 'Window Size (ms)', 'def_val': 10, 'min_val': 5, 'max_val': 50},
            'p_lim': {'text': 'Confidence Interval (%)', 'def_val': 99.99, 'min_val': 90.0, 'max_val': 100.0 - 1e-6},
            'f_cutoff': {'text': 'Frequency Cutoff (kHz)', 'def_val': 5, 'min_val': 1},
        }
        self.add_func(type='Single Experiment Analysis',
                      name='Cross-Correlogram',
                      func='plot_cluster_cross_ccgram',
                      multi_fig=['i_comp'],
                      para=para)

        # initialises the parameter dictionary
        self.init_para_dict()

    def add_func(self, type, **kwargs):
        '''

        :param name:
        :param type:
        :param scope:
        :param para:
        :return:
        '''

        # allocates memory for the new type field (if not set)
        if type not in self.details:
            self.details[type] = []

        fcn_name = [x['name'] for x in self.details[type]]
        if kwargs['name'] not in fcn_name:
            self.details[type].append(kwargs)
        else:
            i_func = fcn_name.index(kwargs['name'])
            self.details[type][i_func]['para'] = kwargs['para']

    def init_para_dict(self):
        '''

        :return:
        '''

        # memory allocation
        self.para = {}

        # loops through each function initialising the parameter values
        for g_type in self.details:
            for fcn_para in self.details[g_type]:
                if fcn_para['para'] is not None:
                    # allocates memory for the current function
                    fcn_name = fcn_para['name']
                    self.para[fcn_name] = {}

                    # sets the parameter values for each parameter in the function
                    for p_name in fcn_para['para']:
                        self.para[fcn_name][p_name] = fcn_para['para'][p_name]['def_val']

    ##################################################
    ####    OBJECT CREATION/DELETION FUNCTIONS    ####
    ##################################################

    def init_para_groupbox(self, h_grp_para):
        '''

        :param h_grp_para:
        :return:
        '''

        # initialisations
        tab_name = ['Calculation Parameters', 'Plotting Parameters']

        # sets the group width
        self.grp_wid = h_grp_para.geometry().width() - dX
        self.grp_wid_p = h_grp_para.geometry().width() - 2.5 * dX

        # creates the calculation/plotting parameter
        self.grp_para_calc = cf.create_groupbox(None, QRect(10, 10, self.grp_wid_p, 51), grp_font_sub2,
                                                "", "calc_para")
        self.grp_para_plot = cf.create_groupbox(None, QRect(10, 10, self.grp_wid_p, 51), grp_font_sub2,
                                                "", "plot_para")

        # initialises the groupbox layout
        self.grp_para_calc.setLayout(QFormLayout())
        self.grp_para_plot.setLayout(QFormLayout())

        # creates the tab object
        self.grp_para_tabs = cf.create_tab(h_grp_para, QRect(5, 55, self.grp_wid, 291), None,
                                           h_tabchild=[self.grp_para_calc, self.grp_para_plot], child_name=tab_name)

        # # hides the groupboxes
        # self.grp_para_calc.hide()
        # self.grp_para_plot.hide()

    def create_para_objects(self, sel_func):
        '''

        :return:
        '''

        # updates the parameters into the global parameter dictionary
        if self.curr_para is not None:
            self.para[self.curr_fcn] = self.curr_para

        # resets the parameter dictionary
        self.curr_fcn = sel_func

        # retrieves the parameters for the currently selected function
        fcn_para = next(x['para'] for x in self.details[self.type] if x['name'] == sel_func)
        if fcn_para is None:
            # if the current function has no parameters, then exit the function
            self.curr_para = None
            self.grp_para_calc.hide()
            self.grp_para_plot.hide()
            return
        else:
            # otherwise, expand the parameters
            self.curr_para = self.para[self.curr_fcn]
            for p in fcn_para:
                fcn_para[p] = self.set_missing_para_field(fcn_para[p])

        # determines if there are any calculation parameters for the current function
        if len(cf.get_para_dict(fcn_para, 'C')):
            # if so, then enable the calculation tab
            self.grp_para_tabs.setTabEnabled(0, True)
        else:
            # otherwise, set the plotting parameter tab as being visible (while disabling the calculation para tab)
            self.grp_para_tabs.setTabEnabled(0, False)
            if self.grp_para_tabs.currentIndex() == 0:
                self.grp_para_tabs.setCurrentWidget(self.grp_para_tabs.findChild(QWidget, 'plot_para'))

        # creates the calculation/plotting parameter objects
        is_calc_sel = self.grp_para_tabs.tabText(self.grp_para_tabs.currentIndex()) == 'Calculation Parameters'
        self.create_group_para(self.grp_para_calc, fcn_para, 'Calculation Parameters', is_calc_sel)
        self.create_group_para(self.grp_para_plot, fcn_para, 'Plotting Parameters', not is_calc_sel)

    def create_group_para(self, h_grp, fcn_para, gtext, is_sel):
        '''

        :param type:
        :return:
        '''

        # initially hides the group
        h_grp.hide()

        # removes all parameters from the layout
        h_layout = h_grp.layout()
        for i_row in range(h_layout.rowCount()):
            h_layout.removeRow(0)

        # filters the parameter for the current group
        grp_para_name = cf.get_para_dict(fcn_para, gtext[0])
        if len(grp_para_name):
            # resets the dimensions of the groupbox
            n_bool = sum([fcn_para[p_name]['type'] == 'B' for p_name in grp_para_name])

            # creates the objects for each of the parameters
            i_count = 0
            for i_para, p_name in enumerate(grp_para_name):
                # if the parameter is a multi-experiment function parameter only, and only single experiments
                # are loaded, then skip creating the parameter
                if (fcn_para[p_name]['is_multi'] and not self.is_multi):
                    if fcn_para[p_name]['type'] == 'B':
                        n_bool -= 1
                    continue

                # sets the top of the object
                i_count += 1

                # creates the parameter object based on the type
                if fcn_para[p_name]['type'] == 'N':
                    self.create_number_para(h_layout, p_name, fcn_para[p_name])
                elif fcn_para[p_name]['type'] == 'B':
                    self.create_boolean_para(h_layout, p_name, fcn_para[p_name])
                elif fcn_para[p_name]['type'] == 'L':
                    self.create_list_para(h_layout, p_name, fcn_para[p_name])
                elif fcn_para[p_name]['type'] == 'CL':
                    self.create_checklist_para(h_layout, p_name, fcn_para[p_name])
                elif fcn_para[p_name]['type'] == 'Sp':
                    self.create_special_para(h_layout, p_name, fcn_para[p_name])

            # shows the
            h_grp.setGeometry(QRect(dX/2, grp_Y0, self.grp_wid_p, 3*dY + i_count*dY_obj - 5*n_bool))
            h_grp.setLayout(h_layout)

            # if the current tab is selected, then show the group
            if is_sel:
                h_grp.show()

    def create_number_para(self, h_layout, p_name, fcn_para):
        '''

        :param h_layout:
        :param p_name:
        :param fcn_para:
        :return:
        '''

        # initialisations
        para_text = '{0}: '.format(fcn_para['text'])
        if isinstance(self.curr_para[p_name], list):
            # case is the number values are a list.
            x = self.curr_para[p_name]
            ind = [0] + list(np.where(np.diff(x) > 1)[0] + 1)

            # combines any continuous numbers by dash-separations
            def_val = ', '.join(
                        ['{}-{}'.format(x[ig[0]], x[ig[-1]]) if len(ig) > 1
                                                else '{}'.format(x[ig[0]]) for ig in cf.setup_index_arr(ind, len(x))]
            )
        else:
            def_val = str(self.curr_para[p_name])

        # creates the label/editbox objects
        h_lbl = cf.create_label(None, txt_font_bold, para_text, align='right')
        h_num = cf.create_edit(None, txt_font, def_val, align='centre', name=p_name)
        h_lbl.setAlignment(Qt.AlignVCenter)

        # sets the callback function
        is_int, min_val, is_list = fcn_para['is_int'], fcn_para['min_val'], fcn_para['is_list']
        cb_fcn = functools.partial(self.update_num_para, h_num, p_name, is_int, min_val, is_list)
        h_num.editingFinished.connect(cb_fcn)

        # if the visiblity flag is set to false, then hide the objects
        if not fcn_para['is_visible']:
            h_lbl.setVisible(False)
            h_num.setVisible(False)
        elif not fcn_para['is_enabled']:
            h_lbl.setEnabled(False)
            h_num.setEnabled(False)

        # adds the widgets to the layout
        h_layout.addRow(h_lbl, h_num)

    def create_boolean_para(self, h_layout, p_name, fcn_para):
        '''

        :return:
        '''

        # initialisations
        cb_fcn = functools.partial(self.update_bool_para, p_name, fcn_para['link_para'])

        # creates the object
        h_chk = cf.create_checkbox(None, txt_font_bold, fcn_para['text'], name=p_name, cb_fcn=cb_fcn)

        # connects the callback function to the checkbox
        h_chk.setChecked(self.curr_para[p_name])
        if fcn_para['link_para'] is not None:
            self.update_bool_para(p_name, fcn_para['link_para'], self.curr_para[p_name])

        # if the visiblity flag is set to false, then hide the objects
        if not fcn_para['is_visible']:
            h_chk.setVisible(False)
        elif not fcn_para['is_enabled']:
            h_chk.setEnabled(False)

        # adds the object to the layout object
        h_layout.addRow(h_chk)

    def create_special_para(self, h_layout, p_name, fcn_para):
        '''

        :param h_layout:
        :param p_name:
        :param fcn_para:
        :param y0:
        :return:
        '''

        # creates the wrapper button
        cb_fcn = functools.partial(self.update_special_para, p_name, fcn_para)
        h_but = cf.create_button(None, None, txt_font_bold, '*** {0} ***'.format(fcn_para['text']), cb_fcn=cb_fcn)
        h_but.setStyleSheet("background-color: red")

        # adds the widgets to the layout
        h_layout.addRow(h_but)

    def create_list_para(self, h_layout, p_name, fcn_para):
        '''

        :return:0
        '''

        # initialisations
        i_ind, recheck_list = 0, False
        para_text = '{0}: '.format(fcn_para['text'])
        link_para, para_reset, list_txt = fcn_para['link_para'], fcn_para['para_reset'], fcn_para['list']

        # resets the list text if a special type
        if list_txt == 'Experiments':
            # case is the experiment names
            list_txt = self.exp_name
        elif list_txt == 'RotationExperiments':
            # case is the rotation experiment names
            is_rot_expt = cf.det_valid_rotation_expt(self.get_data_fcn())
            list_txt = [x for x, y in zip(self.exp_name, is_rot_expt) if y]
        elif list_txt == 'RotationExperimentMD':
            # case is the rotation experiment names
            t_type = ['Black', 'Uniform', 'LandmarkLeft', 'LandmarkRight']
            is_rot_expt = cf.det_valid_rotation_expt(self.get_data_fcn(), t_type=t_type)
            list_txt = [x for x, y in zip(self.exp_name, is_rot_expt) if y]
        else:
            recheck_list = True

        # creates the callback function
        cb_fcn = functools.partial(self.update_list_para, p_name, list_txt, link_para, para_reset, recheck_list)

        # creates the object
        h_lbl = cf.create_label(None, txt_font_bold, para_text, align='right')
        h_list = cf.create_combobox(None, txt_font, list_txt, name=p_name, cb_fcn=cb_fcn)
        h_lbl.setAlignment(Qt.AlignVCenter)

        # sets the callback function
        if self.curr_para[p_name] is None:
            h_list.setCurrentIndex(0)
            self.curr_para[p_name] = list_txt[0]
        else:
            try:
                i_ind = list_txt.index(self.curr_para[p_name])
                h_list.setCurrentIndex(i_ind)
            except:
                h_list.setCurrentIndex(0)
                self.curr_para[p_name] = list_txt[0]

        if link_para is not None:
            self.update_list_para(p_name, list_txt, link_para, None, True, index=i_ind)

        # if the visiblity flag is set to false, then hide the objects
        if not fcn_para['is_visible']:
            h_lbl.setVisible(False)
            h_list.setVisible(False)
        elif not fcn_para['is_enabled']:
            h_lbl.setEnabled(False)
            h_list.setEnabled(False)

        # adds the widgets to the layout
        h_layout.addRow(h_lbl, h_list)

    def create_checklist_para(self, h_layout, p_name, fcn_para):
        '''

        :param h_layout:
        :param p_name:
        :param fcn_para:
        :return:
        '''

        # initialisations
        i_ind = 0
        para_text = '{0}: '.format(fcn_para['text'])
        link_para, list_txt, def_val = fcn_para['link_para'], fcn_para['list'], fcn_para['def_val']

        # creates the callback function
        cb_fcn = functools.partial(self.update_checklist_para, p_name, h_chklist=None)

        # creates the object
        h_lbl = cf.create_label(None, txt_font_bold, para_text, align='right')
        h_chklist = cf.create_checkcombo(None, txt_font, list_txt, name=p_name, cb_fcn=cb_fcn,
                                         first_line='--- Select Trial Conditions ---')
        h_lbl.setAlignment(Qt.AlignVCenter)

        # sets the callback function
        if def_val is None:
            def_val = np.ones(h_chklist.count(), dtype=bool)

        # sets the initial values
        for index in np.where(def_val)[0]:
            h_chklist.setState(index+1, True)

        # if the visiblity flag is set to false, then hide the objects
        if not fcn_para['is_visible']:
            h_lbl.setVisible(False)
            h_chklist.setVisible(False)
        elif not fcn_para['is_enabled']:
            h_lbl.setEnabled(False)
            h_chklist.setEnabled(False)

        # adds the widgets to the layout
        self.update_checklist_para(p_name, h_chklist)
        h_layout.addRow(h_lbl, h_chklist)

    ############################################
    ####    PARAMETER CALLBACK FUNCTIONS    ####
    ############################################

    def update_special_para(self, p_name, fcn_para):
        '''

        :return:
        '''

        #
        if 'para_gui_var' not in fcn_para:
            para_gui_var = None
        else:
            para_gui_var = fcn_para['para_gui_var']

        # runs the
        data, init_data = self.get_data_fcn(), self.curr_para[p_name]
        h_sp = fcn_para['para_gui'](self, init_data=init_data, other_var=para_gui_var)

        # determines if the gui was updated correctly
        exp_info = h_sp.get_info()
        if h_sp.is_ok:
            # updates the current parameter value
            self.curr_para[p_name] = exp_info

        #
        if h_sp.update_plot:
            self.update_plot()

    def update_num_para(self, h_num, p_name, is_int, min_val, is_list):
        '''

        :param h_num:
        :param p_name:
        :param is_int:
        :param is_list:
        :return:
        '''

        #
        if self.is_updating:
            return
        else:
            self.is_updating = True

        # retrieves the string from the editbox
        nw_str = h_num.text()

        # determines if the new value is valid
        if is_list:
            # case is the new value can be a list of numbers
            if len(re.findall('[^0-9,\.\- ]', nw_str)):
                # if there were any invalid characters, then revert back to the previous value
                cf.show_error('Entered string is not a valid number list.','Error!')
                h_num.setText(str(self.curr_para[p_name]))
            else:
                # if any of the strings
                list_tmp = cf.flat_list([cf.expand_dash_number(x.strip()) for x in nw_str.split(',')])
                list_str = np.array([cf.check_edit_num(x.strip(), is_int, min_val=min_val, show_err=False)
                                                                                    for x in list_tmp])
                if np.any([x is None for x in list_str[:, 0]]):
                    # if there was an error, then revert back to the previous value
                    cf.show_error('Entered string is either not a valid number list or has values ' \
                                  'less that {0}.'.format(min_val),'Error!')
                    h_num.setText(str(self.curr_para[p_name]))
                else:
                    # sets the list from the inputted values
                    nw_list = list(list_str[:, 0])
                    if np.any([x is not None for x in list_str[:, 1]]):
                        # if there were any warnings, then reset the string
                        h_num.setText(', '.join([str(x) for x in nw_list]))

                    # updates the parameter list
                    self.curr_para[p_name] = nw_list
        else:
            # case is the new value can only be a single number
            nw_val, e_str = cf.check_edit_num(nw_str, is_int, min_val=min_val)
            if nw_val is None:
                # if there was an error, then revert back to the previous value
                h_num.setText(str(self.curr_para[p_name]))
            else:
                # otherwise, update the parameter value
                self.curr_para[p_name] = nw_val
                if e_str is not None:
                    h_num.setText(str(nw_val))

        # resets the update flag so another change can be made
        self.is_updating = False

    def update_bool_para(self, p_name, link_para, state):
        '''

        :return:
        '''

        # sets the enabled properties for the linked parameters (if they exist)
        if link_para is not None:
            # ensures the link parameters are a list of lists
            if not isinstance(link_para[0], list):
                link_para = [link_para]

            for lp in link_para:
                h_obj = self.find_obj_handle([QLineEdit, QCheckBox, QComboBox], lp[0])
                h_obj[0].setEnabled(bool(state) != lp[1])

        # updates the parameter value
        self.curr_para[p_name] = bool(state)

    def update_list_para(self, p_name, p_list, link_para, para_reset, recheck_list, index):
        '''

        :return:
        '''

        if self.is_updating:
            return

        if recheck_list:
            d_grp = self.details[self.get_plot_grp_fcn()]
            i_grp = next(i for i in range(len(d_grp)) if d_grp[i]['name'] == self.get_plot_fcn())
            p_list = d_grp[i_grp]['para'][p_name]['list']

        # sets the enabled properties for the linked parameters (if they exist)
        if link_para is not None:
            # ensures the link parameters are a list of lists
            if not isinstance(link_para[0], list):
                link_para = [link_para]

            for lp in link_para:
                h_obj = self.find_obj_handle([QLineEdit, QCheckBox, QComboBox], lp[0])
                if isinstance(lp[1], list):
                    h_obj[0].setEnabled(p_list[index] not in lp[1])
                else:
                    h_obj[0].setEnabled(p_list[index] != lp[1])

        # resets the parameters based on the
        if para_reset is not None:
            for pr in para_reset:
                pr[1](pr[0], p_list[index])

        # updates the parameter value
        self.curr_para[p_name] = p_list[index]

    def update_checklist_para(self, p_name, h_chklist=None):
        '''

        :param p_name:
        :param list_txt:
        :param link_para:
        :return:
        '''

        # retrieves the checkbox list handle (if not provided)
        if h_chklist is None:
            h_chklist = self.grp_para_plot.findChild(cf.CheckableComboBox, p_name)

        # retrieves the currently selected items
        self.curr_para[p_name] = h_chklist.getSelectedItems()

    #########################################
    ####    PARAMETER RESET FUNCTIONS    ####
    #########################################

    def reset_vel_rng(self, p_name, dv0):
        '''

        :param p_name:
        :param dv:
        :return:
        '''

        # flag that the parameters are updating
        self.is_updating = True

        # parameters
        v_rng, dv = 80, int(dv0)
        if not isinstance(p_name, list):
            p_name = [p_name]

        # sets the new velocity range list values
        vc_rng = ['{0} to {1}'.format(i * dv - v_rng, (i + 1) * dv - v_rng) for i in range(int(2 * v_rng / dv))]

        # resets the list values for each associated parameter
        for pp in p_name:
            # retrieves the list object
            h_list = self.find_obj_handle([QComboBox], pp)
            if len(h_list):
                # removes the existing items
                for i in range(h_list[0].count()):
                    h_list[0].removeItem(0)

                # adds the new range
                for txt in vc_rng:
                    h_list[0].addItem(txt)

                # resets the associated parameter value
                self.curr_para[pp] = vc_rng[h_list[0].currentIndex()]

        # flag that the parameters are updating
        self.is_updating = False

    def reset_spd_rng(self, p_name, dv0):
        '''

        :param p_name:
        :param dv:
        :return:
        '''

        # flag that the parameters are updating
        self.is_updating = True
        d_grp = self.details[self.get_plot_grp_fcn()]
        i_grp = next(i for i in range(len(d_grp)) if d_grp[i]['name'] == self.get_plot_fcn())

        # parameters
        v_rng, dv = 80, int(dv0)
        if not isinstance(p_name, list):
            p_name = [p_name]

        # sets the new velocity range list values
        sc_rng = ['{0} to {1}'.format(i * dv, (i + 1) * dv) for i in range(int(v_rng / dv))]

        # resets the list values for each associated parameter
        for pp in p_name:
            # retrieves the list object
            h_list = self.find_obj_handle([QComboBox], pp)
            if len(h_list):
                # removes the existing items
                for i in range(h_list[0].count()):
                    h_list[0].removeItem(0)

                # adds the new range
                for txt in sc_rng:
                    h_list[0].addItem(txt)

                # resets the associated parameter value
                self.curr_para[pp] = sc_rng[h_list[0].currentIndex()]
                d_grp[i_grp]['para'][pp]['list'] = sc_rng

            # self.get_plot_grp_fcn = main_obj.get_plot_group
            # self.get_plot_fcn = main_obj.get_plot_func

        # flag that the parameters are updating
        self.is_updating = False

    #######################################
    ####    MISCELLANEOUS FUNCTIONS    ####
    #######################################

    def get_func_names(self, type):
        '''

        :param analy_type:
        :return:
        '''

        # updates the analysis type
        self.type = type

        # returns the function names that are within the scope of the analysis
        return [x['name'] for x in self.details[self.type]]

    def set_func_name(self, curr_fcn):
        '''

        :param fcn_name:
        :return:
        '''

        self.curr_fcn = curr_fcn

    @staticmethod
    def set_missing_para_field(para):
        '''

        :param para:
        :return:
        '''

        # sets the group type flag
        if 'gtype' not in para:
            para['gtype'] = 'P'

        # sets the object type flag
        if 'type' not in para:
            para['type'] = 'N'

        # sets the link type flag
        if 'link_para' not in para:
            para['link_para'] = None

        # sets the link type flag
        if 'is_list' not in para:
            para['is_list'] = False

        # sets the link type flag
        if 'is_int' not in para:
            para['is_int'] = True

        # sets the link type flag
        if 'min_val' not in para:
            para['min_val'] = 1

        # sets the object type flag
        if 'is_multi' not in para:
            para['is_multi'] = False

        # sets the object type flag
        if 'is_enabled' not in para:
            para['is_enabled'] = True

        # sets the object type flag
        if 'is_visible' not in para:
            para['is_visible'] = True

        # sets the object type flag
        if 'para_reset' not in para:
            para['para_reset'] = None

        # returns the parameter dictionary
        return para

    def get_group_offset(self, h_grp):
        '''

        :param h_grp:
        :return:
        '''

        # determines if the groups above exists
        if h_grp.isHidden():
            # if not, then return zero offset
            return 0
        else:
            # otherwise, determine the height of the group
            return dY + (h_grp.geometry().height() - 1)

    def set_exp_name(self, exp_name, is_multi):
        '''

        :param exp_name:
        :return:
        '''

        self.exp_name = exp_name
        self.is_multi = is_multi

    def get_plot_scope(self):
        '''

        :return:
        '''

        # retrieves the scope listbox object handle
        h_scope = self.grp_para_plot.findChild(QComboBox, name='plot_scope')
        if h_scope is None:
            # if it doesn't exist, then return the default value
            return 'Individual Cell'
        else:
            # otherwise, return the current scope
            return h_scope.currentText()

    def find_obj_handle(self, QType, name):
        '''

        :param QType:
        :param name:
        :return:
        '''

        # memory allocation
        h_obj = []

        # finds the object in both the plot/calculation parameters for all widget types
        for qt in QType:
            h_obj += self.grp_para_calc.findChildren(qt, name=name)
            h_obj += self.grp_para_plot.findChildren(qt, name=name)

        # returns the final object
        return h_obj

########################################################################################################################
########################################################################################################################

class AnalysisData(object):
    def __init__(self):
        # field initialisation
        self._cluster = []
        self.cluster = None
        self.comp = ComparisonData()
        self.classify = ClassifyData()
        self.rotation = RotationData()

        self.req_update = True
        self.exc_gen_filt = None

    def update_gen_filter(self):
        '''

        :param c_data:
        :return:
        '''

        # initialises the rotation filter (if not already done so)
        if self.exc_gen_filt is None:
            self.exc_gen_filt = cf.init_general_filter_data()

class ComparisonData(object):
    def __init__(self):
        # initialisation
        self.is_set = False

    def init_comparison_data(self, ind, n_fix, n_free, n_pts, fix_name, free_name):
        '''

        :param ind:
        :param n_fix:
        :param n_pts:
        :return:
        '''

        # initialisations
        self.is_set = True
        self.ind = ind
        self.n_pts = n_pts
        self.fix_name = fix_name
        self.free_name = free_name

        # memory allocation
        self.mu_dist = None
        self.is_accept = np.zeros(n_fix, dtype=bool)
        self.i_match = -np.ones(n_fix, dtype=int)
        self.is_accept_old = np.zeros(n_fix, dtype=bool)
        self.i_match_old = -np.ones(n_fix, dtype=int)
        self.i_dtw = np.empty((n_fix, n_free), dtype=object)

        # old discrimination metrics
        self.sig_corr_old = -np.ones(n_fix, dtype=float)
        self.sig_diff_old = -np.ones(n_fix, dtype=float)
        self.d_depth_old = -np.ones(n_fix, dtype=float)
        self.z_score = np.zeros((n_pts, n_fix))

        # new discrimination metrics
        self.match_intersect = np.zeros((n_pts, n_fix))
        self.match_wasserstain = np.zeros((n_pts, n_fix))
        self.match_bhattacharyya = np.zeros((n_pts, n_fix))
        self.d_depth = -np.ones(n_fix, dtype=float)
        self.dtw_scale = -np.ones(n_fix, dtype=float)
        self.sig_corr = -np.ones(n_fix, dtype=float)
        self.sig_diff = -np.ones(n_fix, dtype=float)
        self.sig_intersect = -np.ones(n_fix, dtype=float)
        self.isi_corr = -np.ones(n_fix, dtype=float)
        self.isi_intersect = -np.ones(n_fix, dtype=float)
        # self.isi_wasserstein = -np.ones(n_fix, dtype=float)
        # self.isi_bhattacharyya = -np.ones(n_fix, dtype=float)
        self.signal_feat = -np.ones((n_fix, 4), dtype=float)
        self.total_metrics = -np.ones((n_fix,3), dtype=float)
        self.total_metrics_mean = -np.ones(n_fix, dtype=float)

class ClassifyData(object):
    def __init__(self):
        # initialisation
        self.is_set = False

    def init_classify_fields(self, expt_name, clust_id):
        '''

        :param n_expt:
        :return:
        '''

        # initialisation
        self.is_set = False
        self.class_set = False
        self.action_set = False

        # memory allocation
        n_expt = len(expt_name)
        A = np.empty(n_expt, dtype=object)

        # sets the experiment names
        self.expt_name = expt_name
        self.clust_id = clust_id

        # initialises the cluster classification data arrays
        self.all_metrics = None
        self.class_para = dcopy(A)
        self.x_clust = dcopy(A)
        self.grp_str = dcopy(A)

        # initialises the cluster action data arrays
        self.action_para = dcopy(A)
        self.c_type = dcopy(A)
        self.t_dur = dcopy(A)
        self.t_event = dcopy(A)
        self.ci_lo = dcopy(A)
        self.ci_hi = dcopy(A)
        self.ccG_T = dcopy(A)
        self.is_acc = dcopy(A)
        self.act_type = dcopy(A)

    def set_classification_data(self, data, class_para, expt_id, x_clust, grp_str, all_metrics):
        '''

        :param x_clust:
        :param grp_idx:
        :param cluster_met:
        :param expt_name:
        :return:
        '''

        # updates the initialisation flag
        self.is_set = True
        self.class_set = True
        self.all_metrics = all_metrics

        # sets the classification data for all field relating to the experiments that were analysed
        for ex_name in np.unique(expt_id):
            # retrieves the experiment index
            i_expt = self.expt_name.index(ex_name)
            ii = expt_id == ex_name

            #
            cl_ind = cfcn.get_inclusion_filt_indices(data._cluster[i_expt], data.exc_gen_filt)
            self.x_clust[i_expt] = -np.ones((len(cl_ind), np.shape(x_clust)[1]))
            self.grp_str[i_expt] = np.array(['N/A'] * len(cl_ind))

            # sets the values into the classification object
            self.class_para[i_expt] = dcopy(class_para)
            self.x_clust[i_expt][cl_ind, :] = x_clust[ii, :]
            self.grp_str[i_expt][cl_ind] = grp_str[ii]

    def set_action_data(self, action_para, c_type, t_dur, t_event, ci_lo, ci_hi, ccG_T, i_expt, act_type):
        '''

        :param c_type:
        :param t_dur:
        :param t_event:
        :param i_expt:
        :return:
        '''

        # updates the initialisation flag
        self.is_set = True
        self.action_set = True

        # sets the action data fields within the classification class object
        for i, i_ex in enumerate(i_expt):
            # sets the main values
            self.action_para[i_ex] = dcopy(action_para)
            self.c_type[i_ex] = c_type[i]
            self.t_dur[i_ex] = t_dur[i]
            self.t_event[i_ex] = t_event[i]

            # signal arrays
            self.ci_lo[i_ex] = ci_lo[i]
            self.ci_hi[i_ex] = ci_hi[i]
            self.ccG_T[i_ex] = ccG_T[i]
            self.act_type[i_ex] = act_type[i]

            # sets the acceptance flags for each action type
            self.is_acc[i_ex] = [np.ones(len(x), dtype=bool) if i < 2 else
                                 np.zeros(len(x), dtype=bool) for i, x in enumerate(t_dur[i])]

class RotationData(object):
    def __init__(self):
        # initialisation
        self.is_set = False
        self.exc_rot_filt = None
        self.exc_ud_filt = None
        # self.init_rot_fields()

    def init_rot_fields(self):
        '''

        :return:
        '''

        # general initialisations
        self.n_boot_phase_grp = -1
        self.n_boot_cond_grp = -1
        self.n_boot_cond_ci = -1
        self.r_obj_black = None                 # black trial type rotation filter object
        self.r_obj_cond = None                  # condition rotation filter objects

        # phase cell group roc parameters
        self.phase_roc = None                   # phase r roc objects
        self.phase_roc_xy = None                # phase roc curve x/y coordinates
        self.phase_roc_auc = None               # phase roc curve integrals
        self.phase_ci_lo = None                 # phase lower confidence interval
        self.phase_ci_hi = None                 # phase higher confidence interval
        self.phase_gtype = None                 # phase cell group type
        self.phase_auc_sig = None               # phase auc significance flags
        self.phase_grp_stats_type = None        # phase cell grouping statistics type

        # condition cell group roc parameters
        self.cond_roc = None                    # condition r roc objects
        self.cond_roc_xy = None                 # condition roc curve x/y coordinates
        self.cond_roc_auc = None                # condition roc curve integrals
        self.cond_ci_lo = None                  # condition lower confidence interval
        self.cond_ci_hi = None                  # condition higher confidence interval
        self.cond_gtype = None                  # condition cell group type
        self.cond_auc_sig = None                # condition auc significance flags
        self.cond_i_expt = None                 # condition cell experiment index
        self.cond_cl_id = None                  # condition cell cluster index
        self.cond_grp_stats_type = None         # condition cell grouping statistics type
        self.cond_auc_stats_type = None         # condition auc statistics type

        # direction selection group type parameters
        self.r_obj_rot_ds = None                #
        self.ds_gtype = None
        self.ds_gtype_pr = None
        self.ds_gtype_N = None
        self.pd_type = None
        self.pd_type_pr = None
        self.pd_type_N = None
        self.ds_p_value = -1

        #
        self.vel_bin = -1                       #
        self.n_boot_cond_kine = -1              # kinemetics bootstrapping
        self.n_boot_kine_ci = -1
        self.kine_auc_stats_type = None         # kinemetics auc statistics type
        self.r_obj_kine = None                  # kinematic rotation filter object
        self.i_bin_spd = None                   #
        self.i_bin_vel = None                   #
        self.comp_spd = -1                      #
        self.n_rs = -1                          #
        self.is_equal_time = False              #
        self.pn_comp = False

        # velocity roc calculations
        self.vel_dt = None                      # velocity bin durations
        self.vel_xi = None                      # velocity bin values
        self.vel_sf = None                      # velocity spike frequencies
        self.vel_sf_rs = None                   # velocity spike frequencies (resampled)
        self.vel_roc = None                     # velocity roc objects
        self.vel_roc_xy = None                  # velocity roc curve x/y coordinates
        self.vel_roc_auc = None                 # velocity roc curve integrals
        self.vel_ci_lo = None                   # velocity roc lower confidence interval
        self.vel_ci_hi = None                   # velocity roc upper confidence interval

        # speed roc calculations
        self.spd_dt = None                      # speed bin durations
        self.spd_xi = None                      # speed bin values
        self.spd_sf = None                      # speed spike frequencies
        self.spd_sf_rs = None                   # speed spike frequencies (resampled)
        self.spd_roc = None                     # speed roc objects
        self.spd_roc_xy = None                  # speed roc curve x/y coordinates
        self.spd_roc_auc = None                 # speed roc curve integrals
        self.spd_ci_lo = None                   # speed roc lower confidence interval
        self.spd_ci_hi = None                   # speed roc upper confidence interval

    def update_rot_filter(self):
        '''

        :param c_data:
        :return:
        '''

        # initialises the rotation filter (if not already done so)
        if self.exc_rot_filt is None:
            self.exc_rot_filt = cf.init_rotation_filter_data(False, is_empty=True)

    def update_ud_filter(self):
        '''

        :param c_data:
        :return:
        '''

        # initialises the uniform drifting filter (if not already done so)
        if self.exc_ud_filt is None:
            self.exc_ud_filt = cf.init_rotation_filter_data(True, is_empty=True)

########################################################################################################################
########################################################################################################################

class OutputData(object):
    def __init__(self, data):
        # REMOVE ME LATER
        pass

########################################################################################################################
########################################################################################################################

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):

        # creates the figure object
        grp_sz = parent.geometry()
        fig = Figure(figsize=((grp_sz.width()-(2*dX+1))/dpi,(grp_sz.height()-(2*dX+1))/dpi), dpi=dpi,
                     tight_layout=True)

        # creates the figure class
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.ax = None
        self.fig = fig

        # rc('text', usetex=True)

    def setup_plot_axis(self, n_plot=None, n_row=None, n_col=None, is_3d=False, is_polar=False, proj_type=None):
        '''

        :return:
        '''

        # sets the plot count and row/column dimensions
        if n_plot is None:
            # if no dimensions are given, then use a single subplot axes
            if (n_row is None) or (n_col is None):
                n_plot, n_row, n_col = 1, 1, 1
            else:
                n_plot = n_row * n_col
        elif (n_row is None) or (n_col is None):
            # otherwise, if the plot count is given but not the dimensions then give some "nice" dimensions
            n_col, n_row = cf.det_subplot_dim(n_plot)

        # creates the subplots
        self.ax = np.empty(n_plot, dtype=object)
        for i_plot in range(n_plot):
            plt.rcParams["axes.edgecolor"] = 'black'
            plt.rcParams["axes.linewidth"] = 1.25

            if proj_type is not None:
                if proj_type[i_plot] is None:
                    self.ax[i_plot] = self.figure.add_subplot(n_row, n_col, i_plot + 1)
                    self.ax[i_plot].set_frame_on(True)
                else:
                    self.ax[i_plot] = self.figure.add_subplot(n_row, n_col, i_plot + 1, projection=proj_type[i_plot])
            else:
                if is_3d:
                    self.ax[i_plot] = self.figure.add_subplot(n_row, n_col, i_plot + 1, projection='3d')
                elif is_polar:
                    self.ax[i_plot] = self.figure.add_subplot(n_row, n_col, i_plot + 1, projection='polar')
                else:
                    self.ax[i_plot] = self.figure.add_subplot(n_row, n_col, i_plot + 1)
                    self.ax[i_plot].set_frame_on(True)

################## IMPORTANT CODE ##################

# DELETEING ALL WIDGETS IN A GROUPBOX
# for hh in self.grp_func.findChildren(QWidget):
#     hh.deleteLater()

# CONVERTING .ui FILES FROM QT DESIGNER
# from PyQt5.uic import *;
# py_file = 'C:\\Work\\EPhys\\Code\\Sepi\\analysis_guis\\main_analysis_tmp.py';
# ui_file = 'C:\\Work\\EPhys\\Code\\Sepi\\analysis_guis\\main_analysis.ui';
# fp = open(py_file, "w");
# compileUi(ui_file, fp);
# fp.close()


# def plot_signal_pair(self, y1, y2, ax=None, col=None):
#     '''
#
#     :param y1:
#     :param y2:
#     :return:
#     '''
#
#     #
#     if col is None:
#         col = 'br'
#
#     #
#     if ax is None:
#         plt.figure()
#         plt.plot(y1, c=col[0])
#         plt.plot(y2, c=col[1])
#     else:
#         ax.plot(y1, c=col[0])
#         ax.plot(y2, c=col[1])

# def reset_axis_lim(self, ax, ax_type='y', min_tick=3, max_tick=4):
#     '''
#
#     :param ax:
#     :param ax_lim:
#     :return:
#     '''
#
#     #
#     dt = [0.0010, 0.0020, 0.0025, 0.005, 0.010, 0.020, 0.025, 0.050, 0.10, 0.20,
#           0.25, 0.50, 1.00, 1.50, 2.00, 2.50, 5.00, 10.0, 20.0, 25.0, 50.0, 100.0]
#
#     # retrieves the limits
#     ax_lim = eval('ax.get_{0}lim()'.format(ax_type))
#     n_tick = np.ceil(np.diff(ax_lim) / dt).astype(int)
#
#     # determines the number of tick labels
#     i_tick = next((i for i in range(len(n_tick)) if ((n_tick[i] >= min_tick) and (n_tick[i] <= max_tick))), None)
#
#     #
#     if i_tick is None:
#         return
#
#     #
#     d_tick = dt[i_tick]
#     ax_lim_nw = np.arange(d_tick * np.floor(ax_lim[0] / d_tick), d_tick * np.ceil(ax_lim[1] / d_tick) + 1, d_tick)
#     eval('ax.{0}axis.set_major_locator(FixedLocator(ax_lim_nw))'.format(ax_type))
#     eval('ax.set_{0}lim([ax_lim_nw[0], ax_lim_nw[-1]])'.format(ax_type))