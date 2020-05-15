# module import
import os
import re
import time
import copy
import warnings
import datetime
import functools
import math as m
import xlsxwriter
import pickle as p
import numpy as np
import pandas as pd
from venn import venn
from random import sample
from numpy.matlib import repmat
from mpldatacursor import datacursor, HighlightingDataCursor

# seaborn module import/initialisation
f_scale = 1.2
import seaborn as sns
sns.set()
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=f_scale)

# matplotlib module imports
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba_array
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Polygon, Patch
from matplotlib.colors import ListedColormap
from matplotlib_venn import venn3, venn2
import matplotlib.style
import matplotlib as mpl
from matplotlib.pyplot import rc

# rpy2 module imports
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects import FloatVector, FactorVector
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()
r_pair = importr("pairwise")
r_stats = importr("stats")
r_pROC = importr("pROC")

# try:
#     r_pHOC = importr("PMCMRplus")
# except:
#     pass

# scipy module imports
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator as pchip
from scipy.spatial import ConvexHull as CHull
from scipy.stats import linregress, bartlett, ks_2samp, kruskal
from scipy.signal import find_peaks

# sklearn module imports
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.mixture import BayesianGaussianMixture as GMM

# pyqt5 module imports
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import (QMainWindow, QWidget, QGroupBox, QLineEdit, QComboBox, QPushButton,
                             QCheckBox, QDialog,  QFormLayout, QMessageBox, QFileDialog)

# custom module import
import analysis_guis.common_func as cf
import analysis_guis.calc_functions as cfcn
import analysis_guis.rotational_analysis as rot
from analysis_guis.dialogs.file_dialog import FileDialogModal
from analysis_guis.dialogs import load_expt, config_dialog, expt_compare
from analysis_guis.dialogs.rotation_filter import RotationFilter, RotationFilteredData
from analysis_guis.dialogs.info_dialog import InfoDialog, ParaFieldDialog
from analysis_guis.dialogs.lda_para import LDASolverPara
from analysis_guis.threads import thread_workers

# other parameters
dX = 10
dY = 10
dY_obj = 25
grp_Y0 = 10
n_plot_max = 25
table_fsize = 12
fig_hght = 891

# font objects
txt_font = cf.create_font_obj()
table_font = cf.create_font_obj(size=12)
table_font_small = cf.create_font_obj(size=10)
table_font_smaller = cf.create_font_obj(size=8)
table_font_smallest = cf.create_font_obj(size=6)
txt_font_bold = cf.create_font_obj(is_bold=True, font_weight=QFont.Bold)
grp_font_sub = cf.create_font_obj(size=10, is_bold=True, font_weight=QFont.Bold)
grp_font_sub2 = cf.create_font_obj(size=9, is_bold=True, font_weight=QFont.Bold)
grp_font_main = cf.create_font_obj(size=12, is_bold=True, font_weight=QFont.Bold)

# general group width sizes
grp_wid = 401
grp_inner = grp_wid - 2 * dX
grp_inner2 = grp_inner - 2 * dX
plot_left = grp_wid + 2 * dX

# lambda function declarations
lin_func = lambda x, a: a * x
ebar_col = lambda x: 'r' if x else 'k'
get_list_fields = lambda comp, c_field: np.concatenate([getattr(x, c_field) for x in comp])
formatter = lambda **kwargs: ', '.join(kwargs['point_label'])
formatter_lbl = lambda **kwargs: kwargs['label']
setup_heatmap_bins = lambda t_stim, dt: np.arange(t_stim + dt / 1000, step=dt / 1000.0)
txt_fcn = lambda l, t: np.any([t in ll for ll in l])
remove_uscore = lambda x: x.replace('_', '').lower()

# other initialisations
dcopy = copy.deepcopy
func_types = np.array(['Cluster Matching', 'Cluster Classification', 'Freely Moving Cell Types', 'Eye Tracking',
                       'Spiking Frequency Correlation', 'Rotation Analysis', 'UniformDrift Analysis', 'ROC Analysis',
                       'Combined Analysis', 'Depth-Based Analysis', 'Direction LDA', 'Speed LDA',
                       'Single Experiment Analysis', 'Miscellaneous Functions'])
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
        cf.set_sns_colour_palette()

        # other initialisations
        self.is_multi = False
        self.calc_ok = True
        self.calc_cancel = False
        self.thread_calc_error = False
        self.can_close = False
        self.initialising = True
        self.analysis_scope = 'Unprocessed'
        self.file_type = -1
        self.func_data = None

        # determines if the default data file has been set
        if os.path.isfile(cf.default_dir_file):
            # if so, then the data from the file
            with open(cf.default_dir_file, 'rb') as fp:
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
        self.resize(1677, 931)
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
        self.grp_info = cf.create_groupbox(self.centralwidget, QRect(10, 10, grp_wid, 96), grp_font_main,
                                           "Experiment Information", "grp_info")

        # creates the header label objects
        self.lbl_expt_count_h = cf.create_label(self.grp_info, txt_font_bold, "Loaded Experiment Count: ",
                                                QRect(10, 30, 181, 20), "lbl_expt_count_h", 'right')
        self.lbl_analy_scope_h = cf.create_label(self.grp_info, txt_font_bold, "Analysis Scope: ",
                                                 QRect(10, 50, 181, 20), "lbl_analy_scope_h", 'right')
        self.lbl_expt_type_h = cf.create_label(self.grp_info, txt_font_bold, "Experiments Type: ",
                                              QRect(10, 70, 181, 20), "lbl_expt_type_h", 'right')

        # creates the text field label objects
        self.lbl_expt_count = cf.create_label(self.grp_info, txt_font, "0", QRect(195, 30, 40, 20), "lbl_expt_count")
        self.lbl_analy_scope = cf.create_label(self.grp_info, txt_font, "N/A", QRect(195, 50, 150, 20), "lbl_analy_scope")
        self.lbl_expt_type = cf.create_label(self.grp_info, txt_font, "N/A", QRect(195, 70, 150, 20), "lbl_expt_type")

        # disables the objects within the group
        self.set_group_enabled_props(self.grp_info, False)

    def init_func_group(self):
        '''

        :return:
        '''

        # creates the groupbox objects
        self.grp_func = cf.create_groupbox(self.centralwidget, QRect(10, 115, grp_wid, 696), grp_font_main,
                                           "Analysis Functions", "grp_func")
        self.grp_scope = cf.create_groupbox(self.grp_func, QRect(10, 30, grp_inner, 55), grp_font_sub,
                                           "Analysis Type", "grp_scope")
        self.grp_funcsel = cf.create_groupbox(self.grp_func, QRect(10, 95, grp_inner, 171), grp_font_sub,
                                           "Function Select", "grp_funcsel")
        self.grp_para = cf.create_groupbox(self.grp_func, QRect(10, 275, grp_inner, 411), grp_font_sub,
                                           "Function Parameters", "grp_para")

        # creates the combobox objectsF
        self.combo_scope = cf.create_combobox(self.grp_scope, txt_font, func_types, QRect(10, 25, grp_inner2, 20),
                                              "combo_scope", self.change_scope)

        # creates the listbox objects
        self.list_funcsel = cf.create_listbox(self.grp_funcsel, QRect(10, 20, grp_inner2, 141), txt_font,
                                              None, "list_funcsel", cb_fcn=self.func_select)

        # creates the pushbutton objects
        self.push_update = cf.create_button(self.grp_para, QRect(10, 25, grp_inner2, 22), txt_font_bold,
                                            "Update Plot Figure", 'push_update', cb_fcn=self.update_click)

        # other initialis/ations and setting object properties
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
        plot_frm_pos = cfcn.get_plot_canvas_pos(plot_left, dY, fig_hght)
        self.grp_plot = cf.create_groupbox(self.centralwidget, plot_frm_pos, None, "", "frm_plot")
        self.reshape_plot_group_dim(reset_plot_frm=False, reset_plot_canvas=False)

        # creates the plot figure object
        self.plot_fig = None

    def reshape_plot_group_dim(self, reset_plot_frm=True, reset_plot_canvas=True):
        '''

        :return:
        '''

        # creates the groupbox objects
        plot_frm_pos, fig_pos = cfcn.get_plot_canvas_pos(plot_left, dY, fig_hght), self.geometry()

        #
        fig_pos.setWidth(plot_frm_pos.left() + plot_frm_pos.width() + dX)
        self.setMinimumSize(0, 0)
        self.setGeometry(fig_pos)
        self.setFixedSize(fig_pos.width(), fig_pos.height())

        # resets the plot frame (if required)
        if reset_plot_frm:
            self.grp_plot.setGeometry(plot_frm_pos)

        #
        try:
            if reset_plot_canvas:
                plot_canvas_pos = self.plot_fig.geometry()
                plot_canvas_pos.setWidth(plot_frm_pos.width() - 2 * dX)
                self.plot_fig.setGeometry(plot_canvas_pos)
        except:
            pass

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
        self.menu_output_data = cf.create_menu(self.menu_file, "Output Datasets", "save_data")
        self.menu_load_general = cf.create_menu(self.menu_file, "Load General File/Directory", "load_general")
        self.menu_default = cf.create_menuitem(self, "Set Default Directories", "menu_default", self.set_default,
                                               s_cut='Ctrl+D')
        self.menu_global_para = cf.create_menuitem(self, "Global Parameters", "global_para", self.update_glob_para,
                                                   s_cut='Ctrl+P')
        self.menu_exit = cf.create_menuitem(self, "Exit Program", "menu_exit", self.exit_program, s_cut='Ctrl+X')

        # adds the menu items to the file menu
        self.menu_file.addAction(self.menu_cluster_data.menuAction())
        self.menu_file.addAction(self.menu_output_data.menuAction())
        self.menu_file.addAction(self.menu_load_general.menuAction())
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.menu_default)
        self.menu_file.addAction(self.menu_global_para)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.menu_exit)

        # disables the output data menu item
        self.menu_output_data.setEnabled(False)
        self.menu_load_general.setEnabled(False)

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

        ###############################################
        ###    GENERAL FILE/DIRECTORY MENU ITEMS    ###
        ###############################################

        # creates the menu items
        self.menu_gen_dir = cf.create_menuitem(self, 'Load Directories', 'load_gen_dir',
                                               self.load_general_dir, s_cut='Ctrl+Q')
        self.menu_gen_file = cf.create_menuitem(self, 'Load Files', 'load_gen_file',
                                                self.load_general_file, s_cut='Ctrl+Z')

        # adds the menu items to the data output menu
        self.menu_load_general.addAction(self.menu_gen_dir)
        self.menu_load_general.addAction(self.menu_gen_file)

        #############################
        ###    DATA MENU ITEMS    ###
        #############################

        # creates the menu items
        self.menu_show_info = cf.create_menuitem(self, 'Show Dataset Information', 'show_info', self.show_info,
                                                 s_cut='Ctrl+I')
        self.menu_alt_fields = cf.create_menuitem(self, 'Alter Parameter Fields', 'alt_fields', self.alt_fields,
                                                 s_cut='Ctrl+A')
        self.menu_init_filt = cf.create_menu(self, "Set Exclusion Filter Fields", "init_rotdata")

        # adds the menu items to the file menu
        self.menu_data.addAction(self.menu_show_info)
        self.menu_data.addAction(self.menu_alt_fields)
        self.menu_data.addSeparator()
        self.menu_data.addAction(self.menu_init_filt.menuAction())

        # disables the output data menu item
        self.menu_data.setEnabled(False)
        self.menu_show_info.setEnabled(False)
        self.menu_alt_fields.setEnabled(False)

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
                  'w_sig_feat': '0.25', 'w_sig_comp': '1.00', 'w_isi': '0.25', 'roc_clvl': '0.99',
                  'lda_trial_type': 'One-Trial Out', 'w_ratio': 1.4}
        def_dir = {'configDir': data_dir, 'inputDir': data_dir, 'dataDir': data_dir, 'figDir': data_dir}

        # sets the final default data dictionary
        def_data = {'dir': def_dir, 'g_para': g_para}

        # writes the data to file
        with open(cf.default_dir_file, 'wb') as fw:
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

        :param iw:
        :param worker_data:
        :param manual_exit:
        :param is_plot:
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

                # initialisations
                init_data, init_comp = True, len(self.data.comp.data) == 0
                has_free_data, has_eyetrack_data = False, False

                # sets the file data type
                _, f_extn = os.path.splitext(self.worker[iw].thread_job_para[0].exp_files[0])
                self.file_type = {'.cdata': 1, '.ccomp': 2, '.mdata': 3, '.mcomp': 4}[f_extn]

                # sets the experiment type
                expt_type = {1: 'Single Experiments', 2: 'Single Comparison Matches',
                             3: 'Multiple Experiments', 4: 'Multiple Comparison Matches'}
                self.lbl_expt_type.setText(expt_type[self.file_type])

                # loops through all the new file names and loads
                if (worker_data is not None) and len(worker_data):
                    for i_d, loaded_data in enumerate(worker_data):
                        if self.file_type == 1:
                            # case is a single data file
                            self.data._cluster.append(loaded_data)

                        elif self.file_type == 2:
                            # case is a single free/fixed comparison data file

                            # sets the fixed/free experiment text labels
                            init_comp = False
                            fix_name = cf.extract_file_name(loaded_data['data'][0]['expFile'])
                            free_name = cf.extract_file_name(loaded_data['data'][1]['expFile'])

                            # appends the new data to the overall cluster/comparison data fields
                            if not self.is_loaded_file(loaded_data['data'][0]['expFile']):
                                self.data._cluster.append(loaded_data['data'][0])

                            if not self.is_loaded_file(loaded_data['data'][1]['expFile']):
                                self.data._cluster.append(loaded_data['data'][1])

                            # determines if the new combination files already exist
                            chk_status, _ = cfcn.check_existing_compare(self.data.comp.data, fix_name, free_name)
                            if chk_status == 0:
                                self.data.comp.data.append(loaded_data['c_data'])

                        elif (self.file_type == 3) or (not isinstance(loaded_data, dict)):
                            # case is a multi-experiment data file
                            names, files = dcopy(self.data.multi.names), dcopy(self.data.multi.files)
                            self.data, init_data = loaded_data, False
                            init_comp = self.file_type == 3

                            if (self.file_type == 4) and hasattr(self.data.externd, 'free_data'):
                                has_free_data = any([len(x)>0 for x in self.data.externd.free_data.cell_type])

                            if hasattr(self.data, 'externd'):
                                has_eyetrack_data = hasattr(self.data.externd, 'eye_track')

                            # initialises the multi file data field (if not provided)
                            if not hasattr(loaded_data, 'multi'):
                                self.data.multi = MultiFileData()
                                self.data.multi.set_multi_file_data(self.worker[iw].thread_job_para[0])

                            # adds in any missing
                            self.data.check_missing_fields()
                            self.data.multi.is_multi, self.data.multi.files, self.data.multi.names = True, names, files

                        elif self.file_type == 4:
                            # case is a multi-experiment comparison data file
                            init_comp = False
                            n_file = len(loaded_data['c_data'])

                            # if the loaded data correlation calculations have been made, then update that field
                            if loaded_data['ff_corr'].is_set:
                                self.data.comp.ff_corr = loaded_data['ff_corr']

                            # if the loaded data correlation calculations have been made, then update that field
                            if 'f_data' in loaded_data:
                                if loaded_data['f_data'] is not None:
                                    self.data.externd.free_data = loaded_data['f_data']
                                    has_free_data = any([len(x) > 0 for x in self.data.externd.free_data.cell_type])

                            for i_file in range(n_file):
                                # retrieves the fixed/free clusters
                                fix_clust, free_clust = loaded_data['data'][i_file, 0], loaded_data['data'][i_file, 1]

                                # retrieves the fixed/free file names (for the given experiment comparison)
                                fix_name = cf.extract_file_name(fix_clust['expFile'])
                                free_name = cf.extract_file_name(free_clust['expFile'])

                                # appends the new data to the overall cluster/comparison data fields
                                if not self.is_loaded_file(fix_clust['expFile']):
                                    self.data._cluster.append(fix_clust)

                                if not self.is_loaded_file(free_clust['expFile']):
                                    self.data._cluster.append(free_clust)

                                # determines if the new combination files already exist
                                chk_status, _ = cfcn.check_existing_compare(self.data.comp.data, fix_name, free_name)
                                if chk_status == 0:
                                    self.data.comp.data.append(loaded_data['c_data'][i_file])

                    if init_data:
                        self.data.exc_rot_filt = cf.init_rotation_filter_data(False, is_empty=True)
                        self.data.exc_ud_filt = cf.init_rotation_filter_data(True, is_empty=True)

                # checks the data fields (if loading single/multi-experiment files only)
                if self.file_type in [1, 3]:
                    self.check_data_fields()

                # sets up the analysis functions and resets the trial type strings
                self.reset_trial_type_strings()
                self.fcn_data.init_all_func()

                # updates the cluster parameters (if loading comparison data file types)
                if self.file_type in [2, 4]:
                    self.fcn_data.reset_cluster_para()

                # sets the analysis function groupbox properties
                self.set_group_enabled_props(self.grp_info, len(self.data._cluster) > 0)
                self.set_group_enabled_props(self.grp_func, len(self.data._cluster) > 0)
                self.lbl_expt_count.setText(str(len(self.data._cluster)))
                self.fcn_data.set_exp_name([cf.extract_file_name(x['expFile']) for x in self.data._cluster],
                                           self.is_multi)

                # sets the enabled properties of the menu items
                self.menu_output_data.setEnabled(True)
                self.menu_load_general.setEnabled(True)
                self.menu_save_file.setEnabled(len(self.data._cluster) > 1 or (self.is_multi))

                # initialises the classification data fields
                exp_name = [cf.extract_file_name(x['expFile']) for x in self.data._cluster]
                clust_id = [x['clustID'] for x in self.data._cluster]

                # re-initialises the data fields for all data types
                if init_data:
                    self.data.classify.init_classify_fields(exp_name, clust_id)
                    self.data.rotation.init_rot_fields()
                    self.data.depth.init_rot_fields()
                    self.data.discrim.init_discrim_fields()

                    if init_comp:
                        self.data.comp.init_comparison_data()

                # updates the comparison flag
                self.data.comp.is_set = self.data.comp.is_set or (self.file_type in [2, 4])

                # enables the menu item
                self.menu_data.setEnabled(True)
                self.menu_show_info.setEnabled(True)
                self.menu_alt_fields.setEnabled(True)

                # adds the force calculation parameter (if not present)
                if not hasattr(self.data, 'force_calc'):
                    self.data.force_calc = False

                # determines which experimental types are available
                has_rot_expt = any(cf.det_valid_rotation_expt(self.data))
                has_vis_expt, has_ud_expt, has_md_expt = cf.det_valid_vis_expt(self.data)
                has_both = has_vis_expt and has_rot_expt


                # if single experiments are loaded, then determine the function types
                new_func_types = dcopy(func_types)
                if not self.is_multi:
                    is_keep = [
                        True,               # Cluster Matching
                        True,               # Cluster Classification
                        False,              # Freely Moving Cell Types
                        False,              # Eye Tracking
                        has_rot_expt,       # Spiking Frequency Correlation
                        has_rot_expt,       # Rotation Analysis
                        has_ud_expt,        # UniformDrift Analysis
                        has_rot_expt,       # ROC Analysis
                        has_both,           # Combined Analysis
                        has_both,           # Depth-Based Analysis
                        has_rot_expt,       # Direction LDA
                        has_rot_expt,       # Speed LDA
                        True,               # Single Experiment Analysis
                        has_rot_expt        # Miscellaneous Functions
                    ]
                    new_func_types = func_types[np.array(is_keep)]
                else:
                    # if the free-data is not set, then the function types from the list
                    if not has_free_data:
                        new_func_types = new_func_types[new_func_types != 'Freely Moving Cell Types']

                    # if the eye-tracking data is not set, then the function types from the list
                    if not has_eyetrack_data:
                        new_func_types = new_func_types[new_func_types != 'Eye Tracking']

                # ensures any missing fields are added to the exclusion filter
                if self.data.exc_gen_filt is not None:
                    f_fld = ['lesion', 'record_state']
                    for ff in f_fld:
                        if ff not in self.data.exc_gen_filt:
                            self.data.exc_gen_filt[ff] = []

                # otherwise, enable the cluster matching comparison menu item
                self.menu_set_compare.setEnabled(self.file_type == 1)
                self.menu_init_filt.setEnabled(has_rot_expt)
                self.menu_gen_filt.setEnabled(True)
                self.menu_rot_filt.setEnabled(has_rot_expt)
                self.menu_ud_filt.setEnabled(has_ud_expt)

                # updates the general exclusion filter
                self.data.update_gen_filter()

                # updates the exclusion rotation filter (if any experiments contain rotational data)
                if has_rot_expt:
                    self.data.update_rot_filter()

                # updates the exclusion uniformdrifting filter (if any experiments contain uniformdrifting data)
                if has_ud_expt:
                    self.data.update_ud_filter()

                if self.combo_scope.count() != len(new_func_types):
                    # sets the flag which disables the function type callback function
                    self.initialising = True

                    # re-adds all the items
                    self.combo_scope.clear()
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

            elif self.worker[iw].thread_job_primary == 'run_calc_func':
                # case is the calculation functions

                # if the calculation failed, then exit without updating
                if not self.worker[iw].is_ok:
                    self.thread_calc_error = True
                    return

                # retrieves the data from the worker thread
                calc_data = worker_data
                self.thread_calc_error = False
                self.calc_cancel = False
                self.data.force_calc = False

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

                elif self.worker[iw].thread_job_secondary == 'Fixed/Free Cluster Matching':
                    # case is calculating the cluster matches

                    # updates the comparison data struct
                    self.fcn_data.update_comp_expts()
                    self.data.comp.is_set = np.any([x.is_set for x in self.data.comp.data])

                # re-runs the plotting function
                self.update_click()

    def error_thread_job(self, e_str, title):
        '''

        :param e_str:
        :param title:
        :return:
        '''

        cf.show_error(e_str, title)

    def det_data_field_vals(self, chk_flds):
        '''

        :return:
        '''

        # memory allocation
        n_expt, n_flds = len(self.data._cluster), len(chk_flds)
        fld_vals = np.empty((n_expt, n_flds), dtype=object)

        #
        for i_c, c in enumerate(self.data._cluster):
            for i_cfld, cfld in enumerate(chk_flds):
                if cfld in c['expInfo']:
                    if c['expInfo'][cfld] is not None:
                        fld_vals[i_c, i_cfld] = c['expInfo'][cfld]

        # returns the field values
        return fld_vals

    def check_data_fields(self, check_missing_only=True):
        '''

        :return:
        '''

        # initialisations
        chk_flds = ['probe_depth']
        fld_vals = self.det_data_field_vals(chk_flds)
        f_name = np.array([cf.extract_file_name(c['expFile']) for c in self.data._cluster])

        # determines which fields are missing values
        if check_missing_only:
            fld_missing = np.any(fld_vals == None, axis=1)
            is_missing = np.where(fld_missing)[0]

            # is_fix = np.array([c['rotInfo'] is not None for c in self.data._cluster]).reshape(1, -1)
            # is_missing = np.where(np.logical_and(fld_missing, is_fix))[0]

            if len(is_missing):
                # if there are missing parameters, then reduce down to the experiments that are missing values
                fld_vals, f_name, t_str = fld_vals[is_missing, :], f_name[is_missing], 'Missing Parameters'

            else:
                # otherwise, exit the function
                return
        else:
            is_missing, t_str = None, 'Alter Parameters'

        # opens up the config dialog box and retrieves the final file information
        ParaFieldDialog(self,
                        title=t_str,
                        chk_flds=chk_flds,
                        fld_vals=dcopy(fld_vals),
                        f_name=f_name,
                        cl_ind=is_missing)

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

    def reset_trial_type_strings(self):
        '''

        :return:
        '''

        # loops through each of the data clusters
        for ic, c in enumerate(self.data._cluster):
            # if there is no rotation information (i.e, free expt) then continue
            if c['rotInfo'] is None:
                continue

            # determines if there are any trial type names that need to be altered
            tt_0 = list(c['rotInfo']['wfm_para'].keys())
            tt_nw = [[x, y] for x, y in zip(tt_0, [cf.convert_trial_type(x) for x in tt_0]) if x != y]

            # if there are alterations that need to be changed, then
            if len(tt_nw):
                for tt in tt_nw:
                    # replaces the values within the rotation information field
                    c['rotInfo']['trial_type'] = np.where(c['rotInfo']['trial_type'] == tt[0], tt[1],
                                                          c['rotInfo']['trial_type'])
                    c['rotInfo']['t_spike'][tt[1]] = c['rotInfo']['t_spike'].pop(tt[0])
                    c['rotInfo']['wfm_para'][tt[1]] = c['rotInfo']['wfm_para'].pop(tt[0])

                    # determines if the temporary cluster field has been set
                    if self.data.cluster is not None:
                        # if so, then retrieve the temporary cluster field (for the current experiment)
                        cc = self.data.cluster[ic]

                        # replaces the values within the rotation information field
                        cc['rotInfo']['trial_type'] = np.where(cc['rotInfo']['trial_type'] == tt[0], tt[1],
                                                               cc['rotInfo']['trial_type'])
                        cc['rotInfo']['t_spike'][tt[1]] = cc['rotInfo']['t_spike'].pop(tt[0])
                        cc['rotInfo']['wfm_para'][tt[1]] = cc['rotInfo']['wfm_para'].pop(tt[0])

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

        def has_multi_file(multi):
            '''

            :param multi:
            :return:
            '''

            if multi.is_multi:
                return '.mdata' in multi.files[0]
            else:
                return False

        # if data output is currently in progress then output an error an exit the function
        if self.is_thread_running():
            return

        # if the loaded data is not
        load_dlg = load_expt.LoadExpt(data=self.data, def_dir=self.def_data['dir'], f_type=self.file_type)
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
                # case is multi data file
                self.is_multi = True
                self.lbl_analy_scope.setText('Multi-Experiment')
            else:
                # case is single data file(s)
                self.is_multi = False
                self.lbl_analy_scope.setText('Single Experiment')

            # removes any loaded data not in the final selection
            if has_multi_file(self.data.multi):
                # re-initialises the data class
                if load_dlg.exp_files[0] == self.data.multi.files[0]:
                    return
                else:
                    self.data = AnalysisData()
            else:
                for i in reversed(range(len(self.data._cluster))):
                    if cf.extract_file_name(self.data._cluster[i]['expFile']) not in load_dlg.exp_name:
                        self.data._cluster.pop(i)

            if self.file_type == 1:
                for i in reversed(range(len(self.data.comp.data))):
                    if self.data.comp.data[i].fix_name not in load_dlg.exp_name:
                        self.data.comp.data.pop(i)

                self.data.comp.is_set = len(self.data.comp.data) > 0

            # updates the multi-file data struct
            if self.is_multi:
                self.data.multi.set_multi_file_data(load_dlg)
            else:
                self.data.multi.set_multi_file_data(None)

            # starts the worker thread
            iw = self.det_avail_thread_worker()
            self.worker[iw].set_worker_func_type('load_data_files', thread_job_para=[load_dlg, loaded_exp, self.is_multi])
            self.worker[iw].start()

    def load_general_dir(self):
        '''

        :return:
        '''

        # runs the directory open dialog
        file_dlg = FileDialogModal(caption='Select General File Directories To Search',
                                   directory=cfcn.get_dir_para('inputDir'),
                                   dir_only=True)

        # determines if the user selected a valid file
        if (file_dlg.exec() == QDialog.Accepted):
            # otherwise, set the output file name
            input_dir = file_dlg.selectedFiles()
            input_file0 = cf.flat_list([cf.flat_list([
                [os.path.join(x[0], z) for z in x[2]] for x in os.walk(y) if len(x[2])]) for y in input_dir])
        else:
            # if the user cancelled then exit
            return

        # determines the number of unique file extensions from each of the files, and returns the input files that have
        # the most likely filetype extension
        f_extn0, n_extn = np.unique([cf.extract_file_extn(x) for x in input_file0], return_counts=True)
        f_extn = f_extn0[np.argmax(n_extn)]
        input_file = [x for x in input_file0 if f_extn in x]

        if f_extn == '.csv':
            # case is the input data files are csv files

            # reads and extracts the eye-tracking features from each csv file
            for i_f, in_f in enumerate(input_file):
                # updates the progress bar
                w_str = 'Loading File {0} of {1}'.format(i_f + 1, len(input_file))
                self.update_thread_job(w_str, 100. * (i_f + 1) / (len(input_file) + 1))

                if hasattr(self.data.externd, 'eye_track'):
                    # if the eye tracking field has been set in the external data field, then append the new data
                    self.data.externd.eye_track.append_data(self.data, in_f)
                else:
                    # otherwise, create a new field to store the eye tracking data
                    setattr(self.data.externd, 'eye_track', EyeTrackingData(self.data, in_f))

            # updates the free experiments
            if self.data.externd.eye_track.n_file > 0:
                self.data.req_update = True
                self.fcn_data.update_extern_expts(self.combo_scope, 'Eye Tracking')

        # updates the progressbar
        self.update_thread_job('File Load Complete!', 100.)
        time.sleep(0.5)
        self.update_thread_job('Waiting For Process...', 0.)

    def load_general_file(self):
        '''

        :return:
        '''

        # sets the file types
        file_types = ['All Files (*.*)',
                      'CSV Files (*.csv)',
                      'Text Files (*.txt)']

        # runs the file open dialog
        file_dlg = FileDialogModal(caption='Select General File(s) To Open',
                                   filter=';;'.join(file_types),
                                   directory=cfcn.get_dir_para('inputDir'))
        file_dlg.setFileMode(QFileDialog.ExistingFiles)

        # determines if the user selected a valid file
        if (file_dlg.exec() == QDialog.Accepted):
            # otherwise, set the output file name
            filt_type = file_dlg.selectedNameFilter()
            input_file = file_dlg.selectedFiles()
        else:
            # if the user cancelled then exit
            return

        if 'All Files' in filt_type:
            # determines what type of analysis is to be performed based on the opened file
            _, f_extn = os.path.splitext(input_file[0])

            if f_extn == '.opend':
                # case is the freely moving data (as calculated/exported by Adam's analysis code)

                # loads the general data file as a binary
                for i_f, in_f in enumerate(input_file):
                    # updates the progress bar
                    w_str = 'Loading File {0} of {1}'.format(i_f + 1, len(input_file))
                    self.update_thread_job(w_str, 100. * (i_f + 1) / (len(input_file) + 1))
                    # opens the new file
                    with open(in_f, 'rb') as fp:
                        f_data = p.load(fp)

                    # sets the new data based on the type
                    if hasattr(self.data.externd, 'free_data'):
                        self.data.externd.free_data.append_data(self.data, f_data)
                    else:
                        setattr(self.data.externd, 'free_data', FreelyMovingData(self.data, f_data))

                # updates the free experiments
                if self.data.externd.free_data.n_file > 0:
                    self.data.req_update = True
                    self.fcn_data.update_extern_expts(self.combo_scope, 'Freely Moving Cell Types')

        elif 'CSV Files' in filt_type:
            # PUT CODE IN HERE DEPENDING ON FILE TYPE
            pass

        elif 'Text Files' in filt_type:
            # PUT CODE IN HERE DEPENDING ON FILE TYPE
            pass

        # updates the progressbar
        self.update_thread_job('File Load Complete!', 100.)
        time.sleep(0.5)
        self.update_thread_job('Waiting For Process...', 0.)

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
                data_fix, data_free = cf.get_comp_datasets(self.data, ind=ind, is_full=True)

                # sets the fixed/free experiment text labels
                fix_name = cf.extract_file_name(data_fix['expFile'])
                free_name = cf.extract_file_name(data_free['expFile'])

                # determines if the new combination files already exist
                chk_status, i_comp = cfcn.check_existing_compare(self.data.comp.data, fix_name, free_name)
                if chk_status == 3:
                    # file comparison combination is already stored
                    return
                elif chk_status in [1, 2]:
                    # case is either the fixed or free file is already within an existing comparison combination
                    c_data = self.data.comp.data[i_comp]

                    # prompts the user if they wish to remove the existing combination for the new one
                    q_str = 'The {0} data file is currently being used for another comparison match:\n\n' \
                            ' * Comparison File = {1}\n' \
                            ' * Existing Matching File = {2}\n' \
                            ' * New Matching File = {3}\n\n' \
                            'Do you wish to remove the current comparison match for the new match?'.format(
                                {1: 'fixed', 2: 'free'}[chk_status],
                                fix_name if chk_status == 1 else free_name,
                                free_name if chk_status == 2 else fix_name,
                                c_data.fix_name if chk_status == 2 else c_data.free_name
                            )
                    u_choice = QMessageBox.question(self, 'Output Images To Sub-Directory?', q_str,
                                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if u_choice == QMessageBox.No:
                        # if the user refused, then exit the function
                        return
                    else:
                        # otherwise, remove the data field belonging to the existing match
                        self.data.comp.data.pop(i_comp)

                # creates a new field for the comparison field
                self.data.comp.set_comparison_data(ind, n_fix=data_fix['nC'],
                                                    n_free=data_free['nC'],
                                                    n_pts=data_fix['nPts'],
                                                    fix_name=fix_name,
                                                    free_name=free_name)

                #
                self.data.comp.is_set = True
                self.fcn_data.update_comp_expts()

                # # starts the worker thread
                # iw = self.det_avail_thread_worker()
                # thread_job_para = [self.data.comp.data[-1], data_fix, data_free, self.def_data['g_para']]
                # self.worker[iw].set_worker_func_type('cluster_matches', thread_job_para=thread_job_para)
                # self.worker[iw].start()

        else:
            # otherwise, create the error message
            cf.show_error(e_str, 'Comparison Experiment Set Error!')

            # sets the information label properties
            if self.data.comp.is_set:
                self.menu_output_data.setEnabled(True)

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

        # REMOVE ME LATER!
        a = 1

        # # determines which flags have been set
        # is_set = np.array([self.data.comp.is_set, self.data.classify.is_set, self.data.rotation.is_set])
        # if not np.any(is_set):
        #     return
        #
        # # sets the initial output data dictionary
        # radio_options = np.array(func_types[:-1])[is_set]
        # file_type = ['Comma Separated Value (.csv)', 'Excel Spreadsheet (.xlsx)']
        # out_data = {'dataDir': self.def_data['dir']['dataDir'], 'dataName': '',
        #             'fileType': file_type[0], 'outData': radio_options[0]}
        #
        # # initialisations
        # dlg_info = [
        #     ['Data File Output Directory', 'dataDir', 'Directory', self.def_data['dir']['dataDir'], True, False, 0],
        #     ['Data File Name', 'dataName', 'String', '', True, False, 1],
        #     ['Data File Type', 'fileType', 'List', file_type, True, False, 1],
        #     ['Experimental Data To Output', 'outData', 'Radio', radio_options, True, False, 2],
        # ]
        #
        # # opens up the config dialog box and retrieves the final file information
        # cfig_dlg = config_dialog.ConfigDialog(dlg_info=dlg_info,
        #                                       title='Data Output Options',
        #                                       width=500,
        #                                       init_data=out_data)
        #
        # # retrieves the information from the dialog
        # out_info = cfig_dlg.get_info()
        # if out_info is not None:
        #     # sets the output file name
        #     out_tmp = os.path.join(out_info['dataDir'], out_info['dataName'])
        #     out_file = cf.set_file_name(out_tmp, out_info['fileType'])
        #     is_csv = out_file[-3:] == 'csv'
        #
        #     # outputs the data depending on the type
        #     if out_info['outData'] == 'Cluster Matching':
        #         # case is the cluster matching
        #         self.output_cluster_matching_data(out_file, is_csv)
        #     elif out_info['outData'] == 'Cluster Classification':
        #         # case is the cell classification (FINISH ME!)
        #         self.output_cell_classification_data(out_file, is_csv)
        #     elif out_info['outData'] == 'Rotation Analysis':
        #         # case is the rotation analysis (FINISH ME!)
        #         self.output_rotation_analysis_data(out_file, is_csv)

    def save_file(self):
        '''

        :return:
        '''

        # determines which file type is being combined
        file_type, is_comp = self.file_type, True
        if file_type == 1:
            # case is individual experiment files are currently loaded
            is_comp = self.data.comp.is_set
            s_str = 'Multi-Comparison' if is_comp else 'Multi-Experiment'

        elif file_type == 2:
            # case is individual comparison experiment files are currently loaded
            s_str = 'Multi-Comparison'

        elif file_type == 3:
            # case is a multi-experiment data file
            s_str = 'Multi-Experiment'

        else:
            # case is a multi-comparison data file
            s_str = 'Multi-Comparison'

        # sets the initial output data dictionary
        out_data = {'inputDir': self.def_data['dir']['inputDir'], 'dataName': ''}

        # initialisations
        dlg_info = [
            ['{0} Data File Output Directory'.format(s_str), 'inputDir', 'Directory',
             self.def_data['dir']['inputDir'], True, False, 0],
            ['{0} Data File Name'.format(s_str), 'dataName', 'String', '', True, False, 1],
        ]

        if file_type == 1:
            if is_comp:
                # case is outputting a single comparison data file
                calc_comp = self.fcn_data.det_comp_expt_names(True)
                dlg_info += [['Fixed Experiment Name', 'exptName', 'List', calc_comp, True, False, 2]]
                out_data['exptName'] = calc_comp

            # else:
            #     # case is outputting a multi-experiment data file
            #     exp_file = [x['expInfo']['name'] for x in self.data._cluster]
            #     dlg_info += [['Experiment File Name', 'exptName', 'CheckCombo', exp_file, True, False, 2]]
            #     out_data['expOut'] = exp_file

        elif file_type == 2:
            # case is outputting a multi-experiment comparison data file
            calc_comp = self.fcn_data.det_comp_expt_names(True)
            dlg_info += [['Comparison File Experiment Names', 'exptName', 'CheckCombo', calc_comp, True, False, 2]]
            out_data['exptName'] = calc_comp

        # opens up the config dialog box and retrieves the final file information
        cfig_dlg = config_dialog.ConfigDialog(dlg_info=dlg_info,
                                              title='Combined Cluster Matching Data File Output Options',
                                              width=500,
                                              init_data=out_data,
                                              use_first_line=False)

        # retrieves the information from the dialog
        out_info = cfig_dlg.get_info()
        if out_info is not None:
            # case is a comparison data file is being output
            if self.file_type == 1:
                # case is a single comparison experiment data file is being output
                if is_comp:
                    cf.save_single_comp_file(self, out_info)
                else:
                    # case is a multi-experiment data file is being output
                    cf.save_multi_data_file(self, out_info)

            elif self.file_type == 2:
                # case is a multi comparison experiment data file is being output
                cf.save_multi_comp_file(self, out_info)

            else:
                # case is a multi-experiment data file is being output
                cf.save_multi_data_file(self, out_info, self.file_type == 3)

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
            with open(cf.default_dir_file, 'wb') as fw:
                p.dump(self.def_data, fw)

    def update_glob_para(self):
        '''

        :return:
        '''

        # parameter lists
        lda_type = ['One-Trial Out', 'One-Phase Out']

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
            ['ROC Conf. Interval Level', 'roc_clvl', 'Number', '', True, False, 5, _bright_red],

            ['LDA Trial Setup Type', 'lda_trial_type', 'List', lda_type, True, False, 6, _gray],
            ['Plot Width/Height Ratio', 'w_ratio', 'Number', '', True, False, 6, _bright_purple],
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
            with open(cf.default_dir_file, 'wb') as fw:
                p.dump(self.def_data, fw)

            # reshapes the plot groups
            self.reshape_plot_group_dim()

    def show_info(self):
        '''

        :return:
        '''

        # opens the data file information dialog
        InfoDialog(self)

    def alt_fields(self):
        '''

        :return:
        '''

        # runs the parameter field dialog
        self.check_data_fields(False)

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
            self.data.force_calc = True

    def init_rotfilt(self):
        '''

        :return:
        '''

        # runs the rotation filter in exclustion mode
        r_filt = RotationFilter(self.fcn_data, init_data=self.data.exc_rot_filt, is_exc=True)

        # determines if the gui was updated correctly
        if r_filt.is_ok:
            # updates the current parameter value
            self.data.exc_rot_filt = r_filt.get_info()
            self.data.req_update = True
            self.data.force_calc = True

    def init_udfilt(self):
        '''

        :return:
        '''

        # runs the uniformdrifting filter in exclustion mode
        r_filt = RotationFilter(self.fcn_data, init_data=self.data.exc_ud_filt, is_exc=True)

        # determines if the gui was updated correctly
        if r_filt.is_ok:
            # updates the current parameter value
            self.data.exc_ud_filt = r_filt.get_info()
            self.data.req_update = True
            self.data.force_calc = True

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

            # if the user cancelled a calculation function then reset
            if self.worker[iw].thread_job_primary == 'run_calc_func':
                self.calc_cancel = True

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
        self.func_data = None
        calc_para, plot_para, self.output_var = self.split_func_para()

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
            if self.det_calc_para_change(calc_para, plot_para, current_fcn, plot_scope):
                # if so, then run the calculation thread
                self.fcn_data.prev_fcn = dcopy(current_fcn)
                self.fcn_data.prev_calc_para = dcopy(calc_para)
                self.fcn_data.prev_plot_para = dcopy(plot_para)
                self.set_group_enabled_props(self.grp_prog, True)

                # runs the worker thread
                fcn_para = [calc_para, plot_para, self.data, self.fcn_data.pool, self.def_data['g_para']]
                self.text_prog.setText('Initialising Function...')
                self.pbar_prog.setValue(0)

                # runs the calculation on the next available thread
                iw = self.det_avail_thread_worker()
                self.worker[iw].set_worker_func_type('run_calc_func', current_fcn, fcn_para)
                self.worker[iw].start()
                return
        else:
            # if there aren't any calculation parameters, then reset the previous calculation parameter object
            self.fcn_data.prev_calc_para = None
            self.fcn_data.prev_plot_para = None
            self.fcn_data.prev_fcn = dcopy(current_fcn)

        # creates the new plot canvas
        self.plot_fig = PlotCanvas(self)
        self.plot_fig.move(10, 10)
        self.plot_fig.show()

        try:
            self.update_thread_job('Creating Plot Figure', 100.0 / 2.0)
            eval('self.{0}(**plot_para)'.format(func_name))
        except:
            a = 1

        if self.calc_ok:
            # if the calculation/plotting when successfully, then show the figure
            self.plot_fig.draw()
            self.plot_fig.fig.canvas.update()
            self.menu_save_figure.setEnabled(True)
            self.menu_save_data.setEnabled(self.func_data is not None)
            self.update_thread_job('Plot Figure Complete!', 100.0)
        else:
            # otherwise,
            self.plot_fig.hide()
            self.menu_save_figure.setEnabled(False)
            self.menu_save_data.setEnabled(False)
            self.update_thread_job('Plot Figure Error!', 100.0)

        # updates te progress bar to indicate the plot has finished being created
        time.sleep(0.1)
        self.finished_thread_job(None, is_plot=True)

    ###################################################
    ####    CLUSTER MATCHING ANALYSIS FUNCTIONS    ####
    ###################################################

    def plot_cluster_match_signals(self, i_cluster, plot_all, m_type, plot_grid=True):
        '''

        :return:
        '''

        def setup_plot_axes(plot_fig, n_plot, n_c, n_r):
            '''

            :return:
            '''

            # sets up the axes dimensions
            top, bottom, pH, wspace, hspace = 0.91, 0.06, 0.01, 0.25, 0.35

            # creates the gridspec object
            gs = gridspec.GridSpec(n_r, n_c, width_ratios=[1 / n_c] * n_c, height_ratios=[1 / n_r] * n_r,
                                   figure=plot_fig.fig, wspace=wspace, hspace=hspace, left=0.075, right=0.98,
                                   bottom=bottom, top=top)

            # creates the subplots
            plot_fig.ax = np.empty(n_plot, dtype=object)
            for i_plot in range(n_r * n_c):
                i_r, i_c = int(i_plot / n_c), i_plot % n_c
                if i_plot >= n_plot:
                    ax = plot_fig.figure.add_subplot(gs[i_r, i_c])
                    ax.axis('off')
                else:
                    plot_fig.ax[i_plot] = plot_fig.figure.add_subplot(gs[i_r, i_c])

        # retrieves the
        i_comp = self.data.comp.last_comp
        c_data = dcopy(self.data.comp.data[i_comp])

        # retrieves the fixed/free datasets
        data_fix, _ = cf.get_comp_datasets(self.data, c_data=c_data)
        _data_fix, data_free = cf.get_comp_datasets(self.data, c_data=c_data, is_full=True)

        # retrieves the fixed/free cluster inclusion indices
        cl_inc_fix = cfcn.get_inclusion_filt_indices(_data_fix, self.data.exc_gen_filt)
        cl_inc_free = cfcn.get_inclusion_filt_indices(data_free, self.data.exc_gen_filt)

        # removes any excluded cells from the free dataset
        ii = np.where(c_data.i_match >= 0)[0]
        jj = c_data.i_match[ii]
        c_data.i_match[ii[np.logical_not(cl_inc_free[jj])]] = -1
        c_data.i_match_old[ii[np.logical_not(cl_inc_free[jj])]] = -1

        # reduces down the match indices to only include the feasible fixed dataset indices
        c_data.i_match, c_data.i_match_old = c_data.i_match[cl_inc_fix], c_data.i_match_old[cl_inc_fix]
        c_data.is_accept, c_data.is_accept_old = c_data.is_accept[cl_inc_fix], c_data.is_accept_old[cl_inc_fix]

        # sets the match/acceptance flags
        if m_type == 'New Method':
            i_match = c_data.i_match
            is_acc = c_data.is_accept
        else:
            i_match = c_data.i_match_old
            is_acc = c_data.is_accept_old

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
        n_c, n_r = cf.det_subplot_dim(n_plot)
        setup_plot_axes(self.plot_fig, n_plot, n_c, n_r)
        ax = self.plot_fig.ax

        # plots the values over all subplots
        n_ct = n_plot % n_c
        for i_plot in range(n_plot):
            # sets the actual fixed/free plot indices
            j_plot = i_cluster[i_plot] - 1
            i_match_new = i_match[j_plot]
            id_fix = data_fix['clustID'][j_plot]

            # plots the fixed signal
            ax[i_plot].plot(T, data_fix['vMu'][:, j_plot], linewidth=3.0)
            if i_match_new >= 0:
                # if there was a match, then plot the mean matches
                ax[i_plot].plot(T, data_free['vMu'][:, i_match_new], 'r', linewidth=2.0)

                # set the title match/colour
                t_str = 'Fixed #{0}/Free #{1}'.format(id_fix, data_free['clustID'][i_match_new])
                t_col = col[int(is_acc[j_plot])]
            else:
                # otherwise, there was no feasible match (set reduced title which is to be black)
                t_str, t_col = 'Fixed #{0}'.format(id_fix), 'k'

            # sets the plot values
            ax[i_plot].set_xlim(T[0], T[-1])
            ax[i_plot].set_title(t_str, color=t_col)
            ax[i_plot].grid(plot_grid)

            i_c, i_r = i_plot % n_c, int(i_plot / n_c)
            is_bot_row = ((i_r + 1) == n_r) if ((i_c < n_ct) or (n_ct == 0)) else ((i_r + 2) == n_r)
            if is_bot_row:
                ax[i_plot].set_xlabel('Time (ms)')

            if i_c == 0:
                ax[i_plot].set_ylabel('Voltage ({0}V)'.format(cf._mu))

            # creates the legend (first plot only)
            if i_plot == 0:
                bbox = ax[i_plot].get_position()
                L, B, W, H = bbox.p0[0], bbox.p0[1], bbox.width, bbox.height
                TOP = B + H
                T_lg = TOP + 3 * (1 - TOP) / 4

                bbox_lg = ((0.5 - L) / W, (T_lg - B) / H)
                ax[i_plot].legend(['Fixed', 'Free'], ncol=2, loc=10, bbox_to_anchor=bbox_lg)

    def plot_single_match_mean(self, plot_comp, i_cluster, n_trace, is_horz, rej_outlier, plot_grid=True):
        '''

        :return:
        '''

        # retrieves the comparison data struct belonging to the selected experiment
        i_comp = cf.det_comp_dataset_index(self.data.comp.data, plot_comp)
        c_data = dcopy(self.data.comp.data[i_comp])

        # retrieves the fixed/free datasets
        data_fix, data_free = cf.get_comp_datasets(self.data, c_data=c_data, is_full=True)

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
        i_match = c_data.i_match[i_cluster]

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
        fix_ID, free_ID = data_fix['clustID'][i_cluster], data_free['clustID'][c_data.i_match[i_cluster]]

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

        # sets the grid properties for the plot axes
        for ax in self.plot_fig.ax[:3]:
            ax.grid(plot_grid)

        # resets the x-axis limits
        for ax in self.plot_fig.ax:
            ax.set_xlim(T[0], T[-1])
            ax.set_ylim(yL)

    def plot_signal_metrics(self, plot_comp, all_expt, is_3d, m_type, plot_grid):
        '''

        :return:
        '''

        # initialisations
        pWL, pWU = 0.90, 1.10

        # retrieves the fixed/free data sets based on the type
        if all_expt:
            # case is multi-experiment files have been loaded
            data_fix, data_free, c_data = self.get_multi_comp_datasets(all_expt, None)

            # sets the experiment names (if multi-experiment)
            exp_files = np.array(cf.flat_list([[x.fix_name] * len(x.i_match) for x in c_data]))
        else:
            # retrieves the comparison data struct belonging to the selected experiment
            i_comp = cf.det_comp_dataset_index(self.data.comp.data, plot_comp)
            c_data = dcopy(self.data.comp.data[i_comp])

            # case is only single experiment files have been loaded
            data_fix, data_free = cf.get_comp_datasets(self.data, c_data=c_data, is_full=True)

        # ensures the data files are stored in lists
        if not isinstance(data_fix, list):
            data_fix, data_free, c_data = [data_fix], [data_free], [c_data]

        # sets the z-scores to be plotted based on type
        if m_type == 'New Method':
            # sets the match indices and acceptance flags
            if all_expt:
                # sets the match/acceptance flags
                i_match = np.array(cf.flat_list([list(x.i_match) for x in c_data]))
                is_accept = np.array(cf.flat_list([list(x.is_accept) for x in c_data]))

                # sets the x, y and z-axis plot values
                x_plot = np.array(cf.flat_list([list(x.sig_diff) for x in c_data]))
                y_plot = np.array(cf.flat_list([list(np.min(x.signal_feat, axis=1)) for x in c_data]))
                z_plot = np.array(cf.flat_list([list(x.isi_corr) for x in c_data]))

                # sets the free cell cluster ID#
                id_free = np.array(cf.flat_list(
                    [self.get_free_cluster_match_ids(x['clustID'], y.i_match) for x, y in zip(data_free, c_data)]
                ))

            else:
                # sets the x, y and z-axis plot values
                i_match, is_accept = c_data[0].i_match, c_data[0].is_accept
                x_plot, y_plot, z_plot = c_data[0].sig_diff, np.min(c_data[0].signal_feat, axis=1), c_data[0].isi_corr

                # sets the free cell cluster ID#
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
            if all_expt:
                # sets the match/acceptance flags
                i_match = np.array(cf.flat_list([list(x.i_match_old) for x in c_data]))
                is_accept = np.array(cf.flat_list([list(x.is_accept_old) for x in c_data]))

                # sets the x, y and z-axis plot values
                x_plot = np.array(cf.flat_list([list(x.sig_corr) for x in c_data]))
                y_plot = np.array(cf.flat_list([list(x.sig_diff_old) for x in c_data]))
                z_plot = np.array(cf.flat_list([list(np.nanmax(np.abs(x.z_score), axis=0)) for x in c_data]))

                # sets the free cell cluster ID#
                id_free = np.array(cf.flat_list(
                    [self.get_free_cluster_match_ids(x['clustID'], y.i_match) for x, y in zip(data_free, c_data)]
                ))
            else:
                # sets the x, y and z-axis plot values
                i_match, is_accept = c_data[0].i_match_old, c_data[0].is_accept_old
                x_plot, y_plot = c_data[0].sig_corr, c_data[0].sig_diff_old
                z_plot = np.nanmax(np.abs(c_data[0].z_score), axis=0)

                # sets the free cell cluster ID#
                id_free = np.array(self.get_free_cluster_match_ids(data_free[0]['clustID'], i_match))

            # sets the axis labels
            x_label, y_label, z_label = 'Correlation Coefficient', 'L2 Norm', 'Max Z-Score'
            x_lim = [pWL * np.nanmin(x_plot), 1.0]
            y_lim = [pWL * np.nanmin(y_plot), pWU * np.nanmax(y_plot)]
            z_lim = [pWL * np.nanmin(z_plot), pWU * np.nanmax(z_plot)]

        # sets up the figure/axis
        self.init_plot_axes(is_3d=is_3d)
        ax = self.plot_fig.ax[0]

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
            h = ax.scatter(x_plot[is_plot], y_plot[is_plot], z_plot[is_plot], marker='o', c=cm, alpha=1)
            self.remove_scatterplot_spines(ax)
            ax.view_init(20, -45)
        else:
            # case is a 2D plot
            h = ax.scatter(x_plot[is_plot], y_plot[is_plot], marker='o', c=cm, alpha=1)

        # creates the cursor object
        datacursor(h, formatter=formatter, point_labels=lbl, hover=True)

        # # creates the cursor object
        # cursor = mplcursors.cursor(h, hover=True)
        # cursor.connect("add", lambda sel: sel.annotation.set_text(lbl[sel.target.index]))

        # sets the scatterplot properties
        # ax.legend(['Accepted', 'Rejected'], loc=3)
        ax.set_xlabel('{0} (X)'.format(x_label))
        ax.set_ylabel('{0} (Y)'.format(y_label))
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(plot_grid)
        self.plot_fig.draw()

        # adds the z-axis label (if a 3D plot)
        if is_3d:
            ax.set_zlabel('{0} (Z)'.format(z_label))
            ax.set_zlim(z_lim)

    def plot_new_cluster_signals(self, plot_comp, i_cluster, plot_all, sig_type, plot_grid=True):
        '''

        :return:
        '''

        # initialisation
        reset_ylim, y_lim = True, [0, 0]

        # retrieves the comparison data struct belonging to the selected experiment
        i_comp = cf.det_comp_dataset_index(self.data.comp.data, plot_comp)
        c_data = dcopy(self.data.comp.data[i_comp])

        # retrieves the fixed/free datasets
        data_fix, data_free = cf.get_comp_datasets(self.data, c_data=c_data, is_full=True)

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
            y_data = c_data.match_intersect
            y_label = 'Intersection'
            y_lim, reset_ylim = [0, 1], False
        elif sig_type == 'Wasserstein Distance':
            # case is the wasserstain (earth-mover) distance
            y_data = c_data.match_wasserstain
            y_label = 'Wasser. Distance'
        else:
            # case is the bhattacharyya distance
            y_data = c_data.match_bhattacharyya
            y_label = 'BHA Distance'

        # sets up the figure/axis
        n_plot = len(i_cluster)
        self.init_plot_axes(n_plot=n_plot)

        # plots the values over all subplots
        for i_plot in range(n_plot):
            # creates the new subplot
            j_plot = i_cluster[i_plot] - 1
            i_match = c_data.i_match[j_plot]
            id_fix = data_fix['clustID'][j_plot]

            # only plot data values if there was a match
            if i_match >= 0:
                # plots the z-scores and the upper/lower limits
                self.plot_fig.ax[i_plot].plot(T,y_data[:, j_plot], 'b')

                # sets the title properties
                id_free = data_free['clustID'][i_match]
                t_str = 'Fixed #{0}/Free #{1}'.format(id_fix, id_free)
                t_col = col[int(c_data.is_accept[j_plot])]
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

        # retrieves the fixed/free datasets
        data_fix, data_free = cf.get_comp_datasets(self.data)

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
            x_plot = self.data.c_data.mu_dist[i_plot, :, :].T

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

    def plot_cluster_isi(self, plot_comp, i_cluster, plot_all, t_lim, plot_all_bin, is_norm, equal_ax, plot_grid=True):
        '''

        :param i_cluster:
        :param plot_all:
        :param is_norm:
        :return:
        '''

        # retrieves the comparison data struct belonging to the selected experiment
        i_comp = cf.det_comp_dataset_index(self.data.comp.data, plot_comp)
        c_data = dcopy(self.data.comp.data[i_comp])

        # retrieves the fixed/free datasets
        data_fix, data_free = cf.get_comp_datasets(self.data, c_data=c_data)

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
            i_match = c_data.i_match[j_plot]
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
                t_col = col[int(c_data.is_accept[j_plot])]
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

    def plot_fix_free_corr(self, rot_filt, ff_cluster, free_exp_name, show_trend, bin_sz, lcond_type, plot_type,
                                 vel_dir, plot_grid, plot_scope):
        '''

        :param rot_filt:
        :param ff_cluster:
        :param free_exp_name:
        :param show_trend:
        :param bin_sz:
        :param lcond_type:
        :param plot_type:
        :param vel_dir:
        :param plot_grid:
        :param plot_scope:
        :return:
        '''

        def det_reverse_indices(i_cell_b, ind_gff):
            '''

            :param i_cell_b:
            :param ind_gff:
            :return:
            '''

            _, _, ind_rev = np.intersect1d(i_cell_b, ind_gff, return_indices=True)
            return ind_rev

        def det_expt_index(ind_gfilt, ind_corr):
            '''

            :param ind_gfilt:
            :param ind_corr:
            :return:
            '''

            # memory allocation
            ind_ex = -np.ones(len(ind_gfilt), dtype=int)

            #
            for i_expt in range(len(ind_corr)):
                ind_ex[np.in1d(ind_gfilt, ind_corr[i_expt])] = i_expt

            # returns the index array
            return ind_ex

        def setup_plot_axes(plot_fig, n_filt):
            '''

            :param plot_fig:
            :param n_filt:
            :return:
            '''

            # sets up the axes dimensions
            n_r, n_c = n_filt, 7
            top, bottom, pH, wspace, hspace = 0.95, 0.06, 0.01, 0.30, 0.20

            # creates the gridspec object
            gs = gridspec.GridSpec(n_r, n_c, width_ratios=[1 / n_c] * n_c, height_ratios=[1 / n_r] * n_r,
                                   figure=plot_fig.fig, wspace=wspace, hspace=hspace, left=0.05, right=0.98,
                                   bottom=bottom, top=top)

            # creates the subplots
            plot_fig.ax = np.empty(n_filt + 2, dtype=object)
            for i_filt in range(n_filt):
                plot_fig.ax[i_filt] = plot_fig.figure.add_subplot(gs[i_filt, :(n_c - 3)])

            # sets up the other axes
            plot_fig.ax[n_filt] = plot_fig.figure.add_subplot(gs[:, (n_c - 3):(n_c - 1)])
            plot_fig.ax[n_filt + 1] = plot_fig.figure.add_subplot(gs[:, -1])

            cf.set_axis_limits(plot_fig.ax[n_filt], [0, 1], [0, 1])
            plot_fig.ax[n_filt].plot([-2, -1], [-2, -1], 'w')
            plot_fig.ax[n_filt].axis('off')

        # initialises the rotation filter (if not set)
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # initialisations
        ff_corr = self.data.comp.ff_corr
        f_data = self.data.externd.free_data
        tt_key = {'DARK1': 'Black', 'LIGHT1': 'Uniform', 'LIGHT2': 'Uniform'}
        collapse_arr = lambda y, i_c: np.array(cf.flat_list([x[i_c] for x in y]))
        n_bin_h = int(80 / ff_corr.vel_bin)

        # sets the grouping index
        if ff_corr.split_vel:
            i_grp = ['Negative', 'Positive'].index(vel_dir)
            ind_grp = [np.arange(n_bin_h), np.arange(n_bin_h, 2 * n_bin_h)][i_grp]
        else:
            i_grp, ind_grp = 0, np.arange(2 * n_bin_h)

        if plot_type == 'Individual Cell Correlation':
            ############################################
            ####    INDIVIDUAL CELL CORRELATIONS    ####
            ############################################

            # determines the matching cells against the freely moving experiment file
            i_expt_f2f, f2f_map = cf.det_matching_fix_free_cells(self.data, exp_name=[free_exp_name])

            # calculates the number of time bins
            axLmx = [1e10, -1e10]
            p_value = 2.5
            n_tt = len(f_data.t_type)

            # sets the cdf xi-values
            xi_cdf = np.linspace(-1, 1, 201)
            x_cdf = 0.5 * (xi_cdf[:-1] + xi_cdf[1:])

            # retrieves the fixed/free mapping indices
            i_expt = f_data.exp_name.index(free_exp_name)
            is_ok = np.where(f2f_map[0][:, 1] > 0)[0]

            # determines the index of the selected cluster
            clust_id = np.array(re.findall(r'\d+', ff_cluster)).astype(int)
            i_clust = is_ok[np.where(np.logical_and(ff_corr.clust_id[i_expt][:, 0] == clust_id[0],
                                                    ff_corr.clust_id[i_expt][:, 1] == clust_id[1]))[0][0]]

            # retrieves the fixed/free spiking frequencies (across all the group types)
            sf_fix = [x[i_clust, ind_grp] for x in ff_corr.sf_fix[i_expt, :]]
            sf_free = [x[i_clust, ind_grp] for x in ff_corr.sf_free[i_expt, :]]
            sf_corr = [sf_c[i_clust, i_grp] for sf_c in ff_corr.sf_corr[i_expt]]
            sf_grad = [sf_g[i_clust, :, i_grp] for sf_g in ff_corr.sf_grad[i_expt]]
            sf_corr_sh = [sf_sh[i_clust, :, i_grp] for sf_sh in ff_corr.sf_corr_sh[i_expt]]
            sf_sig = [(sf_s[i_clust, i_grp] != 0) for sf_s in ff_corr.sf_corr_sig[i_expt]]

            # initialises the plot axes
            self.init_plot_axes(n_row=n_tt, n_col=2)
            ax = self.plot_fig.ax

            # creates the subplots for each condition type
            for i_tt in range(n_tt):
                # sets the left/right axis indices
                iL, iR = 2 * i_tt, 2 * i_tt + 1

                # creates the scatterplot (ensures free spiking frequencies are reversed)
                ax[iL].plot(sf_fix[i_tt], dcopy(sf_free[i_tt]), 'o')

                # resets the axis limits
                axLmx = [min(axLmx[0], min(ax[iL].get_xlim()[0], ax[iL].get_ylim()[0])),
                         max(axLmx[1], max(ax[iL].get_xlim()[1], ax[iL].get_ylim()[1]))]

                # sets the axis properties
                tt_nw = f_data.t_type[i_tt]
                ax[iL].set_title('Spiking Frequency Relation ({0}/{1})'.format(tt_nw, tt_key[tt_nw]))
                ax[iL].set_ylabel('Free Spiking Frequency (Hz)')
                ax[iL].grid(plot_grid)

                # calculates the cumulative distribution values
                sf_corr_hist = np.histogram(sf_corr_sh[i_tt], bins=xi_cdf, normed=False)[0]
                sf_corr_cdf = 100. * np.cumsum(sf_corr_hist / np.sum(sf_corr_hist))
                i_cdf_bin = np.where(x_cdf > sf_corr[i_tt])[0][0]
                t_str = 'Correlation = {:5.3f}{}'.format(sf_corr[i_tt], '*' if sf_sig[i_tt] else '')

                # plots the cumulative distribution traces
                ax[iR].plot(x_cdf, sf_corr_cdf, 'k')
                ax[iR].plot(x_cdf[i_cdf_bin] * np.ones(2), [0, sf_corr_cdf[i_cdf_bin]], 'r--')
                ax[iR].plot([-1, x_cdf[i_cdf_bin]], sf_corr_cdf[i_cdf_bin] * np.ones(2), 'r--')
                ax[iR].plot([-1, 1], p_value * np.ones(2), '-', c='r', linewidth=2)
                ax[iR].plot([-1, 1], (100 - p_value) * np.ones(2), '-', c='r', linewidth=2)

                # sets the axis properties
                ax[iR].set_title(t_str)
                ax[iR].set_xlim([-1, 1])
                ax[iR].set_ylim([0, 100])
                ax[iR].set_ylabel('Percentage')
                ax[iR].grid(plot_grid)

            for i_tt in range(n_tt):
                _ax = ax[2 * i_tt]
                cf.set_axis_limits(_ax, axLmx, axLmx)

            # adds in the trendline (if required)
            if show_trend:
                #
                for i_tt in range(n_tt):
                    # sets the lower limit coordinates
                    p0 = np.vstack([
                        [axLmx[0], sf_grad[i_tt][0] * axLmx[0] + sf_grad[i_tt][1]],
                        [(axLmx[0] - sf_grad[i_tt][1]) / sf_grad[i_tt][0], axLmx[0]],
                    ])

                    # sets the upper limit coordinates
                    p1 = np.vstack([
                        [axLmx[1], sf_grad[i_tt][0] * axLmx[1] + sf_grad[i_tt][1]],
                        [(axLmx[1] - sf_grad[i_tt][1]) / sf_grad[i_tt][0], axLmx[1]],
                    ])

                    # determines which points are to be plotted
                    i0 = np.where(np.all(np.logical_and(p0 >= axLmx[0], p0 <= axLmx[1]), axis=1))[0][0]
                    i1 = np.where(np.all(np.logical_and(p1 >= axLmx[0], p1 <= axLmx[1]), axis=1))[0][0]

                    # sets the coordinates of the trendline's second point
                    ax[2 * i_tt].plot([p0[i0, 0], p1[i1, 0]], [p0[i0, 1], p1[i1, 1]], 'r--', linewidth=2)

            # sets the axis properties
            ax[-2].set_xlabel('Fixed Spiking Frequency (Hz)')
            ax[-1].set_xlabel('Correlation')

        else:
            # sets the reverse trial type condition dictionary key
            tt_key_rev = {'Black': 'DARK1', 'Uniform': lcond_type}

            # determines the matching cells against the freely moving experiment file
            i_expt_f2f, f2f_map = cf.det_matching_fix_free_cells(self.data, exp_name=f_data.exp_name)
            is_ok0 = [x[:, 0] >= 0 for x in f2f_map]
            is_ok = np.array(cf.flat_list(is_ok0))

            # retrieves the common filtered indices
            r_obj_wc = RotationFilteredData(self.data, rot_filt, None, None, True, 'Whole Experiment', False)
            t_type_full = [x['t_type'][0] for x in r_obj_wc.rot_filt_tot]
            i_cell_b, r_obj_tt = cfcn.get_common_filtered_cell_indices(self.data, r_obj_wc, t_type_full, True)

            # sets the global-to-local and trial condition indices
            ind_gff = cf.flat_list([list(x) for x in ff_corr.ind_g])
            ind_gfilt0 = [det_reverse_indices(ic, ind_gff) for ic in i_cell_b]
            ind_gfilt = [x[ok] for x, ok in zip(ind_gfilt0, [is_ok[ig] for ig in ind_gfilt0])]
            i_cond = [np.where(f_data.t_type == tt_key_rev[tt])[0][0] for tt in t_type_full]

            # sets the spiking frequency correlation/significance values
            sf_corr = [collapse_arr(ff_corr.sf_corr, i_c)[i_g, i_grp] for i_c, i_g in zip(i_cond, ind_gfilt)]
            sf_sig = [collapse_arr(ff_corr.sf_corr_sig, i_c)[i_g, i_grp] > 0 for i_c, i_g in zip(i_cond, ind_gfilt)]
            sf_sig_all = [np.any(collapse_arr(ff_corr.sf_corr_sig, i_c)[i_g, :] > 0, axis=1)
                                                                        for i_c, i_g in zip(i_cond, ind_gfilt)]

            #
            n_cell, n_free = [len(sf) for sf in sf_sig], len(i_expt_f2f)
            ind_ex = np.array(cf.flat_list([[i] * len(x) for i, x in enumerate(ff_corr.ind_g)]))

            if plot_type == 'Correlation Histogram':
                #####################################
                ####    CORRELATION HISTOGRAM    ####
                #####################################

                # REMOVE ME LATER
                is_norm = False

                # initialisations
                yLmx, n_filt = 0, r_obj_wc.n_filt
                col = cf.get_plot_col(n_filt)
                p_sig_tot, tab_str = np.empty(n_filt, dtype=object), np.empty(n_filt, dtype=object)

                # sets the cdf xi-values
                xi_cdf = np.linspace(-1, 1, int(2 / bin_sz) + 1)
                x_cdf = 0.5 * (xi_cdf[:-1] + xi_cdf[1:])
                b_wid = 0.85 * (x_cdf[1] - x_cdf[0])

                # initialises the plot axes
                # n_col, n_row = cf.det_subplot_dim(n_filt)
                setup_plot_axes(self.plot_fig, n_filt)
                ax = self.plot_fig.ax

                # creates the histograms for each filter option
                for i_filt in range(n_filt):
                    #############################################
                    ####    CORRELATION HISTOGRAM SUBPLOT    ####
                    #############################################

                    # sets the index of the column to be analysed
                    t_str = r_obj_wc.lg_str[i_filt].replace('\n', ', ')

                    if len(sf_corr[i_filt]):
                        # calculates the histogram of the significant cells
                        sf_corr_hist = np.histogram(sf_corr[i_filt], bins=xi_cdf, normed=False)[0]
                        sf_corr_hist_sig = np.histogram(sf_corr[i_filt][sf_sig[i_filt]], bins=xi_cdf, normed=False)[0]

                        # calculates the proportions
                        if is_norm:
                            sf_corr_hist_sum = np.sum(sf_corr_hist)
                            p_sig = 100. * sf_corr_hist_sig / sf_corr_hist_sum
                            p_nsig = 100. * (sf_corr_hist - sf_corr_hist_sig) / sf_corr_hist_sum
                        else:
                            p_sig = sf_corr_hist_sig
                            p_nsig = sf_corr_hist - sf_corr_hist_sig

                        # case is the significant values so normalise using the provided value
                        ax[i_filt].bar(x_cdf, p_sig, width=b_wid, edgecolor=col[i_filt], color=col[i_filt])
                        ax[i_filt].bar(x_cdf, p_nsig, width=b_wid, bottom=p_sig, edgecolor=col[i_filt], color='None')

                    # sets the axis properties
                    ax[i_filt].set_title('{0} (#{1})'.format(t_str, i_filt + 1))
                    ax[i_filt].set_xlim([-1, 1])
                    ax[i_filt].grid(plot_grid)

                    if (i_filt + 1) < n_filt:
                        ax[i_filt].set_xticklabels([])

                    # sets the y-axis limits
                    yLmx = max(yLmx, ax[i_filt].get_ylim()[1])

                    ##############################################
                    ####    CELL SIGNIFICANCE CALCULATIONS    ####
                    ##############################################

                    # calculates the percentage of significant cells (over all experiments)
                    i_ex_gf = ind_ex[ind_gfilt[i_filt]]
                    n_sig = sum([np.sum(sf_sig_all[i_filt][i_ex_gf == i_ex]) for i_ex in range(n_free)])
                    p_sig_tot[i_filt] = [np.mean(sf_sig_all[i_filt][i_ex_gf == i_ex]) for i_ex in range(n_free)]

                    # sets up the table values
                    n_sig_tab = np.array([n_sig, n_cell[i_filt] - n_sig, n_cell[i_filt]])
                    p_sig_tab = ['{:.1f}'.format(100 * x / n_cell[i_filt]) for x in n_sig_tab]
                    tab_str[i_filt] = np.vstack((n_sig_tab, p_sig_tab))

                ############################################
                ####    FINAL FIGURE PROPERTY UPDATE    ####
                ############################################

                # sets the axis properties
                for i_ax, _ax in enumerate(ax[:n_filt]):
                    # resets the overall y-axis limit
                    _ax.set_ylim([0, yLmx])

                    # sets the x-axis labels (last row only)
                    if (i_ax + 1) == n_filt:
                        _ax.set_xlabel('Correlation')

                    # sets the y-axis labels
                    _ax.set_ylabel('Percentage' if is_norm else 'Cell Count')

                ################################################
                ####    CELL SIGNIFICANCE SUBPLOT/TABLES    ####
                ################################################

                # initialisations
                xi = np.arange(n_filt) + 1
                col_sig, col_row = cf.get_plot_col(n_filt + 1, n_filt), cf.get_plot_col(2, 2 * n_filt + 1)
                t_props = np.empty(n_filt, dtype=object)
                col_hdr, row_hdr = ['Matched', 'Unmatched', 'Total'], ['Count', '%age']

                # calculates the mean/sem percentages
                p_sig_mu = 100. * np.mean(np.vstack(p_sig_tot).T, axis=0)
                p_sig_sem = 100. * np.std(np.vstack(p_sig_tot).T, axis=0) / np.sqrt(n_free)

                # creates the bar graph
                for i in range(len(xi)):
                    ax[-1].bar(xi[i], p_sig_mu[i], width=0.9, color=col[i], yerr=p_sig_sem[i])

                # updates the axis properties
                ax[-1].set_title('%age Matched')
                ax[-1].set_xticks(xi)
                ax[-1].set_xticklabels(['#{0}'.format(x) for x in xi])
                ax[-1].grid(plot_grid)
                cf.set_axis_limits(ax[-1], [0.5, xi[-1]+0.5], [-1, 101])

                # creates the statistics tables for each filter type
                for i in range(n_filt):
                    # creates the table
                    t_props[i] = cf.add_plot_table(self.plot_fig, ax[n_filt], table_font_small, tab_str[i], row_hdr,
                                                   col_hdr, col_row, col_sig, 'top')

                    # resets the table dimensions
                    t_props[i][0]._bbox[0] = max(t_props[i][0]._bbox[0], -0.05)
                    t_props[i][0]._bbox[1] = 1 - (i / n_filt) * (1 + 1.25 * t_props[i][0]._bbox[3])
                    t_props[i][0]._bbox[2] = min(t_props[i][0]._bbox[2], 1.0)

            elif plot_type == 'Correlation Scatterplot':
                #######################################
                ####    CORRELATION SCATTERPLOT    ####
                #######################################

                # parameters
                h_sig = []
                m_size, mlt = 20, 3

                # significance colours
                sig_col = [cf.convert_rgb_col([147, 149, 152])[0],      # non-significant markers
                           cf.convert_rgb_col(_green)[0],               # black-only significant markers
                           cf.convert_rgb_col(_bright_purple)[0],       # uniform-only significant markers
                           cf.convert_rgb_col(_bright_red)[0]]          # both condition significant spikes

                # initialises the plot axes
                n_plot = int(r_obj_wc.n_filt / 2)
                n_col, n_row = cf.det_subplot_dim(n_plot)
                self.init_plot_axes(n_plot=n_plot, n_row=n_row, n_col=n_col)
                ax = self.plot_fig.ax

                #
                for i_plot in range(n_plot):
                    # calculates the significance scores
                    ii = 2 * i_plot + np.array([0, 1])
                    sig_score = sf_sig[ii[0]] + 2 * sf_sig[ii[1]]
                    sf_x, sf_y = sf_corr[ii[0]], sf_corr[ii[1]]

                    # plots the significant values
                    for i_sig in range(1, 4):
                        # creates the legend markers (first subplot only)
                        if i_plot == 0:
                            h_sig.append(ax[i_plot].scatter(-2, -2, marker='o', s=mlt*m_size, facecolor=sig_col[i_sig]))

                        # determines if there are any significant cells
                        is_sig = sig_score == i_sig
                        if np.any(is_sig):
                            # if there are significant cells, then
                            ax[i_plot].scatter(sf_x[is_sig], sf_y[is_sig], marker='o',
                                               s=mlt*m_size, facecolor=sig_col[i_sig])

                    # plots the scatterplot values
                    ax[i_plot].scatter(sf_x, sf_y, marker='o', s=m_size, facecolor=sig_col[0])
                    ax[i_plot].plot([-1, 1], [0, 0], 'r--')
                    ax[i_plot].plot([0, 0], [-1, 1], 'r--')
                    cf.set_axis_limits(ax[i_plot], [-1, 1], [-1, 1])

                    # sets the subplot title
                    t_str = ', '.join(r_obj_wc.lg_str[ii[0]].split()[:-1]) if '\n' in r_obj_wc.lg_str[0] else 'All Cells'
                    ax[i_plot].set_title(t_str)

                    # sets the other axis properties
                    ax[i_plot].grid(plot_grid)
                    ax[i_plot].set_xlabel('Black Correlation')
                    ax[i_plot].set_ylabel('Uniform Correlation')

                    # creates the legend (first subplot only)
                    if i_plot == 0:
                        ax[i_plot].legend(h_sig, ['Black Sig.', 'Uniform Sig.', 'Both Sig.'])

    def plot_old_cluster_signals(self, plot_comp, i_cluster, plot_all, plot_grid=True):
        '''

        :return:
        '''

        # retrieves the comparison data struct belonging to the selected experiment
        i_comp = cf.det_comp_dataset_index(self.data.comp.data, plot_comp)
        c_data = dcopy(self.data.comp.data[i_comp])

        # retrieves the fixed/free datasets
        n_pts = c_data.n_pts
        data_fix, data_free = cf.get_comp_datasets(self.data, c_data=c_data, is_full=True)

        # resets the cluster index if plotting all clusters
        i_cluster, e_str = self.check_cluster_index_input(i_cluster, plot_all, data_fix['nC'])
        if e_str is not None:
            cf.show_error(e_str, 'Infeasible Cluster Indices')
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
            i_match = c_data.i_match[j_plot]
            id_fix = data_fix['clustID'][j_plot]

            # only plot data values if there was a match
            if i_match >= 0:
                # plots the z-scores and the upper/lower limits
                self.plot_fig.ax[i_plot].plot(T, c_data.z_score[:, j_plot], 'b')
                self.plot_fig.ax[i_plot].plot([0, n_pts], [1, 1], 'r--')
                self.plot_fig.ax[i_plot].plot([0, n_pts], [-1, -1], 'r--')

                # sets the title properties
                id_free = data_free['clustID'][i_match]
                t_str = 'Fixed #{0}/Free #{1}'.format(id_fix, id_free)
                t_col = col[int(c_data.is_accept[j_plot])]
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

    ######################################################
    ####    CELL CLASSIFICATION ANALYSIS FUNCTIONS    ####
    ######################################################

    def plot_classification_metrics(self, exp_name, all_expt, c_met1, c_met2, c_met3, use_3met,
                                    class_type, m_size, plot_grid):
        '''

        :return:
        '''

        ###################################################
        ####    INITIALISATIONS & MEMORY ALLOCATION    ####
        ###################################################

        # initialisations
        cluster = self.data.cluster
        t_lim, n_met, p_size = [-1e10, 1e10], 7, 2.5
        lg_str = ['Narrow Spikes', 'Wide Spikes']

        # sets up the marker colours
        # col = ['b', 'r', convert_rgb_col(_bright_cyan), convert_rgb_col(_bright_yellow)]
        col = [cf.convert_rgb_col([33, 71, 97])[0],           # narrow spikes
               cf.convert_rgb_col([163, 77, 72])[0],          # wide spikes
               cf.convert_rgb_col([104, 201, 210])[0],        # excitatory spikes
               cf.convert_rgb_col([252, 195, 150])[0]]        # inhibitory spikes

        # initialises the subplot axes
        self.clear_plot_axes()
        self.plot_fig.ax = np.empty(4, dtype=object)

        # sets up the classification scatterplot
        if use_3met:
            m_size_s = m_size / 2
            self.plot_fig.ax[0] = self.plot_fig.figure.add_subplot(1, 2, 1, projection='3d')
        else:
            m_size_s = m_size
            self.plot_fig.ax[0] = self.plot_fig.figure.add_subplot(1, 2, 1)

        # sets up the signal subplots
        for i in range(3):
            self.plot_fig.ax[i+1] = self.plot_fig.figure.add_subplot(3, 2, 2*(i+1))

        # initialisations
        c_metric = ['Trough to 2nd Peak Time (ms)', '2nd Peak to Trough Ratio', 'Trough Half-Width (ms)',
                    'Peak Ratio'] #, '2nd Peak Half-Width (ms)', '2nd Peak Relaxation Time (ms)',
                    # 'Firing Rate (Hz)']

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
        # if use_pca:
        #     # transforms the data using PCA
        #     pca = PCA(n_components=np.size(x_clust_scaled, axis=1))
        #     x_clust_scaled = pca.fit_transform(x_clust_scaled)

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
        x_label, y_label, z_label = c_met1, c_met2, c_met3

        # sets the scatterplot tooltip strings
        lbl = ['Expt = {0}\n==========\nID# = {1}\nGroup = {2}\n=========='.format(
            x, y, z+1) for x, y, z in zip(expt_id, expt_cl, grp_idx)
        ]
        lbl = ['{}\nX = {:5.3f}\nY = {:5.3f}'.format(x, y, z) for x, y, z in zip(lbl, x_clust[:, 0], x_clust[:, 1])]

        #
        if use_3met:
            lbl = ['{}\nZ = {:5.3f}'.format(x, y) for x, y in zip(lbl, x_clust[:, 2])]

        #
        if self.data.classify.action_set:
            lbl = ['{0}\n==========\n{1}'.format(x, act_str[y]) for x, y in zip(lbl, act_type)]
            i1, i2 = act_type == 1, act_type == 2

        # creates the scatterplot
        if use_3met:
            if self.data.classify.action_set:
                self.plot_fig.ax[0].scatter(x_clust_plt[i1, 0], x_clust_plt[i1, 1], x_clust_plt[i1, 2],
                                            marker='o', c=col[2], s=p_size*m_size_s)
                self.plot_fig.ax[0].scatter(x_clust_plt[i2, 0], x_clust_plt[i2, 1], x_clust_plt[i2, 2],
                                            marker='o', c=col[3], s=p_size*m_size_s)

            # case is a 3D plot
            h = self.plot_fig.ax[0].scatter(x_clust_plt[:, 0], x_clust_plt[:, 1], x_clust_plt[:, 2],
                                            marker='o', c=cm, s=m_size_s)
            self.remove_scatterplot_spines(self.plot_fig.ax[0])
            self.plot_fig.ax[0].view_init(20, -45)
        else:
            if self.data.classify.action_set:
                self.plot_fig.ax[0].scatter(x_clust_plt[i1, 0], x_clust_plt[i1, 1],
                                            marker='o', c=col[2], s=p_size*m_size_s)
                self.plot_fig.ax[0].scatter(x_clust_plt[i2, 0], x_clust_plt[i2, 1],
                                            marker='o', c=col[3], s=p_size*m_size_s)

            # case is a 2D plot
            h = self.plot_fig.ax[0].scatter(x_clust_plt[:, 0], x_clust_plt[:, 1], marker='o', c=cm, s=m_size_s)
            self.plot_fig.ax[0].grid(plot_grid)

        # retrieves the axis limits
        x_lim = self.plot_fig.ax[0].get_xlim()
        y_lim = self.plot_fig.ax[0].get_ylim()

        if use_3met:
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

        if use_3met:
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
                self.plot_fig.ax[i+1].plot(T_plot, Vm_N, c=tuple(col[0]))
                if (i == 0):
                    self.plot_fig.ax[i + 1].plot(T_plot, np.mean(Vm_N, axis=1), 'k', linewidth=3)

            # case is the wide spikes
            if i in [1, 2]:
                self.plot_fig.ax[i+1].plot(T_plot, Vm_W, c=tuple(col[1]))
                if (i == 1):
                    self.plot_fig.ax[i + 1].plot(T_plot, np.mean(Vm_W, axis=1), 'k', linewidth=3)

            # plots the spike line
            self.plot_fig.ax[i + 1].plot([0, 0], self.plot_fig.ax[i+1].get_ylim(), 'k--')

            # sets the plot
            self.plot_fig.ax[i + 1].set_xlim(T_plot[0], T_plot[-1])
            self.plot_fig.ax[i + 1].set_title(t_str[i])
            self.plot_fig.ax[i + 1].set_xlabel('Time (ms)')
            self.plot_fig.ax[i + 1].set_ylabel('Normalised Voltage')
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

    #################################################
    ####    FREELY MOVING CELL TYPE FUNCTIONS    ####
    #################################################

    def plot_free_cell_stats(self, free_exp_name, plot_all, vel_bin, use_pcent, use_place, plot_grid):
        '''

        :param free_exp_name:
        :param plot_all:
        :param vel_bin:
        :param plot_grid:
        :return:
        '''

        def setup_plot_axes(plot_fig):
            '''

            :param plot_fig:
            :return:
            '''

            # sets up the axes dimensions
            nR, nC = 10, 4
            top, bottom, pH, wspace, hspace = 0.97, 0.06, 0.01, 0.4, 0.05

            # creates the gridspec object
            gs = gridspec.GridSpec(nR, nC, width_ratios=[1 / nC] * nC, height_ratios=[1 / nR] * nR,
                                   figure=plot_fig.fig, wspace=wspace, hspace=hspace, left=0.075, right=0.98,
                                   bottom=bottom, top=top)

            # creates the subplots
            plot_fig.ax = np.empty(3, dtype=object)
            plot_fig.ax[0] = plot_fig.figure.add_subplot(gs[:, :2])
            plot_fig.ax[1] = plot_fig.figure.add_subplot(gs[1:, 2:])
            plot_fig.ax[2] = plot_fig.figure.add_subplot(gs[0, 2:])

            plot_fig.ax[2].axis('off')

        # initialisations
        i_bin = ['5', '10'].index(vel_bin)
        cell_type_all = self.data.externd.free_data.cell_type
        ahv_score_all = self.data.externd.free_data.ahv_score
        c_key = list(cell_type_all[0][0].columns)

        # sets the plot values
        if plot_all:
            # case is plotting all experiments
            cell_type, ahv_score = [x[i_bin] for x in cell_type_all], [x[i_bin] for x in ahv_score_all]
        else:
            # case is plotting a single experiment
            i_expt = self.data.externd.free_data.exp_name.index(free_exp_name)
            cell_type, ahv_score = [cell_type_all[i_expt][i_bin]], [ahv_score_all[i_expt][i_bin]]

        # initialises the plot axes
        setup_plot_axes(self.plot_fig)
        ax = self.plot_fig.ax

        # other initialisations
        n_expt = len(cell_type)
        n_plot = len(c_key) - (1 + int(not use_place))
        n_cell = [np.shape(x)[0] for x in cell_type]

        # sets the bar graph colours
        col_b = cf.get_plot_col(n_plot+1)
        col_pp = cf.get_plot_col(2, n_plot+1)

        #############################################
        ####    CELL TYPE PERCENTAGES SUBPLOT    ####
        #############################################

        # calculates the freely moving cell types
        n_type_cell = np.vstack([np.array(np.sum(x, axis=0)) for x in cell_type])
        p_type_cell = 100. * np.vstack([np.array(np.mean(x, axis=0)) for x in cell_type])
        p_type_mu = np.mean(p_type_cell, axis=0)[1:]
        p_type_sem = np.std(p_type_cell, axis=0)[1:] / np.sqrt(n_expt)
        x_lbl = c_key[1:(n_plot+1)]

        # creates the bar graphs for each cell type
        for i_type in range(n_plot):
            ax[0].bar(i_type, p_type_mu[i_type], width=0.9, color=col_b[i_type],
                                        edgecolor=col_b[i_type], yerr=p_type_sem[i_type])
            ax[0].bar(i_type, 100 - p_type_mu[i_type], bottom=p_type_mu[i_type],
                                        width=0.9, color='w', edgecolor=col_b[i_type])

        # sets the axis properties
        ax[0].set_xticks(np.arange(n_plot))
        ax[0].set_xticklabels(x_lbl)
        ax[0].set_ylabel('Percentage of Cells')
        ax[0].set_ylim([-1, 101])
        ax[0].grid(plot_grid)

        # creates the output table
        p_type_str = np.vstack([np.sum(n_type_cell, axis=0)[1:(n_plot+1)],
                                np.array(['{:.1f}'.format(pt) for pt in p_type_mu[:n_plot]])])
        p_type_str = np.hstack(([p_type_str, np.array([str(sum(n_cell)), '100.0']).reshape(-1, 1)]))
        cf.add_plot_table(self.plot_fig, ax[0], table_font, p_type_str, ['Count', '%age'],
                                         x_lbl + ['Total Cells'], col_pp, col_b, 'bottom')

        ####################################
        ####    VENN DIAGRAM SUBPLOT    ####
        ####################################

        # memory allocation
        ct_dict = {}

        # sets the type matches over all experiments/cell types
        for i_ct, ct in enumerate(cell_type):
            for ck in x_lbl:
                # determines the number of matches with the current cell-type
                n_match = np.where(ct[ck])[0]
                if len(n_match) > 0:
                    # if there are any matches, then add it to the match dictionary
                    nw_match = set(['{0}|{1}'.format(i_ct+1, x) for x in n_match])
                    if ck in ct_dict:
                        ct_dict[ck] = ct_dict[ck].union(nw_match)
                    else:
                        ct_dict[ck] = nw_match

        # creates the venn diagram
        if len(list(ct_dict.keys())) == 4:
            # case is 4 cell-types groups have been found
            if use_pcent:
                # case is using percentages to represent venn diagram set regions
                venn(ct_dict, fmt="{percentage:.1f}%", fontsize=8, legend_loc="upper left", cmap=col_b, ax=ax[1])
            else:
                # case is using counts to represent venn diagram set regions
                venn(ct_dict, fontsize=8, legend_loc="upper left", ax=ax[1], cmap=col_b)
        else:
            # case is less than 4 groups have been found
            ct_lbl = set([x for x in x_lbl if x in ct_dict])
            ct_set = [set(ct_dict[ct]) for ct in ct_lbl]
            col_venn = [col_b[x_lbl.index(ct)] for ct in ct_lbl]

            # creates the venn diagrams based on the type group count
            if use_pcent:
                n_total = sum(n_cell)
                lbl_fmt = lambda x: '{:.1%}'.format(x/n_total)

                if len(ct_lbl) == 3:
                    venn3(ct_set, ct_lbl, ax=ax[1], set_colors=col_venn, subset_label_formatter=lbl_fmt)
                else:
                    venn2(ct_set, ct_lbl, ax=ax[1], set_colors=col_venn, subset_label_formatter=lbl_fmt)
            else:
                if len(ct_lbl) == 3:
                    venn3(ct_set, ct_lbl, ax=ax[1], set_colors=col_venn)
                else:
                    venn2(ct_set, ct_lbl, ax=ax[1], set_colors=col_venn)

        ###########################################
        ####    AHV TYPE PERCENTAGE SUBPLOT    ####
        ###########################################

        # calculates the head direction cells
        n_hd = sum([sum(ct['HD']) for ct in cell_type])
        n_hdmod = sum([sum(ct['HDMod']) for ct in cell_type])

        # creates the output table
        n_hd_str = np.array([str(x) for x in [n_hd, n_hdmod]]).reshape(1, -1)
        t_props_1 = cf.add_plot_table(self.plot_fig, ax[2], table_font, n_hd_str, ['Count'],
                                      ['HD Cells', 'HDMod Cells'], [col_pp[0]], col_b[:2], 'top')

        l_min = 0.025
        t_props_1[0]._bbox[0] = max(t_props_1[0]._bbox[0], l_min)
        t_props_1[0]._bbox[2] = min(t_props_1[0]._bbox[2], 1 - 2 * l_min)

        ########################################
        ####    HDMOD PERCENTAGE SUBPLOT    ####
        ########################################

        # calculates the total/symmetric ahv cell counts
        n_ahv_sig = sum([sum(x > 0) for x in ahv_score])
        n_ahv_sym = sum([sum(x == 3) for x in ahv_score])

        # creates the output table
        n_ahv_str = np.array([str(x) for x in [n_ahv_sym, n_ahv_sig - n_ahv_sym, n_ahv_sig]]).reshape(1, -1)
        t_props_2 = cf.add_plot_table(self.plot_fig, ax[2], table_font, n_ahv_str, ['Count'],
                                      ['Symmetric', 'Asymmetric', 'Total'], [col_pp[0]], col_b[:3], 'top')

        # resets the bottom location of the upper table
        c_hght = t_props_1[0]._bbox[3] / (np.size(n_hd_str, axis=0) + 1)
        t_props_2[0]._bbox[0] = max(t_props_2[0]._bbox[0], l_min)
        t_props_2[0]._bbox[1] = t_props_1[0]._bbox[1] - (t_props_1[0]._bbox[3] + c_hght)
        t_props_2[0]._bbox[2] = min(t_props_2[0]._bbox[2], 1 - 2 * l_min)

    ######################################
    ####    EYE TRACKING FUNCTIONS    ####
    ######################################

    def plot_eye_movement_signals(self, etrack_exp_name, plot_all, etrack_tt, plot_mean, plot_grid):
        '''

        :param etrack_exp_name:
        :param etrack_tt:
        :param plot_all:
        :param plot_grid:
        :return:
        '''

        # initialisations
        n_sd, dp_ax = 3, 10
        extn_data = self.data.externd.eye_track
        et_tt = [x.lower() for x in etrack_tt]
        t_str0 = ['Medial to Temporal', 'Temporal to Medial']

        # memory allocation
        n_expt, n_tt, n_evnt = len(extn_data.et_data) if plot_all else 1, len(etrack_tt), 2
        y_sig = np.empty((n_tt, n_evnt), dtype=object)

        # retrieves the eye-tracking data based on the plot type
        if plot_all:
            # case is plotting all the experiments
            et_data, y_evnt = extn_data.et_data, extn_data.y_evnt
        else:
            # case is plotting a single experiment
            i_expt = extn_data.exp_name.index(etrack_exp_name)
            et_data, y_evnt = [extn_data.et_data[i_expt]], [extn_data.y_evnt[i_expt]]

        # retrieves the signal values over all experiments/trial types
        for i_tt in range(len(et_tt)):
            for i_evnt in range(n_evnt):
                # retrieves the signal values over all experiments (for the current trial type/eye-movement type)
                y_sig_tmp, is_ok = np.empty(n_expt, dtype=object), np.ones(n_expt, dtype=bool)
                for i_exp, et_d in enumerate(et_data):
                    if et_tt[i_tt] in et_d.t_type:
                        i = et_d.t_type.index(et_tt[i_tt])
                        if len(y_evnt[i_exp][i][i_evnt]):
                            y_sig_tmp[i_exp] = y_evnt[i_exp][i][i_evnt]
                        else:
                            is_ok[i_exp] = False
                    else:
                        is_ok[i_exp] = False

                # combines the data over all experiments into a single array
                if np.any(is_ok):
                    y_sig[i_tt, i_evnt] = np.vstack(y_sig_tmp[is_ok])
                else:
                    y_sig[i_tt, i_evnt] = None

                    # initialises the plot axes
        self.init_plot_axes(n_row=n_tt, n_col=n_evnt)
        ax = self.plot_fig.ax

        # sets the time signal
        dt_sig = 1000 / extn_data.fps
        t_sig = np.arange(-extn_data.n_pre, extn_data.n_post + 1) * dt_sig

        #
        for i_tt in range(n_tt):
            for i_evnt in range(n_evnt):
                # sets the plot index
                i_plt = i_tt * n_evnt + i_evnt

                # creates the signal plot
                if y_sig[i_tt, i_evnt] is not None:
                    # sets the plot values and calculates the min/max, mean/std dev values
                    y_sig_plt = y_sig[i_tt, i_evnt]

                    # plots the all the sub-signals 
                    ax[i_plt].plot(t_sig, y_sig_plt.T, 'b', linewidth=1, alpha=0.2)
                    if plot_mean:
                        # plots the mean sub-signal (if required)
                        ax[i_plt].plot(t_sig, np.mean(y_sig_plt, axis=0), 'k', linewidth=3)

                # sets the x-axis label (last row only)
                if (i_tt + 1) == n_tt:
                    ax[i_plt].set_xlabel('Time (ms)')

                # sets the y-axis label (first column only)
                if i_evnt == 0:
                    ax[i_plt].set_ylabel('Relative Position (um)')

                # sets the other axis properties
                ax[i_plt].set_xlim([t_sig[0], t_sig[-1]])
                ax[i_plt].grid(plot_grid)
                ax[i_plt].set_title('{0} ({1})'.format(t_str0[i_evnt], etrack_tt[i_tt]))

    def plot_eye_movement_correlation_indiv(self, i_cell, plot_avg, exp_name, etrack_tt, m_size, show_err, plot_grid):
        '''

        :param i_cell:
        :param exp_name:
        :param etrack_tt:
        :param show_err:
        :param plot_grid:
        :return:
        '''

        # initialisations
        et_d = self.data.externd.eye_track
        i_expt = et_d.exp_name.index(exp_name)

        # determines if the trial type exists for the selected experiment
        if etrack_tt.lower() not in et_d.et_data[i_expt].t_type:
            # if not then output an error screen
            e_str = 'The "{0}" trial type is not present for this experiment.\n' \
                    'Either reload the experimental data or select another experiment.'.format(etrack_tt)
            cf.show_error(e_str, 'Missing Trial Type')

            # exit flagging an error
            self.calc_ok = False
            return

        else:
            # determines the index of the trial type
            i_tt = et_d.et_data[i_expt].t_type.index(etrack_tt.lower())
            n_cell = np.shape(et_d.p_corr[i_expt][i_tt])[0] - 1

            #
            if plot_avg:
                # case is the experiment average
                j_cell = n_cell
            else:
                # determines if the cluster index is valid
                if i_cell > n_cell:
                    # if not then output an error to screen
                    e_str = 'The cell index {0} exceeds the cell count for this experiment {1}.\n' \
                            'Re-run the function with a valid cell index.'.format(i_cell, n_cell)
                    cf.show_error(e_str, 'Missing Trial Type')

                    # exit flagging an error
                    self.calc_ok = False
                    return
                else:
                    # otherwise, set the
                    j_cell = i_cell - 1

        # retrieves the plot values
        dt_et = (1000 / et_d.fps)
        t_ex = np.arange(-et_d.n_pre, et_d.n_post + 1) * dt_et
        y_corr, p_corr = et_d.y_corr[i_expt][i_tt][j_cell, :], et_d.p_corr[i_expt][i_tt][j_cell, :]
        sp_evnt = [et_d.fps * sp_evnt[:, :, j_cell] for sp_evnt in et_d.sp_evnt[i_expt][i_tt]]

        # calculates the mean spiking frequency/eye-movement position sub-signals
        sp_evnt_mn = [np.mean(sp_ev, axis=0) for sp_ev in sp_evnt]
        y_evnt_mn = [np.mean(y_evnt, axis=0) for y_evnt in et_d.y_evnt[i_expt][i_tt]]

        # calculates the event spiking frequency (if plotting the signal error)
        if show_err:
            n_evnt = [np.shape(x)[0] for x in sp_evnt]
            sp_evnt_sem = [np.std(sp_ev, axis=0) / np.sqrt(n) for sp_ev, n in zip(sp_evnt, n_evnt)]

        # sets up the plot axes
        self.plot_fig.setup_plot_axis(n_row=2, n_col=2)
        ax = self.plot_fig.ax

        # sets the event string
        evnt_str = ['M-to-T', 'T-to-M']

        ##############################################
        ####    SPIKE EVENT HISTOGRAM SUBPLOTS    ####
        ##############################################

        # initialisations
        y_lim = [1e6, -1e6]

        for i_ax, _ax in enumerate(ax[:2]):
            # plots the error area (if required)
            if show_err:
                cf.create_step_area_patch(_ax, t_ex, sp_evnt_mn[i_ax], sp_evnt_sem[i_ax], 'b')

            # plots the mean spiking rate
            _ax.step(t_ex, sp_evnt_mn[i_ax], color='k', where='mid')

            # determines the overall axis limits
            yL = _ax.get_ylim()
            y_lim = [min(yL[0], y_lim[0]), max(yL[1], y_lim[1])]

            # sets the axes properties
            _ax.set_title('Firing Rate Histogram ({0})'.format(evnt_str[i_ax]))
            _ax.set_xlabel('Time (ms)')
            _ax.set_ylabel('Firing Rate (Hz)')
            _ax.grid(plot_grid)

        for i_ax, _ax in enumerate(ax[:2]):
            # resets the y-axis limits
            _ax.set_ylim(y_lim)

            # resets the x-axis limits
            if show_err:
                _ax.set_xlim([(t_ex[0] - dt_et / 2), (t_ex[-1] + dt_et / 2)])
            else:
                _ax.set_xlim([t_ex[0], t_ex[-1]])

        ############################################
        ####    SPIKE RATE/POSITION SUBPLOTS    ####
        ############################################

        # initialisations
        n_slope, p_value = 1000., 0.05
        x_lim, y_lim = [1e6, -1e6], [1e6, -1e6]

        # creates the subplots for each event type
        for i_ax, _ax in enumerate(ax[2:]):
            # creates the scatter-plot of values
            _ax.scatter(y_evnt_mn[i_ax], sp_evnt_mn[i_ax], marker='o', s=m_size, facecolor='b')

            # determines the overall axis limits
            xL, yL = _ax.get_xlim(), _ax.get_ylim()
            x_lim = [min(xL[0], x_lim[0]), max(xL[1], x_lim[1])]
            y_lim = [min(yL[0], y_lim[0]), max(yL[1], y_lim[1])]

            # sets the axes properties
            p_str = '{:.4f}{}'.format(p_corr[i_ax], '*' if p_corr[i_ax] < p_value else '')
            t_str = 'Firing Rate vs Position - {} (c = {:.4f}, p = {})'.format(evnt_str[i_ax], y_corr[i_ax], p_str)
            _ax.set_title(t_str)
            _ax.set_xlabel('Relative Position (um)')
            _ax.set_ylabel('Firing Rate (Hz)')
            _ax.grid(plot_grid)

        for i_ax, _ax in enumerate(ax[2:]):
            # resets the x/y-axis limits
            _ax.set_xlim(x_lim)
            _ax.set_ylim(y_lim)

            # plots the best fit line
            p = linregress(y_evnt_mn[i_ax], sp_evnt_mn[i_ax])
            y = p.slope * n_slope * np.array(x_lim) + p.intercept
            _ax.plot(n_slope * np.array(x_lim), y, 'r', linewidth=3)

        #######################################
        ####    OTHER FIGURE PROPERTIES    ####
        #######################################

        # sets the title string
        if plot_avg:
            # case is the experiment average
            t_str = 'Trial Type = {0}\n(Experiment Average)'.format(etrack_tt)
        else:
            # case is for a single cell
            i_clust = [cf.extract_file_name(x['expFile']) for x in self.data.cluster].index(exp_name)
            cl_id, ch_id = self.data.cluster[i_clust]['clustID'][j_cell], self.data.cluster[i_clust]['chDepth'][j_cell]
            t_str = 'Trial Type = {0}\nCluster #{1} (Channel #{2})'.format(etrack_tt, cl_id, ch_id)

        # resizes the figure to include the super-title
        self.plot_fig.fig.set_tight_layout(False)
        self.plot_fig.fig.suptitle(t_str, fontsize=16, fontweight='bold')
        self.plot_fig.fig.tight_layout(rect=[0, 0.01, 1, 0.935])

    def plot_eye_movement_correlation_exp(self, etrack_tt, corr_type, pr_type, plot_grid):
        '''

        :param etrack_exp_name:
        :param plot_all:
        :param etrack_tt:
        :param plot_grid:
        :return:
        '''

        def setup_plot_axes(plot_fig, n_filt):
            '''

            :param plot_fig:
            :param n_filt:
            :return:
            '''

            # sets up the axes dimensions
            n_r1, n_r2, n_c = 2, 1, 2
            tbl_hght = 0.05 + 0.02 * n_filt
            left, top, bottom, wspace, hspace, ax_gap = 0.1, 0.95, 0.025, 0.15, 0.3 + (0.025 * (n_filt - 1)), 0.025

            # creates the gridspec object
            gs1 = gridspec.GridSpec(n_r1, n_c, width_ratios=[1 / n_c] * n_c, height_ratios=[1 / n_r1] * n_r1,
                                    figure=plot_fig.fig, wspace=wspace, hspace=hspace, left=left, right=0.98,
                                    bottom=bottom + (tbl_hght + ax_gap), top=top)
            gs2 = gridspec.GridSpec(n_r2, n_c, width_ratios=[1 / n_c] * n_c, figure=plot_fig.fig, left=left,
                                    right=0.98, bottom=bottom, top=bottom + tbl_hght)

            # creates the correlation/significance proportion subplots
            plot_fig.ax = np.empty(6, dtype=object)
            for i_r in range(2):
                for i_c in range(2):
                    plot_fig.ax[2 * i_r + i_c] = plot_fig.figure.add_subplot(gs1[i_r, i_c])

            # if the count table row, then disable the axis
            for i_c in range(2):
                plot_fig.ax[4 + i_c] = plot_fig.figure.add_subplot(gs2[0, i_c])
                plot_fig.ax[4 + i_c].axis('off')

        def split_data_by_event_type(y0, is_comb=False):
            '''

            :param y_data:
            :return:
            '''

            # array dimensioning
            n_evnt = 2
            n_exp, n_filt = np.shape(y0)
            y_evnt = np.empty((n_exp, n_evnt), dtype=object)

            # combines the data by event type (over each experiment)
            for i_exp in range(n_exp):
                # combines the data for the current experiment
                y0_exp = np.hstack(y0[i_exp, :])
                for i_evnt in range(n_evnt):
                    y_evnt[i_exp, i_evnt] = y0_exp[:, i_evnt::n_evnt]

            # combines the data over all experiment (if required)
            if is_comb:
                y_evnt = [np.vstack(y_evnt[:, i_evnt]) for i_evnt in range(n_evnt)]

            # returns the array
            return y_evnt

        # initialisations
        p_value = 0.05
        n_filt, n_evnt = len(etrack_tt), 2
        et_d = self.data.externd.eye_track

        # determines the indices of the trial types over each experiment
        etrack_tt_lo = [x.lower() for x in etrack_tt]
        ind_tt = [np.array([x.index(y) for y in etrack_tt_lo]) for x in [x.t_type for x in et_d.et_data]]

        # initialises the plot axes
        setup_plot_axes(self.plot_fig, n_filt)
        ax = self.plot_fig.ax

        # sets the event string
        evnt_str = ['M-to-T', 'T-to-M']

        ###################################
        ####    CORRELATION SUBPLOT    ####
        ###################################

        # creates the graph
        c = cf.get_plot_col(n_filt)

        # retrieves the correlation values (set only for the selected trial types and split by the event types)
        y_corr0 = np.vstack([x[i] for i, x in zip(ind_tt, et_d.y_corr)])
        y_corr = split_data_by_event_type(y_corr0, True)

        for i_evnt in range(n_evnt):
            if corr_type == 'Boxplot':
                # case is the correlation plot type is a boxplot

                # sets up the plot values
                xi = np.arange(n_filt)
                y_plt_g = [y for y in y_corr[i_evnt].T]
                ax[i_evnt].boxplot(y_plt_g, positions=xi, vert=True, patch_artist=True, widths=0.9)

            else:
                # case is the correlation plot type is a violin/swarmplot

                # sets the x/y plot values
                x_plt = cf.flat_list([[j + 1] * len(x) for j, x in enumerate(y_corr[i_evnt].T)])
                y_plt = np.hstack(y_corr[i_evnt].T)

                # sets the violin/swarmplot dictionaries
                vl_dict = cf.setup_sns_plot_dict(ax=ax[i_evnt], x=x_plt, y=y_plt, inner=None, bw=1, cut=1)
                sw_dict = cf.setup_sns_plot_dict(ax=ax[i_evnt], x=x_plt, y=y_plt, color='white', edgecolor='gray', size=3)

                # creates the violin/swarmplot
                sns.violinplot(**vl_dict)
                sns.swarmplot(**sw_dict)

            # sets the axis properties
            t_str = '{0} ({1})'.format('Correlation', evnt_str[i_evnt])
            ax[i_evnt].set_xticklabels(etrack_tt)
            ax[i_evnt].set_title(t_str, fontweight='bold', fontsize=16)
            ax[i_evnt].grid(plot_grid)
            ax[i_evnt].set_ylim([-1, 1])

            # sets the y-axis label (first column only)
            if i_evnt == 0:
                ax[i_evnt].set_ylabel('Correlation')

            # sets the separation line (only if more than one filter type)
            if n_filt > 1:
                ax[i_evnt].plot([0.5, 0.5], [-1, 1], 'k--')

        ###############################################
        ####    SIGNIFICANCE PROPORTION SUBPLOT    ####
        ###############################################

        # creates the graph
        c2 = cf.get_plot_col(n_filt)

        # retrieves the correlation values (set only for the selected trial types and split by the event types)
        p_corr0 = np.vstack([x[i] for i, x in zip(ind_tt, et_d.p_corr)])
        p_corr = split_data_by_event_type(p_corr0)

        # calculates the number/proportion of significant cells over each experiment
        n_cell = [np.size(x[0], axis=0) for x in p_corr]
        n_sig = [np.vstack([np.sum(pp < p_value, axis=0) for pp in p_corr[:, i_evnt]]) for i_evnt in range(n_evnt)]

        #
        if pr_type == 'Stacked Bar':
            #
            p_sig0 = [(100. * np.sum(ns, axis=0) / sum(n_cell)).reshape(1, -1) for ns in n_sig]
            p_sig = [np.vstack((ps, 100 - ps)) for ps in p_sig0]
        else:
            # calculates the significance proportions for each experiment
            p_sig0 = [100. * np.divide(ns, repmat(n_cell, 2, 1).T) for ns in n_sig]
            p_sig = [np.reshape(ps.T, [n_filt, np.size(ps, axis=0), 1], 'A') for ps in p_sig0]

        # creates the graphs for the motion sensitive/direction selectivity plots
        for i in range(n_evnt):
            # creates the graph
            h_plt = cf.create_general_group_plot(ax[i + n_evnt], p_sig[i], pr_type, c2)
            cf.reset_axes_dim(ax[i + n_evnt], 'bottom', 0.075, True)

            # sets the legend/x-ticklabels based on the grouping type
            t_str = 'Significance Proportion ({0})'.format(evnt_str[i])
            ax[i + n_evnt].set_title(t_str, fontsize=16, fontweight='bold')

            # only set the y-axis label for the first subplot
            if i == 0:
                ax[n_evnt].set_ylabel('Population %')

            # sets the y-axis limits based on type
            if 'Bar' in pr_type:
                ax[i + n_evnt].set_ylim([0, 100])
            else:
                ax[i + n_evnt].set_ylim([-2.5, 102.5])

            # updates the axis properties
            ax[i + n_evnt].grid(plot_grid)
            ax[i + n_evnt].set_xticklabels(etrack_tt)

        #########################################
        ####    SIGNIFICANCE COUNT TABLES    ####
        #########################################

        # class strings
        class_str0 = ['Significant', 'Insignificant']
        cT = cf.get_plot_col(max(n_filt, 2), 2 * n_evnt)

        # calculates the
        n_sig_tot0 = [np.sum(ns, axis=0).reshape(-1, 1) for ns in n_sig]
        n_sig_tot = [np.hstack((ns_tot, (sum(n_cell) - ns_tot))) for ns_tot in n_sig_tot0]

        # sets the
        n_sig_N0 = [np.hstack((x, np.sum(x, axis=1).reshape(-1, 1))) for x in n_sig_tot]
        n_sig_N = [np.vstack((x, np.sum(x, axis=0))) for x in n_sig_N0]
        col_hdr = class_str0 + ['Total Cells']

        # creates the title text object
        t_font = cf.get_table_font_size(3)
        if (n_filt == 1) and (etrack_tt[0] == 'Black'):
            row_hdr = ['All Cells', 'Total']
        else:
            row_hdr = etrack_tt + ['Total']

        # creates the graphs for the motion sensitive/direction selectivity plots
        for i in range(2):
            # creates the new table
            j, nT = i + 2 * n_evnt, len(col_hdr)
            cf.add_plot_table(self.plot_fig, ax[j], t_font, n_sig_N[i].astype(int), row_hdr, col_hdr,
                              cT[:n_filt] + [(0.75, 0.75, 0.75)], cT[:nT] + [(0.75, 0.75, 0.75)],
                              'fixed', n_col=len(class_str0))

    #######################################################
    ####    SPIKING FREQUENCY CORRELATION FUNCTIONS    ####
    #######################################################

    def plot_freq_corr_indiv(self, rot_filt, i_cluster, plot_exp_name, plot_shuffle, plot_grid, plot_scope):
        '''

        :param rot_filt:
        :param i_cluster:
        :param plot_exp_name:
        :param plot_shuffle:
        :param plot_grid:
        :param plot_scope:
        :param is_fixed:
        :return:
        '''

        # initialisations
        r_data = self.data.rotation

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        r_obj_wc = RotationFilteredData(self.data, rot_filt, None, None, True, 'Whole Experiment', False)
        if not r_obj_wc.is_ok:
            self.calc_ok = False
            return

        ########################################
        ####    INDIVIDUAL CELL ANALYSIS    ####
        ########################################

        # parameters
        p_value = 2.5

        # initialisations
        xi_cdf = np.linspace(-1, 1, 2001)
        x_cdf = 0.5 * (xi_cdf[:-1] + xi_cdf[1:])
        xi, h_plt, h_plt2 = np.mean(r_data.vel_xi, axis=1), [], []

        # retrieves the base
        rot_filt_base = cf.init_rotation_filter_data(False)
        r_obj_b = RotationFilteredData(self.data, rot_filt_base, None, None, False, 'Whole Experiment', False)

        # retrieves the indices of the cells that correspond to the selected experiment
        i_expt = cf.get_expt_index(plot_exp_name, self.data.cluster)
        ind_expt = np.where(r_obj_b.i_expt[0] == i_expt)[0]
        if i_cluster >= len(ind_expt):
            # if the cluster index is invalid, then exit the function
            e_str = 'You have specified an index greater than the number of clusters ({0}) for the selected ' \
                    'experiment. Reset the cluster index and re-run the function.'.format(len(ind_expt))
            cf.show_error(e_str, 'Infeasible Cluster Indices')
            self.calc_ok = False
            return

        # otherwise, determine the global cell index
        i_cell_g, t_type = ind_expt[i_cluster - 1], rot_filt['t_type']
        n_grp = np.size(r_data.vel_sf_sig[t_type[0]], axis=1)
        col = cf.get_plot_col(len(t_type))

        # sets the velocity range indices
        if r_data.split_vel:
            # case is the velocity range is split into negative/positive ranges
            i_nw = [np.arange(int(len(xi) / 2)), np.arange(int(len(xi) / 2), len(xi))]
        else:
            # case is the velocity range is not split
            i_nw = [np.arange(len(xi))]

        # creates the plot axis
        self.init_plot_axes(n_row=2, n_col=n_grp)
        ax = self.plot_fig.ax

        # creates the sub-plot figures for each velocity group type/trial condition
        for i_grp in range(n_grp):
            # initialisations
            lg_str_cdf = []
            i_ax1, i_ax2 = i_grp, n_grp + i_grp

            for i_tt, tt in enumerate(t_type):
                ########################################
                ####    SPIKING FREQUENCY OUTPUT    ####
                ########################################

                # retrieves the mean/shuffled signal values
                vel_sf_mean = r_data.vel_sf_mean[tt][i_cell_g, i_grp]
                vel_sf_shuffle = r_data.vel_sf_shuffle[tt][i_cell_g, i_grp].T

                # creates the signal plot
                if plot_shuffle:
                    ax[i_ax1].plot(xi[i_nw[i_grp]], vel_sf_shuffle, col[i_tt], linewidth=1, alpha=0.05)
                h_plt.append(ax[i_ax1].plot(xi[i_nw[i_grp]], vel_sf_mean, col[i_tt], linewidth=2))

                # creates the legend (final plot)
                ax[i_ax1].grid(plot_grid)
                if (i_tt + 1) == len(t_type):
                    ax[i_ax1].legend([x[0] for x in h_plt], t_type)

                ########################################
                ####    SPIKING FREQUENCY OUTPUT    ####
                ########################################

                # retrieves the correlation/significance values
                vel_sf_corr = r_data.vel_sf_corr[tt][:, i_cell_g, i_grp]
                vel_sf_corr_mn = r_data.vel_sf_corr_mn[tt][i_cell_g, i_grp]
                vel_sf_sig = r_data.vel_sf_sig[tt][i_cell_g, i_grp]

                # calculates the cumulative distribution values
                sf_corr_hist = np.histogram(vel_sf_corr, bins=xi_cdf, normed=True)[0]
                sf_corr_cdf = 100. * np.cumsum(sf_corr_hist / np.sum(sf_corr_hist))

                # creates
                i_cdf_bin = max(0, np.where(x_cdf > vel_sf_corr_mn)[0][0] - 1)
                ax[i_ax2].plot(x_cdf, sf_corr_cdf, col[i_tt], linewidth=2)
                ax[i_ax2].plot(x_cdf[i_cdf_bin] * np.ones(2), [0, sf_corr_cdf[i_cdf_bin]], '--', c=col[i_tt])
                h_plt2.append(ax[i_ax2].plot([-1, x_cdf[i_cdf_bin]], sf_corr_cdf[i_cdf_bin] * np.ones(2),
                                                 '--', c=col[i_tt]))

                # appends the legend string and updates the axis properties (if final condition type)
                lg_str_cdf.append('{} (Corr = {:5.3f}{})'.format(tt, vel_sf_corr_mn, '*' if vel_sf_sig else ''))
                if (i_tt + 1) == len(t_type):
                    ax[i_ax2].plot([-1, 1], p_value * np.ones(2), '-', c='r', linewidth=2)
                    ax[i_ax2].plot([-1, 1], (100 - p_value) * np.ones(2), '-', c='r', linewidth=2)
                    ax[i_ax2].legend([x[0] for x in h_plt2], lg_str_cdf, loc=4)

                # sets the axis limits
                ax[i_ax2].set_xlim([-1, 1])
                ax[i_ax2].set_ylim([-0.1, 100.1])
                ax[i_ax2].grid(plot_grid)

        # sets the x/y-labels for the first subplot
        ax[0].set_ylabel('Spiking Frequency (Hz)')
        ax[0].set_xlabel('Angular Velocity (deg/s)')

        # sets the cumulative distribution properties for the first subplot
        ax[n_grp].set_ylabel('Percentage')
        ax[n_grp].set_xlabel('Correlation')

        if n_grp > 1:
            # sets the final axis limits
            y_lim0, y_lim1 = ax[0].get_ylim(), ax[1].get_ylim()
            y_lim_fin = [min(y_lim0[0], y_lim1[0]), max(y_lim0[1], y_lim1[1])]

            # negative velocity axis properties
            ax[0].set_title('Negative Velocities')
            ax[0].set_ylim(y_lim_fin)

            # positive velocity axis properties
            ax[1].set_title('Positive Velocities')
            ax[1].set_xlabel('Angular Velocity (deg/s)')
            ax[1].set_ylim(y_lim_fin)

            # positive velocity correlation cumulative distribution properties
            ax[3].set_xlabel('Correlation')

        else:
            # sets the axis title
            ax[0].set_title('All Velocities')

        # creates the super-title
        r_obj = RotationFilteredData(self.data, rot_filt, i_cluster, plot_exp_name, False, 'Individual Cell', False)
        self.plot_fig.fig.set_tight_layout(False)
        self.plot_fig.fig.tight_layout(rect=[0, 0.01, 1, 0.955])
        self.plot_fig.fig.suptitle('Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0],
                                   int(r_obj.ch_id[0][0])), fontsize=16, fontweight='bold')

    def plot_freq_corr_hist(self, rot_filt, dist_type, bin_size, comb_all, vel_dir, plot_grid, plot_scope, is_fixed):
        '''

        :param rot_filt:
        :param dist_type:
        :param bin_size:
        :param comb_all:
        :param vel_dir:
        :param plot_grid:
        :param plot_scope:
        :param is_fixed:
        :return:
        '''

        def setup_plot_axes(plot_fig, n_grp):
            '''

            :param plot_fig:
            :param n_filt:
            :return:
            '''

            # sets up the axes dimensions
            nR, nC = n_grp, 3
            top, bottom, pH, wspace, hspace = 0.97, 0.06, 0.01, 0.25, 0.225

            # creates the gridspec object
            gs = gridspec.GridSpec(nR, nC, width_ratios=[1 / nC] * nC, height_ratios=[1 / nR] * nR,
                                   figure=plot_fig.fig, wspace=wspace, hspace=hspace, left=0.075, right=0.98,
                                   bottom=bottom, top=top)

            # creates the subplots
            plot_fig.ax = np.empty(n_grp + 1, dtype=object)
            for i_grp in range(n_grp):
                # sets up the distribution/significance axis
                plot_fig.ax[i_grp] = plot_fig.figure.add_subplot(gs[i_grp, :2])

            # turns off the significance axis off
            plot_fig.ax[-1] = plot_fig.figure.add_subplot(gs[:, -1])
            plot_fig.ax[-1].axis('off')

        def calc_ks_stats(sf_cdf):
            '''

            :return:
            '''

            # parameters
            p_val = 0.05

            # memory allocation
            n_grp = len(sf_cdf)
            ks_stats = np.empty((n_grp, n_grp), dtype=object)

            # sets up the stats strings for each comparison
            for i_grp in range(n_grp):
                for j_grp in range(i_grp, n_grp):
                    if i_grp == j_grp:
                        # case is the comparison is against itself
                        ks_stats[i_grp, j_grp] = 'N/A'
                    else:
                        # case is the comparison is against another cdf distribution
                        ks_val = cfcn.calc_ks2_stat(sf_cdf[i_grp], sf_cdf[j_grp])
                        ks_stats_nw = '{:.3f}{}'.format(ks_val, cf.sig_str_fcn(ks_val, p_val))
                        ks_stats[i_grp, j_grp] = ks_stats[j_grp, i_grp] = ks_stats_nw

            # returns the final stats array
            return ks_stats

        ###################################
        ####    DATA PRE-PROCESSING    ####
        ###################################

        # initialisations
        r_data = self.data.rotation
        is_hist = dist_type == 'Histogram'
        i_grp = ['Negative', 'Positive'].index(vel_dir)

        # sets the cdf xi-values
        xi_cdf = np.linspace(-1, 1, int(2 / bin_size) + 1)
        x_cdf = 0.5 * (xi_cdf[:-1] + xi_cdf[1:])

        if is_fixed:
            ###################################
            ####    FIXED HEAD ANALYSIS    ####
            ###################################

            # if there was an error setting up the rotation calculation object, then exit the function with an error
            r_obj_wc = RotationFilteredData(self.data, rot_filt, None, None, True, 'Whole Experiment', False)
            if not r_obj_wc.is_ok:
                self.calc_ok = False
                return

            # memory allocation
            n_filt = r_obj_wc.n_filt
            A = np.empty(n_filt, dtype=object)
            v_sf_sig, v_sf_corr, v_sf_score, sf_cdf = dcopy(A), dcopy(A), dcopy(A), dcopy(A)

            # retrieves the indices of the cells that are common across all trial types
            t_type_filt = rot_filt['t_type']
            t_type_full = [x['t_type'][0] for x in r_obj_wc.rot_filt_tot]
            i_cell_b, r_obj_tt = cfcn.get_common_filtered_cell_indices(self.data, r_obj_wc, t_type_full, True)

            # sets the spiking frequency significance/correlation
            for i_filt, rr in enumerate(r_obj_wc.rot_filt_tot):
                # retrieves the cell indices that correspond to the current filter
                ind_cell = i_cell_b[i_filt]

                # retrieves the significance flags/correlation values for each cell
                v_sf_sig[i_filt] = r_data.vel_sf_sig[rr['t_type'][0]][i_cell_b[i_filt], :]
                v_sf_corr[i_filt] = r_data.vel_sf_corr_mn[rr['t_type'][0]][ind_cell, i_grp]

                # calculates the significance scores
                if r_data.split_vel:
                    v_sf_score[i_filt] = v_sf_sig[i_filt][:, 0].astype(int) + 2 * v_sf_sig[i_filt][:, 1].astype(int)
                else:
                    v_sf_score[i_filt] = v_sf_sig[i_filt].astype(int)

        else:
            ######################################
            ####    FREELY MOVING ANALYSIS    ####
            ######################################

            # initialisations
            i_bin = ['5', '10'].index(r_data.vel_bin_corr)
            t_type_filt = dcopy(rot_filt)

            # determines the number of selected trial types
            v_sf_sig, v_sf_corr = self.get_free_moving_data(t_type_filt, i_bin, i_grp)
            if v_sf_sig is None:
                # if no trial types are selected, then output an error to screen
                e_str = 'Error! At least one trial type must be selected to run this function.'
                cf.show_error(e_str, 'Incorrect Function Parameters')

                # exits function with an error flag
                self.calc_ok = False
                return

        #####################################
        ####    DISTRIBUTION SUBPLOTS    ####
        #####################################

        # parameters
        p_wid = 0.85
        n_filt = len(v_sf_sig)
        sf_cdf = np.empty(n_filt, dtype=object)

        # sets the legend strings
        if is_fixed:
            lg_str_d0 = [', '.join(x.split('\n')) for x in r_obj_wc.lg_str] if comb_all else t_type_filt
        else:
            lg_str_d0 = t_type_filt
        lg_str_d = ['#{0} - {1}'.format(i + 1, x) for i, x in enumerate(lg_str_d0)]

        # determines the number of the trial types/groups
        n_tt = len(t_type_filt)
        n_grp = int(n_filt / n_tt)
        h_plt, col = [], cf.get_plot_col(n_filt) if comb_all else cf.get_plot_col(n_tt) * n_grp

        # sets up the subplot axis
        n_plot = 1 if comb_all else n_grp
        setup_plot_axes(self.plot_fig, n_plot)
        ax = self.plot_fig.ax

        # sets the spiking frequency significance/correlation
        for i_filt in range(n_filt):
            # sets the axis
            i_axD = 0 if comb_all else int(i_filt / n_tt)
            i_tt = i_filt if comb_all else (i_filt % n_tt)

            # calculates the cumulative distribution values
            sf_corr_hist = np.histogram(v_sf_corr[i_filt], bins=xi_cdf, normed=False)[0]
            sf_corr_hist_sum = np.sum(sf_corr_hist)
            sf_cdf[i_filt] = 100. * np.cumsum(sf_corr_hist / sf_corr_hist_sum)

            # creates the subplot based on the type
            if is_hist:
                # sets the bar graph offsets
                if i_filt == 0:
                    n_wid = n_filt if comb_all else n_tt
                    dx = np.linspace(-1, 1, 2 * n_wid + 1)[1::2]
                    b_wid = p_wid * (x_cdf[1] - x_cdf[0]) / 2

                # case is plotting the histogram
                sf_corr_hist_sig = np.histogram(v_sf_corr[i_filt][v_sf_sig[i_filt][:, i_grp]],
                                                bins=xi_cdf, normed=False)[0]

                # calculates the proportions
                p_sig = 100. * sf_corr_hist_sig / sf_corr_hist_sum
                p_nsig = 100. * (sf_corr_hist - sf_corr_hist_sig) / sf_corr_hist_sum

                # case is the significant values so normalise using the provided value
                x_cdf_tt = x_cdf + dx[i_tt] * b_wid
                ax[i_axD].bar(x_cdf_tt, p_sig, width=2 * b_wid / n_wid, edgecolor=col[i_filt], color=col[i_filt])
                ax[i_axD].bar(x_cdf_tt, p_nsig, width=2 * b_wid / n_wid, bottom=p_sig, edgecolor=col[i_filt], color='None')
                h_plt.append(ax[i_axD].bar(-10, 0.01, edgecolor=col[i_filt], color=col[i_filt]))

            else:
                # case is plotting the cumulative distribution
                h_plt.append(ax[i_axD].plot(x_cdf, sf_cdf[i_filt], c=col[i_filt]))

        # sets the title strings (based on the type)
        if is_fixed:
            if ('\n' not in r_obj_wc.lg_str[0].replace('\n', '')) or comb_all:
                # case is no special filter has been applied (except for trial type) or all filters are combined
                t_str = ['All Cells']
            else:
                # case is there is some sort of additional filter being applied
                t_str = [', '.join(x.replace('\n', '').split('\n')[:-1]) for x in r_obj_wc.lg_str]
        else:
            # case is freely moving cells
            t_str = ['All Cells']

        # sets the properties for each of the subplot axes
        for i_ax, _ax in enumerate(ax[:n_plot]):
            _ax.set_ylabel('Percentage')
            _ax.set_title(t_str[2 * i_ax])
            _ax.grid(plot_grid)

            # sets the axis limits and legend for the sub-plot
            if is_hist:
                # sets the y-axis limits (first subplot only)
                if i_ax == 0:
                    yL_min = np.min([_ax.get_ylim()[0] for _ax in ax[:n_plot]])
                    yL_max = np.max([_ax.get_ylim()[1] for _ax in ax[:n_plot]])

                # case is the histogram distribution
                _ax.set_xlim([-1, 1])
                _ax.set_ylim([yL_min, yL_max])
                _ax.legend(h_plt, lg_str_d)

                # plots the histogram bin demarkation markers
                for xx in 0.5 * (x_cdf[:-1] + x_cdf[1:]):
                    _ax.plot([xx] * 2, [yL_min, yL_max], 'k--', linewidth=0.5)

            else:
                # sets the x-axis limits (first subplot only)
                if i_ax == 0:
                    xL_min = np.min([_ax.get_xlim()[0] for _ax in ax[:n_plot]])
                    xL_max = np.max([_ax.get_xlim()[1] for _ax in ax[:n_plot]])

                # case is the cumulative distribution
                _ax.set_xlim([xL_min, xL_max])
                _ax.set_ylim([-0.1, 100.1])
                _ax.legend([x[0] for x in h_plt], lg_str_d)

            # sets the x-axis label
            if (i_ax + 1) < n_plot:
                # case is the non-final row
                _ax.set_xticklabels([])
            else:
                # case is the final row
                _ax.set_xlabel('Correlation')

        #######################################
        ####    DISTRIBUTION STATISTICS    ####
        #######################################

        # sets the indices for each stats comparison group
        if comb_all:
            ind_grp = [np.arange(n_filt)]
        else:
            ind_grp = [np.arange(n_tt) + i_grp * n_tt for i_grp in range(n_grp)]

        # initialisations
        t_props = np.empty(len(ind_grp), dtype=object)
        cl_hdr = rw_hdr = ['#{0}'.format(i + 1) for i in range(len(lg_str_d))]

        #
        for i_grp, ig in enumerate(ind_grp):
            # sets up the statistic strings for the current group comparisons
            ks_stats = calc_ks_stats(sf_cdf[ig])

            # creates the table
            col_tab = list(np.array(col)[ig])
            t_props[i_grp] = cf.add_plot_table(self.plot_fig, ax[-1], table_font_small, ks_stats, rw_hdr,
                                               cl_hdr, col_tab, col_tab, None)

        # resets the table positions
        cf.reset_table_pos(self.plot_fig, ax[-1], t_props)

    def plot_freq_corr_scatter(self, rot_filt, x_plot, y_plot, comb_all, vel_dir, plot_grid, plot_scope, is_fixed):
        '''

        :param rot_filt:
        :param x_plot:
        :param y_plot:
        :param comb_all:
        :param vel_dir:
        :param plot_grid:
        :param plot_scope:
        :param is_fixed:
        :return:
        '''

        if x_plot == y_plot:
            # if the x/y axis trial types are the same, then output an error to screen
            e_str = 'The trial types for the X and Y axis must be unique to run this function.'
            cf.show_error(e_str, 'Incorrect Plotting Parameters')

            # exits the function with an error flag
            self.calc_ok = False
            return

        ###################################
        ####    DATA PRE-PROCESSING    ####
        ###################################

        # initialisations
        r_data = self.data.rotation
        tt_plot = [x_plot, y_plot]
        i_grp = ['Negative', 'Positive'].index(vel_dir)

        if is_fixed:
            ###################################
            ####    FIXED HEAD ANALYSIS    ####
            ###################################

            # if there was an error setting up the rotation calculation object, then exit the function with an error
            r_obj_wc = RotationFilteredData(self.data, rot_filt, None, None, True, 'Whole Experiment', False)
            if not r_obj_wc.is_ok:
                # if there was an error, then exit with an error flag
                self.calc_ok = False
                return

            # retrieves the indices of the cells that are common across all trial types (for the selected trial types)
            t_type_info = np.vstack([[i, x['t_type'][0]] for i, x in
                                     enumerate(r_obj_wc.rot_filt_tot) if (x['t_type'][0] in tt_plot)])
            i_cell_b, r_obj_tt = cfcn.get_common_filtered_cell_indices(self.data, r_obj_wc, t_type_info[:, 1], True)

            # memory allocation
            n_plot_tt = len(i_cell_b)
            A = np.empty(n_plot_tt, dtype=object)
            v_sf_sig, v_sf_corr = dcopy(A), dcopy(A)

            # sets the spiking frequency significance/correlation values for the selected trial types
            for i_filt, tt in enumerate(t_type_info[:, 1]):
                # retrieves the cell indices that correspond to the current filter
                ind_cell = i_cell_b[i_filt]

                # retrieves the significance flags/correlation values for each cell
                v_sf_sig[i_filt] = r_data.vel_sf_sig[tt][i_cell_b[i_filt], i_grp]
                v_sf_corr[i_filt] = r_data.vel_sf_corr_mn[tt][ind_cell, i_grp]

        else:
            ######################################
            ####    FREELY MOVING ANALYSIS    ####
            ######################################

            # initialisations
            i_bin = ['5', '10'].index(r_data.vel_bin_corr)
            t_type_filt = dcopy(rot_filt)
            n_plot_tt = 2

            # determines the number of selected trial types
            v_sf_sig, v_sf_corr = self.get_free_moving_data(t_type_filt, i_bin, i_grp, use_all_sig=False)
            if v_sf_sig is None:
                # if no trial types are selected, then output an error to screen
                e_str = 'Error! At least one trial type must be selected to run this function.'
                cf.show_error(e_str, 'Incorrect Function Parameters')

                # exits function with an error flag
                self.calc_ok = False
                return

        ########################################
        ####    CORRELATION SCATTERPLOTS    ####
        ########################################

        # parameters
        m_size, mlt = 20, 3

        # initialises the plot axes
        n_grp, n_tt = int(n_plot_tt / 2), 2
        n_plot = 1 if comb_all else int(n_plot_tt / n_tt)
        n_row, n_col = int(np.ceil(n_plot / 2)), 1 if (n_plot == 1) else 2
        self.init_plot_axes(n_plot=n_plot, n_row=n_row, n_col=n_col)
        ax = self.plot_fig.ax

        # determines the number of the trial types/groups
        h_sig, col = [], cf.get_plot_col(n_plot_tt) if comb_all else cf.get_plot_col(n_tt) * n_grp
        ind_grp = [np.arange(n_tt) + i_grp * n_tt for i_grp in range(n_grp)]

        # significance colours
        sig_col = [cf.convert_rgb_col([147, 149, 152])[0],      # non-significant markers
                   cf.convert_rgb_col(_green)[0],               # black-only significant markers
                   cf.convert_rgb_col(_bright_purple)[0],       # uniform-only significant markers
                   cf.convert_rgb_col(_bright_red)[0]]          # both condition significant spikes

        # sets the title strings (based on the type)
        if is_fixed:
            n_str = int(len(r_obj_wc.lg_str) / n_grp)
            if ('\n' not in r_obj_wc.lg_str[0].replace('\n', '')) or comb_all:
                # case is no special filter has been applied (except for trial type) or all filters are combined
                t_str = ['All Cells']
            else:
                # case is there is some sort of additional filter being applied
                t_str = [', '.join(x.replace('\n', '').split('\n')[:-1]) for x in r_obj_wc.lg_str]

        else:
            n_str = 1
            t_str = ['All Cells']

        for i_grp in range(n_grp):
            # sets the plot index
            i_plot = 0 if comb_all else i_grp

            # sets the correlation/significance values
            x_sig, y_sig = v_sf_sig[ind_grp[i_grp][0]], v_sf_sig[ind_grp[i_grp][1]]
            x_corr, y_corr = v_sf_corr[ind_grp[i_grp][0]], v_sf_corr[ind_grp[i_grp][1]]
            sig_score = x_sig + 2 * y_sig

            # plots the significant values
            for i_sig in range(1, 4):
                # creates the legend markers (first subplot only)
                if i_plot == 0:
                    h_sig.append(ax[i_plot].scatter(-2, -2, marker='o', s=mlt * m_size, facecolor=sig_col[i_sig]))

                # determines if there are any significant cells
                is_sig = sig_score == i_sig
                if np.any(is_sig):
                    # if there are significant cells, then
                    ax[i_plot].scatter(x_corr[is_sig], y_corr[is_sig], marker='o',
                                       s=mlt * m_size, facecolor=sig_col[i_sig])

            # plots the scatterplot values
            ax[i_plot].scatter(x_corr, y_corr, marker='o', s=m_size, facecolor=sig_col[0])
            ax[i_plot].plot([-1, 1], [0, 0], 'r--')
            ax[i_plot].plot([0, 0], [-1, 1], 'r--')
            cf.set_axis_limits(ax[i_plot], [-1, 1], [-1, 1])

            # sets the other axis properties
            ax[i_plot].set_title(t_str[n_str * i_plot])
            ax[i_plot].grid(plot_grid)

            # creates the legend (first subplot only)
            if ((i_plot == 0) and comb_all) or (not comb_all):
                ax[i_plot].set_xlabel('{0} Correlation'.format(x_plot))
                ax[i_plot].set_ylabel('{0} Correlation'.format(y_plot))
                ax[i_plot].legend(h_sig, ['{0} Sig.'.format(x_plot), '{0} Sig.'.format(y_plot), 'Both Sig.'])

    def plot_freq_corr_significance(self, rot_filt, grp_plot_type, plot_grid, p_value, grp_by_filt,
                                    show_stats, plot_scope, is_fixed):
        '''

        :param rot_filt:
        :param grp_plot_type:
        :param plot_grid:
        :param p_value:
        :param grp_by_filt:
        :param show_stats:
        :param plot_scope:
        :param is_fixed:
        :return:
        '''

        def calc_kw_stats(p_sig, n_filt):
            '''

            :param p_sig:
            :return:
            '''

            # parameters
            p_val = 0.05

            # initialisations
            kw_stats = np.empty((n_filt, n_filt), dtype=object)

            for i_filt in range(n_filt):
                for j_filt in range(i_filt, n_filt):
                    if i_filt == j_filt:
                        kw_stats[i_filt, j_filt] = 'N/A'
                    elif np.array_equal(p_sig[j_filt], p_sig[i_filt]):
                        kw_stats[i_filt, j_filt] = kw_stats[j_filt, i_filt] = '1.000'
                    else:
                        _, kw_val = kruskal(p_sig[i_filt], p_sig[j_filt])
                        kw_str = '{:.3f}{}'.format(kw_val, cf.sig_str_fcn(kw_val, p_val))
                        kw_stats[i_filt, j_filt] = kw_stats[j_filt, i_filt] = kw_str

            # returns the stats array
            return kw_stats

        def setup_plot_axis(plot_fig, n_filt):
            '''

            :param plot_fig:
            :param n_filt:
            :return:
            '''

            # sets up the axes dimensions
            n_r, n_c = 2, 1
            top, bottom, wspace, hspace = 0.925, 0.04, 0.25, 0.05
            tbl_hght = 0.05 + 0.025 * n_filt
            height_ratios = [1 - tbl_hght, tbl_hght]

            # creates the gridspec object
            gs = gridspec.GridSpec(n_r, n_c, width_ratios=[1 / n_c] * n_c, height_ratios=height_ratios,
                                   figure=plot_fig.fig, wspace=wspace, hspace=hspace, left=0.075, right=0.98,
                                   bottom=bottom, top=top)

            # creates the subplots
            plot_fig.ax = np.empty(n_r * n_c, dtype=object)
            plot_fig.ax[0] = plot_fig.figure.add_subplot(gs[0, 0])
            plot_fig.ax[1] = plot_fig.figure.add_subplot(gs[1, 0])

            # turns off axis for the table row
            plot_fig.ax[1].axis('off')

        ###################################
        ####    DATA PRE-PROCESSING    ####
        ###################################

        # checks to see if the current configuration is feasible (exit function if not)
        if not self.check_group_plot_feas(grp_plot_type, grp_by_filt, show_stats):
            return

        # initialisations
        n_grp = [4]
        r_data = self.data.rotation

        if is_fixed:
            ###################################
            ####    FIXED HEAD ANALYSIS    ####
            ###################################

            # if there was an error setting up the rotation calculation object, then exit the function with an error
            r_obj_wc = RotationFilteredData(self.data, rot_filt, None, None, True, 'Whole Experiment', False)
            if not r_obj_wc.is_ok:
                # if there was an error, then exit with an error flag
                self.calc_ok = False
                return

            # retrieves the indices of the cells that are common across all trial types
            n_filt, lg_str = r_obj_wc.n_filt, r_obj_wc.lg_str
            t_type_full = [x['t_type'][0] for x in r_obj_wc.rot_filt_tot]
            i_cell_b, r_obj_tt = cfcn.get_common_filtered_cell_indices(self.data, r_obj_wc, t_type_full, True)
            i_glob, i_expt_int = cf.get_global_index_arr(r_obj_wc, False)
            i_loc = cf.get_global_index_arr(r_obj_wc, False, i_expt_int)

            # memory allocation
            n_filt_ex = np.zeros(n_filt, dtype=int)
            n_cell_ex0 = np.empty(n_filt, dtype=object)
            n_type_ex0 = np.empty(n_filt, dtype=object)

            # sets the spiking frequency significance values
            for i_filt, rr in enumerate(r_obj_wc.rot_filt_tot):
                # retrieves the indices of the cells within each experiment
                i_ex = i_loc[i_filt]
                n_filt_ex[i_filt] = len(i_ex)
                n_cell_ex0[i_filt] = repmat(np.array([len(x) for x in i_ex]).reshape(-1, 1), 1, 3)

                # retrieves the significance flags for the current filter type
                v_sf_sig_nw = r_data.vel_sf_sig[rr['t_type'][0]][i_cell_b[i_filt], :]

                # determines the number of cells for each significance type:
                #   =0 - No direction is significance
                #   =1 - Negative direction only is significance
                #   =2 - Positive direction only is significance
                #   =3 - Both directions are significance
                v_sf_sig_score = v_sf_sig_nw[:, 0] + 2 * v_sf_sig_nw[:, 1]
                v_sf_sig_ex = [v_sf_sig_score[_i_ex] for _i_ex in i_ex]
                n_type_ex0[i_filt] = np.vstack([[sum(v_sf == i) for i in range(1, 4)] for v_sf in v_sf_sig_ex])

        else:
            ######################################
            ####    FREELY MOVING ANALYSIS    ####
            ######################################

            # initialisations
            i_bin = ['5', '10'].index(r_data.vel_bin_corr)
            t_type_filt = lg_str = dcopy(rot_filt)
            n_filt = len(t_type_filt)

            # determines the number of selected trial types
            v_sf_sig, _ = self.get_free_moving_data(t_type_filt, i_bin, 0, stack_arr=False)
            if v_sf_sig is None:
                # if no trial types are selected, then output an error to screen
                e_str = 'Error! At least one trial type must be selected to run this function.'
                cf.show_error(e_str, 'Incorrect Function Parameters')

                # exits function with an error flag
                self.calc_ok = False
                return

            # memory allocation
            n_type_ex0, n_cell_ex0 = np.empty(n_filt, dtype=object), np.empty(n_filt, dtype=object)
            for i_filt in range(n_filt):
                v_sf_sig_ex = [x[:, 0] + 2 * x[:, 1] for x in v_sf_sig[i_filt]]
                n_cell_ex0[i_filt] = repmat(np.array([np.shape(v_sf)[0] for v_sf in v_sf_sig[0]]).reshape(-1, 1), 1, 3)
                n_type_ex0[i_filt] = np.vstack([[sum(v_sf == i) for i in range(1, 4)] for v_sf in v_sf_sig_ex])

        # includes the any significant column within the data
        for i_filt in range(n_filt):
            n_type_ex0[i_filt] = np.hstack((n_type_ex0[i_filt], np.sum(n_type_ex0[i_filt], axis=1).reshape(-1, 1)))
            n_cell_ex0[i_filt] = np.hstack((n_cell_ex0[i_filt], n_cell_ex0[i_filt][:, 0].reshape(-1, 1)))

        # calculates the proportions of the groups over each experiment/filter type
        sf_type_pr = [100. * np.divide(n_type_ex0, n_cell_ex0)]
        if not grp_by_filt:
            sf_type_pr = [self.reorder_array_dim(sf_type_pr[0])]

        # calculates the group posthoc statistics
        stats_ph = self.calc_group_posthoc_stats(sf_type_pr, n_filt, n_grp)

        #################################################
        ####    CORRELATION SIGNIFICANCE SUBPLOTS    ####
        #################################################

        # sets the legend strings based on the type
        tt_class = ['Negative', 'Positive', 'Both', 'Any Significant']
        if (n_filt == 1) and (lg_str[0] == 'Black'):
            tt_filt = ['(#1) - All Cells']
        else:
            tt_filt = ['(#{0}) - {1}'.format(i + 1, '/'.join(lg.split('\n'))) for i, lg in enumerate(lg_str)]

        if show_stats:
            ##########################################################
            ####    CORRELATION SIGNIFICANCE STATISTICS TABLES    ####
            ##########################################################

            # sets the table/plot colours
            col = cf.get_plot_col(max([max(n_grp), n_filt]), 1)

            # sets up the plot axes
            main_title = ['Correlation Significance']
            self.setup_posthoc_stats_table_axes(self.plot_fig, main_title, n_filt, n_grp)

            # sets the table header strings
            hdr_class = ['Neg.', 'Pos.', 'Both', 'Any']
            if (n_filt == 1) and (lg_str[0] == 'Black'):
                hdr_filt = ['All Cells']
            else:
                hdr_filt = ['(#{0})'.format(i + 1) for i in range(n_filt)]

            # sets the base title string
            if n_filt == 1:
                hdr_str, t_str = [hdr_filt], [tt_class]
            elif grp_by_filt:
                t_str, hdr_str = [tt_class, tt_filt], [hdr_filt, hdr_class]
            else:
                t_str, hdr_str = [tt_filt, tt_class], [hdr_class, hdr_filt]

            # creates the posthoc statistics table
            self.create_posthoc_stats_table(stats_ph, hdr_str, t_str, col, p_value)

        else:
            ##########################################################
            ####    CORRELATION SIGNIFICANCE PROPORTION FIGURE    ####
            ##########################################################

            # sets the legend/x-ticklabels based on the grouping type
            lg_str = tt_class if grp_by_filt else tt_filt
            x_ticklbl = tt_filt if grp_by_filt else tt_class

            # creates the subplot axis
            setup_plot_axis(self.plot_fig, n_filt)
            ax = self.plot_fig.ax

            # creates the graph
            c = cf.get_plot_col(np.max([n_filt, max(n_grp)]))
            h_plt = cf.create_general_group_plot(ax[0], sf_type_pr[0], grp_plot_type, c)

            # creates the xticklabels
            ax[0].set_xticklabels(x_ticklbl)

            # updates the axis properties
            ax[0].grid(plot_grid)
            ax[0].set_ylabel('Population %')
            ax[0].legend([x[0] for x in h_plt], lg_str, ncol=max([2, int(len(lg_str) / 2)]), loc='upper center',
                                 columnspacing=0.125, bbox_to_anchor=(0.5, 1.075))

            # sets the y-axis limits based on type
            if 'Bar' in grp_plot_type:
                ax[0].set_ylim([0, 100])
            else:
                ax[0].set_ylim([-2.5, 102.5])

            ####################################################
            ####    CORRELATION SIGNIFICANCE COUNT TABLE    ####
            ####################################################

            if (n_filt == 1) and (lg_str[0] == 'Black'):
                tt_filt_N = ['All Cells']
            else:
                tt_filt_N = ['(#{0})'.format(i + 1) for i in range(n_filt)]

            # table parameters
            t_font, tot_col = cf.get_table_font_size(2), [(0.75, 0.75, 0.75)]
            col_hdr, row_hdr = ['None'] + tt_class + ['Total Cell Count'], tt_filt_N + ['Total Count']
            col_table = cf.get_plot_col(max([len(col_hdr), len(row_hdr)]))

            # calculates the total cell counts (over all filter types/classification groups)
            sf_type_N0 = np.vstack([np.concatenate((np.sum(n_t, axis=0), [np.sum(n_ex[:, 0])]))
                                    for n_t, n_ex in zip(n_type_ex0, n_cell_ex0)])
            sf_type_N0 = np.hstack(((sf_type_N0[:,-1] - np.sum(sf_type_N0[:, :n_grp[0]-1],
                                    axis=1)).reshape(-1, 1), sf_type_N0))   # REMOVE THIS LINE FOR NONE COLUMN
            sf_type_N = np.vstack((sf_type_N0, np.sum(sf_type_N0, axis=0)))

            # creates the table
            cf.add_plot_table(self.plot_fig, ax[1], t_font, sf_type_N, row_hdr, col_hdr,
                              col_table[:n_filt] + tot_col, col_table[:(len(col_hdr) - 1)] + tot_col, 'fixed')

    #############################################
    ####    ROTATIONAL ANALYSIS FUNCTIONS    ####
    #############################################

    def plot_rotation_trial_spikes(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, show_pref_dir,
                                   n_bin, show_err, plot_grid):
        '''

        :param plot_scope:
        :return:
        '''

        # applies the rotation filter to the dataset
        r_obj = RotationFilteredData(self.data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, False)
        self.create_raster_hist(r_obj, n_bin, show_pref_dir, show_err, plot_grid)

    def plot_phase_spike_freq(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, ms_prop, grp_plot_type,
                                    plot_scope, plot_trend, m_size, plot_grid, p_value, grp_by_filt, show_stats):
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

        # checks to see if the current configuration is feasible (exit function if not)
        if not self.check_group_plot_feas(grp_plot_type, grp_by_filt, show_stats):
            return

        # creates the spike frequency plot/statistics tables
        self.create_spike_freq_plot(r_obj, plot_grid, plot_trend, p_value, grp_plot_type, ms_prop,
                                    m_size, grp_by_filt, show_stats)

    def plot_spike_freq_heatmap(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, norm_type,
                                mean_type, plot_scope, dt):
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
        self.create_spike_heatmap(r_obj, dt, norm_type, mean_type)

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
            self.plot_fig.fig.suptitle('Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0], int(r_obj.ch_id[0][0])),
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

    def plot_firing_rate_distributions(self, rot_filt, plot_exp_name, plot_all_expt, comp_type, n_smooth,
                                       smooth_hist, plot_grid):
        '''

        :param rot_filt:
        :param plot_exp_name:
        :param plot_all_expt:
        :param comp_type:
        :param plot_grid:
        :param plot_scope:
        :return:
        '''

        # ensures the smoothing window is an odd integer
        if smooth_hist:
            if n_smooth % 2 != 1:
                # if not, then exit with an error
                e_str = 'The median smoothing filter window span must be an odd integer.'
                cf.show_error(e_str, 'Incorrect Smoothing Window Span')
                self.calc_ok = False
                return

        # initialises the rotation filter (if not set)
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        r_obj = RotationFilteredData(self.data, rot_filt, None, plot_exp_name, plot_all_expt, 'Whole Experiment', False)
        if not r_obj.is_ok:
            self.calc_ok = False
            return

        # initialisations
        c_ind, A, h_plt = 1 + ('CCW' in comp_type), np.empty(r_obj.n_filt, dtype=object), []
        s_plt, sp_f_hist, dsp_f, dsp_f_hist = dcopy(A), dcopy(A), dcopy(A), dcopy(A)
        sp_f_mn, dsp_f_mn, sp_f_mx, dsp_f_mx  = 0.01, 1e6, -1e6, -1e6

        # calculates mean spiking rates for all cells
        _, sp_f = cf.calc_phase_spike_freq(r_obj)

        # sets the plot/table properties
        c = cf.get_plot_col(r_obj.n_filt)

        # initialises the plot axes
        # self.plot_fig.fig.set_tight_layout(False)
        # self.plot_fig.fig.tight_layout(rect=[0.01, 0.02, 0.98, 0.97])
        self.init_plot_axes(n_row=1, n_col=2)

        # calculates the difference in spiking rate between the comparison index and baseline phases
        for i_filt in range(r_obj.n_filt):
            # determines the neurons whose spiking rate is greater than minimum
            ii = sp_f[i_filt][:, 0] >= sp_f_mn

            # sets the mean baseline spiking rates (determining the min/max values)
            s_plt[i_filt] = sp_f[i_filt][ii, 0]
            sp_f_mx = max([sp_f_mx, np.max(s_plt[i_filt])])

            # sets the change in mean spiking rates (determining the min/max values)
            dsp_f[i_filt] = sp_f[i_filt][ii, c_ind] - s_plt[i_filt]
            dsp_f_mx = max([dsp_f_mx, np.max(dsp_f[i_filt])])
            dsp_f_mn = min([dsp_f_mn, np.min(dsp_f[i_filt])])

        #
        xi_f = np.arange(np.log10(sp_f_mn), 0.1 * (1 + np.ceil(np.log10(sp_f_mx) / 0.1)), 0.1)
        xi_df = np.arange(np.floor(dsp_f_mn), np.ceil(dsp_f_mx))

        # calculates the spiking rate/rate change histograms over all filter types
        for i_filt in range(r_obj.n_filt):
            # calculates the histograms for each of the filter options
            sp_f_hist_tmp = np.histogram(np.log10(s_plt[i_filt]), bins=xi_f)[0]
            sp_f_hist[i_filt] = sp_f_hist_tmp / np.sum(sp_f_hist_tmp)

            #
            dsp_f_hist_tmp = np.histogram(dsp_f[i_filt], bins=xi_df)[0]
            dsp_f_hist[i_filt] = dsp_f_hist_tmp / np.sum(dsp_f_hist_tmp)

        ############################
        ####    FIGURE SETUP    ####
        ############################

        # sets the x-axis plot points
        x_f = 10 ** (0.5 * (xi_f[1:] + xi_f[:-1]))
        dx_f, x_f = 0.5 * (xi_df[1:] + xi_df[:-1]), np.concatenate(([x_f[0]], x_f))

        # creates both histograms for each of the filter types
        for i_filt in range(r_obj.n_filt):
            # sets plot values (smoothes the data if required)
            sp_f_plt = sp_f_hist[i_filt]
            if smooth_hist:
                sp_f_plt = medfilt(sp_f_plt, n_smooth)

            # creates the area graphs for the baseline spiking rates
            sp_f_plt = np.concatenate(([0], sp_f_plt))
            h_plt.append(self.plot_fig.ax[0].fill_between(x_f, sp_f_plt, color=c[i_filt], alpha=0.4))
            self.plot_fig.ax[0].plot(x_f, sp_f_plt, color=c[i_filt], linewidth=1.5)

            # creates the area graphs for the baseline spiking rates
            self.plot_fig.ax[1].plot(dx_f, dsp_f_hist[i_filt], color=c[i_filt])

        # creates the legend object
        self.plot_fig.ax[0].legend(h_plt, r_obj.lg_str, loc=0)

        # sets the graph titles/axis properties
        self.plot_fig.ax[0].set_title("Neuron Firing Rate Histograms")
        self.plot_fig.ax[0].set_ylabel("Fraction of Neurons")
        self.plot_fig.ax[0].set_xlabel("Firing Rate (Spike/Sec)")
        self.plot_fig.ax[0].set_xlim([sp_f_mn, 100.])
        self.plot_fig.ax[0].set_xscale("log")

        # sets the graph titles/axis properties
        self.plot_fig.ax[1].set_title("Neuron Firing Rate Change")
        self.plot_fig.ax[1].set_ylabel("Fraction of Neurons")
        self.plot_fig.ax[1].set_xlabel("{0}Firing Rate (Spike/Sec)".format(cf._delta))

        # readjusts the y-axis limits of both graphs
        yL_0, yL_1 = self.plot_fig.ax[0].get_ylim(), self.plot_fig.ax[1].get_ylim()
        self.plot_fig.ax[0].set_ylim([0, yL_0[1]])
        self.plot_fig.ax[1].plot([0, 0], [0, yL_1[1]], 'r--')
        self.plot_fig.ax[1].set_ylim([0, yL_1[1]])

    def plot_spike_freq_kinematics(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, spread_type, plot_scope,
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

        def create_kinematic_plots(r_obj, b_sz, n_smooth, is_smooth, is_single_cell, plot_grid):
            '''

            :param r_obj:
            :param b_sz:
            :param n_smooth:
            :param is_smooth:
            :param is_single_cell:
            :param plot_grid:
            :return:
            '''

            def get_kinematic_plot_values(k_sf, i_plot, is_single_cell):
                '''

                :param k_sf:
                :param i_plot:
                :return:
                '''

                # removes any singleton dimensions
                if is_single_cell:
                    k_sf = np.squeeze(k_sf)

                # sets the temporary spiking frequency arrays
                if i_plot == 0:
                    # case is decreasing position
                    k_sf_raw = k_sf[:, :, 0]
                elif i_plot == 1:
                    # case is increasing position
                    k_sf_raw = k_sf[:, :, 1]
                else:
                    # case is pooled position
                    k_sf_raw = np.mean(k_sf, axis=2)

                # calculates the mean spiking frequency over all cells
                n_cell = np.size(k_sf_raw, axis=0)
                return k_sf_raw, np.mean(k_sf_raw, axis=0), np.std(k_sf_raw, axis=0) / (n_cell ** 0.5)

            def create_pos_polar_plots(ax, r_obj, k_sf, xi_bin, k_rng, b_sz, is_single_cell):
                '''

                :return:
                '''

                # initialisations
                n_plot, c, mlt = len(ax), cf.get_plot_col(r_obj.n_filt), -1
                t_str = ['Position ({0})'.format(x) for x in ['Decreasing', 'Increasing', 'Averaged']]
                xi_tot = np.hstack((xi_bin[:, 0], -xi_bin[0, 0]))

                # sets the bin values
                xi_mid, xi_min = np.mean(xi_bin, axis=1), xi_bin[0, 0]
                xi_mid = np.pi * (1 - (xi_mid - xi_min) / np.abs(2 * xi_min))
                x_tick = np.linspace(-k_rng, k_rng, 7)

                # creates the radial plots for each of the filter types
                for i_plot in range(n_plot):
                    # memory allocation
                    h_plt = []

                    for i_filt in range(r_obj.n_filt):
                        # retrieves the mean plot values
                        _, k_sf_mn, _ = get_kinematic_plot_values(k_sf[i_filt], i_plot, is_single_cell)

                        # creates the polar plot
                        d_xi = 0.5 * (xi_mid[0] - xi_mid[1]) * (((2 * i_filt + 1) / r_obj.n_filt) - 1)
                        h_plt.append(ax[i_plot].bar(xi_mid - d_xi, height=k_sf_mn, color=c[i_filt],
                                                    width=np.deg2rad(b_sz) / r_obj.n_filt, linewidth=0))

                        # sets the plot axis title/x-axis properties
                        if i_filt == 0:
                            ax[i_plot].set_title(t_str[i_plot])
                            ax[i_plot].set_xticks(np.pi * (x_tick - xi_min) / np.abs(2 * xi_min))
                            ax[i_plot].set_xticklabels([str(int(np.round(mlt * x))) for x in x_tick])

                    # sets the legend (first subplot only)
                    if i_plot == 0:
                        ax[i_plot].legend(r_obj.lg_str, loc=2)

                # determines the overall limits
                yL = [0, max([_ax.get_ylim()[1] for _ax in ax])]

                for _ax in ax:
                    # adds in the bin lines for the polar plot
                    _ax.set_ylim(yL)
                    for xi in (np.pi / 2) * (1 + (xi_tot / k_rng)):
                        _ax.plot([xi, xi], 2 * np.array(yL), 'k--')

                    # resets the axis limits
                    _ax.set_thetamin(0)
                    _ax.set_thetamax(180)

            def create_vel_line_plots(ax, r_obj, k_sf, xi_bin, k_rng, is_smooth, is_single_cell, plot_indiv):
                '''

                :param ax:
                :param r_obj:
                :param k_sf:
                :param xi_bin:
                :param k_rng:
                :param is_smooth:
                :param is_single_cell:
                :return:
                '''

                # initialisations
                n_plot, c, mlt, yL = len(ax), cf.get_plot_col(r_obj.n_filt), 1, [1e6, -1e6]
                t_str = ['Velocity ({0})'.format(x) for x in ['Decreasing', 'Increasing', 'Averaged']]
                sf_min, sf_max = 1e6, -1e6

                # sets the bin values
                xi_mid = np.mean(xi_bin, axis=1)
                x_tick = np.linspace(-k_rng, k_rng, 9)

                # creates the radial plots for each of the filter types
                for i_plot in range(n_plot):
                    # memory allocation
                    h_plt = []

                    # creates the plots for each of the
                    for i_filt in range(r_obj.n_filt):
                        # retrieves the plot values
                        k_sf_raw, k_sf_mn, k_sf_sem = get_kinematic_plot_values(k_sf[i_filt], i_plot, is_single_cell)

                        # smooths the signal (if required)
                        if is_smooth:
                            k_sf_mn = medfilt(k_sf_mn, n_smooth)

                        # creates the line plot
                        h_plt.append(ax[i_plot].plot(xi_mid, k_sf_mn, 'o-', color=c[i_filt]))
                        if is_single_cell and plot_indiv:
                            #
                            for i_trial in range(np.shape(k_sf_raw)[0]):
                                # outputs the trace
                                k_sf_raw_nw = k_sf_raw[i_trial, :]
                                if is_smooth:
                                    k_sf_raw_nw = medfilt(k_sf_raw_nw, n_smooth)

                                # creates the new raw trace
                                sf_min, sf_max = min([sf_min, np.min(k_sf_raw_nw)]), max([sf_max, np.max(k_sf_raw_nw)])
                                ax[i_plot].plot(xi_mid, k_sf_raw_nw, c[i_filt], linewidth=1, alpha=0.25)

                            # plots the mean spiking frequency values
                            ax[i_plot].plot(xi_mid, k_sf_mn, c[i_filt], linewidth=2)

                            # sets the axis limits
                            if (i_plot + 1) == n_plot:
                                yL = [sf_min - 0.05 * (sf_max - sf_min), sf_max + 0.05 * (sf_max - sf_min)]
                        else:
                            cf.create_error_area_patch(ax[i_plot], xi_mid, k_sf_mn, k_sf_sem, c[i_filt])

                            # determines the overall min/max axis values
                            yL = [min(0.95 * min(k_sf_mn - k_sf_sem), yL[0]), max(1.05 * max(k_sf_mn + k_sf_sem), yL[1])]

                        # sets the plot axis title/x-axis properties
                        if i_filt == 0:
                            ax[i_plot].set_title(t_str[i_plot])
                            ax[i_plot].set_xticks(x_tick)
                            ax[i_plot].set_xticklabels([str(int(np.round(mlt * x))) for x in x_tick])

                # resets the axis limits over all axes
                for _ax in ax:
                    _ax.set_ylim(yL)
                    _ax.grid(plot_grid)

            # if there was an error setting up the rotation calculation object, then exit the function with an error
            if not r_obj.is_ok:
                self.calc_ok = False
                return

            # initialisations
            plot_indiv = spread_type == 'Individual Trial Traces'
            c, k_rng = cf.get_plot_col(r_obj.n_filt), [90, 80]
            proj_type = ['polar', 'polar', 'polar', None, None, None]

            # calculates the position/velocity values over all trials/cells
            k_sf, xi_bin, _ = rot.calc_kinemetic_spike_freq(self.data, r_obj, b_sz, calc_type=2+is_single_cell)

            # creates the plot outlay and titles
            self.init_plot_axes(n_row=2, n_col=3, proj_type=proj_type)

            # creates the position polar/velocity line plots
            create_pos_polar_plots(self.plot_fig.ax[:3], r_obj, k_sf[0], xi_bin[0], k_rng[0], b_sz[0], is_single_cell)
            create_vel_line_plots(self.plot_fig.ax[3:], r_obj, k_sf[1], xi_bin[1], k_rng[1], is_smooth,
                                  is_single_cell, plot_indiv)

            # resets the subplot layout
            self.plot_fig.fig.set_tight_layout(False)
            if r_obj.is_single_cell:
                # sets the layout size
                self.plot_fig.fig.tight_layout(rect=[0, 0.01, 1, 0.955])

                # sets the cell cluster ID
                self.plot_fig.fig.suptitle('Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0],
                                           int(r_obj.ch_id[0][0])), fontsize=16, fontweight='bold')
            else:
                # sets the layout size
                self.plot_fig.fig.tight_layout(rect=[0, 0, 1, 1])

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

        # determines what analysis type is being used
        is_single_cell = plot_scope == 'Individual Cell'

        # applies the rotation filter to the dataset
        r_obj = RotationFilteredData(self.data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, False)
        create_kinematic_plots(r_obj, [float(pos_bin), float(vel_bin)], n_smooth, is_smooth, is_single_cell, plot_grid)

    def plot_overall_direction_bias(self, rot_filt, grp_plot_type, plot_grid, p_value, grp_by_filt, show_stats, plot_scope,
                                    plot_exp_name, plot_all_expt):
        '''

        :param rot_filt:
        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_grid:
        :return:
        '''

        def setup_plot_axis(plot_fig, n_filt):
            '''

            :param plot_fig:
            :param n_filt:
            :return:
            '''

            # sets up the axes dimensions
            n_r, n_c = 2, 1
            top, bottom, wspace, hspace = 0.925, 0.04, 0.25, 0.05
            tbl_hght = 0.05 + 0.025 * n_filt
            height_ratios = [1 - tbl_hght, tbl_hght]

            # creates the gridspec object
            gs = gridspec.GridSpec(n_r, n_c, width_ratios=[1 / n_c] * n_c, height_ratios=height_ratios,
                                   figure=plot_fig.fig, wspace=wspace, hspace=hspace, left=0.075, right=0.98,
                                   bottom=bottom, top=top)

            # creates the subplots
            plot_fig.ax = np.empty(n_r * n_c, dtype=object)
            plot_fig.ax[0] = plot_fig.figure.add_subplot(gs[0, 0])
            plot_fig.ax[1] = plot_fig.figure.add_subplot(gs[1, 0])

            # turns off axis for the table row
            plot_fig.ax[1].axis('off')

        # checks to see if the current configuration is feasible (exit function if not)
        if not self.check_group_plot_feas(grp_plot_type, grp_by_filt, show_stats):
            return

        # initialises the rotation filter (if not set)
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        r_obj = RotationFilteredData(self.data, rot_filt, None, plot_exp_name, plot_all_expt, 'Whole Experiment', False)
        if not r_obj.is_ok:
            self.calc_ok = False
            return

        # memory allocation and parameters
        n_grp, ind = [1], np.arange(1, 2)
        p_lim, n_filt = 0.05, r_obj.n_filt
        i_glob, _ = cf.get_global_index_arr(r_obj)
        main_title = ['CCW Preferred Direction Proportion']
        lg_str_f = r_obj.lg_str

        # retrieves the spiking frequency values
        _, _sp_f = cf.calc_phase_spike_freq(r_obj)

        # calculates the proportions of the preferred directions for each experiment (overall filter types)
        #  =0 - the clockwise direction is preferred
        #  =1 - the counter-clockwise direction is preferred
        p_pref_dir0, n_pref_dir = np.empty(n_filt, dtype=object), np.empty(n_filt, dtype=object)
        for i_filt in range(n_filt):
            # retrieves the spiking frequencies/global indices for the current filter type
            sp_f, ig = _sp_f[i_filt], i_glob[i_filt]

            # calculates the preferred directions proportions for each experiment (pad any missing rows with NaN's)
            n_pref_dir[i_filt] = np.vstack(
                [cf.calc_rel_prop((sp_f[_ig, 2] / sp_f[_ig, 1] > 1).astype(int),
                                            2, return_counts=True) for _ig in ig if len(_ig)]
            )
            p_pref_dir0[i_filt] = np.vstack(
                [cf.calc_rel_prop((sp_f[_ig, 2] / sp_f[_ig, 1] > 1).astype(int), 2, ind=ind) for _ig in ig if len(_ig)]
            )

        # sets the preferred direction proportions
        p_pref_dir = [dcopy(p_pref_dir0)]
        if not grp_by_filt:
            # if grouping the classification group, then re-order
            p_pref_dir = [self.reorder_array_dim(dcopy(p_pref_dir0))]

        # calculates the group posthoc statistics
        stats_ph = self.calc_group_posthoc_stats(p_pref_dir, n_filt, [1])

        # sets the legend strings based on the type
        if (n_filt == 1) and (lg_str_f[0] == 'Black'):
            tt_filt = ['(#1) - All Cells']
        else:
            tt_filt = ['(#{0}) - {1}'.format(i + 1, '/'.join(lg.split('\n'))) for i, lg in enumerate(lg_str_f)]

        if show_stats:
            #################################
            ####    STATISTICS TABLES    ####
            #################################

            # sets up the plot axes
            col_table = cf.get_plot_col(max([max(n_grp), n_filt]), 1)
            self.setup_posthoc_stats_table_axes(self.plot_fig, main_title, n_filt, n_grp)

            # sets the table header strings
            hdr_str = ['(#{0})'.format(i + 1) for i in range(n_filt)]
            t_str = ['Preferred CCW Direction']

            # sets the final header/title strings
            if n_filt == 1:
                return

            # creates the posthoc statistics table
            self.create_posthoc_stats_table(stats_ph, [hdr_str], [t_str], col_table, p_value)

        else:
            #####################################################
            ####    PREFERRED DIRECTION PROPORTION FIGURE    ####
            #####################################################

            # sets the legend/x-ticklabels based on the grouping type
            lg_str = main_title if grp_by_filt else tt_filt
            x_ticklbl = tt_filt if grp_by_filt else main_title

            # creates the subplot axis
            setup_plot_axis(self.plot_fig, n_filt)
            ax = self.plot_fig.ax

            # creates the graph
            c = cf.get_plot_col(np.max([n_filt, max(n_grp)]))
            h_plt = cf.create_general_group_plot(ax[0], p_pref_dir[0], grp_plot_type, c)

            # creates the xticklabels
            ax[0].set_xticklabels(x_ticklbl)

            # updates the axis properties
            ax[0].grid(plot_grid)
            ax[0].set_ylabel('Population %')

            if len(lg_str) > 1:
                ax[0].legend([x[0] for x in h_plt], lg_str, ncol=max([2, int(len(lg_str) / 2)]), loc='upper center',
                                     columnspacing=0.125, bbox_to_anchor=(0.5, 1.075))
            else:
                ax[0].set_title('Preferred CCW Direction Proportion', fontsize=18, fontweight='bold')

            # sets the y-axis limits based on type
            if 'Bar' in grp_plot_type:
                ax[0].set_ylim([0, 100])
            else:
                ax[0].set_ylim([-2.5, 102.5])

            ###############################################
            ####    PREFERRED DIRECTION COUNT TABLE    ####
            ###############################################

            if (n_filt == 1) and (lg_str_f[0] == 'Black'):
                tt_filt_N = ['All Cells']
            else:
                tt_filt_N = ['(#{0})'.format(i + 1) for i in range(n_filt)]

            # table parameter
            t_font, tot_col = cf.get_table_font_size(2), [(0.75, 0.75, 0.75)]
            col_hdr, row_hdr = ['CW Preferred', 'CCW Preferred', 'Total Cell Count'], tt_filt_N + ['Total Count']
            col_table = cf.get_plot_col(max([n_filt, len(col_hdr) - 1]))

            # calculates the total cell counts (over all filter types/classification groups)
            n_pref_dir0 = np.vstack([np.sum(x, axis=0) for x in n_pref_dir])
            n_pref_dir0 = np.vstack((n_pref_dir0, np.sum(n_pref_dir0, axis=0).reshape(1, -1)))
            n_pref_dir_fin = np.hstack((n_pref_dir0, np.sum(n_pref_dir0, axis=1).reshape(-1, 1))).astype('U50')

            # creates the count table
            cf.add_plot_table(self.plot_fig, ax[1], t_font, n_pref_dir_fin, row_hdr, col_hdr,
                              col_table[:n_filt] + tot_col, col_table[:(len(col_hdr) - 1)] + tot_col, 'fixed')

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

    def plot_unidrift_trial_spikes(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope,
                                   n_bin, show_err, plot_grid, rmv_median, show_pref_dir):
        '''

        :param rot_filter:
        :param plot_scope:
        :return:
        '''

        # applies the rotation filter to the dataset
        r_obj = RotationFilteredData(self.data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, True)
        self.create_raster_hist(r_obj, n_bin, show_pref_dir, show_err, plot_grid)

    def plot_unidrift_spike_freq(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, ms_prop, grp_plot_type,
                                 plot_scope, plot_trend, m_size, plot_grid, p_value, grp_by_filt, show_stats):
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

        # checks to see if the current configuration is feasible (exit function if not)
        if not self.check_group_plot_feas(grp_plot_type, grp_by_filt, show_stats):
            return

        # splits up the uniform drifting into CW/CCW phases
        t_phase, t_ofs = self.fcn_data.curr_para['t_phase_vis'], self.fcn_data.curr_para['t_ofs_vis']
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
        self.create_spike_freq_plot(r_obj, plot_grid, plot_trend, p_value, grp_plot_type, ms_prop,
                                    m_size, grp_by_filt, show_stats, ind_type=ind_type)

    def plot_unidrift_spike_heatmap(self, rot_filt, i_cluster, plot_exp_name, plot_all_expt, norm_type,
                                    mean_type, plot_scope, dt):
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
        self.create_spike_heatmap(r_obj, dt, norm_type, mean_type)

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

                        p_value_str = '{:5.3f}{}'.format(p_value, cf.sig_str_fcn(p_value, 0.05))
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
        roc_auc = [r_data.cond_roc_auc[x][y, 2] for x, y in zip(t_type, cell_ind)]
        roc_xy = [r_data.cond_roc_xy[x][y] for x, y in zip(t_type, cell_ind)]

        # if the black auROC value is <0.5, then calculate the complimentary auROC/coordinates
        i_black = t_type.index('Black')
        if roc_auc[i_black] < 0.5:
            roc_auc = [1.0 - x for x in roc_auc]
            roc_xy = [1.0 - x for x in roc_xy]

        # sets the confidence interval values into a list
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
        self.plot_fig.fig.suptitle('Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0], int(r_obj.ch_id[0][0])),
                                   fontsize=16, fontweight='bold')
        self.plot_fig.fig.tight_layout(rect=[0, 0.01, 1, 0.955])

        if r_obj.n_filt > 1:
            # sets the column/row headers
            auc_stats = calc_cond_significance(roc, r_data)
            row_hdr = col_hdr = ['#{0}'.format(str(x+1)) for x in range(n_filt)]

            # calculates the table dimensions
            cf.add_plot_table(self.plot_fig, 1, table_font, auc_stats, row_hdr, col_hdr, c, c,
                              'bottom', pfig_sz=0.955)

    def plot_direction_roc_curves_whole(self, rot_filt, plot_exp_name, plot_all_expt, use_avg, connect_lines, violin_bw,
                                        m_size, cell_grp_type, auc_plot_type, plot_grid, plot_grp_type, plot_scope):
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

        self.create_dir_roc_curve_plot(rot_filt, plot_exp_name, plot_all_expt, use_avg, connect_lines, violin_bw, m_size,
                                       plot_grp_type, cell_grp_type, auc_plot_type, plot_grid, plot_scope, False)

    def plot_direction_roc_auc_histograms(self, rot_filt, plot_exp_name, plot_all_expt, bin_sz, phase_type,
                                          show_sig_cells, plot_grid, plot_scope):
        '''

        :param rot_filt:
        :param plot_exp_name:
        :param plot_all_expt:
        :param cell_grp_type:
        :param plot_grid:
        :param plot_scope:
        :return:
        '''

        def create_auc_hist(ax, plt_vals, xi, c, is_sig, p_norm=None):
            '''

            :param ax:
            :param plt_vals:
            :param xi:
            :param c:
            :param is_sig:
            :param p_norm:
            :return:
            '''

            # sets the plot values
            xi_p = 0.5 * (xi[1:] + xi[:-1])

            # calculates the histogram values
            n_hist = np.histogram(plt_vals, bins=xi)[0]

            # creates the bar graph
            if is_sig:
                # case is the significant values so normalise using the provided value
                ax.bar(xi_p, n_hist / p_norm, width=xi[1] - xi[0], color=c)
            else:
                # calculates the normalised histogram values
                p_norm = np.sum(n_hist)
                p_hist = n_hist / p_norm

                # creates the bar plot
                ax.bar(xi_p, p_hist, width=xi[1] - xi[0], edgecolor=c, color='None')

                # sets the axis properties
                cf.set_axis_limits(ax, [0, 1], [0, 1.05 * max(p_hist)])

            # returns the normalisation value
            return p_norm

        # initialises the rotation filter (if not set)
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        r_obj = RotationFilteredData(self.data, rot_filt, None, plot_exp_name, plot_all_expt, 'Whole Experiment', False)
        if not r_obj.is_ok:
            self.calc_ok = False
            return

        # initialisations
        lg_str, n_filt, n_bin = r_obj.lg_str, r_obj.n_filt, int(100. / float(bin_sz))
        xi, r_data = np.linspace(0, 1, n_bin + 1), self.data.rotation
        c, yL_mx = cf.get_plot_col(n_filt), 0.
        f_keys = list(r_data.cond_roc_auc.keys())
        i_type = ['CW vs BL', 'CCW vs BL', 'CCW vs CW'].index(phase_type)

        # initialises the plot axis
        n_col, n_row = cf.det_subplot_dim(n_filt)
        self.init_plot_axes(n_plot=n_filt)

        #
        for i_col in range(n_col):
            for i_row in range(n_row):
                # sets the global plot index
                i_plot = i_row * n_col + i_col
                if i_plot == n_filt:
                    continue

                # retrieves the current trial type
                tt = f_keys[i_plot]

                # creates the auc histogram
                roc_auc = r_data.cond_roc_auc[tt][:, i_type]
                p_norm = create_auc_hist(self.plot_fig.ax[i_plot], roc_auc, xi, c[i_plot], False)

                # creates the significant auc histogram (if required)
                if show_sig_cells:
                    roc_auc_sig = roc_auc[r_data.cond_auc_sig[tt][:, i_type]]
                    create_auc_hist(self.plot_fig.ax[i_plot], roc_auc_sig, xi, c[i_plot], True, p_norm=p_norm)

                # sets the other axis properties
                self.plot_fig.ax[i_plot].grid(plot_grid)
                self.plot_fig.ax[i_plot].set_title(lg_str[i_plot])

                # sets the y-axis label (first column only)
                if i_col == 0:
                    self.plot_fig.ax[i_plot].set_ylabel('Fraction of Neurons')

                # sets the x-axis label (bottom row only)
                if (i_row + 1) == n_row:
                    self.plot_fig.ax[i_plot].set_xlabel('auROC')

                # determines the overall maximum
                yL_mx = max([yL_mx, self.plot_fig.ax[i_plot].get_ylim()[1]])

        # sets equal axis limits
        if n_filt > 1:
            for i_plot in range(n_filt):
                self.plot_fig.ax[i_plot].set_ylim([0, yL_mx])

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
            i_bin = list(r_data.vel_xi[:, 0]).index(int(vel_y_rng.split()[0]))
            _roc_xy, _roc_auc = dcopy(r_data.vel_roc_xy), dcopy(r_data.vel_roc_auc)
            _ci_lo, _ci_hi = dcopy(r_data.vel_ci_lo), dcopy(r_data.vel_ci_hi)
            xi_bin = dcopy(np.mean(r_data.vel_xi, axis=1))
        else:
            i_bin = list(r_data.spd_xi[:, 0]).index(int(spd_y_rng.split()[0]))
            _roc_xy, _roc_auc = dcopy(r_data.spd_roc_xy), dcopy(r_data.spd_roc_auc)
            _ci_lo, _ci_hi = dcopy(r_data.spd_ci_lo), dcopy(r_data.spd_ci_hi)
            xi_bin = dcopy(np.mean(r_data.spd_xi, axis=1))

        # determines the indices of the cell in the overall array
        B = np.zeros(len(xi_bin))
        t_type_base = list(r_data.spd_sf_rs.keys()) if r_data.is_equal_time else list(r_data.spd_sf.keys())
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
        self.plot_fig.fig.suptitle('Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0], int(r_obj.ch_id[0][0])),
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

    def plot_velocity_significance(self, rot_filt, plot_err, plot_exp_name, plot_all_expt, pool_expt, plot_cond, i_bin,
                                   use_vel, plot_type, plot_grid, plot_scope):
        '''

        :param rot_filt:
        :param plot_err:
        :param plot_exp_name:
        :param plot_all_expt:
        :param pool_expt:
        :param plot_cond:
        :param i_bin:
        :param use_vel:
        :param plot_type:
        :param plot_grid:
        :param plot_scope:
        :return:
        '''

        def create_sig_plot_axes(plot_fig):
            '''

            :return:
            '''

            # sets up the axes dimensions
            nC, top, bottom, wspace, hspace = 4, 0.95, 0.06, 0.2, 0.25

            # creates the gridspec object
            gs = gridspec.GridSpec(1, nC, width_ratios=[1 / nC] * nC, figure=plot_fig.fig, wspace=wspace,
                                   hspace=hspace, left=0.05, right=0.99, bottom=bottom, top=top)

            # creates the subplots
            ax = np.empty(2, dtype=object)
            ax[0] = plot_fig.figure.add_subplot(gs[0, :(nC-1)])
            ax[1] = plot_fig.figure.add_subplot(gs[0, -1])

            # returns the axis array
            return ax

        # initialises the rotation filter (if not set)
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        r_obj = RotationFilteredData(self.data, rot_filt, None, plot_exp_name, True, 'Whole Experiment', False)
        if not r_obj.is_ok:
            # sets the exit flag and exits the function
            self.calc_ok = False
            return

        # initialisations and other array indexing
        n_filt, r_data = r_obj.n_filt, self.data.rotation
        lg_str, c = r_obj.lg_str, cf.get_plot_col(n_filt)
        is_boot = int(r_data.kine_auc_stats_type == 'Bootstrapping')
        xi_bin = np.mean(r_data.vel_xi, axis=1) if use_vel else np.mean(r_data.spd_xi, axis=1)
        roc_sig_sem = None

        ###########################################
        ####    AUC ROC SIGNIFICANCE CURVES    ####
        ###########################################

        if plot_type == 'auROC Significance':

            ################################
            ####    PRE-CALCULATIONS    ####
            ################################

            # retrieves the trial type
            tt_calc = list(r_data.vel_roc_auc.keys())
            tt_robj = [x['t_type'][0] for x in r_obj.rot_filt_tot]

            # retrieves the roc significance values based on type and statistical calculation type
            roc_sig0 = dcopy(r_data.vel_roc_sig[:, is_boot]) if use_vel else dcopy(r_data.spd_roc_sig[:, is_boot])
            roc_sig = [roc_sig0[tt_calc.index(tt)] for tt in tt_robj]

            #
            i_expt = [np.unique(i_ex) for i_ex in r_obj.i_expt]
            n_c = np.array([x['nC'] for x in self.data.cluster[:-1]])
            c_ofs = [np.cumsum([0]+list(n_c[i_ex[:-1]])) for i_ex in i_expt]
            n_ex = [len(x) for x in i_expt]

            #
            if plot_all_expt:
                # retrieves the experiment significance values for each experiment/filter
                roc_sig_expt = np.empty(r_obj.n_filt, dtype=object)
                for i_filt in range(r_obj.n_filt):
                    roc_sig_expt[i_filt] = cfcn.calc_expt_roc_sig(r_obj, roc_sig[i_filt], c_ofs[i_filt], i_filt)

                # case is using all the experiment
                if pool_expt:
                    # pools all experiment data into a single group for mean calculations
                    roc_sig_mn = [100. * np.mean(np.vstack(r_sig), axis=0) for r_sig in roc_sig_expt]
                else:
                    # calculates the mean significance over each experiment
                    roc_sig_expt_mn = np.empty(n_filt, dtype=object)
                    for i_filt in range(n_filt):
                        roc_sig_expt_mn[i_filt] = np.empty(n_ex[i_filt], dtype=object)
                        for i_ex in range(n_ex[i_filt]):
                            roc_sig_expt_mn[i_filt][i_ex] = np.mean(roc_sig_expt[i_filt][i_ex], axis=0)

                    # calculates the final mean/SEM values over each filter
                    roc_sig_mn = [100. * np.nanmean(np.vstack(r_sig), axis=0) for r_sig in roc_sig_expt_mn]
                    roc_sig_sem = [100. * np.nanstd(np.vstack(r_sig), axis=0) / N ** 0.5 for r_sig, N in zip(roc_sig_expt_mn,n_ex)]

            else:
                # case is using an individual experiment
                i_expt = cf.get_expt_index(plot_exp_name, self.data.cluster)
                roc_sig_mn = np.empty(r_obj.n_filt, dtype=object)

                for i_filt in range(r_obj.n_filt):
                    if any(r_obj.i_expt[i_filt] == i_expt):
                        roc_sig_mn[i_filt] = cfcn.calc_expt_roc_sig(r_obj, roc_sig[i_filt], c_ofs[i_filt],
                                                                    i_filt, calc_mean=True, i_expt=[i_expt])[0]

            ################################
            ####    SUBPLOT CREATION    ####
            ################################

            # initialisations
            w_bar = 0.9
            c, h_plt = cf.get_plot_col(n_filt), []
            is_ok = np.zeros(r_obj.n_filt,dtype=bool)

            # initialises the plot axes
            self.plot_fig.ax = create_sig_plot_axes(self.plot_fig)
            ax = self.plot_fig.ax

            # plots the full auROC curves for each filter type
            for i_filt in range(n_filt):
                # plots the auc values
                if roc_sig_mn[i_filt] is not None:
                    is_ok[i_filt] = True
                    h_plt.append(ax[0].plot(xi_bin, roc_sig_mn[i_filt], 'o-', c=c[i_filt]))
                    if plot_err and (roc_sig_sem is not None):
                        cf.create_error_area_patch(ax[0], xi_bin, roc_sig_mn[i_filt], roc_sig_sem[i_filt], c[i_filt])

            # sets the axis properties
            ax[0].grid(plot_grid)
            ax[0].set_ylabel('%age Significant Cells')
            ax[0].legend([x[0] for x in h_plt], np.array(r_obj.lg_str)[is_ok], loc=0)
            cf.set_axis_limits(ax[0], [-80 * use_vel, 80], [0, ax[0].get_ylim()[1]])

            # sets the x-label depending on the speed type
            i_mu = np.ones(len(xi_bin), dtype=bool)
            if use_vel:
                i_mu[r_data.i_bin_vel] = False
                ax[0].set_xlabel('Velocity (deg/s)')
            else:
                i_mu[r_data.i_bin_spd] = False
                ax[0].set_xlabel('Speed (deg/s)')

            # plots the mean accuracy values
            x_bar = np.arange(n_filt)
            r_mu = [np.mean(x[i_mu]) if x is not None else None for x in roc_sig_mn]
            r_sem = [np.std(x[i_mu]) / np.sqrt(sum(i_mu)) if x is not None else None for x in roc_sig_mn]

            # creates the bargraph + SEM errorbar
            k = 0
            for i in range(n_filt):
                if r_mu[i] is not None:
                    ax[1].bar(k, r_mu[i], width=w_bar, color=c[i], zorder=1)
                    ax[1].errorbar(k, r_mu[i], yerr=r_sem[i], capsize=40 / n_filt, color='k')
                    k += 1

            # sets the axis properties
            ax[1].grid(plot_grid)
            ax[1].set_xlim([-0.5, k-0.5])
            ax[1].set_xticks(np.arange(k))
            ax[1].set_ylim(ax[0].get_ylim())
            ax[1].set_xticklabels([])

        #########################################
        ####    AUC CDF/STATISTICS GRAPHS    ####
        #########################################

        elif plot_type == 'auROC CDF/Statistics':

            # determines the index of the bin to be plotted (based on whether speed or velocity is analysed)
            if i_bin == 'All Bins':
                # case is plotting all bins
                i_bin_plt = dcopy(i_bin)
            else:
                # case is plotting an individual velocity bin
                i_bin_sp = np.array(i_bin.split(' '))[np.array([0, 2])].astype(int)
                if use_vel:
                    # case is velocity is being analysed
                    i_bin_plt = next(
                        i for i in range(np.shape(r_data.vel_xi)[0]) if np.all(r_data.vel_xi[i, :] == i_bin_sp))
                else:
                    # case is speed is being analysed
                    i_bin_plt = next(
                        i for i in range(np.shape(r_data.spd_xi)[0]) if np.all(r_data.spd_xi[i, :] == i_bin_sp))

            # creates the auc stats figure
            self.create_auc_stats_figure(r_obj, rot_filt, plot_cond, plot_grid, i_bin=i_bin_plt, use_vel=use_vel)

    def plot_cond_grouping_scatter(self, rot_filt, plot_exp_name, plot_all_expt, plot_cond, m_size, show_grp_markers,
                                   show_sig_markers, mark_type, plot_trend, plot_type, plot_grid, plot_scope):
        '''

        :param rot_filt:
        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_cond:
        :param plot_grid:
        :param plot_scope:
        :return:
        '''

        # initialisations
        is_scatter = plot_type == 'auROC Scatterplot'
        _show_grp_markers, _show_sig_markers = dcopy(show_grp_markers), dcopy(show_sig_markers)

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
        x_trend, is_cong = np.arange(0, 10, 0.02), False
        n_filt, r_data = r_obj.n_filt, self.data.rotation
        st_type_name = ['Wilcoxon Paired Test', 'Delong', 'Bootstrapping']
        A = np.empty(n_filt, dtype=object)

        if (not _show_grp_markers) and (not _show_sig_markers):
            e_str = 'Warning! Neither the group or significance markers have been selected!\n' \
                    'Re-run this with either marker type selected'
            cf.show_error(e_str, 'No Markers Selected!')

        # initialisations
        t_type = cf.flat_list([x['t_type'] for x in r_obj.rot_filt_tot])
        ind_black = [t_type.index('Black')]
        ind_match = [cf.det_matching_filters(r_obj, i) for i in ind_black]
        m = ['o', 'x', '^', 's', 'D', 'H', '*']

        # significance colours
        sig_col = [cf.convert_rgb_col([147, 149, 152])[0],         # non-significant markers
                   cf.convert_rgb_col([33, 72, 98])[0],            # black-only significant markers
                   cf.convert_rgb_col([222, 126, 93])[0],          # uniform-only significant markers
                   cf.convert_rgb_col([147, 126, 148])[0]]         # both condition significant spikes

        if is_scatter:
            ####################################
            ####    SCATTERPLOT CREATION    ####
            ####################################

            # retrieves the condition filtered rotation data
            i_cell_b, r_obj_tt = cfcn.get_cond_filt_data(self.data, r_obj)

            if mark_type == 'Congruency':
                _show_sig_markers, _show_grp_markers = False, True
                grp_type, g_type, is_cong = ['None', 'Congruent', 'Incongruent'], r_data.pd_type + 1, True

            elif (not _show_grp_markers) or (mark_type == 'Motion Sensitivity/Direction Selectivity'):
                st_type = st_type_name.index(r_data.phase_grp_stats_type)
                grp_type, g_type = ['MS/DS', 'MS/Not DS', 'Not MS'], r_data.phase_gtype[:, st_type]
            elif mark_type == 'Rotation/Visual Response':
                grp_type, g_type = ['None', 'Rotation', 'Visual', 'Both'], r_data.ds_gtype

            if mark_type == 'Congruency':
                e_col = [cf.convert_rgb_col(_light_gray), cf.convert_rgb_col(_black), sig_col[2]]
                f_col = ['None', cf.convert_rgb_col(_black), sig_col[2]]
            elif mark_type == 'Rotation/Visual Response':
                e_col = f_col = cf.get_plot_col(len(grp_type), len(sig_col))
            elif mark_type == 'Motion Sensitivity/Direction Selectivity':
                e_col = f_col = cf.get_plot_col(len(grp_type), len(sig_col))

            # legend properties initialisations
            if _show_sig_markers:
                h_plt, lg_str = [], ['Black Sig,', '{0} Sig.'.format(plot_cond), 'Both Sig.']
                if not _show_grp_markers:
                    lg_str = ['Not Sig.'] + lg_str
            else:
                h_plt, lg_str = [], []

            # initialises the plot axes
            self.init_plot_axes(n_plot=1)

            # initialises the plot axes region
            self.plot_fig.ax[0].plot([0, 1], [0, 1], 'k--', linewidth=2)
            self.plot_fig.ax[0].grid(plot_grid)

            #
            for i, im in enumerate(ind_match):
                # sets the black/comparison trial condition cell group type values (for the current match)
                if len(i_cell_b[im[1]]):
                    # retrieves the auc, signal and statistical significance values
                    x_auc, y_auc, g_type_m, xy_sig, i_cell_b = self.get_plot_vals(r_data, r_obj_tt, g_type,
                                                                                  i_cell_b, im, plot_cond, is_cong)

                    # sets the final significance plot colours
                    if _show_sig_markers:
                        if _show_grp_markers:
                            mlt, jj = 3, xy_sig > 0

                            if is_cong:
                                sig_col_plt = face_col_plt = np.array([sig_col[x] for x in xy_sig])
                            else:
                                sig_col_plt = np.array([sig_col[x] if x > 0 else None for x in xy_sig])
                                face_col_plt = np.array([sig_col[x] if x > 0 else 'None' for x in xy_sig], dtype=object)
                        else:
                            jj, mlt = np.ones(len(xy_sig), dtype=bool), 1

                            sig_col_plt = np.array([sig_col[x] for x in xy_sig])
                            face_col_plt = np.array([s_col if x > 0 else 'None' for s_col, x in
                                                     zip(sig_col_plt, xy_sig)], dtype=object)

                        # creates the significance markers
                        self.plot_fig.ax[0].scatter(x_auc[jj], y_auc[jj], marker=m[i],
                                                    s=mlt*m_size, facecolor=face_col_plt[jj], edgecolor=sig_col_plt[jj])

                        # creates the significance legend plot markers
                        if i == 0:
                            if _show_grp_markers:
                                lg_ind = np.unique(xy_sig[xy_sig > 0])
                            else:
                                lg_ind = np.unique(xy_sig)

                            for j in lg_ind:
                                h_plt.append(self.plot_fig.ax[0].scatter(-1, -1, marker=m[i], s=mlt*m_size,
                                                                         facecolor='None' if j == 0 else sig_col[j],
                                                                         edgecolor=sig_col[j]))

                    # creates the markers for each of the phases
                    for igt, gt in enumerate(grp_type):
                        ii = g_type_m == igt
                        if np.any(ii):
                            if _show_grp_markers:
                                self.plot_fig.ax[0].scatter(x_auc[ii], y_auc[ii], marker=m[i], s=m_size, alpha=1,
                                                            facecolor=f_col[igt], edgecolor=e_col[igt])

                                # h_plt[i, igt] = self.plot_fig.ax[0].plot(-1, -1, c=c[igt], marker=m[igt])
                                h_plt.append(self.plot_fig.ax[0].scatter(-1, -1, marker=m[i],
                                             facecolor=f_col[igt], edgecolor=e_col[igt]))
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

        else:
            ############################################
            ####    AUROC BAR/HISTOGRAM CREATION    ####
            ############################################

            self.create_auc_stats_figure(r_obj, rot_filt, plot_cond, plot_grid, is_cong=is_cong)

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

    def create_multi_auc_plot(self, ax, roc_auc, plot_grid, connect_lines, violin_bw, m_size, lg_str, auc_plot_type):
        '''

        :param ax:
        :param roc_auc:
        :param plot_grid:
        :param lg_str:
        :return:
        '''

        # sets the x-indices and title string
        n_filt, x_ofs = len(roc_auc), 0
        xi = np.array(range(n_filt)) + 1

        # creates the bubble plot and the decision line
        if connect_lines:
            cf.create_connected_line_plot(ax, roc_auc)
        elif auc_plot_type == 'Bubbleplot':
            cf.create_bubble_boxplot(ax, roc_auc)
        else:
            #
            x_plt = cf.flat_list([[i + 1] * len(x) for i, x in enumerate(roc_auc)])
            y_plt, x_ofs = np.hstack(roc_auc), 1

            # sets the violin/swarmplot dictionaries
            vl_dict = cf.setup_sns_plot_dict(ax=ax, x=x_plt, y=y_plt, inner=None, bw=violin_bw, cut=1)
            sw_dict = cf.setup_sns_plot_dict(ax=ax, x=x_plt, y=y_plt, color='white', edgecolor='gray', size=m_size)

            # creates the violin/swarmplot
            sns.violinplot(**vl_dict)
            sns.swarmplot(**sw_dict)

        # resets the axis limits
        ax.plot([-1, xi[-1]+1], [0.5, 0.5], 'k--')
        cf.set_axis_limits(ax, np.array([xi[0] - 0.5, xi[-1] + 0.5]) - x_ofs, [0, 1])

        # sets the other axis properties
        ax.set_xticks(xi - x_ofs)
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

                # calculates the kruskal-wallis test
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
                        p_stats[i_comp, i_bin] = '{:5.3f}{}'.format(p_value, cf.sig_str_fcn(p_value, 0.05))

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
            xi_bin = dcopy(np.mean(r_data.vel_xi, axis=1))
            if r_data.pn_comp:
                xi_bin = xi_bin[int(len(xi_bin) / 2):]
        else:
            _roc_auc = dcopy(r_data.spd_roc_auc)
            xi_bin = dcopy(np.mean(r_data.spd_xi, axis=1))

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
                col_hdr = ['{0}/-{0}'.format(int(x)) for x in xi_rng[int(len(xi_rng) / 2):, 1]]
            else:
                col_hdr = ['{0}/{1}'.format(int(xx[0]), int(xx[1])) for xx in xi_rng]

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

    def create_auc_stats_figure(self, r_obj, rot_filt, plot_cond, plot_grid, is_cong=False,
                                i_bin=None, plot_cond_base='Black', use_vel=False):

        def setup_sig_str(y, is_auc_sig):
            '''

            :param p_auc_sig:
            :return:
            '''

            # memory allocation and parameters
            n_tt = len(y)
            p_value, n_ex = 0.05, None
            p_str = np.empty((n_tt, n_tt), dtype=object)

            # sets up the significance strings
            for i_tt in range(n_tt):
                for j_tt in range(n_tt):
                    if i_tt == j_tt:
                        # case is the diagonal case
                        p_str[i_tt, i_tt] = 'N/A'
                    else:
                        if is_auc_sig:
                            if n_ex is None:
                                n_ex = [len(x) for x in y]

                            # runs the pair-wise wilcoxon test
                            i_grp, y_grp = [0] * n_ex[i_tt] + [1] * n_ex[j_tt], y[i_tt] + y[j_tt]
                            results = r_stats.pairwise_wilcox_test(FloatVector(y_grp), FloatVector(i_grp),
                                                                   p_adjust_method='bonf', paired=False)
                            p_val_nw = results[results.names.index('p.value')][0]
                        else:
                            if (y[i_tt] is None) or (y[j_tt] is None):
                                p_str[i_tt, j_tt] = p_str[j_tt, i_tt] = 'NaN'
                                continue
                            else:
                                _, p_val_nw = ks_2samp(y[i_tt], y[j_tt])

                        # retrieves the results and stores them in the results array
                        p_str[i_tt, j_tt] = p_str[j_tt, i_tt] = \
                            '{:5.3f}{}'.format(p_val_nw, cf.sig_str_fcn(p_val_nw, p_value))

            # returns the array
            return p_str

        # initialisations
        e_str, r_data = None, self.data.rotation
        is_dir, st_type_name = i_bin is None, ['Wilcoxon Paired Test', 'Delong', 'Bootstrapping']
        tt_auc, tt_filt = rot_filt['t_type'], cf.flat_list([x['t_type'] for x in r_obj.rot_filt_tot])

        # retrieves the black/matching filter index values
        if plot_cond_base not in tt_filt:
            # case is the base trial condition has not been calculated
            e_str = 'To run this function, you must ensure "{0}" is selected for the ' \
                    'Trial Type in the Filter Parameters GUI.'.format(plot_cond_base)
        elif plot_cond not in tt_filt:
            # case is the dependent trial condition has not been calculated
            e_str = 'To run this function, you must ensure "{0}" is selected for the ' \
                    'Trial Type in the Filter Parameters GUI.'.format(plot_cond)
        else:
            # determines the indices of the cells that are common to all the trial conditions
            ind_cond = cfcn.get_all_match_cond_cells(self.data, tt_filt)
            if len(ind_cond[tt_filt[0]]) == 0:
                # if there are no matching cells, then output an error to screen
                e_str = 'There are no cells that match all the selected trial conditions. \n' \
                        'Try deselecting some of the conditions from the Trial Type in the Filter Parameters GUI.'

        # there is an error, then output the message to screen and exit with an error
        if e_str is not None:
            cf.show_error(e_str, 'Invalid Filter')
            self.calc_ok = False
            return

        # determines the indices of the
        ind_match = [tt_auc.index(plot_cond_base), tt_auc.index(plot_cond)]

        # retrieves the cell indices and unique filter objects
        i_cell_b, r_obj_tt = cfcn.get_common_filtered_cell_indices(self.data, r_obj, tt_filt, use_vel, ind_cond=ind_cond)
        if is_dir:
            st_type = st_type_name.index(r_data.phase_grp_stats_type)
            g_type = r_data.phase_gtype[:, st_type]
        else:
            if use_vel and (i_bin in r_data.i_bin_vel):
                e_str = 'Not possible to plot the auROC stats as the dependent/plotting velocity bins are the same.'
            elif (not use_vel) and (i_bin == r_data.i_bin_spd):
                e_str = 'Not possible to plot the auROC stats as the dependent/plotting speed bins are the same.'

            if e_str is None:
                st_type = st_type_name.index(r_data.kine_auc_stats_type) - 1
                g_type = None
            else:
                cf.show_error(e_str, 'Invalid Bin Index')
                self.calc_ok = False
                return

        #############################
        ####    SUBPLOT SETUP    ####
        #############################

        # sets up the axes dimensions
        nR, nC, nCM = 2, 11, 6
        top, bottom, pH, wspace, hspace = 0.95, 0.06, 0.01, 0.2, 0.25
        # i_expt = [r_obj.i_expt[0] == x for x in np.unique(r_obj.i_expt[0])]

        # creates the gridspec object
        gs = gridspec.GridSpec(nR, nC, width_ratios=[1 / nC] * nC, height_ratios=[1 / nR] * nR,
                               figure=self.plot_fig.fig, wspace=wspace, hspace=hspace, left=0.12, right=0.99,
                               bottom=bottom, top=top)

        # creates the subplots
        self.plot_fig.ax = np.empty(4, dtype=object)
        self.plot_fig.ax[0] = self.plot_fig.figure.add_subplot(gs[0, :nCM])
        self.plot_fig.ax[1] = self.plot_fig.figure.add_subplot(gs[1, :nCM])
        self.plot_fig.ax[2] = self.plot_fig.figure.add_subplot(gs[0, nCM:])
        self.plot_fig.ax[3] = self.plot_fig.figure.add_subplot(gs[1, nCM:])
        ax = self.plot_fig.ax

        # turns off the subplot intended for the stats
        for _ax in ax[:2]:
            _ax.grid(plot_grid)
        for _ax in ax[2:]:
            _ax.axis('off')

        # turns off the axis if analysing all bins
        if i_bin == 'All Bins':
            ax[1].axis('off')

        ###################################
        ####    DATA PRE-PROCESSING    ####
        ###################################

        # memory allocation
        im, n_grp = ind_match, 4
        B = np.empty(len(ind_match), dtype=object)
        is_sig, auc = dcopy(B), dcopy(B)

        #
        r_filt_auc = cf.init_rotation_filter_data(False)
        r_filt_auc['t_type'] = tt_auc
        r_obj_auc_full = RotationFilteredData(self.data, r_filt_auc, None, None, True, 'Whole Experiment', False)

        # sets the base/comparison trial condition cell group type values (for the current match)
        i_cell_auc, r_obj_auc = cfcn.get_common_filtered_cell_indices(self.data, r_obj_auc_full, tt_auc, use_vel)
        if len(i_cell_b[im[1]]):
            # retrieves the auc, signal and statistical significance values
            x_auc, y_auc, _, xy_sig, i_cell_auc = self.get_plot_vals(r_data, r_obj_auc, g_type, i_cell_auc, im, plot_cond,
                                                                     is_cong=is_cong, i_bin=i_bin, use_vel=use_vel)
            if i_bin != 'All Bins':
                # calculates significant cells for each type
                is_sig, auc = [xy_sig == (j + 1) for j in range(3)], np.empty(n_grp, dtype=object)

                # stores the
                for j in range(n_grp):
                    if j == 0:
                        # case are base condition cells that were significant only for base conditions
                        auc[j] = x_auc[is_sig[0]].flatten()
                    elif j == 1:
                        # case are dependent condition cells that were significant only for dependent conditions
                        auc[j] = y_auc[is_sig[1]].flatten()
                    elif j == 2:
                        # case are base condition cells that were significant for BOTH base/dependent conditions
                        auc[j] = x_auc[is_sig[2]].flatten()
                    else:
                        # case are dependent condition cells that were significant for BOTH base/dependent conditions
                        auc[j] = y_auc[is_sig[2]].flatten()

        # proportion of condition auROC values that are significant
        if is_dir:
            i_expt = [[r_data.cond_i_expt[c] == x for x in np.unique(r_data.cond_i_expt[c])] for c in tt_filt]
            p_auc_sig = [[100. * np.mean(r_data.cond_gtype[c][j_ex, st_type] == 0) for j_ex in i_ex]
                         for i_ex, c in zip(i_expt, tt_filt)]
            n_expt = [len(x) for x in i_expt]

        else:
            # determines the indices of the cells that correspond to each experiment (for each filter type)
            i_expt = [[rr.i_expt[0][i_c] == x for x in range(r_obj.n_expt)] for rr, i_c in zip(r_obj_tt, i_cell_b)]
            n_cell = np.vstack([[sum(y) for y in x] for x in i_expt])
            n_expt = [sum(x > 0) for x in n_cell]

            # determines the indices of the trial conditions with respect to the stored data
            i_tt = [tt_auc.index(x) for x in tt_filt]

            if i_bin == 'All Bins':
                # case is all bins are being analysed

                # retrieves the valid bin indices
                i_col = self.setup_valid_bin_indices(r_data, use_vel)

                # sets the significance values based on the kinematic type
                if use_vel:
                    # case is analysing velocity
                    v_roc = [r_data.vel_roc_sig[it, st_type][ic, :][:, i_col] for it, ic in zip(i_tt, i_cell_b)]
                    p_auc_sig = [[100. * np.mean(np.mean(v_r[j_ex, :], axis=1)) for j_ex, n in zip(i_ex, n_c) if n > 0]
                                                     for i_ex, v_r, n_c in zip(i_expt, v_roc, n_cell)]

                else:
                    # case is analysing speed
                    s_roc = [r_data.spd_roc_sig[it, st_type][ic, :][:, i_col] for it, ic in zip(i_tt, i_cell_b)]
                    p_auc_sig = [[100. * np.mean(np.mean(s_r[j_ex, :], axis=1)) for j_ex, n in zip(i_ex, n_c) if n > 0]
                                                     for i_ex, s_r, n_c in zip(i_expt, s_roc, n_cell)]

            else:
                # case is a single bin is being analysed

                # sets the significance values based on the kinematic type
                if use_vel:
                    # case is analysing velocity
                    v_roc = [r_data.vel_roc_sig[it, st_type][ic, :][:, i_bin] for it, ic in zip(i_tt, i_cell_b)]
                    p_auc_sig = [[100. * np.mean(v_r[j_ex]) for j_ex, n in zip(i_ex, n_c) if n > 0]
                                                            for i_ex, v_r, n_c in zip(i_expt, v_roc, n_cell)]

                else:
                    # case is analysing speed
                    s_roc = [r_data.spd_roc_sig[it, st_type][ic, :][:, i_bin] for it, ic in zip(i_tt, i_cell_b)]
                    p_auc_sig = [[100. * np.mean(s_r[j_ex]) for j_ex, n in zip(i_ex, n_c) if n > 0]
                                                            for i_ex, s_r, n_c in zip(i_expt, s_roc, n_cell)]

        # calculates the mean/SEM proportional significances
        p_auc_sig_mn = [np.mean(x) for x in p_auc_sig]
        p_auc_nsig_mn = [100 - x for x in p_auc_sig_mn]
        p_auc_sig_sem = [np.std(x) / (n ** 0.5) for x, n in zip(p_auc_sig, n_expt)]

        #################################################
        ####    SIGNIFICANCE HORIZONTAL BAR GRAPH    ####
        #################################################

        # parameters and other initialisations
        col_b = cf.get_plot_col(len(tt_filt))
        xi_y = np.arange(len(tt_filt)) + 0.5

        # creates the bar graph
        for i_tt, tt in enumerate(tt_filt):
            ax[0].barh(xi_y[i_tt], p_auc_sig_mn[i_tt], height=0.9, color=col_b[i_tt],
                       edgecolor=col_b[i_tt], label=tt, xerr=p_auc_sig_sem[i_tt])
            ax[0].barh(xi_y[i_tt], p_auc_nsig_mn[i_tt], left=p_auc_sig_mn[i_tt],
                       height=0.9, color='w', edgecolor=col_b[i_tt])

        # sets the axis properties
        ax[0].set_xlabel('Percentage Significant')
        ax[0].set_xlim([-0.1, 100.1])
        ax[0].set_ylim([0, len(tt_filt)])
        ax[0].set_yticks(xi_y)
        ax[0].set_yticklabels(['#{0} - {1}'.format(i + 1,lg_str) for i, lg_str in enumerate(r_obj.lg_str)])
        ax[0].invert_yaxis()

        # sets the overall title
        self.plot_fig.fig.suptitle('Significant Cells by Trial Type', fontsize=14, fontweight='bold')

        ############################################
        ####    AUC SIGNIFICANCE STATS TABLE    ####
        ############################################

        #
        n_tt = len(tt_filt)
        col_h = cf.get_plot_col(max(n_grp, n_tt))

        # sets the table headers/values
        # row_hdr = col_hdr = [cf.cond_abb(tt) for tt in tt_filt]
        row_hdr = col_hdr = ['#{0}'.format(i + 1) for i in range(n_tt)]
        t_data = setup_sig_str(p_auc_sig, True)

        # sets up the n-value table
        ax_pos_tbb = dcopy(ax[2].get_tightbbox(self.plot_fig.get_renderer()).bounds)
        cf.add_plot_table(self.plot_fig, ax[2], table_font, t_data, row_hdr, col_hdr, col_h[:n_tt],
                          col_h[:n_tt], 'top', n_row=2, n_col=4, ax_pos_tbb=ax_pos_tbb, p_wid=1.5)

        #
        if i_bin == 'All Bins':
            ax[1].axis('off')
            return

        ######################################
        ####    SIGNIFICANCE HISTOGRAM    ####
        ######################################

        # parameters and other initialisations
        h_plt, lg_str, b_sz = [], [], 0.01
        xi = np.arange(0.5, 1.001, b_sz)
        xi_h = 0.5 * (xi[1:] + xi[:-1])

        #
        tt_abb = [cf.cond_abb(tt) for tt in ['Black', plot_cond]]
        auc_hist_type = cf.flat_list(
            [['{0}{1}{2}'.format(p_str, tt, ')' if len(p_str) else '') for tt in tt_abb] for p_str in
             ['', 'Both (']])

        auc_cdf = np.empty(n_grp, dtype=object)
        for i in range(n_grp):
            # creates the accuracy histograms
            if len(auc[i]):
                auc_hist = np.histogram(auc[i], bins=xi, normed=True)[0]
                auc_cdf[i] = 100. - np.cumsum(auc_hist)

                h_plt.append(ax[1].plot(xi_h, auc_cdf[i], c=col_h[i]))
                lg_str.append(auc_hist_type[i])
            else:
                auc_cdf[i] = None

        # sets the axis properties
        ax[1].set_xlabel('auROC')
        ax[1].set_ylabel('Proportion')
        ax[1].set_xlim([0.5, 1.0])
        ax[1].set_ylim([0.0, 100.0])
        ax[1].legend([x[0] for x in h_plt], lg_str, loc='lower left')

        # sets the overall title
        ax[1].set_title('Significant Cell auROC Histogram', fontsize=14, fontweight='bold')

        ###################################################
        ####    AUC SURVIVABILITY CURVE STATS TABLE    ####
        ###################################################

        # sets the table headers/values
        col_t = cf.get_plot_col(4)
        row_hdr = col_hdr = dcopy(auc_hist_type)
        t_data = setup_sig_str(auc_cdf, False)

        # sets up the n-value table
        ax_pos_tbb = dcopy(ax[3].get_tightbbox(self.plot_fig.get_renderer()).bounds)
        cf.add_plot_table(self.plot_fig, ax[3], table_font, t_data, row_hdr, col_hdr, col_t, col_t,
                          'top', n_row=2, n_col=4, ax_pos_tbb=ax_pos_tbb, p_wid=1.5)

    ###########################################
    ####    COMBINED ANALYSIS FUNCTIONS    ####
    ###########################################

    def plot_combined_stimuli_stats(self, rot_filt, plot_exp_name, plot_all_expt, plot_scope, plot_type,
                                    grp_plot_type, plot_grid, grp_by_filt, show_stats):
        '''

        :return:
        '''

        def setup_plot_axes(plot_fig, n_plot):
            '''

            :param plot_fig:
            :param n_plot:
            :return:
            '''

            # sets up the axes dimensions
            n_r, n_c = 2, n_plot
            top, bottom, wspace, hspace = 0.90, 0.04, 0.25, 0.01
            height_ratios = [0.8, 0.1]

            # creates the gridspec object
            gs = gridspec.GridSpec(n_r, n_c, width_ratios=[1 / n_c] * n_c, height_ratios=height_ratios,
                                   figure=plot_fig.fig, wspace=wspace, hspace=hspace, left=0.075, right=0.98,
                                   bottom=bottom, top=top)

            # creates the subplots
            plot_fig.ax = np.empty(n_r * n_c, dtype=object)
            for i_r in range(n_r):
                for i_c in range(n_c):
                    # adds the subplot
                    i_plt = i_r * n_c + i_c
                    plot_fig.ax[i_plt] = plot_fig.figure.add_subplot(gs[i_r, i_c])

                    # turns off axis for the table row
                    if i_r == 1:
                        plot_fig.ax[i_plt].axis('off')

        # checks to see if the current configuration is feasible (exit function if not)
        if not self.check_group_plot_feas(grp_plot_type, grp_by_filt, show_stats):
            return

        # initialisations and memory allocation
        p_value = 0.05
        r_data = self.data.rotation
        lg_str_f = r_data.r_obj_rot_ds.lg_str
        n_grp0, n_filt = [3, 3, 1], r_data.r_obj_rot_ds.n_filt

        # determines the indices of the metric that are being plotted
        plot_type0 = ['Motion Sensitivity', 'Direction Selectivity', 'Congruency']
        i_grp = np.arange(len(plot_type0)) if show_stats else [i for i, pt in enumerate(plot_type) if pt in plot_type0]
        n_plt, n_grp = len(i_grp), list(np.array(n_grp0)[i_grp])

        # sets the main titles (based on the plot types)
        stats_type = ['Rotation MS', 'Rotation/Visual DS', 'Congruency']
        main_title = list(np.array(stats_type)[i_grp])

        # retrieves the plot values
        type_pr_ex0 = [dcopy(r_data.ms_gtype_ex), dcopy(r_data.ds_gtype_ex), dcopy(r_data.pd_type_ex)]

        if grp_by_filt:
            # data is grouped by filter type
            type_pr_ex = [[x[:, 1:] for x in y] for y in np.array(type_pr_ex0)[i_grp]]
        else:
            # data is grouped by the classification groups
            type_pr_ex = [self.reorder_array_dim([x[:, 1:] for x in y]) for y in np.array(type_pr_ex0)[i_grp]]

        # calculates the group posthoc statistics
        stats_ph = self.calc_group_posthoc_stats(type_pr_ex, n_filt, n_grp)

        if show_stats:
            #################################
            ####    STATISTICS TABLES    ####
            #################################

            # sets up the plot axes
            col_table = cf.get_plot_col(max([max(n_grp), n_filt]), 1)
            self.setup_posthoc_stats_table_axes(self.plot_fig, main_title, n_filt, n_grp)

            # sets the table header strings
            hdr_class = ['Rot.', 'Vis.', 'Both']
            hdr_filt = ['(#{0})'.format(i + 1) for i in range(n_filt)]

            # sets the base title string
            tt_class = ['Rotation', 'Visual', 'Both']
            if (n_filt == 1) and (lg_str_f[0] == 'Black'):
                tt_filt = ['(#1) - All Cells']
            else:
                tt_filt = ['(#{0}) - {1}'.format(i + 1, '/'.join(lg.split('\n'))) for i, lg in enumerate(lg_str_f)]

            # sets the final header/title strings
            if n_filt == 1:
                hdr_str, t_str = [hdr_class, hdr_class], [tt_filt, tt_filt]
            elif grp_by_filt:
                hdr_str = [hdr_filt, hdr_class, hdr_filt, hdr_class, hdr_filt]
                t_str = [tt_class, tt_filt, tt_class, tt_filt, ['Congruency']]
            else:
                hdr_str = [hdr_class, hdr_filt, hdr_class, hdr_filt, hdr_filt]
                t_str = [tt_filt, tt_class, tt_filt, tt_class, ['Congruency']]

            # creates the posthoc statistics table
            self.create_posthoc_stats_table(stats_ph, hdr_str, t_str, col_table, p_value)

        else:
            #############################
            ####    METRIC GRAPHS    ####
            #############################

            # initialisations
            not_stacked = int((grp_plot_type != 'Stacked Bar'))
            c = cf.get_plot_col(max([n_filt, max(n_grp) + (1 - not_stacked)]))

            # sets the classification group strings
            class_str0 = [['None', 'Rotation', 'Visual', 'Both']] * 2 + [['Incongruent', 'Congruent']]
            class_str = [x[not_stacked:] for x in np.array(class_str0)[i_grp]]

            # creates filter group strings
            if (n_filt == 1) and (lg_str_f[0] == 'Black'):
                # no filter applied, so all cells are used
                filt_str = ['(#1) - All Cells']
            else:
                # otherwise, set the filter grouping strings
                filt_str = ['#{0} - {1}'.format(i + 1, x) for i, x in enumerate(dcopy(lg_str_f))]

            # sets up the plot axes
            setup_plot_axes(self.plot_fig, len(n_grp))
            ax = self.plot_fig.ax

            # sets the plot values (based on the plot types)
            if not_stacked:
                # case is the non-stacked bar graphs
                plt_type = dcopy(type_pr_ex)
            else:
                # case is the stacked bar graphs
                plt_type0 = dcopy([r_data.ms_gtype_pr, r_data.ds_gtype_pr, r_data.pd_type_pr])
                plt_type = np.array(plt_type0)[i_grp]

            # creates the graphs for the motion sensitive/direction selectivity plots
            for i in range(n_plt):
                # creates the graph
                h_plt = cf.create_general_group_plot(ax[i], plt_type[i], grp_plot_type, c)
                cf.reset_axes_dim(ax[i], 'bottom', 0.075, True)

                # sets the legend/x-ticklabels based on the grouping type
                lg_str = class_str[i] if grp_by_filt else filt_str
                x_ticklbl = filt_str if grp_by_filt else class_str[i]

                # creates the legend (if more than one bar)
                if len(h_plt):
                    ax[i].legend([x[0] for x in h_plt], lg_str, ncol=len(lg_str), loc='upper center',
                                                                columnspacing=0.125, bbox_to_anchor=(0.5, 1.11))

                # only set the y-axis label for the first subplot
                if i == 0:
                    ax[i].set_ylabel('Population %')

                # only set the x-tick labels if there is more than one label
                ax[i].set_xticklabels(x_ticklbl)

                # sets the y-axis limits based on type
                if 'Bar' in grp_plot_type:
                    ax[i].set_ylim([0, 100])
                else:
                    ax[i].set_ylim([-2.5, 102.5])

                # updates the axis properties
                ax[i].set_title(main_title[i], fontsize=16, fontweight='bold')
                ax[i].grid(plot_grid)

            ###################################
            ####    METRIC COUNT TABLES    ####
            ###################################

            # # enforces tight layout format
            # self.plot_fig.fig.tight_layout()

            # sets the initial motion sensitivity/congruency table values
            n_MS0 = np.vstack([r_data.ms_gtype_N] * 4) * r_data.ms_gtype_pr / 100

            # sets the final table values
            n_MS = np.vstack((n_MS0[0, :], np.sum(n_MS0[1:, ], axis=0)))
            n_DS = np.vstack([r_data.ds_gtype_N] * 4) * r_data.ds_gtype_pr / 100
            n_PD = np.vstack(r_data.pd_type_N)
            t_data = [cf.add_rowcol_sum(n_MS).T, cf.add_rowcol_sum(n_DS).T, cf.add_rowcol_sum(n_PD)]

            # creates the title text object
            t_str = ['{0} N-Values'.format(x) for x in stats_type]
            col_hdr = [['Insensitive', 'Sensitive', 'Total'],
                       ['None', 'Rotation', 'Visual', 'Both', 'Total'],
                       ['Incongruent', 'Congruent', 'Total']]
            t_font = cf.get_table_font_size(3)

            if (n_filt == 1) and (lg_str_f[0] == 'Black'):
                row_hdr = ['All Cells', 'Total']
            else:
                row_hdr = ['#{0}'.format(x + 1) for x in range(n_filt)] + ['Total']

            # creates the graphs for the motion sensitive/direction selectivity plots
            for i in range(n_plt):
                # creates the new table
                j, nT = i + n_plt, len(col_hdr[i])
                cT = cf.get_plot_col(max(n_filt, nT), max(n_grp))
                cf.add_plot_table(self.plot_fig, ax[j], t_font, t_data[i].astype(int), row_hdr, col_hdr[i],
                                  cT[:n_filt] + [(0.75, 0.75, 0.75)], cT[:nT] + [(0.75, 0.75, 0.75)],
                                  'fixed', n_col=n_plt)

    def plot_combined_direction_roc_curves(self, rot_filt, plot_exp_name, plot_all_expt, use_avg, connect_lines, m_size,
                                           violin_bw, plot_grp_type, cell_grp_type, auc_plot_type, plot_grid, plot_scope):
        '''

        :param rot_filt:
        :param plot_exp_name:
        :param plot_all_expt:
        :param use_avg:
        :param connect_lines:
        :param plot_grp_type:
        :param cell_grp_type:
        :param plot_grid:
        :param plot_scope:
        :return:
        '''

        self.create_dir_roc_curve_plot(rot_filt, plot_exp_name, plot_all_expt, use_avg, connect_lines, violin_bw, m_size,
                                       plot_grp_type, cell_grp_type, auc_plot_type, plot_grid, plot_scope, True)

    ##############################################
    ####    DEPTH-BASED ANALYSIS FUNCTIONS    ####
    ##############################################

    def plot_depth_spiking(self, rot_filt, plot_ratio, plot_layer, depth_type, plot_grid, plot_all_expt,  plot_scope):
        '''

        :param rot_filt:
        :param depth_type:
        :param plot_grid:
        :param plot_scope:
        :return:
        '''

        # initialisations
        r_data = self.data.depth
        plt, stats, ind, t_type = r_data.plt, r_data.stats, dcopy(r_data.ind), r_data.r_filt['t_type']
        n_tt, ch_depth0 = len(t_type), dcopy(r_data.ch_depth)
        ch_region0, ch_layer0 = dcopy(r_data.ch_region), dcopy(r_data.ch_layer)

        # other initialisations and parameters
        auc0 = dcopy(r_data.cond_roc_auc)
        col, n_phase, p_value = ['r', 'b'], 3, 0.05

        # creates the rotation filter object
        _data = cfcn.get_rsp_reduced_clusters(self.data)
        r_obj = RotationFilteredData(_data, rot_filt, None, None, True, 'Whole Experiment', False)

        # memory allocation
        n_filt, a = r_obj.n_filt, np.empty(r_obj.n_filt, dtype=object)
        ch_depth, ch_region, ch_layer, auc, ind_g = dcopy(a), dcopy(a), dcopy(a), dcopy(a), dcopy(a)


        for i_filt in range(n_filt):
            # retrieves the trial type for the current filter
            tt = r_obj.rot_filt_tot[i_filt]['t_type'][0]

            # determines of the indices that match the rotational filter
            ind_tmp = [np.where(r_data.cond_i_expt[tt] == x)[0] for x in range(max(r_data.cond_i_expt[tt]) + 1)]
            ind_tmp = [x for x in ind_tmp if len(x)]

            # sets the indices of the values that will be kept for analysis
            ind_g[i_filt] = np.array(cf.flat_list([_ind[x] for _ind, x in zip(ind_tmp, r_obj.clust_ind[i_filt])]))
            ch_depth[i_filt], ch_region[i_filt] = ch_depth0[tt][ind_g[i_filt]], ch_region0[tt][ind_g[i_filt]]
            ch_layer[i_filt], auc[i_filt]  = ch_layer0[tt][ind_g[i_filt]], auc0[tt][ind_g[i_filt]]

        ###################################
        ####    DATA PRE-PROCESSING    ####
        ###################################

        if depth_type == 'Preferred/Baseline FR Difference':
            # determines the preferred direction (direction with the greatest absolute deviation from baseline)
            dCW, dCCW = np.diff(plt[0], axis=0), np.diff(plt[1], axis=0)
            imx = np.argmax(np.vstack((dCW, dCCW)).T, axis=1)

            # retrieves the significant cells
            p_sig = np.array([stats[_imx][i] for i, _imx in enumerate(imx)])
            is_sig = [p_sig[_ind] < p_value for _ind in ind]

            # sets the plot values (based on type)
            if plot_ratio:
                # case is the firing rate ratio
                dsf_Pref_BL = np.divide([plt[_imx][1][i] for i, _imx in enumerate(imx)], np.array(plt[0])[0, :])
                x_plt = [np.log10(dsf_Pref_BL[_ind]) for _ind in ind]
                x_lbl = 'log10[FR(Pref)/FR(BL)]'
            else:
                # case is the firing rate difference
                dsf_Pref_BL = np.array([plt[_imx][1][i] for i, _imx in enumerate(imx)]) -  np.array(plt[0])[0, :]
                x_plt = [dsf_Pref_BL[_ind] for _ind in ind]
                x_lbl = 'FR(Pref) - FR(BL) (spike/s)'

        elif depth_type == 'CW/CCW auROC Difference':
            # retrieves the significant cells
            is_sig = [r_data.cond_auc_sig[tt][:, -1] for tt in t_type]

            # sets the plot values (based on type)
            if plot_ratio:
                # case is the CCW/CW auROC ratio
                x_lbl = 'log10(auROC(CCW)/auROC(CW))'
                x_plt = [np.log10(auc[tt][:, 2] / auc[tt][:, 1]) for tt in t_type]
            else:
                # case is the CCW/CW auROC difference
                x_lbl = 'auROC(CCW) - auROC(CW) (a.u.)'
                x_plt = [np.diff(auc[tt][:, 1:], axis=1) for tt in t_type]

        elif depth_type == 'CW/CCW FR Difference':
            # retrieves the significant cells
            is_sig = [stats[2][_ind] < p_value for _ind in ind]

            # sets the plot values (based on type)
            if plot_ratio:
                # case is the firing rate ratio
                x_lbl = 'log10(FR(CCW)/FR(CW))'
                x_plt = [np.log10(np.divide(plt[2][1], plt[2][0])[_ind]) for _ind in ind]
            else:
                # case is the firing rate difference
                x_lbl = 'FR(CCW) - FR(CW) (a.u.)'
                x_plt = [np.diff(np.array(plt[2])[:, _ind], axis=0) for _ind in ind]

        if plot_ratio:
            is_ok = [np.logical_not(np.isinf(x)) for x in x_plt]
            xL = np.ceil(max(max([np.max(x[i_ok]) for x, i_ok in zip(x_plt, is_ok)]),
                             np.abs(min([np.min(x[i_ok]) for x, i_ok in zip(x_plt, is_ok)]))))
            x_lim = [-xL, xL]
        else:
            is_ok = [np.ones(len(x)) for x in ind]
            x_lim = [-1, 1]

        #############################
        ####    SUBPLOT SETUP    ####
        #############################

        # sets up the axes dimensions
        if plot_layer:
            # initialises the plot axes
            self.plot_fig.setup_plot_axis(n_row=1, n_col=n_filt)

        else:
            top, bottom, pH, wspace, hspace = 0.97, 0.06, 0.01, 0.01, 0.15
            w_ratio, h_ratio = [0.85, 0.15], [1 / n_filt] * n_filt

            # creates the gridspec object
            gs = gridspec.GridSpec(n_filt, 2, width_ratios=w_ratio, height_ratios=h_ratio, wspace=wspace, hspace=hspace,
                                   left=0.075, right=0.98, bottom=bottom, top=top, figure=self.plot_fig.fig)

            # creates the subplots
            self.plot_fig.ax = np.empty(2 * n_filt, dtype=object)
            for i_filt in range(n_filt):
                self.plot_fig.ax[2 * i_filt] = self.plot_fig.figure.add_subplot(gs[i_filt, 0])
                self.plot_fig.ax[2 * i_filt + 1] = self.plot_fig.figure.add_subplot(gs[i_filt, 1])

        ################################
        ####    SUBPLOT CREATION    ####
        ################################

        # plotting parameters
        col, m, yL, y_ofs = ['b', 'r'], ['s', '^'], None, 0.75


        #
        for i_filt in range(n_filt):
            # initialisations
            tt = r_obj.rot_filt_tot[i_filt]['t_type']
            ax = self.plot_fig.ax[(1 + (not plot_layer)) * i_filt]

            # retrieves the indices of the
            is_rspg = (ch_region[i_filt][is_ok[i_filt]] == 'RSPg')
            y_depth, y_layer = ch_depth[i_filt][is_ok[i_filt]], ch_layer[i_filt][is_ok[i_filt]]
            is_sig_tt = is_sig[i_filt][is_ok[i_filt]]

            # sets the grouping indices (for the significant/non-significant RSPd/RSPg groups)
            i_grp = [
                np.logical_and(is_rspg, is_sig_tt),
                np.logical_and(is_rspg, np.logical_not(is_sig_tt)),
                np.logical_and(np.logical_not(is_rspg), is_sig_tt),
                np.logical_and(np.logical_not(is_rspg), np.logical_not(is_sig_tt)),
            ]

            if plot_layer:
                rl_str = np.array(['{0}\n({1})'.format(x, y) for x, y in zip(ch_region[i_filt][is_ok[i_filt]], y_layer)])
                rl_str_uniq = np.unique(rl_str)

                # sorts the region/layer strings by average group cell depth
                i_sort = np.argsort([np.mean(y_depth[rl_str == x]) for x in rl_str_uniq])
                rl_str_uniq = list(rl_str_uniq[i_sort])
                yL = [0.5, (len(rl_str_uniq) + 0.5)]

                # retrieves the depths for each of the
                y_depth = np.array([(rl_str_uniq.index(x) + 1) for x in rl_str]).astype(float)
                y_depth += (1 - 2 * np.random.rand(len(y_depth))) * (y_ofs / 2)

                # plots the group separation markers
                for i in range(1, len(rl_str_uniq)):
                    ax.plot(x_lim, (i + 0.5) * np.ones(2), 'k--')

            ##########################################
            ####    DEPTH/ACCURACY SCATTERPLOT    ####
            ##########################################

            # creates the scatter
            for i in range(2):
                for j in range(2):
                    # sets the plot values/indices
                    k = 2 * i + j
                    x0, is_sig_nw = x_plt[i_filt][is_ok[i_filt]][i_grp[k]], is_sig_tt[i_grp[k]]
                    e_col = [col[is_pos] for is_pos in (x0 < 0)]

                    # creates a scatterplot marker for each member of the group
                    for kk in range(len(x0)):
                        ax.scatter(x0[kk], y_depth[i_grp[k]][kk], marker=m[i],
                                   facecolor=e_col[kk] if j==0 else 'w', edgecolors=e_col[kk], s=120)

            # plots the midline
            if yL is None:
                yL = ax.get_ylim()

            # plots the midline
            ax.plot([0, 0], yL, 'k--')
            ax.set_ylim(yL)

            # creates the legend (first trial type only)
            if i_filt == 0:
                # sets the legend strings
                lg_str = ['{0}{1})'.format(_lg_str, 0) for _lg_str in ['RSPg (>', 'RSPg (<', 'RSPd (>', 'RSPd (<']]

                # creates the legend plot objects
                h_lg = []
                for i in range(2):
                    for j in range(2):
                        h_lg.append(ax.scatter(0, 10000., marker=m[i], facecolor=col[j], edgecolor=col[j]))

                # creates the legend object
                ax.legend(h_lg, lg_str, loc=0)

            # sets the plot axis
            ax.set_title(', '.join(r_obj.lg_str[i_filt].split('\n')))
            ax.invert_yaxis()
            ax.set_xlim(x_lim)
            ax.grid(plot_grid)

            if plot_layer:
                if i_filt == 0:
                    ax.set_yticks(1. + np.arange(len(rl_str_uniq)))
                    ax.set_yticklabels(rl_str_uniq)
                else:
                    ax.set_yticklabels([])

                ax.set_xlabel(x_lbl)
            else:
                ax.set_ylabel('Depth beneath Surface ({0}m)'.format(cf._mu))

                # sets the y-axis label
                if (i_filt + 1) == n_filt:
                    ax.set_xlabel(x_lbl)

            # continue if plotting the layers
            if plot_layer:
                continue

            ###################################
            ####    CELL LAYER LINEPLOT    ####
            ###################################

            # retrieves the depth/layer axes
            ax_l = self.plot_fig.ax[2 * i_filt + 1]

            # sets the depths of each layer
            y_layer_uniq = np.unique(y_layer)
            y_depth_layer = np.array([y_depth[y_layer == y_l] for y_l in y_layer_uniq])

            # sorts the layers by depth
            i_sort_dl = np.argsort([np.mean(x) for x in y_depth_layer])
            y_layer_uniq, y_depth_layer = y_layer_uniq[i_sort_dl], y_depth_layer[i_sort_dl]

            # plots the markers
            n_layer, l_col = len(y_layer_uniq), cf.get_plot_col(len(y_layer_uniq))
            for i_yl in range(n_layer):
                # plots the markers
                y_dl, x = y_depth_layer[i_yl], (i_yl + 1)
                ax_l.plot(x * np.ones(len(y_dl)), y_dl, 'o', color=l_col[i_yl], linewidth=2)

                # adds the text markers to the lines
                ax_l.text(x + 0.15, np.min(y_dl), y_layer_uniq[i_yl], rotation=-90,
                          rotation_mode='anchor', fontweight='bold')

            # sets the axis properties
            ax_l.set_xlim([0.5, n_layer + 0.5])
            ax_l.set_ylim(ax.get_ylim())
            ax_l.set_xticks([])
            ax_l.set_yticks([])
            ax_l.grid(plot_grid)

    def plot_depth_spiking_multi(self, rot_filt, plot_ratio, plot_layer, depth_type, plot_grid, plot_all_expt, plot_scope):
        '''

        :param rot_filt:
        :param depth_type:
        :param plot_grid:
        :param plot_scope:
        :return:
        '''

        # REMOVE ME LATER
        a = 1

    #########################################################
    ####    ROTATION DISCRIMINATION ANALYSIS FUNCTIONS   ####
    #########################################################

    def plot_rotation_dir_lda(self, plot_transform, s_factor, plot_exp_name, plot_all_expt, acc_type,
                              add_accuracy_trend, output_stats, plot_grid):
        '''

        :param plot_transform:
        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_grid:
        :return:
        '''

        def plot_transform_values(d_data, plot_exp_name, plot_grid):
            '''

            :param r_data:
            :param plot_exp_name:
            :param plot_all_expt:
            :param plot_grid:
            :return:
            '''

            # retrieves the plot values
            # i_expt = list(d_data.exp_name).index(plot_exp_name)
            i_expt = cf.get_expt_index(plot_exp_name, self.data.cluster)
            lda_X, lda_var_exp = d_data.lda[i_expt]['lda_X'], d_data.lda[i_expt]['lda_var_exp']

            # other initialisations
            h_plt, patches, c = [], [], cf.get_plot_col(len(lda_X))

            # initialises the plot axis
            self.init_plot_axes(n_plot=1)
            ax = self.plot_fig.ax[0]

            # creates the
            for i_grp in range(len(lda_X)):
                # creates the plot object
                h_plt.append(ax.plot(lda_X[i_grp][:, 0], lda_X[i_grp][:, 1], 'o', color=c[i_grp]))

                # creates the convex hull
                p_hull = CHull(lda_X[i_grp])
                patches.append(Polygon(lda_X[i_grp][p_hull.vertices, :], True))

            # creates the patch collection
            p = PatchCollection(patches, facecolors=c, alpha=0.4)
            ax.add_collection(p)

            # creates the legend object
            lg_str = cf.flat_list([['{0} ({1})'.format(tt, d) for d in ['CW', 'CCW']] for tt in d_data.ttype])
            ax.set_title('LDA Components (Variance Explained = {:5.2f}%)'.format(lda_var_exp))
            ax.set_xlabel('LDA Component #1')
            ax.set_ylabel('LDA Component #2')
            ax.grid(plot_grid)
            ax.legend(cf.flat_list(h_plt), lg_str, loc=0)

        def create_heatmap_markers(ax, c_mat, t_type):
            '''

            :param ax:
            :param lbl_str:
            :return:
            '''

            # initialisations
            h, lbl, dXY, xL0, yL0 = [], [], 0.1, -0.5, -0.5
            WH = 1 - 2 * dXY

            # sets up the label strings
            lbl_str = cf.flat_list([['{0} ({1})'.format(tt, d) for d in ['CW', 'CCW']] for tt in t_type])

            #
            for i_row in range(np.size(c_mat, axis=0)):
                for i_col in range(np.size(c_mat, axis=0)):
                    # creates the new patch object
                    x0, y0 = (xL0 + i_col) + dXY, (yL0 + i_row) + dXY
                    h.append(Rectangle((x0, y0), WH, WH))

                    # creates the new string
                    lbl.append('True = {}\nDecoded = {}\nPercentage = {:5.2f}%'.format(
                        lbl_str[i_row], lbl_str[i_col], c_mat[i_row, i_col]
                    ))

            # creates the cursor object
            pc = PatchCollection(h, facecolor='g', alpha=0.0, zorder=10)
            ax.add_collection(pc)
            datacursor(pc, formatter=formatter, point_labels=lbl, hover=True)

        def calc_acc_trend(n_cell, y_acc):
            '''

            :param n_cell:
            :param y_acc:
            :return:
            '''

            # memory allocation
            n_cond = np.size(y_acc, axis=1)
            y_trend, r_2 = np.empty(n_cond, dtype=object), np.zeros(n_cond)

            # calculates the trend values for each type
            for i_col in range(np.size(y_acc, axis=1)):
                m, x0, r_2[i_col], _, _ = linregress(n_cell, y_acc[:, i_col])
                y_trend[i_col] = [m, x0]

            # returns the trend values
            return y_trend, r_2

        def create_marker_line(ax, x_mn, type, i_plt, n_expt, mn_hght, col):
            '''

            :param x_mn:
            :param exp_name:
            :param n_plt:
            :param mn_hght:
            :return:
            '''

            # sets the plot-line label strings
            lbl_t = 'Type = {}\nExperiment Count = {}\nMean Accuracy = {:4.1f}%'.format(type, n_expt, x_mn)

            # creates the line plot and adds it to the datacurson list
            h_mn_nw, = ax.plot(i_plt + (mn_hght / 2) * np.array([-1, 1]), x_mn * np.ones(2), c=col,
                               linewidth=2, zorder=100, label=lbl_t)

            # returns the label string and plot object handle
            return lbl_t, h_mn_nw

        # initialisations
        d_data = self.data.discrim.dir
        n_cond = len(d_data.ttype)

        # determines if the user if trying to plot the transform values with the lsqr solver type
        if plot_transform:
            # determines if the user is using the lsqr solver type
            if d_data.solver == 'lsqr':
                # if so, then output an error to screen
                e_str = 'It is not possible to plot the LDA transform values with the "lsqr" solver. Either re-run ' \
                        'the function with a different solver type or de-select transform plotting.'
                cf.show_error(e_str, 'Invalid Solver Type')

                # exits the function with an error flag
                self.calc_ok = False
                return
            else:
                # otherwise, plot the transform values (exit function afterwards)
                plot_transform_values(d_data, plot_exp_name, plot_grid)
                return

        ###################################
        ####    DATA PRE-PROCESSING    ####
        ###################################

        # retrieves the plot values
        if plot_all_expt:
            # case is using all the experiments
            c_mat = np.dstack([x['c_mat'] / d_data.ntrial for x in d_data.lda])
            c_mat_ch = np.dstack([x['c_mat_chance'] / d_data.ntrial for x in d_data.lda])
            n_expt = np.size(c_mat, axis=2)

            # calculates the mean confusion matrix values
            c_mat_mn, y_acc = np.mean(c_mat, axis=2), d_data.y_acc

            # c_mat_ch_mn = np.mean(c_mat_ch, axis=2)
        else:
            # case is using a specific experiment
            i_expt, n_expt = list(d_data.exp_name).index(plot_exp_name), 1
            c_mat_mn, y_acc = d_data.lda[i_expt]['c_mat'] / d_data.ntrial, d_data.y_acc[i_expt, :]

            # sz_nw = (2 * n_cond, 2 * n_cond, 1)
            # c_mat = np.reshape(c_mat, sz_nw)
            # c_mat_ch = np.reshape(d_data.lda[i_expt]['c_mat_chance'] / d_data.ntrial, sz_nw)

        # sets the chance values
        y_acc = y_acc.reshape(1, -1) if n_expt == 1 else y_acc
        y_acc_ch = 0.5 * np.ones((1, 1 + n_cond))
        n_cell, is_multi = np.array([x['n_cell'] for x in d_data.lda]), len(d_data.lda)  > 1

        #######################################
        ####    SUBPLOT INITIALISATIONS    ####
        #######################################

        # sets up the axes dimensions
        tick_lbls = cf.flat_list(['CW', 'CCW'] * n_cond)
        top, bottom, pH, wspace, hspace = 0.9, 0.06, 0.01, 0.2, 0.2
        x = np.arange(2, 2 * n_cond, 2)

        # width ratio
        w_ratio = [0.6, 0.025, 0.025, 0.0]
        w_ratio[-1] = 1 - np.sum(w_ratio[:2])

        # bar graph dimensioning
        x_bar, w_bar = np.arange(n_cond + 1), 0.9
        bar_lbls = ['Cond'] + ['Dir\n({0})'.format(cf.cond_abb(tt)) for tt in d_data.ttype]
        lg_str = [x.replace('\n', ' ') for x in bar_lbls]

        # creates the gridspec object
        gs = gridspec.GridSpec(2, 4, width_ratios=w_ratio, figure=self.plot_fig.fig,
                               wspace=wspace, left=0.085, right=0.98, bottom=bottom, top=top, hspace=0.14)

        # creates the subplots
        use_extra = is_multi and (not output_stats)
        self.plot_fig.ax = np.empty(4 + is_multi, dtype=object)
        self.plot_fig.ax[0] = self.plot_fig.figure.add_subplot(gs[:, 0])
        self.plot_fig.ax[1] = self.plot_fig.figure.add_subplot(gs[:, 1])
        self.plot_fig.ax[2] = self.plot_fig.figure.add_subplot(gs[:, 2])

        if use_extra:
            self.plot_fig.ax[3] = self.plot_fig.figure.add_subplot(gs[0, 3])
            self.plot_fig.ax[4] = self.plot_fig.figure.add_subplot(gs[1, 3])
        else:
            self.plot_fig.ax[3] = self.plot_fig.figure.add_subplot(gs[:, 3])

        ######################################
        ####    ACCURACY HEATMAP SETUP    ####
        ######################################

        # heatmap setup
        c_ofs = 2 * n_cond
        create_heatmap_markers(self.plot_fig.ax[0], 100. * c_mat_mn, d_data.ttype)
        im = self.plot_fig.ax[0].imshow(100. * c_mat_mn, aspect='auto', cmap='hot', origin='upper')

        # sets the heatmap axis properties
        self.plot_fig.ax[0].grid(False)
        self.plot_fig.ax[0].set_xticks(range(2 * n_cond))
        self.plot_fig.ax[0].set_yticks(range(2 * n_cond))
        self.plot_fig.ax[0].set_xticklabels(tick_lbls)
        self.plot_fig.ax[0].set_yticklabels(tick_lbls)
        self.plot_fig.ax[0].get_xaxis().set_ticks_position('top')
        self.plot_fig.ax[0].tick_params(length=0)
        self.plot_fig.ax[0].text(-0.5 - 0.15 * c_ofs, n_cond - 0.5, 'True Condition', size=16,
                                 verticalalignment='center', rotation=90, weight='bold')
        self.plot_fig.ax[0].text(n_cond - 0.5, -0.5 - 0.075 * c_ofs, 'Decoded Condition', size=16,
                                 horizontalalignment='center', weight='bold')

        # sets the condition titles
        for itt, tt in enumerate(d_data.ttype):
            tt_abb = cf.cond_abb(tt)
            self.plot_fig.ax[0].text(-0.5 - 0.1 * c_ofs, 2 * itt + 0.5, tt_abb, size=14, verticalalignment='center',
                                     rotation=90, weight='bold')
            self.plot_fig.ax[0].text(2 * itt + 0.5, -0.5 - 0.0375 * c_ofs, tt_abb, size=14,
                                     horizontalalignment='center', weight='bold')

        # creates the separation lines
        xL, yL = self.plot_fig.ax[0].get_xlim(), self.plot_fig.ax[0].get_ylim()
        for i_cond in range(len(x)):
            self.plot_fig.ax[0].plot(xL, (x[i_cond] - 0.5) * np.ones(2), 'w', linewidth=2)
            self.plot_fig.ax[0].plot((x[i_cond] - 0.5) * np.ones(2), yL, 'w', linewidth=2)

        # creates the colorbar
        cbar = self.plot_fig.figure.colorbar(im, cax=self.plot_fig.ax[1])
        cbar.set_clim([0., 100.])

        # turns off the 3rd axis (not required - only used for providing gap)
        self.plot_fig.ax[2].axis('off')

        #########################################
        ####    DECODING ACCURACY SUBPLOT    ####
        #########################################

        # other parameters
        yL = [0., 102.]
        n_col = np.size(y_acc, axis=1)
        is_single_cond = n_cond == 1

        if acc_type == 'Bar + Bubbleplot':
            # sets the plot colours and values
            y_acc_l = [100 * y_acc[:, i] for i in range(is_single_cond, n_col)]
            y_acc_lmn = np.array([np.mean(x) for x in y_acc_l])
            x_bar = [(x - is_single_cond) for x in range(is_single_cond, n_col)]
            col, b_col = cf.get_plot_col(len(x_bar),is_single_cond), to_rgba_array(np.array(_light_gray) / 255, 1)

            # plots the mean accuracy values
            self.plot_fig.ax[3].bar(x_bar, y_acc_lmn, width=w_bar, color=col, zorder=1)

            # creates the final plot based on the selected type
            cf.create_bubble_boxplot(self.plot_fig.ax[3], y_acc_l, plot_median=False, X0=x_bar,
                                     col=['k'] * len(y_acc_l), s=s_factor * n_cell)

            # sets the bar plot axis properties
            self.plot_fig.ax[3].set_xticks(x_bar)
            self.plot_fig.ax[3].set_xticklabels(np.array(bar_lbls)[is_single_cond:])
        else:
            # sets the x/y plot values
            x_plt = cf.flat_list([['{0}'.format(x)] * np.size(y_acc, axis=0) for x in bar_lbls])
            y_plt = 100. * y_acc.T.flatten()

            # sets the violin/swarmplot dictionaries
            vl_dict = cf.setup_sns_plot_dict(ax=self.plot_fig.ax[3], x=x_plt, y=y_plt, inner=None)
            sw_dict = cf.setup_sns_plot_dict(ax=self.plot_fig.ax[3], x=x_plt, y=y_plt, color='white', edgecolor='gray')

            # creates the violin/swarmplot
            sns.violinplot(**vl_dict)
            sns.swarmplot(**sw_dict)

            # creates the mean accuracy lines
            h_plt_mn, lbl_plt_mn, n_expt = [], [], np.size(y_acc, axis=0)
            for i_plt in range(is_single_cond, np.size(y_acc, axis=1)):
                # creates the avg. marker line
                y_mn = 100. * np.mean(y_acc[:, i_plt])
                l_nw, h_nw = create_marker_line(self.plot_fig.ax[3], y_mn, lg_str[i_plt], i_plt, n_expt, 0.8, 'k')

                # stores the label string/plot handle
                h_plt_mn.append(h_nw)
                lbl_plt_mn.append(l_nw)

            # creates the datacursor
            datacursor(h_plt_mn, formatter=formatter_lbl, point_labels=lbl_plt_mn, hover=True)

        # creates the separation marker lines
        for i_plt in range(n_col - (1 + is_single_cond)):
            self.plot_fig.ax[3].plot((i_plt + 0.5) * np.ones(2), yL, 'k--')

        # creates the line markers
        xL = self.plot_fig.ax[3].get_xlim()
        self.plot_fig.ax[3].plot(xL, 50. * np.ones(2), 'gray', linewidth=2)
        self.plot_fig.ax[3].set_xlim(xL)

        # sets the axis properties
        self.plot_fig.ax[3].set_ylabel('Decoding Accuracy (%)')
        self.plot_fig.ax[3].set_ylim(yL)
        self.plot_fig.ax[3].grid(plot_grid)

        # only output the stats/trends if there is multiple experiments
        if is_multi:
            if output_stats and (not is_single_cond):
                # case is the statistics output

                # sets up the values for the calculations
                i_grp = repmat(np.arange(np.size(y_acc, axis=1)), np.size(y_acc, axis=0), 1)

                # sets up the stats calculations
                results = r_stats.pairwise_wilcox_test(FloatVector(y_acc[:, 1:].flatten('F')),
                                                       FloatVector(i_grp[:, 1:].flatten('F')),
                                                       p_adjust_method='bonf', paired=True)
                p_vals = np.reshape(np.array(results[list(results.names).index('p.value')]), (n_cond - 1, n_cond - 1))

                # sets up the stats table values
                p_str = np.empty((n_cond + 1, n_cond + 1), dtype=object)
                for i_col in range(n_cond + 1):
                    for i_row in range(i_col, n_cond + 1):
                        if i_row == i_col:
                            # case is the diagonal value
                            p_str[i_row, i_col] = 'N/A'
                        else:
                            if i_col == 0:
                                # case is the condition accuracy statistics

                                # sets up the stats calculations
                                results = r_stats.wilcox_test(FloatVector(y_acc[:, i_col]),
                                                              FloatVector(i_grp[:, i_row]), paired=True)
                                _p_vals = results[list(results.names).index('p.value')][0]
                            else:
                                # case is the trial condition direction accuracy statistics
                                _p_vals = p_vals[i_row - 2, i_col - 1]

                            #
                            p_str_nw = '{:5.3f}{}'.format(_p_vals, cf.sig_str_fcn(_p_vals, 0.05))
                            p_str[i_row, i_col] = p_str[i_col, i_row] = p_str_nw

                # sets up the n-value table
                col = cf.get_plot_col(n_cond + 1)
                cf.add_plot_table(self.plot_fig, self.plot_fig.ax[3], table_font, p_str, lg_str, lg_str,
                                  col, col, 'bottom', p_wid=1.5, n_col=1)
                self.plot_fig.ax[3].set_xlabel('Decoding Type')

            else:
                # plots the accuracy points
                col, h_plt = cf.get_plot_col(len(x_bar)), []
                for i_col in range(len(col)):
                    h_plt.append(self.plot_fig.ax[4].plot(n_cell, 100. * d_data.y_acc[:, i_col], 'o', c=col[i_col]))

                # plots the
                self.plot_fig.ax[4].set_xlabel('Cell Count')
                self.plot_fig.ax[4].set_ylabel('Decoding Accuracy (%)')
                self.plot_fig.ax[4].set_ylim([np.floor(np.min(100. * d_data.y_acc)) - 10, 105])
                self.plot_fig.ax[4].grid(plot_grid)
                self.plot_fig.ax[4].legend([x[0] for x in h_plt], lg_str, loc='bottom right')

                #
                if add_accuracy_trend and (len(n_cell) > 1):
                    # calculates the trend values and plots them
                    x_lim, lbl_trend, h_trend = np.array(self.plot_fig.ax[4].get_xlim()), [], []
                    y_trend, r2 = calc_acc_trend(n_cell, 100. * d_data.y_acc)

                    #
                    for b_lbl, y_t, c, _r2 in zip(bar_lbls, y_trend, col, r2):
                        yt_nw = y_t[0] * x_lim + y_t[1]
                        lbl_t = 'Type = {}\nGradient = {:5.2f}\nOffset = {:5.2f}\nR2 = {:5.2f}'.format(
                            b_lbl.replace('\n', ' '), y_t[0], y_t[1], _r2
                        )
                        lbl_trend.append(lbl_t)

                        h_trend_nw, = self.plot_fig.ax[4].plot(x_lim, yt_nw, '--', c=c, label=lbl_t)
                        h_trend.append(h_trend_nw)

                    # creates the datacursor
                    datacursor(h_trend, formatter=formatter_lbl, point_labels=lbl_trend, hover=True)

    def plot_temporal_lda(self, use_stagger, plot_err, show_stats, plot_grid):
        '''

        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_grid:
        :return:
        '''

        def create_multi_plot(ax, xi, y_acc, use_stagger, plot_err):
            '''

            :param ax:
            :param xi:
            :param y_acc:
            :param t_phase:
            :param c:
            :return:
            '''

            # array dimensions and initialisations
            n_expt, n_cond, n_xi = np.shape(y_acc)
            yL, col = [-1., 101.], cf.get_plot_col(n_cond)
            x = np.arange(n_xi)
            xL, yL, h_plt = [x[0], x[-1] + 1.], [0., 100.], []

            #
            for i_cond in range(n_cond):
                # sets the plot values
                xi_plt = x + ((i_cond + 1) / (n_cond + 1) if use_stagger else 0.5)
                y_acc_c = y_acc[:, i_cond, :]

                # sets the median plot values
                y_acc_md = np.median(y_acc_c, axis=0)

                # adds the median plot lines
                h_plt.append(ax.plot(xi_plt, y_acc_md, c=col[i_cond], linewidth=2))

                # creates the errorbars (if requred)
                if plot_err:
                    # sets the lower/upper quartile errorbars
                    y_acc_lq = np.percentile(y_acc_c, 25., axis=0)
                    y_acc_uq = np.percentile(y_acc_c, 75., axis=0)

                    # creates the area patch
                    cf.create_error_area_patch(ax, xi_plt, None, y_acc_lq, col[i_cond], y_err2=y_acc_uq)

            # creates the vertical marker lines
            for xx in np.arange(xL[0] + 1, xL[1]):
                ax.plot(xx * np.ones(2), yL, 'k--')

            # plots the chance line
            ax.plot(xL, 50. * np.ones(2), c='gray', linewidth=2)
            ax.legend([x[0] for x in h_plt], d_data.ttype, loc=4)

            # sets the axis properties
            ax.set_xlim(xL)
            ax.set_ylim(yL)
            ax.set_xticks(x + 0.5)
            ax.set_xticklabels(xi)
            ax.set_ylabel('Decoding Accuracy (%)')

        def setup_bin_stats_values(xi, ttype, y_acc):
            '''

            :param xi:
            :param ttype:
            :param y_acc:
            :return:
            '''

            # memory allocation
            n_ex, n_tt = np.size(y_acc_phs, axis=0), len(ttype)
            _ttype = np.array(ttype).reshape(-1, 1)

            # converts the xi/condition type values into a FactorVector
            ttype_st = FactorVector(np.tile(_ttype, [n_ex, 1, np.size(y_acc, axis=2)]).flatten())
            xi_st = FactorVector(np.tile(xi, [n_ex, n_tt, 1]).flatten())

            # returns the values for analysis
            return ttype_st, xi_st, y_acc[:, 1:, :].flatten()

        # initialisations
        d_data = self.data.discrim.temp
        ttype = d_data.ttype

        # retrieves the important fields
        y_acc_phs, y_acc_ofs = 100. * np.dstack(d_data.y_acc[0]), 100. * np.dstack(d_data.y_acc[1])

        ##################################
        ####    DATA VISUALISATION    ####
        ##################################

        # initialises the plot axes
        self.init_plot_axes(n_row=1, n_col=2)
        ax = self.plot_fig.ax

        if show_stats:
            ############################
            ####    STATS TABLES    ####
            ############################

            # sets the axis properties for both subplots
            for _ax in ax:
                _ax.axis('off')

            # memory allocation
            ttype_abb, n_cond = [cf.cond_abb(x) for x in ttype], np.size(y_acc_phs, axis=1) - 1
            col = cf.get_plot_col(n_cond)

            # creates the stats table for the differing phase duration
            tt_phs, xi_phs, y_phs = setup_bin_stats_values(d_data.xi_phs, ttype, y_acc_phs)
            c_grp_phs, p_str_phs = cfcn.calc_art_stats(tt_phs, xi_phs, y_phs, 'X1')
            cf.add_plot_table(self.plot_fig, 0, table_font, p_str_phs, c_grp_phs, c_grp_phs,
                              col, col, 'top', p_wid=1.5, n_col=1)

            # creates the stats table for the differing phase offset
            tt_ofs, xi_ofs, y_ofs = setup_bin_stats_values(d_data.xi_ofs, ttype, y_acc_ofs)
            c_grp_ofs, p_str_ofs = cfcn.calc_art_stats(tt_ofs, xi_ofs, y_ofs, 'X1')
            cf.add_plot_table(self.plot_fig, 1, table_font, p_str_ofs, c_grp_ofs, c_grp_ofs,
                              col, col, 'top', p_wid=1.5, n_col=1)

        else:
            #################################
            ####    SUBPLOT CREATIONS    ####
            #################################

            # creates the multiple boxplot
            create_multi_plot(ax[0], d_data.xi_phs, y_acc_phs[:, 1:, :], use_stagger, plot_err)
            create_multi_plot(ax[1], d_data.xi_ofs, y_acc_ofs[:, 1:, :], use_stagger, plot_err)

            # sets the titles for both subplots
            ax[0].set_title('Decoding Accuracy vs Phase Duration\n(Offset = 0s)')
            ax[1].set_title('Decoding Accuracy vs Phase Offset\n(Duration = {:5.2f}s)'.format(d_data.phs_const))

            # sets the axis properties for both subplots
            for _ax, _x_lbl in zip(ax, ['Phase Duration (s)', 'Phase Offset (s)']):
                _ax.set_xlabel(_x_lbl)
                _ax.grid(plot_grid)

    def plot_individual_lda(self, plot_exp_name, plot_all_expt, decode_type, dir_type_1, dir_type_2, m_size, plot_grid):
        '''

        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_grid:
        :return:
        '''

        def create_sv_plot(ax, y_acc, y_acc_pop, decode_type, exp_name, plot_grid):
            '''

            :param ax:
            :param exp_name:
            :param y_acc:
            :return:
            '''

            def create_marker_line(ax, x_mn, exp_name, i_plt, n_plt, mn_hght, col, is_pop):
                '''

                :param x_mn:
                :param exp_name:
                :param n_plt:
                :param mn_hght:
                :return:
                '''

                # sets the plot-line label strings
                lbl_t = 'Expt. Name = {}\nCell Count = {}\n{} Mean Accuracy = {:4.1f}%'.format(
                    exp_name, n_plt, 'Population' if is_pop else 'Individual Cell', x_mn
                )

                # creates the line plot and adds it to the datacurson list
                h_mn_nw, = ax.plot(x_mn * np.ones(2), i_plt + (mn_hght / 2) * np.array([-1, 1]), c=col,
                                   linewidth=2, zorder=100, label=lbl_t)

                # returns the label string and plot object handle
                return lbl_t, h_mn_nw

            # sets the dataframe arrays/parameters
            c_name = ['Decoding Accuracy (%)', 'Expt Number']
            y_acc_plt = [np.concatenate(([-5.], 100.* x, [105.])) for x in y_acc]

            # sets the x/y plot values
            x_plt = cf.flat_list(y_acc_plt)
            y_plt = cf.flat_list([['#{0}'.format(x + 1)] * len(y) for x, y in zip(range(len(y_acc)), y_acc_plt)])

            # creates the violin/swarmplot
            sns.violinplot(**cf.setup_sns_plot_dict(ax=ax, x=x_plt, y=y_plt, inner=None))
            sns.swarmplot(**cf.setup_sns_plot_dict(ax=ax, x=x_plt, y=y_plt, color='white', edgecolor='gray'))

            # creates the mean plot lines
            mn_hght, h_mn, lbl_trend = min(0.8, 0.04 * len(y_acc)), [], []
            for i_plt in range(len(y_acc)):
                # sets the plot values for the current experiment
                n_plt = len(y_acc[i_plt])
                x_mn, x_mn_pop = 100. * np.mean(y_acc[i_plt]), 100. * np.mean(y_acc_pop[i_plt])

                # creates the mean
                lbl_t, h_plt_nw = create_marker_line(ax, x_mn, exp_name[i_plt][0], i_plt, n_plt, mn_hght, 'k', False)
                lbl_trend.append(lbl_t)
                h_mn.append(h_plt_nw)

                # creates the mean
                lbl_t, h_plt_nw = create_marker_line(ax, x_mn_pop, exp_name[i_plt][0], i_plt, n_plt, mn_hght, 'r', True)
                lbl_trend.append(lbl_t)
                h_mn.append(h_plt_nw)

            # creates the datacursor
            datacursor(h_mn, formatter=formatter_lbl, point_labels=lbl_trend, hover=True)

            # plots the chance line
            yL = ax.get_ylim()
            ax.plot(50. * np.ones(2), yL, 'gray', linewidth=2)
            ax.set_ylim(yL)

            # sets the axis properties
            ax.set_xlim([-1, 101])
            ax.set_title(decode_type)
            ax.set_xlabel(c_name[0])
            ax.set_ylabel(c_name[1])
            ax.grid(plot_grid)

        # initialisations
        d_data_i, d_data_d = self.data.discrim.indiv, self.data.discrim.dir
        n_cond, ttype = len(d_data_d.ttype), d_data_d.ttype
        n_h = 4 * d_data_i.ntrial

        # parameters
        dx_p, col = 0.1, cf.get_plot_col(2)
        x_p = np.arange(0., 1.001, dx_p)

        # determines the indices of the direction trial types
        ind_d1, ind_d2 = ttype.index(dir_type_1), ttype.index(dir_type_2)
        if ind_d1 == ind_d2:
            # if the user selected identical direction trial types, then output an error to screen
            e_str = 'It is not possible to run this function with identical direction trial types.\n' \
                    'Re-run this function with unique direction trial types.'
            cf.show_error(e_str, 'Invalid Trial Type Selection')

            # sets the acceptance flag to false and exits the function
            self.calc_ok = False
            return

        ###################################
        ####    DATA PRE-PROCESSING    ####
        ###################################

        # bar graph dimensioning
        d_type = ['Condition'] + ['Dir ({0})'.format(tt) for tt in ttype]
        id_type = d_type.index(decode_type)

        # retrieves the plot values
        if plot_all_expt:
            # case is using all the experiments

            # calculates the mean confusion matrix values
            y_acc, exp_name = d_data_d.y_acc, d_data_i.exp_name
            y_acc_sw = [x[:, id_type] for x in d_data_i.y_acc]
            y_acc_i = np.vstack(d_data_i.y_acc)

        else:
            # case is using a specific experiment
            i_expt, n_expt = list(d_data_d.exp_name).index(plot_exp_name), 1
            y_acc, y_acc_i = d_data_d.y_acc[i_expt, :].reshape(1, -1), d_data_i.y_acc[i_expt]
            y_acc_sw = [d_data_i.y_acc[i_expt][:, id_type]]
            exp_name = [d_data_i.exp_name[i_expt]]

        # combines the individual responses into a single list
        y_acc_mn = np.mean(y_acc, axis=0)

        # sets up the heatmap values
        im_h = np.zeros((n_h+1, n_h+1), dtype=int)
        i_y, i_x = (y_acc_i[:, ind_d1 + 1] * n_h).astype(int), (y_acc_i[:, ind_d2 + 1] * n_h).astype(int)
        ind_h, n_hc = np.unique(np.vstack((i_x, i_y)).T, axis=0, return_counts=True)

        # creates the heatmap
        for i in range(len(n_hc)):
            im_h[ind_h[i, 1], ind_h[i, 0]] = n_hc[i]

        #############################
        ####    SUBPLOT SETUP    ####
        #############################

        # width ratio
        w_ratio = [0.3, 0.0]
        w_ratio[1] = 1 - np.sum(w_ratio)

        # creates the gridspec object
        top, bottom, wspace, hspace = 0.96, 0.06, 0.2, 0.2
        gs = gridspec.GridSpec(1, 2, width_ratios=w_ratio, figure=self.plot_fig.fig,
                               wspace=wspace, left=0.07, right=0.96, bottom=bottom, top=top, hspace=0.15)

        # creates the subplots
        self.plot_fig.ax = np.empty(3, dtype=object)
        self.plot_fig.ax[0] = self.plot_fig.figure.add_subplot(gs[:, 0])
        self.plot_fig.ax[1] = self.plot_fig.figure.add_subplot(gs[:, 1])

        #################################
        ####    SUBPLOT CREATIONS    ####
        #################################

        # creates the individual cell accuracy swarmplot
        create_sv_plot(self.plot_fig.ax[0], y_acc_sw, d_data_d.y_acc[:, id_type], decode_type, exp_name, plot_grid)

        # creates the scatterplot
        i_plt = np.where(im_h > 0)
        x_plt, y_plt, z_plt = i_plt[0], i_plt[1], im_h[i_plt[0], i_plt[1]]

        # sets the datacursor labels
        lbl = [
            '{} = {:4.1f}%\n{} = {:4.1f}%\nCount = {}'.format(
                ttype[ind_d1], (100 / n_h) * x, ttype[ind_d2], (100 / n_h) * y, z
            ) for x, y, z in zip(x_plt, y_plt, z_plt)
        ]

        # creates the scatterplot
        h = self.plot_fig.ax[1].scatter(x_plt, y_plt, edgecolors='k', s=m_size * (z_plt / max(z_plt)))
        datacursor(h, formatter=formatter, point_labels=lbl, hover=True)

        # plots the region demarkation lines
        ax_lim, a = [-dx_p * n_h / 4, n_h * (1 + dx_p / 4)], np.ones(2)
        self.plot_fig.ax[1].plot((n_h / 2) * a, ax_lim, c='gray', linewidth=2)
        self.plot_fig.ax[1].plot(n_h * y_acc_mn[1] * a, ax_lim, 'r--', linewidth=2)
        self.plot_fig.ax[1].plot(ax_lim, (n_h / 2) * a, c='gray', linewidth=2)
        self.plot_fig.ax[1].plot(ax_lim, n_h * y_acc_mn[2] * a, 'r--', linewidth=2)

        # sets the axis properties
        self.plot_fig.ax[1].set_xticks(x_p * n_h)
        self.plot_fig.ax[1].set_yticks(x_p * n_h)
        self.plot_fig.ax[1].set_xticklabels((100. * x_p).astype(int))
        self.plot_fig.ax[1].set_yticklabels((100. * x_p).astype(int))
        self.plot_fig.ax[1].set_xlabel('{0} Decoding Accuracy (%)'.format(dir_type_1))
        self.plot_fig.ax[1].set_ylabel('{0} Decoding Accuracy (%)'.format(dir_type_2))
        self.plot_fig.ax[1].set_xlim(ax_lim)
        self.plot_fig.ax[1].set_ylim(ax_lim)
        self.plot_fig.ax[1].set_title('Direction Decoding Accuracy')
        self.plot_fig.ax[1].grid(plot_grid)

    def plot_shuffled_lda(self, i_cell_1, i_cell_2, plot_exp_name, plot_corr, dir_type_1, dir_type_2,
                          m_size, plot_grid):
        '''

        :param plot_exp_name:
        :param plot_all_expt:
        :param plot_grid:
        :return:
        '''

        def get_pw_corr(d_data):

            # sets the synchronous/non-synchronous pairwise correlations
            pw_corr = [[] for _ in range(len(d_data.pw_corr))]

            #
            for i in range(len(pw_corr)):
                pw_tmp = [[] for _ in range(len(d_data.pw_corr[i]))]
                for j in range(len(pw_tmp)):
                    i_triu = np.triu(d_data.pw_corr[i][j] + 2, k=1)
                    pw_tmp[j] = i_triu[i_triu != 0] - 2

                pw_corr[i] = np.vstack(pw_tmp).T
                ii = np.logical_not(np.any(np.isnan(pw_corr[i]), axis=1))
                pw_corr[i] = pw_corr[i][ii, :]

            # returns the final combined array
            return np.vstack(pw_corr)

        def create_correl_subfig(ax, d_data_s, d_data_ns, ttype, m_size, plot_grid):
            '''

            :param ax:
            :param d_data_d:
            :param d_data_s:
            :param ind:
            :return:
            '''

            sum_norm = lambda x: x / np.sum(x)

            # initialisations
            d_xi, p_str = 0.1, np.empty((1, 2), dtype=object)
            col, h_plt, xi = cf.get_plot_col(2), [], np.arange(-1, 1.01, d_xi)
            lg_str = ['Synchronous', 'Non-Synchronous']

            # creates the
            h_plt.append(ax[0].plot(d_data_s[:, 0], d_data_s[:, 1], '.', color=col[0], zorder=1, markersize=m_size/10))
            h_plt.append(ax[0].plot(d_data_ns[:, 0], d_data_ns[:, 1], '.', color=col[1], zorder=2, markersize=m_size/10))
            ax[0].legend([x[0] for x in h_plt], lg_str, loc=0)

            # creates the histograms for each of the data types
            xi_h = 0.5 * (xi[1:] + xi[:-1])
            for i in range(2):
                # creates the histograms for the synochronous/non-synchronous data
                h_pw_s = sum_norm(np.histogram(d_data_s[:, i], bins=xi)[0])
                h_pw_ns = sum_norm(np.histogram(d_data_ns[:, i], bins=xi)[0])

                # sets the p-value string
                p_val = bartlett(h_pw_s, h_pw_ns)[1]
                p_str[0, i] = '{:5.3e}{}'.format(p_val, cf.sig_str_fcn(p_val, 0.05))

                if i == 0:
                    ax[i + 1].bar(xi_h - d_xi / 4, h_pw_s, color=col[0], width=d_xi / 2)
                    ax[i + 1].bar(xi_h + d_xi / 4, h_pw_ns, color=col[1], width=d_xi / 2)

                    ax[i + 1].set_xlim([-1, 1])
                    ax[i + 1].set_yticks(ax[i + 1].get_ylim())
                else:
                    ax[i + 1].barh(xi_h - d_xi / 4, h_pw_s, color=col[0], height=d_xi / 2)
                    ax[i + 1].barh(xi_h + d_xi / 4, h_pw_ns, color=col[1], height=d_xi / 2)

                    ax[i + 1].set_ylim([-1, 1])
                    ax[i + 1].set_xticks(ax[i + 1].get_xlim())

                # sets the plot grid
                ax[i + 1].grid(plot_grid)

            # plots the zero-correlation lines
            ax[0].plot([0, 0], [-1, 1], 'k', linewidth=2, zorder=100)
            ax[0].plot([-1, 1], [0, 0], 'k', linewidth=2, zorder=100)

            # sets the axis properties
            ax[0].set_xlim([-1, 1])
            ax[0].set_ylim([-1, 1])
            ax[0].set_xlabel('{0} Pairwise Correlation'.format(ttype[0]))
            ax[0].set_ylabel('{0} Pairwise Correlation'.format(ttype[1]))
            ax[0].grid(plot_grid)

            # sets up the n-value table
            cf.add_plot_table(self.plot_fig, 1, table_font, p_str, ['P-Value'], ttype,
                              cf.get_plot_col(1, 4), cf.get_plot_col(2, 2), 'bottom', p_wid=1.5, n_col=1)

            # resets the position of the vertical bar graph
            l0, b0, w0, h0 = ax[0].get_position().bounds
            l2, b2, w2, h2 = ax[2].get_position().bounds
            ax[2].set_position([l2, b0, w2, h0])

        # initialisations
        d_data_s, d_data_d = self.data.discrim.shuffle, self.data.discrim.dir
        n_cond, ttype, nshuffle = len(d_data_d.ttype), d_data_d.ttype, d_data_s.nshuffle
        bar_lbls = ['Cond'] + ['Dir\n({0})'.format(cf.cond_abb(tt)) for tt in ttype]
        e_str, t_str = None, None

        # determines the indices of the direction trial types
        ind_d1, ind_d2 = ttype.index(dir_type_1), ttype.index(dir_type_2)
        if ind_d1 == ind_d2:
            # if the user selected identical direction trial types, then output an error to screen
            e_str = 'It is not possible to run this function with identical direction trial types.\n' \
                    'Re-run this function with unique direction trial types.'
            t_str, 'Invalid Trial Type Selection'

        elif plot_corr:
            # determines if the cell indices are unique
            if i_cell_1 == i_cell_2:
                # if the user selected identical cell indices, then output an error to screen
                e_str = 'It is not possible to run this function with identical cell indices.\n' \
                        'Re-run this function with unique cell indices types.'
                t_str = 'Invalid Cell Indices'

            else:
                # retrieves the experiment index and cell count for the experiment
                i_expt = list(d_data_d.exp_name).index(plot_exp_name)
                n_cell = np.size(d_data_d.z_corr[i_expt][0], axis=0)

                # determines if the input indices don't exceed the number of cells for this experiment
                if (i_cell_1 > n_cell) or (i_cell_2 > n_cell):
                    # if so, then output an error to screen
                    e_str = 'The or or both of the cell indices exceeds the number of cells for this ' \
                            'experiment ({0}). \nRe-run this function feasible cell indices types.'
                    t_str = 'Invalid Cell Indices'

        # determines if there were any errors within the input parameters
        if e_str is not None:
            # if so, then output the error to screen
            cf.show_error(e_str, t_str)

            # sets the acceptance flag to false and exits the function
            self.calc_ok = False
            return

        # array dimensioning
        n_expt = np.size(d_data_d.y_acc, axis=0)

        # creates the graph based on the user selection
        if plot_corr:
            ###################################
            ####    DATA PRE-PROCESSING    ####
            ###################################

            # memory allocation
            d_type = [dir_type_1, dir_type_2]
            z_corr, pw_corr = np.empty(2, dtype=object), np.zeros(2)

            # retrieves the z-score/pair-wise correlation values
            if i_cell_2 < i_cell_1:
                # sets the cell indices
                i_cell = [i_cell_2, i_cell_1]

                # z-score values
                z_corr[0] = d_data_d.z_corr[i_expt][ind_d1][i_cell_2 - 1, i_cell_1 - 1]
                z_corr[1] = d_data_d.z_corr[i_expt][ind_d2][i_cell_2 - 1, i_cell_1 - 1]

                # pair-wise correlation values
                pw_corr[0] = d_data_d.pw_corr[i_expt][ind_d1][i_cell_2 - 1, i_cell_1 - 1]
                pw_corr[1] = d_data_d.pw_corr[i_expt][ind_d2][i_cell_2 - 1, i_cell_1 - 1]
            else:
                # sets the cell indices
                i_cell = [i_cell_1, i_cell_2]

                # z-score values
                z_corr[0] = d_data_d.z_corr[i_expt][ind_d1][i_cell_1 - 1, i_cell_2 - 1]
                z_corr[1] = d_data_d.z_corr[i_expt][ind_d2][i_cell_1 - 1, i_cell_2 - 1]

                # pair-wise correlation values
                pw_corr[0] = d_data_d.pw_corr[i_expt][ind_d1][i_cell_1 - 1, i_cell_2 - 1]
                pw_corr[1] = d_data_d.pw_corr[i_expt][ind_d2][i_cell_1 - 1, i_cell_2 - 1]

            #############################
            ####    SUBPLOT SETUP    ####
            #############################

            # initialises the plot axes
            self.plot_fig.setup_plot_axis(n_row=1, n_col=2)

            #######################################
            ####    MAIN AXIS SUBPLOT SETUP    ####
            #######################################

            # axis properties
            xL = [-3, 3]

            # creates the plots for each type
            for i_ax, ax in enumerate(self.plot_fig.ax):
                # creates the scatter-plot
                ax.scatter(z_corr[i_ax][:, 0], z_corr[i_ax][:, 1], marker='.', c='k', s=m_size)
                ax.plot(xL, xL, 'k', linewidth=2)

                # sets the axis labels/titles
                ax.set_title(d_type[i_ax])
                ax.set_xlabel('Z-Score Response (Cell #{0})'.format(i_cell[0]))
                ax.set_ylabel('Z-Score Response (Cell #{0})'.format(i_cell[1]))

                # sets the axis properties
                ax.set_xlim(xL)
                ax.set_ylim(xL)
                ax.grid(plot_grid)

                # displays the r-value
                ax.text(-2.9, 2.8, 'r = {:4.2f}'.format(pw_corr[i_ax]))
        else:

            ###################################
            ####    DATA PRE-PROCESSING    ####
            ###################################

            # sets up the synchronised data labels
            x_s = x_ns = list(repmat(bar_lbls, n_expt, 1).flatten())
            y_s = list(100. * d_data_d.y_acc.flatten())
            z_s = list(repmat(['Synchronous'], n_expt, n_cond + 1).flatten())

            # sets up the non-sychronised data labels
            y_ns = cf.flat_list([100. * np.mean(x, axis=1).flatten() for x in dcopy(d_data_s.y_acc)])
            z_ns = list(repmat(['Non-Synchronous'], n_expt, n_cond + 1).flatten())

            #############################
            ####    SUBPLOT SETUP    ####
            #############################

            # sets up the axes dimensions
            r_hw = self.plot_fig.height() / self.plot_fig.width()
            top, bottom, pH, wspace, hspace = 0.98, 0.06, 0.01, 0.1, 0.1

            # memory allocation
            n_col, n_row = 9, 5
            w_ratio, h_ratio = np.zeros(n_col), np.zeros(n_row)

            # calculates the width/height ratios of each sub-plot block
            h_ratio[0] = 0.1
            w_ratio[3], w_ratio[-1] = 0.065, r_hw * h_ratio[0]
            w_ratio[w_ratio == 0] = (1 - sum(w_ratio[w_ratio > 0])) / sum(w_ratio == 0)
            h_ratio[h_ratio == 0] = (1 - sum(h_ratio[h_ratio > 0])) / sum(h_ratio == 0)

            # creates the gridspec object
            gs = gridspec.GridSpec(n_row, n_col, figure=self.plot_fig.fig, width_ratios=w_ratio, height_ratios=h_ratio,
                                   wspace=wspace, hspace=hspace, left=0.05, right=0.98, bottom=bottom, top=top)

            # sets up the main plot axis
            self.plot_fig.ax = np.empty(7, dtype=object)
            self.plot_fig.ax[0] = self.plot_fig.figure.add_subplot(gs[:, :3])

            # sets up the first correlation axis
            self.plot_fig.ax[1] = self.plot_fig.figure.add_subplot(gs[1:, 4:8])
            self.plot_fig.ax[2] = self.plot_fig.figure.add_subplot(gs[0, 4:8], xticklabels=[], yticklabels=[])
            self.plot_fig.ax[3] = self.plot_fig.figure.add_subplot(gs[1:, -1], xticklabels=[], yticklabels=[])

            #######################################
            ####    MAIN AXIS SUBPLOT SETUP    ####
            #######################################

            # subplot properties
            p_mx, f_alpha = 102., [0.9, 0.5]
            col = cf.get_plot_col(n_cond + 1)

            # sets up the swarmplot dictionary
            sw_dict = cf.setup_sns_plot_dict(ax=self.plot_fig.ax[0], x=x_s + x_ns, y=y_s + y_ns,
                                             hue=z_s + z_ns, dodge=True, size=6, color='white')
            vl_dict = cf.setup_sns_plot_dict(ax=self.plot_fig.ax[0], x=x_s + x_ns, y=y_s + y_ns,
                                             hue=z_s + z_ns, inner=None)

            # creates the swarmplot and retrieves the collection objects
            sns.violinplot(**vl_dict)
            sns.swarmplot(**sw_dict)

            # retrieves the violin/swarmplot objects
            c = self.plot_fig.ax[0].collections
            c_v = [x for x in c if isinstance(x, matplotlib.collections.PolyCollection)]
            c_s = [x for x in c if isinstance(x, matplotlib.collections.PathCollection)]

            # updates the plot marker colours
            for i in range(n_cond + 1):
                # sets the synchronous violin/swarmplot objects
                c_v[2 * i].set_color(col[i])
                c_v[2 * i].set_alpha(f_alpha[0])
                c_s[2 * i].set_color('w')

                # sets the non-synchronous violin/swarmplot objects
                c_v[2 * i + 1].set_color(col[i])
                c_v[2 * i + 1].set_alpha(f_alpha[1])
                c_s[2 * i + 1].set_color('w')

            # plots the separation markers
            yL = [-2., p_mx]
            for i_plt in range(n_cond):
                self.plot_fig.ax[0].plot((i_plt + 0.5) * np.ones(2), yL, 'k--')

            # plots the chance markerline
            xL = self.plot_fig.ax[0].get_xlim()
            self.plot_fig.ax[0].plot(xL, 50. * np.ones(2), c='gray', linewidth=2)
            self.plot_fig.ax[0].set_xlim(xL)

            # updates the legend
            lg_patch = [Patch(facecolor='k', edgecolor='k', label='Synchronous', alpha=f_alpha[0]),
                        Patch(facecolor='k', edgecolor='k', label='Non-Synchronous', alpha=f_alpha[1])]
            self.plot_fig.ax[0].legend(handles=lg_patch, ncol=2, loc=4)

            # sets the bar plot axis properties
            self.plot_fig.ax[0].set_ylabel('Decoding Accuracy (%)')
            self.plot_fig.ax[0].set_ylim(yL)
            self.plot_fig.ax[0].grid(plot_grid)

            #########################################
            ####    CORRELATION SUBPLOT SETUP    ####
            #########################################

            # retrieves the pairwise correlations
            pw_s, pw_n = get_pw_corr(d_data_d), get_pw_corr(d_data_s)

            # creates the correlation sub-figure plots
            ttype, p_col = [dir_type_1, dir_type_2], np.array([ind_d1, ind_d2])
            create_correl_subfig(self.plot_fig.ax[1:], pw_s[:, p_col], pw_n[:, p_col], ttype, m_size, plot_grid)

    def plot_partial_lda(self, err_type, y_upper, x_max, use_x_max, use_stagger, m_size, plot_grid):
        '''

        :param plot_grid:
        :return:
        '''

        # initialisations
        d_data_p = self.data.discrim.part
        n_cond, ttype = len(d_data_p.ttype), d_data_p.ttype

        ###################################
        ####    DATA PRE-PROCESSING    ####
        ###################################

        # calculates the mean of the decoding accuracy value across all experiments
        n_expt = np.size(d_data_p.y_acc, axis=0)
        y_acc, xi = np.nanmean(d_data_p.y_acc, axis=0), [0] + d_data_p.xi
        zz = np.zeros((np.size(y_acc, axis=0), 1))

        # calculates the mean, min/max and SEM decoding accuracy values over all shuffles
        y_acc_mu = 100. * np.hstack((zz, np.nanmean(y_acc, axis=2)))
        y_acc_md = 100. * np.hstack((zz, np.nanmedian(y_acc, axis=2)))

        # sets up the min/max and SEM values
        y_acc_min = 100. * np.hstack((zz, np.nanmin(y_acc, axis=2)))
        y_acc_max = 100. * np.hstack((zz, np.nanmax(y_acc, axis=2)))
        y_acc_sem = 100. * np.hstack((zz, np.nanstd(y_acc, axis=2) / (d_data_p.nshuffle ** 0.5)))

        # sets up the interquartile range values
        y_acc_uq = 100. * np.hstack((zz, np.nanpercentile(y_acc, 75, axis=2)))
        y_acc_lq = 100. * np.hstack((zz, np.nanpercentile(y_acc, 25, axis=2)))

        #############################
        ####    SUBPLOT SETUP    ####
        #############################

        # sets up the plot axis
        self.plot_fig.setup_plot_axis()

        #################################
        ####    SUBPLOT CREATIONS    ####
        #################################

        # bar graph dimensioning
        plt_lbls = ['Cond'] + ['Dir ({0})'.format(cf.cond_abb(tt)) for tt in ttype]
        col, ax, c_sz, p_x = cf.get_plot_col(len(plt_lbls)), self.plot_fig.ax[0], 10, 0.25
        xi_del, h_plt = use_stagger * p_x * (np.arange(len(col)) - (len(col) - 1) / 2), []

        #
        for i_plt in range(len(col)):
            # plots the mean accuracy values
            xi_nw = [0] + list(np.array(xi[1:]) + xi_del[i_plt])
            if err_type == 'IQR':
                #
                h_plt.append(ax.plot(xi_nw, y_acc_md[i_plt, :], 'o-', c=col[i_plt], markersize=m_size))
                cf.create_error_area_patch(ax, xi_nw, None, y_acc_lq[i_plt, :], col[i_plt], y_err2=y_acc_uq[i_plt, :])
            else:
                h_plt.append(ax.plot(xi_nw, y_acc_mu[i_plt, :], 'o-', c=col[i_plt], markersize=m_size))

                # plots the errorbars
                if err_type == 'None':
                    continue
                elif err_type == 'SEM':
                    # case is plotting the SEM errorbars
                    cf.create_error_area_patch(ax, xi_nw, y_acc_mu[i_plt, :], y_acc_sem[i_plt, :], col[i_plt])
                else:
                    # case is plotting the SEM errorbars
                    cf.create_error_area_patch(ax, xi_nw, None, y_acc_min[i_plt, :], col[i_plt],
                                               y_err2=y_acc_max[i_plt, :])

        # sets the upper x-axis limits
        if use_x_max:
            xL = [0, x_max]
        else:
            xL = ax.get_xlim()

        # plots the chance line
        ax.plot(xL, 50 * np.ones(2), 'r--')
        ax.plot(xL, y_upper * np.ones(2), 'r--')
        ax.set_xlim(xL)

        # updates the ticklabels (if max cell count is not too high)
        if xi[-1] < 100:
            ax.set_xticks(xi)

        # sets the axis properties
        ax.legend([x[0] for x in h_plt], plt_lbls, loc=4)
        ax.set_yticks(np.arange(0, 100.1, 10))
        ax.set_title('Experiment Count = {0}'.format(n_expt))
        ax.set_xlabel('Cell Count')
        ax.set_ylabel('Decoding Accuracy (%)')
        ax.set_ylim([0, 105])
        ax.grid(plot_grid)

    def plot_acc_filt_lda(self, s_factor, acc_type, plot_grid):
        '''

        :param acc_type:
        :param plot_grid:
        :return:
        '''

        # retrieves the data classes
        d_data_d, d_data_f = self.data.discrim.dir, self.data.discrim.filt
        n_cond, ttype = len(d_data_d.ttype), d_data_d.ttype

        ###################################
        ####    DATA PRE-PROCESSING    ####
        ###################################

        # sets the decoding accuracies for the unfiltered/filtered datasets
        y_acc = np.empty(2, dtype=object)
        y_acc[0], y_acc[1] = d_data_d.y_acc, d_data_f.y_acc

        # sets the cell count for the unfiltered/filtered datasets
        n_cell = np.empty(2, dtype=object)
        n_cell[0] = np.array([x['n_cell'] for x in d_data_d.lda])
        n_cell[1] = np.array([x['n_cell'] for x in d_data_f.lda])

        # bar graph dimensioning
        x_bar, w_bar = np.arange(n_cond + 1) - 0.5, 0.425
        bar_lbls = ['Cond'] + ['Dir\n({0})'.format(cf.cond_abb(tt)) for tt in ttype]
        lg_str = ['Unfiltered', 'Filtered (Min={0}%/Max={1}%)'.format(d_data_f.yaccmn, d_data_f.yaccmx)]

        #######################################
        ####    SUBPLOT INITIALISATIONS    ####
        #######################################

        # initialises the plot axis
        self.plot_fig.setup_plot_axis()
        ax = self.plot_fig.ax[0]

        #########################################
        ####    DECODING ACCURACY SUBPLOT    ####
        #########################################

        # other parameters
        yL, xL, alpha = [0., 110.], [-0.5, (n_cond + 0.5)], [1.0, 0.5]
        col, b_col = cf.get_plot_col(len(x_bar)), to_rgba_array(np.array(_light_gray) / 255, 1)

        if acc_type == 'Bar + Bubbleplot':
            # case is the bar + bubble plot

            # sets the legend location
            lg_loc = 2

            # sets the plot colours and values
            for i_c in range(len(y_acc)):
                # sets the new x-location of the bars
                x_bar_nw = x_bar + (2 * i_c + 1) / 4

                # retrieves the individual plot values
                y_acc_mn = 100. * np.mean(y_acc[i_c], axis=0)
                y_acc_l = [100 * y_acc[i_c][:, i] for i in range(np.size(y_acc[i_c], axis=1))]

                # plots the mean accuracy values
                ax.bar(x_bar_nw, y_acc_mn, width=w_bar, color=col, alpha=alpha[i_c], zorder=1)

                # creates the final plot based on the selected type
                cf.create_bubble_boxplot(ax, y_acc_l, plot_median=False, X0=x_bar_nw,
                                         col=['k'] * len(y_acc_l), s=s_factor * n_cell[i_c], wid=0.40)

            # sets the bar plot axis properties
            ax.set_xticks(x_bar + 0.5)
            ax.set_xticklabels(bar_lbls)

        else:
            # case is the swarm/violinplot

            # sets the legend location
            lg_loc = 4

            # sets the x/y plot values
            x_plt = cf.flat_list(
                [cf.flat_list([['{0}'.format(x)] * np.size(y, axis=0) for x in bar_lbls]) for y in y_acc])
            y_plt = cf.flat_list([100. * x.T.flatten() for x in y_acc])
            z_plt = cf.flat_list([[x] * np.prod(np.shape(y)) for x, y in zip(lg_str, y_acc)])

            # sets up the swarmplot dictionary
            sw_dict = cf.setup_sns_plot_dict(ax=ax, x=x_plt, y=y_plt, hue=z_plt, dodge=True, size=6, color='white')
            vl_dict = cf.setup_sns_plot_dict(ax=ax, x=x_plt, y=y_plt, hue=z_plt, inner=None)

            # creates the violin/swarmplot
            sns.violinplot(**vl_dict)
            sns.swarmplot(**sw_dict)

            # retrieves the violin/swarmplot objects
            c = ax.collections
            c_v = [x for x in c if isinstance(x, matplotlib.collections.PolyCollection)]
            c_s = [x for x in c if isinstance(x, matplotlib.collections.PathCollection)]

            # updates the plot marker colours
            for i in range(n_cond + 1):
                # sets the synchronous violin/swarmplot objects
                c_v[2 * i].set_color(col[i])
                c_v[2 * i].set_alpha(alpha[0])
                c_s[2 * i].set_color('w')

                # sets the non-synchronous violin/swarmplot objects
                c_v[2 * i + 1].set_color(col[i])
                c_v[2 * i + 1].set_alpha(alpha[1])
                c_s[2 * i + 1].set_color('w')

        # creates the legend object
        lg_patch = [Patch(facecolor='k', edgecolor='k', label=lg_str[0], alpha=alpha[0]),
                    Patch(facecolor='k', edgecolor='k', label=lg_str[1], alpha=alpha[1])]
        ax.legend(handles=lg_patch, ncol=2, loc=lg_loc)

        # creates the separation marker lines
        for i_plt in range(np.size(y_acc[0], axis=1) - 1):
            ax.plot((i_plt + 0.5) * np.ones(2), yL, 'k--')

        # plots the chance line
        ax.plot(xL, 50. * np.ones(2), 'gray', linewidth=2)

        # sets the other general axis properties
        ax.set_ylim(yL)
        ax.set_xlim(xL)
        ax.grid(plot_grid)
        ax.set_ylabel('Decoding Accuracy (%)')

    def plot_lda_weights(self, error_type, wght_thresh, plot_cond, plot_layer, plot_comp, plot_grid):
        '''

        :param plot_grid:
        :return:
        '''

        def get_channel_depths(data, d_data):
            '''

            :param cluster:
            :return:
            '''

            # retrieves the important indices
            i_expt, c_ind = d_data.i_expt, d_data.c_ind
            i_c = [np.where(x)[0] for x in d_data.i_cell]

            # memory allocation
            n_tt = np.size(c_ind, axis=1)
            A = np.empty(n_tt, dtype=object)
            chL, chD, cW = dcopy(A), dcopy(A), dcopy(A)

            # retrieves the channel depth map/indices over each cluster
            _data = cfcn.reduce_cluster_data(data, i_expt)
            chDepth = [c['chDepth'][_ic] for c, _ic in zip(_data._cluster, i_c)]
            chMap = [c['expInfo']['channel_map'] for c in _data._cluster]

            # retrieves the channel layer/depths for each valid cluster
            chL0 = [c['chLayer'][_ic] for c, _ic in zip(_data._cluster, i_c)]
            chD0 = [np.array([_chM[_chM[:, 1] == x, 3][0] for x in _chD]) for _chM, _chD in zip(chMap, chDepth)]

            # reorders the channel layer/depth values by decreasing
            for i_tt in range(n_tt):
                chL[i_tt] = np.hstack([chL[ind] for chL, ind in zip(chL0, c_ind[:, i_tt])]).reshape(-1, 1)
                chD[i_tt] = np.hstack([chD[ind] for chD, ind in zip(chD0, c_ind[:, i_tt])]).reshape(-1, 1)
                cW[i_tt] = np.abs(np.hstack([x for x in d_data.c_wght0[:, i_tt]])).reshape(-1, 1)

            # return channel layer/depth values
            return chD, chL, cW

        # retrieves the data classes
        d_data = self.data.discrim.wght
        n_cond, ttype = len(d_data.ttype), d_data.ttype
        col, is_sem = cf.get_plot_col(n_cond), error_type == 'SEM'

        #
        if plot_cond:
            if (len(plot_cond) == 0) or (len(plot_layer) == 0):
                # if not valid, then output an error message to screen
                e_str = 'At least one plot condition/layer type must be selected to run this function.'
                cf.show_error(e_str, 'Invalid Plotting Configuration')

                # exit with a error flag and exit the function
                self.calc_ok = False
                return

        ###################################
        ####    DATA PRE-PROCESSING    ####
        ###################################

        if is_sem:
            # calculates the coefficient weight survival curves
            c_wght = [np.mean(x, axis=0) for x in d_data.c_wght]
            c_wght_sem = [np.std(x, axis=0) / (np.size(x, axis=0) ** 0.5) for x in d_data.c_wght]

            # calculates the bottom-removed coefficient accuracy values
            y_top = [100. * np.mean(x, axis=0) for x in d_data.y_acc_top]
            y_top_sem = [100. * np.std(x, axis=0) / (np.size(x, axis=0) ** 0.5) for x in d_data.y_acc_top]

            # calculates the bottom-removed coefficient accuracy values
            y_bot = [100. * np.mean(x, axis=0) for x in d_data.y_acc_bot]
            y_bot_sem = [100. * np.std(x, axis=0) / (np.size(x, axis=0) ** 0.5) for x in d_data.y_acc_bot]

        else:
            # calculates the coefficient weight survival curves
            c_wght = [np.median(x, axis=0) for x in d_data.c_wght]
            c_wght_lq = [np.percentile(x, 25, axis=0) for x in d_data.c_wght]
            c_wght_uq = [np.percentile(x, 75, axis=0) for x in d_data.c_wght]

            # calculates the bottom-removed coefficient accuracy values
            y_top = [100. * np.mean(x, axis=0) for x in d_data.y_acc_top]
            y_top_lq = [100. * np.percentile(x, 25, axis=0) for x in d_data.y_acc_top]
            y_top_uq = [100. * np.percentile(x, 75, axis=0) for x in d_data.y_acc_top]

            # calculates the bottom-removed coefficient accuracy values
            y_bot = [100. * np.mean(x, axis=0) for x in d_data.y_acc_bot]
            y_bot_lq = [100. * np.percentile(x, 25, axis=0) for x in d_data.y_acc_bot]
            y_bot_uq = [100. * np.percentile(x, 75, axis=0) for x in d_data.y_acc_bot]

        #######################################
        ####    SUBPLOT INITIALISATIONS    ####
        #######################################

        # sets up the axes dimensions
        n_row, n_col = 2, 6
        top, bottom, pH, wspace, hspace = 0.95, 0.06, 0.01, 0.5, 0.25

        # creates the gridspec object
        gs = gridspec.GridSpec(n_row, n_col, width_ratios=[1 / n_col] * n_col, height_ratios=[1 / n_row] * n_row,
                               figure=self.plot_fig.fig, wspace=wspace, hspace=hspace, left=0.05, right=0.98,
                               bottom=bottom, top=top)

        # creates the subplots
        self.plot_fig.ax = np.empty(3 + plot_comp, dtype=object)
        self.plot_fig.ax[1] = self.plot_fig.figure.add_subplot(gs[1, :3])
        self.plot_fig.ax[2] = self.plot_fig.figure.add_subplot(gs[1, 3:])

        # creates the top row subplots (based on the plot type)
        if plot_comp:
            # case is plotting the comparison sub-plot
            self.plot_fig.ax[0] = self.plot_fig.figure.add_subplot(gs[0, :4])
            self.plot_fig.ax[3] = self.plot_fig.figure.add_subplot(gs[0, 4:])
        else:
            # case is not plotting the comparison sub-plot
            self.plot_fig.ax[0] = self.plot_fig.figure.add_subplot(gs[0, :])

        # retrieves the axes handles
        ax = self.plot_fig.ax

        ################################
        ####    SUBPLOT CREATION    ####
        ################################

        # initialisations
        xi, h_plt = 100. * d_data.xi, []

        #
        for i_tt, tt in enumerate(ttype):
            # creates the coefficient weighting survival curve
            h_plt.append(ax[0].plot(xi, c_wght[i_tt], col[i_tt]))
            if is_sem:
                cf.create_error_area_patch(ax[0], xi, c_wght[i_tt], c_wght_sem[i_tt], col[i_tt])
            else:
                cf.create_error_area_patch(ax[0], xi, None, c_wght_lq[i_tt], col[i_tt], y_err2=c_wght_uq[i_tt])

            # creates the worst-ranked removed LDA accuracy
            ax[1].plot(xi, y_top[i_tt], col[i_tt])
            if is_sem:
                cf.create_error_area_patch(ax[1], xi, y_top[i_tt], y_top_sem[i_tt], col[i_tt])
            else:
                cf.create_error_area_patch(ax[1], xi, None, y_top_lq[i_tt], col[i_tt], y_err2=y_top_uq[i_tt])

            # creates the best-ranked removed LDA accuracy
            ax[2].plot(xi, y_bot[i_tt], col[i_tt])
            if is_sem:
                cf.create_error_area_patch(ax[2], xi, y_bot[i_tt], y_bot_sem[i_tt], col[i_tt])
            else:
                cf.create_error_area_patch(ax[2], xi, None, y_bot_lq[i_tt], col[i_tt], y_err2=y_bot_uq[i_tt])


        # sets the legend
        ax[0].legend([x[0] for x in h_plt], ttype, loc=0)

        # sets the axis limits
        xL, yL = [0., 100.], [0., 100.]
        for i_ax, _ax in enumerate(ax[:3]):
            # resets x-axis limits
            _ax.set_xlim(xL)
            _ax.grid(plot_grid)

            if i_ax == 0:
                # sets the x/y-labels
                _ax.set_xlabel('% of Cell Population')
                _ax.set_ylabel('Normalised Coefficient Value (a.u.)')

                # sets the y-axis limits
                _ax.set_ylim([0., 1.])
            else:
                # sets the x/y-labels
                _ax.set_xlabel('% of Cell Removed')
                _ax.set_ylabel('% Accuracy')

                # plots the chance line
                _ax.set_ylim(yL)
                _ax.plot(xL, [50., 50.], c='gray', linewidth=2)

        # plots the coefficient weight threshold
        ax[0].plot(xL, wght_thresh * np.ones(2), 'r--', linewidth=2)

        # sets the subplot titles
        ax[0].set_title('Normalised LDA Coefficient Survival Curve')
        ax[1].set_title('LDA Accuracy (Worst-Ranked Removed)')
        ax[2].set_title('LDA Accuracy (Top-Ranked Removed)')

        ################################
        ####    SUBPLOT CREATION    ####
        ################################

        # if not plotting the comparison, then exit the function
        if not plot_comp:
            return

        # retrieves the channel depths/layer and normalised coefficient weights
        chDepth, chLayer, cWght = get_channel_depths(self.data, d_data)

        # sets the x index
        m_sz, m_type, h_plt = 120., 'o^s*', []
        chLayer_uniq = list(np.unique(chLayer[0]))
        mType = [m_type[chLayer_uniq.index(x)] for x in chLayer[0]]

        # creates the scatterplot
        for i_tt in range(len(chLayer)):
            # only plot the values if the trial type is in the user defined parameters
            if ttype[i_tt] in plot_cond:
                for i_c in range(len(mType)):

                    # only plot the values if the layer type is in the user defined parameters
                    if chLayer[0][i_c] in plot_layer:
                        ax[3].scatter(cWght[i_tt][i_c], chDepth[i_tt][i_c], facecolor='none',
                                      edgecolors=col[i_tt], marker=mType[i_c])

        # retrieves the axis limits
        xL, yL, lg_str = ax[3].get_xlim(), list(ax[3].get_ylim()), dcopy(plot_layer)
        yL[1] *= 1.25

        # plots the dummy layer type markers
        for i_chL, chL in enumerate(chLayer_uniq):
            if chL in plot_layer:
                h_plt.append(ax[3].scatter(-1, -1, facecolor='none', edgecolors='k', marker=m_type[i_chL]))

        # plots the dummy layer type markers
        for i_tt, tt in enumerate(ttype):
            if tt in plot_cond:
                lg_str.append(tt)
                h_plt.append(ax[3].scatter(-1, -1, facecolor=col[i_tt], edgecolors=col[i_tt], marker='o'))

        # creates the legend object
        ax[3].legend(h_plt, lg_str, loc='upper right', ncol=2)

        # sets the axis properties
        ax[3].set_title('Depth vs Coefficient')
        ax[3].set_ylabel('Depth (um)')
        ax[3].set_xlim(xL)
        ax[3].set_ylim(yL)

        # # plots the mean marker points
        # ax[3].scatter(x_nw, y_acc_mn[:, i_cond], marker='.', c=col[i_cond], s=m_sz)
        #
        # # plots the individual points
        # for i_ex in range(n_ex):
        #     ax[3].scatter(x_nw, 100. * d_data.y_acc[i_ex, :, i_cond], facecolors='none',
        #                   edgecolors=col[i_cond], s=n_cell[i_ex])

    ######################################################
    ####    SPEED DISCRIMINATION ANALYSIS FUNCTIONS   ####
    ######################################################

    def plot_speed_accuracy_lda(self, s_factor, marker_type, plot_grid):
        '''

        :param plot_grid:
        :return:
        '''

        # initialisations
        self.create_kinematic_lda_plots(self.data.discrim.spdacc, s_factor, marker_type, plot_grid, plot_chance=False)

    def plot_speed_comp_lda(self, m_size, show_cell_sz, show_fit, sep_resp, plot_type, plot_grid):
        '''

        :param m_size:
        :param show_cell_sz:
        :param show_fit:
        :param sep_resp:
        :param plot_type:
        :param plot_grid:
        :return:
        '''

        # initialisations
        d_data = self.data.discrim.spdc
        n_cond = len(d_data.ttype)
        col = cf.get_plot_col(n_cond)

        # sets the x-tick labels
        spd_x = int(d_data.spd_xi[d_data.i_bin_spd, 1])
        spd_str = ['{0}:{1}'.format(spd_x, int(s)) for s in d_data.spd_xi[:, 1]]
        x = np.arange(np.size(d_data.spd_xi, axis=0))
        xL, yL, h_plt = [x[0], x[-1] + 1.], [0., 100.], []

        #######################################
        ####    SUBPLOT INITIALISATIONS    ####
        #######################################

        # sets up the plot axis
        self.plot_fig.setup_plot_axis()
        ax = self.plot_fig.ax[0]

        ##################################
        ####    DATA VISUALISATION    ####
        ##################################

        if plot_type == 'Inter-Quartile Ranges':

            ###################################
            ####    DATA PRE-PROCESSING    ####
            ###################################

            # sets the plotting data values
            y_acc_md = 100. * np.median(d_data.y_acc, axis=0)
            y_acc_lq = 100. * np.percentile(d_data.y_acc, 25, axis=0)
            y_acc_uq = 100. * np.percentile(d_data.y_acc, 75, axis=0)

            ################################
            ####    SUBPLOT CREATION    ####
            ################################

            # plots the plot/errorbar for each condition
            for i_cond in range(n_cond):
                # sets the plot x locations and error bar values
                x_nw = x + ((i_cond + 1) / (n_cond + 1) if sep_resp else 0.5)

                # creates the plot/errorbar
                h_plt.append(ax.plot(x_nw, y_acc_md[:, i_cond], c=col[i_cond]))
                cf.create_error_area_patch(ax, x_nw, None, y_acc_lq[:, i_cond], col[i_cond], y_err2=y_acc_uq[:, i_cond])

            # creates the legend
            ax.legend([x[0] for x in h_plt], d_data.ttype, loc=4)

        else:

            ###################################
            ####    DATA PRE-PROCESSING    ####
            ###################################

            # array dimensioning
            n_ex = np.size(d_data.y_acc, axis=0)

            # sets cell count for each of the experiments (sets to 1 if not showing cell size)
            n_cell_ex = [sum(x) for x in d_data.i_cell]
            n_cell_mx = np.max(n_cell_ex)
            sz_cell = [m_size * (x / n_cell_mx) if show_cell_sz else m_size for x in n_cell_ex]
            y_acc_mn = 100. * np.mean(d_data.y_acc, axis=0)

            ################################
            ####    SUBPLOT CREATION    ####
            ################################

            # plots the data for all points
            for i_cond in range(n_cond):
                # sets the plot x locations and error bar values
                x_nw = x + ((i_cond + 1) / (n_cond + 1) if sep_resp else 0.5)

                # plots the mean marker points
                h_plt.append(ax.scatter(x_nw, y_acc_mn[:, i_cond], marker='.', c=col[i_cond], s=m_size))

                # plots the individual points
                for i_ex in range(n_ex):
                    ax.scatter(x_nw, 100. * d_data.y_acc[i_ex, :, i_cond], facecolors='none',
                                   edgecolors=col[i_cond], s=sz_cell[i_ex])

                # plots the psychometric fit (if required)
                if show_fit:
                    ax.plot(x_nw, 100. * d_data.y_acc_fit[:, i_cond], c=col[i_cond], linewidth=2)

                # creates the legend
                ax.legend(h_plt, d_data.ttype, loc=4)

        # creates the vertical marker lines
        for xx in np.arange(xL[0] + 1, xL[1]):
            ax.plot(xx * np.ones(2), yL, 'k--')

        # plots the chance line
        ax.plot(xL, 50. * np.ones(2), c='gray', linewidth=2)

        # sets the axis properties
        ax.set_xlim(xL)
        ax.set_ylim(yL)
        ax.set_xticks(x + 0.5)
        ax.set_xticklabels(spd_str)
        ax.set_xlabel('Speed Bin Comparison (deg/s)')
        ax.set_ylabel('Decoding Accuracy (%)')
        ax.grid(plot_grid)

    def plot_pooled_speed_comp_lda(self, m_size, plot_markers, plot_cond, plot_cell, plot_para, plot_grid):
        '''

        :param show_fit:
        :param plot_type:
        :param plot_grid:
        :return:
        '''

        # determines if at least one trial condition has been selected
        if len(plot_cond) == 0:
            # if not, then output an error to screen
            e_str = 'At least one trial condition must be selected to run this function.'
            cf.show_error(e_str, 'Incorrect Parameter Selection')

            # sets the acceptance flag to false and exits the function
            self.calc_ok = False
            return

        # initialisations
        d_data = self.data.discrim.spdcp
        n_cell, y_acc = d_data.n_cell, d_data.y_acc

        # sets up the plot parameters
        is_plot = np.array([x in plot_cond for x in d_data.ttype])
        n_cond, ttype = sum(is_plot), np.array(d_data.ttype)[is_plot]

        # calculates the psychometric curves (if not present)
        if not hasattr(d_data, 'p_acc'):
            cfcn.calc_all_psychometric_curves(d_data, np.diff(d_data.spd_xi[0, :])[0])

        #######################################
        ####    SUBPLOT INITIALISATIONS    ####
        #######################################

        # sets up the plot axis
        self.plot_fig.setup_plot_axis(n_row=1, n_col=1+plot_para)

        ################################
        ####    SUBPLOT CREATION    ####
        ################################

        if plot_para:

            ###########################################
            ####    PSYCHOMETRIC FIT PARAMETERS    ####
            ###########################################

            # initialisations
            ax, n_cond = self.plot_fig.ax, sum(is_plot)
            l_col = cf.get_plot_col(n_cond)
            p_type, p_unit = ['Rate Factor', 'Half-Activation Speed'], ['s/deg', 'deg/s']
            p_acc, p_acc_lo, p_acc_hi = d_data.p_acc, d_data.p_acc_lo, d_data.p_acc_hi

            # sets the x-tick labels and axis limits
            x = n_cell
            xL, h_plt = [x[0], x[-1] + 1.], [[] for _ in range(2)]

            # plots the data for all points
            for i, i_cond in enumerate(np.where(is_plot)[0]):
                # sets the plot x locations and error bar values
                _p_acc, _p_acc_lo, _p_acc_hi = p_acc[i_cond], p_acc_lo[i_cond], p_acc_hi[i_cond]

                for j, i_para in enumerate([2, 3]):
                    # plots the parameter values
                    h_plt[j].append(ax[j].plot(x, _p_acc[:, i_para], 'o-', c=l_col[i]))

                    # # creates the errorbars
                    # y_err = np.vstack((_p_acc[:, i_para] - _p_acc_lo[:, i_para],
                    #                    _p_acc_hi[:, i_para] - _p_acc[:, i_para]))
                    # ax[j].errorbar(x, _p_acc[:, i_para], yerr=y_err, ecolor=l_col[i], fmt='.', capsize=10.0 / n_cond)

                    # sets the axis properties
                    if i == 0:
                        ax[j].set_title('{0} vs Cell Count'.format(p_type[j]))
                        ax[j].set_ylabel('{0} ({1})'.format(p_type[j], p_unit[j]))
                        ax[j].set_xlabel('Cell Count')

            # creates the 2nd legend for the conditions (reshows the first)
            ax[0].legend([x[0] for x in h_plt[0]], plot_cond, loc='upper left')
            ax[1].legend([x[0] for x in h_plt[1]], plot_cond, loc='upper right')

            # sets the x-axis scale to logarithmic
            ax[0].set_xscale('log')
            ax[1].set_xscale('log')

        else:

            ##################################
            ####    PSYCHOMETRIC PLOTS    ####
            ##################################

            # parameters and initialisations
            ax, h_plt_cond = self.plot_fig.ax[0], []
            l_col, l_style = cf.get_plot_col(len(plot_cell)), ['-', '--', '-.', ':']
            y_acc_fit = d_data.y_acc_fit

            # sets the x-tick labels and axis limits
            x = np.arange(np.size(d_data.spd_xi, axis=0))
            spd_x = int(d_data.spd_xi[d_data.i_bin_spd, 1])
            x_str = ['{0}:{1}'.format(spd_x, int(s)) for s in d_data.spd_xi[:, 1]]
            xL, yL, h_plt, k = [x[0], x[-1] + 1.], [0., 100.], [], 0

            # sets cell count for each of the experiments (sets to 1 if not showing cell size)
            y_acc_mn = [100. * np.hstack((np.mean(x[:, :, :-1], axis=0), x[0, :, -1].reshape(-1, 1))) for x in y_acc]

            # plots the data for all points
            for i, i_cond in enumerate(np.where(is_plot)[0]):
                # sets the plot x locations and error bar values
                x_nw, k = x + ((i + 1) / (n_cond + 1)), 0
                _y_acc_fit = y_acc_fit[:, :, i_cond]

                # plots the dummy condition marker lines
                h_plt_cond.append(ax.plot([-1, -2], [0, 0], l_style[i], c='k', linewidth=2))

                #
                h_plt_cell = []
                for j, n_c in enumerate(n_cell):
                    # only include the cell if in the selected checklist
                    if str(n_c) in plot_cell:
                        # plots the mean marker points
                        if plot_markers:
                            h_plt_cell_nw = ax.scatter(x_nw, y_acc_mn[i_cond][:, j], marker='.', c=l_col[k], s=m_size)
                            if i == 0:
                                h_plt_cell.append(h_plt_cell_nw)
                        elif i == 0:
                            h_plt_cell.append(ax.scatter([-1], [-1], marker='.', c=l_col[k], s=m_size))

                        # plots the psychometric fit (if required)
                        ax.plot(x_nw, _y_acc_fit[:, j], l_style[i], c=l_col[k], linewidth=2)

                        # increments the colour counter
                        k += 1

                # creates the legend
                if i == 0:
                    h_lg_cell = ax.legend(h_plt_cell, ['N(cell) = {0}'.format(x) for x in plot_cell], loc=4)

            # creates the 2nd legend for the conditions (reshows the first)
            if len(plot_cond) > 1:
                ax.legend([x[0] for x in h_plt_cond], plot_cond, loc='upper left')
                ax.add_artist(h_lg_cell)

            # creates the vertical marker lines
            for xx in np.arange(xL[0] + 1, xL[1]):
                ax.plot(xx * np.ones(2), yL, 'k--')

            # plots the chance line
            ax.plot(xL, 50. * np.ones(2), c='gray', linewidth=2)

            # sets the axis properties
            ax.set_ylim(yL)
            ax.set_ylabel('Decoding Accuracy (%)')
            ax.set_xlabel('Speed Bin Comparison (deg/s)')
            ax.set_xticks(x + 0.5)
            ax.set_xticklabels(x_str)

        # sets the common axis properties
        for _ax in self.plot_fig.ax:
            _ax.set_xlim(xL)
            _ax.grid(plot_grid)

    def plot_speed_dir_lda(self, use_stagger, s_factor, marker_type, plot_grid, show_stats):
        '''

        :param plot_grid:
        :return:
        '''

        # initialisations
        self.create_kinematic_lda_plots(self.data.discrim.spddir, s_factor, marker_type, plot_grid, use_stagger,
                                        show_stats, plot_chance=True)

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
            self.plot_fig.ax[i_plot].set_title('Cluster #{0}'.format(int(id_clust)))
            self.plot_fig.ax[i_plot].set_ylabel('Frequency (Hz)')
            self.plot_fig.ax[i_plot].set_xlabel('Time Lag (ms)')
            self.plot_fig.ax[i_plot].set_xlim(x_lim)

    def plot_cluster_cross_ccgram(self, exp_name, i_ref, i_comp, plot_all, m_size, plot_type,
                                  window_size, p_lim, f_cutoff):
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
                self.plot_fig.ax[i_plot].bar(xi_hist, height=n_hist, width=1/2)
            else:
                self.plot_fig.ax[i_plot].scatter(xi_hist, n_hist, marker='o', c='b', s=m_size)

            # plots the auto-correlogram and confidence interval limits
            self.plot_fig.ax[i_plot].plot(xi_hist, ciN_lo[ind] / f_scale_ref, 'k--')
            self.plot_fig.ax[i_plot].plot(xi_hist, ciN_hi[ind] / f_scale_ref, 'k--')

            # sets the zero time-lag marker
            self.plot_fig.ax[i_plot].plot(np.zeros(2), y_lim, 'k--')

            # sets the axis properties
            self.plot_fig.ax[i_plot].set_title('Cluster #{0} vs #{1}'.format(int(id_clust1), int(id_clust2)))
            self.plot_fig.ax[i_plot].set_ylabel('Frequency (Hz)')
            self.plot_fig.ax[i_plot].set_xlabel('Time Lag (ms)')
            self.plot_fig.ax[i_plot].set_xlim(x_lim)
            self.plot_fig.ax[i_plot].set_ylim(y_lim)

    #########################################
    ####    COMMON ANALYSIS FUNCTIONS    ####
    #########################################

    def output_spiking_freq_dataframe(self, out_name, plot_all_expt, plot_scope):
        '''

        :return:
        '''

        # retrieves the spike frequency dataframe
        sf_df = self.data.spikedf.sf_df

        # sets the base output file name
        base_name = os.path.join(self.def_data['dir']['dataDir'], out_name)

        if os.path.exists('{0}.csv'.format(base_name)):
            # prompts the user if they want to remove the selected item(s)
            u_choice = QMessageBox.question(self, 'Overwrite File?',
                                            "File already exists. Do you wish to overwrite?",
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if u_choice == QMessageBox.No:
                return

        try:
            # outputs the data to csv/pickle files
            sf_df.to_csv('{0}.csv'.format(base_name))
            sf_df.to_pickle('{0}.pkl'.format(base_name))
        except:
            # if there was an error, then output an error to screen
            e_str = 'Error while outputting data files. Either the files are already open or the filename is invalid.'
            cf.show_error(e_str, 'File Output Error')

    #########################################
    ####    COMMON ANALYSIS FUNCTIONS    ####
    #########################################

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

    def create_raster_hist(self, r_obj, n_bin, show_pref_dir, show_err, plot_grid, rmv_median=False):
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
            A = np.empty(np.size(t_spike, axis=0), dtype=object)
            dxi = xi[1] - xi[0]

            # calculates the histograms for each of the trial over all cells
            for i_hist in range(len(n_hist)):
                ind_trial = np.array([(t_spike[i_hist, i] is not None) for i in range(n_trial)])
                n_hist[i_hist] = np.vstack([np.histogram(x, bins=xi)[0] / dxi for x in t_spike[i_hist, ind_trial]])

                A[i_hist] = np.vstack([np.histogram(x, bins=xi)[0] for x in t_spike[i_hist, ind_trial]])

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

        # memory allocation
        t_sp_t = np.empty((r_obj.n_filt, r_obj.n_phase), dtype=object)
        phase_lbl = ['Baseline', 'Preferred', 'Non-Preferred'] if show_pref_dir else r_obj.phase_lbl

        # sets the firing rate string
        if rmv_median:
            fr_str = 'Relative Mean Firing Rate (Hz)'
        else:
            fr_str = 'Mean Firing Rate (Hz)'

        # creates the plot outlay and titles
        self.init_plot_axes(n_row=2, n_col=r_obj.n_phase, n_plot=2 * r_obj.n_phase)

        # sets up the histogram/rasterplot value for each phase/filter
        for i_filt in range(r_obj.n_filt):
            for i_phase in range(r_obj.n_phase):
                # sets the histogram counts for each of the
                t_sp_t[i_filt, i_phase] = dcopy(r_obj.t_spike[i_filt][:, :, i_phase])

            # if showing the preferred direction, then re-order the arrays accordingly
            if show_pref_dir:
                for i_c in range(np.size(t_sp_t[i_filt, 1], axis=0)):
                    # determines which trials need to be swapped
                    is_swap = [False if ((x is None) or (y is None)) else (len(y)) > len(x)
                               for x, y in zip(t_sp_t[i_filt, 1][i_c, :], t_sp_t[i_filt, 2][i_c, :])]
                    for i_s in np.where(is_swap)[0]:
                        # swaps all the required trials
                        t_sp_t[i_filt, 1][i_c, i_s], t_sp_t[i_filt, 2][i_c, i_s] = \
                            t_sp_t[i_filt, 2][i_c, i_s], t_sp_t[i_filt, 1][i_c, i_s]

        # plots the histogram/rasterplot for each phase
        for i_phase in range(r_obj.n_phase):
            # memory allocation
            c_raster, t_raster, h_plot, y_tick, y_tick_lbl = [], [], [], [], []

            for i_filt in range(r_obj.n_filt):
                # calculates the histogram
                t_phase = r_obj.t_phase[i_filt][0]
                xi = np.linspace(0, t_phase, n_bin + 1)
                xi_hist = (xi[:-1] + xi[1:]) / 2.0

                # calculates the spike time histogram for each of the cells over the current phase
                t_sp = t_sp_t[i_filt, i_phase]
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

                # creates the error area patch (if required)
                ax_plot = self.plot_fig.ax[r_obj.n_phase + i_phase]
                if show_err:
                    # creates the histogram SEM shaded area patch
                    n_hist_sem = np.std(n_hist, axis=0) / (np.shape(n_hist)[0] ** 0.5)
                    cf.create_step_area_patch(ax_plot, xi_hist, n_hist_mn, n_hist_sem, c[i_filt])

                # creates the new plot object
                h_plot.append(ax_plot.step(xi_hist, n_hist_mn, color=c[i_filt], where='mid'))
                if i_filt == 0:
                    self.plot_fig.ax[i_phase].set_xlim(0, t_phase)
                    self.plot_fig.ax[i_phase + r_obj.n_phase].set_xlim(0, t_phase)
                    i_ofs = n_trial
                else:
                    self.plot_fig.ax[i_phase].plot([0, t_phase], (i_ofs - 0.5) * np.ones(2), 'k-', linewidth=1)
                    i_ofs += n_trial

                # appends the tick-marks to the y-tick array
                y_tick.append([i_ofs - n_trial / 2])

            # creates the raster-plot
            self.plot_fig.ax[i_phase].eventplot(positions=cf.flat_list(t_raster), orientation='horizontal',
                                                colors=cf.flat_list(c_raster))

            # sets the axis properties
            self.plot_fig.ax[i_phase].set_ylim(-0.5, i_ofs - 0.5)
            self.plot_fig.ax[i_phase].set_title('{0}'.format(phase_lbl[i_phase]))
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
            self.plot_fig.fig.suptitle('Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0], int(r_obj.ch_id[0][0])),
                                       fontsize=16, fontweight='bold')
            self.plot_fig.fig.tight_layout(rect=[0, 0.01, 1, 0.955])

    def create_spike_freq_plot(self, r_obj, plot_grid, plot_trend, p_value, grp_plot_type, ms_prop, m_size,
                               grp_by_filt, show_stats, ind_type=None):
        '''

        :param r_obj:
        :param plot_grid:
        :param plot_trend:
        :param p_value:
        :param grp_plot_type:
        :param stats_type:
        :param ms_prop:
        :param m_size:
        :param ind_type:
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

            # sets the label index to the plot index (if not provided)
            if lbl_ind is None:
                lbl_ind = plt_ind

            # initialisations
            has_stats = sp_stats is not None
            plt_ind, sep_str = np.array(plt_ind), '--------------------------'
            sp_f = np.array(sp_f)[plt_ind]
            if r_obj.is_ud:
                p_lbl = np.array(['BL', 'Stim'])
            else:
                p_lbl = np.array(['BL', 'CW', 'CCW'])[np.array(lbl_ind)]

            # memory allocation
            n_trial, i_ofs = len(sp_f[0]), 2 + int(not r_obj.is_single_cell) * (
                1 + int(r_obj.plot_all_expt + 3 * has_stats))
            lbl_str = np.empty((n_trial, i_ofs + len(plt_ind)), dtype=object)

            # sets the label string based on the analysis scope
            if r_obj.is_single_cell:
                # case is a single cell is being analysed
                lbl_str[:, 0] = np.array(['Trial #{0}'.format(i + 1) for i in range(n_trial)])
                lbl_str[:, 1] = np.array([sep_str] * n_trial)
            else:
                # case is whole experiment analysis
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

        def setup_plot_axes(plot_fig, n_filt):
            '''

            :param plot_fig:
            :param n_filt:
            :return:
            '''

            # sets up the axes dimensions
            n_r1, n_c1, n_r2, n_c2 = 2, 6, 1, 2
            tbl_hght = 0.05 + 0.02 * n_filt
            top, bottom, wspace, hspace, ax_gap = 0.95, 0.025, 0.7, 0.3 + (0.025 * (n_filt - 1)), 0.025

            # creates the gridspec object
            gs1 = gridspec.GridSpec(n_r1, n_c1, width_ratios=[1 / n_c1] * n_c1, height_ratios=[1 / n_r1] * n_r1,
                                    figure=plot_fig.fig, wspace=wspace, hspace=hspace, left=0.075, right=0.98,
                                    bottom=bottom+(tbl_hght+ax_gap), top=top)
            gs2 = gridspec.GridSpec(n_r2, n_c2, width_ratios=[1 / n_c2] * n_c2, figure=plot_fig.fig, left=0.075,
                                    right=0.98, bottom=bottom, top=bottom+tbl_hght)

            # creates the subplots
            plot_fig.ax = np.empty(7, dtype=object)
            for i_c in range(3):
                plot_fig.ax[i_c] = plot_fig.figure.add_subplot(gs1[0, (2 * i_c):(2 * (i_c + 1))])

            # creates the subplots for each column
            plot_fig.ax[3] = plot_fig.figure.add_subplot(gs1[1, :3])
            plot_fig.ax[4] = plot_fig.figure.add_subplot(gs1[1, 3:])

            # if the count table row, then disable the axis
            for i_c in range(2):
                plot_fig.ax[5 + i_c] = plot_fig.figure.add_subplot(gs2[0, i_c])
                plot_fig.ax[5 + i_c].axis('off')

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        if not r_obj.is_ok:
            self.calc_ok = False
            return

        # initialisations
        n_sub, n_grp = 3, [3, 1]
        main_title = ['Reaction Type', 'Direction Selectivity']
        n_filt = int(r_obj.n_filt / 2) if r_obj.is_ud else r_obj.n_filt
        c, c2 = cf.get_plot_col(n_filt), cf.get_plot_col(max([max(n_grp),n_filt,4]), n_filt)

        # memory allocation
        lg_str_f = r_obj.lg_str
        h_plt, i_grp = [], np.empty(2, dtype=object)
        h, lbl = np.empty(n_sub, dtype=object), np.empty(n_sub, dtype=object)

        # calculates the individual trial/mean spiking rates and sets up the plot/stats arrays
        sp_f0, sp_f = cf.calc_phase_spike_freq(r_obj)
        s_plt, sf_trend, sf_stats, i_grp[0] = cf.setup_spike_freq_plot_arrays(r_obj, sp_f0, sp_f, ind_type, n_sub,
                                                                              plot_trend=plot_trend)

        # determines the indices of the experiments that need to be removed
        i_expt_rmv = cfcn.det_matching_ttype_expt(r_obj, self.data.cluster)
        if len(i_expt_rmv):
            # memory allocation
            is_keep = [np.ones(len(x), dtype=bool) for x in i_grp[0]]

            # ensures that only the cells that have all selected trial conditions are used for the analysis
            for i_filt in range(n_filt):
                for i_rmv in i_expt_rmv:
                    is_keep[i_filt][r_obj.i_expt[i_filt] == i_rmv] = False

            # resets the grouping
            i_grp[0] = [ig[x] for ig, x in zip(i_grp[0], is_keep)]

        if not r_obj.is_single_cell:
            ####################################################
            ####    ANALYSIS PRE-PROCESSING CALCULATIONS    ####
            ####################################################

            # memory allocation
            A = np.empty(n_sub - 1, dtype=object)
            n_type_ex, sf_type_pr = dcopy(A), dcopy(A)
            sf_score = cf.calc_ms_scores(s_plt, sf_stats, p_value=p_value)
            score_min, score_sum = np.min(sf_score[:, :2], axis=1), np.sum(sf_score[:, :2], axis=1)
            x_lbl = ['#{0}'.format(i + 1) for i in np.arange(len(i_grp[0]))]

            # determines the reaction type from the score phase types
            #   0 = None
            #   1 = Inhibited
            #   2 = Excited
            #   3 = Mixed
            sf_type = np.max(sf_score[:, :2], axis=1) + (np.sum(sf_score[:, :2], axis=1) == 3).astype(int)

            # determines the direction selective cells, which must meet the following conditions:
            #  1) one direction only produces a significant result, OR
            #  2) both directions are significant AND the CW/CCW comparison is significant
            one_dir_sig = np.logical_and(score_min == 0, score_sum > 0)     # cells where one direction is significant
            both_dir_sig = np.min(sf_score[:, :2], axis=1) > 0              # cells where both CW/CCW is significant
            comb_dir_sig = sf_score[:, -1] > 0                              # cells where CW/CCW difference is significant

            # determines all motion sensitive cells (sf_type > 0)
            is_mot_sens = sf_type > 0
            is_dir_sel = np.logical_or(one_dir_sig, np.logical_and(both_dir_sig, comb_dir_sig)).astype(int)
            i_grp[1] = [x[is_mot_sens[x]] for x in i_grp[0]]

            # sets the reaction type for each filter type/experiment
            ind = [0, 2, 1, 3]
            sf_type_filt, _ = self.group_metrics_by_expt(r_obj, sf_type, i_grp[0])
            n_type_ex[0] = [np.vstack(
                    [cf.calc_rel_prop(x, 4, return_counts=True, ind=ind) for x in y]) for y in sf_type_filt]

            # sets the direction selectivity for each filter type/experiment
            is_dir_sel_filt, n_filt_ex = self.group_metrics_by_expt(r_obj, is_dir_sel, i_grp[1])
            n_type_ex[1] = [np.vstack([
                    cf.calc_rel_prop(x, 2, return_counts=True) for x in y]) for y in is_dir_sel_filt]

            # if displaying the motion selectivity proportion, then
            if not ms_prop:
                for i_filt in range(r_obj.n_filt):
                    n_type_ex[1][i_filt][:, 0] += n_filt_ex[i_filt] - np.sum(n_type_ex[1][i_filt], axis=1)

            # calculates the proportions and total counts for each reaction type (over each filter type)
            n_type_tot = [np.vstack([np.sum(x, axis=0) for x in xx]) for xx in n_type_ex]
            sf_type_pr = [[100. * np.divide(x, np.sum(x, axis=1).reshape(-1, 1))[:, 1:] for x in y] for y in n_type_ex]
            if not grp_by_filt:
                # reorders by classification type (if required)
                sf_type_pr = [self.reorder_array_dim(x) for x in sf_type_pr]

            # sets the plot values based on the group plot type
            not_stacked = grp_plot_type != 'Stacked Bar'
            if not_stacked:
                # case is the other plot types (removes the
                sf_type_plt = dcopy(sf_type_pr)
            else:
                # case is the stacked barplot
                sf_type_plt = [100. * np.divide(x, np.sum(x, axis=1).reshape(-1, 1)).T for x in n_type_tot]

            # calculates the group posthoc statistics
            stats_ph = self.calc_group_posthoc_stats(sf_type_pr, n_filt, n_grp)

        if show_stats:
            ##########################################################################
            ####    MOTION SENSITIVITY/DIRECTION SELECTIVITY STATISTICS TABLES    ####
            ##########################################################################

            # sets up the plot axes
            col_table = cf.get_plot_col(max([max(n_grp), n_filt]), 1)
            self.setup_posthoc_stats_table_axes(self.plot_fig, main_title, n_filt, n_grp)

            # sets the table header strings
            hdr_class = ['Exc.', 'Inh.', 'Mixed']
            tt_class = ['Excited', 'Inhibited', 'Mixed']

            # sets the base title string
            if (n_filt == 1) and (lg_str_f[0] == 'Black'):
                tt_filt = ['(#1) - All Cells']
                hdr_filt = ['All Cells']
            else:
                tt_filt = ['(#{0}) - {1}'.format(i + 1, '/'.join(lg.split('\n'))) for i, lg in enumerate(lg_str_f)]
                hdr_filt = ['(#{0})'.format(i + 1) for i in range(n_filt)]

            # sets the final header/title strings
            if n_filt == 1:
                hdr_str, t_str = [hdr_class], [tt_filt]
            elif grp_by_filt:
                hdr_str = [hdr_filt, hdr_class, hdr_filt]
                t_str = [tt_class, tt_filt, ['Direction Selective']]
            else:
                hdr_str = [hdr_class, hdr_filt, hdr_filt]
                t_str = [tt_filt, tt_class, ['Direction Selective']]

            # creates the posthoc statistics table
            self.create_posthoc_stats_table(stats_ph, hdr_str, t_str, col_table, p_value)

        else:

            ####################################################
            ####    SPIKING FREQUENCY COMPARISON SUBPLOT    ####
            ####################################################

            # initialisations
            p_alpha = 0.8
            c_scatter = cf.flat_list([[c[i_filt]] * np.size(sp_f[i_filt], axis=0) for i_filt in range(n_filt)])

            # sets up the plot axes
            if r_obj.is_single_cell:
                # case is for a single cell
                self.init_plot_axes(n_row=2, n_col=2)
                self.plot_fig.ax[3].axis('off')
            else:
                # case is whole experiment
                setup_plot_axes(self.plot_fig, n_filt)

            # setups up the scatterplots for each subplot phase
            ax = self.plot_fig.ax
            for i_sub in range(n_sub):
                # loop initialisations
                sp, i1, i2 = s_plt[i_sub], 0, 1

                # sets the scatterplot alpha values
                if not r_obj.is_single_cell:
                    c_scatter = [to_rgba_array(x, 1 - p_alpha * (y > p_value)) for x, y in
                                 zip(c_scatter, sf_stats[i_sub])]
                    c_scatter = np.squeeze(c_scatter)

                # creates the plot and row label strings
                i1, i2 = 1 * (i_sub > 1), 1 + (i_sub > 0)
                h[i_sub] = ax[i_sub].scatter(sp[0], sp[1], marker='o', c=c_scatter, s=m_size)
                lbl[i_sub] = set_stim_phase_str(r_obj, sp, sf_stats[i_sub], [0, 1], [i1, i2], p_value=p_value)

                # creates the legend markers (first subplot only)
                if i_sub == 0:
                    h_plt = [ax[0].scatter(-1, -1, marker='o', c=cc) for cc in c]

                # sets the title string (non-3d plot only)
                t_str = '{0} vs {1}'.format(r_obj.phase_lbl[i2], r_obj.phase_lbl[i1])
                ax[i_sub].set_title(t_str)

                # sets the x/y-label and the grid
                ax[i_sub].set_xlabel('{0} Frequency (Hz)'.format(r_obj.phase_lbl[i1]))
                ax[i_sub].set_ylabel('{0} Frequency (Hz)'.format(r_obj.phase_lbl[i2]))
                ax[i_sub].grid(plot_grid)

            # creates the legend (if more than one filter type)
            if n_filt > 1 or (not r_obj.is_single_cell):
                # sets the legend strings
                if r_obj.is_ud and r_obj.n_filt == 1:
                    lg_str = ['All Cells']
                else:
                    lg_str = ['(#{0}) - {1}'.format(i + 1, x) for i, x in enumerate(r_obj.lg_str)]

                # creates the legend
                ax[0].legend(h_plt, lg_str, loc=0)

            # resets the axis limits
            axL = -1e6
            for i_sub in range(n_sub):
                xLnw, yLnw = ax[i_sub].get_xlim(), ax[i_sub].get_ylim()
                axL = max(xLnw[1], yLnw[1], axL)

                # adds the trend-line (if selected)
                if plot_trend:
                    x = np.array([0, 100 * axL])
                    for i_filt in range(n_filt):
                        y = sf_trend[i_sub][i_filt, 0] * x
                        ax[i_sub].plot(x, y, '--', c=c[i_filt])

            # updates the the x/y axis limits
            for i_sub in range(n_sub):
                ax[i_sub].set_xlim(0, axL)
                ax[i_sub].set_ylim(0, axL)
                ax[i_sub].plot([0, axL], [0, axL], 'k--')

            if r_obj.is_single_cell:
                # resets the axis properties
                self.plot_fig.fig.set_tight_layout(False)
                self.plot_fig.fig.tight_layout(rect=[0.0, 0.01, 1, 0.955])
                self.plot_fig.fig.suptitle(
                    'Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0], int(r_obj.ch_id[0][0])),
                    fontsize=16, fontweight='bold')

                # sets up the statistics table values
                col_hdr = ['CW vs BL', 'CCW vs BL', 'CCW vs CW']
                t_data = np.vstack([['{:5.3f}{}'.format(y, cf.sig_str_fcn(y, p_value)) for y in x] for x in sf_stats]).T

                # calculates the table dimensions
                cf.add_plot_table(self.plot_fig, len(ax)-1, table_font, t_data, r_obj.lg_str,
                                  col_hdr, c, cf.get_plot_col(3, len(c)), None, n_row=2, pfig_sz=0.955)

                # exits the function
                return

            #################################################################
            ####    MOTION SENSITIVITY/DIRECTION SELECTIVITY SUBPLOTS    ####
            #################################################################

            # sets the classification group strings
            class_str0 = [['None', 'Excited', 'Inhibited', 'Mixed'], ['Direction Insensitive', 'Direction Sensitive']]
            class_str = [x[not_stacked:] for x in np.array(class_str0)]

            # creates filter group strings
            if (r_obj.is_ud and r_obj.n_filt == 2) or (r_obj.n_filt == 1):
                # no filter applied, so all cells are used
                filt_str = ['#1 - All Cells']
            else:
                # otherwise, set the filter grouping strings
                filt_str = ['#{0}'.format(i + 1) for i in range(len(lg_str_f))]

            # creates the graphs for the motion sensitive/direction selectivity plots
            for i in range(2):
                # creates the graph
                h_plt = cf.create_general_group_plot(ax[i + n_sub], sf_type_plt[i], grp_plot_type, c2)
                cf.reset_axes_dim(ax[i + n_sub], 'bottom', 0.075, True)

                # sets the legend/x-ticklabels based on the grouping type
                lg_str = class_str[i] if grp_by_filt else filt_str
                x_ticklbl = filt_str if grp_by_filt else class_str[i]

                if len(h_plt):
                    y_hg = 1.1 + 0.01 * n_filt
                    ax[i + n_sub].legend([x[0] for x in h_plt], lg_str, ncol=len(lg_str), loc='upper center',
                                                                        columnspacing=0.125, bbox_to_anchor=(0.5, y_hg))
                else:
                    ax[i + n_sub].set_title('Direction Sensitivity Proportion', fontsize=16, fontweight='bold')

                # only set the y-axis label for the first subplot
                if i == 0:
                    ax[n_sub].set_ylabel('Population %')

                # sets the y-axis limits based on type
                if 'Bar' in grp_plot_type:
                    ax[i + n_sub].set_ylim([0, 100])
                else:
                    ax[i + n_sub].set_ylim([-2.5, 102.5])

                # updates the axis properties
                ax[i + n_sub].grid(plot_grid)
                ax[i + n_sub].set_xticklabels(x_ticklbl)

            ###################################
            ####    METRIC COUNT TABLES    ####
            ###################################

            # sets the
            n_type_N0 = [np.hstack((x, np.sum(x, axis=1).reshape(-1, 1))) for x in n_type_tot]
            n_type_N = [np.vstack((x, np.sum(x, axis=0))) for x in n_type_N0]
            col_hdr = [x + ['Total Cells'] for x in class_str0]

            # creates the title text object
            t_font = cf.get_table_font_size(3)
            if (n_filt == 1) and (lg_str_f[0] == 'Black'):
                row_hdr = ['All Cells', 'Total']
            else:
                row_hdr = ['#{0}'.format(x + 1) for x in range(n_filt)] + ['Total']

            # creates the graphs for the motion sensitive/direction selectivity plots
            for i in range(len(class_str0)):
                # creates the new table
                j, nT = i + 5, len(col_hdr[i])
                cT = cf.get_plot_col(max(n_filt, nT), max(n_grp))
                cf.add_plot_table(self.plot_fig, ax[j], t_font, n_type_N[i].astype(int), row_hdr, col_hdr[i],
                                  cT[:n_filt] + [(0.75, 0.75, 0.75)], cT[:nT] + [(0.75, 0.75, 0.75)],
                                  'fixed', n_col=len(class_str0[i]))

        # retrieves the function data values
        self.func_data = self.set_output_data(locals())

    def group_metrics_by_expt(self, r_obj, y_metric, i_grp):
        '''

        :return:
        '''

        # retrieves the indices of the common cells (across all experiments/filter types)
        i_expt_all = np.array(cf.flat_list([list(x) for x in r_obj.i_expt]))
        i_filt_ex = [[np.where(x == i)[0] for i in np.unique(x)] for x in [i_expt_all[x] for x in i_grp]]
        n_filt_ex = [[len(xx) for xx in x] for x in i_filt_ex]

        # groups the metric values by the
        if np.ndim(y_metric) == 1:
            return [[x[yy] for yy in y] for x, y in zip([y_metric[x] for x in i_grp], i_filt_ex)], n_filt_ex
        else:
            return [[x[yy, :] for yy in y] for x, y in zip([y_metric[x, :] for x in i_grp], i_filt_ex)], n_filt_ex

    def create_spike_heatmap(self, r_obj, dt, norm_type, mean_type):
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
            wspace = 2 / 10 if (n_col_hm == 3) else 8 / 50
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
            probe_depth, cell_depth = np.array([c['expInfo']['probe_depth'] for c in cluster]), []
            if np.any(probe_depth == None):
                return None

            # retrieves the depths from each experiment
            for c, pd in zip(cluster, probe_depth):
                chMap = c['expInfo']['channel_map']
                cell_depth.append([pd - chMap[chMap[:, 1] == x, 3][0] for x in c['chDepth']])

            # returns the
            return np.array(cell_depth)

        def setup_spiking_heatmap(t_spike, xi_h, is_single_cell, isort_d, mean_type):
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
                        h_gram = np.histogram(t_sp_flat, bins=xi_h[0, :])[0]
                        dt = np.diff(xi_h[0, :])
                    else:
                        # t_sp_flat = cf.flat_list(
                        #     [list(x) if x is not None else [] for x in t_spike[i_trial, :, i_phase]])

                        is_ok = np.array([x is not None for x in t_spike[i_trial, :, i_phase]])
                        h_gram_tot = np.vstack([np.histogram(t_spike[i_trial, i, i_phase], bins=xi_h[i_trial, :])[0]
                                                for i in np.where(is_ok)[0]])

                        dt = np.diff(xi_h[i_trial, :])
                        if mean_type == 'Mean':
                            h_gram = np.mean(h_gram_tot, axis=0)
                        else:
                            h_gram = np.median(h_gram_tot, axis=0)

                    # normalises the histograms by the duration of the bins
                    I_hm[i_trial, :, i_phase] = h_gram / dt

            # sorts the clusters by depth (whole experiments only)
            if not is_single_cell:
                I_hm = I_hm[isort_d, :, :]

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
            ch_depth = get_channel_depths(data)
            if ch_depth is None:
                # if not all channel depths are set, then output an error to screen
                e_str = 'At least one experimental file does not have the probe depth set. \n' \
                        'Pleast make sure all probe depths are set before running this function.'
                cf.show_error(e_str, 'Missing Probe Depths')

                # exits the function flagging the error
                self.calc_ok = False
                return

            D = [[ch_depth[x][y] for x, y in zip(i_ex, cf.flat_list([list(x) for x in cl_id]))] for i_ex, cl_id in
                 zip(r_obj.i_expt, r_obj.clust_ind)]
        else:
            # case is single cell (no need to sort by depth)
            D = [None] * r_obj.n_filt

        # creates the plot outlay and titles
        init_heatmap_plot_axes(r_obj)
        hm_cmap = ListedColormap(sns.diverging_palette(223, 17, sep=33, l=47, n=11))

        # creates the heatmaps for each filter/phase
        I_hm = np.empty(r_obj.n_filt, dtype=object)
        for i_filt in range(r_obj.n_filt):
            # determines the stimuli times (converts from indices to actual times)
            _, ind = np.unique(r_obj.i_expt[i_filt], return_inverse=True)
            t_stim = [r_obj.t_phase[i_filt][i] for i in ind]

            # sets up the histogram bins
            xi_h0 = [setup_heatmap_bins(x, dt) for x in t_stim]
            for i in range(len(xi_h0)):
                xi_h0[i][-1] = t_stim[i]

            # calculates the spiking frequency histograms
            xi_h, sort_d = np.vstack(xi_h0), np.argsort(-np.array(D[i_filt]))
            I_hm[i_filt] = setup_spiking_heatmap(r_obj.t_spike[i_filt], xi_h, r_obj.is_single_cell, sort_d, mean_type)

            if not r_obj.is_single_cell:
                D[i_filt] = list(np.array(D[i_filt])[sort_d])

        # sorts the clusters by depth
        if (not r_obj.is_single_cell) and (norm_type != 'None'):
            # normalises each trial across each phase/filter type
            # I_hm_norm = np.max(np.vstack([np.max(np.max(x, axis=1), axis=1) for x in I_hm]), axis=0)
            for i_filt in range(len(I_hm)):
                # calculates the min/max values over all trials
                I_hm_f = I_hm[i_filt].reshape(np.size(I_hm[i_filt], axis=0), -1)
                if norm_type == 'Min/Max Normalisation':
                    I_hm_max, I_hm_min = np.max(I_hm_f, axis=1), np.min(I_hm_f, axis=1)
                else:
                    I_hm_med = np.median(I_hm[i_filt][:, :, 0], axis=1)
                    # I_hm_med = np.mean(I_hm[i_filt][:, :, 0], axis=1)

                for i_trial in range(np.size(I_hm[i_filt], axis=0)):
                    # if the denominator is >0, then normalise the heatmap values for the current trial
                    if norm_type == 'Min/Max Normalisation':
                        # calculates the denominator of the normalisation
                        d_hm = I_hm_max[i_trial] - I_hm_min[i_trial]
                        if d_hm > 0:
                            I_hm[i_filt][i_trial, :, :] = (I_hm[i_filt][i_trial, :, :] - I_hm_min[i_trial]) / d_hm
                    else:
                        I_hm[i_filt][i_trial, :, :] -= I_hm_med[i_trial]

                        I_norm = np.max([np.max(I_hm[i_filt][i_trial, :, :]),
                                         np.abs(np.min(I_hm[i_filt][i_trial, :, :]))])
                        if I_norm > 0:
                            I_hm[i_filt][i_trial, :, :] /= I_norm

        # creates the heatmaps for each filter/phase
        for i_filt in range(r_obj.n_filt):
            im = np.empty(r_obj.n_phase, dtype=object)
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
                im[i_phase] = self.plot_fig.ax[i_plot].imshow(I_hm[i_filt][:, :, i_phase], aspect='auto',
                                                              origin='lower', cmap=hm_cmap)
                # im[i_phase] = sns.heatmap(I_hm[i_filt][:, :, i_phase], cmap=pp, ax=self.plot_fig.ax[i_plot])

                # sets the subplot title (first row only)
                if i_filt == 0:
                    self.plot_fig.ax[i_plot].set_title(r_obj.phase_lbl[i_phase])

                # sets the x-axis label (last row only)
                if (i_filt + 1) == r_obj.n_filt:
                    self.plot_fig.ax[i_plot].set_xlabel('Time (s)')

                #
                if ((i_phase + 1) == r_obj.n_phase) and (not r_obj.is_single_cell):
                    D_hm = np.array(D[i_filt]).reshape(-1, 1)
                    im_d = self.plot_fig.ax[i_plot + 1].imshow(D_hm, aspect='auto', cmap='Reds', origin='lower')
                    self.plot_fig.ax[i_plot + 1].grid(False)
                    self.plot_fig.ax[i_plot + 1].get_xaxis().set_visible(False)
                    self.plot_fig.ax[i_plot + 1].get_yaxis().set_visible(False)
                    cbar = self.plot_fig.figure.colorbar(im_d, cax=self.plot_fig.ax[-2], orientation='horizontal')
                    self.plot_fig.ax[-2].set_xlabel('Depth ({0}m)'.format(cf._mu))

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

            c_lim_mx = np.max([_im.get_clim()[1] for _im in im])
            for _im in im:
                if norm_type == 'Baseline Median Subtraction':
                    _im.set_clim(-1, 1)
                    pass
                else:
                    _im.set_clim(0, c_lim_mx)

            #
            if i_filt == 0:
                self.plot_fig.figure.colorbar(im[0], cax=self.plot_fig.ax[-1], orientation='horizontal')

                if r_obj.is_single_cell:
                    x_lbl = 'Firing Rate (Hz)'
                else:
                    norm_type_str = ['Baseline Median Subtraction', 'Min/Max Normalisation', 'None']
                    x_lbl_list = ['Normalised Relative', 'Normalised', '']
                    x_lbl = '{0} Firing Rate (Hz)'.format(x_lbl_list[norm_type_str.index(norm_type)])

                self.plot_fig.ax[-1].set_xlabel(x_lbl)

        # resets the figure layout (single cell only)
        if r_obj.is_single_cell:
            self.plot_fig.fig.set_tight_layout(False)
            self.plot_fig.fig.suptitle('Cluster #{0} (Channel #{1})'.format(r_obj.cl_id[0][0], int(r_obj.ch_id[0][0])),
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

    def create_dir_roc_curve_plot(self, rot_filt, plot_exp_name, plot_all_expt, use_avg, connect_lines, violin_bw,
                                  m_size, plot_grp_type, cell_grp_type, auc_plot_type, plot_grid, plot_scope, is_comb):

        # initialises the rotation filter (if not set)
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # if there was an error setting up the rotation calculation object, then exit the function with an error
        r_obj = RotationFilteredData(self.data, rot_filt, None, plot_exp_name, plot_all_expt, 'Whole Experiment', False)
        if not r_obj.is_ok:
            self.calc_ok = False
            return
        elif connect_lines:
            if len(np.unique([x['t_type'][0] for x in r_obj.rot_filt_tot])) != 2:
                e_str = 'The connect AUC point option requires 2 trial condition filter options. Either de-select ' \
                        'the option or alter the filter options.'
                cf.show_error(e_str, 'AUC Connection Line Error')
                self.calc_ok = False
                return

        # sets the cell group string lists based on the grouping type
        r_data, is_cong = self.data.rotation, False
        if plot_grp_type == 'Motion/Direction Selectivity':
            # case is direction selectivity
            g_type = ['MS/DS', 'MS/Not DS', 'Not MS', 'All Cells']
            st_type = ['Wilcoxon Paired Test', 'Delong', 'Bootstrapping'].index(r_data.phase_grp_stats_type)
            g_type_data = dcopy(r_data.phase_gtype)[:, st_type]

        elif plot_grp_type == 'Rotation/Visual DS':
            # case is preferred direction
            g_type = ['None', 'Rotation', 'Visual', 'Both', 'All Cells']
            g_type_data = dcopy(r_data.ds_gtype)

        else:
            # case is congruency
            g_type, is_cong = ['Congruent', 'Incongruent', 'All Cells'], True
            g_type_data = dcopy(r_data.pd_type)

        # initialisations
        n_filt, ig_type = r_obj.n_filt, g_type.index(cell_grp_type)
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

        # sets the indices of the points to be shown in the bubble plot
        if (ig_type + 1) == len(g_type):
            # case is plotting all the indices
            if is_cong:
                is_sig = g_type_data >= 0
            else:
                is_sig = np.ones(np.size(g_type_data, axis=0), dtype=bool)
        else:
            # case is plotting for a specific case
            is_sig = g_type_data == ig_type

        # sets the significance criteria for each filter type
        i_cell_sig = [is_sig[z] for z in i_cell_b]

        # if there are no valid selections, then exit
        if np.all(np.array([sum(x) for x in i_cell_sig]) == 0):
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

            if is_comb:
                # roc_xy[i_filt] = r_data.part_roc_xy[tt_filt][i_cell_b[i_filt]][i_cell_sig[i_filt]]
                # roc_auc[i_filt] = r_data.part_roc_auc[tt_filt][i_cell_b[i_filt]][i_cell_sig[i_filt], 2]
                roc_xy[i_filt] = r_data.part_roc_xy[tt_filt][i_cell_b[i_filt]][i_cell_sig[i_filt]]
                roc_auc[i_filt] = r_data.part_roc_auc[tt_filt][i_cell_b[i_filt]][i_cell_sig[i_filt], 2]
            else:
                roc_xy[i_filt] = r_data.cond_roc_xy[tt_filt][i_cell_sig[i_filt]]
                roc_auc[i_filt] = r_data.cond_roc_auc[tt_filt][i_cell_sig[i_filt], 2]

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
        self.create_multi_auc_plot(self.plot_fig.ax[1], roc_auc, plot_grid, connect_lines, violin_bw, m_size,
                                   lg_str0, auc_plot_type)

        # sets the axis titles
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

                        p_value_str = '{:5.3f}{}'.format(p_value, cf.sig_str_fcn(p_value, 0.05))
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

        n_grp = len(g_type) - 1
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

        # calculates the table dimensions
        cf.add_plot_table(self.plot_fig, 1, table_font, n_sig_grp, row_hdr,
                          col_hdr, row_cols, [cc[0]] * len(col_hdr), t_loc='bottom')

    def create_kinematic_lda_plots(self, d_data, s_factor, marker_type, plot_grid, use_stagger=False,
                                   show_stats=False, plot_chance=False):
        '''

        :param d_data:
        :param plot_err:
        :param plot_grid:
        :return:
        '''

        # initialisations
        n_cond, n_ex = len(d_data.ttype), np.size(d_data.y_acc, axis=0)
        col = cf.get_plot_col(n_cond)

        # sets the x-tick labels
        spd_str = ['{0}'.format(int(s)) for s in d_data.spd_xi[:, 1]]
        x = np.arange(np.size(d_data.spd_xi, axis=0))
        xL, yL, h_plt = [x[0], x[-1] + 1.], [0., 100.], []

        #######################################
        ####    SUBPLOT INITIALISATIONS    ####
        #######################################

        # sets up the plot axis
        self.plot_fig.setup_plot_axis()
        ax = self.plot_fig.ax[0]

        ###################################
        ####    DATA PRE-PROCESSING    ####
        ###################################

        # sets the plotting data values
        y_acc_md = 100. * np.median(d_data.y_acc, axis=0)
        y_acc_lq = 100. * np.percentile(d_data.y_acc, 25., axis=0)
        y_acc_uq = 100. * np.percentile(d_data.y_acc, 75., axis=0)

        # sets the number of cells/expt
        n_cell = [sum(x) for x in d_data.i_cell]


        ##################################
        ####    DATA VISUALISATION    ####
        ##################################

        # plots the data for all points
        for i_cond in range(n_cond):
            # sets the plot x locations and error bar values
            x_nw = x + ((i_cond + 1) / (n_cond + 1) if use_stagger else 0.5)

            # plots the mean marker points
            h_plt.append(ax.plot(x_nw, y_acc_md[:, i_cond], c=col[i_cond]))

            # plots the individual points
            if marker_type == 'Individual Experiment Markers':
                # case is the individual experiment
                for i_ex in range(n_ex):
                    ax.scatter(x_nw, 100. * d_data.y_acc[i_ex, :, i_cond], facecolors='none',
                               edgecolors=col[i_cond], s=s_factor * n_cell[i_ex])

            elif marker_type == 'Experiment IQR Area':
                # case is the experiment SEM Area
                cf.create_error_area_patch(ax, x_nw, None, y_acc_lq[:, i_cond], col[i_cond], y_err2=y_acc_uq[:, i_cond])

        # creates the legend
        ax.legend([x[0] for x in h_plt], d_data.ttype, loc=0)

        # creates the vertical marker lines
        for xx in np.arange(xL[0] + 1, xL[1]):
            ax.plot(xx * np.ones(2), yL, 'k--')

        # plots the chance line
        if plot_chance:
            ax.plot(xL, 50. * np.ones(2), c='gray', linewidth=2)

        # sets the axis properties
        ax.set_xlim(xL)
        ax.set_ylim(yL)
        ax.set_xticks(x + 0.5)
        ax.set_xticklabels(spd_str)
        ax.set_xlabel('Speed Bin (deg/s)')
        ax.set_ylabel('Decoding Accuracy (%)')
        ax.grid(plot_grid)

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
        if stats_type == 'Motion Sensitivity':
            n_DS = np.vstack([len(y) * x  / 100 for x, y in zip(sf_type_pr[0].T, i_grp[0])]).astype(int)
        else:
            n_DS = sf_type_pr[1]

        # returns the N-value array
        return n_DS

    def create_spike_freq_stats_table(self, ax, n_DS, n_filt, stats_type,
                                      c=None, c2=None, n_row=2, n_col=3, n_PD=None, show_prop=False):
        '''

        :param n_DS:
        :return:
        '''

        # enforces tight layout format
        self.plot_fig.fig.tight_layout()

        # initialisations
        n_type = int
        if stats_type == 'Motion Sensitivity':
            n_DS_Full = cf.add_rowcol_sum(n_DS)
            col_hdr = ['None', 'Inh.', 'Exc.', 'Mixed', 'Total']
        else:
            n_DS_Full = cf.add_rowcol_sum(n_DS.T)
            col_hdr = ['Insensitive', 'Sensitive', 'Total']

        if show_prop:
            col_hdr, n_DS_Full, n_type = col_hdr[:-1], n_DS_Full[:-1, :][:, :-1], str
            n_DS_Full /= repmat(np.sum(n_DS_Full, axis=1), 1, len(col_hdr))
            n_DS_Full = np.round(100. * n_DS_Full, 1)

        # creates the colours (if not provided)
        # if c is None:
        d_ofs = int(not show_prop)
        n_row_d, n_col_d = np.shape(n_DS_Full)
        c_row, c_col = cf.get_plot_col(n_row_d - d_ofs), cf.get_plot_col(n_col_d - d_ofs, n_row_d - d_ofs)
        tot_col = [] if show_prop else [(0.75, 0.75, 0.75)]

        # creates the title text object
        t_str = '{0} N-Values'.format(stats_type)
        h_title = ax.text(0.5, 1, t_str, fontsize=15, horizontalalignment='center')
        row_hdr = ['#{0}'.format(x + 1) for x in range(n_filt)] + ([] if show_prop else ['Total'])

        # sets up the n-value table
        ax_pos_tbb = dcopy(ax.get_tightbbox(self.plot_fig.get_renderer()).bounds)
        t_props = cf.add_plot_table(self.plot_fig, ax, table_font, n_DS_Full.astype(n_type), row_hdr,
                                      col_hdr, c_row + tot_col, c_col  + tot_col, None,
                                      n_row=n_row, n_col=n_col, h_title=h_title, ax_pos_tbb=ax_pos_tbb)

        # calculates the height between the title and the top of the table
        dh_title = h_title.get_position()[1] - (t_props[0]._bbox[1] + t_props[0]._bbox[3])
        c_hght = t_props[0]._bbox[3] / (np.size(n_DS_Full, axis=0) + 1)

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
                            sig_str = '{:5.3f}{}'.format(chi_pval, cf.sig_str_fcn(chi_pval, p_value))

                        chi_stats[i_row, i_col] = chi_stats[i_col, i_row] = sig_str

            # creates the title text object
            t_str_2 = '{0} Statistics'.format(stats_type)
            h_title_2 = ax.text(0.5, 1, t_str_2, fontsize=15, horizontalalignment='center')

            # sets up the table other object properties
            t_ofs = (1 - t_props[1]) * t_props[2]
            row_hdr_1 = col_hdr_1 = ['#{0}'.format(str(x + 1)) for x in range(n_filt)]

            # sets up the n-value table
            c_stats = cf.get_plot_col(len(row_hdr_1))
            t_props_2 = cf.add_plot_table(self.plot_fig, ax, table_font, chi_stats, row_hdr_1, col_hdr_1,
                                          c_stats, c_stats, None, n_row=n_row, n_col=n_col,
                                          ax_pos_tbb=ax_pos_tbb)

            # resets the bottom location of the upper table
            t_props_2[0]._bbox[1] = t_props[0]._bbox[1] - (t_props[0]._bbox[3] + c_hght)

            # resets the title position
            t_pos = list(h_title_2.get_position())
            t_pos[1] = t_props_2[0]._bbox[1] + t_props_2[0]._bbox[3] + dh_title
            h_title_2.set_position(tuple(t_pos))

    ############################################
    ####    POSTHOC STATISTICS FUNCTIONS    ####
    ############################################

    def setup_posthoc_stats_table_axes(self, plot_fig, main_title, n_filt, n_grp):
        '''

        :param plot_fig:
        :return:
        '''

        def create_title_axes(gs, i_cs, i_cm, main_title, sub_title, has_main):
            '''

            :param gs:
            :return:
            '''

            # creates the main title axis
            if has_main:
                n_col = int(('Within' in sub_title) and (n_filt > 1)) + 1
                ax_main = plot_fig.figure.add_subplot(gs[0, i_cm:(i_cm + n_col)])
                ax_main.set_title(main_title, fontsize=18, fontweight='bold')
                ax_main.axis('off')

            # creates the sub-title axis
            ax_sub = plot_fig.figure.add_subplot(gs[1, i_cs])
            ax_sub.set_title(sub_title, fontsize=15)
            ax_sub.axis('off')

        # ensures the grouping count is a list
        if not isinstance(n_grp, list):
            n_grp = [n_grp]

        # ensures the main title is a list
        if not isinstance(main_title, list):
            main_title = [main_title]

        # axes dimensions
        hght_title = 0.01
        top, bottom, left, right, pH, wspace, hspace = 0.95, 0.02, 0.075, 0.98, 0.01, 0.25, 0.15

        # other initialisations
        n_grp_max = max(n_grp)
        if (n_grp_max == 1) and (len(n_grp) == 1):
            n_r = n_c = 1
        else:
            n_r = 1 if n_filt == 1 else np.max([n_filt, n_grp_max])
            n_c = np.sum(np.array(n_grp) > 1) + len(n_grp) * (n_filt > 1)

        # memory allocation
        plot_fig.ax = np.empty(n_r * n_c, dtype=object)

        # calculates the axes height ratios (based on the row count and whether there are main/sub titles)
        t_str = [hght_title] * 2
        height_ratios = t_str + [(1 - sum(t_str)) / n_r] * n_r

        # creates the gridspec object
        gs = gridspec.GridSpec(n_r + 2, n_c, width_ratios=[1 / n_c] * n_c, height_ratios=height_ratios,
                               figure=plot_fig.fig, wspace=wspace, hspace=hspace, left=left, right=right,
                               bottom=bottom, top=top)

        # creates the main title axes (if more than one comparison type)
        i_cs, i_cm, i_ofs = 0, 0, 2
        # ax_main, ax_sub = np.empty(n_c, dtype=object), np.empty(n_c, dtype=object)
        for m_t, n_g in zip(main_title, n_grp):
            # creates the within group stats axis
            if n_g > 1:
                create_title_axes(gs, i_cs, i_cm, m_t, 'Within Group Stats', True)
                i_cs += 1
                i_cm += 1 + int(n_filt > 1)

            # creates the between group stats axis
            if n_filt > 1:
                create_title_axes(gs, i_cs, i_cm, m_t, 'Between Group Stats', n_g == 1)
                i_cs += 1
                i_cm += int(n_g == 1)

        # creates the plot axes
        for i_r in range(n_r):
            for i_c in range(n_c):
                # creates the subplot axes
                i_plot = i_c * n_r + i_r
                plot_fig.ax[i_plot] = plot_fig.figure.add_subplot(gs[i_r + i_ofs, i_c])

                # disables the plot axes (enables when being used)
                plot_fig.ax[i_plot].axis('off')

    def calc_group_posthoc_stats(self, y, n_filt, n_grp, c_ofs=0):
        '''

        :param sf_type_pr:
        :param n_filt:
        :param n_grp:
        :return:
        '''

        # ensures the group counts is a list object
        if not isinstance(n_grp, list):
            n_grp = [n_grp]

        # calculates the within/between filter type statistics
        i_c, n_grp_max = 0, max(n_grp)
        if (n_grp_max == 1) and (len(n_grp) == 1):
            n_r = n_c = 1
        else:
            n_r = 1 if n_filt == 1 else np.max([n_filt, n_grp_max])
            n_c = np.sum(np.array(n_grp) > 1) + len(n_grp) * (n_filt > 1)

        #
        stats_ph = np.empty((n_r, n_c, 2), dtype=object)
        for i in range(len(n_grp)):
            # calculates the within/between group statistics
            p_stats = cfcn.calc_posthoc_stats(y[i], c_ofs=c_ofs)

            # places the values into the overall stats array
            for j in range(len(p_stats)):
                # if there are no stats, then continue
                if p_stats[j] is None:
                    continue

                # sets the stats based on the filter count
                n_row = np.shape(p_stats[j])[0]
                stats_ph[:n_row, i_c, 0] = p_stats[j][:, 0]
                stats_ph[:n_row, i_c, 1] = p_stats[j][:, 1]

                # increments the counter
                i_c += 1

        # returns the stats array
        return stats_ph

    def create_posthoc_stats_table(self, y, hdr_str, t_str0, col, p_value=0.05):

        # initialisations
        ax = self.plot_fig.ax
        if np.ndim(y) == 1:
            n_r = n_c = 1
        else:
            n_r, n_c = np.shape(y)[0], np.shape(y)[1]

        f_size = [14, 12, 10, 8][int(n_r >= 2) + int(n_r >= 4) + int(n_r >= 6)]
        t_font = cf.get_table_font_size(n_r)

        for i_r in range(n_r):
            for i_c in range(n_c):
                # if there is no stats values, then continue
                if y[i_r, i_c, 0] is None:
                    continue

                # creates the subplot axes
                i_plot = i_c * n_r + i_r
                stats_nw = y[i_r, i_c, 1]
                nr_nw = np.shape(stats_nw)[0]

                # creates the table
                suf_str = cf.sig_str_fcn(y[i_r, i_c, 0], p_value)
                t_str = '{}\n(KW Stats = {:.3e}{})'.format(t_str0[i_c][i_r], y[i_r, i_c, 0], suf_str)
                h_title = ax[i_plot].text(0.5, 1, t_str, fontsize=f_size, horizontalalignment='center')
                cf.add_plot_table(self.plot_fig, ax[i_plot], t_font, stats_nw, hdr_str[i_c],
                                  hdr_str[i_c], col[:nr_nw], col[:nr_nw], 'fixed', h_title=h_title)

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

    def det_calc_para_change(self, calc_para, plot_para, current_fcn, plot_scope):
        '''

        :param calc_para:
        :param current_fcn:
        :return:
        '''

        # mandatory update plot scope list
        plot_scope_chk = ['Spiking Frequency Correlation',
                          'ROC Analysis',
                          'Combined Analysis',
                          'Depth-Based Analysis',
                          'Direction LDA',
                          'Speed LDA',
                          'Miscellaneous Functions']

        # mandatory update function list
        func_plot_chk = ['Shuffled Cluster Distances',
                         'Cluster Cross-Correlogram']

        if (self.thread_calc_error) or (self.fcn_data.prev_fcn is None) or (self.calc_cancel) or (self.data.force_calc):
            # if there was an error or initialising, then return a true flag
            return True

        elif self.fcn_data.prev_fcn != current_fcn:
            # if the function has changed, then return a true value
            return True

        elif self.fcn_data.prev_calc_para is None:
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
            if (current_fcn in func_plot_chk) or (plot_scope in plot_scope_chk):
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
               dict([(x, self.fcn_data.curr_para[x]) for x in cf.get_para_dict(fcn_para, 'P')]), \
               dict([(x, self.fcn_data.curr_para[x]) for x in cf.get_para_dict(fcn_para, 'O')])

    def get_rotation_names(self, f_perm, f_key, t_key):
        '''

        :param key:
        :return:
        '''

        return [y if (t_key[x] is None) else t_key[x][y] for x, y in zip(f_key, f_perm)]

    def get_plot_vals(self, r_data, r_obj_tt, g_type, i_cell_b, im, plot_cond, is_cong=False, i_bin=None,
                      use_vel=False, plot_cond2='Black'):
        '''

        :param r_data:
        :param r_obj_tt:
        :param g_type:
        :param i_cell_b:
        :param im:
        :return:
        '''

        # initialisations
        is_dir = i_bin == None

        # if there is a mis-match in cell count, then find the matching cells between conditions
        if len(i_cell_b[im[0]]) != len(i_cell_b[im[1]]):
            i_cell_b[im[0]], _ = cf.det_cell_match_indices(r_obj_tt[im[0]], [0, 0], r_obj_tt[im[1]])

        # retrieves the auc values and cell grouping indices for the current filter combination
        if is_dir:
            g_type_m = g_type[i_cell_b[im[0]]]
            x_auc = r_data.cond_roc_auc[plot_cond2][i_cell_b[im[0]], 2]
            y_auc = r_data.cond_roc_auc[plot_cond][i_cell_b[im[1]], 2]

            # removes any cells where the group type was not calculated
            is_ok = g_type_m >= 0 if not is_cong else np.ones(len(g_type_m), dtype=bool)
            if not np.all(is_ok):
                g_type_m, x_auc, y_auc = g_type_m[is_ok], x_auc[is_ok], y_auc[is_ok]
                i_cell_b[im[0]], i_cell_b[im[1]] = i_cell_b[im[0]][is_ok], i_cell_b[im[1]][is_ok]
        else:
            # case is the kinematic ROC analysis
            g_type_m = None
            if i_bin == 'All Bins':
                # retrieves the valid bin indices
                i_col = self.setup_valid_bin_indices(r_data, use_vel)
                if use_vel:
                    # sets the dependent/independent velocity ROC values
                    x_roc, y_roc = r_data.vel_roc_auc[plot_cond2], r_data.vel_roc_auc[plot_cond]

                else:
                    # sets the dependent/independent speed ROC values
                    x_roc, y_roc = r_data.spd_roc_auc[plot_cond2], r_data.spd_roc_auc[plot_cond]

                # sets the independent/dependent auROC values
                x_auc, y_auc = x_roc[i_cell_b[im[0]], :][:, i_col], y_roc[i_cell_b[im[1]], :][:, i_col]

            else:
                # case is the kinematic ROC analysis
                i_bin_f = np.abs(dcopy(i_bin))
                if use_vel:
                    # case is analysing velocity
                    x_auc = r_data.vel_roc_auc[plot_cond2][i_cell_b[im[0]], i_bin_f]
                    y_auc = r_data.vel_roc_auc[plot_cond][i_cell_b[im[1]], i_bin_f]
                else:
                    # case is analysing speed
                    x_auc = r_data.spd_roc_auc[plot_cond2][i_cell_b[im[0]], i_bin_f]
                    y_auc = r_data.spd_roc_auc[plot_cond][i_cell_b[im[1]], i_bin_f]

        # calculates the compliment of any auc values < 0.5
        ix_c, iy_c = x_auc < 0.5, y_auc < 0.5
        x_auc[ix_c], y_auc[iy_c] = 1 - x_auc[ix_c], 1 - y_auc[iy_c]

        # sets the x/y significance points
        if is_dir:
            x_sig = r_data.phase_auc_sig[i_cell_b[im[0]], 2]
            y_sig = r_data.cond_auc_sig[plot_cond][i_cell_b[im[1]], 2]
        else:
            i_sig = ['Delong', 'Bootstrapping'].index(r_data.kine_auc_stats_type)
            if i_bin == 'All Bins':
                x_sig, y_sig = 0, 0
                # if use_vel:
                #     # case is analysing velocity
                #     x_sig = np.mean(r_data.vel_roc_sig[im[0], i_sig][i_cell_b[im[0]], :][:, i_col], axis=1)
                #     y_sig = np.mean(r_data.vel_roc_sig[im[1], i_sig][i_cell_b[im[1]], :][:, i_col], axis=1)
                # else:
                #     # case is analysing speed
                #     x_sig = np.mean(r_data.spd_roc_sig[im[0], i_sig][i_cell_b[im[0]], :][:, i_col], axis=1)
                #     y_sig = np.mean(r_data.spd_roc_sig[im[1], i_sig][i_cell_b[im[1]], :][:, i_col], axis=1)

            else:
                # case is analysing a specific bin
                if use_vel:
                    # case is analysing velocity
                    x_sig = r_data.vel_roc_sig[im[0], i_sig][i_cell_b[im[0]], i_bin_f]
                    y_sig = r_data.vel_roc_sig[im[1], i_sig][i_cell_b[im[1]], i_bin_f]
                else:
                    # case is analysing speed
                    x_sig = r_data.spd_roc_sig[im[0], i_sig][i_cell_b[im[0]], i_bin_f]
                    y_sig = r_data.spd_roc_sig[im[1], i_sig][i_cell_b[im[1]], i_bin_f]

        # returns the all auc values, group match type and the inter-condition index arrays
        return x_auc, y_auc, g_type_m, x_sig + 2 * y_sig, i_cell_b

    def check_group_plot_feas(self, grp_plot_type, grp_by_filt, show_stats):
        '''

        :param grp_plot_type:
        :param grp_by_filt:
        :return:
        '''

        if (grp_plot_type == 'Stacked Bar') and (not grp_by_filt) and (not show_stats):
            # not possible to generate stacked bar groups when grouping by classification type.
            # outputs error to screen
            m_str = 'Stacked Bar graphs can only be generated when grouping data by filter type.'
            cf.show_error(m_str, 'Incorrect Plotting Parameters')

            # return flag indicating plot type is not feasible
            self.calc_ok = False
            return False
        else:
            # otherwise, plot type is feasible
            return True

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
            data_fix, data_free = cf.get_comp_datasets(self.data)

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

    def get_multi_comp_datasets(self, all_expt, plot_comp, is_list=True):

        if all_expt:
            # memory allocation
            c_data, A = self.data.comp.data, [[] for _ in range(len(self.data.comp.data))]
            data_fix, data_free = dcopy(A), dcopy(A)

            # retrieves the comparison datasets for each experiment
            for i_cd, cd in enumerate(c_data):
                data_fix[i_cd], data_free[i_cd] = cf.get_comp_datasets(self.data, c_data=cd, is_full=True)

        else:
            # retrieves the comparison data struct belonging to the selected experiment
            i_comp = cf.det_comp_dataset_index(self.data.comp.data, plot_comp)
            c_data = dcopy(self.data.comp.data[i_comp])

            # case is only single experiment files have been loaded
            data_fix, data_free = cf.get_comp_datasets(self.data, c_data=c_data, is_full=True)

        # returns the arrays
        return data_fix, data_free, c_data

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

    def get_free_moving_data(self, t_type_filt, i_bin, i_grp, stack_arr=True, use_all_sig=True):
        '''

        :param t_type_filt:
        :param i_bin:
        :return:
        '''

        n_filt = len(t_type_filt)
        if n_filt == 0:
            return None, None

        # parameters
        p_value = 0.05

        # memory allocation
        c_info = self.data.externd.free_data.c_info
        A = np.empty(n_filt, dtype=object)
        v_sf_sig, v_sf_corr = dcopy(A), dcopy(A)

        # dataframe column names
        pval_col = ['ahv_pearson_p_neg', 'ahv_pearson_p_pos']
        rval_col = ['ahv_pearson_r_neg', 'ahv_pearson_r_pos']

        # sets the spiking frequency significance/correlation
        for i_filt, tt in enumerate(t_type_filt):
            # sets the spiking frequency significance/correlation values
            if stack_arr:
                # case is the data values need to be stacked

                # retrieves the correlation values
                v_sf_corr[i_filt] = np.vstack([np.array(ci[i_bin][tt][[rval_col[i_grp]]]) for ci in c_info])

                # sets the significance values
                if use_all_sig:
                    # case is using all significance values
                    v_sf_sig[i_filt] = np.vstack([np.array(ci[i_bin][tt][pval_col]) < p_value for ci in c_info])
                else:
                    # case is using significance values for a given direction
                    v_sf_sig[i_filt] = np.vstack([np.array(ci[i_bin][tt][[pval_col[i_grp]]]) < p_value for ci in c_info])

            else:
                # case is the data values don't need to be stacked

                # retrieves the correlation values
                v_sf_corr[i_filt] = [np.array(ci[i_bin][tt][[rval_col[i_grp]]]) for ci in c_info]

                # sets the significance values
                if use_all_sig:
                    # case is using all significance values
                    v_sf_sig[i_filt] = [np.array(ci[i_bin][tt][pval_col]) < p_value for ci in c_info]
                else:
                    # case is using significance values for a given direction
                    v_sf_sig[i_filt] = [np.array(ci[i_bin][tt][[pval_col[i_grp]]]) < p_value for ci in c_info]

        # returns the arrays
        return v_sf_sig, v_sf_corr

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
                data_fix, _ = self.get_comp_datasets(self.data)
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
        self.update_thread_job('Removing Excluded Cells...', 0.)
        self.data.cluster = dcopy(self.data._cluster)

        # if the function is not
        for i, c in enumerate(self.data.cluster):
            # retrieves the clusters that are to be included
            cl_inc = cfcn.get_inclusion_filt_indices(c, self.data.exc_gen_filt)

            # removes/keeps the voltage spikes depending on the function type
            if current_fcn != 'plot_single_match_mean':
                # if the voltage spikes are not required, then remove them
                if 'vSpike' in c:
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

    def setup_valid_bin_indices(self, r_data, use_vel):
        '''

        :param r_data:
        :param use_vel:
        :return:
        '''

        if use_vel:
            # sets the column arrays (removes the calculation dependent velocity bin)
            i_col = np.ones(np.shape(r_data.vel_xi)[0], dtype=bool)
            i_col[r_data.i_bin_vel] = False

        else:
            # sets the column arrays (removes the calculation dependent velocity bin)
            i_col = np.ones(np.shape(r_data.spd_xi)[0], dtype=bool)
            i_col[r_data.i_bin_spd] = False

        # returns the final array
        return i_col

    def is_loaded_file(self, expFile):
        '''

        :param expFile:
        :return:
        '''

        for c in self.data._cluster:
            if c['expFile'] == expFile:
                return True

        # if no match, then return a false value
        return False

    def set_output_data(self, local_data):
        '''

        :return:
        '''

        # initialisations
        func_data = {}

        # sets the function data (if any variables matched)
        if len(func_data.keys()):
            self.func_data = func_data

    def reorder_array_dim(self, y_arr):
        '''

        :param y_arr:
        :return:
        '''

        # determines the number of rows
        n_row = [x.shape[0] for x in y_arr]

        # ensures the the sub-np object arrays all have the same number of rows
        for i_filt in range(len(y_arr)):
            y_arr[i_filt] = cf.pad_array_with_nans(y_arr[i_filt], n_row=max(n_row) - n_row[i_filt])

        # re-orders the array so that the first dimension is the previous last dimension (and vice versa)
        return np.moveaxis(np.dstack(y_arr), 1, 0)

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
        self.exp_name = None
        self.pool = None

        # initialises the calculation/plotting parameter groupboxes
        self.init_para_groupbox(h_para_grp)

    ################################################
    ####    ANALYSIS FUNCTION INITALISATIONS    ####
    ################################################

    def init_all_func(self):

        # overall declarations
        data, get_gp = self.get_data_fcn(), cfcn.get_glob_para
        has_multi_expt = len(data._cluster) > 1
        init_lda_para, init_def_class_para = cfcn.init_lda_para, cfcn.init_def_class_para

        # determines if the external data fields have been set
        has_free_data = hasattr(data.externd, 'free_data')
        has_eyetrack_data = hasattr(data.externd, 'eye_track')

        # re-initialises the current parameter fields
        self.reset_curr_para_fields()

        #########################################
        ####    CLUSTER MATCHING FUNCTIONS   ####
        #########################################

        # initialisations
        is_split = True
        m_type = ['New Method', 'Old Method']
        scope_txt = ['Individual Cell', 'Whole Experiment']
        plt_list = ['Intersection', 'Wasserstein Distance', 'Bhattacharyya Distance']
        ff_plot_type = ['Individual Cell Correlation', 'Correlation Histogram', 'Correlation Scatterplot']
        vel_dir = ['Negative', 'Positive']
        lcond_type = ['LIGHT1', 'LIGHT2']
        rt_free = ['Black', 'Uniform']

        # retrieves the comparison fixed file names
        calc_comp = self.det_comp_expt_names(True)
        plot_comp = self.det_comp_expt_names(False)

        # initial rotation filter class object
        rot_filt0 = cf.init_rotation_filter_data(False)

        # sets up the freely moving scatterplot rotational filter
        rot_filt_free = cf.init_rotation_filter_data(False)
        rot_filt_free['t_type'] = dcopy(rt_free)

        # initialises the cluster parameters
        free_exp, ff_cluster = self.get_ff_cluster_details()
        def_clust_para = cfcn.init_clust_para(data.comp, free_exp[0])

        # ====> Fixed/Free Cluster Matching
        para = {
            # calculation parameters
            'calc_comp': {
                'gtype': 'C', 'type': 'L', 'text': 'Comparison Fixed Expt', 'list': calc_comp, 'def_val': calc_comp[0],
                'is_enabled': len(calc_comp) > 1, 'para_reset': [[None, self.reset_cluster_para]]
            },
            'd_max': {
                'gtype': 'C', 'text': 'Max Channel Depth Difference', 'def_val': def_clust_para['d_max'],
                'is_int': True, 'min_val': 1
            },
            'r_max': {
                'gtype': 'C', 'text': 'Max Relative Spike Frequency Rate', 'def_val': def_clust_para['r_max'],
                'min_val': 0.01, 'max_val': 1e4
            },
            'sig_corr_min': {
                'gtype': 'C', 'text': 'Signal Correlation Minimum', 'def_val': def_clust_para['sig_corr_min'],
                'min_val': 0.0, 'max_val': 1.0
            },
            'isi_corr_min': {
                'gtype': 'C', 'text': 'ISI Correlation Minimum', 'def_val': def_clust_para['isi_corr_min'],
                'min_val': 0.0, 'max_val': 1.0
            },
            'sig_diff_max': {
                'gtype': 'C', 'text': 'Maximum Proportional Signal Diff', 'def_val': def_clust_para['sig_diff_max'],
                'min_val': 0.0, 'max_val': 1.0
            },
            'sig_feat_min': {
                'gtype': 'C', 'text': 'Signal Feature Difference Minimum', 'def_val': def_clust_para['sig_feat_min'],
                'min_val': 0.0, 'max_val': 1.0
            },
            'w_sig_feat': {
                'gtype': 'C', 'text': 'Signal Feature Score Weight', 'def_val': def_clust_para['w_sig_feat'],
                'min_val': 0.0, 'max_val': 1.0
            },
            'w_sig_comp': {
                'gtype': 'C', 'text': 'Signal Comparison Score Weight', 'def_val': def_clust_para['w_sig_comp'],
                'min_val': 0.0, 'max_val': 1.0
            },
            'w_isi': {
                'gtype': 'C', 'text': 'ISI Score Weight', 'def_val': def_clust_para['w_isi'],
                'min_val': 0.0, 'max_val': 1.0
            },

            # plotting parameters
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1, 'is_list':True},
            'plot_all': {'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para':['i_cluster',True]},
            'm_type': {'type': 'L', 'text': 'Matching Type', 'def_val': m_type[0], 'list': m_type},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Cluster Matching',
                      name='Fixed/Free Cluster Matching',
                      func='plot_cluster_match_signals',
                      multi_fig=['i_cluster'],
                      para=para)

        # ====> Matched Signal Comparison
        para = {
            # plotting parameters
            'plot_comp': {
                'text': 'Comparison Fixed Expt', 'type': 'L', 'list': plot_comp, 'def_val': plot_comp[0],
                'is_enabled': len(plot_comp) > 1
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'n_trace': {'text': 'Raw Trace Count', 'def_val': 500},
            'is_horz': {'type': 'B', 'text': 'Plot Subplots Horizontally', 'def_val': False},
            'rej_outlier': {'type': 'B', 'text': 'Reject Outlier Traces', 'def_val': True},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Cluster Matching',
                      name='Matched Signal Comparison',
                      func='plot_single_match_mean',
                      multi_fig=['i_cluster'],
                      para=para)

        # ====> Cluster Match Metrics
        para = {
            # plotting parameters
            'plot_comp': {
                'text': 'Comparison Fixed Expt', 'type': 'L', 'list': plot_comp, 'def_val': plot_comp[0],
                'is_enabled': len(plot_comp) > 1
            },
            'all_expt': {
                'type': 'B', 'text': 'Plot All Experiments', 'def_val': False,
                'link_para': ['plot_comp', True]
            },
            'is_3d': {'type': 'B', 'text': 'Plot 3D Data', 'def_val': True},
            'm_type': {'type': 'L', 'text': 'Matching Type', 'def_val': m_type[0], 'list': m_type},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Cluster Matching',
                      name='Cluster Match Metrics',
                      func='plot_signal_metrics',
                      para=para)

        # ====> Matched Cluster ISI Metrics (New Method)
        para = {
            # plotting parameters
            'plot_comp': {
                'text': 'Comparison Fixed Expt', 'type': 'L', 'list': plot_comp, 'def_val': plot_comp[0],
                'is_enabled': len(plot_comp) > 1
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1, 'is_list':True},
            'plot_all': {'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para':['i_cluster', True]},
            'sig_type': {'type': 'L', 'text': 'Plot Type', 'def_val': 'Intersection', 'list': plt_list},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Cluster Matching',
                      name='Matched Cluster ISI Metrics (New Method)',
                      func='plot_new_cluster_signals',
                      multi_fig=['i_cluster'],
                      para=para)

        # ====> Inter-Spike Interval Distributions
        para = {
            # plotting parameters
            'plot_comp': {
                'text': 'Comparison Fixed Expt', 'type': 'L', 'list': plot_comp, 'def_val': plot_comp[0],
                'is_enabled': len(plot_comp) > 1
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1, 'is_list': True},
            'plot_all': {'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para': ['i_cluster', True]},
            't_lim': {'text': 'Upper Time Limit', 'def_val': 500, 'min_val': 10},
            'plot_all_bin': {'type': 'B', 'text': 'Plot All Histogram Time Bins',
                             'def_val': False, 'link_para': ['t_lim', True]},
            'is_norm': {'type': 'B', 'text': 'Use ISI Probabilities', 'def_val': True},
            'equal_ax': {'type': 'B', 'text': 'Use Equal Y-Axis Limits', 'def_val': False},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Cluster Matching',
                      name='Inter-Spike Interval Distributions',
                      func='plot_cluster_isi',
                      multi_fig=['i_cluster'],
                      para=para)

        # ====> Fixed/Freely Moving Spiking Frequency Correlation
        para = {
            # calculation parameters
            'n_shuffle': {'gtype': 'C', 'text': 'Correlation Shuffle Count', 'def_val': 100},
            'vel_bin': {
                'gtype': 'C','type': 'L', 'text': 'Velocity Bin Size (deg/s)', 'list': ['5', '10'], 'def_val': '5'
            },
            'n_sample': {'gtype': 'C', 'text': 'Equal Timebin Resampling Count', 'def_val': 100},
            'equal_time': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Equal Timebins', 'def_val': False,
                'link_para': ['n_sample', False]
            },
            'split_vel': {
                'gtype': 'C', 'type': 'B', 'text': 'Split Velocity Range', 'def_val': is_split,
                'para_reset': [[None, self.reset_vel_range]]
            },

            # invisible calculation parameters
            'freq_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Spike Frequency Type', 'list': ['All'],
                'def_val': 'All', 'is_visible': False
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter,
                'para_gui_var': {'rmv_fields': ['t_type', 'match_type']}, 'def_val': rot_filt_free
            },
            'ff_cluster': {'type': 'L', 'text': 'Matched Index', 'def_val': ff_cluster[0], 'list': ff_cluster},
            'free_exp_name': {
                'type': 'L', 'text': 'Free Experiment', 'def_val': free_exp[0], 'list': free_exp,
                'para_reset': [['ff_cluster', self.reset_matched_index]]
            },
            'show_trend': {'type': 'B', 'text': 'Show Regression Trendlines', 'def_val': True},
            'bin_sz': {'text': 'Histogram Bin Size', 'def_val': 0.1, 'min_val': 0.01, 'min_val': 0.5},
            'vel_dir': {
                'type': 'L', 'text': 'Velocity Direction', 'list': vel_dir,
                'def_val': vel_dir[0], 'is_enabled': is_split
            },
            'lcond_type': {
                'type': 'L', 'text': 'Light Condition Type', 'list': lcond_type, 'def_val': lcond_type[0]
            },
            'plot_type': {
                'type': 'L', 'text': 'Plot Type', 'list': ff_plot_type, 'def_val': ff_plot_type[0],
                'link_para': [['ff_cluster', ['Correlation Histogram', 'Correlation Scatterplot']],
                              ['free_exp_name', ['Correlation Histogram', 'Correlation Scatterplot']],
                              ['show_trend', ['Correlation Histogram', 'Correlation Scatterplot']],
                              ['bin_sz', ['Individual Cell Correlation', 'Correlation Scatterplot']],
                              ['lcond_type', 'Individual Cell Correlation'],
                              ['rot_filt', 'Individual Cell Correlation']]
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
        }
        self.add_func(type='Cluster Matching',
                      name='Fixed/Freely Moving Spiking Frequency Correlation',
                      func='plot_fix_free_corr',
                      para=para)

        # ====> Matched Cluster Metrics (Old Method)
        para = {
            # plotting parameters
            'plot_comp': {
                'text': 'Comparison Fixed Expt', 'type': 'L', 'list': plot_comp, 'def_val': plot_comp[0],
                'is_enabled': len(plot_comp) > 1
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1, 'is_list':True},
            'plot_all': {'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para': ['i_cluster', True]},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Cluster Matching',
                      name='Matched Cluster Metrics (Old Method)',
                      func='plot_old_cluster_signals',
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
            # plotting parameters
            'exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'Experiments'},
            'all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['exp_name', True]
            },
            'c_met1': {'type': 'L', 'text': 'Metric #1', 'def_val': c_metric[0], 'list': c_metric},
            'c_met2': {'type': 'L', 'text': 'Metric #2', 'def_val': c_metric[1], 'list': c_metric},
            'c_met3': {'type': 'L', 'text': 'Metric #3', 'def_val': c_metric[2], 'list': c_metric},
            'use_3met': {'type': 'B', 'text': 'Use 3 Metrics For Classification', 'def_val': False,
                      'link_para': ['c_met3', False]},
            'class_type': {'type': 'L', 'text': 'Classification Method', 'def_val': class_type[0], 'list': class_type},
            'm_size': {'text': 'Scatterplot Marker Size', 'def_val': 60},
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

        #################################################
        ####    FREELY MOVING CELL TYPE FUNCTIONS    ####
        #################################################

        # only initialise these functions if there is free data
        if has_free_data:
            # ====> Freely Moving Cell Type Statistics
            para = {
                # plotting parameters
                'free_exp_name': {'type': 'L', 'text': 'Free Experiment', 'def_val': free_exp[0], 'list': free_exp},
                'plot_all': {
                    'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para': ['free_exp_name', True]
                },
                'vel_bin': {'type': 'L', 'text': 'Velocity Bin Size (deg/s)', 'list': ['5', '10'], 'def_val': '5'},
                'use_pcent': {'type': 'B', 'text': 'Use Percentages For Venn Diagram', 'def_val': True},
                'use_place': {'type': 'B', 'text': 'Include Place Cells For Analysis', 'def_val': False},
                'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            }
            self.add_func(type='Freely Moving Cell Types',
                          name='Freely Moving Cell Type Statistics',
                          func='plot_free_cell_stats',
                          para=para)

        ######################################
        ####    EYE TRACKING FUNCTIONS    ####
        ######################################

        # correlation plot type
        tt_eye_track = ['Black', 'Uniform']
        corr_type = ['Violin/Swarmplot', 'Boxplot']
        grp_plot_type = ['Boxplot', 'Separated Bar', 'Violin/Swarmplot', 'Violinplot', 'Stacked Bar']

        if has_eyetrack_data:
            # initialisations
            fld_data = data.externd.eye_track
            etrack_exp = fld_data.exp_name
            etrack_tt = list(np.unique(cf.flat_list([[y.capitalize() for y in x.t_type] for x in fld_data.et_data])))

            # retrieves the eye-tracking parameter
            et_para = cfcn.get_eyetrack_para(fld_data)

            # ====> Eye Movement Event Signals
            para = {
                # calculation parameters
                'use_med_filt': {
                    'gtype': 'C', 'type': 'B', 'text': 'Use Median Filtering', 'def_val': et_para['use_med_filt']
                },
                'rmv_baseline': {
                    'gtype': 'C', 'type': 'B', 'text': 'Remove Derivative Baseline',
                    'def_val': et_para['rmv_baseline']
                },
                'dp_max': {'gtype': 'C', 'text': 'Max Derivative Threshold (mm/s)', 'def_val': et_para['dp_max']},
                'n_sd': {'gtype': 'C', 'text': 'Event Detection Std. Dev. Threshold', 'def_val': et_para['n_sd']},
                'n_pre': {'gtype': 'C', 'text': 'Pre Event Signal Frame Count', 'def_val': et_para['n_pre']},
                'n_post': {'gtype': 'C', 'text': 'Post Event Signal Frame Count', 'def_val': et_para['n_post']},

                # plotting parameters
                'etrack_exp_name': {
                    'type': 'L', 'text': 'Eye Tracking Experiment', 'def_val': etrack_exp[0], 'list': etrack_exp
                },
                'plot_all': {
                    'type': 'B', 'text': 'Plot All Experiments', 'def_val': True, 'link_para': ['etrack_exp_name', True]
                },
                'etrack_tt': {
                    'type': 'CL', 'text': 'Experiment Trial Types', 'list': etrack_tt,
                    'def_val': np.ones(len(etrack_tt), dtype=bool),
                },
                'plot_mean': {'type': 'B', 'text': 'Plot Mean Event Signal', 'def_val': True},
                'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            }
            self.add_func(type='Eye Tracking',
                          name='Eye Movement Event Signals',
                          func='plot_eye_movement_signals',
                          para=para)

            # ====> Eye Movement Event Signals
            para = {
                # calculation parameters
                'use_med_filt': {
                    'gtype': 'C', 'type': 'B', 'text': 'Use Median Filtering', 'def_val': et_para['use_med_filt']
                },
                'rmv_baseline': {
                    'gtype': 'C', 'type': 'B', 'text': 'Remove Derivative Baseline',
                    'def_val': et_para['rmv_baseline']
                },
                'dp_max': {'gtype': 'C', 'text': 'Max Derivative Threshold (mm/s)', 'def_val': et_para['dp_max']},
                'n_sd': {'gtype': 'C', 'text': 'Event Detection Std. Dev. Threshold', 'def_val': et_para['n_sd']},
                'n_pre': {'gtype': 'C', 'text': 'Pre Event Signal Frame Count', 'def_val': et_para['n_pre']},
                'n_post': {'gtype': 'C', 'text': 'Post Event Signal Frame Count', 'def_val': et_para['n_post']},

                # plotting parameters
                'i_cell': {'text': 'Cell Cluster Index', 'def_val': 1, 'min_val': 1},
                'plot_avg': {
                    'type': 'B', 'text': 'Plot Experiment Average', 'def_val': False, 'link_para': ['i_cell', True]
                },
                'exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'EyeTrackExperiments'},
                'etrack_tt': {
                    'type': 'L', 'text': 'Experiment Trial Type', 'list': etrack_tt, 'def_val': etrack_tt[0]
                },
                'm_size': {'text': 'Scatterplot Marker Size', 'def_val': 60},
                'show_err': {'type': 'B', 'text': 'Show SEM Error', 'def_val': True},
                'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            }
            self.add_func(type='Eye Tracking',
                          name='Eye Movement Correlation (Individual Cell)',
                          func='plot_eye_movement_correlation_indiv',
                          para=para)

            # ====> Eye Movement Event Signals (Whole Experiment)
            para = {
                # calculation parameters
                'use_med_filt': {
                    'gtype': 'C', 'type': 'B', 'text': 'Use Median Filtering', 'def_val': et_para['use_med_filt']
                },
                'rmv_baseline': {
                    'gtype': 'C', 'type': 'B', 'text': 'Remove Derivative Baseline',
                    'def_val': et_para['rmv_baseline']
                },
                'dp_max': {'gtype': 'C', 'text': 'Max Derivative Threshold (mm/s)', 'def_val': et_para['dp_max']},
                'n_sd': {'gtype': 'C', 'text': 'Event Detection Std. Dev. Threshold', 'def_val': et_para['n_sd']},
                'n_pre': {'gtype': 'C', 'text': 'Pre Event Signal Frame Count', 'def_val': et_para['n_pre']},
                'n_post': {'gtype': 'C', 'text': 'Post Event Signal Frame Count', 'def_val': et_para['n_post']},

                # plotting parameters
                'etrack_tt': {
                    'type': 'CL', 'text': 'Experiment Trial Types', 'list': etrack_tt,
                    'def_val': np.ones(len(etrack_tt), dtype=bool)
                },
                'corr_type': {
                    'type': 'L', 'text': 'Correlation Plot Type', 'list': corr_type, 'def_val': corr_type[0]
                },
                'pr_type': {
                    'type': 'L', 'text': 'Significance Proportion Plot Type',
                    'list': grp_plot_type, 'def_val': grp_plot_type[1]
                },
                'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            }
            self.add_func(type='Eye Tracking',
                          name='Eye Movement Correlation (Whole Experiment)',
                          func='plot_eye_movement_correlation_exp',
                          para=para)

        ######################################################
        ####    SPIKING FREQUENCY CORRELATION FUNCTIONS   ####
        ######################################################

        # initialisations
        sig_vel_bin = ['5', '10']
        dist_type = ['Cumulative Distribution', 'Histogram']

        # sets up the fixed correlation rotational filter
        rot_filt_corr_fixed = cf.init_rotation_filter_data(False)
        if not hasattr(data.rotation, 'vel_sf_mean'):
            data.rotation.vel_sf_mean = None
            rot_filt_corr_fixed['t_type'] = dcopy(rt_free)
        elif data.rotation.vel_sf_mean is None:
            rot_filt_corr_fixed['t_type'] = dcopy(rt_free)
        else:
            vel_sf_tt = list(data.rotation.vel_sf_mean.keys())
            rot_filt_corr_fixed['t_type'] = dcopy(vel_sf_tt)

        # retrieves the correlation default parameters
        corr_def_para = cfcn.init_corr_para(data.rotation)
        rt_corr = dcopy(rot_filt_corr_fixed['t_type'])
        if len(rt_corr) == 1:
            rt_corr *= 2

        # ====> Individual Cell Correlation (Fixed)
        para = {
            # calculation parameters
            'n_shuffle': {
                'gtype': 'C', 'text': 'Trial Shuffle Count',
                'def_val': cfcn.set_def_para(corr_def_para, 'n_shuffle', 100)
            },
            'vel_bin': {
                'gtype': 'C', 'type': 'L', 'text': 'Velocity Bin Size (deg/s)', 'list': sig_vel_bin,
                'def_val': cfcn.set_def_para(corr_def_para, 'vel_bin', '5')
            },
            'n_smooth': {
                'gtype': 'C', 'text': 'Smoothing Window', 'min_val': 3,
                'def_val': cfcn.set_def_para(corr_def_para, 'n_smooth', 5)
            },
            'is_smooth': {
                'gtype': 'C', 'type': 'B', 'text': 'Smooth Velocity Trace', 'link_para': ['n_smooth', False],
                'def_val': cfcn.set_def_para(corr_def_para, 'is_smooth', False)
            },
            'n_sample': {
                'gtype': 'C', 'text': 'Equal Timebin Resampling Count',
                'def_val': cfcn.set_def_para(corr_def_para, 'n_sample', 100)
            },
            'equal_time': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Equal Timebins', 'link_para': ['n_sample', False],
                'def_val': cfcn.set_def_para(corr_def_para, 'equal_time', False)
            },

            # invisible calculation parameters
            'split_vel': {
                'gtype': 'C', 'type': 'B', 'text': 'Split Velocity Range', 'is_visible': False, 'def_val': True
            },
            'freq_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Spike Frequency Type', 'list': ['All'],
                'def_val': 'All', 'is_visible': False
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'def_val': dcopy(rot_filt0),
                'para_gui_var': {'rmv_fields': ['match_type', 'region_name', 'record_layer', 'sig_type', 'free_ctype']},
                'para_gui': RotationFilter
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_shuffle': {'type': 'B', 'text': 'Plot Shuffled Spiking Frequency Traces', 'def_val': False},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1],
                'is_visible': False
            },
        }
        self.add_func(type='Spiking Frequency Correlation',
                      name='Individual Cell Correlation (Fixed)',
                      func='plot_freq_corr_indiv',
                      para=para)

        # ====> Correlation Distributions (Fixed)
        para = {
            # calculation parameters
            'n_shuffle': {
                'gtype': 'C', 'text': 'Trial Shuffle Count',
                'def_val': cfcn.set_def_para(corr_def_para, 'n_shuffle', 100)
            },
            'vel_bin': {
                'gtype': 'C', 'type': 'L', 'text': 'Velocity Bin Size (deg/s)', 'list': sig_vel_bin,
                'def_val': cfcn.set_def_para(corr_def_para, 'vel_bin', '5')
            },
            'n_smooth': {
                'gtype': 'C', 'text': 'Smoothing Window', 'min_val': 3,
                'def_val': cfcn.set_def_para(corr_def_para, 'n_smooth', 5)
            },
            'is_smooth': {
                'gtype': 'C', 'type': 'B', 'text': 'Smooth Velocity Trace', 'link_para': ['n_smooth', False],
                'def_val': cfcn.set_def_para(corr_def_para, 'is_smooth', False)
            },
            'n_sample': {
                'gtype': 'C', 'text': 'Equal Timebin Resampling Count',
                'def_val': cfcn.set_def_para(corr_def_para, 'n_sample', 100)
            },
            'equal_time': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Equal Timebins', 'link_para': ['n_sample', False],
                'def_val': cfcn.set_def_para(corr_def_para, 'equal_time', False)
            },

            # invisible calculation parameters
            'split_vel': {
                'gtype': 'C', 'type': 'B', 'text': 'Split Velocity Range', 'is_visible': False, 'def_val': True
            },
            'freq_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Spike Frequency Type', 'list': ['All'],
                'def_val': 'All', 'is_visible': False
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter,
                'para_reset': [[None, self.reset_comb_all]], 'para_gui_var': {'rmv_fields': ['match_type']},
                'def_val': dcopy(rot_filt0),
            },
            'dist_type': {'type': 'L', 'text': 'Distribution Type', 'list': dist_type, 'def_val': dist_type[1]},
            'bin_size': {'text': 'Bin Size', 'def_val': 0.1, 'min_val': 0.01, 'max_val': 1.00},
            'comb_all': {
                'type': 'B', 'text': 'Combine Filters Into Single Figure', 'def_val': False, 'is_enabled': False
            },
            'vel_dir': {
                'type': 'L', 'text': 'Velocity Direction', 'list': vel_dir, 'def_val': vel_dir[0]
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1],
                'is_visible': False
            },
            'is_fixed': {'type': 'B', 'text': 'Fixed Expt Setup', 'def_val': True, 'is_visible': False},
        }
        self.add_func(type='Spiking Frequency Correlation',
                      name='Correlation Distributions (Fixed)',
                      func='plot_freq_corr_hist',
                      para=para)

        # ====> Correlation Scatterplot (Fixed)
        para = {
            # calculation parameters
            'n_shuffle': {
                'gtype': 'C', 'text': 'Trial Shuffle Count',
                'def_val': cfcn.set_def_para(corr_def_para, 'n_shuffle', 100)
            },
            'vel_bin': {
                'gtype': 'C', 'type': 'L', 'text': 'Velocity Bin Size (deg/s)', 'list': sig_vel_bin,
                'def_val': cfcn.set_def_para(corr_def_para, 'vel_bin', '5')
            },
            'n_smooth': {
                'gtype': 'C', 'text': 'Smoothing Window', 'min_val': 3,
                'def_val': cfcn.set_def_para(corr_def_para, 'n_smooth', 5)
            },
            'is_smooth': {
                'gtype': 'C', 'type': 'B', 'text': 'Smooth Velocity Trace', 'link_para': ['n_smooth', False],
                'def_val': cfcn.set_def_para(corr_def_para, 'is_smooth', False)
            },
            'n_sample': {
                'gtype': 'C', 'text': 'Equal Timebin Resampling Count',
                'def_val': cfcn.set_def_para(corr_def_para, 'n_sample', 100)
            },
            'equal_time': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Equal Timebins', 'link_para': ['n_sample', False],
                'def_val': cfcn.set_def_para(corr_def_para, 'equal_time', False)
            },

            # invisible calculation parameters
            'split_vel': {
                'gtype': 'C', 'type': 'B', 'text': 'Split Velocity Range', 'is_visible': False, 'def_val': True
            },
            'freq_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Spike Frequency Type', 'list': ['All'],
                'def_val': 'All', 'is_visible': False
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter,
                'para_reset': [[None, self.reset_trial_sel]], 'para_gui_var': {'rmv_fields': ['match_type']},
                'def_val': dcopy(rot_filt0)
            },
            'x_plot': {'type': 'L', 'text': 'X-Axis Trial Type', 'list': rt_corr, 'def_val': rt_corr[0]},
            'y_plot': {'type': 'L', 'text': 'y-Axis Trial Type', 'list': rt_corr, 'def_val': rt_corr[1]},
            'comb_all': {
                'type': 'B', 'text': 'Combine Filters Into Single Figure', 'def_val': False, 'is_enabled': False
            },
            'vel_dir': {
                'type': 'L', 'text': 'Velocity Direction', 'list': vel_dir, 'def_val': vel_dir[0]
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
            'is_fixed': {'type': 'B', 'text': 'Fixed Expt Setup', 'def_val': True, 'is_visible': False},
        }
        self.add_func(type='Spiking Frequency Correlation',
                      name='Correlation Scatterplot (Fixed)',
                      func='plot_freq_corr_scatter',
                      para=para)

        # ====> Correlation Significance (Fixed)
        para = {
            # calculation parameters
            'n_shuffle': {
                'gtype': 'C', 'text': 'Trial Shuffle Count',
                'def_val': cfcn.set_def_para(corr_def_para, 'n_shuffle', 100)
            },
            'vel_bin': {
                'gtype': 'C', 'type': 'L', 'text': 'Velocity Bin Size (deg/s)', 'list': sig_vel_bin,
                'def_val': cfcn.set_def_para(corr_def_para, 'vel_bin', '5')
            },
            'n_smooth': {
                'gtype': 'C', 'text': 'Smoothing Window', 'min_val': 3,
                'def_val': cfcn.set_def_para(corr_def_para, 'n_smooth', 5)
            },
            'is_smooth': {
                'gtype': 'C', 'type': 'B', 'text': 'Smooth Velocity Trace', 'link_para': ['n_smooth', False],
                'def_val': cfcn.set_def_para(corr_def_para, 'is_smooth', False)
            },
            'n_sample': {
                'gtype': 'C', 'text': 'Equal Timebin Resampling Count',
                'def_val': cfcn.set_def_para(corr_def_para, 'n_sample', 100)
            },
            'equal_time': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Equal Timebins', 'link_para': ['n_sample', False],
                'def_val': cfcn.set_def_para(corr_def_para, 'equal_time', False)
            },

            # invisible calculation parameters
            'split_vel': {
                'gtype': 'C', 'type': 'B', 'text': 'Split Velocity Range', 'is_visible': False, 'def_val': True
            },
            'freq_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Spike Frequency Type', 'list': ['All'],
                'def_val': 'All', 'is_visible': False
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter,
                'def_val': dcopy(rot_filt0), 'para_gui_var': {'rmv_fields': ['match_type']}
            },
            'grp_plot_type': {'type': 'L', 'text': 'Plot Type', 'list': grp_plot_type[:-1], 'def_val': grp_plot_type[1]},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            'p_value': {'text': 'Significance Level', 'def_val': 0.05, 'min_val': 0.00, 'max_val': 0.05},
            'grp_by_filt': {'type': 'B', 'text': 'Group Data by Filter Type', 'def_val': True},
            'show_stats': {
                'type': 'B', 'text': 'Show Statistics Tables', 'def_val': False,
                'link_para': [['grp_plot_type', True], ['plot_grid', True], ['p_value', False]]
            },

            # invisible parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
            'is_fixed': {'type': 'B', 'text': 'Fixed Expt Setup', 'def_val': True, 'is_visible': False},
        }
        self.add_func(type='Spiking Frequency Correlation',
                      name='Correlation Significance (Fixed)',
                      func='plot_freq_corr_significance',
                      para=para)

        # only initialise these functions if there is free data
        if has_free_data:
            # retrieves the freely moving trial types
            tt_free = dcopy(list(data.externd.free_data.t_type))

            # ====> Correlation Distributions (Freely Moving)
            para = {
                # calculation parameters
                'vel_bin': {
                    'gtype': 'C', 'type': 'L', 'text': 'Velocity Bin Size (deg/s)', 'list': sig_vel_bin,
                    'def_val': cfcn.set_def_para(corr_def_para, 'vel_bin', '5')
                },

                # plotting parameters
                'rot_filt': {
                    'type': 'CL', 'text': 'Plot Conditions', 'list': tt_free,
                    'def_val': np.ones(len(tt_free), dtype=bool),
                },
                'dist_type': {'type': 'L', 'text': 'Distribution Type', 'list': dist_type, 'def_val': dist_type[1]},
                'bin_size': {'text': 'Bin Size', 'def_val': 0.1, 'min_val': 0.01, 'max_val': 1.00},
                'comb_all': {
                    'type': 'B', 'text': 'Combine Filters Into Single Figure', 'def_val': True, 'is_visible': False
                },
                'vel_dir': {
                    'type': 'L', 'text': 'Velocity Direction', 'list': vel_dir, 'def_val': vel_dir[0]
                },
                'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

                # invisible parameters
                'plot_scope': {
                    'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1],
                    'is_visible': False
                },
                'is_fixed': {'type': 'B', 'text': 'Fixed Expt Setup', 'def_val': False, 'is_visible': False},
            }
            self.add_func(type='Spiking Frequency Correlation',
                          name='Correlation Distributions (Freely Moving)',
                          func='plot_freq_corr_hist',
                          para=para)

            # ====> Correlation Scatterplot (Freely Moving)
            para = {
                # calculation parameters
                'vel_bin': {
                    'gtype': 'C', 'type': 'L', 'text': 'Velocity Bin Size (deg/s)', 'list': sig_vel_bin,
                    'def_val': cfcn.set_def_para(corr_def_para, 'vel_bin', '5')
                },

                # plotting parameters
                'rot_filt': {
                    'type': 'CL', 'text': 'Plot Conditions', 'list': tt_free,
                    'def_val': np.ones(len(tt_free), dtype=bool),
                },
                'x_plot': {'type': 'L', 'text': 'X-Axis Trial Type', 'list': tt_free, 'def_val': tt_free[0]},
                'y_plot': {'type': 'L', 'text': 'y-Axis Trial Type', 'list': tt_free, 'def_val': tt_free[1]},
                'comb_all': {
                    'type': 'B', 'text': 'Combine Filters Into Single Figure', 'def_val': False, 'is_enabled': False
                },
                'vel_dir': {
                    'type': 'L', 'text': 'Velocity Direction', 'list': vel_dir, 'def_val': vel_dir[0]
                },
                'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

                # invisible parameters
                'plot_scope': {
                    'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1],
                    'is_visible': False
                },
                'is_fixed': {'type': 'B', 'text': 'Fixed Expt Setup', 'def_val': False, 'is_visible': False},
            }
            self.add_func(type='Spiking Frequency Correlation',
                          name='Correlation Scatterplot (Freely Moving)',
                          func='plot_freq_corr_scatter',
                          para=para)

            # ====> Correlation Significance (Freely Moving)
            para = {
                # calculation parameters
                'vel_bin': {
                    'gtype': 'C', 'type': 'L', 'text': 'Velocity Bin Size (deg/s)', 'list': sig_vel_bin,
                    'def_val': cfcn.set_def_para(corr_def_para, 'vel_bin', '5')
                },

                # plotting parameters
                'rot_filt': {
                    'type': 'CL', 'text': 'Plot Conditions', 'list': tt_free,
                    'def_val': np.ones(len(tt_free), dtype=bool),
                },
                'grp_plot_type': {
                    'type': 'L', 'text': 'Plot Type', 'list': grp_plot_type[:-1], 'def_val': grp_plot_type[1]
                },
                'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
                'p_value': {'text': 'Significance Level', 'def_val': 0.05, 'min_val': 0.00, 'max_val': 0.05},
                'grp_by_filt': {'type': 'B', 'text': 'Group Data by Filter Type', 'def_val': True},
                'show_stats': {
                    'type': 'B', 'text': 'Show Statistics Tables', 'def_val': False,
                    'link_para': [['grp_plot_type', True], ['plot_grid', True], ['p_value', False]]
                },

                # invisible plotting parameters
                'plot_scope': {
                    'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1],
                    'is_visible': False
                },
                'is_fixed': {'type': 'B', 'text': 'Fixed Expt Setup', 'def_val': False, 'is_visible': False},
            }
            self.add_func(type='Spiking Frequency Correlation',
                          name='Correlation Significance (Freely Moving)',
                          func='plot_freq_corr_significance',
                          para=para)

        ##########################################
        ####    ROTATION ANALYSIS FUNCTIONS   ####
        ##########################################

        if not hasattr(data.rotation, 'vel_sf_mean'):
            data.rotation.vel_sf_mean = None

        # parameters
        t_phase, t_ofs = 1.0, 0.2
        pos_bin = [str(x) for x in [3, 4, 5, 6, 10, 15, 20, 30, 45, 60]]
        vel_bin = [str(x) for x in [4, 5, 8, 10, 16, 20, 40]]

        # type lists
        mean_type = ['Mean', 'Median']
        norm_type = ['Baseline Median Subtraction', 'Min/Max Normalisation', 'None']
        p_cond = list(np.unique(cf.flat_list(cf.det_reqd_cond_types(data, ['Uniform', 'LandmarkLeft', 'LandmarkRight']))))
        spread_type = ['Individual Trial Traces', 'SEM Error Patches']
        s_type = ['Direction Selectivity', 'Motion Sensitivity']
        ksig_type = ['Individual Cell', 'Correlation Histogram', 'Correlation Scatterplot', 'Correlation Significance']
        cell_type = ['All Cells', 'Narrow Spike Cells', 'Wide Spike Cells']
        comp_type = ['CW vs BL', 'CCW vs BL']

        # sets the LDA comparison types
        comp_type = np.unique(
            cf.flat_list([x['rotInfo']['trial_type'] if x['rotInfo'] is not None else [] for x in data._cluster]))
        comp_type = list(comp_type[comp_type != 'Black'])
        ind_comp = [x == 'Uniform' for x in comp_type]

        # initialises the rotation filter for
        rot_filt_comp = cf.init_rotation_filter_data(False)
        rot_filt_comp['t_type'] = ['Black'] + p_cond

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
            'show_pref_dir': {'type': 'B', 'text': 'Sbow Preferred Direction', 'def_val': True},

            'n_bin': {'text': 'Histogram Bin Count', 'def_val': 20, 'min_val': 10},
            'show_err': {'type': 'B', 'text': 'Show SEM Error', 'def_val': True},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Rotation Analysis',
                      name='Rotation Trial Spiking Rates',
                      func='plot_rotation_trial_spikes',
                      para=para)

        # ====> Rotation Trial Spike Rate Comparison (Individual Cell)
        para = {
            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'plot_exp_name': {
                'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'
            },
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': False, 'is_visible': False
            },
            'ms_prop': {
                'type': 'B', 'text': 'Show DS Cell Proportion Of MS Cell Population',
                'def_val': False, 'is_visible': False
            },
            'grp_plot_type': {
                'type': 'L', 'text': 'Plot Type', 'list': grp_plot_type, 'def_val': grp_plot_type[0],
                'is_visible': False
            },
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[0], 'is_visible': False
            },
            'plot_trend': {'type': 'B', 'text': 'Plot Group Trendlines', 'def_val': False},
            'm_size': {'text': 'Scatterplot Marker Size', 'def_val': 30},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            'p_value': {'text': 'Significance Level', 'def_val': 0.05, 'min_val': 0.00, 'max_val': 0.05},
            'grp_by_filt': {'type': 'B', 'text': 'Group Data by Filter Type', 'def_val': True, 'is_visible': False},
            'show_stats': {'type': 'B', 'text': 'Show Statistics Tables', 'def_val': False, 'is_visible': False},

            # output variables

        }
        self.add_func(type='Rotation Analysis',
                      name='Phase Spiking Rate Comparison (Individual Cell)',
                      func='plot_phase_spike_freq',
                      para=para)

        # ====> Rotation Trial Spike Rate Comparison (Whole Experiment)
        para = {
            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1, 'is_visible': False},
            'plot_exp_name': {
                'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments', 'is_visible': False
            },
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True],
                'is_visible': False
            },
            'ms_prop': {'type': 'B', 'text': 'Show DS Cell Proportion Of MS Cell Population', 'def_val': False},
            'grp_plot_type': {'type': 'L', 'text': 'Plot Type', 'list': grp_plot_type, 'def_val': grp_plot_type[1]},
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
            'plot_trend': {'type': 'B', 'text': 'Plot Group Trendlines', 'def_val': False},
            'm_size': {'text': 'Scatterplot Marker Size', 'def_val': 30},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            'p_value': {'text': 'Significance Level', 'def_val': 0.05, 'min_val': 0.00, 'max_val': 0.05},
            'grp_by_filt': {'type': 'B', 'text': 'Group Data by Filter Type', 'def_val': True},
            'show_stats': {
                'type': 'B', 'text': 'Show Statistics Tables', 'def_val': False,
                'link_para': [['ms_prop', True], ['grp_plot_type', True], ['plot_trend', True],
                              ['m_size', True], ['plot_grid', True], ['p_value', False]]
            },
        }
        self.add_func(type='Rotation Analysis',
                      name='Phase Spiking Rate Comparison (Whole Experiment)',
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
            'norm_type': {
                'text': 'Heatmap Normalisation Type', 'type': 'L', 'list': norm_type, 'def_val': norm_type[0]
            },
            'mean_type': {
                'text': 'Histogram Averaging Type', 'type': 'L', 'list': mean_type, 'def_val': mean_type[0]
            },
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1],
                'link_para': [['i_cluster', 'Whole Experiment'], ['plot_exp_name', 'Individual Cell'],
                              ['plot_all_expt', 'Individual Cell'], ['norm_type', 'Individual Cell']]
            },
            'dt': {'text': 'Heatmap Resolution (ms)', 'def_val': 100, 'min_val': 10},
        }
        self.add_func(type='Rotation Analysis',
                      name='Spiking Rate Heatmap',
                      func='plot_spike_freq_heatmap',
                      para=para)

        # ====> Rotation Trial Motion/Direction Selectivity
        para = {
            # # calculation parameters
            # 't_phase_vis': {'gtype': 'C', 'text': 'Visual Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10},
            # 't_ofs_vis': {'gtype': 'C', 'text': 'Visual Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00},

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

            'plot_cond': {
                'type': 'CL', 'text': 'Plot Conditions', 'list': p_cond, 'def_val': np.ones(len(p_cond), dtype=bool)
            },
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

        # ====> Firing Rate Distributions
        para = {
            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None,
                              'list': 'RotationExperimentMD'},
            'plot_all_expt': {'type': 'B', 'text': 'Analyse All Experiments',
                              'def_val': True, 'link_para': ['plot_exp_name', True]},
            'comp_type': {
                'type': 'L', 'text': 'Phase Comparison Type', 'list': comp_type, 'def_val': comp_type[0]
            },

            'n_smooth': {'text': 'Smoothing Window', 'def_val': 5, 'min_val': 3},
            'smooth_hist': {
                'type': 'B', 'text': 'Smooth Firing Rate Histogram', 'def_val': False, 'link_para': ['n_smooth', False],
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Rotation Analysis',
                      name='Firing Rate Distributions',
                      func='plot_firing_rate_distributions',
                      para=para)

        # ====> Kinematic Spiking Frequency
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
            'spread_type': {
                'type': 'L', 'text': 'Individual Trial Spread Type', 'list': spread_type, 'def_val': spread_type[0],
            },
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[0],
                'link_para': [['i_cluster', 'Whole Experiment'], ['plot_exp_name', 'Individual Cell'],
                              ['plot_all_expt', 'Individual Cell'], ['spread_type', 'Whole Experiment']]
            },

            'pos_bin': {'type': 'L', 'text': 'Position Bin Size (deg)', 'list': pos_bin, 'def_val': '10'},
            'vel_bin': {'type': 'L', 'text': 'Velocity Bin Size (deg/s)', 'list': vel_bin, 'def_val': '5'},
            'n_smooth': {'text': 'Smoothing Window', 'def_val': 5, 'min_val': 3},
            'is_smooth': {
                'type': 'B', 'text': 'Smooth Velocity Trace', 'def_val': True, 'link_para': ['n_smooth', False]
            },
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
            'grp_plot_type': {'type': 'L', 'text': 'Plot Type', 'list': grp_plot_type[:-1], 'def_val': grp_plot_type[1]},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            'p_value': {'text': 'Significance Level', 'def_val': 0.05, 'min_val': 0.00, 'max_val': 0.05},
            'grp_by_filt': {'type': 'B', 'text': 'Group Data by Filter Type', 'def_val': True},
            'show_stats': {
                'type': 'B', 'text': 'Show Statistics Tables', 'def_val': False,
                'link_para': [['grp_plot_type', True], ['plot_grid', True], ['p_value', False]]
            },

            # invisible parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
            'plot_exp_name': {
                'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments', 'is_visible': False
            },
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True],
                'is_visible': False
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
            'show_err': {'type': 'B', 'text': 'Show SEM Error', 'def_val': True},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            'rmv_median': {'type': 'B', 'text': 'Remove Baseline Median', 'def_val': True, 'is_visible': False},
            'show_pref_dir': {'type': 'B', 'text': 'Sbow Preferred Direction', 'def_val': False, 'is_visible': False},
        }
        self.add_func(type='UniformDrift Analysis',
                      name='UniformDrift Trial Spiking Rates',
                      func='plot_unidrift_trial_spikes',
                      para=para)

        # ====> UniformDrift Phase Spiking Rate Comparison (Individual Cell)
        para = {
            # calculation parameters
            't_phase_vis': {
                'gtype': 'C', 'text': 'Analysis Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10
            },
            't_ofs_vis': {
                'gtype': 'C', 'text': 'Analysis Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'UniformDrifting Filter Parameters', 'para_gui': RotationFilter,
                'para_gui_var': {'rmv_fields': ['t_freq_dir']}, 'def_val': None
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1},
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperimentUD'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': False, 'is_visible': False
            },
            'ms_prop': {
                'type': 'B', 'text': 'Show DS Cell Proportion Of MS Cell Population',
                'def_val': False, 'is_visible': False
            },
            'grp_plot_type': {
                'type': 'L', 'text': 'Plot Type', 'list': grp_plot_type, 'def_val': grp_plot_type[0],
                'is_visible': False
            },
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[0], 'is_visible': False
            },
            'plot_trend': {'type': 'B', 'text': 'Plot Group Trendlines', 'def_val': False},
            'm_size': {'text': 'Scatterplot Marker Size', 'def_val': 30},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            'p_value': {'text': 'Significance Level', 'def_val': 0.05, 'min_val': 0.00, 'max_val': 0.05},
            'grp_by_filt': {'type': 'B', 'text': 'Group Data by Filter Type', 'def_val': True, 'is_visible': False},
            'show_stats': {'type': 'B', 'text': 'Show Statistics Tables', 'def_val': False, 'is_visible': False},
        }
        self.add_func(type='UniformDrift Analysis',
                      name='UniformDrift Phase Spiking Rate Comparison (Individual Cell)',
                      func='plot_unidrift_spike_freq',
                      para=para)

        # ====> UniformDrift Phase Spiking Rate Comparison (Whole Experiment)
        para = {
            # calculation parameters
            't_phase_vis': {
                'gtype': 'C', 'text': 'Analysis Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10
            },
            't_ofs_vis': {
                'gtype': 'C', 'text': 'Analysis Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'UniformDrifting Filter Parameters', 'para_gui': RotationFilter,
                'para_gui_var': {'rmv_fields': ['t_freq_dir']}, 'def_val': None
            },
            'i_cluster': {'text': 'Cluster Index', 'def_val': 1, 'min_val': 1, 'is_invisible': False},
            'plot_exp_name': {
                'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperimentUD', 'is_visible': False
            },
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True],
                'is_visible': False
            },
            'ms_prop': {'type': 'B', 'text': 'Show DS Cell Proportion Of MS Cell Population', 'def_val': True},
            'grp_plot_type': {'type': 'L', 'text': 'Plot Type', 'list': grp_plot_type, 'def_val': grp_plot_type[1]},
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
            'plot_trend': {'type': 'B', 'text': 'Plot Group Trendlines', 'def_val': False},
            'm_size': {'text': 'Scatterplot Marker Size', 'def_val': 30},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            'p_value': {'text': 'Significance Level', 'def_val': 0.05, 'min_val': 0.00, 'max_val': 0.05},
            'grp_by_filt': {'type': 'B', 'text': 'Group Data by Filter Type', 'def_val': True},
            'show_stats': {
                'type': 'B', 'text': 'Show Statistics Tables', 'def_val': False,
                'link_para': [['ms_prop', True], ['grp_plot_type', True], ['plot_trend', True],
                              ['m_size', True], ['plot_grid', True], ['p_value', False]]
            },
        }
        self.add_func(type='UniformDrift Analysis',
                      name='UniformDrift Phase Spiking Rate Comparison (Whole Experiment)',
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
            'norm_type': {
                'text': 'Heatmap Normalisation Type', 'type': 'L', 'list': norm_type, 'def_val': norm_type[0]
            },
            'mean_type': {
                'text': 'Histogram Averaging Type', 'type': 'L', 'list': mean_type, 'def_val': mean_type[0]
            },
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1],
                'link_para': [['i_cluster', 'Whole Experiment'], ['plot_exp_name', 'Individual Cell'],
                              ['plot_all_expt', 'Individual Cell'], ['norm_type', 'Individual Cell']]
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

        # parameters
        dv, v_rng, n_boot_def = 5, 80, 100

        # type lists
        roc_vel_bin = ['5', '10', '20', '40', '80']
        hist_bin_sz = ['2', '5', '10', '20', '25', '50']
        md_grp_type = ['MS/DS', 'MS/Not DS', 'Not MS', 'All Cells']
        pd_grp_type = ['None', 'Rotation', 'Visual', 'Both', 'All Cells']
        grp_stype = ['Wilcoxon Paired Test', 'Delong', 'Bootstrapping']
        auc_stype = ['Delong', 'Bootstrapping']
        freq_type = ['Decreasing', 'Increasing', 'All']
        exc_type = ['Use All Cells', 'Low Firing Cells', 'High Firing Cells', 'Band Pass']
        phase_comp_type = ['CW vs BL', 'CCW vs BL', 'CCW vs CW']
        resp_grp_type = ['Rotation/Visual Response', 'Motion Sensitivity/Direction Selectivity', 'Congruency']
        auc_plt_type = ['Violinplot + Swarmplot', 'Bubbleplot']

        # determines if any uniform/motor drifting experiments exist + sets the visual experiment type
        has_vis_expt, has_ud_expt, has_md_expt = cf.det_valid_vis_expt(self.get_data_fcn())
        vis_type = list(np.array(['UniformDrifting', 'MotorDrifting'])[np.array([has_ud_expt, has_md_expt])])
        vis_type_0 = vis_type[0] if len(vis_type) else 'N/A'
        cell_desc_type = ['Motion/Direction Selectivity', 'Rotation/Visual DS', 'Congruency']
        vel_sig_type = ['auROC Significance', 'auROC CDF/Statistics']
        ms_scat_type = ['auROC Scatterplot', 'auROC Significance/Histogram']

        # velocity/speed ranges
        vc_rng = cfcn.get_kinematic_range_strings(dv, True, v_rng)
        sc_rng = cfcn.get_kinematic_range_strings(dv, False, v_rng)
        p_cond_vel = list(np.unique(cf.flat_list(cf.det_reqd_cond_types(data, ['Uniform', 'MotorDrifting']))))

        #
        rot_filt_grp = cf.init_rotation_filter_data(False)
        rot_filt_grp['t_type'] = ['Black'] + ['Uniform'] + ['MotorDrifting'] if has_md_expt else []

        #
        vel_roc_para = cfcn.init_roc_para(data.rotation, 'vel_roc_sig')
        vc_rng_sig = cfcn.get_kinematic_range_strings(float(vel_roc_para['vel_bin']), True, v_rng)

        # ====> Direction ROC Curves (Single Cell)
        para = {
            # calculation parameters
            'n_boot': {'gtype': 'C', 'text': 'Number bootstrapping shuffles', 'def_val': n_boot_def, 'min_val': 100},
            'auc_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'AUC CI Calculation Type', 'list': auc_stype,
                'def_val': auc_stype[0], 'link_para': ['n_boot', 'Delong']
            },
            't_phase_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Duration (s)', 'def_val': 1.0, 'min_val': 0.10
            },
            't_ofs_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Offset (s)', 'def_val': 0.2, 'min_val': 0.00
            },
            'use_full_rot': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Rotation Phase', 'def_val': True,
                'link_para': [['t_phase_rot', True], ['t_ofs_rot', True]]
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
                'def_val': grp_stype[0], 'link_para': ['n_boot', ['Delong', 'Wilcoxon Paired Test']]
            },
            'auc_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'AUC CI Calculation Type', 'list': auc_stype,
                'def_val': auc_stype[0], 'link_para': ['n_boot', 'Delong']
            },
            't_phase_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10
            },
            't_ofs_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00
            },
            'use_full_rot': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Rotation Phase', 'def_val': True,
                'link_para': [['t_phase_rot', True], ['t_ofs_rot', True]]
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
            'violin_bw': {'text': 'Violinplot Width Scale Factor', 'def_val': 1, 'min_val': 0},
            'm_size': {'text': 'Swarmplot Marker Size', 'def_val': 3, 'min_val': 1},
            'cell_grp_type': {
                'type': 'L', 'text': 'Cell Grouping Type', 'list': md_grp_type, 'def_val': md_grp_type[-1],
            },
            'auc_plot_type': {
                'type': 'L', 'text': 'auROC Plot Type', 'list': auc_plt_type, 'def_val': auc_plt_type[0],
                'link_para': [['violin_bw', 'Bubbleplot'], ['m_size', 'Bubbleplot']]
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_grp_type': {
                'type': 'L', 'text': 'Cell Discrimination Type', 'list': [cell_desc_type[0]],
                'def_val': cell_desc_type[0], 'is_visible': False
            },
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
        }
        self.add_func(type='ROC Analysis',
                      name='Direction ROC Curves (Whole Experiment)',
                      func='plot_direction_roc_curves_whole',
                      para=para)

        # ====> Direction ROC AUC Histograms
        para = {
            # calculation parameters
            'n_boot': {'gtype': 'C', 'text': 'Number bootstrapping shuffles', 'def_val': n_boot_def, 'min_val': 100},
            'grp_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'Cell Grouping Significance Test', 'list': grp_stype,
                'def_val': grp_stype[0], 'link_para': ['n_boot', ['Delong', 'Wilcoxon Paired Test']]
            },
            'auc_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'AUC CI Calculation Type', 'list': auc_stype,
                'def_val': auc_stype[0], 'link_para': ['n_boot', 'Delong']
            },
            't_phase_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10
            },
            't_ofs_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00
            },
            'use_full_rot': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Rotation Phase', 'def_val': True,
                'link_para': [['t_phase_rot', True], ['t_ofs_rot', True]]
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True]
            },
            'bin_sz': {
                'type': 'L', 'text': 'Histogram Bin Size', 'list': hist_bin_sz, 'def_val': hist_bin_sz[1],
            },
            'phase_type': {
                'type': 'L', 'text': 'Phase Comparison Type', 'list': phase_comp_type, 'def_val': phase_comp_type[-1],
            },
            'show_sig_cells': {'type': 'B', 'text': 'Show Significant Cells', 'def_val': True},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
        }
        self.add_func(type='ROC Analysis',
                      name='Direction ROC AUC Histograms',
                      func='plot_direction_roc_auc_histograms',
                      para=para)

        # ====> Velocity ROC Curves (Single Cell)
        para = {
            # calculation parameters
            'n_boot': {'gtype': 'C', 'text': 'Number bootstrapping shuffles', 'def_val': n_boot_def, 'min_val': 100},
            'auc_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'AUC CI Calculation Type', 'list': auc_stype, 'def_val': auc_stype[0],
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
            'freq_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Spike Frequency Type', 'list': freq_type, 'def_val': freq_type[-1]
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
                'gtype': 'C', 'type': 'L', 'text': 'AUC CI Calculation Type', 'list': auc_stype, 'def_val': auc_stype[0],
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
            'freq_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Spike Frequency Type', 'list': freq_type, 'def_val': freq_type[-1]
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
                'gtype': 'C', 'type': 'L', 'text': 'AUC CI Calculation Type', 'list': auc_stype, 'def_val': auc_stype[0],
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
            'freq_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Spike Frequency Type', 'list': freq_type, 'def_val': freq_type[-1]
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

        # ====> Velocity ROC Significance
        para = {
            # calculation parameters
            'n_boot': {
                'gtype': 'C', 'text': 'Number bootstrapping shuffles', 'def_val': vel_roc_para['n_boot']
            },
            'auc_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'AUC CI Calculation Type', 'list': auc_stype,
                'def_val': vel_roc_para['auc_stype'], 'link_para': ['n_boot', 'Delong']
            },
            'spd_x_rng': {
                'gtype': 'C', 'type': 'L', 'text': 'Comparison Speed Range', 'list': sc_rng,
                'def_val': vel_roc_para['spd_x_rng']
            },
            'vel_bin': {
                'gtype': 'C', 'type': 'L', 'text': 'Velocity Bin Size (deg/sec)', 'list': roc_vel_bin,
                'para_reset': [['spd_x_rng', self.reset_spd_rng], ['i_bin', self.reset_plot_index]],
                'def_val': vel_roc_para['vel_bin'],
            },
            'n_sample': {'gtype': 'C', 'text': 'Equal Timebin Resampling Count', 'def_val': vel_roc_para['n_sample']},
            'equal_time': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Equal Timebins', 'def_val': vel_roc_para['equal_time'],
                'link_para': ['n_sample', False]
            },

            # invisible calculation parameters
            'freq_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Spike Frequency Type', 'list': ['All'], 'def_val': 'All',
                'is_visible': False
            },
            'pn_calc': {'gtype': 'C', 'text': 'Use Pos/Neg', 'def_val': False, 'is_visible': False},

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Filter Parameters', 'para_gui': RotationFilter, 'def_val': None
            },

            'plot_err': {'type': 'B', 'text': 'Plot Error Patch', 'def_val': True},
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True,
                'link_para': [['plot_exp_name', True], ['plot_err', True]]
            },
            'pool_expt': {
                'type': 'B', 'text': 'Pool All Experiments', 'def_val': False,
                'link_para': [['plot_exp_name', True], ['plot_all_expt', True], ['plot_err', True]]
            },

            'plot_cond': {'type': 'L', 'text': 'Plot Conditions', 'list': p_cond_vel, 'def_val': 'Uniform'},
            'i_bin': {
                'type': 'L', 'text': 'Speed/Velocity Bin Index', 'list': vc_rng_sig + ['All Bins'],
                'def_val': '0 to {0}'.format(dv),
            },
            'use_vel': {
                'type': 'B', 'text': 'Plot Velocity ROC Values', 'def_val': True,
                'para_reset': [['i_bin', self.reset_plot_index]]
            },
            'plot_type': {
                'type': 'L', 'text': 'Analysis Plot Type', 'list': vel_sig_type, 'def_val': vel_sig_type[0],
                'link_para': [['plot_cond', vel_sig_type[0]], ['i_bin', vel_sig_type[0]],
                              ['plot_err', vel_sig_type[1]], ['plot_exp_name', vel_sig_type[1]],
                              ['plot_all_expt', vel_sig_type[1]], ['pool_expt', vel_sig_type[1]]]
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible plotting parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
        }
        self.add_func(type='ROC Analysis',
                      name='Velocity ROC Significance',
                      func='plot_velocity_significance',
                      para=para)

        # ====> Condition ROC Curve Comparison
        para = {
            # calculation parameters
            'n_boot': {'gtype': 'C', 'text': 'Number bootstrapping shuffles', 'def_val': n_boot_def, 'min_val': 100},
            'grp_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'Cell Grouping Significance Test', 'list': grp_stype,
                'def_val': grp_stype[0], 'link_para': ['n_boot', ['Delong', 'Wilcoxon Paired Test']]
            },
            'auc_stype': {'gtype': 'C', 'type': 'L', 'text': 'AUC CI Calculation Type', 'list': auc_stype,
                         'def_val': auc_stype[0], 'link_para': ['n_boot', 'Delong']},
            't_phase_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10
            },
            't_ofs_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00
            },
            'use_full_rot': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Rotation Phase', 'def_val': True,
                'link_para': [['t_phase_rot', True], ['t_ofs_rot', True]]
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter,
                'para_gui_var': {'rmv_fields': ['t_type']}, 'def_val': rot_filt_comp
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
                'def_val': grp_stype[1], 'link_para': ['n_boot', ['Delong', 'Wilcoxon Paired Test']]
            },
            'auc_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'AUC CI Calculation Type', 'list': auc_stype, 'def_val': auc_stype[0],
                'link_para': ['n_boot', 'Delong']
            },
            't_phase_vis': {
                'gtype': 'C', 'text': 'Visual Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10,
                'is_enabled': has_vis_expt
            },
            't_ofs_vis': {
                'gtype': 'C', 'text': 'Visual Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00,
                'is_enabled': has_vis_expt
            },
            'use_full_vis': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Visual Phase', 'def_val': False,
                'link_para': [['t_phase_vis', True], ['t_ofs_vis', True]], 'is_enabled': has_vis_expt
            },
            'vis_expt_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Visual Experiment Type', 'list': vis_type, 'def_val': vis_type_0
            },
            't_phase_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10
            },
            't_ofs_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00
            },
            'use_full_rot': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Rotation Phase', 'def_val': True,
                'link_para': [['t_phase_rot', True], ['t_ofs_rot', True]]
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Rotation Filter Parameters', 'para_gui': RotationFilter,
                'para_gui_var': {'rmv_fields': ['t_type']}, 'def_val': rot_filt_grp
            },
            'plot_exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments'},
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'link_para': ['plot_exp_name', True]
            },
            'plot_cond': {
                'type': 'L', 'text': 'Comparison Condition', 'list': rot_filt_grp['t_type'], 'def_val': 'Uniform',
            },
            'm_size': {'text': 'Scatterplot Marker Size', 'def_val': 50},
            'plot_trend': {'type': 'B', 'text': 'Plot Group Trendlines', 'def_val': False},
            'show_sig_markers': {
                'type': 'B', 'text': 'Show Cell Significance', 'def_val': True
            },
            # 'use_resp_grp_type': {
            #     'type': 'B', 'text': 'Use Response Cell Grouping', 'def_val': False, 'is_enabled': has_vis_expt,
            # },
            'mark_type': {
                'type': 'L', 'text': 'Grouping Type', 'list': resp_grp_type, 'def_val': resp_grp_type[0],
                'link_para': ['show_sig_markers', 'Congruency']
            },
            'show_grp_markers': {
                'type': 'B', 'text': 'Show Grouping Type Markers', 'def_val': False, 'link_para': ['mark_type', False]
            },
            'plot_type': {
                'type': 'L', 'text': 'Analysis Plot Type', 'list': ms_scat_type, 'def_val': ms_scat_type[0],
                'link_para': [['mark_type', ms_scat_type[1]], ['show_grp_markers', ms_scat_type[1]],
                              ['plot_trend', ms_scat_type[1]]]
            },
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
        comb_type = ['Motion Sensitivity', 'Direction Selectivity', 'Congruency']

        # ====> Combined Stimuli Statistics
        para = {
            # calculation parameters
            'n_boot': {'gtype': 'C', 'text': 'Number bootstrapping shuffles', 'def_val': n_boot_def, 'min_val': 100},
            'grp_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'Cell Grouping Significance Test', 'list': grp_stype,
                'def_val': grp_stype[0], 'link_para': ['n_boot', ['Delong', 'Wilcoxon Paired Test']]
            },
            't_phase_vis': {
                'gtype': 'C', 'text': 'Visual Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10
            },
            't_ofs_vis': {
                'gtype': 'C', 'text': 'Visual Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00
            },
            'use_full_vis': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Visual Phase', 'def_val': False,
                'link_para': [['t_phase_vis', True], ['t_ofs_vis', True]]
            },
            'vis_expt_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Visual Experiment Type', 'list': vis_type, 'def_val': vis_type_0
            },
            'p_value': {'gtype': 'C', 'text': 'Significance Level', 'def_val': 0.05, 'min_val': 0.00, 'max_val': 0.05},

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Filter Parameters', 'para_gui': RotationFilter, 'def_val': None,
                'para_gui_var': {'rmv_fields': ['t_freq_dir', 't_type', 't_cycle']}
            },
            'plot_exp_name': {
                'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments', 'is_visible': False
            },
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'is_visible': False
            },
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
            'plot_type': {
                'type': 'CL', 'text': 'Plot Metrics', 'list': comb_type, 'def_val': np.ones(len(comb_type), dtype=bool),
                'other_para': '--- Select Plot Metrics ---'
            },
            'grp_plot_type': {'type': 'L', 'text': 'Plot Type', 'list': grp_plot_type, 'def_val': grp_plot_type[1]},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            'grp_by_filt': {'type': 'B', 'text': 'Group Data by Filter Type', 'def_val': True},
            'show_stats': {
                'type': 'B', 'text': 'Show Statistics Tables', 'def_val': False,
                'link_para': [['plot_type', True], ['grp_plot_type', True], ['plot_grid', True]]
            },
        }
        self.add_func(type='Combined Analysis',
                      name='Rotation/Visual Stimuli Response Statistics',
                      func='plot_combined_stimuli_stats',
                      para=para)

        # ====> Combined Stimuli Statistics
        para = {
            # calculation parameters
            'n_boot': {'gtype': 'C', 'text': 'Number bootstrapping shuffles', 'def_val': n_boot_def, 'min_val': 100},
            'grp_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'Cell Grouping Significance Test', 'list': grp_stype,
                'def_val': grp_stype[0], 'link_para': ['n_boot', ['Delong', 'Wilcoxon Paired Test']]
            },
            # 'auc_stype': {
            #     'gtype': 'C', 'type': 'L', 'text': 'AUC CI Calculation Type', 'list': auc_stype,
            #     'def_val': auc_stype[0], 'link_para': ['n_boot', 'Delong']
            # },
            't_phase_vis': {
                'gtype': 'C', 'text': 'Visual Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10
            },
            't_ofs_vis': {
                'gtype': 'C', 'text': 'Visual Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00
            },
            'use_full_vis': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Visual Phase', 'def_val': False,
                'link_para': [['t_phase_vis', True], ['t_ofs_vis', True]]
            },
            'vis_expt_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Visual Experiment Type', 'list': vis_type, 'def_val': vis_type_0
            },
            't_phase_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10
            },
            't_ofs_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00
            },
            'use_full_rot': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Rotation Phase', 'def_val': True,
                'link_para': [['t_phase_rot', True], ['t_ofs_rot', True]]
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
            'violin_bw': {'text': 'Violinplot Width Scale Factor', 'def_val': 1, 'min_val': 0},
            'm_size': {'text': 'Swarmplot Marker Size', 'def_val': 3, 'min_val': 1},
            'plot_grp_type': {
                'type': 'L', 'text': 'Cell Discrimination Type', 'list': cell_desc_type[1:],
                'def_val': cell_desc_type[1], 'para_reset': [['cell_grp_type', self.reset_grp_type]]
            },
            'cell_grp_type': {
                'type': 'L', 'text': 'Cell Grouping Type', 'list': pd_grp_type, 'def_val': pd_grp_type[0]
            },
            'auc_plot_type': {
                'type': 'L', 'text': 'auROC Plot Type', 'list': auc_plt_type, 'def_val': auc_plt_type[0],
                'link_para': [['violin_bw', 'Bubbleplot'], ['m_size', 'Bubbleplot']]
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
        }
        self.add_func(type='Combined Analysis',
                      name='Combined Direction ROC Curves (Whole Experiment)',
                      func='plot_combined_direction_roc_curves',
                      para=para)

        #############################################
        ####    DEPTH BASED ANALYSIS FUNCTIONS   ####
        #############################################

        # parameter initialisations
        r_filt_depth = cf.init_rotation_filter_data(False)

        # parameter lists
        depth_type = ['Preferred/Baseline FR Difference', 'CW/CCW auROC Difference', 'CW/CCW FR Difference']

        # ====> Depth Spiking Rate Comparison
        para = {
            # calculation parameters
            't_phase_vis': {
                'gtype': 'C', 'text': 'Visual Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10
            },
            't_ofs_vis': {
                'gtype': 'C', 'text': 'Visual Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00
            },
            'use_full_vis': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Visual Phase', 'def_val': False,
                'link_para': [['t_phase_vis', True], ['t_ofs_vis', True]]
            },
            't_phase_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10
            },
            't_ofs_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00
            },
            'use_full_rot': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Rotation Phase', 'def_val': True,
                'link_para': [['t_phase_rot', True], ['t_ofs_rot', True]]
            },

            # invisible calculation parameters
            'grp_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'Cell Grouping Significance Test', 'list': ['Wilcoxon Paired Test'],
                'def_val': 'Wilcoxon Paired Test', 'is_visible': False
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Filter Parameters', 'para_gui': RotationFilter, 'def_val': dcopy(r_filt_depth),
                'para_gui_var': {'rmv_fields': ['region_name']}
            },
            'plot_ratio': {'type': 'B', 'text': 'Plot Ratio Values', 'def_val': True},
            'plot_layer': {'type': 'B', 'text': 'Group Cells By Layer', 'def_val': True},
            'depth_type': {
                'type': 'L', 'text': 'Analysis Type', 'list': depth_type, 'def_val': depth_type[0],
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible plotting parameters
            'plot_all_expt': {'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'is_visible': False},
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
        }
        self.add_func(type='Depth-Based Analysis',
                      name='Depth Spiking Rate Comparison',
                      func='plot_depth_spiking',
                      para=para)

        # ====> Depth Spiking Rate Comparison (Multi-Sensory)
        para = {
            # calculation parameters
            't_phase_vis': {
                'gtype': 'C', 'text': 'Visual Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10
            },
            't_ofs_vis': {
                'gtype': 'C', 'text': 'Visual Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00
            },
            'use_full_vis': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Visual Phase', 'def_val': False,
                'link_para': [['t_phase_vis', True], ['t_ofs_vis', True]]
            },
            'vis_expt_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Visual Experiment Type', 'list': vis_type, 'def_val': vis_type_0
            },
            't_phase_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Duration (s)', 'def_val': t_phase, 'min_val': 0.10
            },
            't_ofs_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Offset (s)', 'def_val': t_ofs, 'min_val': 0.00
            },
            'use_full_rot': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Rotation Phase', 'def_val': True,
                'link_para': [['t_phase_rot', True], ['t_ofs_rot', True]]
            },

            # invisible calculation parameters
            'plot_all_expt': {'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'is_visible': False},
            'grp_stype': {
                'gtype': 'C', 'type': 'L', 'text': 'Cell Grouping Significance Test', 'list': ['Wilcoxon Paired Test'],
                'def_val': 'Wilcoxon Paired Test', 'is_visible': False
            },

            # plotting parameters
            'rot_filt': {
                'type': 'Sp', 'text': 'Filter Parameters', 'para_gui': RotationFilter, 'def_val': dcopy(r_filt_depth),
                'para_gui_var': {'rmv_fields': ['region_name']}
            },
            'plot_ratio': {'type': 'B', 'text': 'Plot Ratio Values', 'def_val': True},
            'plot_layer': {'type': 'B', 'text': 'Group Cells By Layer', 'def_val': True},
            'depth_type': {
                'type': 'L', 'text': 'Analysis Type', 'list': depth_type, 'def_val': depth_type[0],
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},

            # invisible plotting parameters
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
        }
        self.add_func(type='Depth-Based Analysis',
                      name='Depth Spiking Rate Comparison (Multi-Sensory)',
                      func='plot_depth_spiking_multi',
                      para=para)

        #########################################################
        ####    ROTATION DISCRIMINATION ANALYSIS FUNCTIONS   ####
        #########################################################

        # rotation LDA parameters
        rot_lda_para, rot_def_para = init_lda_para(data.discrim.dir)
        temp_lda_para, temp_def_para = init_lda_para(data.discrim.temp)
        indiv_lda_para, indiv_def_para  = init_lda_para(data.discrim.indiv)
        shuff_lda_para, shuff_def_para = init_lda_para(data.discrim.shuffle)
        part_lda_para, part_def_para = init_lda_para(data.discrim.part)
        filt_lda_para, filt_def_para = init_lda_para(data.discrim, 'filt', SubDiscriminationData('IndivFilt'))
        wght_lda_para, wght_def_para = init_lda_para(data.discrim, 'wght', SubDiscriminationData('LDAWeight'))

        # parameter lists
        err_type = ['IQR', 'SEM', 'Min/Max', 'None']
        decode_type = ['Condition'] + ['Dir ({0})'.format(x) for x in indiv_lda_para['comp_cond']]
        wght_plot_cond = wght_lda_para['comp_cond']
        wght_plot_err = ['IQR', 'SEM']
        acc_type = ['Bar + Bubbleplot', 'Violinplot + Swarmplot']
        wght_plot_layer = cfcn.det_uniq_channel_layers(data, wght_lda_para)

        # ====> Rotation Direction LDA
        para = {
            # calculation parameters
            'lda_para': {
                'gtype': 'C', 'type': 'Sp', 'text': 'LDA Solver Parameters', 'para_gui': LDASolverPara,
                'def_val': rot_lda_para
            },
            't_phase_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Duration (s)', 'min_val': 0.1,
                'def_val': cfcn.set_def_para(rot_def_para, 'tphase', t_phase)
            },
            't_ofs_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Offset (s)', 'min_val': 0.00,
                'def_val': cfcn.set_def_para(rot_def_para, 'tofs', t_ofs)
            },
            'use_full_rot': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Rotation Phase',
                'def_val': cfcn.set_def_para(rot_def_para, 'usefull', True),
                'link_para': [['t_phase_rot', True], ['t_ofs_rot', True]]
            },

            # plotting parameters
            'plot_transform': {'type': 'B', 'text': 'Plot LDA Transform Values', 'def_val': False},
            's_factor': {'text': 'Cell Marker Size Scale Factor', 'def_val': 2, 'min_val': 1},
            'plot_exp_name': {
                'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments',
                'is_enabled': has_multi_expt
            },
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': has_multi_expt,
                'link_para': [['plot_exp_name', True], ['plot_transform', True]], 'is_enabled': has_multi_expt
            },
            'acc_type': {
                'type': 'L', 'text': 'Accuracy Plot Type', 'list': acc_type, 'def_val': acc_type[0],
                'link_para': ['s_factor', 'Violinplot + Swarmplot']
            },
            'add_accuracy_trend': {
                'type': 'B', 'text': 'Add Accuracy Trendlines', 'def_val': has_multi_expt, 'is_enabled': has_multi_expt,
            },
            'output_stats': {
                'type': 'B', 'text': 'Show Statistics Table', 'def_val': has_multi_expt, 'is_enabled': has_multi_expt,
                'link_para': ['add_accuracy_trend', True]
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Direction LDA',
                      name='Rotation Direction LDA',
                      func='plot_rotation_dir_lda',
                      para=para)

        # ====> Temporal Duration/Offset LDA
        para = {
            # calculation parameters
            'lda_para': {
                'gtype': 'C', 'type': 'Sp', 'text': 'LDA Solver Parameters', 'para_gui': LDASolverPara,
                'def_val': temp_lda_para
            },
            'dt_phase': {
                'gtype': 'C', 'text': 'Phase Duration Step-Size (s)', 'min_val': 0.10, 'max_val': 1.0,
                'def_val': cfcn.set_def_para(temp_def_para, 'dt_phs', 0.5)
            },
            'dt_ofs': {
                'gtype': 'C', 'text': 'Phase Offset Step-Size (s)', 'min_val': 0.10, 'max_val': 1.0,
                'def_val': cfcn.set_def_para(temp_def_para, 'dt_ofs', 0.5)
            },
            't_phase_const': {
                'gtype': 'C', 'text': 'Constant Phase Duration (s)', 'min_val': 0.10, 'max_val': 1.0,
                'def_val': cfcn.set_def_para(temp_def_para, 'phs_const', 0.5)
            },

            # plotting parameters
            'use_stagger': {'type': 'B', 'text': 'Horizontally Separate Conditions', 'def_val': False},
            'plot_err': {'type': 'B', 'text': 'Show Error Shaded Region', 'def_val': True},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            'show_stats': {
                'type': 'B', 'text': 'Show Statistics Table', 'def_val': False, 'link_para': ['plot_grid', True]
            },
        }
        self.add_func(type='Direction LDA',
                      name='Temporal Duration/Offset LDA',
                      func='plot_temporal_lda',
                      para=para)

        # ====> Individual LDA
        para = {
            # calculation parameters
            'lda_para': {
                'gtype': 'C', 'type': 'Sp', 'text': 'LDA Solver Parameters', 'para_gui': LDASolverPara,
                'def_val': indiv_lda_para, 'para_gui_var': {'rmv_fields': ['y_acc_max', 'y_acc_min']},
                'para_reset': [['decode_type', self.reset_decode_type], ['dir_type_1', self.reset_dir_acc_type],
                               ['dir_type_2', self.reset_dir_acc_type]]
            },
            't_phase_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Duration (s)', 'min_val': 0.10,
                'def_val': cfcn.set_def_para(indiv_def_para, 'tphase', t_phase)
            },
            't_ofs_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Offset (s)', 'min_val': 0.00,
                'def_val': cfcn.set_def_para(indiv_def_para, 'tofs', t_ofs)
            },
            'use_full_rot': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Rotation Phase',
                'def_val': cfcn.set_def_para(indiv_def_para, 'usefull', True),
                'link_para': [['t_phase_rot', True], ['t_ofs_rot', True]]
            },

            # plotting parameters
            'plot_exp_name': {
                'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments',
                'is_enabled': has_multi_expt
            },
            'plot_all_expt': {
                'type': 'B', 'text': 'Analyse All Experiments', 'def_val': has_multi_expt,
                'link_para': ['plot_exp_name', True], 'is_enabled': has_multi_expt
            },
            'decode_type': {
                'type': 'L', 'text': 'Swarmplot Accuracy Type', 'list': decode_type, 'def_val': decode_type[0]
            },
            'dir_type_1': {
                'type': 'L', 'text': '1st Direction Trial Type', 'list': indiv_lda_para['comp_cond'],
                'def_val': indiv_lda_para['comp_cond'][0]
            },
            'dir_type_2': {
                'type': 'L', 'text': '2nd Direction Trial Type', 'list': indiv_lda_para['comp_cond'],
                'def_val': indiv_lda_para['comp_cond'][1]
            },
            'm_size': {'text': 'Maximum Scatterplot Marker Size', 'def_val': 60},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Direction LDA',
                      name='Individual LDA',
                      func='plot_individual_lda',
                      para=para)

        # ====> Shuffled LDA
        para = {
            # calculation parameters
            'lda_para': {
                'gtype': 'C', 'type': 'Sp', 'text': 'LDA Solver Parameters', 'para_gui': LDASolverPara,
                'para_reset': [['dir_type_1', self.reset_dir_acc_type], ['dir_type_2', self.reset_dir_acc_type]],
                'def_val': shuff_lda_para
            },
            't_phase_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Duration (s)', 'min_val': 0.10,
                'def_val': cfcn.set_def_para(shuff_def_para, 'tphase', t_phase)
            },
            't_ofs_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Offset (s)', 'min_val': 0.00,
                'def_val': cfcn.set_def_para(shuff_def_para, 'tofs', t_ofs)
            },
            'use_full_rot': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Rotation Phase',
                'def_val': cfcn.set_def_para(shuff_def_para, 'usefull', True),
                'link_para': [['t_phase_rot', True], ['t_ofs_rot', True]]
            },
            'n_shuffle': {
                'gtype': 'C', 'text': 'Trial Shuffle Count',
                'def_val': cfcn.set_def_para(shuff_def_para, 'nshuffle', 10)
            },

            # plotting parameters
            'i_cell_1': {'text': 'First Cell Index Number', 'def_val': 1, 'min_val': 1},
            'i_cell_2': {'text': 'Second Cell Index Number', 'def_val': 2, 'min_val': 1},
            'plot_exp_name': {
                'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'RotationExperiments',
            },
            'plot_corr': {
                'type': 'B', 'text': 'Plot Z-Score Correlations', 'def_val': False,
                'link_para': [['i_cell_1', False], ['i_cell_2', False], ['plot_exp_name', False]]
            },
            'dir_type_1': {
                'type': 'L', 'text': '1st Direction Trial Type', 'list': indiv_lda_para['comp_cond'],
                'def_val': indiv_lda_para['comp_cond'][0]
            },
            'dir_type_2': {
                'type': 'L', 'text': '2nd Direction Trial Type', 'list': indiv_lda_para['comp_cond'],
                'def_val': indiv_lda_para['comp_cond'][1]
            },
            'm_size': {'text': 'Scatterplot Marker Size', 'def_val': 60},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Direction LDA',
                      name='Shuffled LDA',
                      func='plot_shuffled_lda',
                      para=para)

        # ====> Pooled Neuron LDA
        para = {
            # calculation parameters
            'lda_para': {
                'gtype': 'C', 'type': 'Sp', 'text': 'LDA Solver Parameters', 'para_gui': LDASolverPara,
                'def_val': part_lda_para
            },
            't_phase_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Duration (s)', 'min_val': 0.10,
                'def_val': cfcn.set_def_para(part_def_para, 'tphase', t_phase)
            },
            't_ofs_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Offset (s)', 'min_val': 0.00,
                'def_val': cfcn.set_def_para(part_def_para, 'tofs', t_ofs)
            },
            'use_full_rot': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Rotation Phase',
                'def_val': cfcn.set_def_para(part_def_para, 'usefull', True),
                'link_para': [['t_phase_rot', True], ['t_ofs_rot', True]]
            },
            'n_shuffle': {
                'gtype': 'C', 'text': 'Partial Cell Shuffle Count',
                'def_val': cfcn.set_def_para(part_def_para, 'nshuffle', 10)
            },
            'pool_expt': {
                'gtype': 'C', 'type': 'B', 'text': 'Pool All Experiments',
                'def_val': cfcn.set_def_para(part_def_para, 'poolexpt', False),
            },

            # plotting parameters
            'err_type': {'type': 'L', 'text': 'Error Type', 'list': err_type, 'def_val': err_type[0]},
            'y_upper': {'text': 'Upper Accuracy Threshold Location', 'def_val': 95, 'min_val': 50, 'max_val': 100},
            'x_max': {'text': 'X-Axis Upper Limit', 'def_val': 100},
            'use_x_max': {
                'type': 'B', 'text': 'Use X-Axis Upper Limit', 'def_val': False, 'link_para': ['x_max', False]
            },
            'use_stagger': {'type': 'B', 'text': 'Horizontally Separate Conditions', 'def_val': False},
            'm_size': {'text': 'Plot Marker Size', 'def_val': 6},
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Direction LDA',
                      name='Pooled Neuron LDA',
                      func='plot_partial_lda',
                      para=para)

        # ====> Individual Cell Accuracy Filtered LDA
        para = {
            # calculation parameters
            'lda_para': {
                'gtype': 'C', 'type': 'Sp', 'text': 'LDA Solver Parameters', 'para_gui': LDASolverPara,
                'def_val': filt_lda_para, 'para_gui_var': {'rmv_fields': ['y_acc_max', 'y_acc_min']}
            },
            't_phase_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Duration (s)', 'min_val': 0.1,
                'def_val': cfcn.set_def_para(filt_def_para, 'tphase', t_phase)
            },
            't_ofs_rot': {
                'gtype': 'C', 'text': 'Rotation Phase Offset (s)', 'min_val': 0.00,
                'def_val': cfcn.set_def_para(filt_def_para, 'tofs', t_ofs)
            },
            'use_full_rot': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Full Rotation Phase',
                'def_val': cfcn.set_def_para(filt_def_para, 'usefull', True),
                'link_para': [['t_phase_rot', True], ['t_ofs_rot', True]]
            },
            'y_acc_min': {
                'gtype': 'C', 'text': 'Min Individual Cell Accuracy (%)',
                'def_val': cfcn.set_def_para(filt_def_para, 'yaccmn', 0)
            },
            'y_acc_max': {
                'gtype': 'C', 'text': 'Max Individual Cell Accuracy (%)',
                'def_val': cfcn.set_def_para(filt_def_para, 'yaccmx', 100)
            },

            # plotting parameters
            's_factor': {'text': 'Marker Size Scale Factor', 'def_val': 3},
            'acc_type': {
                'type': 'L', 'text': 'Accuracy Plot Type', 'list': acc_type, 'def_val': acc_type[0],
                'link_para': ['s_factor', 'Violinplot + Swarmplot']
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Direction LDA',
                      name='Individual Cell Accuracy Filtered LDA',
                      func='plot_acc_filt_lda',
                      para=para)

        # ====> LDA Group Weightings
        if wght_plot_layer is not None:
            para = {
                # calculation parameters
                'lda_para': {
                    'gtype': 'C', 'type': 'Sp', 'text': 'LDA Solver Parameters', 'para_gui': LDASolverPara,
                    'def_val': wght_lda_para, 'para_reset': [['plot_cond', self.reset_plot_cond_cl],
                                                             ['plot_layer', self.reset_plot_layer]],
                },
                't_phase_rot': {
                    'gtype': 'C', 'text': 'Rotation Phase Duration (s)', 'min_val': 0.1,
                    'def_val': cfcn.set_def_para(wght_def_para, 'tphase', t_phase)
                },
                't_ofs_rot': {
                    'gtype': 'C', 'text': 'Rotation Phase Offset (s)', 'min_val': 0.00,
                    'def_val': cfcn.set_def_para(wght_def_para, 'tofs', t_ofs)
                },
                'use_full_rot': {
                    'gtype': 'C', 'type': 'B', 'text': 'Use Full Rotation Phase',
                    'def_val': cfcn.set_def_para(wght_def_para, 'usefull', True),
                    'link_para': [['t_phase_rot', True], ['t_ofs_rot', True]]
                },

                # plotting parameters
                'error_type': {
                    'type': 'L', 'text': 'Signal Error Type', 'list': wght_plot_err, 'def_val': wght_plot_err[0]
                },
                'wght_thresh': {
                    'text': 'Coefficient Weight Threshold', 'def_val': 0.05, 'min_val': 0.0, 'max_val': 1.0
                },
                'plot_cond': {
                    'type': 'CL', 'text': 'Plot Conditions', 'list': wght_plot_cond,
                    'def_val': np.ones(len(wght_plot_cond), dtype=bool),
                },
                'plot_layer': {
                    'type': 'CL', 'text': 'Plot Channel Layers', 'list': wght_plot_layer,
                    'def_val': np.ones(len(wght_plot_layer), dtype=bool), 'other_para': '--- Select Layer Types ---'
                },
                'plot_comp': {'type': 'B', 'text': 'Show Coefficient/Depth Comparison', 'def_val': False,
                              'link_para': [['plot_cond', False], ['plot_layer', False]]},
                'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            }
            self.add_func(type='Direction LDA',
                          name='LDA Group Weightings',
                          func='plot_lda_weights',
                          para=para)

        ######################################
        ####    KINEMATIC LDA FUNCTIONS   ####
        ######################################

        # velocity LDA parameters
        spdacc_lda_para, spdacc_def_para = init_lda_para(data.discrim, 'spdacc', SubDiscriminationData('SpdAcc'))
        spdc_lda_para, spdc_def_para = init_lda_para(data.discrim, 'spdc', SubDiscriminationData('SpdComp'))
        spdcp_lda_para, spdcp_def_para = init_lda_para(data.discrim, 'spdcp', SubDiscriminationData('SpdCompPool'))
        spddir_lda_para, spddir_def_para = init_lda_para(data.discrim, 'spddir', SubDiscriminationData('SpdCompDir'))

        # combobox parameter lists
        plot_type_spd = ['Inter-Quartile Ranges', 'Individual Cell Responses']
        lda_plot_cond = spdcp_lda_para['comp_cond']
        spr_type = ['Experiment IQR Area', 'Individual Experiment Markers', 'No Markers']

        # determines the cell count checklist values
        n_cell_list = [str(x) for x in cfcn.get_pool_cell_counts(data, spdcp_lda_para)]

        # ====> Speed LDA Accuracy
        para = {
            # calculation parameters
            'lda_para': {
                'gtype': 'C', 'type': 'Sp', 'text': 'LDA Solver Parameters', 'para_gui': LDASolverPara,
                'def_val': spdacc_lda_para
            },

            'vel_bin': {
                'gtype': 'C', 'type': 'L', 'text': 'Speed Bin Size (deg/sec)', 'list': roc_vel_bin,
                'def_val': cfcn.set_def_para(spdacc_def_para, 'vel_bin', '5'),
                'para_reset': [['spd_x_rng', self.reset_spd_rng]]
            },
            'n_sample': {
                'gtype': 'C', 'text': 'Equal Timebin Resampling Count',
                'def_val': cfcn.set_def_para(spdacc_def_para, 'n_sample', 100)
            },
            'equal_time': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Equal Timebins',
                'def_val': cfcn.set_def_para(spdacc_def_para, 'equal_time', False),
                'link_para': ['n_sample', False]
            },

            # invisible parameters
            'freq_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Spike Frequency Type', 'list': ['All'], 'def_val': 'All',
                'is_visible': False
            },

            # plotting parameters
            's_factor': {'text': 'Cell Marker Size Scale Factor', 'def_val': 1, 'min_val': 0},
            'marker_type': {
                'type': 'L', 'text': 'Spread Plot Type', 'list': spr_type, 'def_val': spr_type[0],
                'link_para': [['s_factor', ['Experiment IQR Area', 'No Markers']]]
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Speed LDA',
                      name='Speed LDA Accuracy',
                      func='plot_speed_accuracy_lda',
                      para=para)

        # ====> Speed LDA Comparison (Individual Experiments)
        para = {
            # calculation parameters
            'lda_para': {
                'gtype': 'C', 'type': 'Sp', 'text': 'LDA Solver Parameters', 'para_gui': LDASolverPara,
                'def_val': spdc_lda_para
            },

            'spd_x_rng': {
                'gtype': 'C', 'type': 'L', 'text': 'Dependent Speed Range', 'list': sc_rng,
                'def_val': cfcn.set_def_para(spdc_def_para, 'spd_xrng', '0 to 5')
            },
            'vel_bin': {
                'gtype': 'C', 'type': 'L', 'text': 'Speed Bin Size (deg/sec)', 'list': roc_vel_bin,
                'def_val': cfcn.set_def_para(spdc_def_para, 'vel_bin', '5'),
                'para_reset': [['spd_x_rng', self.reset_spd_rng]]
            },
            'n_sample': {
                'gtype': 'C', 'text': 'Equal Timebin Resampling Count',
                'def_val': cfcn.set_def_para(spdc_def_para, 'n_sample', 100)
            },
            'equal_time': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Equal Timebins',
                'def_val': cfcn.set_def_para(spdc_def_para, 'equal_time', False),
                'link_para': ['n_sample', False]
            },

            # invisible parameters
            'freq_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Spike Frequency Type', 'list': ['All'], 'def_val': 'All',
                'is_visible': False
            },
            'pn_calc': {'gtype': 'C', 'text': 'Use Pos/Neg', 'def_val': False, 'is_visible': False},

            # plotting parameters
            'm_size': {'text': 'Maximum Cell Marker Size', 'def_val': 60},
            'show_cell_sz': {
                'type': 'B', 'text': 'Show Relative Cell Size', 'def_val': False,
            },
            'show_fit': {'type': 'B', 'text': 'Show Psychometric Fit', 'def_val': True},
            'sep_resp': {'type': 'B', 'text': 'Separate Condition Type Responses', 'def_val': False},
            'plot_type': {
                'type': 'L', 'text': 'Plot Type', 'list': plot_type_spd, 'def_val': plot_type_spd[0],
                'link_para': [['show_fit', 'Inter-Quartile Ranges'],
                              ['show_cell_sz', 'Inter-Quartile Ranges'],
                              ['m_size', 'Inter-Quartile Ranges']]
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Speed LDA',
                      name='Speed LDA Comparison (Individual Experiments)',
                      func='plot_speed_comp_lda',
                      para=para)

        # ====> Speed LDA Comparison (Pooled Experiments)
        para = {
            # calculation parameters
            'lda_para': {
                'gtype': 'C', 'type': 'Sp', 'text': 'LDA Solver Parameters', 'para_gui': LDASolverPara,
                'def_val': spdcp_lda_para, 'para_reset': [['plot_cond', self.reset_plot_cond_cl],
                                                          ['plot_cell', self.reset_plot_cell_cl]],
            },

            'spd_x_rng': {
                'gtype': 'C', 'type': 'L', 'text': 'Dependent Speed Range', 'list': sc_rng,
                'def_val': cfcn.set_def_para(spdcp_def_para, 'spd_xrng', '0 to 5')
            },
            'vel_bin': {
                'gtype': 'C', 'type': 'L', 'text': 'Speed Bin Size (deg/sec)', 'list': roc_vel_bin,
                'def_val': cfcn.set_def_para(spdcp_def_para, 'vel_bin', '5'),
                'para_reset': [['spd_x_rng', self.reset_spd_rng]]
            },
            'n_sample': {
                'gtype': 'C', 'text': 'Equal Timebin Resampling Count',
                'def_val': cfcn.set_def_para(spdcp_def_para, 'n_sample', 100)
            },
            'equal_time': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Equal Timebins',
                'def_val': cfcn.set_def_para(spdcp_def_para, 'equal_time', False),
                'link_para': ['n_sample', False]
            },
            'n_shuffle': {
                'gtype': 'C', 'text': 'Pooled Experiment Shuffle Count',
                'def_val': cfcn.set_def_para(spdcp_def_para, 'nshuffle', 5), 'min_val': 5
            },
            'pool_expt': {
                'gtype': 'C', 'type': 'B', 'text': 'Pool All Experiments',
                'def_val': cfcn.set_def_para(spdcp_def_para, 'poolexpt', False),
            },

            # invisible parameters
            'freq_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Spike Frequency Type', 'list': ['All'], 'def_val': 'All',
                'is_visible': False
            },
            'pn_calc': {'gtype': 'C', 'text': 'Use Pos/Neg', 'def_val': False, 'is_visible': False},

            # plotting parameters
            'm_size': {'text': 'Mean Value Marker Size', 'def_val': 240},
            'plot_markers': {
                'type': 'B', 'text': 'Plot Mean Value Markers', 'def_val': True, 'link_para': ['m_size', False]
            },
            'plot_cond': {
                'type': 'CL', 'text': 'Plot Conditions', 'list': lda_plot_cond,
                'def_val': np.ones(len(lda_plot_cond), dtype=bool),
            },
            'plot_cell': {
                'type': 'CL', 'text': 'Plot Cell Counts', 'list': n_cell_list,
                'def_val': np.ones(len(n_cell_list), dtype=bool), 'other_para': '--- Select Plot Cell Counts ---'
            },
            'plot_para': {
                'type': 'B', 'text': 'Plot Fit Parameters', 'def_val': False, 'link_para': ['plot_cell', True],
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
        }
        self.add_func(type='Speed LDA',
                      name='Speed LDA Comparison (Pooled Experiments)',
                      func='plot_pooled_speed_comp_lda',
                      para=para)

        # ====> Speed Direction Discrimination LDA
        para = {
            # calculation parameters
            'lda_para': {
                'gtype': 'C', 'type': 'Sp', 'text': 'LDA Solver Parameters', 'para_gui': LDASolverPara,
                'def_val': spdacc_lda_para
            },

            'vel_bin': {
                'gtype': 'C', 'type': 'L', 'text': 'Speed Bin Size (deg/sec)', 'list': roc_vel_bin,
                'def_val': cfcn.set_def_para(spddir_def_para, 'vel_bin', '5'),
                'para_reset': [['spd_x_rng', self.reset_spd_rng]]
            },
            'n_sample': {
                'gtype': 'C', 'text': 'Equal Timebin Resampling Count',
                'def_val': cfcn.set_def_para(spddir_def_para, 'n_sample', 100)
            },
            'equal_time': {
                'gtype': 'C', 'type': 'B', 'text': 'Use Equal Timebins',
                'def_val': cfcn.set_def_para(spddir_def_para, 'equal_time', False),
                'link_para': ['n_sample', False]
            },

            # invisible parameters
            'freq_type': {
                'gtype': 'C', 'type': 'L', 'text': 'Spike Frequency Type', 'list': ['Increasing'],
                'def_val': 'Increasing', 'is_visible': False
            },

            # plotting parameters
            'use_stagger': {'type': 'B', 'text': 'Horizontally Separate Conditions', 'def_val': False},
            's_factor': {'text': 'Cell Marker Size Scale Factor', 'def_val': 1, 'min_val': 0},
            'marker_type': {
                'type': 'L', 'text': 'Spread Plot Type', 'list': spr_type, 'def_val': spr_type[0],
                'link_para': [['s_factor', ['Experiment IQR Area', 'No Markers']]]
            },
            'plot_grid': {'type': 'B', 'text': 'Show Axes Grid', 'def_val': False},
            'show_stats': {
                'type': 'B', 'text': 'Show Statistics Table', 'def_val': False, 'link_para': ['plot_grid', True]
            },
        }
        self.add_func(type='Speed LDA',
                      name='Velocity Direction Discrimination LDA',
                      func='plot_speed_dir_lda',
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

        ######################################
        ####    MISCELLANEOUS FUNCTIONS   ####
        ######################################

        # sets the default class parameters
        sf_def_para = init_def_class_para(data, 'spikedf', SpikingFreqData())
        rot_filt_df = dcopy(cf.init_rotation_filter_data(False))

        # ====> Velocity Multilinear Regression Dataframe Output
        para = {
            # calculation parameters
            'rot_filt': {
                'gtype': 'C', 'type': 'Sp', 'text': 'Filter Parameters', 'para_gui': RotationFilter,
                'def_val': cfcn.set_def_para(sf_def_para, 'rot_filt', rot_filt_df)
            },

            'bin_sz': {
                'gtype': 'C', 'text': 'Time Bin Size (ms)', 'min_val': 10, 'max_val': 1000,
                'def_val': cfcn.set_def_para(sf_def_para, 'bin_sz', 100)
            },
            't_over': {
                'gtype': 'C', 'text': 'Bin Overlap Duration (ms)', 'min_val': 0, 'max_val': 1000,
                'def_val': cfcn.set_def_para(sf_def_para, 't_over', 50)
            },

            # invisible parameters
            'out_name': {
                'type': 'T', 'text': 'Output File Name',
                'def_val': cfcn.set_def_para(sf_def_para, 'out_name', 'Spike_Freq_Dataframe')
            },
            'plot_all_expt': {'type': 'B', 'text': 'Analyse All Experiments', 'def_val': True, 'is_visible': False},
            'plot_scope': {
                'type': 'L', 'text': 'Analysis Scope', 'list': scope_txt, 'def_val': scope_txt[1], 'is_visible': False
            },
        }
        self.add_func(type='Miscellaneous Functions',
                      name='Velocity Multilinear Regression Dataframe Output',
                      func='output_spiking_freq_dataframe',
                      para=para)

        ##############################
        ####    OTHER FUNCTIONS   ####
        ##############################

        # ====> Cross Cross-Correlogram
        para = {
            # plotting parameters
            'exp_name': {'type': 'L', 'text': 'Experiment', 'def_val': None, 'list': 'Experiments'},
            'i_ref': {'text': 'Reference Cluster Index', 'def_val': 1, 'min_val': 1},
            'i_comp': {'text': 'Comparison Cluster Indices', 'def_val': 1, 'min_val': 1, 'is_list': True},
            'plot_all': {'type': 'B', 'text': 'Plot All Clusters', 'def_val': True, 'link_para':['i_comp', True]},
            'm_size': {'text': 'Scatterplot Markersize', 'def_val': 30},
            'plot_type': {
                'type': 'L', 'text': 'Plot Type', 'def_val': 'bar', 'list': ['bar', 'scatterplot'],
                'link_para': [['m_size', 'bar']]
            },
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

    ##################################################
    ####    OBJECT CREATION/DELETION FUNCTIONS    ####
    ##################################################

    def reset_curr_para_fields(self):
        '''

        :return:
        '''

        self.curr_fcn = None
        self.curr_para = None
        self.prev_fcn = None
        self.prev_calc_para = None
        self.prev_plot_para = None
        self.is_updating = False

    def set_pool_worker(self, pool):
        '''

        :param pool:
        :return:
        '''

        # sets the pool worker
        self.pool = pool

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

    def update_comp_expts(self):
        '''

        :return:
        '''

        # retrieves the comparison fixed file names
        p_str = ['calc_comp', 'plot_comp']
        ex_name = [self.det_comp_expt_names(True), self.det_comp_expt_names(False)]

        # updates the parameters for each function associated with the comparison experiment file names
        for fd in self.details['Cluster Matching']:
            for i_pp, pp in enumerate(p_str):
                if pp in fd['para']:
                    # updates the parameter list and enabled values
                    fd['para'][pp]['list'] = dcopy(ex_name[i_pp])
                    fd['para'][pp]['is_enabled'] = len(ex_name[i_pp]) > 1

                    # updates the default value (if not in the current list)
                    if fd['para'][pp]['def_val'] not in ex_name[i_pp]:
                        fd['para'][pp]['def_val'] = dcopy(ex_name[i_pp][0])

                    # if the current experiment
                    if fd['name'] == self.curr_fcn:
                        # retrieves the list object
                        h_list = self.find_obj_handle([QComboBox], pp)
                        if len(h_list):
                            # retrieves the current index
                            i_sel0 = h_list[0].currentIndex()

                            # removes the existing items
                            for i in range(h_list[0].count()):
                                h_list[0].removeItem(0)

                            # adds the new range
                            for txt in ex_name[i_pp]:
                                h_list[0].addItem(txt)

                            # resets the selected index (if the indices exceed the list count)
                            if i_sel0 > len(ex_name[i_pp]):
                                i_sel0 = 0

                            # resets the associated parameter value
                            self.curr_para[pp] = ex_name[i_pp][i_sel0]
                            h_list[0].setCurrentIndex(i_sel0)
                            h_list[0].setEnabled(len(ex_name[i_pp]) > 1)

    def update_extern_expts(self, h_combo, fcn_str):
        '''

        :return:
        '''

        # initialisations
        data = self.get_data_fcn()
        h_combo_list = [h_combo.itemText(i) for i in range(h_combo.count())]

        # determines if the function type is in the function dropdown list. if not, then add it in
        if fcn_str not in h_combo_list:
            # appends the function list the freely moving cell types
            h_combo_list.insert(np.where(func_types == fcn_str)[0][0], fcn_str)
            i_sel0 = h_combo.currentIndex()

            # removes the function types
            for i in range(h_combo.count() - 1):
                h_combo.removeItem(h_combo.count() - 1)

            # re-adds the function types
            for l_str in h_combo_list[1:]:
                h_combo.addItem(l_str)

            # resets the current index
            h_combo.setCurrentIndex(i_sel0)

        #
        if fcn_str == 'Freely Moving Cell Types':
            # retrieves the external data field and corresponding parameter name
            f_data, pp = data.externd.free_data, 'free_exp_name'

            # retrieves the function data sub-dictionary
            fcn_d = next(d for d in self.details['Cluster Matching'] if 'Fixed/Freely Moving Spiking' in d['name'])
            if 'No Fixed/Free' in fcn_d['para'][pp]['def_val']:
                # if there are experiments set, then reset the default value
                fcn_d['para'][pp]['def_val'] = f_data.exp_name[0]

        elif fcn_str == 'Eye Tracking':
            # retrieves the external data field and corresponding parameter name
            f_data, pp = data.externd.eye_track, 'etrack_exp_name'

            # retrieves the function data sub-dictionary
            if 'Eye Tracking' in self.details.keys():
                # retrieves the sub-dictionary if the eye-tracking data field has been setup
                fcn_d = next(d for d in self.details['Eye Tracking'] if 'Eye Movement Event Signals' in d['name'])
            else:
                # otherwise, set an empty sub-dictionary
                fcn_d = None

        if fcn_d is not None:
            # updates the functions parameter/default value fields
            fcn_d['para'][pp]['list'] = f_data.exp_name

            # if the currently selected experiment is the function corresponding to the loaded data, then update the
            # list fields with the new experiment names
            if fcn_d['name'] == self.curr_fcn:
                # retrieves the list object
                h_list = self.find_obj_handle([QComboBox], pp)[0]
                i_sel0 = h_list.currentIndex()

                # removes the existing items
                for i in range(h_list.count()):
                    h_list.removeItem(0)

                # adds the new range
                for txt in f_data.exp_name:
                    h_list.addItem(txt)

                # updates the selected index
                h_list.setCurrentIndex(i_sel0)
            else:
                # sets the initial selection value to zero
                i_sel0 = 0

            # resets the matching indices
            if fcn_str == 'Freely Moving Cell Types':
                self.reset_matched_index('ff_cluster', f_data.exp_name[i_sel0])

        # re-initalises all the function data
        self.init_all_func()

    ##################################################
    ####    OBJECT CREATION/DELETION FUNCTIONS    ####
    ##################################################

    def init_para_groupbox(self, h_grp_para):
        '''

        :param h_grp_para:
        :return:
        '''

        # initialisations
        tab_name, tab_hght = ['Calculation Parameters', 'Plotting Parameters'], 351

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
        self.grp_para_tabs = cf.create_tab(h_grp_para, QRect(5, 55, self.grp_wid, tab_hght), None,
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
                elif not fcn_para[p_name]['is_visible']:
                    continue

                # sets the top of the object
                i_count += 1

                # creates the parameter object based on the type
                if fcn_para[p_name]['type'] == 'N':
                    self.create_number_para(h_layout, p_name, fcn_para[p_name])
                elif fcn_para[p_name]['type'] == 'T':
                    self.create_text_para(h_layout, p_name, fcn_para[p_name])
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

    def create_text_para(self, h_layout, p_name, fcn_para):
        '''

        :param h_layout:
        :param p_name:
        :param fcn_para:
        :return:
        '''

        # initialisations
        para_text = '{0}: '.format(fcn_para['text'])
        def_val = self.curr_para[p_name]

        # creates the label/editbox objects
        h_lbl = cf.create_label(None, txt_font_bold, para_text, align='right')
        h_txt = cf.create_edit(None, txt_font, def_val, align='centre', name=p_name)
        h_lbl.setAlignment(Qt.AlignVCenter)

        # sets the callback function
        cb_fcn = functools.partial(self.update_text_para, h_txt, p_name)
        h_txt.editingFinished.connect(cb_fcn)

        # if the visiblity flag is set to false, then hide the objects
        if not fcn_para['is_visible']:
            h_lbl.setVisible(False)
            h_txt.setVisible(False)
        elif not fcn_para['is_enabled']:
            h_lbl.setEnabled(False)
            h_txt.setEnabled(False)

        # adds the widgets to the layout
        h_layout.addRow(h_lbl, h_txt)

    def create_boolean_para(self, h_layout, p_name, fcn_para):
        '''

        :return:
        '''

        # initialisations
        link_para, para_reset = fcn_para['link_para'], fcn_para['para_reset']
        cb_fcn = functools.partial(self.update_bool_para, p_name, link_para, para_reset)

        # creates the object
        h_chk = cf.create_checkbox(None, txt_font_bold, fcn_para['text'], name=p_name, cb_fcn=cb_fcn)

        # connects the callback function to the checkbox
        h_chk.setChecked(self.curr_para[p_name])
        if fcn_para['link_para'] is not None:
            self.update_bool_para(p_name, link_para, para_reset, self.curr_para[p_name])

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
        link_para, list_txt = fcn_para['link_para'], fcn_para['list']
        para_reset, reset_func = fcn_para['para_reset'], fcn_para['reset_func']

        # resets the list text if a special type
        if list_txt == 'Experiments':
            # case is the experiment names
            list_txt = self.exp_name

        elif list_txt == 'RotationExperiments':
            # case is the rotation experiment names
            is_rot_expt = cf.det_valid_rotation_expt(self.get_data_fcn())
            list_txt = [x for x, y in zip(self.exp_name, is_rot_expt) if y]

        elif list_txt == 'RotationExperimentUD':
            # case is the uniform drifting experiment names
            is_ud_expt = cf.det_valid_rotation_expt(self.get_data_fcn(), is_ud=True)
            list_txt = [x for x, y in zip(self.exp_name, is_ud_expt) if y]

        elif list_txt == 'RotationExperimentMD':
            # case is the rotation experiment names
            t_type = ['Black', 'Uniform', 'LandmarkLeft', 'LandmarkRight']
            is_rot_expt = cf.det_valid_rotation_expt(self.get_data_fcn(), t_type=t_type)
            list_txt = [x for x, y in zip(self.exp_name, is_rot_expt) if y]

        elif list_txt == 'EyeTrackExperiments':
            # case is the eye tracking experiment names
            list_txt = self.get_data_fcn().externd.eye_track.exp_name

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

        # updates the list parameter (if the link parameter field is not None)
        if link_para is not None:
            self.update_list_para(p_name, list_txt, link_para, None, True, index=i_ind)

        # runs the reset function (if not None)
        if reset_func is not None:
            reset_func(h_list, h_lbl, self.get_data_fcn())

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

        #
        first_line = fcn_para['other_para']
        if first_line is None:
            first_line = '--- Select Trial Conditions ---'

        # creates the object
        h_lbl = cf.create_label(None, txt_font_bold, para_text, align='right')
        h_chklist = cf.create_checkcombo(None, txt_font, list_txt, name=p_name, first_line=first_line)
        h_lbl.setAlignment(Qt.AlignVCenter)

        #
        cb_fcn = functools.partial(self.update_checklist_para, p_name, h_chklist=h_chklist)
        h_chklist.view().pressed.connect(cb_fcn)

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

        :param p_name:
        :param fcn_para:
        :return:
        '''

        #
        if 'para_gui_var' not in fcn_para:
            para_gui_var = None
        else:
            para_gui_var = fcn_para['para_gui_var']

        # runs the
        para_reset = fcn_para['para_reset']
        data, init_data = self.get_data_fcn(), self.curr_para[p_name]
        h_sp = fcn_para['para_gui'](self, init_data=init_data, other_var=para_gui_var)

        # determines if the gui was updated correctly
        exp_info = h_sp.get_info()
        if h_sp.is_ok:
            # updates the current parameter value
            self.curr_para[p_name] = exp_info

            # resets the parameters based on the
            if para_reset is not None:
                # flag that the parameters are updating
                self.is_updating = True

                # runs the parameter reset functions
                for pr in para_reset:
                    pr[1](exp_info, pr[0])

                # flag that the parameters are finished updating
                self.is_updating = False

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

    def update_text_para(self, h_txt, p_name):
        '''

        :param h_txt:
        :param p_name:
        :return:
        '''

        if self.is_updating:
            return
        else:
            self.is_updating = True

        # retrieves the string from the editbox
        self.curr_para[p_name] = h_txt.text()

        # resets the update flag so another change can be made
        self.is_updating = False

    def update_bool_para(self, p_name, link_para, para_reset, state):
        '''

        :return:
        '''

        if self.is_updating:
            return

        # sets the enabled properties for the linked parameters (if they exist)
        if link_para is not None:
            # ensures the link parameters are a list of lists
            if not isinstance(link_para[0], list):
                link_para = [link_para]

            for lp in link_para:
                # retrieves the object handle of the parameter (continue if not found)
                h_obj = self.find_obj_handle([QLineEdit, QCheckBox, QComboBox], lp[0])
                if len(h_obj) == 0:
                    continue

                h_obj[0].setEnabled(bool(state) != lp[1])

        # resets the parameters based on the
        if para_reset is not None:
            # flag that the parameters are updating
            self.is_updating = True

            # runs the parameter reset functions
            for pr in para_reset:
                pr[1](pr[0], state)

            # flag that the parameters are finished updating
            self.is_updating = False

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
                # retrieves the object handle of the parameter (continue if not found)
                h_obj = self.find_obj_handle([QLineEdit, QCheckBox, QComboBox, QPushButton], lp[0])
                if len(h_obj) == 0:
                    continue

                # self.get_para_details(lp[0])
                if isinstance(lp[1], list):
                    is_enabled = p_list[index] not in lp[1]
                else:
                    is_enabled = p_list[index] != lp[1]

                # updates the enabled properties of the object
                h_obj[0].setEnabled(is_enabled)

                # checks to see if the link parameter is a checkbox and is being disabled
                if isinstance(h_obj[0],QCheckBox) and (not is_enabled):
                    # if so, then uncheck the checkbox
                    self.is_updating = True
                    h_obj[0].setCheckState(False)
                    self.is_updating = False

                    # determines if the checkbox has any other link parameters
                    para_d = self.get_para_details(lp[0])
                    if para_d['link_para'] is not None:
                        # if so, then update the link parameter of the link parameter
                        if len(para_d['link_para']) == 2:
                            self.update_bool_para(lp[0], para_d['link_para'], None, False)

        # resets the parameters based on the
        if para_reset is not None:
            # flag that the parameters are updating
            self.is_updating = True

            # runs the parameter reset functions
            for pr in para_reset:
                pr[1](pr[0], p_list[index])

            # flag that the parameters are finished updating
            self.is_updating = False

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

    def reset_spd_rng(self, p_name, dv0):
        '''

        :param p_name:
        :param dv:
        :return:
        '''

        # determines the plot function that is currently selected
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

    def reset_plot_index(self, p_name, dv0):
        '''

        :param p_name:
        :param dv0:
        :return:
        '''

        # object handles
        h_vb, h_uv = self.find_obj_handle([QComboBox], 'vel_bin'), self.find_obj_handle([QCheckBox], 'use_vel')
        if len(h_vb) and len(h_uv):
            h_vb, h_uv = h_vb[0], h_uv[0]
        else:
            return

        # determines the plot function that is currently selected
        d_grp = self.details[self.get_plot_grp_fcn()]
        i_grp = next(i for i in range(len(d_grp)) if d_grp[i]['name'] == self.get_plot_fcn())

        # parameters
        v_rng, dv = 80, int(h_vb.currentText())
        if not isinstance(p_name, list):
            p_name = [p_name]

        # sets the new velocity range list values
        if h_uv.checkState() == Qt.Checked:
            nw_rng = ['{0} to {1}'.format(i * dv - v_rng, (i + 1) * dv - v_rng) for i in range(int(2 * v_rng / dv))]
        else:
            nw_rng = ['{0} to {1}'.format(i * dv, (i + 1) * dv) for i in range(int(v_rng / dv))]

        # resets the list values for each associated parameter
        nw_rng += ['All Bins']
        for pp in p_name:
            # retrieves the list object
            h_list = self.find_obj_handle([QComboBox], pp)
            if len(h_list):
                # removes the existing items
                for i in range(h_list[0].count()):
                    h_list[0].removeItem(0)

                # adds the new range
                for txt in nw_rng:
                    h_list[0].addItem(txt)

                i_sel = h_list[0].currentIndex()
                if (i_sel + 1) >= len(nw_rng):
                    i_sel = 0

                # resets the associated parameter value
                self.curr_para[pp] = nw_rng[i_sel]
                d_grp[i_grp]['para'][pp]['list'] = nw_rng

    def reset_grp_type(self, p_name, g_type):
        '''

        :return:
        '''

        # determines the cell grouping type that was selected
        gtype_list = ['Direction Selectivity', 'Rotation/Visual DS', 'Congruency']
        ig_type = gtype_list.index(g_type)

        # determines the plot function that is currently selected
        d_grp = self.details[self.get_plot_grp_fcn()]
        i_grp = next(i for i in range(len(d_grp)) if d_grp[i]['name'] == self.get_plot_fcn())

        # sets the new list values based on the selected type
        if gtype_list[ig_type] == 'Direction Selectivity':
            # case is the direction selectivity
            nw_list = ['MS/DS', 'MS/Not DS', 'Not MS', 'All Cells']
        elif gtype_list[ig_type] == 'Rotation/Visual DS':
            # case is the preferred direction
            nw_list = ['None', 'Rotation', 'Visual', 'Both', 'All Cells']
        else:
            # case is the congruency
            nw_list = ['Congruent', 'Incongruent', 'All Cells']

        # retrieves the list object
        h_list = self.find_obj_handle([QComboBox], p_name)
        if len(h_list):
            # removes the existing items
            for i in range(h_list[0].count()):
                h_list[0].removeItem(0)

            # adds the new range
            for txt in nw_list:
                h_list[0].addItem(txt)

            # resets the associated parameter values
            self.curr_para[p_name] = nw_list[h_list[0].currentIndex()]
            d_grp[i_grp]['para'][p_name]['list'] = nw_list

    def reset_cell_types(self, h_list, h_lbl, data):
        '''

        :param data:
        :return:
        '''

        # only enable the parameter if the cell classification has been performed
        h_lbl.setEnabled(data.classify.is_set)
        h_list.setEnabled(data.classify.is_set)

    def reset_decode_type(self, exp_info, p_name):
        '''

        :param exp_info:
        :param p_name:
        :return:
        '''

        # retrieves the list object corresponding to the parameter
        h_list = self.find_obj_handle([QComboBox], p_name)[0]
        curr_txt = [h_list.itemText(i) for i in range(h_list.count())]

        # checks if there is a change in the comparison conditions
        nw_txt = ['Condition'] + ['Dir ({0})'.format(x) for x in dcopy(exp_info['comp_cond'])]
        if set(nw_txt) != set(curr_txt):
            # determines the plot function that is currently selected
            d_grp = self.details[self.get_plot_grp_fcn()]
            i_grp = next(i for i in range(len(d_grp)) if d_grp[i]['name'] == self.get_plot_fcn())

            # sets the list selection index (if current selection is gone, then set to zero)
            i_sel = next((i for i in range(len(nw_txt)) if nw_txt[i] == h_list.currentText()), 0)

            # removes the existing items
            for i in range(h_list.count()):
                h_list.removeItem(0)

            # adds the new range
            for txt in nw_txt:
                h_list.addItem(txt)

            # resets the associated parameter value
            h_list.setCurrentIndex(i_sel)
            self.curr_para[p_name] = nw_txt[i_sel]
            d_grp[i_grp]['para'][p_name]['list'] = nw_txt

    def reset_dir_acc_type(self, exp_info, p_name):
        '''

        :param exp_info:
        :param p_name:
        :return:
        '''

        # retrieves the list object corresponding to the parameter
        h_list = self.find_obj_handle([QComboBox], p_name)[0]
        curr_txt = [h_list.itemText(i) for i in range(h_list.count())]

        # checks if there is a change in the comparison conditions
        nw_txt = dcopy(exp_info['comp_cond'])
        if set(nw_txt) != set(curr_txt):
            # determines the plot function that is currently selected
            d_grp = self.details[self.get_plot_grp_fcn()]
            i_grp = next(i for i in range(len(d_grp)) if d_grp[i]['name'] == self.get_plot_fcn())

            # sets the list selection index (if current selection is gone, then set to zero)
            i_sel0 = int(p_name[-1]) - 1
            i_sel = next((i for i in range(len(nw_txt)) if nw_txt[i] == h_list.currentText()), i_sel0)

            # removes the existing items
            for i in range(h_list.count()):
                h_list.removeItem(0)

            # adds the new range
            for txt in nw_txt:
                h_list.addItem(txt)

            # resets the associated parameter value
            h_list.setCurrentIndex(i_sel)
            self.curr_para[p_name] = nw_txt[i_sel]
            d_grp[i_grp]['para'][p_name]['list'] = nw_txt

    def reset_plot_cond(self, exp_info, p_name):
        '''

        :param exp_info:
        :param p_name:
        :return:
        '''

        # retrieves the list object corresponding to the parameter
        h_list = self.find_obj_handle([QComboBox], p_name)[0]
        curr_txt = [h_list.itemText(i) for i in range(h_list.count())]

        # checks if there is a change in the comparison conditions
        nw_txt = dcopy(exp_info['comp_cond'])
        if set(nw_txt) != set(curr_txt):
            # determines the plot function that is currently selected
            d_grp = self.details[self.get_plot_grp_fcn()]
            i_grp = next(i for i in range(len(d_grp)) if d_grp[i]['name'] == self.get_plot_fcn())

            # sets the list selection index (if current selection is gone, then set to zero)
            i_sel = next((i for i in range(len(nw_txt)) if nw_txt[i] == h_list.currentText()), 0)

            # removes the existing items
            for i in range(h_list.count()):
                h_list.removeItem(0)

            # adds the new range
            for txt in nw_txt:
                h_list.addItem(txt)

            # resets the associated parameter value
            h_list.setCurrentIndex(i_sel)
            self.curr_para[p_name] = nw_txt[i_sel]
            d_grp[i_grp]['para'][p_name]['list'] = nw_txt

    def reset_plot_cond_cl(self, exp_info, p_name):
        '''

        :param exp_info:
        :param p_name:
        :return:
        '''

        # retrieves the list object corresponding to the parameter
        h_list = self.find_obj_handle([QComboBox], p_name)[0]
        curr_txt = [h_list.itemText(i) for i in range(h_list.count())]

        # retrieves the new text list
        nw_txt = dcopy(exp_info['comp_cond'])

        # checks if there is a change in the comparison conditions
        if set(nw_txt) != set(curr_txt[1:]):
            # determines the plot function that is currently selected
            d_grp = self.details[self.get_plot_grp_fcn()]
            i_grp = next(i for i in range(len(d_grp)) if d_grp[i]['name'] == self.get_plot_fcn())

            # sets the list selection index (if current selection is gone, then set to zero)
            is_match = [x in self.curr_para[p_name] for x in nw_txt]

            # removes the existing items
            for i in range(1, h_list.count()):
                h_list.removeItem(1)

            # adds the new range
            for i, txt in enumerate(nw_txt):
                h_list.addItem(txt, True)
                nw_item = h_list.model().item(i + 1)
                nw_item.setCheckState(Qt.Checked if is_match[i] else Qt.Unchecked)

            # resets the associated parameter value
            h_list.setCurrentIndex(0)
            self.curr_para[p_name] = list(np.array(nw_txt)[is_match])
            d_grp[i_grp]['para'][p_name]['list'] = nw_txt

    def reset_plot_layer(self, exp_info, p_name):
        '''

        :param exp_info:
        :param p_name:
        :return:
        '''

        # retrieves the list object corresponding to the parameter
        h_list = self.find_obj_handle([QComboBox], p_name)[0]
        curr_txt = [h_list.itemText(i) for i in range(h_list.count())]

        # retrieves the new text list
        nw_txt = cfcn.det_uniq_channel_layers(self.get_data_fcn(), exp_info)

        # checks if there is a change in the comparison conditions
        if set(nw_txt) != set(curr_txt[1:]):
            # determines the plot function that is currently selected
            d_grp = self.details[self.get_plot_grp_fcn()]
            i_grp = next(i for i in range(len(d_grp)) if d_grp[i]['name'] == self.get_plot_fcn())

            # sets the list selection index (if current selection is gone, then set to zero)
            is_match = [x in self.curr_para[p_name] for x in nw_txt]

            # removes the existing items
            for i in range(1, h_list.count()):
                h_list.removeItem(1)

            # adds the new range
            for i, txt in enumerate(nw_txt):
                h_list.addItem(txt, True)
                nw_item = h_list.model().item(i + 1)
                nw_item.setCheckState(Qt.Checked if is_match[i] else Qt.Unchecked)

            # resets the associated parameter value
            h_list.setCurrentIndex(0)
            self.curr_para[p_name] = list(np.array(nw_txt)[is_match])
            d_grp[i_grp]['para'][p_name]['list'] = nw_txt

    def reset_plot_cell_cl(self, exp_info, p_name):
        '''

        :param exp_info:
        :param p_name:
        :return:
        '''

        # retrieves the list object corresponding to the parameter
        h_list = self.find_obj_handle([QComboBox], p_name)[0]
        curr_txt = [h_list.itemText(i) for i in range(h_list.count())]

        # retrieves the new text list
        nw_txt = [str(x) for x in cfcn.get_pool_cell_counts(self.get_data_fcn(), exp_info)]

        # checks if there is a change in the comparison conditions
        if set(nw_txt) != set(curr_txt[1:]):
            # determines the plot function that is currently selected
            d_grp = self.details[self.get_plot_grp_fcn()]
            i_grp = next(i for i in range(len(d_grp)) if d_grp[i]['name'] == self.get_plot_fcn())

            # sets the list selection index (if current selection is gone, then set to zero)
            is_match = [x in self.curr_para[p_name] for x in nw_txt]
            is_match[-1] = True

            # removes the existing items
            for i in range(1, h_list.count()):
                h_list.removeItem(1)

            # adds the new range
            for i, txt in enumerate(nw_txt):
                h_list.addItem(txt, True)
                nw_item = h_list.model().item(i + 1)
                nw_item.setCheckState(Qt.Checked if is_match[i] else Qt.Unchecked)

            # resets the associated parameter value
            h_list.setCurrentIndex(0)
            self.curr_para[p_name] = list(np.array(nw_txt)[is_match])
            d_grp[i_grp]['para'][p_name]['list'] = nw_txt

    def reset_matched_index(self, p_name, exp_name):
        '''

        :param p_name:
        :param exp_name:
        :return:
        '''

        # initialisations
        data = self.get_data_fcn()

        # retrieves the list object corresponding to the parameter
        fcn_d = next(d for d in self.details['Cluster Matching'] if 'Fixed/Freely Moving Spiking' in d['name'])

        # retrieves the experiment names and the currently selected index
        exp_name = fcn_d['para']['free_exp_name']['list']
        h_list_exp = self.find_obj_handle([QComboBox], 'free_exp_name')
        if len(h_list_exp):
            i_sel_exp = h_list_exp[0].currentIndex()
            if i_sel_exp < 0:
                i_sel_exp = 0
        else:
            i_sel_exp = exp_name.index(fcn_d['para']['free_exp_name']['def_val'])

        # updates the parameter with the new values
        nw_txt = cfcn.get_matching_fix_free_strings(data, exp_name[i_sel_exp])
        fcn_d['para'][p_name]['list'] = nw_txt
        fcn_d['para'][p_name]['def_val'] = nw_txt[0]

        # resets the current parameter value
        self.curr_para[p_name] = nw_txt[0]

        # updates the experiment list items (if currently visible)
        h_list = self.find_obj_handle([QComboBox], p_name)
        if len(h_list):
            # removes the existing items
            for i in range(h_list[0].count()):
                h_list[0].removeItem(0)

            # adds the new range
            for txt in nw_txt:
                h_list[0].addItem(txt)

            # updates the selected index
            h_list[0].setCurrentIndex(0)

    def reset_cluster_para(self, p_name=None, exp_name=None):
        '''

        :param p_name:
        :param exp_name:
        :return:
        '''

        # initialisations
        data = self.get_data_fcn()
        h_list0 = self.find_obj_handle([QComboBox], 'calc_comp')

        if len(h_list0) == 0:
            return
        else:
            h_list = h_list0[0]

        # retrieves the list object corresponding to the parameter
        fcn_d = next(d for d in self.details['Cluster Matching'] if 'Fixed/Free Cluster Matching' in d['name'])
        para_new = cfcn.init_clust_para(data.comp, data.comp.data[h_list.currentIndex()].free_name)

        for pf in para_new:
            # continue if the calculation combination parameter
            if pf == 'calc_comp':
                continue

            # retrieves the parameter object
            h_obj = self.find_obj_handle([QLineEdit], pf)[0]
            h_obj.setText(str(para_new[pf]))

    def reset_vel_range(self, p_name, state):
        '''

        :param p_name:
        :param state:
        :return:
        '''

        # updates the parameter enabled value for all associated parameters
        d_grp = self.details[self.get_plot_grp_fcn()]

        # updates the split velocity parameter value
        for i_grp in [i for i in range(len(d_grp)) if 'split_vel' in d_grp[i]['para']]:
            d_grp[i_grp]['para']['split_vel']['def_val'] = state > 0
            self.para[d_grp[i_grp]['name']]['split_vel'] = state > 0

        # updates the velocity direction enabled parameter
        for i_grp in [i for i in range(len(d_grp)) if 'vel_dir' in d_grp[i]['para']]:
            d_grp[i_grp]['para']['vel_dir']['is_enabled'] = state > 0

        # updates the split velocity parameter
        self.curr_para['split_vel'] = state

        # if the velocity direction parameter is present, then update it
        h_list = self.find_obj_handle([QComboBox], 'vel_dir')
        if len(h_list):
            h_list[0].setEnabled(state)

    def reset_comb_all(self, exp_info, p_name):
        '''

        :param exp_info:
        :param p_name:
        :return:
        '''

        # resets the combine all field
        self.reset_comb_all_field(exp_info)

    def reset_trial_sel(self, exp_info, p_name):
        '''

        :param exp_info:
        :param p_name:
        :return:
        '''

        # resets the combine all field
        self.reset_comb_all_field(exp_info)

        # retrieves the new trial type
        tt_new = exp_info['t_type']

        # updates the parameter enabled value for all associated parameters
        d_grp = self.details[self.get_plot_grp_fcn()]
        i_grp = next(i for i in range(len(d_grp)) if d_grp[i]['name'] == self.get_plot_fcn())

        # updates the list parameters
        d_grp[i_grp]['para']['x_plot']['list'] = dcopy(tt_new)
        d_grp[i_grp]['para']['y_plot']['list'] = dcopy(tt_new)

        # retrieves the x/y plot checkbox objects
        h_chk_x = self.find_obj_handle([QComboBox], 'x_plot')[0]
        h_chk_y = self.find_obj_handle([QComboBox], 'y_plot')[0]

        # updates the x/y axis indices based on the user's selection
        if len(tt_new) == 1:
            i_sel_x = i_sel_y = 0
            self.curr_para['x_plot'] = self.curr_para['y_plot'] = dcopy(tt_new[0])
        else:
            # determines if the x/y-axis trial types are in the new trial type selection
            x_in, y_in = self.curr_para['x_plot'] in tt_new, self.curr_para['y_plot'] in tt_new

            # updates the x-axis trial type index
            if x_in:
                # if the trial type is in the trial type selections, then determine the index
                i_sel_x = tt_new.index(self.curr_para['x_plot'])

            elif y_in:
                # if the x-axis is not, but the y-axis trial type is, then reset the index
                i_sel_x = 1 if (tt_new.index(self.curr_para['y_plot']) == 0) else 0

            else:
                # otherwise, set the new index to the first trial type
                i_sel_x = 0

            # updates the y-axis trial type index
            if y_in:
                # if the trial type is in the trial type selections, then determine the index
                i_sel_y = tt_new.index(self.curr_para['y_plot'])

            else:
                # otherwise, set the index based on the x-axis index
                i_sel_y = 1 if (i_sel_x == 0) else 0

        # updates the x-axis checklist values
        h_chk_x.clear()
        for tt in tt_new:
            h_chk_x.addItem(tt)

        # updates the y-axis checklist values
        h_chk_y.clear()
        for tt in tt_new:
            h_chk_y.addItem(tt)

        # updates the indices
        h_chk_x.setCurrentIndex(i_sel_x)
        h_chk_y.setCurrentIndex(i_sel_y)

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

        # retrieves the plot scope parameter value
        if 'plot_scope' in self.curr_para:
            # if it exists, then return the value
            return self.curr_para['plot_scope']
        else:
            # if it doesn't exist, then return the default value
            return 'Individual Cell'

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

    def det_comp_expt_names(self, is_calc):
        '''

        :return:
        '''

        # retrieves the comparison data fields
        comp = self.get_data_fcn().comp
        if hasattr(comp, 'data'):
            c_data = comp.data
        else:
            c_data = []

        if is_calc:
            # case is determining the feasible comparison datasets for calculation functions
            if len(c_data):
                # case is at least one comparison dataset has been set
                return [cf.get_comb_file_names(x.fix_name, x.free_name) for x in c_data]
            else:
                # case is no comparison datasets have been set
                return ['No Fixed/Free Comparisons Set!']

        else:
            # case is determining the feasible comparison datasets for plotting functions
            if len(c_data):
                # determines the fixed experiments for each of the comparison datasets (that have been calculated)
                f_name = [cf.get_comb_file_names(x.fix_name, x.free_name) for x in c_data if x.is_set]
                if len(f_name):
                    # if there is at least one where the calculations have been made, then return the fixed file names
                    return f_name
                else:
                    # otherwise, there are no feasible comparison datasets for plotting
                    return ['No Matching Calculations Made!']
            else:
                # case is there are no comparison datasets
                return ['No Fixed/Free Comparisons Set!']

    def get_ff_cluster_details(self, i_file=0):
        '''

        :return:
        '''

        # overall declarations
        data = self.get_data_fcn()
        free_exp, ff_cluster = None, None

        if hasattr(data, 'externd'):
            if hasattr(data.externd, 'free_data'):
                free_exp = data.externd.free_data.exp_name
                ff_cluster = cfcn.get_matching_fix_free_strings(data, [free_exp[i_file]])

        if free_exp is None:
            free_exp = ['No Fixed/Free Data Loaded!']
            ff_cluster = ['No Fixed/Free Data Loaded!']

        # returns the data arrays
        return free_exp, ff_cluster

    def reset_comb_all_field(self, exp_info):
        '''

        :param exp_info:
        :param p_name:
        :return:
        '''

        # sets the number of selections for each filter type (except trial type and key)
        multi_sel = np.any(np.array([len(exp_info[x]) for x in exp_info if x not in ['t_type', 't_key']]) > 1)

        # updates the checkbox properties based on the user rotation filter selections
        h_chk = self.find_obj_handle([QCheckBox], 'comb_all')[0]
        h_chk.setEnabled(multi_sel)
        if not multi_sel:
            h_chk.setChecked(False)

        # updates the parameter enabled value for all associated parameters
        d_grp = self.details[self.get_plot_grp_fcn()]
        i_grp = next(i for i in range(len(d_grp)) if d_grp[i]['name'] == self.get_plot_fcn())
        d_grp[i_grp]['para']['comb_all']['is_enabled'] = multi_sel

    def get_para_details(self, p_name):
        '''

        :return:
        '''

        # determines the index of the current function
        fcn_d = self.details[self.get_plot_grp_fcn()]
        i_fcn = next(i for i, x in enumerate(fcn_d) if x['name'] == self.get_plot_fcn())

        # returns the parameter dictionary for the current parameter
        return fcn_d[i_fcn]['para'][p_name]

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

        # sets the multiple cell flag
        if 'is_multi' not in para:
            para['is_multi'] = False

        # sets the enabled flag
        if 'is_enabled' not in para:
            para['is_enabled'] = True

        # sets the visible flag
        if 'is_visible' not in para:
            para['is_visible'] = True

        # sets the parameter reset function type
        if 'para_reset' not in para:
            para['para_reset'] = None

        # sets the reset function type
        if 'reset_func' not in para:
            para['reset_func'] = None

        # sets the object type flag
        if 'other_para' not in para:
            para['other_para'] = None

        # returns the parameter dictionary
        return para

########################################################################################################################
########################################################################################################################


class AnalysisData(object):
    def __init__(self):
        # field initialisation
        self._cluster = []
        self.cluster = None
        self.comp = ComparisonData()
        self.classify = ClassifyData()
        self.rotation = RotationData('rotation')
        self.depth = RotationData('depth')
        self.discrim = DiscriminationData()
        self.multi = MultiFileData()
        self.spikedf = SpikingFreqData()
        self.externd = ExternalData()

        # exclusion filter fields
        self.exc_gen_filt = None
        self.exc_rot_filt = None
        self.exc_ud_filt = None

        # other flags
        self.req_update = True
        self.force_calc = True
        self.files = None

    def check_missing_fields(self):
        '''

        :param f_name:
        :return:
        '''

        # field strings to check
        fld_str = ['_cluster', 'cluster', 'comp', 'classify', 'rotation', 'depth', 'discrim', 'spikedf']

        # changes the location of the rotation/uniformdrifting exclusion filters (if in the wrong location)
        if hasattr(self, 'rotation'):
            if not hasattr(self.rotation, 'vel_sf_corr'):
                setattr(self.rotation, 'n_shuffle_corr', -1)
                setattr(self.rotation, 'vel_bin_corr', -1)
                setattr(self.rotation, 'vel_sf_nsm', -1)
                setattr(self.rotation, 'vel_sf_corr', None)
                setattr(self.rotation, 'vel_sf_corr_rs', None)

            if not hasattr(self, 'exc_rot_filt'):
                setattr(self, 'exc_rot_filt', self.rotation.exc_rot_filt)
                setattr(self, 'exc_ud_filt', self.rotation.exc_ud_filt)


        # checks if the above fields exist in the data object. if not, then add them in
        for fs in fld_str:
            if not hasattr(self, fs):
                if fs == '_cluster':
                    setattr(self, fs, [])
                elif fs == 'cluster':
                    setattr(self, fs, None)
                elif fs == 'comp':
                    setattr(self, fs, ComparisonData())
                    self.comp.init_comparison_data()
                elif fs == 'classify':
                    setattr(self, fs, ClassifyData())
                elif fs == 'rotation':
                    setattr(self, fs, RotationData('rotation'))
                    self.rotation.init_rot_fields()
                elif fs == 'depth':
                    setattr(self, fs, RotationData('depth'))
                    self.depth.init_rot_fields()
                elif fs == 'discrim':
                    setattr(self, fs, DiscriminationData())
                    self.discrim.init_discrim_fields()
                elif fs == 'spikedf':
                    setattr(self, fs, SpikingFreqData())

    def update_gen_filter(self):
        '''

        :param c_data:
        :return:
        '''

        # initialises the rotation filter (if not already done so)
        if self.exc_gen_filt is None:
            self.exc_gen_filt = cf.init_general_filter_data()

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


class ComparisonDataObj(object):
    def __init__(self):
        # calculation flag
        self.is_set = False

        # initial parameter fields
        self.d_max = int(cfcn.get_glob_para('d_max'))
        self.r_max = float(cfcn.get_glob_para('r_max'))
        self.sig_corr_min = float(cfcn.get_glob_para('sig_corr_min'))
        self.isi_corr_min = float(cfcn.get_glob_para('isi_corr_min'))
        self.sig_diff_max = float(cfcn.get_glob_para('sig_diff_max'))
        self.sig_feat_min = float(cfcn.get_glob_para('sig_feat_min'))
        self.w_sig_feat = float(cfcn.get_glob_para('w_sig_feat'))
        self.w_sig_comp = float(cfcn.get_glob_para('w_sig_comp'))
        self.w_isi = float(cfcn.get_glob_para('w_isi'))

    def init_class_fields(self, ind, n_fix, n_free, n_pts, fix_name, free_name):
        '''

        :param ind:
        :param n_fix:
        :param n_free:
        :param n_pts:
        :param fix_name:
        :param free_name:
        :return:
        '''

        # initialisations
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
        self.signal_feat = -np.ones((n_fix, 4), dtype=float)
        self.total_metrics = -np.ones((n_fix,3), dtype=float)
        self.total_metrics_mean = -np.ones(n_fix, dtype=float)


class FixedFreeCorr(object):
    def __init__(self):

        # parameters
        self.is_set = False
        self.vel_bin = -1
        self.n_shuffle_corr = -1
        self.split_vel = False
        self.force_update = False

        # other arrays
        self.sf_fix = None
        self.sf_free = None
        self.sf_corr = None
        self.sf_corr_sh = None
        self.sf_corr_sig = None
        self.sf_grad = None
        self.clust_id = None
        self.ind_g = None


class ComparisonData(object):
    def __init__(self):

        # initialises the comparison data
        self.init_comparison_data()

    def init_comparison_data(self):
        '''

        :return:
        '''

        # initialisation
        self.is_set = False
        self.data = []
        self.last_comp = -1

        # initialises the fixed/free correlation object
        self.ff_corr = FixedFreeCorr()

    def set_comparison_data(self, ind, n_fix, n_free, n_pts, fix_name, free_name):
        '''

        :param ind:
        :param n_fix:
        :param n_pts:
        :return:
        '''

        # creates a new class object
        DataNw = ComparisonDataObj()
        DataNw.init_class_fields(ind, n_fix, n_free, n_pts, fix_name, free_name)

        # appends the data to the data array
        self.data.append(DataNw)


class ClassifyData(object):
    def __init__(self):
        # initialisation
        self.is_set = False
        self.class_set = False
        self.action_set = False

    def init_classify_fields(self, expt_name, clust_id):
        '''

        :param expt_name:
        :param clust_id:
        :return:
        '''

        # initialisation
        self.is_set = False

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
    def __init__(self, type):

        # sets the type flag
        self.type = type

        # initialisation
        self.is_set = False

    def init_rot_fields(self):
        '''

        :return:
        '''

        # general initialisations
        self.n_boot_phase_grp = -1
        self.n_boot_cond_grp = -1
        self.n_boot_cond_ci = -1
        self.n_boot_comb_grp = -1
        self.t_ofs_rot = -1
        self.t_phase_rot = -1
        self.t_ofs_vis = -1
        self.t_phase_vis = -1
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

        # uniform drifting cell group roc parameters
        self.phase_roc_ud = None                # phase r roc objects
        self.phase_roc_xy_ud = None             # phase roc curve x/y coordinates
        self.phase_roc_auc_ud = None            # phase roc curve integrals

        # uniform drifting cell group roc parameters
        self.part_roc = {}                      # partial r roc objects
        self.part_roc_xy = {}                   # partial roc curve x/y coordinates
        self.part_roc_auc = {}                  # partial roc curve integrals

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
        self.ms_gtype_ex = None
        self.ms_gtype_pr = None
        self.ms_gtype_N = None
        self.ds_gtype_ex = None
        self.ds_gtype_pr = None
        self.ds_gtype_N = None
        self.pd_type_ex = None
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
        self.freq_type = ''

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
        self.vel_roc_sig = None

        # velocity shuffled correlation fields
        self.vel_sf_nsm = -1
        self.vel_bin_corr = -1
        self.n_shuffle_corr = -1
        self.vel_sf_eqlt = False
        self.split_vel = True
        self.vel_shuffle_calc = False
        self.vel_sf_mean = None
        self.vel_sf_shuffle = None
        self.vel_sf_corr = None
        self.vel_sf_corr_mn = None
        self.vel_sf_sig = None

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
        self.spd_roc_sig = None

        # sets the depth class specific fields
        if self.type == 'depth':
            self.ch_depth, self.ch_region, self.ch_layer = None, None, None
            self.ch_depth_ms, self.ch_region_ms, self.ch_layer_ms = None, None, None

            # initialises the multi-sensory depth data fields
            self.plt, self.stats, self.ind, self.r_filt = None, None, None, None
            self.plt_rms, self.stats_rms, self.ind_rms, self.r_filt_rms = None, None, None, None
            self.plt_vms, self.stats_vms, self.ind_vms, self.r_filt_vms = None, None, None, None


class DiscriminationData(object):
    def __init__(self):
        # initialisation
        self.is_set = False
        self.init_discrim_fields()

    def init_discrim_fields(self):
        '''

        :return:
        '''

        # rotation discrimination analysis
        self.dir = SubDiscriminationData('Direction')
        self.temp = SubDiscriminationData('Temporal')
        self.indiv = SubDiscriminationData('Individual')
        self.shuffle = SubDiscriminationData('TrialShuffle')
        self.part = SubDiscriminationData('Partial')
        self.filt = SubDiscriminationData('IndivFilt')
        self.wght = SubDiscriminationData('LDAWeight')

        # kinematic discrimination analysis
        self.spdacc = SubDiscriminationData('SpdAcc')
        self.spdc = SubDiscriminationData('SpdComp')
        self.spdcp = SubDiscriminationData('SpdCompPool')
        self.spddir = SubDiscriminationData('SpdCompDir')


class SubDiscriminationData(object):
    def __init__(self, type):

        # sets the type flag
        self.type = type

        self.lda = None
        self.i_expt = None
        self.i_cell = None
        self.y_acc = None
        self.exp_name = None
        self.pw_corr = None
        self.z_corr = None
        self.lda_trial_type = None

        # lda calculation/parameter elements
        self.ntrial = -1
        self.solver = None
        self.shrinkage = -1
        self.norm = -1
        self.ctype = None
        self.ttype = None
        self.cellmin = -1
        self.trialmin = -1
        self.yaccmx = -1
        self.yaccmn = -1
        self.yaucmx = -1
        self.yaucmn = -1

        if type in ['Direction', 'Individual', 'TrialShuffle', 'Partial', 'IndivFilt', 'LDAWeight']:
            # case is the direction LDA analysis
            self.tofs = -1
            self.tphase = -1
            self.usefull = -1

            if type == 'TrialShuffle':
                # case is the shuffled LDA analysis
                self.nshuffle = -1
                self.bsz = -1

            elif type == 'Partial':
                # case is the partial LDA analysis
                self.nshuffle = -1
                self.poolexpt = False
                self.xi = None

            elif type == 'IndivFilt':
                # case is the individual filtered LDA analysis
                self.yaccmn = -1
                self.yaccmx = -1

            elif type == 'LDAWeight':
                # case is the LDA weights
                self.xi = None
                self.c_wght = None
                self.c_wght0 = None
                self.y_acc_bot = None
                self.y_acc_top = None

        elif type in ['Temporal']:
            # case is the temporal LDA analysis
            self.dt_phs = -1
            self.dt_ofs = -1
            self.phs_const = -1
            self.xi_phs = None
            self.xi_ofs = None

        elif type in ['SpdAcc', 'SpdComp', 'SpdCompPool', 'SpdCompDir']:
            # case is the speed comparison LDA
            self.spd_xi = None
            self.i_bin_spd = -1
            self.vel_bin = -1
            self.n_sample = -1
            self.equal_time = -1

            if type in ['SpdComp', 'SpdCompPool']:
                self.y_acc_fit = None
                self.spd_xrng = -1

            if type in ['SpdCompPool']:
                # case is the pooled speed comparison LDA
                self.n_cell = None
                self.p_acc = None
                self.p_acc_lo = None
                self.p_acc_hi = None
                self.nshuffle = -1


class MultiFileData(object):
    def __init__(self, is_multi=False):

        # initialisation
        self.is_multi = is_multi
        self.names = None
        self.files = None

    def set_multi_file_data(self, dlg_info):
        '''

        :param exp_info:
        :return:
        '''

        if dlg_info is None:
            self.is_multi, self.names, self.files = False, None, None
        else:
            self.is_multi, self.names, self.files = True, dcopy(dlg_info.exp_name), dcopy(dlg_info.exp_files)


class SpikingFreqData(object):
    def __init__(self):
        # initialises the class fields
        self.is_set = False
        self.sf_df = None

        # calculation parameters
        self.bin_sz = -1
        self.t_over = -1
        self.out_name = 'Spike_Freq_Dataframe'
        self.rot_filt = cf.init_rotation_filter_data(False)

########################################################################################################################
########################################################################################################################


class ExternalData(object):
    def __init__(self):

        # creates an empty class
        pass


class FreelyMovingData(object):
    def __init__(self, data, f_data):

        # initialises the static object fields
        self.n_file = 0
        self.v_bin = f_data['AHV_bin_size']
        self.v_max = f_data['AHV_max_magnitude']
        self.t_type = np.sort(list(f_data['cell_data'][0].keys()))

        # initialises the other
        self.exp_name = []
        self.cell_id = []
        self.s_freq = []
        self.c_info = []
        self.p_sig_neg = []
        self.p_sig_pos = []
        self.cell_type = []
        self.ahv_score = []

        # creates the objects for each experiment
        self.append_data(data, f_data)

    def append_data(self, data, f_data):
        '''

        :param data:
        :param f_data:
        :return:
        '''

        # parameters
        v_min_hd = 0.2                  # min vec for head direction cells
        v_min_hd_mod_lo = 0.1           # min vec for head modulated direction cells (lower limit)
        v_min_hd_mod_hi = 0.2           # min vec for head modulated direction cells (upper limit)
        p_rayleight_hd = 0.01           # rayleigh t-test p-value for head direction cells
        p_rayleight_hd_mod = 0.01       # rayleigh t-test p-value for head direction modulated cells
        # p_tile_hd = 97.0                # min mean vec percentile for head direction cells
        # p_tile_hd_mod = 97.0            # min mean vec percentile for head direction modulated cells
        p_tile_ahv = 95.0               # min mean vec percentile for angular head velocity cells
        p_tile_speed = 95.0             # min mean vec percentile for speed cells
        p_tile_place = 95.0             # min mean vec percentile for place cells

        def check_data_fields(c_info):
            '''

            :param c_info:
            :return:
            '''

            # cell information field keys
            ci_key = ['mean_vec_length', 'mean_vec_percentile', 'pearson_neg_percentile',
                      'pearson_pos_percentile', 'pearson_percentile', 'peak_percentile']

            # checks that all the fields (for all trial types) are not empty
            for tt in c_info.keys():
                for ck in ci_key:
                    # if the field is empty, then exit with a false flag
                    if isinstance(c_info[tt][ck].ix[0], list):
                        return False

            # flag that
            return True

        if f_data['experiment_name'] in self.exp_name:
            # if the data already exists in the class object, then exit
            return
        else:
            # determines if the new data file matches any of the loaded experiments
            free_name = [cf.extract_file_name(x['expFile']) for x in data._cluster if x['rotInfo'] is None]
            i_expt_nw = cf.det_likely_filename_match(free_name, f_data['experiment_name'])
            if i_expt_nw is None:
                # if there is no matching expt, then exit without appending
                return

        # initialisations
        has_missing_fields = False
        c_type = ['HD', 'HDMod', 'AHV', 'Speed', 'Place']

        # increments the file count
        self.n_file += 1

        # converts the cell ID strings to integers
        cell_id = [int(x.split('_')[1]) for x in f_data['cell_data'][0][self.t_type[0]]['cell_list_order']]

        # appends the new experiment name and cell IDs
        self.exp_name.append(f_data['experiment_name'])
        self.cell_id.append(cell_id)

        # memory allocation
        n_cell = len(cell_id)
        A = np.empty((len(self.v_bin), len(self.t_type)), dtype=object)
        B = np.zeros((n_cell, len(self.v_bin), len(self.t_type)))
        C = np.empty(len(self.v_bin), dtype=object)
        s_freq, p_sig_neg, p_sig_pos = dcopy(A), dcopy(B), dcopy(B)
        cell_type, ahv_score, c_info = dcopy(C), dcopy(C), dcopy(C)

        # retrieves the necessary information from trial condition/velocity bin size
        for i_bin, v_bin in enumerate(self.v_bin):
            # initialises the cell information data dictionary
            c_info[i_bin] = {}
            for i_tt, tt in enumerate(self.t_type):
                # retrieves the cell data for the given trial condition/velocity bin size
                c_data = f_data['cell_data'][i_bin][tt]
                c_info[i_bin][tt] = c_data['cell_information']

                # sets the spiking frequencies (reverses them to match fixed spiking frequencies)
                s_freq[i_bin, i_tt] = [x[::-1] for x in dcopy(c_data['AHV_spiking_frequency'])]

                # corrects the correlation values for the negative velocity range
                c_info[i_bin][tt]['ahv_pearson_r_neg'] *= -1

                # retrieves the significant cells
                p_sig_neg[:, i_bin, i_tt] = c_info[i_bin][tt]['ahv_pearson_p_neg']
                p_sig_pos[:, i_bin, i_tt] = c_info[i_bin][tt]['ahv_pearson_p_pos']

            ########################################
            ####    FREE CELL CLASSIFICATION    ####
            ########################################

            # if the required fields are not set, then continue
            if (not check_data_fields(c_info[i_bin])) or has_missing_fields:
                has_missing_fields = True
                continue

            # head direction cell significance
            #   => for BOTH light1 and light2 conditions:
            #     * v_min_hd_mod_lo < mean_vec_length < v_min_hd_mod_hi
            #     * rayleigh_test_p < p_rayleight_hd_mod
            hd_sig = np.logical_and(
                np.logical_and(c_info[i_bin]['LIGHT1']['mean_vec_length'] >= v_min_hd,
                               c_info[i_bin]['LIGHT2']['mean_vec_length'] >= v_min_hd),
                np.logical_and(c_info[i_bin]['LIGHT1']['rayleigh_test_p'] < p_rayleight_hd,
                               c_info[i_bin]['LIGHT2']['rayleigh_test_p'] < p_rayleight_hd)
            )

            # head direction modulated cell significance
            #   => for BOTH light1 and light2 conditions:
            #     * mean_vec_length >= v_min_hd_mod
            #     * rayleigh_test_p < p_rayleight_hd
            hd_mod_sig_vec = np.logical_and(
                np.logical_and(c_info[i_bin]['LIGHT1']['mean_vec_length'] > v_min_hd_mod_lo,
                               c_info[i_bin]['LIGHT2']['mean_vec_length'] > v_min_hd_mod_lo),
                np.logical_and(c_info[i_bin]['LIGHT1']['mean_vec_length'] < v_min_hd_mod_hi,
                               c_info[i_bin]['LIGHT2']['mean_vec_length'] < v_min_hd_mod_hi),
            )
            hd_mod_sig = np.logical_and(
                np.logical_and(c_info[i_bin]['LIGHT1']['rayleigh_test_p'] < p_rayleight_hd_mod,
                               c_info[i_bin]['LIGHT2']['rayleigh_test_p'] < p_rayleight_hd_mod),
                hd_mod_sig_vec
            )

            # angular head velocity cell significance
            ahv_sig_pos = c_info[i_bin]['DARK1']['pearson_pos_percentile'] > p_tile_ahv
            ahv_sig_neg = c_info[i_bin]['DARK1']['pearson_neg_percentile'] > p_tile_ahv
            ahv_score[i_bin] = np.array(ahv_sig_neg).astype(int) + 2 * np.array(ahv_sig_pos).astype(int)
            ahv_sig = ahv_score[i_bin] > 0

            # speed cell significance
            spd_sig = c_info[i_bin]['DARK1']['pearson_percentile'] > p_tile_speed

            # place cell significance
            pl_sig = np.logical_and(c_info[i_bin]['LIGHT1']['peak_percentile'] > p_tile_place,
                                    c_info[i_bin]['LIGHT2']['peak_percentile'] > p_tile_place)

            # combines the significance arrays into a single dataframe
            cell_type[i_bin] = pd.DataFrame(np.vstack((hd_sig, hd_mod_sig, ahv_sig, spd_sig, pl_sig)).T, columns=c_type)

        # appends the new fields to the class object
        self.s_freq.append(s_freq)
        self.c_info.append(c_info)
        self.p_sig_neg.append(p_sig_neg)
        self.p_sig_pos.append(p_sig_pos)

        # sets the cell types (depending if fields were missing or not)
        if has_missing_fields:
            self.cell_type.append([])
            self.ahv_score.append([])
        else:
            self.cell_type.append(cell_type)
            self.ahv_score.append(ahv_score)


class EyeTrackingData(object):
    def __init__(self, data, f_file):

        # initialises the static object fields
        self.fps = 40
        self.n_file = 0

        # calculation parameters
        self.use_med_filt = True
        self.rmv_baseline = True
        self.dp_max = 1.0
        self.n_sd = 2.5
        self.n_pre = 3
        self.n_post = 8

        # initialises the other fields
        self.exp_name = []
        self.et_data = []
        self.cl_data = []

        # initialises the calculation fields
        self.t_evnt = []
        self.y_evnt = []
        self.t_sp_h = []
        self.sp_evnt = []
        self.y_corr = []
        self.p_corr = []

        # creates the objects for each experiment
        self.append_data(data, f_file)

    def append_data(self, data, f_file):
        '''

        :param data:
        :param f_data:
        :return:
        '''

        # resets the set flag
        self.is_set = False

        # reads the csv file
        f_data = pd.read_csv(f_file)
        f_name = cf.extract_file_name(f_file)
        exp_name_nw, t_type_nw = '_'.join(f_name.split('_')[:-1]), f_name.split('_')[-1]
        _exp_name_nw = remove_uscore(exp_name_nw)

        #
        exp_name_cluster = [cf.extract_file_name(c['expFile']) for c in data._cluster]
        _exp_name_cluster = [remove_uscore(x) for x in exp_name_cluster]

        # determines if the experiment matching the eye-tracking data file has been loaded
        if _exp_name_nw in _exp_name_cluster:
            # if so, then append the experiment data to experiment data
            i_expt = _exp_name_cluster.index(_exp_name_nw)
            if exp_name_cluster[i_expt] in self.exp_name:
                # if the experiment is set, then determine the index counter
                self.et_data[self.exp_name.index(exp_name_cluster[i_expt])].append_data(f_data, t_type_nw)

            else:
                # if the experiment is not set, then increment the file counter and appends the experiment name
                i_expt = _exp_name_cluster.index(_exp_name_nw)
                self.exp_name.append(exp_name_cluster[i_expt])
                cl_data = data._cluster[i_expt]
                self.et_data.append(EyeTrackingDataSub(cl_data, f_data, t_type_nw))

                # increments the experiment counter
                self.n_file += 1

class EyeTrackingDataSub(object):
    def __init__(self, cl_data, f_data, t_type):

        # sets/initialises the static object fields
        self.nt_type = 0
        self.c = cl_data

        # other field initialisations
        self.t_type = []
        self.p_pos = []

        # appends the data to the
        self.append_data(f_data, t_type)

    def append_data(self, f_data, t_type):
        '''

        :param f_data:
        :param t_type:
        :return:
        '''

        if t_type in self.t_type:
            # if the trial type is already set, then exit the function
            return
        else:
            # otherwise, append the data to the class object
            self.nt_type += 1
            self.t_type.append(t_type)

        # retrieves the position/diameter locations
        col_str = list(f_data.ix[0, :])
        p_pos = self.get_metric_values(f_data, col_str, 'pupil position')
        # p_dia = self.get_metric_values(f_data, col_str, 'pupil diameter')

        # adds the position/position derivatives to the overall storage arrays
        self.p_pos.append(p_pos)

    def get_metric_values(self, f_data, col_str, met_str):
        '''

        :param f_data:
        :param col_str:
        :param met_str:)
        :return:
        '''

        # retrieves the index of the column of interest
        ind = next(i for i in range(len(col_str)) if met_str in col_str[i])
        return f_data.ix[1:, ind]

########################################################################################################################
########################################################################################################################


class OutputData(object):
    def __init__(self, f_data):
        # REMOVE ME LATER
        pass

########################################################################################################################
########################################################################################################################


class PlotCanvas(FigureCanvas):
    def __init__(self, main_obj, dpi=100):

        # creates the figure object
        grp_sz = main_obj.grp_plot.geometry()
        fig = Figure(figsize=((grp_sz.width() - (2*dX + 1)) / dpi,(grp_sz.height() - (2*dX + 1)) / dpi),
                     dpi=dpi, tight_layout=True)

        # creates the figure class
        FigureCanvas.__init__(self, fig)
        self.setParent(main_obj.grp_plot)
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

########################################################################################################################
########################################################################################################################

