# module imports
import copy
import functools
import numpy as np

# pyqt5 module import
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QPushButton, QGridLayout, QGroupBox)

# custom module import
import analysis_guis.common_func as cf
import analysis_guis.rotational_analysis as rot

# parameters
dX = 10
dY = 10
width = 450

# font objects
txt_font = cf.create_font_obj(size=8, is_bold=False)
txt_font_bold = cf.create_font_obj(size=8, is_bold=True, font_weight=75)
button_font = cf.create_font_obj(size=9, is_bold=True, font_weight=75)
grp_font = cf.create_font_obj(size=10, is_bold=True, font_weight=75)

# other function declarations
dcopy = copy.deepcopy
get_field = lambda wfm_para, f_key: np.unique(cf.flat_list([list(x[f_key]) for x in wfm_para]))

########################################################################################################################
########################################################################################################################


class RotationFilter(QDialog):
    def __init__(self, main_obj, parent=None, init_data=None, other_var=None, is_exc=False, is_gen=False):
        # creates the object
        super(RotationFilter, self).__init__(parent)

        # other initialisations
        self.grp_type = main_obj.get_plot_grp_fcn()
        self.can_close = False
        self.update_plot = False
        self.is_ok = True
        self.is_init = False
        self.use_both = False
        self.is_exc = is_exc
        self.is_gen = is_gen
        self.data = main_obj.get_data_fcn()

        if self.is_exc:
            self.plot_all_expt = True
            self.is_multi_cell = True

            if self.is_gen:
                self.rmv_fields = ['t_type', 'record_coord', 'sig_type', 'match_type']
                self.is_ud = False
            else:
                self.rmv_fields = None
                self.is_ud = init_data['is_ud'][0]

        else:
            self.plot_all_expt = main_obj.grp_para_plot.findChild(QCheckBox, 'plot_all_expt').checkState()
            if other_var is not None:
                if 'use_ud' in other_var:
                    self.is_ud = main_obj.grp_para_plot.findChildren(QCheckBox, other_var['use_ud'])[0].checkState() > 0
                    if self.is_ud:
                        self.use_both = True
                        self.rmv_fields = other_var['rmv_fields']
                    else:
                        self.rmv_fields = None
                else:
                    self.rmv_fields = other_var['rmv_fields']
                    # self.is_ud = self.grp_type in ['UniformDrift Analysis', 'Combined Analysis']
                    self.is_ud = self.grp_type in ['UniformDrift Analysis']
            else:
                self.rmv_fields = None
                # self.is_ud = self.grp_type in ['UniformDrift Analysis', 'Combined Analysis']
                self.is_ud = self.grp_type in ['UniformDrift Analysis']

            plot_scope = main_obj.get_plot_scope()
            if plot_scope is None:
                self.is_multi_cell = False
            else:
                self.is_multi_cell = plot_scope != 'Individual Cell'

        # sets/iniitialises the filter data dictionary based on the input values state
        if init_data is None:
            # case is
            self.init_filter_data()
        else:
            self.f_data = init_data

        # determines the feasible filter fields
        self.init_filter_fields()
        if self.n_grp == 0:
            # if there are no
            e_str = 'There does not appear to be feasible filter fields associated with these experiments.'
            cf.show_error(e_str, 'No Feasible Filter Fields')
            return

        # initialises all the other GUI objects
        self.init_dialog()
        self.init_filter_groups()

        # ensures the gui has a fixed size
        cf.set_obj_fixed_size(self, width)
        self.is_init = True

        # shows the final GUI
        self.show()
        self.exec()

    #################################################
    ####     OBJECT INITIALISATION FUNCTIONS     ####
    #################################################

    def init_dialog(self):
        '''

        :return:
        '''

        if self.is_exc:
            if self.is_gen:
                title = "General Exclusion Filter"
            elif self.is_ud:
                title = "UniformDrift Exclusion Filter"
            else:
                title = "Rotational Exclusion Filter"
        else:
            if self.grp_type == 'Rotation Analysis':
                title = "Rotational Analysis Plot Filter"
            elif self.grp_type == 'ROC Analysis':
                title = "ROC Analysis"
            elif self.grp_type == 'Combined Analysis':
                title = "Combined Analysis Plot Filter"
            else:
                title = "UniformDrift Analysis Plot Filter"

        self.setObjectName("RotationFilter")
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

    def init_filter_data(self):
        '''

        :return:
        '''

        self.f_data = cf.init_rotation_filter_data(self.is_ud and (not self.use_both))

    def init_filter_fields(self):
        '''

        :return:
        '''

        # retrieves the data clusters for each of the valid rotation experiments
        is_rot_expt = cf.det_valid_rotation_expt(self.data)
        d_clust = [x for x, y in zip(self.data._cluster, is_rot_expt) if y]

        # retrieves the trial-types from each experiment
        if (self.grp_type in ['Rotation Analysis', 'ROC Analysis', 'Combined Analysis']) or \
           (self.is_exc and (not self.is_ud)):
            t_list0 = cf.flat_list([list(x['rotInfo']['trial_type']) for x in d_clust])
            if self.use_both:
                trial_type = [x for x in np.unique(t_list0) if x != 'UniformDrifting']
            else:
                trial_type = list(np.unique([x for x in t_list0 if 'UniformDrifting' not in x]))
        else:
            trial_type = ['']

        # sets the field combobox lists
        sig_type = ['Narrow Spikes', 'Wide Spikes']
        match_type = ['Matched Clusters', 'Unmatched Clusters']
        region_type = np.unique(cf.flat_list([list(np.unique(x['chRegion'])) for x in d_clust]))
        record_layer = np.unique(cf.flat_list([list(np.unique(x['chLayer'])) for x in d_clust]))
        record_coord = list(np.unique([x['expInfo']['record_coord'] for x in d_clust]))

        # sets the filter field parameter information
        self.fields = [
            ['Trial Type', 'CheckCombo', 't_type', trial_type, True],
            ['Cluster Signal Type', 'CheckCombo', 'sig_type', sig_type, self.data.classify.is_set],
            ['Cluster Match Type', 'CheckCombo', 'match_type', match_type, self.data.comp.is_set],
            ['Region Name', 'CheckCombo', 'region_name', list(region_type), self.is_multi_cell and self.plot_all_expt],
            ['Recording Layer', 'CheckCombo', 'record_layer', list(record_layer), self.is_multi_cell and self.plot_all_expt],
            ['Recording Coordinate', 'CheckCombo', 'record_coord', record_coord, True],
        ]

        # appends on additional query fields for if analysing uniform-drifting
        if self.is_ud:
            # combines the wave-form parameters over all experiments
            wfm_para = [x['rotInfo']['wfm_para']['UniformDrifting'] for x in
                                                    d_clust if 'UniformDrifting' in x['rotInfo']['wfm_para']]

            # retrieves the unique uniform-drifting stimuli parameters
            temp_freq = [str(x) for x in get_field(wfm_para, 'tFreq')]
            temp_freq_dir = [str(x) for x in get_field(wfm_para, 'yDir').astype(int)]
            temp_cycle = [str(x) for x in get_field(wfm_para, 'tCycle').astype(int)]

            # adds the new data fields
            self.fields = self.fields[int(not self.use_both):] + [
                ['Temporal Frequency Magnitude', 'CheckCombo', 't_freq', temp_freq, True],
                ['Temporal Frequency Direction', 'CheckCombo', 't_freq_dir', temp_freq_dir, True],
                ['Spatial Frequency', 'CheckCombo', 't_cycle', temp_cycle, True],
            ]

        # removes any other fields (if specified)
        if self.rmv_fields is not None:
            for r_field in self.rmv_fields:
                i_rmv = next((i for i in range(len(self.fields)) if self.fields[i][2] == r_field), None)
                if i_rmv is not None:
                    self.fields[i_rmv][4] = False

        # if running the filter in non-exclusion mode, then remove any field values that have been excluded
        f_str = [x[2] for x in self.fields]
        if self.is_exc:
            # removes the black phase from the rotation filter
            if not self.is_ud and not self.is_gen:
                i_field = f_str.index('t_type')
                self.fields[i_field][3].pop(self.fields[i_field][3].index('Black'))
        else:
            # sets the exclusion filter based on the analysis type
            if self.is_ud:
                # case is uniformdrifting analysis
                exc_filt = self.data.rotation.exc_ud_filt
            else:
                # case is rotation analysis
                exc_filt = self.data.rotation.exc_rot_filt

            # loops through each of the filter keys removing any
            for fk in exc_filt.keys():
                if len(exc_filt[fk]) and (fk in f_str):
                    # retrieves the field that the filter key corresponds to
                    i_field = f_str.index(fk)

                    # removes the fields values that have been excluded
                    for fv in exc_filt[fk]:
                        self.fields[i_field][3].pop(self.fields[i_field][3].index(fv))

        # removes any groups that don't have more than one query value
        for i_row in reversed(range(len(self.fields))):
            if (len(self.fields[i_row][3]) == 1) or (not self.fields[i_row][-1]):
                self.fields.pop(i_row)

        # sets the number of filter groups
        self.n_grp = len(self.fields)

    def init_filter_groups(self):
        '''

        :return:
        '''

        # memory allocation
        self.h_grpbx = np.empty((self.n_grp+1,1), dtype=object)
        self.grp_width, n_grp = width - (2 * dX), dcopy(self.n_grp)

        # creates the main layout widget
        mainLayout = QGridLayout()

        # creates the progressbar and layout objects
        for i_grp in range(self.n_grp):
            if self.fields[i_grp][2] in self.f_data:
                self.create_single_group(i_grp)
                mainLayout.addWidget(self.h_grpbx[i_grp][0], i_grp, 0)

        # creates the control buttons
        self.create_control_buttons()
        mainLayout.addWidget(self.h_grpbx[self.n_grp][0], self.n_grp, 0)

        # sets the main progress-bar layout
        self.setLayout(mainLayout)

    def create_single_group(self, i_grp):
        '''

        :return:
        '''

        # initialisations
        grp_type = self.fields[i_grp][1]

        # creates the group box object
        self.h_grpbx[i_grp] = QGroupBox(self.fields[i_grp][0])
        cf.set_obj_fixed_size(self.h_grpbx[i_grp][0], width=self.grp_width, fix_size=False)
        self.h_grpbx[i_grp][0].setFont(grp_font)

        #
        hP = self.h_grpbx[i_grp][0]
        if grp_type == 'CheckCombo':
            # creates
            chk_list = self.fields[i_grp][3]
            if not self.is_exc:
                if len(chk_list) > 1 and chk_list[0] != 'All':
                    chk_list = ['All'] + chk_list
                    self.fields[i_grp][3] = chk_list
                elif len(chk_list) == 1 and self.f_data[self.fields[i_grp][2]][0] == 'All':
                    self.f_data[self.fields[i_grp][2]] = [self.fields[i_grp][3][0]]

            #
            any_sel = len(self.f_data[self.fields[i_grp][2]])
            if any_sel:
                first_line = '--- Selection: {0} ---'.format(', '.join(self.f_data[self.fields[i_grp][2]]))
            else:
                first_line = '--- Selection: None ---'

            if self.is_exc:
                h_obj = cf.create_checkcombo(hP, None, chk_list, has_all=False, first_line=first_line)
            else:
                h_obj = cf.create_checkcombo(hP, None, chk_list, has_all=len(chk_list)>1, first_line=first_line)

            #
            cb_func = functools.partial(self.checkComboUpdate, h_obj, chk_list, i_grp)
            h_obj.view().pressed.connect(cb_func)

            # sets the initial states
            if any_sel:
                for i_sel in [self.fields[i_grp][3].index(x) for x in self.f_data[self.fields[i_grp][2]]]:
                    h_obj.handleItemPressed(i_sel+1)

        elif grp_type == 'NumberGroup':
            # case is a number group group

            # initialisations
            n_num, n_txt = len(self.fields[i_grp][3]), self.fields[i_grp][3]
            n_val = self.f_data[self.fields[i_grp][2]]
            w_num = (self.grp_width - 2*dX) / n_num
            h_obj = [[] for _ in range(2 * n_num)]

            #
            for i_num in range(n_num):
                # creates the label text
                ind_txt, ind_num = i_num * 2, i_num * 2 + 1
                dim_txt = QRect(dX + (2 * i_num) * w_num, dY/2, w_num, 17)
                h_obj[ind_txt] = cf.create_label(hP, txt_font_bold, n_txt[i_num], dim=dim_txt, align='right')

                # creates the number edit boxes
                dim_num = QRect(dX + (2 * i_num + 1) * w_num, dY, w_num, 21)
                h_obj[ind_num] = cf.create_edit(hP, txt_font, str(n_val[i_num]), dim=dim_num)

        # sets the widgets into the box layout
        layout = QHBoxLayout()
        if isinstance(h_obj, list):
            for hh in h_obj:
                layout.addWidget(hh)
        else:
            layout.addWidget(h_obj)

        # sets the groupbox layout and enabled properties
        self.h_grpbx[i_grp][0].setLayout(layout)
        cf.set_group_enabled_props(self.h_grpbx[i_grp][0], self.fields[i_grp][4])

    def get_info(self):
        '''

        :return:
        '''

        if not self.is_ok:
            # user cancelled
            return None
        elif all([(len(x)>0) for x in zip(list(self.f_data.values()))]):
            # all fields were filled out correctly
            return self.f_data
        else:
            # not all the fields were filled out correctly
            return None

    def create_control_buttons(self):
        '''

        :return:
        '''

        # initialisations
        layout = QHBoxLayout()

        #
        if self.is_exc:
            b_txt = ['Update Exclusion Filter', 'Cancel']
            cb_fcn = [self.update_exc_filter_only, self.user_cancel]
            b_name = ['update_exc_filter', 'user_cancel']
        else:
            b_txt = ['Update Axes Plot', 'Update Filter Only', 'Cancel']
            cb_fcn = [self.update_filter_plot, self.update_filter_only, self.user_cancel]
            b_name = ['update_filter_plot', 'update_filter_only', 'user_cancel']

        # group box object
        b_wid = (self.grp_width - (1 + len(b_txt)) * dX) / len(b_txt)
        self.h_grpbx[self.n_grp] = QGroupBox("")

        # creates the load config file object
        for i in range(len(b_txt)):
            # creates the button object
            b_dim = QRect((i + 1)*dX + i * b_wid, dY, b_wid, 21)
            h_but = cf.create_button(self.h_grpbx[self.n_grp][0], b_dim, button_font, b_txt[i],
                                     cb_fcn=cb_fcn[i], name=b_name[i])
            h_but.setAutoDefault(False)

            # adds the objects to the layout
            layout.addWidget(h_but)

        # sets the box layout
        self.h_grpbx[self.n_grp][0].setLayout(layout)

    def set_button_enabled_props(self):
        '''

        :return:
        '''

        if not self.is_init:
            return

        # initialisations
        if self.is_exc:
            # determines if not all values have been selected
            f_keys, f_values = [x[2] for x in self.fields], [x[3] for x in self.fields]
            is_ok = not np.any([len(self.f_data[fk]) == len(fv) for fk, fv in zip(f_keys, f_values)])

            # retrieves the save button object and determines if all paths are correct
            hUpdateOnly = self.h_grpbx[self.n_grp][0].findChild(QPushButton, 'update_exc_filter')
            if hUpdateOnly is not None:
                hUpdateOnly.setEnabled(is_ok)
        else:
            # determines if at least one has been selected
            is_ok = all([(len(fv) > 0) for fv in list(self.f_data.values())])

            # retrieves the save button object and determines if all paths are correct
            hUpdateOnly = self.h_grpbx[self.n_grp][0].findChild(QPushButton, 'update_filter_only')
            if hUpdateOnly is not None:
                hUpdateOnly.setEnabled(is_ok)

            # retrieves the save button object and determines if all paths are correct
            hUpdatePlot = self.h_grpbx[self.n_grp][0].findChild(QPushButton, 'update_filter_plot')
            if hUpdatePlot is not None:
                hUpdatePlot.setEnabled(is_ok)

    ####################################
    ####     CALLBACK FUNCTIONS     ####
    ####################################

    def checkComboUpdate(self, h_obj, chk_list, i_grp, index):
        '''

        :return:
        '''

        #
        item, i_sel = h_obj.model().itemFromIndex(index), index.row()
        is_Checked = item.checkState() == Qt.Checked

        if is_Checked:
            if chk_list[i_sel - 1] == 'All':
                self.f_data[self.fields[i_grp][2]] = ['All']
            else:
                self.f_data[self.fields[i_grp][2]].append(chk_list[i_sel - 1])
        else:
            i_rmv = self.f_data[self.fields[i_grp][2]].index(chk_list[i_sel - 1])
            self.f_data[self.fields[i_grp][2]].pop(i_rmv)

        #
        if len(self.f_data[self.fields[i_grp][2]]):
            first_line = '--- Selection: {0} ---'.format(', '.join(self.f_data[self.fields[i_grp][2]]))
        else:
            first_line = '--- Selection: None ---'

        #
        h_obj.model().item(0).setText(first_line)
        self.set_button_enabled_props()

    def update_exc_filter_only(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        self.can_close = True
        self.close()

    def update_filter_only(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        self.f_data['is_ud'] = [self.is_ud]
        self.can_close = True
        self.close()

    def update_filter_plot(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        self.f_data['is_ud'] = [self.is_ud]
        self.can_close = True
        self.update_plot = True
        self.close()

    def user_cancel(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        self.is_ok = False
        self.can_close = True
        self.close()

    def closeEvent(self, evnt):
        '''

        :param evnt:
        :return:
        '''

        if self.can_close:
            super(RotationFilter, self).closeEvent(evnt)
        else:
            evnt.ignore()


########################################################################################################################
########################################################################################################################

class RotationFilteredData(object):
    def __init__(self, data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, is_ud,
                 t_ofs=None, t_phase=None):

        # initialisations
        self.e_str = None
        self.is_ok = True
        self.is_ud = is_ud
        self._t_ofs = t_ofs
        self._t_phase = t_phase
        self.plot_exp_name = plot_exp_name
        self.plot_all_expt = plot_all_expt
        self.plot_scope = plot_scope

        # sets the phase labels based on the experiment stimuli type
        if self.is_ud:
            self.phase_lbl = ['Baseline', 'Stimuli']
        else:
            self.phase_lbl = ['Baseline', 'Clockwise', 'Counter-Clockwise']

        # sets the rotation filter (depending if one has been previously set)
        if rot_filt is None:
            self.rot_filt = cf.init_rotation_filter_data(self.is_ud)
        else:
            self.rot_filt = rot_filt

        # sets the other fields
        self.i_expt0 = None
        self.n_phase = len(self.phase_lbl)
        self.is_single_cell = plot_scope == 'Individual Cell'

        #
        if data.cluster is None:
            self.n_expt = 1 + plot_all_expt * (len(data._cluster) - 1)
        else:
            self.n_expt = 1 + plot_all_expt * (len(data.cluster) - 1)

        # applies the filter and sets up the other plotting field values
        self.apply_rotation_filter(data)
        self.set_spike_arrays(data, i_cluster)
        self.set_legend_str()
        self.set_final_data_arrays()

    #####################################
    ####    MAIN FILTER FUNCTIONS    ####
    #####################################

    def apply_rotation_filter(self, data):
        '''

        :return:
        '''

        # applies the rotational filter to the spike time dataset
        if self.plot_scope == 'Individual Cell':
            # case is filtering on a single cell level
            expt_filt_lvl = 0
            self.is_single_cell = True
        else:
            # case is filtering on a whole cell/multiple experiment level
            expt_filt_lvl = 1 + self.plot_all_expt
            self.is_single_cell = False

        # sets the experiment name (set to none if filtering all experiments)
        exp_name = self.plot_exp_name if expt_filt_lvl < 2 else None

        # applies all unique filters to the loaded experiments
        self.t_spike0, self.wvm_para, self.trial_ind, self.clust_ind, self.i_expt0, self.f_perm, self.f_key, \
                    self.rot_filt_tot = rot.apply_rot_filter(data, self.rot_filt, expt_filt_lvl, exp_name)

        # determines the number of plots to be displayed
        self.n_filt = len(self.rot_filt_tot)

    def set_spike_arrays(self, data, i_cluster):
        '''

        :param data:
        :param i_cluster:
        :return:
        '''

        # sets the experiment indices
        clust_ind, trial_ind, e_str = self.clust_ind, self.trial_ind, None

        if len(clust_ind) == 0:
            # if the cluster index is not valid, then output an error to screen
            e_str = 'The input cluster index does not have a feasible match. Please try again with a ' \
                    'different index or rotation analysis filter.'

        if self.is_single_cell:
            if i_cluster not in (clust_ind[0][0] + 1):
                # if the cluster index is not valid, then output an error to screen
                e_str = 'The input cluster index does not have a feasible match. Please try again with a ' \
                        'different index or rotation analysis filter.'
            else:
                # otherwise, set the cluster index value for the given experiment
                clust_ind = [[np.array([i_cluster-1], dtype=int)] for _ in range(self.n_filt)]
                i_expt0 = cf.get_expt_index(self.plot_exp_name, data.cluster, cf.det_valid_rotation_expt(data))
                self.i_expt0 = [np.array([i_expt0]) for _ in range(self.n_filt)]

        # if there was an error then output a message to screen and exit the function
        if e_str is not None:
            cf.show_error(e_str, 'Infeasible Cluster Indices')
            self.is_ok = False
            return

        if data.cluster is None:
            s_freq = [[data._cluster[i]['sFreq'] for i in x] for x in self.i_expt0]
        else:
            s_freq = [[data.cluster[i]['sFreq'] for i in x] for x in self.i_expt0]

        # retrieves the sampling frequencies and trial/cell count
        n_trial = [[len(x) if x is not None else 0 for x in ss] for ss in trial_ind]
        n_cell = [[len(x) if x is not None else 0 for x in ss] for ss in clust_ind]

        # sets the stimuli phase duration (depending on the trial type)
        if 'tPeriod' in self.wvm_para[0][0].dtype.names:
            # case is a sinusoidal pattern
            self.t_phase = [
                [np.floor(wv['tPeriod'][0] / 2) / jj for wv, jj in zip(wvp, sf)]
                                                    for wvp, sf in zip(self.wvm_para, s_freq)
            ]
        else:
            # case is a flat pattern
            self.t_phase = [
                [np.floor(wv['nPts'][0] / 2) / jj for wv, jj in zip(wvp, sf)]
                                                    for wvp, sf in zip(self.wvm_para, s_freq)
            ]

        # sets the cluster/channel ID flags
        if data.cluster is None:
            self.cl_id = [sum([list(np.array(data._cluster[x]['clustID'])[y])
                            for x, y in zip(i_ex, cl_ind)], []) for i_ex, cl_ind in zip(self.i_expt0, clust_ind)]
            self.ch_id = [sum([list(np.array(data._cluster[x]['chDepth'])[y])
                            for x, y in zip(i_ex, cl_ind)], []) for i_ex, cl_ind in zip(self.i_expt0, clust_ind)]
        else:
            self.cl_id = [sum([list(np.array(data.cluster[x]['clustID'])[y])
                            for x, y in zip(i_ex, cl_ind)], []) for i_ex, cl_ind in zip(self.i_expt0, clust_ind)]
            self.ch_id = [sum([list(np.array(data.cluster[x]['chDepth'])[y])
                            for x, y in zip(i_ex, cl_ind)], []) for i_ex, cl_ind in zip(self.i_expt0, clust_ind)]

        # memory allocation sets the other important values for each cell/experiment
        A, dcopy = np.empty(self.n_filt, dtype=object), copy.deepcopy
        self.n_trial, self.s_freq, self.t_spike, self.i_expt = dcopy(A), dcopy(A), dcopy(A), dcopy(A)
        for i_filt in range(self.n_filt):
            for ii, j_expt in enumerate(self.i_expt0[i_filt]):
                # sets the number of cells
                nC = n_cell[i_filt][ii]

                # stores the spike times for the current filter/experiment
                tSp = self.t_spike0[i_filt][ii] / s_freq[i_filt][ii]
                if self.is_single_cell:
                    tSp = tSp[clust_ind[i_filt][ii], :, :]
                self.t_spike[i_filt] = cf.combine_nd_arrays(self.t_spike[i_filt], tSp)

                # sets the values for the other field values
                if ii == 0:
                    # case is the storage array is empty, so assign the values
                    self.n_trial[i_filt] = np.array([n_trial[i_filt][ii]] * nC)
                    self.s_freq[i_filt] = np.array([s_freq[i_filt][ii]] * nC)
                    self.i_expt[i_filt] = np.array([j_expt] * nC, dtype=int)
                else:
                    # otherwise, append the new values to the existing arrays
                    self.n_trial[i_filt] = np.append(self.n_trial[i_filt], np.array([n_trial[i_filt][ii]] * nC))
                    self.s_freq[i_filt] = np.append(self.s_freq[i_filt], np.array([s_freq[i_filt][ii]] * nC))
                    self.i_expt[i_filt] = np.append(self.i_expt[i_filt] , np.array([j_expt] * nC, dtype=int))

    def set_legend_str(self):
        '''

        :return:
        '''

        # initialisations
        f_perm = copy.deepcopy(self.f_perm)
        f_key = copy.deepcopy(self.f_key)

        # sets the legend/y-axis labels strings
        if f_perm is not None:
            # case is specific filter permutations have been set
            if ('t_type' not in f_key) and (not self.rot_filt['is_ud']):
                # if the trial has not been specifically filtered for a non uniform-drifting expt, then add
                # this field to the key
                f_key = ['t_type'] + f_key
                f_perm = np.array([[self.rot_filt['t_type'][0]] + list(x) for x in f_perm])

            # sets the final legend strings
            self.lg_str = [
                '\n'.join(self.get_rotation_names(f_perm[i, :], f_key, self.rot_filt['t_key']))
                                                            for i in range(np.size(f_perm, axis=0))
            ]
        elif self.rot_filt['is_ud'][0]:
            # if no filter permutation have been set, but is uniform-drifting, then use this as the legend string
            self.lg_str = ['UniformDrifting']
        else:
            # otherwise, set the trial type as the legend string
            self.lg_str = [self.rot_filt['t_type'][0]]

    def set_final_data_arrays(self):
        '''

        :return:
        '''

        # determines if each filter has at least one cell
        is_ok = np.array([np.size(x, axis=0) for x in self.t_spike]) > 0
        if any(np.logical_not(is_ok)):
            # if there are any filters with no cells, then reduce the class arrays
            self.n_filt = sum(is_ok)

            # numpy array reduction
            self.t_spike = self.t_spike[is_ok]
            self.t_spike0 = self.t_spike0[is_ok]
            self.i_expt = self.i_expt[is_ok]
            self.n_trial = self.n_trial[is_ok]
            self.s_freq = self.s_freq[is_ok]
            self.wvm_para = self.wvm_para[is_ok]
            self.i_expt0 = self.i_expt0[is_ok]
            self.trial_ind = self.trial_ind[is_ok]
            self.clust_ind = self.clust_ind[is_ok]
            self.f_perm = self.f_perm[is_ok, :]

            # list array reduction
            self.ch_id = list(np.array(self.ch_id))
            self.cl_id = list(np.array(self.cl_id))
            self.lg_str = list(np.array(self.lg_str)[is_ok])

            # other array reduction
            self.t_phase = [x for x, y in zip(self.t_phase, is_ok) if y]
            self.rot_filt_tot = [x for x, y in zip(self.rot_filt_tot, is_ok) if y]

        #
        if self._t_ofs is not None:
            # determines if the phase duration is greater than
            t_phase = self.t_phase[0][0]
            if (self._t_ofs + self._t_phase) > t_phase:
                self.is_ok = False
                self.e_str = 'The entered analysis duration and offset is ' \
                             'greater than the experimental phase duration:\n\n' \
                             '  * Analysis Duration + Offset = {0}s.\n * Experiment Phase Duration = {1}s.\n\n' \
                             'Enter a correct analysis duration/offset combination before re-running ' \
                             'the function.'.format(self._t_ofs + self._t_phase, np.round(t_phase, 3))
            else:
                # reduces the time spike arrays to include only the valid offset/duration
                for i_filt in range(self.n_filt):
                    # array dimensioning
                    t_phase0 = dcopy(self.t_phase[i_filt][0])
                    n_cell, n_trial, n_phase = np.shape(self.t_spike[i_filt])
                    self.t_phase[i_filt][0] = self._t_phase

                    #
                    for i_phase in range(n_phase):
                        for i_trial in range(n_trial):
                            for i_cell in range(n_cell):
                                # # resets the full time-spike arrays
                                # t_sp0 = self.t_spike0[i_filt][i_cell, i_trial, i_phase]
                                # jj = np.logical_and(t_sp0 >= self._t_ofs, t_sp0 <= (self._t_ofs + self._t_phase))
                                # t_sp0 = t_sp0[jj]

                                # reshapes the other time-spike arrays
                                t_sp = self.t_spike[i_filt][i_cell, i_trial, i_phase]
                                if t_sp is not None:
                                    if i_phase == 0:
                                        ii = t_sp >= (t_phase0 - self._t_phase)
                                    else:
                                        ii = np.logical_and(t_sp >= self._t_ofs, t_sp <= (self._t_ofs + self._t_phase))

                                    self.t_spike[i_filt][i_cell, i_trial, i_phase] = \
                                                            self.t_spike[i_filt][i_cell, i_trial, i_phase][ii]

    #######################################
    ####    MISCELLANEOUS FUNCTIONS    ####
    #######################################

    @staticmethod
    def get_rotation_names(f_perm, f_key, t_key):
        '''

        :param key:
        :return:
        '''

        return [y if (t_key[x] is None) else t_key[x][y] for x, y in zip(f_key, f_perm)]