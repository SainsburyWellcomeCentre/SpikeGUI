# module imports
import copy
import functools
import numpy as np

# pyqt5 module import
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QPushButton, QGridLayout, QGroupBox, QComboBox,
                             QLineEdit, QCheckBox)

# custom module import
import analysis_guis.common_func as cf

# parameters
dX = 10
dY = 10
width = 350

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

class LDASolverPara(QDialog):
    def __init__(self, main_obj, parent=None, init_data=None, other_var=None):
        # creates the object
        super(LDASolverPara, self).__init__(parent)

        # field initialisations
        self.is_ok = True
        self.can_close = False
        self.is_updating = True
        self.update_plot = False
        self.data = main_obj.get_data_fcn()
        self.rmv_fields = None
        self.f_data = init_data

        # sets the fields to be removed (if provided)
        if other_var is not None:
            self.rmv_fields = other_var['rmv_fields']

        # initialises all the other GUI objects
        self.init_filter_fields()
        self.init_dialog()
        self.init_filter_groups()

        # ensures the gui has a fixed size
        cf.set_obj_fixed_size(self, width)
        self.is_updating = False

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

        # initialises the dialog properties
        self.setObjectName("LDAParameters")
        self.setWindowTitle("LDA Solver Parameters")
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

    def init_filter_data(self):
        '''

        :return:
        '''

        self.f_data = cfcn.init_lda_solver_para()

    def init_filter_fields(self):
        '''

        :return:
        '''

        # retrieves the data clusters for each of the valid rotation experiments
        is_rot_expt = cf.det_valid_rotation_expt(self.data)
        d_clust = [x for x, y in zip(self.data._cluster, is_rot_expt) if y]

        # retrieves the trial-types from each experiment
        comp_cond = list(np.unique(cf.flat_list([list(x['rotInfo']['trial_type']) for x in d_clust])))
        # comp_cond = [x for x in t_list0 if (x != 'Black')]

        # sets the field combobox lists
        solver_type = ['eigen', 'lsqr', 'svd']
        cell_types = ['All Cells', 'Narrow Spike Cells', 'Wide Spike Cells']
        num_types = ['Min Cell Count: ', 'Min Trial Count: ']
        check_types = ['Normalise Counts? ', 'Use LDA Shrinkage? ']
        acc_types = ['Max Individual Decoding Accuracy']

        # sets the filter field parameter information
        self.fields = [
            ['', 'NumberGroup', ['n_cell_min', 'n_trial_min'], num_types, True, None],
            ['', 'CheckGroup', ['is_norm', 'use_shrinkage'], check_types, True, None],
            ['LDA Solver Type', 'ListGroup', 'solver_type', solver_type, True, ['use_shrinkage', 'svd']],
            ['Comparison Conditions', 'CheckCombo', 'comp_cond', comp_cond, True, None],
            ['Cell Signal Types', 'ListGroup', 'cell_types', cell_types, self.data.classify.is_set, None],
            ['', 'NumberGroup', ['y_acc_max'], acc_types, self.data.discrim.indiv.lda is not None, None],
        ]

        # removes any other fields (if specified)
        if self.rmv_fields is not None:
            for r_field in self.rmv_fields:
                # loops through each of the fields determining the matching parameters
                i_rmv = None
                for i in range(len(self.fields)):
                    # determines if there is a match (depending on the parameter format type)
                    if isinstance(self.fields[i][2], list):
                        # case is the parameter type is a list
                        if r_field in self.fields[i][2]:
                            # if there is a match, then set the row index to remove and exits the loop
                            i_rmv = i
                            break
                    elif self.fields[i][2] == r_field:
                        # if there is a match, then set the row index to remove and exits the loop
                        i_rmv = i
                        break

                # if there was a match, then remove that field
                if i_rmv is not None:
                    self.fields[i_rmv][-2] = False

        # removes any groups that don't have more than one query value
        for i_row in reversed(range(len(self.fields))):
            if (not self.fields[i_row][-2]):
                self.fields.pop(i_row)

        # sets the number of filter groups
        self.n_grp = len(self.fields)

    def init_filter_groups(self):
        '''

        :return:
        '''

        # memory allocation
        self.h_grpbx = np.empty((self.n_grp + 1,1), dtype=object)
        self.grp_width, n_grp = width - (2 * dX), dcopy(self.n_grp)

        # creates the main layout widget
        mainLayout = QGridLayout()

        # creates the progressbar and layout objects
        for i_grp in range(self.n_grp):
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
            # case is a check combobox group

            # retrieves the checklist strings
            chk_list = self.fields[i_grp][3]

            # determines if any fields have been selected
            any_sel = len(self.f_data[self.fields[i_grp][2]])
            if any_sel:
                first_line = '--- Selection: {0} ---'.format(', '.join(self.f_data[self.fields[i_grp][2]]))
            else:
                first_line = '--- Selection: None ---'

            # creates the checkcombobox object
            h_obj = cf.create_checkcombo(hP, None, chk_list, has_all=False, first_line=first_line)
            cb_func = functools.partial(self.checkComboUpdate, h_obj, chk_list, i_grp)
            h_obj.view().pressed.connect(cb_func)

            # sets the initial states
            if any_sel:
                for i_sel in [self.fields[i_grp][3].index(x) for x in self.f_data[self.fields[i_grp][2]]]:
                    h_obj.handleItemPressed(i_sel+1)

        elif grp_type == 'ListGroup':
            # case is a number group group

            # retrieves the list text
            list_txt = self.fields[i_grp][3]

            # creates the combobox object
            h_obj = cf.create_combobox(hP, None, list_txt)
            cb_func = functools.partial(self.listGroupUpdate, list_txt, i_grp)
            h_obj.currentIndexChanged.connect(cb_func)

            # sets the initial list selected item
            i_sel = self.fields[i_grp][3].index(self.f_data[self.fields[i_grp][2]])
            h_obj.setCurrentIndex(i_sel)

        elif grp_type == 'NumberGroup':
            # case is a number group group

            # initialisations
            n_num, n_txt = len(self.fields[i_grp][3]), self.fields[i_grp][3]
            n_val = [self.f_data[x] for x in self.fields[i_grp][2]]
            h_obj = [[] for _ in range(2 * n_num)]

            #
            for i_num in range(n_num):
                # array indices
                ind_txt, ind_num = i_num * 2, i_num * 2 + 1

                # creates the label text
                h_obj[ind_txt] = cf.create_label(hP, txt_font_bold, n_txt[i_num], dim=None, align='right')
                h_obj[ind_txt].setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

                # creates the number edit boxes and sets the callback function
                p_name = self.fields[i_grp][2][i_num]
                h_obj[ind_num] = cf.create_edit(hP, txt_font, str(n_val[i_num]), dim=None, name=p_name)
                cb_func = functools.partial(self.numGroupUpdate, h_obj[ind_num], i_grp, i_num)
                h_obj[ind_num].editingFinished.connect(cb_func)

        elif grp_type == 'CheckGroup':
            # case is a number group group

            # initialisations
            n_num, n_txt = len(self.fields[i_grp][3]), self.fields[i_grp][3]
            n_val = [self.f_data[x] for x in self.fields[i_grp][2]]
            h_obj = [[] for _ in range(n_num)]

            #
            for i_num in range(n_num):
                # creates the checkbox object and sets the callback function
                p_name = self.fields[i_grp][2][i_num]
                h_obj[i_num] = cf.create_checkbox(hP, txt_font_bold, n_txt[i_num], state=n_val[i_num], name=p_name)
                cb_func = functools.partial(self.checkGroupUpdate, h_obj[i_num], i_grp, i_num)
                h_obj[i_num].stateChanged.connect(cb_func)

        # sets the widgets into the box layout
        layout = QHBoxLayout()
        if isinstance(h_obj, list):
            for hh in h_obj:
                layout.addWidget(hh)
        else:
            layout.addWidget(h_obj)

        # sets the groupbox layout and enabled properties
        self.h_grpbx[i_grp][0].setLayout(layout)

    def get_info(self):
        '''

        :return:
        '''

        if not self.is_ok:
            # user cancelled
            return None
        else:
            # all fields were filled out correctly
            return self.f_data

    def create_control_buttons(self):
        '''

        :return:
        '''

        # initialisations
        layout = QHBoxLayout()
        b_txt = ['Update Parameters', 'Cancel']
        cb_fcn = [self.update_parameters, self.user_cancel]
        b_name = ['update_parameters', 'user_cancel']

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

    ####################################
    ####     CALLBACK FUNCTIONS     ####
    ####################################

    def checkComboUpdate(self, h_obj, chk_list, i_grp, index):
        '''

        :return:
        '''

        # initialisations
        item, i_sel = h_obj.model().itemFromIndex(index), index.row()
        is_Checked = item.checkState() == Qt.Checked

        if is_Checked:
            self.f_data[self.fields[i_grp][2]].append(chk_list[i_sel - 1])
        else:
            i_rmv = self.f_data[self.fields[i_grp][2]].index(chk_list[i_sel - 1])
            self.f_data[self.fields[i_grp][2]].pop(i_rmv)

        # sets the first line of the combobox based on the selections made
        if len(self.f_data[self.fields[i_grp][2]]):
            first_line = '--- Selection: {0} ---'.format(', '.join(self.f_data[self.fields[i_grp][2]]))
        else:
            first_line = '--- Selection: None ---'

        # resets the first line for the combobox
        h_obj.model().item(0).setText(first_line)

    def listGroupUpdate(self, chk_list, i_grp, index):
        '''

        :return:
        '''

        # updates the parameter value
        self.f_data[self.fields[i_grp][2]] = chk_list[index]

        #
        if self.fields[i_grp][-1] is not None:
            h_obj = self.find_obj_handle([QLineEdit, QCheckBox, QComboBox], self.fields[i_grp][-1][0])
            if len(h_obj):
                h_obj[0].setEnabled(chk_list[index] != self.fields[i_grp][-1][1])

    def numGroupUpdate(self, h_obj, i_grp, i_num):
        '''

        :param h_obj:
        :param i_grp:
        :return:
        '''

        # determines if updating
        if self.is_updating:
            # if so, then exit the function
            return
        else:
            # otherwise, retrieve the current text and flag that updating is occuring
            nw_str = h_obj.text()
            self.is_updating = True

        # determines if the new number is valid
        nw_val, e_str = cf.check_edit_num(nw_str, min_val=0, is_int=True)
        if e_str is None:
            # if so, then update the parameter value
            self.f_data[self.fields[i_grp][2][i_num]] = int(nw_str)
        else:
            # otherwise, revert back to the previous valid value
            h_obj.setText(str(self.f_data[self.fields[i_grp][2][i_num]]))

        # resets the update flag
        self.is_updating = False

    def checkGroupUpdate(self, h_obj, i_grp, i_num, state):
        '''

        :param h_obj:
        :param i_grp:
        :param i_num:
        :param state:
        :return:
        '''

        # updates the parameter value
        self.f_data[self.fields[i_grp][2][i_num]] = state

    def update_parameters(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        self.can_close = True
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
            super(LDASolverPara, self).closeEvent(evnt)
        else:
            evnt.ignore()

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
            h_obj += cf.flat_list([x[0].findChildren(qt, name) if x[0] is not None else [] for x in self.h_grpbx])

        # returns the final object
        return h_obj