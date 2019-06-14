# module import
import numpy as np

# pyqt5 module imports
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QHBoxLayout, QDialog, QFormLayout, QPushButton, QGridLayout, QGroupBox)

# custom module import
import analysis_guis.common_func as cf
import analysis_guis.calc_functions as cfcn

# font objects
txt_font = cf.create_font_obj()
txt_font_bold = cf.create_font_obj(is_bold=True, font_weight=QFont.Bold)
grp_font_sub = cf.create_font_obj(size=10, is_bold=True, font_weight=QFont.Bold)
grp_font_sub2 = cf.create_font_obj(size=9, is_bold=True, font_weight=QFont.Bold)
grp_font_main = cf.create_font_obj(size=12, is_bold=True, font_weight=QFont.Bold)

# other initialisations
dX = 10
dY = 10

# sets the style data
styleData="""
QPushButton
{
    font-size: 10;
    font-weight: bold;
}
QGroupBox
{
    font-weight: bold;
    font-size: 14;
}
# QLabel { 
#     background-color: white;
# }
"""

########################################################################################################################
########################################################################################################################

class InfoDialog(QDialog):
    def __init__(self, main_obj, parent=None, width=950, height=500, rot_filt=None):
        # creates the gui object
        super(InfoDialog, self).__init__(parent)

        # field initialisations
        self.main_obj = main_obj
        self.get_data_fcn = main_obj.get_data
        self.rot_filt = rot_filt
        self.can_close = False

        #
        self.init_gui_objects(width, height)
        self.init_all_expt_groups()
        self.create_control_buttons()

        # shows and executes the dialog box
        self.show()
        self.exec()

    def init_gui_objects(self, width, height):
        '''

        :return:
        '''

        # retrieves the loaded data object from the main gui
        self.data = self.get_data_fcn()

        # width dimensions
        self.gui_width = width
        self.grp_wid_main = self.gui_width - 2 * dX
        self.grp_wid_expt = self.grp_wid_main - 0.5 * dX
        self.grp_wid_info = self.grp_wid_expt - 0.6 * dX

        # height dimensions
        self.gui_hght = height
        self.grp_hght_main = self.gui_hght - (2*dY + 55)
        self.grp_hght_expt = self.grp_hght_main - (2 * dY)
        self.grp_hght_info = self.grp_hght_expt - (2 * dY)

        # memory allocation
        self.n_expt = len(self.data._cluster)
        self.h_expt = np.empty(self.n_expt, dtype=object)
        self.h_info = np.empty((self.n_expt, 2), dtype=object)
        self.h_grpbx = np.empty(2, dtype=object)

        # main layout object
        self.mainLayout = QGridLayout()

        # sets the final window properties
        self.setWindowTitle('Experiment Information')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setStyleSheet(styleData)

    def init_all_expt_groups(self):
        '''

        :param data:
        :return:
        '''

        # creates the tab groups for each experiment
        for i_expt in range(self.n_expt):
            self.create_expt_group(i_expt)

        # creates the tab object
        self.h_grpbx[0]= cf.create_tab(None, QRect(10, 10, self.grp_wid_main, self.grp_hght_main), None,
                                       h_tabchild=[x for x in self.h_expt],
                                       child_name=['Expt #{0}'.format(i_expt+1) for i_expt in range(self.n_expt)])
        cf.set_obj_fixed_size(self.h_grpbx[0], width=self.grp_wid_main, height=self.grp_hght_main)

        # sets the main widget into the GUI
        self.mainLayout.addWidget(self.h_grpbx[0], 0, 0)
        self.setLayout(self.mainLayout)

        # sets the gui's fixed dimensions
        cf.set_obj_fixed_size(self, width=self.gui_width, height=self.gui_hght)

    def create_control_buttons(self):
        '''

        :return:
        '''

        # initialisations
        b_txt = ['Refresh', 'Close Window']
        cb_fcn = [self.refresh_fields, self.close_window]
        b_name = ['refresh_fields', 'close_window']

        # group box object
        self.h_grpbx[1] = QGroupBox("")
        layout = QHBoxLayout()

        # creates the load config file object
        for i in range(len(b_txt)):
            # creates the button object
            hButton = QPushButton(b_txt[i])
            hButton.clicked.connect(cb_fcn[i])
            hButton.setObjectName(b_name[i])
            hButton.setAutoDefault(False)
            cf.update_obj_font(hButton, pointSize=9)

            # adds the objects to the layout
            layout.addWidget(hButton)

        # sets the box layout
        self.h_grpbx[1].setLayout(layout)
        self.mainLayout.addWidget(self.h_grpbx[1], 1, 0)

    def create_expt_group(self, i_expt):
        '''

        :param i_expt:
        :return:
        '''

        # creates the calculation/plotting parameter
        self.h_info[i_expt, 0] = cf.create_groupbox(None, QRect(10, 10, self.grp_wid_info, self.grp_hght_info),
                                                    grp_font_sub2, "", "calc_para")
        self.h_info[i_expt, 1] = cf.create_groupbox(None, QRect(10, 10, self.grp_wid_info, self.grp_hght_info),
                                                    grp_font_sub2, "", "plot_para")
        for hh in self.h_info[i_expt, :]:
            cf.set_obj_fixed_size(hh, width=self.grp_wid_info, height=self.grp_hght_info)

        # creates the tab object
        self.h_expt[i_expt]= cf.create_tab(None, QRect(5, 55, self.grp_wid_expt, self.grp_hght_expt), None,
                                           h_tabchild=[self.h_info[i_expt, 0], self.h_info[i_expt, 1]],
                                           child_name=['Experiment Info', 'Cluster Info'])
        cf.set_obj_fixed_size(self.h_expt[i_expt], width=self.grp_wid_expt, height=self.grp_hght_expt)

        # initialises the groupbox layout
        self.h_info[i_expt, 0].setLayout(QFormLayout())
        self.h_info[i_expt, 1].setLayout(QFormLayout())

        #
        self.setup_expt_info(i_expt)
        self.setup_cluster_info(i_expt)

    def setup_expt_info(self, i_expt):
        '''

        :param i_expt:
        :return:
        '''

        # retrieves the cluster data
        c_data = self.data._cluster[i_expt]

        # removes all parameters from the layout
        h_layout = self.h_info[i_expt, 0].layout()
        for i_row in range(h_layout.rowCount()):
            h_layout.removeRow(0)

        # sets the experiment information fields
        expt_info = [
            ['Experiment Name', 'name', True],
            ['Experiment Date', 'date', True],
            ['Experiment Condition', 'cond', True],
            ['Experiment Type', 'type', True],
            ['Specimen Sex', 'sex', True],
            ['Specimen Age', 'age', True],
            ['Probe Name', 'probe', True],
            ['Lesion Location', 'lesion', True],
            ['Cluster Types', 'cluster_type', True],
            ['Recording State', 'record_state', True],
            ['Recording Coordinate', 'record_coord', True],
            ['Cluster Count', 'nC', False],
            ['Experiment Duration (s)', 'tExp', False],
            ['Sampling Frequency (Hz)', 'sFreq', False],
        ]

        #
        for tt in expt_info:
            # sets the label value
            if tt[2]:
                lbl_str = '{0}'.format(eval('c_data["expInfo"]["{0}"]'.format(tt[1])))
            else:
                lbl_str = '{0}'.format(eval('c_data["{0}"]'.format(tt[1])))

            # creates the label objects
            h_lbl = cf.create_label(None, txt_font_bold, '{0}: '.format(tt[0]), align='right')
            h_lbl_str = cf.create_label(None, txt_font, lbl_str, align='left')

            # adds the widgets to the layout
            h_layout.addRow(h_lbl, h_lbl_str)

        # sets the horizontal spacer
        h_layout.setHorizontalSpacing(250)

    def setup_cluster_info(self, i_expt):
        '''

        :param i_expt:
        :return:
        '''

        # retrieves the cluster data
        c_data = self.data._cluster[i_expt]
        nC, is_fixed = c_data['nC'], c_data['expInfo']['cond'] == 'Fixed'

        # determines the indices that are excluded due to the general filter
        cl_inc = cfcn.get_inclusion_filt_indices(c_data, self.main_obj.data.exc_gen_filt)
        cl_exc = np.where(np.logical_xor(c_data['expInfo']['clInclude'], cl_inc))[0]

        # sets the experiment information fields
        cl_info = [
            ['Include?', 'special'],
            ['Cluster\nIndex', 'special'],
            ['Cluster\nID#', 'clustID'],
            ['Channel\nDepth', 'chDepth'],
            ['Channel\nDepth ({0}m)'.format(cf._mu), 'special'],
            ['Channel\nRegion', 'chRegion'],
            ['Channel\nLayer', 'chLayer'],
            ['Spiking\nFrequency', 'special'],
            ['Matching\nCluster', 'special'],
            ['Spike\nClassification', 'special'],
            ['Action\nType', 'special'],
        ]

        # removes all parameters from the layout
        h_layout = self.h_info[i_expt, 1].layout()
        for i_row in range(h_layout.rowCount()):
            h_layout.removeRow(0)

        #
        t_data = np.empty((nC, len(cl_info)), dtype=object)
        for itt, tt in enumerate(cl_info):
            # sets the label value
            if tt[1] == 'special':
                if tt[0] == 'Include?':
                    nw_data = cl_inc

                if tt[0] == 'Channel\nDepth ({0}m)'.format(cf._mu):
                    ch_map = c_data['expInfo']['channel_map']
                    nw_data = np.array([ch_map[ch_map[:, 1] == x, 3][0] for x in c_data['chDepth']]).astype(str)

                elif tt[0] == 'Cluster\nIndex':
                    nw_data = (np.array(list(range(nC))) + 1).astype(str)

                elif tt[0] == 'Spiking\nFrequency':
                    nw_data = np.array(['{:5.3f}'.format(len(x) / c_data['tExp']) for x in c_data['tSpike']])

                elif tt[0] == 'Matching\nCluster':
                    if self.data.comp.is_set:
                        #
                        data_fix, data_free = self.main_obj.get_comp_datasets(is_full=True)
                        i_fix = np.where(self.data.comp.is_accept)[0]
                        i_free = self.data.comp.i_match[self.data.comp.is_accept]

                        if is_fixed:
                            clustID = data_free['clustID']
                            i_ref, i_comp = i_fix, i_free
                        else:
                            clustID = data_fix['clustID']
                            i_ref, i_comp = i_free, i_fix

                        nw_data = np.array(['N/A'] * nC)
                        nw_data[i_ref] = np.array([clustID[x] for x in i_comp])
                        nw_data[np.logical_not(cl_inc)] = 'N/A'
                    else:
                        nw_data = np.array(['---'] * nC)

                elif tt[0] == 'Spike\nClassification':
                    if self.data.classify.class_set:
                        nw_data = np.array(['N/A'] * nC)
                        nw_data[cl_inc] = self.data.classify.grp_str[i_expt][cl_inc]
                    else:
                        nw_data = np.array(['---'] * nC)

                elif tt[0] == 'Action\nType':
                    if self.data.classify.action_set:
                        nw_data, act_str = np.array(['N/A'] * nC), np.array(['---', 'Inhibitory', 'Excitatory'])
                        nw_data[cl_inc] = act_str[self.data.classify.act_type[i_expt][cl_inc]]
                    else:
                        nw_data = np.array(['---'] * nC)

            else:
                nw_data = np.array(eval('c_data["{0}"]'.format(tt[1]))).astype(str)

            # appends the new data to the table data array
            t_data[:, itt] = nw_data

        # creates the label objects
        col_hdr = [tt[0] for tt in cl_info]
        h_table = cf.create_table(None, txt_font, data=t_data, col_hdr=col_hdr, n_row=nC, max_disprows=20,
                                  check_col=[0], check_fcn=self.includeCheck, exc_rows=cl_exc)
        h_table.verticalHeader().setVisible(False)

        nrow_table = min(15, nC)
        cf.set_obj_fixed_size(h_table, height=(40 - nrow_table) + nrow_table * 22, width=self.grp_wid_info - 2*dX)

        # adds the widgets to the layout
        h_layout.addRow(h_table)

    def includeCheck(self, i_row, i_col, state):
        '''

        :return:
        '''

        # retrieves the base inclusion indices
        i_expt = self.h_grpbx[0].currentIndex()
        cl_inc = self.data._cluster[i_expt]['expInfo']['clInclude']

        # determines the indices that are excluded due to the general filter
        cl_inc_full = cfcn.get_inclusion_filt_indices(self.data._cluster[i_expt], self.main_obj.data.exc_gen_filt)
        cl_inc_xor = np.logical_xor(cl_inc, cl_inc_full)

        # determines if the selected row is part of the general exclusion filter indices
        if cl_inc_xor[i_row]:
            # if so, then output a message and revert back to the original state
            a = 1
        else:
            # flag that an update is required and updates the inclusion flag for the cell
            self.main_obj.data.req_update = True
            cl_inc[i_row] = state > 0

    def refresh_fields(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        pass

    def close_window(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        self.can_close = True
        self.close()

    def closeEvent(self, evnt):

        if self.can_close:
            super(InfoDialog, self).closeEvent(evnt)
        else:
            evnt.ignore()