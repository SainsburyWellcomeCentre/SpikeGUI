# module import
import os
import re
import sys
import copy
import functools
import numpy as np

# pyqt5 module imports
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (QApplication, QDialog, QHBoxLayout, QPushButton, QMessageBox, QComboBox, QTableWidget,
                             QGridLayout, QGroupBox, QLabel, QStyleFactory, QLineEdit, QRadioButton, QTableWidgetItem)

# custom module import
import analysis_guis.common_func as cf
from analysis_guis.dialogs.file_dialog import FileDialogModal

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

# other initialisations
dX = 10
bSz = 24
dcopy = copy.deepcopy
iconDir = os.path.join(os.getcwd(), 'analysis_guis', 'icons')

########################################################################################################################
########################################################################################################################


class ConfigDialog(QDialog):
    def __init__(self, dlg_info, title=None, parent=None, width=1000, init_data=None, def_dir=None,
                 has_reset=True, use_first_line=True):
        # creates the gui object
        super(ConfigDialog, self).__init__(parent)

        # sets a default title if not provided
        if title is None:
            title = 'Configuration Dialog'

        # initialisations
        self.def_dir = def_dir
        self.init_data = init_data
        self.dlg_info = dlg_info
        self.is_init = False
        self.is_ok = True
        self.n_grp = len(self.dlg_info)
        self.can_close = False
        self.has_reset = has_reset
        self.init_fields()
        self.is_updating = False
        self.use_first_line = use_first_line

        # creates all the groups
        if init_data is None:
            self.get_config_info(None)
        self.init_gui_objects(width)

        # sets the final window properties
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        # resizes the GUI and sets the other style properties
        self.setModal(False)
        self.setStyleSheet(styleData)

        # ensures the gui has a fixed size
        cf.set_obj_fixed_size(self, self.width(), self.height())

        # shows and executes the dialog box
        self.show()
        self.exec()

    def init_gui_objects(self, width):
        '''

        :return:
        '''

        # memory allocation
        self.p_bar = np.empty((self.n_grp+1,1), dtype=object)

        #
        mainLayout = QGridLayout()
        if self.has_form_layout:
            h_obj = [[] for _ in range(self.dlg_info[-1][6]+1)]
            i_row = np.array([x[6] for x in self.dlg_info])

        # creates the progressbar and layout objects
        for i_grp in range(self.n_grp):
            if self.has_form_layout:
                self.create_single_group(i_grp, width, sum(i_row == i_row[i_grp]))
                h_obj[self.dlg_info[i_grp][6]].append(self.p_bar[i_grp][0])

            else:
                self.create_single_group(i_grp, width, 1.0)
                mainLayout.addWidget(self.p_bar[i_grp][0], i_grp, 0)

        # creates the control buttons
        self.create_control_buttons()
        if self.has_form_layout:
            #
            n_max = np.max([len(x) for x in h_obj])

            #
            for i_row, hh in enumerate(h_obj):
                col_span = n_max / len(hh)
                for i_widget, h_widget in enumerate(hh):
                    mainLayout.addWidget(h_widget, i_row, i_widget*col_span, 1, col_span)

            #
            mainLayout.addWidget(self.p_bar[self.n_grp][0], self.n_grp, 0, 1, n_max)
        else:
            mainLayout.addWidget(self.p_bar[self.n_grp][0], self.n_grp, 0)

        # sets the main progress-bar layout
        self.setLayout(mainLayout)

        # fixes the gui size
        cf.set_obj_fixed_size(self, width=width)
        self.is_init = True

        # sets the save button enabled properties
        self.set_button_enabled_props()

    def get_info(self):
        '''

        :return:
        '''

        if not self.is_ok:
            # user cancelled
            return None
        elif all([((len(x)>0) or not y[4]) for x,y in zip(list(self.fInfo.values()), self.dlg_info)]):
            # all fields were filled out correctly
            return self.fInfo
        else:
            # not all the fields were filled out correctly
            return None

    def create_control_buttons(self):
        '''

        :return:
        '''

        # initialisations
        b_txt = ['Save Config File', 'Reset All Fields', 'Continue', 'Cancel']
        cb_fcn = [self.save_config, self.reset_fields, self.close_window, self.user_cancel]
        b_name = ['save_config', 'reset_all', 'close_window', 'user_cancel']

        is_feas = np.ones(len(b_txt), dtype=bool)
        is_feas[0], is_feas[1] = any([x[5] for x in self.dlg_info]), self.has_reset

        # group box object
        self.p_bar[self.n_grp] = QGroupBox("")
        layout = QHBoxLayout()

        # creates the load config file object
        for i in range(len(b_txt)):
            if is_feas[i]:
                # creates the button object
                hButton = QPushButton(b_txt[i])
                hButton.clicked.connect(cb_fcn[i])
                hButton.setObjectName(b_name[i])
                hButton.setAutoDefault(False)
                cf.update_obj_font(hButton, pointSize=9)

                # adds the objects to the layout
                layout.addWidget(hButton)

        # sets the box layout
        self.p_bar[self.n_grp][0].setLayout(layout)

    def create_single_group(self, i_grp, width, mlt):
        '''

        :return:
        '''

        # object dimensioning
        h_button, f_type = None, self.dlg_info[i_grp][2]
        grp_wid = width - 2*(dX + 1)
        txt_wid = grp_wid - (4*dX + bSz)
        is_file = any([f_type == x for x in ['File', 'Directory']])

        #
        if f_type != 'TableCombo':
            grp_file = eval('self.fInfo["{0}"]'.format(self.dlg_info[i_grp][1]))

        # creates the group box object
        self.p_bar[i_grp] = QGroupBox(self.dlg_info[i_grp][0])
        cf.set_obj_fixed_size(self.p_bar[i_grp][0], width=txt_wid, fix_size=False)

        #
        if self.has_gbox_col:
            g_col = self.dlg_info[i_grp][7]
            self.p_bar[i_grp][0].setStyleSheet("color: rgb({0},{1},{2})".format(g_col[0], g_col[1], g_col[2]))

        # creates the text label object
        if is_file:
            # creates the text label
            h_obj = [QLabel(),QPushButton('')]
            h_obj[0].setAlignment(Qt.AlignLeft)
            cf.set_obj_fixed_size(h_obj[0], width=txt_wid, height=20)

            # creates the push-button
            cb_func = functools.partial(self.get_new_file, i_grp)
            h_obj[1].clicked.connect(cb_func)
            h_obj[1].setAutoDefault(False)
            cf.set_obj_fixed_size(h_obj[1], width=bSz, height=bSz)

        elif f_type == 'List':
            # case is a dropdown list
            h_obj = QComboBox()
            h_obj.addItems(self.dlg_info[i_grp][3])
            self.fInfo[self.dlg_info[i_grp][1]] = self.dlg_info[i_grp][3][0]
            cf.set_obj_fixed_size(h_obj, width=grp_wid / mlt - bSz, height=22)

            if self.init_data is None:
                h_obj.setCurrentIndex(0)
            else:
                try:
                    i_sel = self.dlg_info[i_grp][3].index(self.init_data[self.dlg_info[i_grp][1]])
                except:
                    i_sel = 0

                h_obj.setCurrentIndex(i_sel)

            # sets the callback function
            cb_func = functools.partial(self.pop_change, i_grp, self.dlg_info[i_grp][3])
            h_obj.activated.connect(cb_func)

        elif f_type == 'Radio':
            # case is a radio button group
            h_obj = [QRadioButton(x) for x in self.dlg_info[i_grp][3]]

            # initialises the checked button value
            radio_opt = self.dlg_info[i_grp][3]
            i_sel = next(i for i in range(len(radio_opt)) if self.fInfo[self.dlg_info[i_grp][1]] == radio_opt[i])
            h_obj[i_sel].setChecked(True)

            # sets the callback functions for the radio buttons in the group
            for i_obj, hh in enumerate(h_obj):
                cb_func = functools.partial(self.radio_change, i_grp, i_obj)
                hh.toggled.connect(cb_func)

        elif f_type == 'CheckCombo':
            # sets the first line
            sel_str = self.fInfo[self.dlg_info[i_grp][1]]
            if self.use_first_line:
                first_line = '--- Selection: {0} ---'.format(', '.join(sel_str))
            else:
                first_line = '--- {0} Item{1} Selected ---'.format(len(sel_str), '' if len(sel_str) == 0 else 's')

            # creates the combobox
            combocheck_text = dcopy(self.dlg_info[i_grp][3])
            h_obj = cf.create_checkcombo(None, None, combocheck_text, first_line=first_line)

            # sets the callback functions for the radio buttons in the group
            cb_func = functools.partial(self.checkcombo_change, h_obj, combocheck_text, i_grp)
            h_obj.view().pressed.connect(cb_func)

            # if there is any initial data values, then sets the flags
            if len(self.fInfo[self.dlg_info[i_grp][1]]):
                for cc_text in self.fInfo[self.dlg_info[i_grp][1]]:
                    index = combocheck_text.index(cc_text)
                    h_obj.handleItemPressed(index + 1)
            else:
                self.fInfo[self.dlg_info[i_grp][1]] = []

        elif f_type == 'TableCombo':
            # initialisations
            col_hdr = self.dlg_info[i_grp][3][0]
            col_opt = self.dlg_info[i_grp][3][2]
            n_row = self.dlg_info[i_grp][3][4]
            row_hdr = ['{0} #{1}'.format(self.dlg_info[i_grp][3][3], x+1) for x in range(n_row)]

            # creates the table combobox object
            h_obj = cf.create_tablecombo(None, None, col_opt, col_hdr, row_hdr, n_row,
                                         combo_fcn=[self.tablecombo_change,i_grp])
            self.set_table_data(h_obj, self.dlg_info[i_grp][1], col_opt)

            # sets the table callback function
            cb_func = functools.partial(self.tablecombo_change, i_grp, h_obj)
            h_obj.cellChanged.connect(cb_func)

        else:
            # case is a number/string editbox
            h_obj = QLineEdit()
            h_obj.setAlignment(Qt.AlignLeft)
            cf.set_obj_fixed_size(h_obj, width=grp_wid / mlt - bSz, height=22)

            # sets the text box callback function based on the type
            if len(self.fInfo[self.dlg_info[i_grp][1]]) == 0:
                self.fInfo[self.dlg_info[i_grp][1]] = self.dlg_info[i_grp][3]

            if f_type == 'Number':
                # case is the field is a number
                cb_func = functools.partial(self.num_change, h_obj, i_grp)
            else:
                # case is the field is a string
                cb_func = functools.partial(self.string_change, h_obj, i_grp)

            # sets the text box callback function
            h_obj.editingFinished.connect(cb_func)

        # sets the widgets into the box layout
        layout = QHBoxLayout()
        if isinstance(h_obj, list):
            for hh in h_obj:
                layout.addWidget(hh)
                cf.update_obj_font(hh, pointSize=8)
        else:
            layout.addWidget(h_obj)
            cf.update_obj_font(h_obj, pointSize=8)

        #
        if h_button is not None:
            layout.addWidget(h_button)

        # sets the groupbox layout
        self.p_bar[i_grp][0].setLayout(layout)

        # updates the object fonts
        cf.update_obj_font(self.p_bar[i_grp][0], pointSize=10, weight=QFont.Bold)

        #
        if is_file:
            self.set_group_props(grp_file, i_grp)
        elif (f_type == 'Number') or (f_type == 'String'):
            if len(self.fInfo[self.dlg_info[i_grp][1]]):
                h_obj.setText(self.fInfo[self.dlg_info[i_grp][1]])
            else:
                h_obj.setText(self.dlg_info[i_grp][3])

    def num_change(self, h_obj, i_grp):
        '''

        :param i_grp:
        :return:
        '''

        #
        if self.is_updating:
            return
        else:
            nw_str = h_obj.text()
            self.is_updating = True

        #
        nw_val, e_str = cf.check_edit_num(nw_str, min_val=0)
        if e_str is None:
            self.fInfo[self.dlg_info[i_grp][1]] = nw_str
            self.set_button_enabled_props()
        else:
            # updates the save button enabled properties
            h_obj.setText(str(self.fInfo[self.dlg_info[i_grp][1]]))

        #
        self.is_updating = False

    def radio_change(self, i_grp, i_obj, is_sel):
        '''

        :param i_grp:
        :param i_sel:
        :return:
        '''

        # updates the radio button selection (if the radio button is selected)
        if is_sel:
            self.fInfo[self.dlg_info[i_grp][1]] = self.dlg_info[i_grp][3][i_obj]

    def string_change(self, h_obj, i_grp):
        '''

        :param i_grp:
        :return:
        '''

        # updates the save button enabled properties
        self.fInfo[self.dlg_info[i_grp][1]] = h_obj.text()
        self.set_button_enabled_props()

    def pop_change(self, i_grp, popup_text, i_select):
        '''

        :param i_select:
        :return:
        '''

        # updates the save button enabled properties
        self.fInfo[self.dlg_info[i_grp][1]] = popup_text[i_select]

    def checkcombo_change(self, h_obj, checkcombo_text, i_grp, index):
        '''

        :param i_grp:
        :param index:
        :return:
        '''

        # ensures the variables are stored in a list
        if not isinstance(self.fInfo[self.dlg_info[i_grp][1]], list):
            self.fInfo[self.dlg_info[i_grp][1]] = list(self.fInfo[self.dlg_info[i_grp][1]])

        #
        i_sel = index.row()
        if h_obj.model().item(i_sel).checkState() == Qt.Checked:
            try:
                self.fInfo[self.dlg_info[i_grp][1]].append(checkcombo_text[i_sel - 1])
            except:
                a = 1
        else:
            i_remove = self.fInfo[self.dlg_info[i_grp][1]].index(checkcombo_text[i_sel - 1])
            self.fInfo[self.dlg_info[i_grp][1]].pop(i_remove)


        #
        sel_str = self.fInfo[self.dlg_info[i_grp][1]]
        if self.use_first_line:
            if len(self.fInfo[self.dlg_info[i_grp][1]]):
                first_line = '--- Selection: {0} ---'.format(', '.join(sel_str))
            else:
                first_line = '--- Selection: None ---'
        else:
            first_line = '--- {0} Item{1} Selected ---'.format(len(sel_str), '' if len(sel_str) == 0 else 's')

        # updates the save button enabled properties
        h_obj.model().item(0).setText(first_line)
        self.set_button_enabled_props()

    def tablecombo_change(self, i_grp, h_table, i_row, i_col, i_sel=None):
        '''

        :param i_grp:
        :param t_para:
        :param i_row:
        :param i_col:
        :return:
        '''

        # if already updating, then exit the function
        if self.is_updating:
            return
        else:
            self.is_updating = True

        # initialisations
        t_para, table_info = self.dlg_info[i_grp][1], self.dlg_info[i_grp][3]
        n_row, h_cell = len(self.fInfo[t_para[i_col]]), h_table.cellWidget(i_row, i_col)

        #
        if i_row > n_row:
            e_str = 'There must be an entry on previous rows before you can add new data to the table.'
            cf.show_error(e_str, 'Table Entry Error')

            # resets the cell contents based on the type
            if isinstance(h_cell, QLineEdit):
                # cell is a number/string
                h_table.setItem(i_row, i_col, QTableWidgetItem(''))
            elif isinstance(h_cell, QComboBox):
                # cell is a combobox
                h_cell.setCurrentIndex(0)
        else:
            # if the cell being editted is a number, then check the value is valid
            is_combo = isinstance(h_cell, QComboBox)
            if (not is_combo) and (table_info[1][i_col] == 'Number'):
                nw_text = h_cell.text()
                if len(nw_text):
                    _, e_str = cf.check_edit_num(nw_text, False, min_val=0)
                    if e_str is not None:
                        # if there was an error, then reset the cell value
                        if i_row == n_row:
                            # no information has been added for the row, so set an empty cell
                            h_table.setItem(i_row, i_col, QTableWidgetItem(''))
                        else:
                            # otherwise, revert back to the previous parameter value
                            h_table.setItem(i_row, i_col, QTableWidgetItem(self.fInfo[t_para[i_col]][i_row]))

                        # exits the function
                        self.is_updating = False
                        return

            # otherwise, update the cells based on the information provided
            if i_row == n_row:
                for tt in t_para:
                    self.fInfo[tt].append('')

            # updates the parameter values
            if is_combo:
                self.fInfo[t_para[i_col]][i_row] = h_cell.currentText()
            else:
                self.fInfo[t_para[i_col]][i_row] = h_cell.text()

            # determines if all of the strings are empty for a current row.
            if i_row == (n_row - 1):
                if all([len(self.fInfo[tp][i_row]) == 0 for tp in t_para]):
                    # if so, then remove the row from the data
                    for tt in t_para:
                        self.fInfo[tt].pop(i_row)

        # resets the update flag
        self.set_button_enabled_props()
        self.is_updating = False

    def get_new_file(self, i_grp):
        '''

        :param headers:
        :return:
        '''

        # prompts the user for the file/directory
        if self.dlg_info[i_grp][2] == 'Directory':
            # case is a directory
            file_dlg = FileDialogModal(caption='Select Directory',
                                       directory=self.fInfo[self.dlg_info[i_grp][1]],
                                       dir_only=True)
        else:
            # sets the default file depending on the type
            if self.dlg_info[i_grp][5] and (len(self.fInfo[self.dlg_info[i_grp][1]]) == 0) and (self.def_dir is not None):
                # case is for a config file that hasn't current been set (and there is a default directory provided)
                def_dir = self.def_dir['configDir']
            else:
                # otherwise, use the currently set value
                def_dir = os.path.dirname(self.fInfo[self.dlg_info[i_grp][1]])

            # opens the file dialog
            file_dlg = FileDialogModal(caption='Select New File',
                                       filter=self.dlg_info[i_grp][3],
                                       directory=def_dir)

        # prompts the user for the file they want to set
        if (file_dlg.exec() == QDialog.Accepted):
            # updates the field associated with the parameter
            file_info = file_dlg.selectedFiles()
            file_name = os.path.normpath(file_info[0])

            # updates the file information and the button object properties
            self.fInfo[self.dlg_info[i_grp][1]] = file_name
            self.set_group_props(file_name, i_grp)

            # if the file is a config file, then load all of the corrsponding file names
            if self.dlg_info[i_grp][5]:
                # retrieves the parameter names from the list
                para_name = [x[1] if isinstance(x, list) else [x[1]] for x in self.dlg_info]

                for i_line, line in enumerate(open(file_info[0], 'r')):
                    # determines the group that the current line belongs to
                    line_sp = line.split('|')
                    j_grp = next(i for i in range(len(para_name)) if line_sp[0] in para_name[i])

                    # splits the line into constituent components
                    if i_grp != j_grp:
                        # updates the group properties
                        fld_info = line_sp[1].strip()
                        if (len(fld_info)) and (fld_info[0] == '['):
                            # object is a table so determine the column index and sets the parameters
                            i_col = para_name[j_grp].index(line_sp[0])
                            self.fInfo[line_sp[0]] = re.findall(r"'(.*?)'", fld_info)

                            # updates the table object with the parameters
                            h_table = self.p_bar[j_grp][0].findChildren(QTableWidget)[0]
                            self.set_table_column(h_table, self.fInfo[line_sp[0]], i_col)
                        else:
                            self.fInfo[line_sp[0]] = fld_info
                            if self.dlg_info[j_grp][2] == 'List':
                                self.set_group_props(self.dlg_info[j_grp][3].index(fld_info), j_grp)
                            else:
                                self.set_group_props(fld_info, j_grp)

        # updates the save button enabled properties
        self.set_button_enabled_props()

    def load_config(self):
        '''

        :return:
        '''

        self.get_new_file(0)

    def save_config(self):
        '''

        :return:
        '''

        # retrieves the file type filter
        f_type = self.dlg_info[0][3]

        # creates a file dialog object
        file_dlg = FileDialogModal(caption='Set Configuration File',
                                   filter=f_type,
                                   is_save=True)

        # prompts the user for the file they want to set
        if (file_dlg.exec() == QDialog.Accepted):
            # updates the field associated with the configuration file
            cfig_name = os.path.normpath(cf.set_file_name(file_dlg.selectedFiles()[0], file_dlg.selectedNameFilter()))
            self.fInfo[self.dlg_info[0][1]] = cfig_name

            # if the file is valid, then output the data to file
            self.output_config_file(cfig_name)
            self.set_group_props(cfig_name, 0)

    def init_fields(self):
        '''

        :return:
        '''

        # memory allocation
        self.fInfo = {}

        self.has_gbox_col = len(self.dlg_info[0]) == 8
        self.has_form_layout = len(self.dlg_info[0]) >= 7 and isinstance(self.dlg_info[0][6], int)

        # resets the fields for all groups
        for h in self.dlg_info:
            if self.init_data is None:
                if isinstance(h[1], list):
                    for hh in h[1]:
                        self.fInfo[hh] = ""
                else:
                    self.fInfo[h[1]] = ""
            else:
                if isinstance(h[1], list):
                    for hh in h[1]:
                        self.fInfo[hh] = str(self.init_data[hh])
                elif h[2] == 'CheckCombo':
                    self.fInfo[h[1]] = self.init_data[h[1]]
                else:
                    self.fInfo[h[1]] = str(self.init_data[h[1]])

    def reset_fields(self):
        '''

        :return:
        '''

        # prompts the user if they want to reset all the fields
        u_choice = QMessageBox.question(self, 'Reset All Fields?', "Are you sure you want to reset all fields?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if u_choice == QMessageBox.No:
            return

        # resets the configuration information
        self.get_config_info(None)

        # resets the fields for all groups
        for i_grp in range(self.n_grp):
            if self.dlg_info[i_grp][2] == 'List':
                self.set_group_props(0, i_grp)
            elif self.dlg_info[i_grp][2] == 'Number':
                self.set_group_props(self.dlg_info[i_grp][3], i_grp)
            else:
                self.set_group_props("", i_grp)

        # sets the save button enabled properties
        self.set_button_enabled_props()

    def close_window(self):
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

    def set_table_column(self, h_table, data, i_col):
        '''

        :param h_table:
        :param data:
        :param i_col:
        :return:
        '''

        # sets the updating flag
        self.is_updating = True

        for i_row in range(len(data)):
            # retrieves the current cell object and determines if is a combobox object
            h_cell = h_table.cellWidget(i_row, i_col)
            if i_row == 0:
                is_combo = isinstance(h_cell, QComboBox)

            # updates the table values depending on the cell type
            if is_combo:
                # cell is a combobox
                h_cell.setCurrentIndex(h_cell.findText(data[i_row]))
            else:
                # cell is a number/string
                h_table.setItem(i_row, i_col, QTableWidgetItem(data[i_row]))

        # sets the updating flag
        self.is_updating = False

    def set_group_props(self, grp_file, i_grp, i_para=None):
        '''

        :param grp_file:
        :param i_grp:
        :return:
        '''

        # initialisations
        tt_str = ''
        f_type = self.dlg_info[i_grp][2]
        is_file = any([f_type == x for x in ['File' , 'Directory']])

        # sets the button logo/text
        if is_file and len(grp_file) == 0:
            # case is the file has not been set
            lbl_text, button_type, txt_col = '{0} Has Not Been Set!'.format(f_type), 'invalid', 'red'
        elif is_file and (not self.is_file_valid(grp_file, f_type)):
            # case is the file missing or corrupt
            lbl_text, button_type, txt_col = '{0} Is Missing Or Corrupt!'.format(f_type), 'missing', 'red'
        elif f_type == 'List':
            # case is a combobox object
            combo_index = grp_file
        else:
            # case is the
            lbl_text, tt_str, button_type, txt_col = grp_file, grp_file, 'correct', 'black'

        # sets the text properties
        if (f_type == 'List'):
            h_combo = self.p_bar[i_grp][0].findChild(QComboBox)
            h_combo.setCurrentIndex(combo_index)
        elif (f_type == 'Number') or (f_type == 'String'):
            h_text = self.p_bar[i_grp][0].findChild(QLineEdit)
            h_text.setText(lbl_text)
        else:
            h_text = self.p_bar[i_grp][0].findChild(QLabel)
            h_text.setText(cf.set_text_colour(lbl_text, col=txt_col))
            h_text.setToolTip(tt_str)

        # sets the button logo
        h_button = self.p_bar[i_grp][0].findChild(QPushButton)
        if h_button is not None:
            # updates the buttons logo
            self.set_button_logo(h_button, i_grp, button_type)

    def set_table_data(self, h_table, table_para, col_opt):
        '''

        :param h_obj:
        :param table_para:
        :param col_opt:
        :return:
        '''

        # only add table data if there are values to add
        if len(self.fInfo[table_para[0]]):
            # retrieves the group number for the table parameters
            para_name = [x[1] if isinstance(x, list) else [x[1]] for x in self.dlg_info]
            j_grp = next(i for i in range(len(para_name)) if table_para[0] in para_name[i])

            # sets the data for each of the table parameters
            for j in range(len(table_para)):
                # retrieves the column index and the data list
                i_col = self.dlg_info[j_grp][1].index(table_para[j])
                data = re.findall(r"'(.*?)'", self.fInfo[table_para[j]])

                # sets the data for the table column
                self.set_table_column(h_table, data, i_col)

    def set_button_logo(self, h_button, i_grp, type='invalid'):
        '''

        :param h_button:
        :param type:
        :return:
        '''

        # sets the icon
        if not self.dlg_info[i_grp][4]:
            qI = QIcon(os.path.join(iconDir, 'open.png'))
        elif type == 'invalid':
            qI = QIcon(os.path.join(iconDir, 'cross.png'))
        elif type == 'missing':
            qI = QIcon(os.path.join(iconDir, 'exclamation.png'))
        else:
            qI = QIcon(os.path.join(iconDir, 'tick.png'))

        # sets the button icon
        h_button.setIcon(qI)
        h_button.setIconSize(QSize(bSz, bSz))

    def set_button_enabled_props(self):
        '''

        :return:
        '''

        if not self.is_init:
            return

        # initialisations
        f_vals = list(self.fInfo.values())
        reqd_set = cf.flat_list([[x[4]] * len(x[1]) if isinstance(x[1], list) else [x[4]] for x in self.dlg_info])
        is_ok = all([(len(fv) or not r_set) if not isinstance(fv, list) else
                     (all([len(x) for x in fv]) and len(fv))
                     for fv, r_set in zip(f_vals, reqd_set)])

        # retrieves the save button object and determines if all paths are correct
        hSave = self.p_bar[self.n_grp][0].findChild(QPushButton, 'save_config')
        if hSave is not None:
            hSave.setEnabled(is_ok)

        # retrieves the close button object and determines if all paths are correct
        hClose = self.p_bar[self.n_grp][0].findChild(QPushButton, 'close_window')
        if hClose is not None:
            hClose.setEnabled(is_ok)

        # sets the reset button enabled properties
        hReset = self.p_bar[self.n_grp][0].findChild(QPushButton, 'reset_all')
        if hReset is not None:
            hReset.setEnabled(any([(len(fv) and dinfo[4]) for fv, dinfo in zip(f_vals, self.dlg_info)]))

    def get_config_info(self, cFigFile):
        '''

        :param cFigFile:
        :return:
        '''

        if (cFigFile is None) or not os.path.isfile(cFigFile):
            # no file info, or configuration file is missing
            for h in self.dlg_info:
                if h[2] != 'Number':
                    if isinstance(h[1], list):
                        for hh in h[1]:
                            self.fInfo[hh] = []
                    else:
                        self.fInfo[h[1]] = ""
        else:
            # otherwise, read the config file and set all the fields
            for line in open(cFigFile, 'r'):
                line_sp = line.split('|')
                self.fInfo[line_sp[0]] = line_sp[1].strip()

    def closeEvent(self, evnt):

        if self.can_close:
            super(ConfigDialog, self).closeEvent(evnt)
        else:
            evnt.ignore()

    def changeStyle(self, styleName):
        '''

        :param styleName:
        :return:
        '''

        QApplication.setStyle(QStyleFactory.create(styleName))
        QApplication.setPalette(QApplication.style().standardPalette())

    def output_config_file(self, cfig_file):
        '''

        :param cfig_file:
        :param f_type:
        :return:
        '''

        # creates the output file object
        out_file = open(cfig_file, 'w')

        # outputs
        for i_grp in range(self.n_grp):
            if isinstance(self.dlg_info[i_grp][1], list):
                n_para = len(self.dlg_info[i_grp][1])
                for i_name, p_name in enumerate(self.dlg_info[i_grp][1]):
                    suffix_str = '' if ((i_grp + 1) == self.n_grp) and ((i_name + 1) == n_para) else '\n'
                    nw_val = eval('self.fInfo["{0}"]'.format(p_name))
                    out_file.write('{0}|{1}{2}'.format(p_name, nw_val, suffix_str))
            else:
                suffix_str = '\n' if (i_grp + 1) < self.n_grp else ''
                nw_val = eval('self.fInfo["{0}"]'.format(self.dlg_info[i_grp][1]))
                out_file.write('{0}|{1}{2}'.format(self.dlg_info[i_grp][1], nw_val, suffix_str))

        # closes the file
        out_file.close()

    @staticmethod
    def is_file_valid(grp_file, f_type):
        '''

        :param grp_file:
        :param i_grp:
        :return:
        '''

        # determines if the file exists. if not, then return a false value
        if f_type == 'Directory':
            return os.path.isdir(grp_file)
        else:
            return os.path.isfile(grp_file)
