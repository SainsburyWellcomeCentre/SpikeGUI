# module import
import os
import os.path
import numpy as np

# pyqt5 module import
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QRect, QSize, Qt
from PyQt5.QtWidgets import QDialog, QAbstractItemView, QMessageBox

# custom module import
import analysis_guis.common_func as cf
from analysis_guis.dialogs.file_dialog import FileDialogModal

# object font objects
l_font = cf.create_font_obj()
b_font = cf.create_font_obj(size=10, is_bold=True, font_weight=75)
grp_font = cf.create_font_obj(size=10, is_bold=True, font_weight=75)

# other initialisations
iconDir = os.path.join(os.getcwd(), 'analysis_guis', 'icons')

########################################################################################################################
########################################################################################################################


class LoadExpt(QDialog):
    def __init__(self, parent=None, data=[], def_dir=None):
        # creates the object
        super(LoadExpt, self).__init__(parent)

        # initialisations
        loaded_data, i_type = data._cluster, 0

        # attribute initialisations
        self.can_close = False
        self.is_ok = False
        self.def_dir = def_dir
        self.is_multi = False
        self.multi_file = True

        # sets the loaded experimental file names
        if len(loaded_data):
            # if there is previously loaded data,  then set the experiment file locations/names
            if data.multi.is_multi:
                is_mdata = '.mdata' in data.multi.names[0]
                i_type = 1 + is_mdata

                f_name0 = cf.extract_file_name(data.multi.names[0])
                self.exp_name = ['{0}.{1}'.format(f_name0, 'mdata' if is_mdata else 'mcomp')]
                self.exp_files = data.multi.files
            else:
                self.exp_name = [cf.extract_file_name(x['expFile']) for x in loaded_data]
                self.exp_files = [x['expFile'] for x in loaded_data]
        else:
            # otherwise, then set empty data arrays
            self.exp_name, self.exp_files = [], []

        # initialises the experiment/control button objects
        self.init_dialog_obj()
        self.init_expt_obj(i_type)
        self.init_control_obj()

        # ensures the gui has a fixed size
        cf.set_obj_fixed_size(self, self.width(), self.height())

        # shows the final GUI
        self.show()
        self.exec()

    #################################################
    ####     OBJECT INITIALISATION FUNCTIONS     ####
    #################################################

    def init_dialog_obj(self):
        '''

        :return:
        '''

        self.resize(430, 300)
        self.setObjectName("LoadExptData")
        self.setWindowTitle("Load Experimental Data")
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

    def init_expt_obj(self, i_type):
        '''

        :return:
        '''

        # creates the pushbutton icons
        icon_add = QIcon(os.path.join(iconDir, 'add.png'))
        icon_rmv = QIcon(os.path.join(iconDir, 'remove.png'))

        # creates the groupboxes
        self.group_expt = cf.create_groupbox(self, QRect(10, 10, 400, 230),
                                             grp_font, "Experimental Data Files", "group_expt")

        # creates the listbox object
        self.list_expt = cf.create_listbox(self.group_expt, QRect(10, 20, 341, 171), l_font,
                                           self.exp_name, "list_expt", cb_fcn=self.select_expt)
        self.list_expt.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # creates the experiment list group buttons
        self.push_add = cf.create_button(self.group_expt, QRect(361, 65, 31, 31), grp_font,"", 'push_add',
                                         icon=icon_add, tooltip='Add Experiments', cb_fcn=self.add_expts)
        self.push_rmv = cf.create_button(self.group_expt, QRect(361, 105, 31, 31), grp_font,"", 'push_rmv',
                                         icon=icon_rmv, tooltip='Remove Experiments', cb_fcn=self.remove_expts)

        # experiment type combobox
        expt_text = ['Single Processed Experimental Data File (*.cdata)',
                     'Multiple Cluster Comparison Data File (*.mcomp)',
                     'Multiple Processed Experimental Data File (*.mdata)']
        self.expt_type = cf.create_combobox(self.group_expt, l_font, expt_text, dim=QRect(10, 200, 381, 21))
        self.expt_type.setEnabled(len(self.exp_name) == 0)
        self.expt_type.setCurrentIndex(i_type)

        # sets the other button objects
        self.push_rmv.setEnabled(len(self.exp_files))
        self.push_add.setIconSize(QSize(31, 31))
        self.push_rmv.setIconSize(QSize(31, 31))

        # if a multi-data file is already loaded, then disable the add button
        if '*.mdata' in expt_text[i_type]:
            self.push_add.setEnabled(False)

    def init_control_obj(self):
        '''

        :return:
        '''

        # initialises the group object
        self.group_buttons = cf.create_groupbox(self, QRect(10, 250, 401, 41),
                                                grp_font, "", "group_buttons")

        # creates the control buttons
        self.push_continue = cf.create_button(self.group_buttons, QRect(10, 10, 186, 23), b_font,
                                              "Continue", "push_continue", cb_fcn=self.continue_click)
        self.push_cancel = cf.create_button(self.group_buttons, QRect(205, 10, 186, 23), b_font,
                                            "Cancel", "push_cancel", cb_fcn=self.cancel_click)

        # sets the other button objects
        self.push_continue.setEnabled(len(self.exp_name)>0)

    ####################################
    ####     CALLBACK FUNCTIONS     ####
    ####################################

    def add_expts(self):
        '''

        :return:
        '''

        # sets the default data directory (if already set)
        if self.def_dir is None:
            # path not set, so use current directory
            def_dir = os.getcwd()
        else:
            # otherwise, use the default path
            def_dir = self.def_dir['inputDir']

        # sets the file type based on the user's choice
        if '*.cdata' in self.expt_type.currentText():
            # case is single experiments
            self.is_multi = False
            file_type = 'Single Experiment Files (*.cdata)'

        elif '*.mdata' in self.expt_type.currentText():
            # case is multi-experiments
            self.is_multi = True
            file_type = 'Multi-Experiment Files (*.mdata)'
            self.multi_file = False

        else:
            # case is combined cluster files
            self.is_multi = True
            file_type = 'Multi-Cluster Comparison Files (*.mcomp)'

        # opens the file dialog
        file_dlg = FileDialogModal(caption='Select Data File(s)',
                                   filter=file_type,
                                   directory=def_dir,
                                   is_multi=self.multi_file)

        # loads the window and determines if the user accepts
        if (file_dlg.exec() == QDialog.Accepted):
            # if the user didn't cancel, then set the new file names
            file_info = file_dlg.selectedFiles()
            for exp_files in file_info:
                # retrieves the experiment name and determines if the file already exists in the current list
                exp_name = cf.extract_file_name(exp_files)
                if exp_name not in self.exp_name:
                    # if not, then append the experimant name to the list widget
                    self.list_expt.addItem(exp_name)

                    # appends the file name to the overall experiment name/file lists
                    self.exp_name.append(exp_name)
                    self.exp_files.append(exp_files)

            # re-enables the experiment type (if there are no experiments selected)
            if len(self.exp_name):
                self.expt_type.setEnabled(False)
                self.push_add.setEnabled(self.multi_file)

            # enables the remove/continue push-buttons
            self.push_continue.setEnabled(True)

    def remove_expts(self):
        '''

        :return:
        '''

        # prompts the user if they want to remove the selected item(s)
        u_choice = QMessageBox.question(self, 'Remove Selected Experiments?',
                                        "Are you sure you want to removed the selected experiment(s)?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if u_choice == QMessageBox.Yes:
            # if so, then reset the listbox items and the corresponding experiment file/names
            expt_select = self.list_expt.selectedItems()
            expt_rmv = [x.text() for x in expt_select]

            # removes the
            is_keep = np.array([x not in expt_rmv for x in self.exp_name])
            self.exp_name = list(np.array(self.exp_name)[is_keep])
            self.exp_files = list(np.array(self.exp_files)[is_keep])

            # removes the items from the listbox
            for item in expt_select:
                self.list_expt.takeItem(self.list_expt.row(item))

            # sets the button enabled properties
            self.push_add.setEnabled(self.is_multi or (len(self.exp_name) == 0))
            self.push_rmv.setEnabled(len(self.exp_name))
            self.expt_type.setEnabled(len(self.exp_name) == 0)
            self.push_continue.setEnabled(len(self.exp_name))

            # re-enables the experiment type (if there are no experiments selected)
            if len(self.exp_name) == 0:
                self.expt_type.setEnabled(True)

    def continue_click(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        self.is_ok = True
        self.can_close = True
        self.close()

    def cancel_click(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        self.can_close = True
        self.close()

    def select_expt(self):
        '''

        :return:
        '''

        self.push_rmv.setEnabled(len(self.list_expt.selectedItems()))

    def closeEvent(self, evnt):
        '''

        :param evnt:
        :return:
        '''

        if self.can_close:
            super(LoadExpt, self).closeEvent(evnt)
        else:
            evnt.ignore()

