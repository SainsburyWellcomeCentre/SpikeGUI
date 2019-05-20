# module import
import numpy as np

# pyqt5 module import
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QDialog

# custom module import
import analysis_guis.common_func as cf

# font objects
txt_font = cf.create_font_obj()
grp_font = cf.create_font_obj(size=10, is_bold=True, font_weight=75)
button_font = cf.create_font_obj(size=10, is_bold=True, font_weight=75)

########################################################################################################################
########################################################################################################################

class ExptCompare(QDialog):
    def __init__(self, parent=None, exp_type=None, exp_name=None, comp_data=None):
        # creates the object
        super(ExptCompare, self).__init__(parent)

        # other initialisations
        self.can_close = False
        self.is_ok = True
        self.exp_name = np.array(exp_name)

        # sets the comparison indices
        if comp_data.is_set:
            self.comp_ind = comp_data.ind
        else:
            self.comp_ind = None

        # sets the indices of the fixed/free experiments
        is_fixed = [x == 'Fixed' for x in exp_type]
        self.ind_fix, self.ind_free = np.where(is_fixed)[0], np.where(np.logical_not(is_fixed))[0]

        # initialises the gui objects
        self.init_main_window()
        self.init_control_buttons()
        self.init_expt_sel()

        # ensures the gui has a fixed size
        cf.set_obj_fixed_size(self, self.width(), self.height())

        # shows the final GUI
        self.show()
        self.exec()

    #################################################
    ####     OBJECT INITIALISATION FUNCTIONS     ####
    #################################################

    def init_main_window(self):
        '''

        :return:
        '''

        # sets the main window properties
        self.resize(510, 270)
        self.setObjectName("ExptCompare")
        self.setWindowTitle("Experiment Comparison Setup")
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

    def init_expt_sel(self):
        '''

        :return:
        '''

        # creates the groupbox objects
        self.grp_fixed = cf.create_groupbox(self, QRect(10, 10, 241, 200), grp_font,
                                            "Fixed Preparation Experiments", "grp_fixed")
        self.grp_free = cf.create_groupbox(self, QRect(260, 10, 241, 200), grp_font,
                                            "Free Preparation Experiments", "grp_free")

        # creates the listbox objects
        self.list_fixed = cf.create_listbox(self.grp_fixed, QRect(10, 20, 221, 171), txt_font,
                                            self.exp_name[self.ind_fix], "list_fixed", self.list_select)
        self.list_free = cf.create_listbox(self.grp_free, QRect(10, 20, 221, 171), txt_font,
                                           self.exp_name[self.ind_free], "list_free", self.list_select)

        # if the previous indices have already been selected, then update them
        if self.comp_ind is not None:
            self.list_fixed.setCurrentRow(np.where(self.ind_fix == self.comp_ind[0])[0][0])
            self.list_free.setCurrentRow(np.where(self.ind_free == self.comp_ind[1])[0][0])

    def init_control_buttons(self):
        '''

        :return:
        '''

        # creates the groupbox object
        self.grp_control = cf.create_groupbox(self, QRect(10, 220, 491, 41), grp_font, "", "grp_control")

        # creates the button objects
        self.push_continue = cf.create_button(self.grp_control, QRect(10, 10, 231, 23), button_font,
                                              "Continue", "grp_continue", cb_fcn=self.continue_click)
        self.push_cancel = cf.create_button(self.grp_control, QRect(250, 10, 231, 23), button_font,
                                              "Cancel", "push_cancel", cb_fcn=self.cancel_click)

        # sets the continue button enabled properties
        self.push_continue.setEnabled(False)

    ####################################
    ####     CALLBACK FUNCTIONS     ####
    ####################################

    def list_select(self):
        '''

        :return:
        '''

        # determines if an item from both lists have been select and enables/disables the continue button accordingly
        both_selected = len(self.list_fixed.selectedItems()) and len(self.list_free.selectedItems())
        self.push_continue.setEnabled(both_selected)

    def continue_click(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        self.can_close = True
        self.close()

    def cancel_click(self):
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
            super(ExptCompare, self).closeEvent(evnt)
        else:
            evnt.ignore()