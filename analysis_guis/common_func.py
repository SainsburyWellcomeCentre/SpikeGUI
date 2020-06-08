# module import
import os
import re
import sys
import copy
import functools
import math as m
import numpy as np
import pandas as pd
import pickle as p
import seaborn as sns
from numpy import ndarray
from skimage import measure
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz, process

# matplotlib module import
from matplotlib.patches import Polygon
from matplotlib.text import Annotation

# pyqt5 module import
from PyQt5.QtGui import QFont, QFontMetrics, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QGroupBox, QPushButton, QListWidget, QComboBox, QMenuBar, QProgressBar, QHeaderView,
                             QMenu, QAction, QLabel, QWidget, QLineEdit, QCheckBox, QMessageBox, QTableWidget,
                             QTabWidget, QTableWidgetItem, QHBoxLayout)

#
from scipy.optimize import curve_fit
from matplotlib.colors import to_rgba_array

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects import FloatVector, BoolVector, StrVector, IntVector
from rpy2.robjects.packages import importr
from rpy2.robjects.functions import SignatureTranslatedFunction
rpy2.robjects.numpy2ri.activate()
r_stats = importr("stats")
r_pROC = importr("pROC")

#
_roc = r_pROC.roc
_ci_auc = SignatureTranslatedFunction(r_pROC.ci_auc,
                                      init_prm_translate = {'boot_n': 'boot.n', 'conf_level': 'conf.level'})
_roc_test = SignatureTranslatedFunction(r_pROC.roc_test,
                                        init_prm_translate = {'boot_n': 'boot.n'})

# lambda function declarations
lin_func = lambda x, a: a * x
lin_func_const = lambda x, a, b: a * x + b
spike_count_fcn = lambda t_sp: np.array([len(x) for x in t_sp])
swap_array = lambda x1, x2, is_swap: np.array([x if is_sw else y for x, y, is_sw in zip(x1, x2, is_swap)])
# combine_spike_freq = lambda sp_freq, i_dim: flat_list([list(sp_freq[i_filt][:, i_dim]) for i_filt in range(len(sp_freq))])
calc_rel_count = lambda x, n: np.array([sum(x == i) for i in range(n)])
convert_rgb_col = lambda col: to_rgba_array(np.array(col) / 255, 1)
sig_str_fcn = lambda x, p_value: '*' if x < p_value else ''
get_field = lambda wfm_para, f_key: np.unique(flat_list([list(x[f_key]) for x in wfm_para]))

# vectorisation function declarations
sp_freq = lambda x, t_phase: len(x) / t_phase if x is not None else 0
sp_freq_fcn = np.vectorize(sp_freq)

# unicode characters
_bullet_point = '\u2022'
_mu = '\u03bc'
_delta = '\u0394'
_plusminus = '\u00b1'

# other initialisations
t_wid_f = 0.99
dcopy = copy.deepcopy
is_linux = sys.platform == 'linux'
default_dir_file = os.path.join(os.getcwd(), 'default_dir.p')

_red, _black, _green = [140, 0, 0], [0, 0, 0], [47, 150, 0]
_blue, _gray, _light_gray, _orange = [0, 30, 150], [90, 90, 50], [200, 200, 200], [255, 110, 0]
_bright_red, _bright_cyan, _bright_purple = (249, 2, 2), (2, 241, 249), (245, 2, 249)
_bright_yellow = (249, 221, 2)

custom_col = [_bright_yellow, _bright_red, _bright_cyan, _bright_purple, _red,
              _black, _green, _blue, _gray, _light_gray, _orange]

def flat_list(l):
    '''

    :param l:
    :return:
    '''

    #
    if len(l) == 0:
        return []
    elif isinstance(l[0], list) or isinstance(l[0], ndarray):
        return [item for sublist in l for item in sublist]
    else:
        return l

def calc_rel_prop(x, n, N=None, return_counts=False, ind=None):
    '''

    :param x:
    :param n:
    :param N:
    :return:
    '''

    if ind is None:
        ind = np.arange(n)

    if return_counts:
        return np.array([sum(x == i) for i in ind])
    elif N is None:
        return 100 * np.array([sum(x == i) for i in ind]) / len(x)
    else:
        return 0 if (N == 0) else 100 * np.array([sum(x == i) for i in ind]) / N

class CheckableComboBox(QComboBox):
    def __init__(self, parent=None, has_all=False, first_line=None):
        super(CheckableComboBox, self).__init__(parent)
        self.view().pressed.connect(self.handleItemPressed)
        self.n_item = 0
        self.has_all = has_all
        self.first_line = first_line

    def addItem(self, item, can_check):
        '''

        :param item:
        :param can_check:
        :return:
        '''

        super(CheckableComboBox, self).addItem(item)
        item = self.model().item(self.count()-1,0)
        self.n_item += 1

        if can_check:
            # item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setFlags(Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
        else:
            item.setFlags(Qt.NoItemFlags)

    # def itemChecked(self, index):
    #     item = self.model().item(index, 0)
    #     return item.checkState() == Qt.Checked

    def getSelectedItems(self):
        '''

        :return:
        '''

        # initialisations
        txt_sel = []

        # retrieves the checkbox text for selected items
        for i in range(self.count()):
            item = self.model().item(i)
            if item.checkState() == Qt.Checked:
                txt_sel.append(item.text())

        # returns the selected text
        return txt_sel

    def setState(self, index, state):
        '''

        :param index:
        :param state:
        :return:
        '''

        # retrieves the item corresponding to the current index
        if (index == 0) and (self.first_line is not None):
            return

        item = self.model().item(index)
        item.setCheckState(Qt.Checked if state else Qt.Unchecked)

    def handleItemPressed(self, index, is_checked=None):
        '''

        :param index:
        :param is_checked:
        :return:
        '''

        #
        if (index == 0) and (self.first_line is not None):
            return

        #
        if isinstance(index, int):
            item, i_sel = self.model().item(index), index
        else:
            item = self.model().itemFromIndex(index)
            i_sel = item.row()

        #
        if is_checked is None:
            is_checked = (item.checkState() == Qt.Checked)
            item.setCheckState(Qt.Unchecked if is_checked else Qt.Checked)

        #
        if (i_sel == 1) and self.has_all:
            for i_item in range(2, self.n_item):
                item_new = self.model().item(i_item)
                if is_checked:
                    item_new.setFlags(Qt.ItemIsEnabled)
                else:
                    item_new.setCheckState(Qt.Unchecked)
                    item_new.setFlags(Qt.NoItemFlags)

#########################################
####    OBJECT PROPERTY FUNCTIONS    ####
#########################################

def create_font_obj(size=8, is_bold=False, font_weight=QFont.Normal):
    '''

    :param is_bold:
    :param font_weight:
    :return:
    '''

    # creates the font object
    font = QFont()

    # sets the font properties
    font.setPointSize(size)
    font.setBold(is_bold)
    font.setWeight(font_weight)

    # returns the font object
    return font


def update_obj_font(h_obj, pointSize=8, weight=QFont.Normal):
    '''

    :param hObj:
    :param pointSize:
    :param weight:
    :return:
    '''

    mainFont = h_obj.font().family()
    qF = QFont(mainFont, pointSize=pointSize, weight=weight)
    h_obj.setFont(qF)


def set_obj_fixed_size(h_obj, width=None, height=None, fix_size=True):
    '''

    '''

    # retrieves the suggested object object size
    obj_sz = h_obj.sizeHint()

    if width is None:
        width = obj_sz.width()

    if height is None:
        height = obj_sz.height()

    # resets the object size
    if fix_size:
        h_obj.setFixedSize(width, height)
    else:
        h_obj.resize(width, height)


def set_text_colour(text, col='black'):
    '''

    :param text:
    :param col:
    :return:
    '''

    return '<span style="color:{0}">{1}</span>'.format(col, text)

#####################################################
####    PYQT5 OBJECT INITIALISATION FUNCTIONS    ####
#####################################################

def create_groupbox(parent, dim, font, title, name=None):
    '''

    :param parent:
    :param dim:
    :param font:
    :param name:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the groupbox object
    h_group = QGroupBox(parent)

    # sets the object properties
    h_group.setGeometry(dim)
    h_group.setFont(font)
    h_group.setTitle(title)

    # sets the object name (if provided)
    if name is not None:
        h_group.setObjectName(name)

    # returns the group object
    return h_group


def create_label(parent, font, text, dim=None, name=None, align='left'):
    '''

    :param parent:
    :param dim:
    :param font:
    :param text:
    :param name:
    :param align:
    :return:
    '''

    # creates the label object
    h_lbl = QLabel(parent)

    # sets the label properties
    h_lbl.setFont(font)
    h_lbl.setText(text)

    # set the object dimensions (if not None)
    if dim is not None:
        h_lbl.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_lbl.setObjectName(name)

    # sets the horizontal alignment of the label
    if align == 'centre':
        h_lbl.setAlignment(Qt.AlignCenter)
    elif align == 'left':
        h_lbl.setAlignment(Qt.AlignLeft)
    else:
        h_lbl.setAlignment(Qt.AlignRight)

    # returns the label object
    return h_lbl


def create_edit(parent, font, text, dim=None, name=None, cb_fcn=None, align='centre'):
    '''

    :param font:
    :param text:
    :param dim:
    :param parent:
    :param name:
    :param cb_fcn:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the editbox object
    h_edit = QLineEdit(parent)

    # sets the label properties
    h_edit.setFont(font)
    h_edit.setText(text)

    # sets the object name (if provided)
    if name is not None:
        h_edit.setObjectName(name)

    # set the object dimensions (if not None)
    if dim is not None:
        h_edit.setGeometry(dim)

    # sets the horizontal alignment of the label
    if align == 'centre':
        h_edit.setAlignment(Qt.AlignCenter)
    elif align == 'left':
        h_edit.setAlignment(Qt.AlignLeft)
    else:
        h_edit.setAlignment(Qt.AlignRight)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_edit.editingFinished.connect(cb_fcn)

    # returns the object
    return h_edit


def create_button(parent, dim, font, text, name=None, icon=None, tooltip=None, cb_fcn=None):
    '''

    :param dim:
    :param font:
    :param name:
    :param icon:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the button object
    h_button = QPushButton(parent)

    # sets the button properties
    h_button.setFont(font)
    h_button.setText(text)

    #
    if dim is not None:
        h_button.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_button.setObjectName(name)

    # sets the icon (if provided)
    if icon is not None:
        h_button.setIcon(icon)

    # sets the tooltip string (if provided)
    if tooltip is not None:
        h_button.setToolTip(tooltip)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_button.clicked.connect(cb_fcn)

    # returns the button object
    return h_button


def create_checkbox(parent, font, text, dim=None, name=None, state=False, cb_fcn=None):
    '''

    :param parent:
    :param dim:
    :param font:
    :param text:
    :param name:
    :param state:
    :param cb_fcn:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the listbox object
    h_chk = QCheckBox(parent)

    #
    h_chk.setText(text)
    h_chk.setFont(font)
    h_chk.setChecked(state)

    # set the object dimensions (if not None)
    if dim is not None:
        h_chk.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_chk.setObjectName(name)

    # sets the callback function
    if cb_fcn is not None:
        h_chk.stateChanged.connect(cb_fcn)

    # returns the checkbox object
    return h_chk


def create_listbox(parent, dim, font, text, name=None, cb_fcn=None):
    '''

    :param parent:
    :param dim:
    :param text:
    :param name:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the listbox object
    h_list = QListWidget(parent)

    # sets the listbox object properties
    h_list.setFont(font)

    # set the object dimensions (if not None)
    if dim is not None:
        h_list.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_list.setObjectName(name)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_list.itemSelectionChanged.connect(cb_fcn)

    # sets the listbox text (if provided)
    if text is not None:
        for t in text:
            h_list.addItem(t)

    # returns the listbox object
    return h_list


def create_progressbar(parent, dim, font, text=None, init_val=0, name=None, max_val=100.0):
    '''

    :param parent:
    :param font:
    :param text:
    :param dim:
    :param init_val:
    :param name:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the listbox object
    h_pbar = QProgressBar(parent)

    # sets the listbox object properties
    h_pbar.setGeometry(dim)
    h_pbar.setFont(font)
    h_pbar.setValue(init_val)
    h_pbar.setMaximum(max_val)

    # sets the object name (if provided)
    if name is not None:
        h_pbar.setObjectName(name)

    # removes the text if not provided
    if text is None:
        h_pbar.setTextVisible(False)

    # returns the progressbar object
    return h_pbar


def create_combobox(parent, font, text, dim=None, name=None, cb_fcn=None):
    '''

    :param parent:
    :param dim:
    :param font:
    :param list:
    :param name:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the listbox object
    h_combo = QComboBox(parent)

    # sets the combobox object properties
    h_combo.setFont(font)

    # sets the object dimensions (if provided)
    if dim is not None:
        h_combo.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_combo.setObjectName(name)

    # sets the combobox text (if provided)
    if text is not None:
        for t in text:
            h_combo.addItem(t)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_combo.currentIndexChanged.connect(cb_fcn)

    # returns the listbox object
    return h_combo

def create_checkcombo(parent, font, text, dim=None, name=None, cb_fcn=None,
                      first_line='--- Select From Options List Below ---', has_all=False):
    '''

    :param parent:
    :param font:
    :param combo_opt:
    :param dim:
    :param name:
    :param cb_fcn:
    :param combo_fcn:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the listbox object
    h_chkcombo = CheckableComboBox(parent, has_all, first_line)

    # sets the combobox object properties
    h_chkcombo.setFont(font)

    # sets the object dimensions (if provided)
    if dim is not None:
        h_chkcombo.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_chkcombo.setObjectName(name)

    # sets the combobox text (if provided)
    if text is not None:
        if first_line is not None:
            text = [first_line] + text

        for i, t in enumerate(text):
            h_chkcombo.addItem(t, i>0)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_chkcombo.view().pressed.connect(cb_fcn)

    # returns the listbox object
    return h_chkcombo

def create_table(parent, font, data=None, col_hdr=None, row_hdr=None, n_row=None, dim=None, name=None,
                 cb_fcn=None, combo_fcn=None, max_disprows=3, check_col=None, check_fcn=None, exc_rows=None):
    '''

    :param parent:
    :param font:
    :param col_hdr:
    :param row_hdr:
    :param n_row:
    :param dim:
    :param name:
    :param cb_fcn:
    :param combo_fcn:
    :return:
    '''

    #
    n_col = len(col_hdr)
    if n_row is None:
        n_row = max_disprows

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the table object
    h_table = QTableWidget(parent)

    # sets the object properties
    h_table.setRowCount(n_row)
    h_table.setColumnCount(n_col)
    h_table.setFont(font)

    if col_hdr is not None:
        h_table.setHorizontalHeaderLabels(col_hdr)

    if row_hdr is not None:
        h_table.setVerticalHeaderLabels(row_hdr)

    # sets the object dimensions (if provided)
    if dim is not None:
        h_table.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_table.setObjectName(name)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_table.cellChanged.connect(cb_fcn)

    # sets the table dimensions
    h_table.setMaximumHeight(20 + min(max_disprows, n_row) * 22)
    h_table.resizeRowsToContents()

    # sets the table headers
    h_hdr = h_table.horizontalHeader()
    for i_col in range(len(col_hdr)):
        h_hdr.setSectionResizeMode(i_col, QHeaderView.Stretch)

    #
    if data is not None:
        for i_row in range(n_row):
            for i_col in range(n_col):
                if check_col is not None:
                    if i_col in check_col:
                        # creates the checkbox widget
                        h_chk = QCheckBox()
                        h_chk.setCheckState(Qt.Checked if data[i_row, i_col] else Qt.Unchecked)

                        if check_fcn is not None:
                            check_fcn_full = functools.partial(check_fcn, i_row, i_col)
                            h_chk.stateChanged.connect(check_fcn_full)

                        # creates the widget object
                        h_cell = QWidget()
                        h_layout = QHBoxLayout(h_cell)
                        h_layout.addWidget(h_chk)
                        h_layout.setAlignment(Qt.AlignCenter)
                        h_layout.setContentsMargins(0, 0, 0, 0)
                        h_cell.setLayout(h_layout)

                        # if the row is excluded
                        if exc_rows is not None:
                            if i_row in exc_rows:
                                item = QTableWidgetItem('')
                                item.setBackground(QColor(200, 200, 200))
                                h_table.setItem(i_row, i_col, item)

                                h_cell.setEnabled(False)

                        # continues to the next column
                        h_table.setCellWidget(i_row, i_col, h_cell)
                        continue

                # retrieves the current cell object and determines if is a combobox object
                item = QTableWidgetItem(data[i_row, i_col])

                # resets the background colour (if the row is excluded)
                if exc_rows is not None:
                    if i_row in exc_rows:
                        item.setBackground(QColor(200, 200, 200))

                # adds the item to the table
                item.setTextAlignment(Qt.AlignHCenter)
                h_table.setItem(i_row, i_col, item)

                # # if the column is checkable, then modify the cell item properties
                # if check_col is not None:
                #     if i_col in check_col:
                #         item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                #         item.setCheckState(Qt.Checked if data[i_row, i_col] else Qt.Unchecked)

    # returns the table object
    return h_table


def create_tablecombo(parent, font, combo_opt, col_hdr=None, row_hdr=None, n_row=None,
                      dim=None, name=None, cb_fcn=None, combo_fcn=None):
    '''

    :param parent:
    :param font:
    :param col_hdr:
    :param combo_opt:
    :param dim:
    :param name:
    :param cb_fcb:
    :return:
    '''

    #
    if n_row is None:
        n_row = 3

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the table object
    h_table = QTableWidget(parent)

    # sets the object properties
    h_table.setRowCount(n_row)
    h_table.setColumnCount(len(col_hdr))
    h_table.setFont(font)

    # sets the
    for opt_col in combo_opt:
        for i_row in range(n_row):
            # sets the combobox callback function (if provided)
            if combo_fcn is None:
                cb_fcn_combo = None
            else:
                cb_fcn_combo = functools.partial(combo_fcn[0], combo_fcn[1], h_table, i_row, opt_col)

            # creates the combo-box object
            h_combocell = create_combobox(h_table, font, combo_opt[opt_col], cb_fcn=cb_fcn_combo)

            # creates the combobox object and fills in the options
            h_table.setCellWidget(i_row, opt_col, h_combocell)

    if col_hdr is not None:
        h_table.setHorizontalHeaderLabels(col_hdr)

    if row_hdr is not None:
        h_table.setVerticalHeaderLabels(row_hdr)

    # sets the object dimensions (if provided)
    if dim is not None:
        h_table.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_table.setObjectName(name)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_table.cellChanged.connect(cb_fcn)

    #
    h_table.setMaximumHeight(20 + min(3, n_row) * 22)
    h_table.resizeRowsToContents()

    h_hdr = h_table.horizontalHeader()
    for i_col in range(len(col_hdr)):
        h_hdr.setSectionResizeMode(i_col, QHeaderView.Stretch)

    # returns the table object
    return h_table


def create_tab(parent, dim, font, h_tabchild=None, child_name=None, name=None, cb_fcn=None):
    '''

    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the tab object
    h_tab = QTabWidget(parent)

    # sets the listbox object properties
    h_tab.setGeometry(dim)
    h_tab.setFont(font)

    # adds any children widgets (if provided)
    if (h_tabchild is not None) and (child_name is not None):
        for h_tc, c_n in zip(h_tabchild, child_name):
            h_tab.addTab(h_tc, c_n)

    # sets the object name (if provided)
    if name is not None:
        h_tab.setObjectName(name)

    # sets the tab changed callback function (if provided)
    if cb_fcn is not None:
        h_tab.currentChanged.connect(cb_fcn)

    # returns the tab object
    return h_tab


def create_menubar(parent, dim, name=None):
    '''

    :param parent:
    :param dim:
    :param name:
    :return:
    '''

    # creates the menubar object
    h_menubar = QMenuBar(parent)

    # sets the menubar properties
    h_menubar.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_menubar.setObjectName(name)

    # returns the menubar object
    return h_menubar


def create_menu(parent, title, name=None):
    '''

    :param parent:
    :param title:
    :param name:
    :return:
    '''

    # creates the menu item
    h_menu = QMenu(parent)

    # sets the menu properties
    h_menu.setTitle(title)

    # sets the object name (if provided)
    if name is not None:
        h_menu.setObjectName(name)

    # returns the menu object
    return h_menu


def create_menuitem(parent, text, name=None, cb_fcn=None, s_cut=None):
    '''

    :param parent:
    :param title:
    :param name:
    :return:
    '''

    # creates the menu item object
    h_menuitem = QAction(parent)

    # sets the menuitem properties
    h_menuitem.setText(text)

    # sets the object name (if provided)
    if name is not None:
        h_menuitem.setObjectName(name)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_menuitem.triggered.connect(cb_fcn)

    # sets the callback function (if provided)
    if s_cut is not None:
        h_menuitem.setShortcut(s_cut)

    # returns the menu item object
    return h_menuitem


def delete_widget_children(h_grp):
    '''

    :param h_grp:
    :return:
    '''

    # deletes all widgets that are children to the groupbox object
    for hh in h_grp.findChildren(QWidget):
        hh.deleteLater()


#######################################
####    MISCELLANEOUS FUNCTIONS    ####
#######################################


def set_file_name(f_name, f_type):
    '''

    :param f_name:
    :param f_type:
    :return:
    '''

    f_type_ex = re.search('\(([^)]+)', f_type).group(1)
    if f_type_ex[1:] not in f_name:
        return '{0}.{1}'.format(f_name, f_type_ex[1:])
    else:
        return f_name


def check_edit_num(nw_str, is_int=False, min_val=-1e100, max_val=1e10, show_err=True):
    '''

    :param nw_str:
    :param is_int:
    :return:
    '''

    # initialisations
    nw_val, e_str = None, None

    if is_int:
        # case is the string must be a float
        try:
            nw_val = int(nw_str)
        except:
            try:
                # if there was an error, then determine if the string was a float
                nw_val = float(nw_str)
                if nw_val % 1 == 0:
                    # if the float is actually an integer, then return the value
                    nw_val, e_str = int(nw_val), 1
                else:
                    # otherwise,
                    e_str = 'Entered value is not an integer.'
            except:
                # case is the string was not a valid number
                e_str = 'Entered value is not a valid number.'
    else:
        # case is the string must be a float
        try:
            nw_val = float(nw_str)
        except:
            # case is the string is not a valid number
            e_str = 'Entered value is not a valid number.'

    # determines if the new value meets the min/max value requirements
    if nw_val is not None:
        if nw_val < min_val:
            e_str = 'Entered value must be greater than or equal to {0}'.format(min_val)
        elif nw_val > max_val:
            e_str = 'Entered value must be less than or equal to {0}'.format(max_val)
        else:
            return nw_val, e_str

    # shows the error message (if required)
    if show_err:
        show_error(e_str, 'Error!')

    # shows the error and returns a None value
    return None, e_str


def expand_dash_number(num_str):
    '''

    :param x:
    :return:
    '''

    if '-' in num_str:
        if num_str.count('-') > 1:
            return 'NaN'
        else:
            i_dash = num_str.index('-')
            return [str(x) for x in list(range(int(num_str[0:i_dash]),int(num_str[(i_dash+1):])+1))]
    else:
        return [num_str]


def calc_text_width(font, text, w_ofs=0):
    '''

    :param font:
    :param text:
    :return:
    '''

    # creates the font metrics object
    fm = QFontMetrics(font)

    # returns the text width based on the type
    if isinstance(text, list):
        # case is a list, so return the maximum width of all text strings
        return max([fm.width(t) for t in text]) + w_ofs
    else:
        # otherwise, return the width of the text string
        return fm.width(text) + w_ofs


def det_subplot_dim(n_plot):
    '''

    :param n_plot:
    :return:
    '''

    #
    return m.ceil(0.5 * (1 + m.sqrt(1 + 4 * n_plot))) - 1, m.ceil(m.sqrt(n_plot))


def setup_index_arr(ind, n_ele):
    '''
        sets the index arrays for the unique groups
    '''

    # memory allocation
    ind_grp = np.zeros(len(ind), dtype=object)

    # sets the indices of the sub-groups
    for i in range(len(ind)):
        if i == (len(ind) - 1):
            ind_grp[i] = np.array(range(ind[i], n_ele))
        else:
            ind_grp[i] = np.array(range(ind[i], ind[i+1]))

    # returns the index array
    return ind_grp


def show_error(text, title):
    '''

    :param text:
    :param title:
    :return:
    '''

    # otherwise, create the error message
    err_dlg = QMessageBox()
    err_dlg.setText(text)
    err_dlg.setWindowTitle(title)
    err_dlg.setWindowFlags(Qt.WindowStaysOnTopHint)

    # shows the final message
    err_dlg.exec()

def get_index_groups(b_arr):
    '''

    :param b_arr:
    :return:
    '''

    if not any(b_arr):
        return []
    else:
        labels = measure.label(b_arr)
        return [np.where(labels == (i + 1))[0] for i in range(max(labels))]


def expand_index_groups(i_grp, n_exp, n_max):
    '''

    :param i_grp:
    :param n_exp:
    :param n_max:
    :return:
    '''

    if len(i_grp):
        for i in range(len(i_grp)):
            i_grp[i] = np.arange(max(0, i_grp[i][0] - n_exp), min(n_max, i_grp[i][-1] + (n_exp + 1)))

    return i_grp

def det_largest_index_group(b_arr):
    '''

    :param b_arr:
    :return:
    '''

    # determines the index groups from the binary array
    i_grp = get_index_groups(b_arr)

    # returns the largest group of all the index groups
    return i_grp[np.argmax([len(x) for x in i_grp])]

def set_binary_groups(sz, ind):
    '''

    :param sz:
    :param ind:
    :return:
    '''

    if not isinstance(ind, list):
        ind = [ind]

    b_arr = np.zeros(sz, dtype=bool)
    for i in range(len(ind)):
        b_arr[ind[i]] = True

    # returns the final binary array
    return b_arr


def extract_file_name(f_file):
    '''

    :param f_name:
    :return:
    '''

    if '.' in f_file:
        f_name = os.path.basename(f_file)
        return f_name[:f_name.rfind('.')]
    else:
        return f_file

def extract_file_extn(f_file):
    '''

    :param f_name:
    :return:
    '''

    if '.' in f_file:
        f_name = os.path.basename(f_file)
        return f_name[f_name.rfind('.'):]
    else:
        return ''

def get_expt_index(exp_name, cluster, ind_arr=None):
    '''

    :param exp_name:
    :param cluster:
    :return:
    '''

    # returns the index of the experiment corresponding to the experiment with name, exp_name
    i_expt = next(i for i in range(len(cluster)) if exp_name.lower() in extract_file_name(cluster[i]['expFile']).lower())
    if ind_arr is None:
        return i_expt
    else:
        return np.where(np.where(ind_arr)[0] == i_expt)[0][0]

def get_para_dict(fcn_para, f_type):
    '''

    :return:
    '''

    return [p for p in fcn_para if fcn_para[p]['gtype'] == f_type]


def set_group_enabled_props(h_groupbox, is_enabled=True):
    '''

    :param h_groupbox:
    :return:
    '''

    for h_obj in h_groupbox.findChildren(QWidget):
        h_obj.setEnabled(is_enabled)
        if isinstance(h_obj, QGroupBox):
            set_group_enabled_props(h_obj, is_enabled)


def init_general_filter_data():
    '''

    :return:
    '''

    f_data = {
        'region_name': [],
        'record_layer': [],
        'free_ctype': [],
        'lesion': [],
        'record_state': [],
    }

    # returns the field data
    return f_data


def init_rotation_filter_data(is_ud, is_empty=False):
    '''

    :return:
    '''

    # initialisations
    t_type0 = [['Black'], ['UniformDrifting']]

    t_key = {
        't_type': None,
        'sig_type': None,
        'match_type': None,
        'region_name': None,
        'record_layer': None,
        'record_coord': None,
        'lesion': None,
        'record_state': None,
        't_freq': {'0.5': '0.5 Hz', '2.0': '2 Hz', '4.0': '4 Hz'},
        't_freq_dir': {'-1': 'CW', '1': 'CCW'},
        't_cycle': {'15': '15 Hz', '120': '120 Hz'},
        'free_ctype': None,
    }

    if is_empty:
        f_data = {
            't_type': [],
            'sig_type': [],
            'match_type': [],
            'region_name': [],
            'record_layer': [],
            'record_coord': [],
            'lesion': [],
            'record_state': [],
            't_freq': [],
            't_freq_dir': [],
            't_cycle': [],
            'free_ctype': [],
            'is_ud': [is_ud],
            't_key': t_key,
        }
    else:
        f_data = {
            't_type': t_type0[int(is_ud)],
            'sig_type': ['All'],
            'match_type': ['All'],
            'region_name': ['All'],
            'record_layer': ['All'],
            'record_coord': ['All'],
            'lesion': ['All'],
            'record_state': ['All'],
            'record_data': ['All'],
            't_freq': ['All'],
            't_freq_dir': ['All'],
            't_cycle': ['All'],
            'free_ctype': ['All'],
            'is_ud': [is_ud],
            't_key': t_key,
        }

    # returns the field data
    return f_data


def get_plot_col(n_plot=1, i_ofs=0):
    '''

    :param index:
    :return:
    '''

    def get_new_colour(index):

        #
        if index < 10:
            return 'C{0}'.format(index)
        else:
            return convert_rgb_col(custom_col[index-10])[0]

    c = []
    for i_plot in range(n_plot):
        c.append(get_new_colour(i_plot+i_ofs))

    return c


def det_valid_rotation_expt(data, is_ud=False, t_type=None, min_count=2):
    '''

    :return: 
    '''

    # determines which experiments has rotational analysis information associated with them
    is_valid = [x['rotInfo'] is not None for x in data._cluster]
    if is_ud:
        # if experiment is uniform drifting, determine if these trials were performed
        is_valid = [('UniformDrifting' in x['rotInfo']['trial_type'])
                                                    if y else False for x, y in zip(data._cluster, is_valid)]
    elif t_type is not None:
        # if the trial types are given, then ensure that at least 2 trial types are within the experiment
        is_valid = [sum([z in x['rotInfo']['trial_type'] for z in t_type]) >= min_count
                                                    if y else False for x, y in zip(data._cluster, is_valid)]

    # returns the array
    return is_valid


def det_valid_vis_expt(data, is_vis_only=False):
    '''

    :param data:
    :return:
    '''

    # determines if there are any valid uniform/motor drifting experiments currently loaded
    has_ud_expt = any(det_valid_rotation_expt(data, True))
    has_md_expt = any(det_valid_rotation_expt(data, t_type=['MotorDrifting'], min_count=1))

    # returns the boolean flags
    if is_vis_only:
        return has_ud_expt or has_md_expt
    else:
        return has_ud_expt or has_md_expt, has_ud_expt, has_md_expt


def set_axis_limits(ax, x_lim, y_lim):
    '''

    :param ax:
    :param x_lim:
    :param y_lim:
    :return:
    '''

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

def set_equal_axis_limits(ax, ind):
    '''

    :param ax:
    :param ind:
    :return:
    '''

    # initialisations
    xL, yL = [1e6, -1e6], [1e6, -1e6]

    # determines the overall x/y limits over the axis indices
    for i in ind:
        # retrieves the x/y axis limits
        x_lim, y_lim = ax[i].get_xlim(), ax[i].get_ylim()

        # determines the min/max limits from the
        xL[0], xL[1] = min(xL[0], x_lim[0]), max(xL[1], x_lim[1])
        yL[0], yL[1] = min(yL[0], y_lim[0]), max(yL[1], y_lim[1])

    # resets the axis limits
    xx = [min(xL[0], yL[0]), max(xL[1], yL[1])]
    for i in ind:
        set_axis_limits(ax[i], xx, xx)

def reset_plot_axes_limits(ax, ax_lim, ax_str, is_high):
    '''

    :param ax_lim:
    :param ax_str:
    :param is_low:
    :return:
    '''

    if ax_str == 'x':
        axL = list(ax.get_xlim())
        axL[is_high] = ax_lim
        ax.set_xlim(axL)
    elif ax_str == 'y':
        axL = list(ax.get_ylim())
        axL[is_high] = ax_lim
        ax.set_ylim(axL)
    else:
        axL = list(ax.get_zlim())
        axL[is_high] = ax_lim
        ax.set_zlim(axL)


def combine_nd_arrays(A, B, dim=1, dim_append=0):
    '''

    :param A0:
    :param A:
    :return:
    '''

    # if the original array is empty, then return the new array
    if A is None:
        return B

    #
    n_szA, n_szB = np.shape(A), np.shape(B)

    # appends columns to the original/new arrays if they are not the correct size
    if n_szA[dim] > n_szB[dim]:
        # if the new
        d_dim = n_szA[dim] - n_szB[dim]
        Bnw = np.empty([x if i != dim else d_dim for i, x in enumerate(n_szB)], dtype=object)
        B = np.append(B, Bnw, axis=dim)

    elif n_szA[dim] < n_szB[dim]:
        d_dim = n_szB[dim] - n_szA[dim]
        Anw = np.empty([x if i != dim else d_dim for i, x in enumerate(n_szA)], dtype=object)
        A = np.append(A, Anw, axis=dim)

    # returns the arrays appended across the rows
    return np.append(A, B, axis=dim_append)


def create_general_group_plot(ax, y_plt, grp_plot_type, col):
    '''

    :param ax:
    :param y_plt:
    :param grp_plot_type:
    :param col:
    :return:
    '''

    # creates the plot based on type
    if grp_plot_type == 'Stacked Bar':
        # case is a stacked bar plot
        return create_stacked_bar(ax, y_plt, col)

    else:
        # initialisations
        n_grp, n_type = len(y_plt), np.shape(y_plt[0])[1]
        xi_type = np.arange(n_type)

        if grp_plot_type in ['Violin/Swarmplot', 'Violinplot']:
            # initialisations
            vl_col = {}
            x1, x2, y = [], [], []

            # sets the pallete colours for each type
            for i_type in range(n_type):
                vl_col[i_type] = col[i_type]

            for i_grp in range(n_grp):
                # sets the violin/swarmplot dictionaries
                x1.append([i_grp] * np.prod(y_plt[i_grp].shape))
                x2.append(flat_list([[i] * len(y) for i, y in enumerate(y_plt[i_grp].T)]))
                y.append(y_plt[i_grp].T.flatten())

                # plots the separation line
                if i_grp > 0:
                    ax.plot([i_grp - 0.5] * 2, [-1e6, 1e6], 'k--')

            # sets up the plot dictionary
            _x1, _x2, y = flat_list(x1), flat_list(x2), flat_list(y)
            if grp_plot_type == 'Violin/Swarmplot':
                # sets up the violin/swarmplot dictionary
                vl_dict = setup_sns_plot_dict(ax=ax, x=_x1, y=y, inner=None, hue=_x2, palette=vl_col)
                st_dict = setup_sns_plot_dict(ax=ax, x=_x1, y=y, edgecolor='gray', hue=_x2,
                                                     split=True, linewidth=1, palette=vl_col)

                # creates the violin/swarmplot
                h_vl = sns.violinplot(**vl_dict)
                h_st = sns.stripplot(**st_dict)

                # removes the legend (if only one group)
                if n_type == 1:
                    h_vl._remove_legend(h_vl.get_legend())
                    h_st._remove_legend(h_st.get_legend())
            else:
                # sets up the violinplot dictionary
                vl_dict = setup_sns_plot_dict(ax=ax, x=_x1, y=y, palette=vl_col)
                if n_type > 1:
                    vl_dict['hue'] = _x2

                # creates the violin/swarmplot
                h_vl = sns.violinplot(**vl_dict)

                # removes the legend (if only one group)
                if n_type == 1:
                    h_vl._remove_legend(h_vl.get_legend())

            # sets the x-axis tick marks
            ax.set_xlim(ax.get_xlim())
            ax.set_xticks(np.arange(n_grp))

        else:
            # initialisations
            xi_tick = np.zeros(n_grp)

            for i_grp in range(n_grp):
                # sets the x-values for the current group
                xi = xi_type + i_grp * (n_type + 1)
                n_ex, xi_tick[i_grp] = np.shape(y_plt[i_grp])[0], np.mean(xi)

                # plots the separation line
                if i_grp > 0:
                    ax.plot([xi[0] - 1] * 2, [-1e6, 1e6], 'k--')

                # creates the graph based on the type
                if grp_plot_type == 'Separated Bar':
                    # case is a separated bar graph

                    # sets the mean/sem plot values
                    n_ex = np.sum(~np.isnan(y_plt[i_grp]), axis=0) ** 0.5
                    y_plt_mn = np.nanmean(y_plt[i_grp], axis=0)
                    y_plt_sem = np.nanstd(y_plt[i_grp], axis=0) / n_ex

                    # creates the bar graph
                    ax.bar(xi, y_plt_mn, yerr=y_plt_sem, color=col[:n_type])

                elif grp_plot_type == 'Boxplot':
                    # case is a boxplot

                    # creates the boxplot
                    if np.ndim(y_plt[i_grp]) == 1:
                        ii = ~np.isnan(y_plt[i_grp])
                        h_bbox = ax.boxplot(y_plt[i_grp][ii], positions=xi, vert=True, patch_artist=True, widths=0.9)
                    else:
                        y_plt_g = [y[~np.isnan(y)] for y in y_plt[i_grp].T]
                        h_bbox = ax.boxplot(y_plt_g, positions=xi, vert=True, patch_artist=True, widths=0.9)

                    # resets the colour of the boxplot patches
                    for i_patch, patch in enumerate(h_bbox['boxes']):
                        patch.set_facecolor(col[i_patch])

                    for h_md in h_bbox['medians']:
                        h_md.set_color('k')

            # sets the x-axis tick marks
            ax.set_xlim([-1, xi[-1] + 1])
            ax.set_xticks(xi_tick)

        # creates the
        h_plt = []
        if n_type > 1:
            for i_type in range(n_type):
                h_plt.append(ax.bar(-10, 1, color=col[i_type]))

        # returns the plot objects
        return h_plt


def create_stacked_bar(ax, Y, c):
    '''

    :param ax:
    :param Y:
    :param c:
    :return:
    '''

    # initialisations
    h_bar, xi_ind = [], np.array(range(np.size(Y, axis=1)))

    # creates/appends to the stacked bar graph
    for i_type in range(np.size(Y, axis=0)):
        if i_type == 0:
            # case is the first bar plot stack
            h_bar.append(ax.bar(xi_ind, Y[i_type, :], color=c[i_type]))
            bar_bottom = Y[i_type, :]
        else:
            # case is the other bar-plot stacks
            h_bar.append(ax.bar(xi_ind, Y[i_type, :], bottom=bar_bottom, color=c[i_type]))
            bar_bottom += Y[i_type, :]

    # sets the x-axis tick marks
    ax.set_xticks(xi_ind)

    # returns the bar graph handles
    return h_bar


def create_plot_table(ax, data, row_hdr, col_hdr, loc='bottom', bbox=None, rowColours=None,
                      colColours=None, f_sz=None, colWidths=None, cellColours=None):
    '''

    :param ax:
    :param data:
    :param row_hdr:
    :param col_hdr:
    :return:
    '''

    # creates the table object
    h_table = ax.table(cellText=data, rowLabels=row_hdr, colLabels=col_hdr, loc=loc, rowLoc='center',
                       cellLoc='center', bbox=bbox, rowColours=rowColours, colColours=colColours,
                       cellColours=cellColours, colWidths=colWidths)
    # h_table.auto_set_column_width(False)

    # sets the table font size (if provided)
    if f_sz is not None:
        h_table.auto_set_font_size(False)
        h_table.set_fontsize(f_sz)

    # returns the table object
    return h_table

def add_rowcol_sum(A):
    '''

    :param A:
    :return:
    '''


    A_csum = np.hstack((A, np.reshape(np.sum(A, axis=1), (-1, 1))))
    return np.vstack((A_csum, np.sum(A_csum, axis=0)))


def create_bubble_boxplot(ax, Y, wid=0.75, plot_median=True, s=60, X0=None, col=None):
    '''

    :param Y:
    :return:
    '''

    # initialisations
    n_plot = len(Y)
    if col is None:
        col = get_plot_col(len(Y))

    #
    for i_plot in range(n_plot):
        #
        dX = wid * (0.5 - np.random.rand(len(Y[i_plot])))
        dX -= np.mean(dX)

        # creates the bubble plot
        if X0 is None:
            X = (i_plot + 1) + dX
            ax.scatter(X, Y[i_plot], s=s, facecolors='none', edgecolors=col[i_plot])
        else:
            X = X0[i_plot] + dX
            ax.scatter(X, Y[i_plot], s=s, facecolors='none', edgecolors=col[i_plot], zorder=10)

        # plots the median line (if required)
        if plot_median:
            Ymd = np.median(Y[i_plot])
            ax.plot((i_plot + 1) + (wid / 2) * np.array([-1, 1]), Ymd * np.ones(2), linewidth=2)

    # sets the x-axis limits/ticks
    if X0 is None:
        ax.set_xlim(0.5, n_plot + 0.5)
        ax.set_xticks(np.array(range(n_plot)) + 1)


def create_connected_line_plot(ax, Y, s=60, col=None, X0=None, plot_mean=True):
    '''

    :param ax:
    :param Y:
    :param s:
    :return:
    '''

    # initialisations
    n_plot, n_cell = len(Y), len(Y[0])
    y_mn0, y_mn1 = np.mean(Y[0]), np.mean(Y[1])

    #
    if col is None:
        col = get_plot_col(len(Y))

    #
    if X0 is None:
        X = np.ones((n_cell, 2))
        X[:, 1] *= 2
    else:
        X = repmat(X0, n_cell, 1)

    # plots the connecting lines
    for i_cell in range(n_cell):
        ax.plot(X[i_cell, :], [Y[0][i_cell], Y[1][i_cell]], 'k--')

    # creates the scatter plots
    ax.scatter(X[:, 0], Y[0], s=s, facecolors='none', edgecolors=col[0], zorder=10)
    ax.scatter(X[:, 1], Y[1], s=s, facecolors='none', edgecolors=col[1], zorder=10)

    # creates the mean scatter plot points
    if plot_mean:
        ax.plot([1, 2], [y_mn0, y_mn1], 'k', linewidth=4)
        ax.scatter(1, y_mn0, s=2 * s, edgecolors=col[0], zorder=11)
        ax.scatter(2, y_mn1, s=2 * s, edgecolors=col[1], zorder=11)

    # sets the x-axis limits/ticks
    if X0 is None:
        ax.set_xlim(0.5, n_plot + 0.5)
        ax.set_xticks(np.array(range(n_plot)) + 1)


def det_reqd_cond_types(data, t_type):
    '''

    :param t_type:
    :return:
    '''

    is_rot_expt = det_valid_rotation_expt(data, t_type=t_type, min_count=1)
    return [[z for z in t_type if z in x['rotInfo']['trial_type']]
                                                for x, y in zip(data._cluster, is_rot_expt) if y]


def get_r_stats_values(r_stats_obj, f_key, is_arr=False):
    '''

    :param r_stats_obj:
    :param f_key:
    :return:
    '''

    try:
        r_stats_val = r_stats_obj[r_stats_obj.names.index(f_key)]
    except:
        r_stats_val = list(r_stats_obj)[np.where(r_stats_obj.names == f_key)[0][0]]

    if is_arr:
        return r_stats_val
    else:
        return r_stats_val[0]


def lcm(x, y):
   """This function takes two
   integers and returns the L.C.M."""

   # choose the greater number
   if x > y:
       greater = x
   else:
       greater = y

   while(True):
       if((greater % x == 0) and (greater % y == 0)):
           lcm = greater
           break

       greater += 1

   return lcm


def combine_stacks(x, y):
    '''

    :param x:
    :param y:
    :return:
    '''

    #
    n_col_x, n_col_y = np.size(x, axis=1), np.size(y, axis=1)

    #
    if n_col_x > n_col_y:
        n_row_y = np.size(y, axis=0)
        y = np.concatenate((y, np.empty((n_row_y, n_col_x - n_col_y), dtype=object)), axis=1)
    elif n_col_x < n_col_y:
        n_row_x = np.size(x, axis=0)
        x = np.concatenate((x, np.empty((n_row_x, n_col_y - n_col_x), dtype=object)), axis=1)

    #
    return np.dstack((x, y))


def calc_phase_spike_freq(r_obj):
    '''

    :param r_obj:
    :return:
    '''

    # sets the spiking frequency across all trials
    sp_f0 = [sp_freq_fcn(x, y[0]) if np.size(x, axis=0) else None for x, y in zip(r_obj.t_spike, r_obj.t_phase)]
    if r_obj.is_single_cell:
        sp_f = [np.squeeze(x) if x is not None else None for x in sp_f0]
    else:
        # if not single cell, then calculate average over all trials
        sp_f = [np.mean(x, axis=1) if x is not None else None for x in sp_f0]

    # returns the total/mean spiking frequency arrays
    return sp_f0, sp_f


def combine_spike_freq(sp_freq, i_dim):

    return flat_list([list(sp_freq[i_filt][:, i_dim]) if sp_freq[i_filt] is not None else []
                      for i_filt in range(len(sp_freq))])

def setup_spike_freq_plot_arrays(r_obj, sp_f0, sp_f, ind_type, n_sub=3, plot_trend=False, is_3d=False):
    '''

    :param sp_f0:
    :param sp_f:
    :return:
    '''

    # memory allocation
    A, i_grp = np.empty(n_sub, dtype=object), None
    s_plt, sf_trend, sf_stats = dcopy(A), dcopy(A), dcopy(A)

    # combines the all the data from each phase type
    for i_sub in range(n_sub):
        if is_3d:
            # case is a 3d scatter plot
            s_plt[i_sub] = [combine_spike_freq(sp_f, i) for i in range(3)]
        elif r_obj.is_ud:
            # case is uniform drifting
            if (i_sub + 1) == n_sub:
                # case is the CW vs CCW phase
                sp_sub = [np.vstack((sp_f[x][:, 1], sp_f[y][:, 1])).T if sp_f0[x] is not None else None
                          for x, y in zip(ind_type[0], ind_type[1])]
                sp_f0_sub = [combine_stacks(sp_f0[x][:, :, 1], sp_f0[y][:, :, 1]) if sp_f0[x] is not None else None
                             for x, y in zip(ind_type[0], ind_type[1])]
            else:
                # case is the CW/CCW vs BL phases
                sp_sub = np.array(sp_f)[ind_type[i_sub]]
                sp_f0_sub = [sp_f0[x] if sp_f0[x] is not None else [] for x in ind_type[i_sub]]

            # sets the plot values
            s_plt[i_sub] = [combine_spike_freq(sp_sub, i) for i in range(2)]

            # calculates the wilcoxon signed rank test between the baseline/stimuli phases
            # if not r_obj.is_single_cell:
            sf_stats[i_sub] = calc_spike_freq_stats(sp_f0_sub, [0, 1])

            # adds the trend-line (if selected)
            if plot_trend:
                sf_trend[i_sub] = calc_spike_freq_correlation(sp_sub, [0, 1])

            # sets the cell group indices (over each filter type)
            if i_sub == 0:
                ii = np.append(0, np.cumsum([np.size(x, axis=0) for x in sp_f0_sub]))
        else:
            # case is the default plot
            i1, i2 = 1 * (i_sub > 1), 1 + (i_sub > 0)
            s_plt[i_sub] = [combine_spike_freq(sp_f, i) for i in [i1, i2]]

            # calculates the wilcoxon signed rank test between the stimuli phases
            # if not r_obj.is_single_cell:
            sf_stats[i_sub] = calc_spike_freq_stats(sp_f0, [i1, i2])

            # adds the trend-line (if selected)
            if plot_trend:
                sf_trend[i_sub] = calc_spike_freq_correlation(sp_f, [i1, i2])

            # sets the cell group indices (over each filter type)
            if i_sub == 0:
                N = [np.size(x, axis=0) if x is not None else 0 for x in sp_f0]
                ii = np.append(0, np.cumsum(N))

    # sets the indices for each filter type grouping
    if (not is_3d):
        i_grp = [np.array(range(ii[i], ii[i + 1])) for i in range(len(ii) - 1)]

    # returns the important arrays
    return s_plt, sf_trend, sf_stats, i_grp


def calc_spike_freq_stats(sp_f0, ind, concat_results=True):
    '''

    :param sp_f0:
    :param ind:
    :return:
    '''

    # memory allocation
    n_filt = len(sp_f0)
    n_row = [np.size(x, axis=0) if (x is not None) else 0 for x in sp_f0]
    sf_stats = [np.zeros(nr) for nr in n_row]

    # calculates the p-values for each of the trials
    for i_filt in range(n_filt):
        if n_row[i_filt] > 0:
            for i_row in range(n_row[i_filt]):
                x, y = sp_f0[i_filt][i_row, :, ind[0]], sp_f0[i_filt][i_row, :, ind[1]]
                ii = np.logical_and(~np.equal(x, None), ~np.equal(y, None))
                results = r_stats.wilcox_test(FloatVector(x[ii]), FloatVector(y[ii]), paired=True, exact=True)
                sf_stats[i_filt][i_row] = get_r_stats_values(results, 'p.value')

    # returns the stats array
    if concat_results:
        return np.concatenate(sf_stats)
    else:
        return sf_stats


def calc_spike_freq_correlation(sp_f, ind):
    '''

    :param sp_f:
    :param ind:
    :param is_single_cell:
    :return:
    '''

    # memory allocation
    n_filt = np.size(sp_f, axis=0)
    sp_corr = np.nan * np.ones((n_filt, 1))

    #
    for i_filt in range(n_filt):
        # sets the x/y points for the correlation calculation
        if sp_f[i_filt] is not None:
            x, y = sp_f[i_filt][:, ind[0]], sp_f[i_filt][:, ind[1]]
            sp_corr[i_filt], _ = curve_fit(lin_func, x, y)

    # returns the correlation array
    return sp_corr

##################################################
####    ROC ANALYSIS CALCULATION FUNCTIONS    ####
##################################################

def get_roc_xy_values(roc, is_comp=None):
    '''

    :param roc:
    :return:
    '''



    # retrieves the roc coordinates and returns them in a combined array
    roc_ss, roc_sp = get_r_stats_values(roc, 'sensitivities', True), get_r_stats_values(roc, 'specificities', True)
    return np.vstack((1-np.array(roc_ss), np.array(roc_sp))).T


def get_roc_auc_value(roc):
    '''

    :param roc:
    :return:
    '''

    # returns the roc curve integral
    return get_r_stats_values(roc, 'auc')


def calc_inter_roc_significance(roc1, roc2, method, boot_n):
    '''

    :param roc1:
    :param roc2:
    :return:
    '''

    # runs the test and returns the p-value
    results = _roc_test(roc1, roc2, method=method[0].lower(), boot_n=boot_n, progress='none')
    return get_r_stats_values(results, 'p.value')


def calc_roc_curves(comp_vals, roc_type='Cell Spike Times', x_grp=None, y_grp=None, ind=[1, 2]):
    '''

    :param t_spike:
    :return:
    '''

    # sets up the x/y groupings and threshold values based on type
    if (x_grp is None) or (y_grp is None):
        if roc_type == 'Cell Spike Times':
            # case is the cell spike times

            # sets the cw/ccw trial spike arrays
            n_trial = np.sum([(x is not None) for x in comp_vals[:, 0]])
            t_sp_cc = [comp_vals[i, ind[0]] for i in range(n_trial)]       # CW trial spikes
            t_sp_ccw = [comp_vals[i, ind[1]] for i in range(n_trial)]      # CCW trial spikes

            # determines the spike counts for the cc/ccw trials
            x_grp, y_grp = spike_count_fcn(t_sp_cc), spike_count_fcn(t_sp_ccw)

        elif roc_type == 'Cell Spike Counts':
            # case is the cell spike counts

            # sets the pooled neuron preferred/non-preferred trial spike counts
            x_grp, y_grp = comp_vals[:, 0], comp_vals[:, 1]

    # sets up the roc
    nn = len(x_grp)
    roc_pred, roc_class = np.hstack((np.zeros(nn), np.ones(nn))), np.hstack((x_grp, y_grp))
    return r_pROC.roc(FloatVector(roc_pred), FloatVector(roc_class), direction = "<", quiet=True)


# def calc_cell_roc_bootstrap_wrapper(p_data):
#     '''
#
#     :param p_data:
#     :return:
#     '''
#
#     # initialisations
#     t_spike, n_boot, ind = p_data[0], p_data[1], p_data[2]
#
#     # sets the cw/ccw trial spike arrays
#     n_trial = np.sum([(x is not None) for x in t_spike[:, 0]])
#     t_sp_p1 = [t_spike[i, ind[0]] for i in range(n_trial)]  # 1st phase trial spikes
#     t_sp_p2 = [t_spike[i, ind[1]] for i in range(n_trial)]  # 2nd phase trial spikes
#
#     # determines the spike counts for the cc/ccw trials
#     n_spike = [spike_count_fcn(t_sp_p1), spike_count_fcn(t_sp_p2)]
#
#     return calc_cell_roc_bootstrap(None, n_spike, n_boot=n_boot, ind=ind)


def calc_roc_conf_intervals(p_data):
    '''

    :param roc:
    :param type:
    :param n_boot:
    :return:
    '''

    # parameters and input arguments
    roc, grp_stype, n_boot, c_lvl = p_data[0], p_data[1], p_data[2], p_data[3]

    # calculates the roc curve integral
    results = _ci_auc(roc, method=grp_stype[0].lower(), boot_n=n_boot, conf_level=c_lvl, progress='none')
    return [results[1] - results[0], results[2] - results[1]]

def calc_cell_group_types(auc_sig, stats_type):
    '''

    :param auc_sig:
    :return:
    '''

    # memory allocation

    # Cell group type convention
    #   =0 - Both MS/DS
    #   =1 - MS but not DS
    #   =2 - Not MS
    g_type = 2 * np.ones(np.size(auc_sig, axis=0), dtype=int)

    # determines which cells are motion/direction sensitive
    #  * Motion Sensitive - either (one direction only is significant), OR (both are significant AND
    #                       the CW/CCW phase difference is significant)
    n_sig = np.sum(auc_sig[:, :2], axis=1)
    is_ms = n_sig > 0

    #
    if stats_type == 'Wilcoxon Paired Test':
        # case is if phase statistics ws calculated via Wilcoxon paried test
        is_ds = np.logical_or(n_sig == 1, np.logical_and(n_sig == 2, auc_sig[:, 2]))
    else:
        # case is if phase stats was calculated using ROC analysis
        is_ds = auc_sig[:, 2]

    # sets the MS/DS and MS/Not DS indices
    g_type[np.logical_and(is_ms, is_ds)] = 0            # case is both MS/DS
    g_type[np.logical_and(is_ms, ~is_ds)] = 1           # case is MS but not DS

    # returns the group type array
    return g_type


# # calculates the table dimensions
# bbox, t_data, cell_col, c_wids = cf.calc_table_dim(self.plot_fig, 1, table_font, n_sig_grp, row_hdr,
#                                                    row_cols, g_type, t_loc='bottom')

# bbox, t_data, cell_col, c_wids = cf.add_plot_table(self.plot_fig, 1, table_font, n_sig_grp, row_hdr,
#                                                    g_type, row_cols, col_cols, table_fsize, t_loc='bottom')


# calculates the table dimensions
def add_plot_table(fig, ax, font, data, row_hdr, col_hdr, row_cols, col_cols, t_loc, cell_cols=None,
                   n_row=1, n_col=2, pfig_sz=1.0, t_ofs=0, h_title=None, p_wid=1.5, ax_pos_tbb=None):
    '''

    :param ax:
    :param font:
    :param data:
    :param row_hdr:
    :param col_hdr:
    :return:
    '''

    # initialisations
    n_line, title_hght, pWT = 0, 0, 0.5
    y0, w_gap, h_gap, cell_wid, cell_wid_row, n_row_data = 0.01, 10 * p_wid, 2, 0, 0, np.size(data, axis=0)

    # creates the font metrics object
    fm, f_sz0 = QFontMetrics(font), font.pointSize()

    # sets the axis object (if the axis index was provided)
    if isinstance(ax, int):
        ax = fig.ax[ax]

    # retrieves the bounding box position array (if not provided)
    if ax_pos_tbb is None:
        ax_pos_tbb = ax.get_tightbbox(fig.get_renderer()).bounds

    # objection dimensioning
    fig_wid, fig_hght = fig.width(), fig.height() * pfig_sz
    cell_hght0, ax_wid, ax_hght, ax_pos = fm.height(), ax.bbox.width, ax.bbox.height, ax.get_position()

    # if there is a title, then retrieve the title height
    if h_title is not None:
        fm_title = QFontMetrics(create_font_obj(size=h_title.get_fontsize()))
        title_hght = fm_title.height()

    if t_loc == 'bottom':
        # case is the table is located at the bottom of the axes

        # if there is an x-label then increment the line counter
        if ax.xaxis.label is not None:
            n_line += 1

        # if there is an x-ticklabels then increment the line counter depending on the number of lines
        h_xticklbl = ax.get_xticklabels()
        if h_xticklbl is not None:
            n_line += np.max([(1 + x._text.count('\n')) for x in h_xticklbl])

    elif t_loc == 'top':
        # case is the table is located at the top of the axes

        # case is the title
        if len(ax.title._text):
            n_line += 1

    # parameters and other dimensioning
    n_rowhdr_line, n_colhdr_line = row_hdr[0].count('\n') + 1, col_hdr[0].count('\n') + 1

    ############################################
    ####    CELL ROW/HEIGHT CALCULATIONS    ####
    ############################################

    # calculates the maximum column header width
    if col_hdr is not None:
        cell_wid = np.max([fm.width(x) for x in col_hdr]) * p_wid

    # calculates the maximum of the cell widths
    for i_row in range(n_row_data):
        cell_wid = max(cell_wid, np.max([fm.width(str(x)) for x in data[i_row, :]]) * p_wid)

    # calculates the maximum row header width
    if row_hdr is not None:
        cell_wid_row = np.max([fm.width(x) for x in row_hdr]) * p_wid

    # sets the row header/whole table widths and cell/whole table heights
    table_wid = ((cell_wid_row > 0) * (cell_wid_row + w_gap) + len(col_hdr) * (cell_wid + w_gap)) / ax_wid
    table_hght = (n_colhdr_line * cell_hght0 + (n_colhdr_line + 1) * h_gap + \
                  n_row_data * (cell_hght0 + (n_rowhdr_line + 1) * h_gap))

    # if the table width it too large, then rescale
    sp_width = get_axes_tight_bbox(fig, ax_pos_tbb, pfig_sz)[2] / ax.get_position().width
    if table_wid > sp_width:
        ptable_wid = sp_width / table_wid
        cell_wid, cell_wid_row = ptable_wid * cell_wid, ptable_wid * cell_wid_row
        table_wid = sp_width

    if t_loc == 'bottom':
        ax_bot = np.floor(ax_pos.y1 / (1 / n_row)) / n_row
        ax_y0, ax_y1 = ax_bot * fig_hght + table_hght + cell_hght0 * (1.5 + n_line), ax.bbox.y1
    elif t_loc == 'top':
        ax_top = np.ceil(ax_pos.y1 / (1 / n_row)) / n_row
        ax_y0, ax_y1 = ax.bbox.y0, ax_top * fig_hght - (table_hght + 2 * cell_hght0)
    elif t_loc == 'fixed':
        ax_y0, ax_y1 = ax.bbox.y0, ax.bbox.y1
    else:
        ax_y0 = fig_hght * np.floor(ax_pos.y0 / (1 / n_row)) / n_row
        ax_y1 = fig_hght * np.ceil(ax_pos.y1 / (1 / n_row)) / n_row

    # sets the bounding box dimensions
    ax_hght_new = pfig_sz * (ax_y1 - ax_y0) / fig_hght
    ax_fig_hght = ax_hght_new * fig_hght
    table_x0 = get_axis_scaled_xval(ax, fig, ax_pos_tbb, (1 - table_wid) / 2, pfig_sz)

    if t_loc == 'bottom':
        table_y0 = -(table_hght + (1 + pWT) * title_hght + cell_hght0 * (1 + n_line)) / ax_fig_hght
        bbox = [table_x0, table_y0, table_wid, table_hght / ax_fig_hght]
    elif t_loc == 'top':
        table_y0 = 1 + (cell_hght0 + (1 + pWT) * title_hght) / ax_fig_hght
        bbox = [table_x0, table_y0, table_wid, table_hght / ax_fig_hght]
    else:
        table_y0 = 1 - (t_ofs + table_hght + (1 + pWT) * title_hght + cell_hght0) / ax_fig_hght
        bbox = [table_x0, table_y0, table_wid, table_hght / ax_fig_hght]

    ####################################################
    ####    AXIS RE-POSITIONING & TABLE CREATION    ####
    ####################################################

    # resets the axis position to accomodate the table
    if t_loc != 'fixed':
        ax_pos_nw = [ax_pos.x0, ax_y0 / fig_hght, ax_pos.width, ax_hght_new]
        ax.set_position(ax_pos_nw)

    # resets the position of the title object
    if h_title is not None:
        x_title = get_axis_scaled_xval(ax, fig, ax_pos_tbb, 0.5, pfig_sz, False)
        h_title.set_position([x_title, table_y0 + (table_hght + pWT * title_hght) / ax_fig_hght])

    # sets the table parameters based on whether there is a row header column
    if cell_wid_row == 0:
        # case is there is no row header column
        c_wids = [cell_wid + w_gap] * len(col_hdr)
    else:
        # case is there is a row header column
        c_wids = [cell_wid_row + w_gap] + [cell_wid + w_gap] * len(col_hdr)

        #
        if cell_cols is None:
            cell_cols = np.vstack([['w'] * np.size(data, axis=1)] * np.size(data, axis=0))

        # resets the data and column header arrays
        data = np.hstack((np.array(row_hdr).reshape(-1, 1), data))
        col_hdr, col_cols = [''] + col_hdr, ['w'] + col_cols

        #
        if np.size(data, axis=0) == 1:
            cell_cols = np.array(row_cols + list(cell_cols[0]), dtype=object).reshape(-1, 1).T
        else:
            cell_cols = np.hstack((np.array(row_cols, dtype=object).reshape(-1, 1), cell_cols))

    # creates the table
    h_table = create_plot_table(ax, data, None, col_hdr, bbox=bbox, colWidths=c_wids, cellColours=cell_cols,
                                colColours=col_cols, f_sz=f_sz0)

    # removes the outline from the top-left cell
    h_table._cells[(0, 0)].set_linewidth(0)

    # returns the cell width/height
    return [h_table, table_y0, ax_fig_hght]


def get_axes_tight_bbox(fig, ax_pos_tbb, pfig_sz=1.):
    '''

    :param fig:
    :param ax:
    :return:
    '''

    # retrieves the figure width/height
    fig_wid, fig_hght = fig.width(), fig.height() * pfig_sz
    r_fig_pos = np.array([fig_wid, fig_hght, fig_wid, fig_hght])

    # returns the
    return ax_pos_tbb / r_fig_pos


def get_subplot_width(fig, ax, n_col):
    '''

    :param ax:
    :return:
    '''

    return (t_wid_f / n_col) / ax.get_position().width


def get_axis_scaled_xval(ax, fig, ax_pos_tbb, x, pfig_sz, is_scaled=True):
    '''

    :param ax:
    :param x:
    :return:
    '''

    # retrieves the axis normal/tight position vector
    ax_pos = np.array(ax.get_position().bounds)
    ax_pos_t = get_axes_tight_bbox(fig, ax_pos_tbb, pfig_sz)

    # sets the column locations (for each column)
    # pp = np.linspace(x_ofs + (ax_pos_t[0] - ax_pos[0]) / ax_pos[2],
    #                  (1 - x_ofs) + ((ax_pos_t[0] + ax_pos_t[2]) - (ax_pos[2] + ax_pos[0])) / ax_pos[2], n_col + 1)
    # i_col = int(np.floor(ax_pos[0] / (1 / n_col)) / n_col)

    # calculates the subplot axis left/right location
    x_ofs = (1 - t_wid_f) / (2 * ax_pos_t[2])
    sp_left = x_ofs + (ax_pos_t[0] - ax_pos[0]) / ax_pos[2]
    sp_right = (1 - x_ofs) + ((ax_pos_t[0] + ax_pos_t[2]) - (ax_pos[2] + ax_pos[0])) / ax_pos[2]

    # returns the scaled value
    if is_scaled:
        return sp_left + x * (sp_right - sp_left)
    else:
        return sp_left + x * (sp_right - sp_left)


def reset_axes_dim(ax, d_type, d_val, is_prop):
    '''

    :param ax:
    :param d_type:
    :param d_val:
    :param is_prop:
    :return:
    '''

    #
    ax_pos0 = ax.get_position()
    ax_pos = [ax_pos0.x0, ax_pos0.y0, ax_pos0.width, ax_pos0.height]
    i_dim = ['left', 'bottom', 'width', 'height'].index(d_type.lower())

    if is_prop:
        ax_pos[i_dim] *= (1 + d_val)
    else:
        ax_pos[i_dim] = d_val

    #
    ax.set_position(ax_pos)


def setup_trial_condition_filter(rot_filt, plot_cond):
    '''

    :param plot_cond:
    :return:
    '''

    if not isinstance(plot_cond, list):
        plot_cond = [plot_cond]

    # determines the unique trial types within the experiment that match the required list
    if len(plot_cond) == 0:
        # if the black phase is not in any of the experiments, then output an error to screen
        e_str = 'At least one trial condition type must be selected before running this function.'
        return None, e_str, 'No Trial Conditions Selected'
    else:
        t_type_exp = ['Black'] + plot_cond

    # initialises the rotation filter (if not set)
    if rot_filt is None:
        rot_filt = init_rotation_filter_data(False)

    # sets the trial types into the rotation filter
    rot_filt['t_type'] = list(np.unique(flat_list(t_type_exp)))
    if 'Black' not in rot_filt['t_type']:
        # if the black phase is not in any of the experiments, then output an error to screen
        e_str = 'The loaded experiments do not include the "Black" trial condition. To run this function ' \
                'you will need to load an experiment with this trial condition.'
        return None, e_str, 'Invalid Data For Analysis'

    elif len(rot_filt['t_type']) == 1:
        # if there are insufficient trial types in the loaded experiments, then create the error string
        e_str = 'The loaded experiments only has the "{0}" trial condition. To run this function you will ' \
                'need to load a file with the following trial condition:\n'.format(rot_filt['t_type'][0])
        for tt in plot_cond:
            e_str = '{0}\n => {1}'.format(e_str, tt)

        # outputs the error to screen and exits the function
        return None, e_str, 'Invalid Data For Analysis'

    # otherwise, return the rotational filter
    return rot_filt, None, None


def det_matching_filters(r_obj, ind):
    '''

    :param r_obj:
    :param ind:
    :return:
    '''

    # sets the candidate rotation filter dictionary
    r_filt0 = r_obj.rot_filt_tot[ind]

    # loops through each of the filter dictionaries determining the match
    for i in range(len(r_obj.rot_filt_tot)):
        # if the current index is
        if i == ind:
            continue

        # loops through each of the field values determining if they all match
        is_match = True
        for f_key in r_filt0.keys():
            # no need to consider the trial type field
            if f_key == 't_type':
                continue

            # if the field values do not match, then update the match flag and exit the loop
            if r_filt0[f_key] != r_obj.rot_filt_tot[i][f_key]:
                is_match = False
                break

        # if all the fields match, then return the matching index
        if is_match:
            return [ind, i]

def calc_ms_scores(s_plt, sf_stats, p_value=0.05):
    '''

    :param s_plt:
    :param sf_stats:
    :return:
    '''

    #
    if p_value is not None:
        # calculates the relative change for CW/CCW from baseline, and CCW to CW
        grad_CW = np.array(s_plt[0][1]) / np.array(s_plt[0][0])         # CW to BL
        grad_CCW = np.array(s_plt[1][1]) / np.array(s_plt[1][0])        # CCW to BL
        grad_CCW_CW = np.array(s_plt[1][1]) / np.array(s_plt[0][1])     # CCW to CW

        # calculates the score type for the CW/CCW phases
        sf_score = np.zeros((len(grad_CW), 3), dtype=int)

        # case is the statistical significance has already been calculated (which is the case for ROC)
        sf_score[:, 0] = (sf_stats[0] < p_value).astype(int) * (1 + (grad_CW > 1).astype(int))
        sf_score[:, 1] = (sf_stats[1] < p_value).astype(int) * (1 + (grad_CCW > 1).astype(int))

        # if both CW and CCW are significant (wrt the baseline phase), then determine from these cells which
        # cells have a significant CW/CCW difference (1 for CW, 2 for CCW, 0 otherwise):
        #   1) significant for a single direction (either CW or CCW preferred)
        #   2) significant for both, but the CCW/CW gradient is either > 1 (for CCW preferred) or < 1 (for CW preferred)
        # case is the statistical significance needs to be calculated
        both_sig = np.logical_and(sf_score[:, 0] > 0, sf_score[:, 1] > 0)
        sf_score[both_sig, 2] = (sf_stats[2][both_sig] < p_value).astype(int) * \
                                (1 + (grad_CCW_CW[both_sig] > 1).astype(int))
    else:
        # calculates the score type for the CW/CCW phases
        sf_score = np.zeros((np.size(s_plt, axis=0), 3), dtype=int)

        # case is the statistical significance needs to be calculated
        sf_score[:, 0] = sf_stats[:, 0].astype(int) * (1 + (s_plt[:, 0] > 0.5).astype(int))
        sf_score[:, 1] = sf_stats[:, 1].astype(int) * (1 + (s_plt[:, 1] > 0.5).astype(int))
        sf_score[:, 2] = sf_stats[:, 2].astype(int) * (1 + (s_plt[:, 2] > 0.5).astype(int))

    # returns the scores array
    return sf_score

def det_cell_match_indices(r_obj, ind, r_obj2=None):
    '''

    :param r_obj:
    :param ind:
    :return:
    '''

    if r_obj2 is None:
        # determines the cell id's which overlap each other
        cell_id = [10000 * r_obj.i_expt[i] + np.array(flat_list(r_obj.clust_ind[i])) for i in ind]
        id_match = np.array(list(set(cell_id[0]).intersection(set(cell_id[1]))))

        # returns the indices of the matching
        return np.searchsorted(cell_id[0], id_match), np.searchsorted(cell_id[1], id_match)
    else:
        #
        if isinstance(ind, int):
            cell_id_1 = 10000 * r_obj.i_expt[ind] + np.array(flat_list(r_obj.clust_ind[ind]))
            cell_id_2 = 10000 * r_obj2.i_expt[ind] + np.array(flat_list(r_obj2.clust_ind[ind]))
        else:
            cell_id_1 = 10000 * r_obj.i_expt[ind[0]] + np.array(flat_list(r_obj.clust_ind[ind[0]]))
            cell_id_2 = 10000 * r_obj2.i_expt[ind[1]] + np.array(flat_list(r_obj2.clust_ind[ind[1]]))

        # returns the indices of the matching cells
        id_match = np.sort(np.array(list(set(cell_id_1).intersection(set(cell_id_2)))))
        return np.searchsorted(cell_id_1, id_match), np.searchsorted(cell_id_2, id_match)

def split_unidrift_phases(data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, dt, t0=0):
    '''

    :param rot_filt:
    :return:
    '''

    from analysis_guis.dialogs.rotation_filter import RotationFilteredData

    # parameters
    rot_filt['t_freq_dir'] = ['-1', '1']

    # splits the data by the forward/reverse directions
    r_obj = RotationFilteredData(data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, True,
                                 t_ofs=t0, t_phase=dt)

    # shortens the stimuli phases to the last/first dt of the baseline/stimuli phases
    t_phase, n_filt = r_obj.t_phase[0][0], int(r_obj.n_filt / 2)
    ind_type = [np.where([tt in r_filt['t_freq_dir'] for r_filt in
                          r_obj.rot_filt_tot])[0] for tt in rot_filt['t_freq_dir']]

    # reduces down the fields to account for the manditory direction filtering
    r_obj.t_phase = [[dt] * len(x) for x in r_obj.t_phase]
    r_obj.phase_lbl = ['Baseline', 'Clockwise', 'Counter-Clockwise']
    r_obj.lg_str = [x.replace('Reverse\n', '') for x in r_obj.lg_str[:n_filt]]
    r_obj.i_expt, r_obj.ch_id, r_obj.cl_id = r_obj.i_expt[:n_filt], r_obj.ch_id[:n_filt], r_obj.cl_id[:n_filt]

    # # loops through each stimuli direction (CW/CCW) and for each filter reducing the stimuli phases
    # for i_dir in range(len(rot_filt['t_freq_dir'])):
    #     for i_filt in range(n_filt):
    #
    #         # for each cell (in each phase) reduce the spikes to fit the shortened stimuli range
    #         ii = ind_type[i_dir][i_filt]
    #         for i_cell in range(np.size(r_obj.t_spike[ii], axis=0)):
    #             for i_trial in range(np.size(r_obj.t_spike[ii], axis=1)):
    #                 for i_phase in range(np.size(r_obj.t_spike[ii], axis=2)):
    #                     # reduces the spikes by the shortened stimuli phase depending on the phase
    #                     if r_obj.t_spike[ii][i_cell, i_trial, i_phase] is not None:
    #                         x = dcopy(r_obj.t_spike[ii][i_cell, i_trial, i_phase])
    #                         if i_phase == 0:
    #                             # case is the baseline phase
    #                             r_obj.t_spike[ii][i_cell, i_trial, i_phase] = x[x > (t_phase - dt)]
    #                         else:
    #                             # case is the stimuli phase
    #                             jj = np.logical_and(x >= t0, x < (t0 + dt))
    #                             r_obj.t_spike[ii][i_cell, i_trial, i_phase] = x[jj]

    # returns the rotational analysis object
    return r_obj, ind_type

def eval_class_func(funcName, *args):
    '''

    :param funcName:
    :param args:
    :return:
    '''

    return funcName(*args)


def set_box_color(bp, color):
    '''

    :param bp:
    :param color:
    :return:
    '''

    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def cond_abb(tt):
    '''

    :param tt:
    :return:
    '''

    # sets up the abbreviation dictionary
    abb_txt = {
        'black': 'B',
        'uniform': 'U',
        'motordrifting': 'MD',
        'uniformdrifting': 'UD',
        'landmarkleft': 'LL',
        'landmarkright': 'LR',
        'black40': 'B40',
        'black45': 'B45',
        'mismatch1': 'MM1',
        'mismatch2': 'MM2',
        'black_discard': 'BD',
    }

    # returns the matching abbreviation
    return abb_txt[convert_trial_type(tt).lower()]


def convert_trial_type(tt_str):
    '''

    :param tt_str:
    :return:
    '''

    # sets the trial type key dictionary
    tt_key = {
        'Black40': ['Black_40', 'Black40deg'],
        'Black45': ['Black_45', 'Black45deg'],
        'Mismatch1': ['mismatch1'],
        'Mismatch2': ['mismatch2'],
        'Black_Discard': ['Black_discard', 'Black1_Discard']
    }

    # determines if the trial type string is in any of the conversion dictionary fields
    for tt in tt_key:
        if tt_str in tt_key[tt]:
            return tt

    # if no matches are made, then return the original strings
    return tt_str


def pad_array_with_nans(y, n_row=0, n_col=0):
    '''

    :param y:
    :param n_row:
    :param n_col:
    :return:
    '''

    # creates a copy of the array
    yy = dcopy(y)

    # expands the rows (if required)
    if n_row > 0:
        yy = np.lib.pad(yy, ((0, n_row), (0, 0)), 'constant', constant_values=np.NaN)

    # expands the columns (if required)
    if n_col > 0:
        yy = np.lib.pad(yy, ((0, 0), (0, n_col)), 'constant', constant_values=np.NaN)

    # returns the final array
    return yy


def calc_pointwise_diff(x1, x2):
    '''

    :param X1:
    :param X2:
    :return:
    '''

    #
    X1, X2 = np.meshgrid(x1, x2)
    return np.abs(X1 - X2)


def check_existing_file(hParent, out_file):
    '''

    :param out_file:
    :return:
    '''

    if os.path.exists(out_file):
        # prompts the user if they want to remove the selected item(s)
        u_choice = QMessageBox.question(hParent, 'Overwrite Existing File?',
                                        "File already exists. Are you sure you wish to overwrite this file?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        # returns if user wants to overwrite this file
        return u_choice == QMessageBox.Yes
    else:
        # file doesn't exist so continue as normal
        return True


def setup_sns_plot_dict(**kwargs):
    '''

    :param kwargs:
    :return:
    '''

    # initialisations
    sns_dict = {}

    # sets up the swarmplot data dictionary
    for key, value in kwargs.items():
        sns_dict[key] = value

    # returns the plot dictionary
    return sns_dict


def create_error_area_patch(ax, x, y_mn, y_err, col, f_alpha=0.2, y_err2=None, l_style='-', edge_color=None):
    '''

    :param ax:
    :param x:
    :param y_mn:
    :param y_err:
    :param col:
    :param f_alpha:
    :param y_err2:
    :return:
    '''

    # removes the non-NaN values
    is_ok = ~np.isnan(y_err)
    y_err, x = dcopy(y_err)[is_ok], np.array(dcopy(x))[is_ok]

    if edge_color is None:
        edge_color = col

    #
    if y_mn is not None:
        y_mn = dcopy(y_mn)[is_ok]

    # sets up the error patch vertices
    if y_err2 is None:
        if y_mn is None:
            err_vert = [*zip(x, y_err), *zip(x[::-1], y_err[::-1])]
        else:
            err_vert = [*zip(x, y_mn + y_err), *zip(x[::-1], y_mn[::-1] - y_err[::-1])]
    else:
        # removes the non-NaN values
        y_err2 = dcopy(y_err2)[is_ok]

        if y_mn is None:
            err_vert = [*zip(x, y_err), *zip(x[::-1], y_err2[::-1])]
        else:
            err_vert = [*zip(x, y_mn + y_err), *zip(x[::-1], y_mn[::-1] - y_err2[::-1])]

    # creates the polygon object and adds it to the axis
    poly = Polygon(err_vert, facecolor=col, alpha=f_alpha, edgecolor=edge_color, linewidth=3, linestyle=l_style)
    ax.add_patch(poly)


def create_step_area_patch(ax, x, y_mn, y_err, col, f_alpha=0.2):
    '''

    :param ax:
    :param x:
    :param y_mn:
    :param y_err:
    :param col:
    :param f_alpha:
    :return:
    '''

    # determines the new x-axis interpolation points
    d_xi = x[1] - x[0]
    x_interp = np.arange(x[0] - d_xi / 2, x[-1] + (0.001 + d_xi / 2), d_xi)

    # sets the x-locations for each of the steps
    ii = np.arange(len(x)).astype(int)
    x_step = x_interp[np.vstack((ii, ii+1)).T].flatten()

    # sets the lower/upper bound step values
    jj = repmat(ii, 2, 1).T
    y_lo, y_hi = (y_mn[jj] - y_err[jj]).flatten(), (y_mn[jj] + y_err[jj]).flatten()

    # creates the polygon object and adds it to the axis
    step_vert = [*zip(x_step, y_lo), *zip(x_step[::-1], y_hi[::-1])]
    poly = Polygon(step_vert, facecolor=col, alpha=f_alpha, edgecolor=col, linewidth=4)
    ax.add_patch(poly)


def set_sns_colour_palette(type='Default'):
    '''

    :param type:
    :return:
    '''

    if type == 'Default':
        colors = sns.xkcd_palette(["dark slate blue", "dark peach", "dull teal", "purpley grey", "maize", "sea blue",
                  "dark salmon", "teal", "dusty lavender", "sandy", "turquoise blue", "terracota", "dark seafoam",
                  "dark lilac", "buff"])

    # updates the colour palette
    sns.set_palette(colors)


def det_closest_file_match(f_grp, f_new):
    '''

    :param f_grp:
    :param f_new:
    :return:
    '''

    #
    ind_m = next((i for i in range(len(f_grp)) if f_grp[i] == f_new), None)

    if ind_m is None:
        # determines the best match and returns the matching file name/score
        f_score = np.array([fuzz.partial_ratio(x.lower(), f_new.lower()) for x in f_grp])

        # sorts the scores/file names by descending score
        i_sort = np.argsort(-f_score)
        f_score, f_grp = f_score[i_sort], np.array(f_grp)[i_sort]

        # returns the top matching values
        return f_grp[0], f_score[0]
    else:
        # otherwise, return the exact match
        return f_grp[ind_m], 100.

def get_cfig_line(cfig_file, fld_name):
    '''

    :param cfg_file:
    :param fld_name:
    :return:
    '''

    return next(c.rstrip('\n').split('|')[1] for c in open(cfig_file) if fld_name in c)


def save_single_file(f_name, data):
    '''

    :param f_name:
    :param data:
    :return:
    '''

    with open(f_name, 'wb') as fw:
        p.dump(data, fw)


def save_multi_comp_file(main_obj, out_info, force_update=False):
    '''

    :param main_obj:
    :param out_info:
    :param force_update:
    :return:
    '''

    # sets the output file name
    out_file = os.path.join(out_info['inputDir'], '{0}.mcomp'.format(out_info['dataName']))
    if not force_update:
        if not check_existing_file(main_obj, out_file):
            # if the file does exists and the user doesn't want to overwrite then exit
            return

    # starts the worker thread
    iw = main_obj.det_avail_thread_worker()
    main_obj.worker[iw].set_worker_func_type('save_multi_comp_file', thread_job_para=[main_obj.data, out_info])
    main_obj.worker[iw].start()


def save_single_comp_file(main_obj, out_info, force_update=False):
    '''

    :return:
    '''

    # sets the output file name
    out_file = os.path.join(out_info['inputDir'], '{0}.ccomp'.format(out_info['dataName']))
    if not force_update:
        if not check_existing_file(main_obj, out_file):
            # if the file does exists and the user doesn't want to overwrite then exit
            return

    # retrieves the index of the data field corresponding to the current experiment
    i_comp = det_comp_dataset_index(main_obj.data.comp.data, out_info['exptName'])

    # creates the multi-experiment data file based on the type
    data_out = {'data': [[] for _ in range(2)], 'c_data': main_obj.data.comp.data[i_comp],
                'ex_data': None, 'gen_filt': main_obj.data.exc_gen_filt}
    data_out['data'][0], data_out['data'][1] = get_comp_datasets(main_obj.data, c_data=data_out['c_data'], is_full=True)

    # outputs the external data (if it exists)
    if hasattr(main_obj.data, 'externd'):
        if hasattr(main_obj.data.externd, 'free_data'):
            data_out['ex_data'] = main_obj.data.externd.free_data

    # outputs the data to file
    with open(out_file, 'wb') as fw:
        p.dump(data_out, fw)


def save_multi_data_file(main_obj, out_info, is_multi=True, force_update=False):
    '''

    :return:
    '''

    # initialisations
    f_extn = 'mdata' if is_multi else 'mcomp'

    # determines if the file exists
    out_file = os.path.join(out_info['inputDir'], '{0}.{1}'.format(out_info['dataName'], f_extn))
    if not force_update:
        if not check_existing_file(main_obj, out_file):
            # if the file does exists and the user doesn't want to overwrite then exit
            return

    # starts the worker thread
    iw = main_obj.det_avail_thread_worker()
    main_obj.worker[iw].set_worker_func_type('save_multi_expt_file', thread_job_para=[main_obj.data, out_info])
    main_obj.worker[iw].start()


def det_comp_dataset_index(c_data, exp_name, is_fix=True):
    '''

    :param c_data:
    :param exp_name:
    :return:
    '''

    # removes any forward slashes (if present)
    if '/' in exp_name:
        exp_name = exp_name.split('/')[0]

    f_name = [cd.fix_name for cd in c_data] if is_fix else [cd.free_name for cd in c_data]
    return det_likely_filename_match(f_name, exp_name)

def det_likely_filename_match(f_name, exp_name):
    '''

    :param f_name:
    :param exp_name:
    :return:
    '''

    # sets the search name
    exp_name_search = exp_name[0] if isinstance(exp_name, list) else exp_name

    # determines the comparison dataset with the matching freely moving experiment file name
    i_expt_nw = next((i for i, x in enumerate(f_name) if x.lower() == exp_name_search.lower()), None)
    if i_expt_nw is None:
        # if there isn't an exact match, then determine the
        m_score_fuzz = np.array([fuzz.partial_ratio(x, exp_name_search) for x in f_name])
        i_match_fuzz = np.where(m_score_fuzz > 90)[0]

        #
        if len(i_match_fuzz) == 0:
            # case is there are no matches
            return None

        elif len(i_match_fuzz) > 1:
            # case is there is more than one match
            m_score_fuzz_2 = np.array([fuzz.ratio(x, exp_name_search) for x in f_name])
            i_expt_nw = np.argmax(np.multiply(m_score_fuzz, m_score_fuzz_2))

        else:
            # case is there is only one match
            i_expt_nw = i_match_fuzz[0]

    # returns the index of the matching file
    return i_expt_nw

def get_comp_datasets(data, c_data=None, ind=None, is_full=False):
    '''

    :return:
    '''

    # sets the cluster type
    c = data._cluster if (use_raw_clust(data) or is_full) else data.cluster

    # retrieves the fixed/free datasets based on type
    if ind is None:
        return c[get_expt_index(c_data.fix_name, c)], c[get_expt_index(c_data.free_name, c)]
    else:
        return c[ind[0]], c[ind[1]]


def get_comb_file_names(str_1, str_2):
    '''

    :param str_1:
    :param str_2:
    :return:
    '''

    # initialisations
    N = min(len(str_1), len(str_2))
    _str_1, _str_2 = str_1.lower(), str_2.lower()

    # determines the mutual components of each string and combines them into a single string
    i_match = next((i for i in range(N) if _str_1[i] != _str_2[i]), N + 1) - 1
    return '{0}/{1}'.format(str_1, str_2[i_match:])


def use_raw_clust(data):
    '''

    :param data:
    :return:
    '''

    return (data.cluster is None) or (len(data.cluster) != len(data._cluster))


def get_global_expt_index(data, c_data):
    '''

    :param data:
    :param c_data:
    :return:
    '''

    return [extract_file_name(c['expFile']) for c in data._cluster].index(c_data.fix_name)


def has_free_ctype(data):
    '''

    :param data:
    :return:
    '''

    # determines if the freely moving data field has been set into the external data field of the main data object
    if hasattr(data, 'externd'):
        if hasattr(data.externd, 'free_data'):
            # if so, determine if the cell type information has been set for at least one experiment
            return np.any([len(x) > 0 for x in data.externd.free_data.cell_type])
        else:
            # otherwise, return a false flag value
            return False
    else:
        # if no external data field, then return a false flag value
        return False


def det_matching_fix_free_cells(data, exp_name=None, cl_ind=None, apply_filter=False, r_obj=None):
    '''

    :param data:
    :return:
    '''

    import analysis_guis.calc_functions as cfcn
    from analysis_guis.dialogs.rotation_filter import RotationFilteredData

    # sets the experiment file names
    if exp_name is None:
        exp_name = data.externd.free_data.exp_name
    elif not isinstance(exp_name, list):
        exp_name = list(exp_name)

    # initialisations
    free_file, free_data = [x.free_name for x in data.comp.data], data.externd.free_data
    i_file_free = [free_data.exp_name.index(ex_name) for ex_name in exp_name]

    # retrieves the cluster indices (if not provided)
    if cl_ind is None:
        r_filt = init_rotation_filter_data(False)
        r_filt['t_type'] += ['Uniform']
        r_obj0 = RotationFilteredData(data, r_filt, None, None, True, 'Whole Experiment', False, use_raw=True)
        cl_ind = r_obj0.clust_ind[0]

    # memory allocation
    n_file = len(exp_name)
    is_ok = np.ones(n_file, dtype=bool)
    i_expt = -np.ones(n_file, dtype=int)
    f2f_map = np.empty(n_file, dtype=object)

    for i_file, ex_name in enumerate(exp_name):
        # determines if there is a match between the freely moving experiment file and that stored within the
        # freely moving data field
        if det_likely_filename_match(free_data.exp_name, ex_name) is None:
            # if not, then flag the file as being invalid and continue
            is_ok[i_file] = False
            continue
        else:
            # otherwise, determine if there is a match within the comparison dataset freely moving data files
            i_expt_nw = det_likely_filename_match(free_file, ex_name)
            if i_expt_nw is None:
                # if not, then flag the file as being invalid and continue
                is_ok[i_file] = False
                continue

        # retrieves the fixed/free datasets
        i_expt[i_file] = i_expt_nw
        c_data = data.comp.data[i_expt_nw]
        data_fix, data_free = get_comp_datasets(data, c_data=c_data, is_full=True)

        # sets the match array (removes non-inclusion cells and non-accepted matched cells)
        cl_ind_nw = cl_ind[i_expt_nw]
        i_match = c_data.i_match[cl_ind_nw]
        i_match[~c_data.is_accept[cl_ind_nw]] = -1

        # removes any cells that are excluded by the general filter
        if apply_filter:
            cl_inc_fix = cfcn.get_inclusion_filt_indices(data_fix, data.exc_gen_filt)
            i_match[np.logical_not(cl_inc_fix)] = -1

        # if there is a secondary rotation filter object, then remove any non-included indices
        if r_obj is not None:
            b_arr = set_binary_groups(len(i_match), r_obj.clust_ind[0][i_expt_nw])
            i_match[~b_arr] = -1

        # determines the overlapping cell indices between the free dataset and those from the cdata file
        _, i_cell_free_f, i_cell_free = \
                np.intersect1d(dcopy(free_data.cell_id[i_file_free[i_file]]), dcopy(data_free['clustID']),
                               assume_unique=True, return_indices=True)

        # determines the fixed-to-free mapping index arrays
        _, i_cell_fix, i_free_match = np.intersect1d(i_match, i_cell_free, return_indices=True)
        f2f_map[i_file] = -np.ones((len(cl_ind_nw),2), dtype=int)
        f2f_map[i_file][i_cell_fix, 0] = i_cell_free[i_free_match]
        f2f_map[i_file][i_cell_fix, 1] = i_cell_free_f[i_free_match]

    # returns the experiment index/fixed-to-free mapping indices
    return i_expt, f2f_map


def det_reverse_indices(i_cell_b, ind_gff):
    '''

    :param i_cell_b:
    :param ind_gff:
    :return:
    '''

    _, _, ind_rev = np.intersect1d(i_cell_b, ind_gff, return_indices=True)
    return ind_rev


def reset_table_pos(fig, ax_t, t_props):
    '''

    :param fig:
    :param ax:
    :param t_props:
    :return:
    '''

    # no need to reset positions if only one table
    n_table = len(t_props)
    if n_table == 1:
        return


    # initialisations
    f_rend = fig.get_renderer()
    ax_pos = ax_t.get_tightbbox(f_rend).bounds
    ax_hght = ax_pos[1] + ax_pos[3]

    #
    t_pos = [tp[0].get_tightbbox(f_rend).bounds for tp in t_props]
    t_pos_bb = [tp[0]._bbox for tp in t_props]

    #
    for i_table in range(1, n_table):
        #
        p_hght = 1 - i_table / n_table
        y_nw = (p_hght * ax_hght) - t_pos[i_table][3]

        #
        t_props[i_table][0]._bbox[1] = y_nw / t_props[i_table][2]
        t_props[i_table][0]._bbox[0] = t_pos_bb[0][0] + (t_pos_bb[0][2] - t_pos_bb[i_table][2]) / 2


def get_table_font_size(n_grp):
    '''

    :param n_grp:
    :return:
    '''

    if n_grp <= 2:
        return create_font_obj(size=10)
    elif n_grp <= 4:
        return create_font_obj(size=8)
    else:
        return create_font_obj(size=6)


def get_cluster_id_flag(cl_id, i_expt=None):

    if i_expt is None:
        # case is the experiment index is not provided
        return [(i * 10000) + np.array(y) for i, y in enumerate(cl_id)]
    else:
        # case is the experiment index is provided
        return [(i * 10000) + y for i, y in zip(i_expt, cl_id)]


def get_array_lengths(Y, fld_key=None):
    '''

    :param Y:
    :param fld_key:
    :return:
    '''

    if fld_key is None:
        return np.array([len(x) for x in Y])
    else:
        return np.array([eval('x["{0}"]'.format(fld_key)) for x in Y])


def get_global_index_arr(r_obj, return_all=True, i_expt_int=None):
    '''

    :param r_obj:
    :return:
    '''

    def setup_global_index_arr(nC, i_expt_int, i_expt0, return_all):
        '''

        :param nC:
        :param i_expt0:
        :return:
        '''

        ii = np.append(0, np.cumsum(nC))
        if return_all:
            return [np.arange(ii[i], ii[i + 1]) if (i_expt0[i] in i_expt_int)
                                                    and (ii[i + 1] - ii[i]) > 0 else [] for i in range(len(ii) - 1)]
        else:
            return [np.arange(ii[i], ii[i + 1]) for i in range(len(ii) - 1)
                                                    if (i_expt0[i] in i_expt_int) and (ii[i + 1] - ii[i]) > 0]

    # determines the indices of the experiments that are common to all trial types
    if i_expt_int is None:
        i_expt_int = set(r_obj.i_expt0[0])
        for i_ex in r_obj.i_expt0[1:]:
            i_expt_int = i_expt_int.intersection(set(i_ex))

        # retrieves the global indices of the cells wrt to the filtered spiking frequency values
        return [setup_global_index_arr(get_array_lengths(cl_id), i_expt_int, i_ex, return_all)
                               for cl_id, i_ex in zip(r_obj.clust_ind, r_obj.i_expt0)], np.array(list(i_expt_int))
    else:
        # retrieves the global indices of the cells wrt to the filtered spiking frequency values
        cl_id = [[x for i, x in zip(i_ex0, cl_id) if i in i_expt_int] for i_ex0, cl_id in
                 zip(r_obj.i_expt0, r_obj.clust_ind)]
        return [setup_global_index_arr(get_array_lengths(_cl_id), i_expt_int, i_expt_int, return_all)
                               for _cl_id in cl_id]


def reset_integer_tick(ax, ax_type):
    '''

    :param ax:
    :param ax_type:
    :return:
    '''

    if ax_type == 'x':
        get_tick_fcn, set_tick_fcn, set_lbl_fcn = ax.get_xticks, ax.set_xticks, ax.set_xticklabels
    else:
        get_tick_fcn, set_tick_fcn, set_lbl_fcn = ax.get_yticks, ax.set_yticks, ax.set_yticklabels

    # retrieves the tick values and determines if they are integers
    t_vals = get_tick_fcn()
    is_ok = t_vals % 1 == 0

    # if there are any non-integer values then remove them
    if np.any(~is_ok):
        set_tick_fcn(t_vals[is_ok])
        set_lbl_fcn([Annotation('{:d}'.format(int(y)),[0, y]) for y in t_vals[is_ok]])


def get_all_filter_indices(data, rot_filt):
    '''

    :param data:
    :param rot_filt:
    :return:
    '''

    # module import
    from analysis_guis.dialogs.rotation_filter import RotationFilteredData

    # retrieves the data clusters for each of the valid rotation experiments
    is_rot_expt = det_valid_rotation_expt(data)
    d_clust = [x for x, y in zip(data._cluster, is_rot_expt) if y]
    wfm_para = [x['rotInfo']['wfm_para']['UniformDrifting'] for x in
                              d_clust if 'UniformDrifting' in x['rotInfo']['wfm_para']]

    # adds any non-empty filter objects onto the rotation filter object
    for gf in data.exc_gen_filt:
        if len(data.exc_gen_filt[gf]):
            # retrieves the field values
            if gf in ['cell_type']:
                # case is the freely moving data types
                fld_vals = get_unique_group_types(d_clust, gf, c_type=data.externd.free_data.cell_type)
            elif gf in ['temp_freq', 'temp_freq_dir', 'temp_cycle']:
                # case is the uniform drifting data types
                fld_vals = get_unique_group_types(d_clust, gf, wfm_para=wfm_para)
            else:
                # case is the other rotation data types
                fld_vals = get_unique_group_types(d_clust, gf)

            # retrieves the fields values to be added
            add_fld = list(set(fld_vals) - set(data.exc_gen_filt[gf]))

            if 'All' in rot_filt[gf]:
                rot_filt[gf] = add_fld
            else:
                rot_filt[gf] = list(np.union1d(add_fld, rot_filt[gf]))

    # retrieves the rotation filter data class object
    r_obj = RotationFilteredData(data, rot_filt, None, None, True, 'Whole Experiment', False,
                                 rmv_empty=0, use_raw=True)

    # returns the cluster indices
    return r_obj.clust_ind


def get_unique_group_types(d_clust, f_type, wfm_para=None, c_type=None):
    '''

    :param d_clust:
    :param f_type:
    :return:
    '''

    # retrieves the field values based on the type and inputs
    if wfm_para is not None:
        # case is the uniform-drifting values
        if f_type == 'temp_freq':
            return [str(x) for x in get_field(wfm_para, 'tFreq')]
        elif f_type == 'temp_freq_dir':
            return [str(x) for x in get_field(wfm_para, 'yDir').astype(int)]
        elif f_type == 'temp_cycle':
            return [str(x) for x in get_field(wfm_para, 'tCycle').astype(int)]

    elif c_type is not None:
        if f_type in ['c_type', 'free_ctype']:
            c_type0 = pd.concat([x[0] for x in c_type if len(x)], axis=0)
            c_none = ['No Type'] if any(np.sum(c_type0, axis=1)==0) else []
            return [ct for ct, ct_any in zip(c_type0.columns, np.any(c_type0, axis=0)) if ct_any] + c_none

    else:
        if f_type == 'sig_type':
            return ['Narrow Spikes', 'Wide Spikes']
        elif f_type == 'match_type':
            return ['Matched Clusters', 'Unmatched Clusters']
        elif f_type in ['t_type', 'trial_type']:
            return flat_list([list(x['rotInfo']['trial_type']) for x in d_clust])
        elif f_type in ['region_type', 'region_name']:
            return list(np.unique(flat_list([list(np.unique(x['chRegion'])) for x in d_clust])))
        elif f_type == 'record_layer':
            return list(np.unique(flat_list([list(np.unique(x['chLayer'])) for x in d_clust])))
        elif f_type == 'record_coord':
            return list(np.unique([x['expInfo']['record_coord'] for x in d_clust]))
        elif f_type in ['lesion_type', 'lesion']:
            return list(np.unique([x['expInfo']['lesion'] for x in d_clust]))
        elif f_type == 'record_state':
            return list(np.unique([x['expInfo']['record_state'] for x in d_clust]))