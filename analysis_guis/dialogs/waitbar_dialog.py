# module import
import sys
import time
import numpy as np
import functools

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QProgressBar, QApplication, QDialog, QVBoxLayout, QPushButton,
                             QGridLayout, QGroupBox, QLabel, QStyleFactory)

# sets the style data
styleData="""
QPushButton
{
    font-weight: bold;
}
QProgressBar::chunk
{
    background-color: #0000FF;
}
QGroupBox
{
    font-weight: bold;
    font-size: 14;"
}
"""


class Waitbar(QDialog):
    def __init__(self, parent=None, n_bar=1, p_min=0, p_max=100, title=None, width=400, headers=None, is_test=False):
        # creates the object
        super(Waitbar, self).__init__(parent)
        self.originalPalette = QApplication.palette()

        # memory allocation and initialisations
        self.n_bar = n_bar
        self.is_cancel = False

        # sets up the progress bars
        self.setup_limits(p_min, p_max)
        self.setup_progressbars(headers)
        cf.set_obj_fixed_size(self, width=width)

        # sets up the test timers
        if is_test:
            self.setup_test_timers()

        # sets the final window properties
        if title is None:
            self.setWindowTitle(" ")
        else:
            self.setWindowTitle(title)

        # resizes the GUI and sets the other style properties
        self.setModal(1)
        self.setStyleSheet(styleData)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.show()

        for i in range(self.n_bar):
            self.update(i, 0)

    def update(self, i_bar, value, text=None):
        '''

        :param i_bar:
        :param value:
        :param text:
        :return:
        '''

        # retrieves the progress bar/text label object handle
        hProg = self.p_bar[i_bar][0].findChild(QProgressBar)
        hText = self.p_bar[i_bar][0].findChild(QLabel)

        # sets the label text (if not provided)
        if text is None:
            maxVal = hProg.maximum()
            text = 'Percentage = {0}%'.format(int(100.0*value/maxVal))

        # updates the label text and progress bar value
        hText.setText(text)
        hProg.setValue(value)
        time.sleep(0.01)


    def changeStyle(self, styleName):
        '''

        :param styleName:
        :return:
        '''

        QApplication.setStyle(QStyleFactory.create(styleName))
        QApplication.setPalette(QApplication.style().standardPalette())


    def closeEvent(self, evnt):

        if self.is_cancel:
            super(Waitbar, self).closeEvent(evnt)
        else:
            evnt.ignore()


    def closeWindow(self):

        self.is_cancel = True
        self.close()


    def setup_progressbars(self, headers):

        # memory allocation
        self.p_bar = np.empty((self.n_bar,1), dtype=object)

        # creates the progressbar and layout objects
        mainLayout = QGridLayout()
        for i_bar in range(self.n_bar):
            self.create_progress_bar(i_bar, headers)
            mainLayout.addWidget(self.p_bar[i_bar][0], i_bar, 0)

        #
        self.create_close_button()
        mainLayout.addWidget(self.p_close, self.n_bar, 0)

        # sets the main progress-bar layout
        self.setLayout(mainLayout)


    def create_progress_bar(self, i_bar, headers):

        # group box object
        if headers is not None:
            self.p_bar[i_bar] = QGroupBox(headers[i_bar])
        else:
            self.p_bar[i_bar] = QGroupBox("")

        # creates the text label object
        hText = QLabel("Waitbar #{0}".format(i_bar+1))
        hText.setAlignment(Qt.AlignCenter)

        # creates the progress bar object
        hProg = QProgressBar()
        hProg.setRange(self.p_lim[i_bar,0], self.p_lim[i_bar,1])
        hProg.setValue(0)
        hProg.setTextVisible(False)

        # sets the box layout
        layout = QVBoxLayout()
        layout.addWidget(hText)
        layout.addWidget(hProg)
        layout.addStretch(1)
        self.p_bar[i_bar][0].setLayout(layout)

        # updates the object fonts
        cf.update_obj_font(self.p_bar[i_bar][0], pointSize=10, weight=QFont.Bold)
        cf.update_obj_font(hText, pointSize=8)


    def create_close_button(self):

        # group box object
        self.p_close = QGroupBox("")

        # creates the button object
        hButton = QPushButton('Cancel Operation')
        hButton.setDefault(True)
        hButton.clicked.connect(self.closeWindow)
        cf.update_obj_font(hButton, pointSize=8, weight=QFont.Bold)

        # sets the box layout
        layout = QVBoxLayout()
        layout.addWidget(hButton)
        layout.addStretch(1)
        self.p_close.setLayout(layout)


    def setup_limits(self, p_min, p_max):
        '''

        :param p_min:
        :param p_max:
        :return:
        '''

        # memory allocation
        self.p_lim = np.zeros((self.n_bar, 2), dtype=int)

        # sets the lower/upper limits
        self.set_limit(p_min, 0)
        self.set_limit(p_max, 1)


    def set_limit(self, p, ind):
        '''

        :param p:
        :param ind:
        :return:
        '''

        # sets the limit based on the value type
        if isinstance(p, int):
            # case is a constant integer value. set for all
            self.p_lim[:, ind] = p
        else:
            # case is list/array
            self.p_lim[:, ind] = np.array(p)

    ###########################
    #    TESTING FUNCTIONS    #
    ###########################

    def setup_test_timers(self):
        '''

        :return:
        '''

        # creates the timer callback function
        timerCallback = functools.partial(self.testProgressBar, i_bar=0)

        # sets up the timer object
        timer = QTimer(self)
        timer.timeout.connect(timerCallback)
        timer.start(5)


        for i_bar in range(self.n_bar):
            if i_bar == 0:
                hProg = self.p_bar[i_bar][0].findChild(QProgressBar)
                mxVal = hProg.maximum()

            # initialises the progress bar
            self.update(i_bar, 0, text='{0} of {1} (0%)'.format(0, mxVal))


    def testProgressBar(self, i_bar):
        '''

        '''

        # retrieves the current/maximum value
        hProg = self.p_bar[i_bar][0].findChild(QProgressBar)
        cVal = hProg.value()
        mxVal = hProg.maximum()

        #
        if cVal == mxVal:
            if (i_bar + 1) == self.n_bar:
                sys.exit()
            else:
                self.update(i_bar, 0, text='{0} of {1} (0%)'.format(0, mxVal))
                self.testProgressBar(i_bar + 1)
        else:
            pW = int(100.0*float(cVal+1)/float(mxVal))
            self.update(i_bar, cVal+1, text='{0} of {1} ({2}%)'.format(cVal+1, mxVal, pW))