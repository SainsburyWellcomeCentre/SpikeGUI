# module import
import os
import sys

if sys.platform == 'linux':
    os.environ['R_HOME'] = '/home/skeshav/miniconda3/envs/rotation_analysis/lib/R/'

from PyQt5.QtWidgets import QApplication
from analysis_guis import main_analysis

import seaborn as sns
sns.set_palette(sns.color_palette("husl"))

if __name__ == '__main__':

    if sys.platform == 'linux':
        QApplication.setStyle('windows')

    # runs the analysis GUIs
    app = QApplication([])
    h_main = main_analysis.AnalysisGUI()
    app.exec()
