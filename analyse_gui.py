# module import
import os
import sys

if sys.platform == 'linux':
    os.environ['R_HOME'] = '/home/skeshav/miniconda3/envs/rotation_analysis/lib/R/'

from PyQt5.QtWidgets import QApplication
from analysis_guis import main_analysis

import seaborn as sns
#sns.set_palette(sns.color_palette("Paired", 8))
#colors = ["cobalt", "light orange", "teal", "dusty lavender", "sea blue", "maize", "dull teal", "purpley grey",
#          "turquoise blue", "light mustard", "dark seafoam", "dark lilac"]
colors = ["cobalt", "dark peach", "dull teal", "purpley grey", "maize", "sea blue", "dark salmon", "teal",
          "dusty lavender", "light orange", "turquoise blue", "terracota", "dark seafoam",
          "dark lilac", "mustard yellow"]

sns.set_palette(sns.xkcd_palette(colors))



if __name__ == '__main__':

    if sys.platform == 'linux':
        QApplication.setStyle('windows')

    # runs the analysis GUIs
    app = QApplication([])
    h_main = main_analysis.AnalysisGUI()
    app.exec()