# pyqt5 module import
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

########################################################################################################################
########################################################################################################################

class FileDialogModal(QFileDialog):
    def __init__(self, parent=None, caption=None, filter=None,
                       directory=None, is_save=False, dir_only=False, is_multi=False):
        # creates the object
        super(FileDialogModal, self).__init__(parent=parent, caption=caption, filter=filter, directory=directory)

        # sets the file dialog parameters
        self.setModal(True)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        # sets the file dialog to open if
        if is_save:
            self.setAcceptMode(QFileDialog.AcceptSave)

        # sets the file mode to directory (if directory only)
        if dir_only:
            self.setFileMode(QFileDialog.DirectoryOnly)

        # sets the multi-select flag to true (if required)
        if is_multi:
            self.setFileMode(QFileDialog.ExistingFiles)