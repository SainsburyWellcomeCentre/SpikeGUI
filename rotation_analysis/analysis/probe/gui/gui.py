import os
import sys

if sys.platform.startswith('linux'):
    from OpenGL import GL

from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtWidgets import QApplication

from analysis.probe.gui.backend_classes import PythonBackendClass1, Logger
from analysis.probe.gui.image_providers import PyplotImageProvider

DEBUG = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    appEngine = QQmlApplicationEngine()
    context = appEngine.rootContext()

    analysis_image_provider1 = PyplotImageProvider(fig=None)
    appEngine.addImageProvider("analysisprovider1", analysis_image_provider1)
    analysis_image_provider2 = PyplotImageProvider(fig=None)
    appEngine.addImageProvider("analysisprovider2", analysis_image_provider2)


    # ALL THE ADDIMAGEPROVIDER LINES BELOW ARE REQUIRED TO MAKE QML BELIEVE THE PROVIDER IS VALID BEFORE ITS CREATION
    # appEngine.addImageProvider('viewerprovider', CvImageProvider())

    # analysis_image_provider = PyplotImageProvider(fig=None)
    # appEngine.addImageProvider("analysisprovider", analysis_image_provider)

    conf = {
        'shared_directory': './'  # FIXME: this is obviously BS
    }

    qml_source_path = os.path.join(conf['shared_directory'], 'qml', 'gui_qtquick', 'gui_qtquick.qml')
    if not os.path.isfile(qml_source_path):
        raise ValueError("Qml code not found at {}, please verify your installation".format(qml_source_path))
    appEngine.load(qml_source_path)

    try:
        win = appEngine.rootObjects()[0]
    except IndexError:
        raise ValueError("Could not start the QT GUI")

    if not DEBUG:
        logger = Logger(context, win, "log")
        sys.stdout = logger

    print('Hello world')

    # icon = QIcon(os.path.join(conf.shared_directory, 'resources', 'icons', 'pyper.png'))
    # win.setIcon(icon)

    backend = PythonBackendClass1(app, context, win, analysis_image_provider1, analysis_image_provider2)  # create instance of backend
    context.setContextProperty('py_iface', backend)  # register backend python object with qml code under variable name py_iface

    win.show()

    sys.exit(app.exec_())
