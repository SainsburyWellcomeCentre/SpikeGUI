import matplotlib
matplotlib.use('qt5agg')
matplotlib.interactive(False)

import pandas as pd

from PyQt5.QtCore import QObject, pyqtSlot
from analysis.probe.probe_io import metadata_db_io
from matplotlib import pyplot as plt


class Logger(QObject):
    """
    A qml object for logging the events
    In production, the standard out is redirected to it.
    It is not meant to be interacted with directly (use print() instead)
    """

    def __init__(self, context, parent=None, log_object_name="log"):
        QObject.__init__(self, parent)
        self.win = parent
        self.ctx = context
        self.log = self.win.findChild(QObject, log_object_name)

    def write(self, text):
        """
        The method to make it compatible with sys.stdout
        The text gets printed in the corresponding qml component

        :param string text: The text to append at the end of the current qml component text
        """
        if text:
            previous_text = self.log.property('text')
            output_text = '{}\n>>>{}'.format(previous_text, text)

            self.log.setProperty('text', output_text)


class PythonBackendClass1(QObject):
    """
    The QObject derived class that stores most of the parameters from the graphical interface
    for the other QT interfaces
    """
    def __init__(self, app, context, parent, img_provider, img_provider_2):
        """
        :param app: The QT application
        :param context:
        :param parent: the parent window
        """
        QObject.__init__(self, parent)
        self.app = app  # necessary to avoid QPixmap bug: Must construct a QGuiApplication before
        self.win = parent
        self.ctx = context
        self.src_path = ''
        self.dest_path = ''
        self.data = pd.read_csv('/home/skeshav/Desktop/db_test_fix_relative_events.csv')
        self.timer_period = 1

        self._set_defaults()
        self.condition_dictionaries = [{}, {}]
        self.db_io = metadata_db_io.DatabaseIo('/home/skeshav/Desktop/db_test_fix_relative_events.csv')
        self.dfs = [pd.DataFrame(), pd.DataFrame()]
        self.results = pd.DataFrame()

        self.img_provider_1 = img_provider
        self.img_provider_2 = img_provider_2

    def _set_defaults(self):
        """
        Reset the parameters to default.
        To customise the defaults, users should do this in the config file.
        """
        # self.bg_frame_idx = config['tracker']['frames']['ref']
        # self.n_bg_frames = config['tracker']['sd_mode']['n_background_frames']

    @pyqtSlot(int)
    def set_timer_period(self, timer_period):
        self.timer_period = timer_period

    @pyqtSlot(result=int)
    def get_timer_period(self):
        return self.timer_period

    @pyqtSlot(str)
    def print_warning(self, name):
        print('take the time to go fast {}'.format(name))

    @pyqtSlot(result=str)
    def get_bs(self):
        return 'BS'

    @pyqtSlot(int, result=str)
    def get_key_at(self, idx):
        return self.keys()[idx]

    @pyqtSlot(result=int)
    def get_n_keys(self):
        return len(self.keys())

    @pyqtSlot(int, str, str, str)
    def update_condition_dictionary(self, idx, key, value, comparator):

        if not value:
            return

        if key in ['cell_id', 'trial_id']:
            value = value.split(',')
            value = [int(v) for v in value]

        comparator_value_string = '{} {}'.format(comparator, value)

        self.condition_dictionaries[idx].setdefault(key, comparator_value_string)

    @pyqtSlot()
    def reset_conditions(self):
        self.condition_dictionaries = [{}, {}]
        self.dfs = [pd.DataFrame(), pd.DataFrame()]

    @pyqtSlot(int, result=str)
    def display_table(self, idx):
        self.dfs[0] = self.db_io.filter_df(self.condition_dictionaries[0])
        self.dfs[1] = self.db_io.filter_df(self.condition_dictionaries[1])
        return self.dfs[idx].to_html()

    def keys(self):
        keys = list(self.data.keys())
        keys.remove('Unnamed: 0')
        return keys

    def get_comparators(self):
        return list(self.db_io.get_comparator_functions().keys())

    @pyqtSlot(int, result=str)
    def get_comparator_at(self, idx):
        return self.get_comparators()[idx]

    @pyqtSlot(result=int)
    def get_n_comparators(self):
        return len(self.get_comparators())

    @pyqtSlot(result=str)
    def compare(self):
        """

        :param key:
        :return:
        """
        key = 'cell_id'
        self.results = pd.DataFrame()

        conditions_a_dict = self.condition_dictionaries[0]
        conditions_b_dict = self.condition_dictionaries[1]

        for option in list(set(self.db_io.db[key])):

            conditions_a_dict[key] = '== {}'.format(option)
            conditions_b_dict[key] = '== {}'.format(option)

            avg_a, avg_b, wilcox_p = self.db_io.compare_groups(conditions_a_dict, conditions_b_dict)

            new_dict = {}
            for item_a, item_b in zip(conditions_a_dict.items(), conditions_b_dict.items()):
                new_val = [item_a[1], item_b[1]]
                new_key = item_a[0]
                assert new_key == item_b[0]
                new_dict[new_key] = new_val

            results_dict = {key: [option, option],
                            'average_values': [avg_a, avg_b],
                            'p_value': [wilcox_p, wilcox_p],
                            }

            new_dict.update(results_dict)

            results_df = pd.DataFrame(new_dict)
            self.results = self.results.append(results_df, ignore_index=True)

        return self.results.to_html()

    @pyqtSlot()
    def generate_plots(self):
        eventplot_data_1 = self.eventplot_from_df(self.db_io.filter_df(self.condition_dictionaries[0]))
        eventplot_data_2 = self.eventplot_from_df(self.db_io.filter_df(self.condition_dictionaries[1]))

        fig, axes = plt.subplots()
        self.format_plot(axes)

        plt.eventplot(eventplot_data_1)

        self.img_provider_1._fig = fig

        fig, axes = plt.subplots()
        plt.eventplot(eventplot_data_2)
        self.format_plot(axes)

        self.img_provider_2._fig = fig

    def eventplot_from_df(self, df):
        eventplot_data = []
        events_as_strs_all_trials = df['event_locs']
        for trial in events_as_strs_all_trials:
            if len(trial) > 2:
                event_locs_in_trial = [int(float(v.strip())) for v in trial.strip('[]').split(',')]
                eventplot_data.append(event_locs_in_trial)
        return eventplot_data

    def format_plot(self, ax):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('sample number')
        plt.ylabel('trials')
