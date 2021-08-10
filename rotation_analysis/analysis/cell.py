import os
import sys
import pickle

import numpy as np
import pandas as pd

from rotation_analysis.analysis.probe.ctxt_manager_decorator import temp_setattr
from rotation_analysis.analysis.block import Block
np.seterr(divide='ignore', invalid='ignore')  # TODO: improve (avoids invalid value in true_divide)


class Cell(object):
    """
    Would make sense to have a set of Cell objects that belong to the same depth
    """

    CELL_TYPES = ['Pyramid', 'Interneuron']

    def __init__(self, src_dir, exp_id, depth, cell_idx, recordings, extension, use_bsl_2):
        """
        recordings is a list of list (n_angles list of n recordings)
        a cell has a set of lists of lists such that shape is n_angles, n_trials
        for each relevant parameter
        
        :param str src_dir: The source directory
        :param int cell_idx: The index of the cell in the list of ROIs
        :param dict recordings: a list of Recording objects that have fluorescence profiles for the cell
        :param str extension: The file extension to save the figures
        
        :var str dir: The depth directory
        
        :var OrderedDict of lists of OrderedDict trials: The profiles of all the trials for that cell (x angles, y trials, z channels)
        :var OrderedDict of lists of OrderedDict neuropile_trials: The neuropile profiles of all the trials for that cell (x angles, y trials, z channels)
        :var OrderedDict of lists of OrderedDict noises: The standard deviations of the high pass filtered data (x angles, y trials, z channels)
        :var OrderedDict of lists of OrderedDicts of lists eventsParams: The list of parameters (eventsPos, peaksPos, halfRises,  _) for (x angles, y trials, z channels)
        :var OrderedDict metadata: The ini metadata for that cell (1 per angle)
        :var OrderedDict commands: The commands (input to motor) for that cell (1 per angle)
        
        """
        self.analysed_metrics = ('frequency', 'amplitude', 'delta_f')  # WARNING: could change
        self.dir = src_dir
        self.ext = extension

        self.id = cell_idx
        self.exp_id = exp_id
        self.depth = depth

        self.cell_type = 'Pyramid'  # TODO: config
        self.skip = False

        self.angles = list(recordings.keys())  # FIXME: duplicate with block
        self.block = Block(self, recordings, use_bsl_2)
        self.main_dir = None

    def __str__(self):
        return 'exp_{}_depth_{}_cell_{}'.format(self.exp_id, self.depth, str(self.id).zfill(3))

    def set_main_dir(self, main_dir):
        self.main_dir = main_dir

    def pickle(self):
        pkl_file_path = os.path.join(self.dir, '{}.pkl'.format(self))
        with open(pkl_file_path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file, pickle.HIGHEST_PROTOCOL)

    @property
    def n_angles(self):
        return len(self.angles)

    def remove_trials(self, bad_trials_list):  # TODO: see if can be dynamic instead
        """
        Given a list of trials indices, remove the trials in the list
        """
        self.block.remove_trials(bad_trials_list)  # FIXME: check if removal should depend on condition

    def save_detection(self, processing_type):
        for angle in self.angles:
            with temp_setattr(self.block, 'current_condition', {'keep': True, 'angle': angle}):
                self.block.save_detection(angle, processing_type)

    def analyse_all(self):
        self.analyse()

    def analyse(self, processed=True):
        """
        Loop over all recordings for each angle
            self.noises with the high frequency noise of each recording
            and self.events_params with the tuple of lists: (events_pos, peaks_pos, half_rises, peak_ampls)
        """
        for angle in self.angles:
            with temp_setattr(self.block, 'current_condition', {'keep': True, 'angle': angle}):
                self.block.analyse(angle, processed=processed)

    def plot_all(self, angle):
        with temp_setattr(self.block, 'current_condition', {'keep': True, 'angle': angle}):
            self.block.plot()

    def analyse_block(self, angle):
        with temp_setattr(self.block, 'current_condition', {'keep': True, 'angle': angle}):
            self.block.save_stats()

    def _get_base_df(self, angle):
        return pd.DataFrame({'cell_id': self.id,
                             'cell_type': self.cell_type,
                             'angle': angle},
                            index=[0]
                            )

    def get_results_as_df(self, angle):
        with temp_setattr(self.block, 'current_condition', {'keep': True, 'angle': angle}):
            return pd.concat([self._get_base_df(angle), self.block.get_results_df()], axis=1)
