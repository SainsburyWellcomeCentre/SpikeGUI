import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tqdm import trange, tqdm

from margrie_libs.margrie_libs.stats.stats import wilcoxon
from margrie_libs.margrie_libs.signal_processing.cross_correlation import normalised_periodic_cross_cor

from rotation_analysis.analysis.block import BlockException
from rotation_analysis.analysis.probe.ctxt_manager_decorator import temp_setattr
from rotation_analysis.analysis.stimulus import RotationStimulus


class FieldOfViewError(Exception):
    pass


class FieldOfView(object):
    """
    A class representing a given field of view (i.e. typically at a given depth)
    Its main attribute is the list of cells in that FOV
    """
    def __init__(self, cells, trials_to_remove, main_dir, depth):
        self.stats = pd.DataFrame()
        self.cells = cells
        self.trials_to_remove = trials_to_remove
        self.main_dir = main_dir
        self.depth = depth

        self.ref_vars = ('position', 'distance', 'velocity', 'acceleration')
        self.spiking_tables = {}
        for angle in self.angles:
            self.spiking_tables[angle] = {}
            for ref_var_name in self.ref_vars:
                self.spiking_tables[angle][ref_var_name] = None

        self.default_cmap = plt.get_cmap('inferno')

        self.imshow_params = {'interpolation': 'none',
                              'aspect': 'auto',
                              'origin': 'lower',
                              'cmap': self.default_cmap}

    def __len__(self):
        return [c.skip for c in self.cells].count(False)

    def __getitem__(self, item):
        return self.analysed_cells[item]

    def get_str(self, angle):
        return 'depth_{}_angle_{}'.format(self.depth, angle)

    @property
    def analysed_cells(self):
        return [c for c in self.cells if not c.skip]

    @property
    def first_cell(self):
        for cell in self.cells:
            if not cell.skip:
                return cell

    @property
    def analysed_metrics(self):
        return 'freq', 'weighted_ampl', 'fluo'

    @property
    def condition_pairs(self):
        return RotationStimulus.condition_pairs

    @property
    def angles(self):
        return self.first_cell.angles

    @staticmethod
    def check_all_trials_have_same_n_repeats(block):
        if len(set([t.stimulus.n_repeats for t in block.kept_trials])) != 1:
            raise BlockException('All trials must have the same number of repeats, got: {}'
                                 .format([t.stimulus.n_repeats for t in block.kept_trials]))

    def get_all_cells_all_trials_spiking_tables(self, angle, levels_var_name='position'):
        spiking_tables = []
        for i, cell in enumerate(tqdm(self)):
            block = cell.block
            with temp_setattr(block, 'current_condition', {'keep': True, 'angle': angle}):
                FieldOfView.check_all_trials_have_same_n_repeats(block)   # TODO: ideally check all cells have same ?
                spiking_table, levels, cmd = block.get_freqs_from_timetable(levels_var_name)
                spiking_tables.append(spiking_table)
        spiking_tables = np.dstack(spiking_tables)  # TEST: check shape

        spiking_tables = np.transpose(spiking_tables, (2, 0, 1))
        return levels, spiking_tables, cmd

    def get_spiking_table(self, angle, levels_var_name='position'):
        if self.spiking_tables[angle][levels_var_name] is None:
            self.spiking_tables[angle][levels_var_name] = self._get_spiking_table(angle, levels_var_name)
        return self.spiking_tables[angle][levels_var_name]

    def _get_spiking_table(self, angle, levels_var_name='position'):
        levels, table, cmd = self.get_all_cells_all_trials_spiking_tables(angle, levels_var_name)
        if not table.shape[0] == len(self):
            raise FieldOfViewError('Table shape is incorrect. should have n_cells ({}) in the first dimension,'
                                   ' got {}. Table shape is {}'
                                   .format(len(self), table.shape[2], table.shape))
        return levels, table.mean(axis=2), cmd

    def do_cell_stats(self):
        """
        Wilcoxon of all matching columns and output to DF -> csv

        :return:
        """
        for angle in self.angles:
            df_dict = {}
            angle_stats = self.stats[self.stats['angle'] == angle]
            for c1, c2 in self.condition_pairs:  # FIXME: check which baseline in FOV vs block
                for metric in self.analysed_metrics:
                    if c1 == 'bsl_short':
                        col1 = '{}_{}_{}'.format(c1, c2, metric)
                    else:
                        col1 = '{}_{}'.format(c1, metric)
                    col2 = '{}_{}'.format(c2, metric)
                    col_name = '{}_vs_{}_{}'.format(c1, c2, metric)
                    df_dict[col_name] = wilcoxon(angle_stats[col1], angle_stats[col2])
            df = pd.DataFrame(df_dict, index=[0])
            csv_filename = '{}_stats.csv'.format(self.get_str(angle))
            csv_file_path = os.path.join(self.main_dir, csv_filename)
            df.to_csv(csv_file_path)

    def plot_psth(self, extension):
        for angle in self.angles:
            plt.figure()
            angle_bsl_psth_y = []
            angle_spin_psth_y = []
            angle_bsl2_psth_y = []
            for cell in self:
                with temp_setattr(cell.block, 'current_condition', {'angle': angle}):
                    bsl_psth, spin_psth, bsl2_psth = cell.block.get_psth()
                    bsl_psth_x, bsl_psth_y = bsl_psth
                    spin_psth_x, spin_psth_y = spin_psth
                    bsl2_psth_x, bsl2_psth_y = bsl2_psth
                    angle_bsl_psth_y.append(bsl_psth_y)
                    angle_spin_psth_y.append(spin_psth_y)
                    angle_bsl2_psth_y.append(bsl2_psth_y)
            angle_bsl_psth_y = np.array(angle_bsl_psth_y, dtype=np.float64).sum(0)
            angle_spin_psth_y = np.array(angle_spin_psth_y, dtype=np.float64).sum(0)
            angle_bsl2_psth_y = np.array(angle_bsl2_psth_y, dtype=np.float64).sum(0)
            plt.step(bsl_psth_x, angle_bsl_psth_y, where='post', color='blue')
            plt.step(spin_psth_x, angle_spin_psth_y, where='post', color='green')
            plt.step(bsl2_psth_x, angle_bsl2_psth_y, where='post', color='red')
            plt.savefig(os.path.join(self.main_dir, '{}_PSTH.{}'.format(self.get_str(angle), extension)))
            plt.close()

    def analyse(self, do_stats, do_plots, extension):
        if do_stats:
            figs = {a: plt.figure(a) for a in self.angles}

        for i, cell in enumerate(tqdm(self, desc='Performing stats of cells')):
            cell.remove_trials(self.trials_to_remove)

            if do_stats:
                for angle in cell.angles:
                    plt.figure(angle)  # set active

                    # TRIALS BASED
                    cell.set_main_dir(self.main_dir)
                    cell.analyse_block(angle)

        if do_plots:
            self.plot_psth(extension)  # TODO: deal with depth iterate angle first, cell second
            for angle in self.angles:
                self.plot_spiking_heatmaps(angle)
                self.plot_spiking_histogram(angle)

        if do_stats:  # SUMMARY (after all cells all angles)
            for cell in self:
                for angle in self.angles:
                    # CELL BASED
                    row = cell.get_results_as_df(angle)
                    self.stats = self.stats.append(row)  # FIXME: necessary ?
                    if do_plots:
                        cell.plot_all(angle)

            for angle in self.angles:

                csv_file_name = '{}_all_cells.csv'.format(self.get_str(angle))
                csv_file_path = os.path.join(self.main_dir, csv_file_name)
                self.stats[self.stats.angle == angle].to_csv(csv_file_path)
            self.do_cell_stats()

    def plot_spiking_heatmaps(self, angle):
        for ref_var_name in self.ref_vars:
            self._plot_spiking_heatmap(angle, ref_var_name)

    def _plot_spiking_heatmap(self, angle, ref_var_name='position'):
        f = plt.figure()
        _, spiking_table, cmd = self.get_spiking_table(angle, ref_var_name)
        plt.imshow(spiking_table,
                   extent=(cmd.min(), cmd.max(),  # TODO: for ditance: 0-cmd.max() - cmd.min()
                           0, len(self)),
                   **self.imshow_params)
        plt.title('{}_{}'.format(self.get_str(angle), ref_var_name))
        plt.colorbar()
        # plt.show()

        fig_name = '{}_{}_spiking_heatmap.{}'.format(self.get_str(angle), ref_var_name, self.first_cell.ext)
        fig_path = os.path.join(self.main_dir, fig_name)
        plt.savefig(fig_path)
        plt.close(f)
        
        df = pd.DataFrame(spiking_table, columns=np.linspace(cmd.min(), cmd.max(), spiking_table.shape[1]))
        csv_name = '{}_{}_spiking_heatmap.csv'.format(self.get_str(angle), ref_var_name)
        df.to_csv(os.path.join(self.main_dir, csv_name))

    def plot_spiking_histogram(self, angle):
        for ref_var_name in self.ref_vars:
            self._plot_spiking_histogram(angle, ref_var_name)

    def _plot_spiking_histogram(self, angle, ref_var_name='position'):
        levels, spiking_table, _ = self.get_spiking_table(angle, ref_var_name)

        spiking_means = spiking_table.mean(axis=0)
        spiking_stds = spiking_table.std(axis=0)

        f = plt.figure()

        bar_width = RotationStimulus.get_spacing_from_ref_var_name(ref_var_name)
        plot_degrees = levels[:len(spiking_means)]  # FIXME: should always be -1
        plt.bar(plot_degrees, spiking_means, width=bar_width, yerr=spiking_stds)
        plt.ylabel('frequency (Hz)')
        plt.xticks(plot_degrees)

        plt.title('{}_{}'.format(self.get_str(angle), ref_var_name))

        fig_name = '{}_{}_spiking_histogram.{}'.format(self.get_str(angle), ref_var_name, self.first_cell.ext)
        fig_path = os.path.join(self.main_dir, fig_name)

        print(os.path.join(self.main_dir, self, fig_name))

        plt.savefig(fig_path)
        plt.close(f)

    # def plot_cross_cor(self, angle):
    #     spin_xcor = self.make_cells_spin_correlation_matrix(angle)
    #     plt.imshow(spin_xcor, **self.imshow_params); plt.show()
    #     bsl_xcor = self.make_cells_bsl_correlation_matrix(angle)
    #     plt.imshow(bsl_xcor, **self.imshow_params); plt.show()
    #     plt.imshow(spin_xcor / bsl_xcor, **self.imshow_params)
    #     plt.colorbar(); plt.show()

    # def make_cells_spin_correlation_matrix(self, angle):
    #     mat = np.zeros((len(self), len(self)), dtype=np.float64)
    #     for cell in self:
    #         cell.blocks[angle].concatenated_spin = cell.blocks[angle].concatenate_centre_spins()
    #     for i in trange(len(self), desc='Analysing correlation'):
    #         for j in range(i):
    #             k = j-i
    #             mat[k, j] = normalised_periodic_cross_cor(self[k].blocks[angle].concatenated_spin,
    #                                                       self[j].blocks[angle].concatenated_spin).max()
    #     return mat
    #
    # def make_cells_bsl_correlation_matrix(self, angle):
    #     mat = np.zeros((len(self), len(self)), dtype=np.float64)
    #     for cell in self:
    #         cell.blocks[angle].concatenated_bsl = cell.blocks[angle].concatenate_baselines()
    #     for i in trange(len(self), desc='Analysing correlation'):
    #         for j in range(i):
    #             k = j-i
    #             mat[k, j] = normalised_periodic_cross_cor(self[k].blocks[angle].concatenated_bsl,
    #                                                       self[j].blocks[angle].concatenated_bsl).max()
    #     return mat
