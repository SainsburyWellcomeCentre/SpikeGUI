import csv
import warnings
from math import floor, ceil

warnings.filterwarnings('error')

import numpy as np

from src.plotting.polar_graphs import PolarPlot, ScatterPlot
from src.signal_processing import mat_utils


class ResampledMatrix(object):
    """
    A class to represent a matrix of resampled data from igor vestibular stim experiment.
    The data should be of shape (nTrials, nDegrees, nHalfCycles) 
    and will be transposed to (nDegrees, nTrials, nHalfCycles). 
    """
    
    def __init__(self, exp, name, ext='png', stats_path='stats.txt', lines=True):
        """
        :param exp: The source experiment
        :type exp: Experiment
        :param str name: The name of the matrix
        :param str ext: The extension without the leading dot
        :param str stats_path: The suffix to save the statistics
        :param bool lines: whether to TODO
        """
    
        self.name = name
        self.exp = exp
        self.group_name = self._get_group_name()
        self.activity_type = self._get_activity_type()

        self.save_csv = False
        self.plot_graphs = False

        self.data = (exp.data[name]).transpose( (1, 0, 2) ) # flip dim 1 and 0
        
        if self.activity_type == 'spiking':  # Hack to overwrite
            if self.group_name == 'position':
                self.data = self.exp.position_spiking
                self.durations = self.exp.position_durations
            elif self.group_name == 'velocity':
                self.data = self.exp.velocity_spiking
                self.durations = self.exp.velocity_durations
            elif self.group_name == 'acceleration':
                self.data = self.exp.acceleration_spiking
                self.durations = self.exp.acceleration_durations
        
        self.shuffled_data = mat_utils.shuffle(self.data)

        self.x_data = self.bins = None  # declared here before set in self.get_groups()
        self.get_groups()

        matrix_name = '{}_{}'.format(self.group_name, self.activity_type)
        self.matrix_name = matrix_name
        self.c_wise_name = '{}_clockwise'.format(matrix_name)
        self.c_c_wise_name = '{}_counter_clockwise'.format(matrix_name)
        self.both_name = '{}_all_pooled'.format(matrix_name)

        self.binned_data = self.bin(self.data)
        self.binned_shuffled_data = self.bin(self.shuffled_data)
        if self.activity_type == 'spiking':
            self.binned_durations = self.bin(self.durations)
            self.binned_data /= self.binned_durations
        self.avg = mat_utils.avg(self.binned_data)  # TODO: check if not better on original
        self.lines = lines
        self.ext = ext
        self.stats_path = stats_path
        
    def _get_activity_type(self):
        """
        Checks whether this is sub ('Vm') or suprathreshold activity ('spiking')
        """
        name = self.name.lower()
        return 'spiking' if 'raster' in name else 'Vm'
        
    def _get_group_name(self):
        """
        Checks whether this is position, velocity or acceleration
        """
        name = self.name.lower()
        name = name.replace('matrix', '')
        name = name.replace('normalised', '')
        name = name.replace('raster', '')
        name = name if name else 'position'
        if name not in ['position', 'velocity', 'acceleration']:
            raise AssertionError('Unknown x_data: {}'.format(name))
        return name
        
    def get_groups(self):
        """
        Exctracts self.x_data
        Bins self.x_data to create self.bins
        """
        if self.group_name.startswith('vel'):
            key = 'velocities'
        elif self.group_name.startswith('acc'):
            key = 'accelerations'
        elif self.group_name.startswith('position'):
            key = 'degrees'
        else:
            raise AttributeError("Unknown group: {}".format(self.group_name))
        self.x_data = self.exp.data[key]
#            bin_size = round((self.x_data.max() - self.x_data.min()) / self.x_data.max())
        bin_size = 5
        self.bins = self.bin_x_axis(bin_size)
        
    def get_freq(self, spiking, durations):
        """
        Computes the frequency ignoring NaNs (zero division error) and similar warnings
        """
        warnings.filterwarnings('ignore')
        freq = spiking / durations
        warnings.filterwarnings('error')
        return freq

    def bin_x_axis(self, bin_size):
        """
        Bins self.x_data with 'bin_size' and returns the result
        """
        x_data = self.x_data
        min_angle = floor(x_data[0]); max_angle = ceil(x_data[-1])  # FIXME: check if need floor and ceil since fix x_data
        binned_x_data = np.arange(min_angle, max_angle, bin_size)
        binned_x_data = binned_x_data.astype(np.float32)
        return binned_x_data
        
    def bin(self, mat):
        """
        Bins the input array using self.bins
        Assumes a 3D array
        """
        bins = self.bins
        binned_mat = np.empty((len(bins), mat.shape[1], mat.shape[2]), dtype=np.float64)
        for i in range(mat.shape[1]):
            for j in range(mat.shape[2]):
                binned_mat[:, i, j] = self._bin_vector(mat[:, i, j])
        return binned_mat
        
    def _bin_vector(self, vect):
        """
        Bins the input 1D array using self.bins and returns the result
        Constructs bins by sum for activity type spiking and by average
        for activity type Vm
        """
        bins = self.bins
        x_data = self.x_data
        if len(vect) != len(x_data):
            raise ValueError("Argument vector and x_data length must match, got {} and {}".
                             format(len(vect), len(x_data)))
        binned_vect = np.zeros(len(bins))
        levels = np.digitize(x_data, bins)

        for i, level in enumerate(set(levels)):
            if self.activity_type == 'spiking':
                binned_vect[i] = (vect[levels == level]).sum()
            else:
                binned_vect[i] = (vect[levels == level]).mean()
        return binned_vect
        
    def save_binned_data(self, dest):
        """
        Writes binned data to csv file
        """
        if self.group_name == 'position':
            data = self.binned_data
        else:
            data = np.dsplit(self.binned_data, self.binned_data.shape[2])
            data = np.concatenate(data, axis=1)  # FIXME: use dstack or similar
        if self.save_csv:
            self.write_mat(data, dest)
        
    def _get_initial_orientation(self):
        """
        Checks whether the command starts clockwise or counterclockwise
        Assumes cmd.max ~= -(cmd.min())
        returns 1 for clockWise 0 for counterclockwise.
        """
        cmd = self.exp.data['cpgCommandCut']
        epsilon = 1  # +/- 1 degree
        c_min = -(cmd.min())
        c_max = cmd.max()
        assert (c_min - epsilon) < c_max < (c_min + epsilon), "Command not centered on zero"
        return 1 if (cmd[0] < 0) else 0
        
    def _pool_orientations(self, binned_data):
        """
        Returns 3 2D arrays from a 3D one.
        The first one is the clockwise (all even or odd layers depending on the initial orientation)
        The second one is the counterclockwise
        The third is the concatenation of the 2
        """
        initial_orientation = self._get_initial_orientation()
        layers = [binned_data[:, :, i] for i in range((1 - initial_orientation), binned_data.shape[2], 2)]
        c_wise = np.dstack(layers)
        layers = [binned_data[:, :, i] for i in range(initial_orientation, binned_data.shape[2], 2)]
        c_c_wise = np.dstack(layers)
        both = binned_data.copy()
        
        return c_wise, c_c_wise, both
            
    def analyse(self, do_spiking_difference=False, do_spiking_ratio=False):
        """
        Perform plots and anovas on the data.
        """
        c_wise, c_c_wise, bothOri = self._pool_orientations(self.binned_data)
        shuffled_c_wise, shuffled_c_c_wise, shuffled_both_ori = self._pool_orientations(self.binned_shuffled_data)
        # WARNING: bothOri != self.binned_data (only 2D so no '-----' separation in csv)
        plot_type = 'polar' if (self.group_name == 'position') else 'scatter'

        if self.plot_graphs:
            self.analyse_mat(bothOri, self.both_name, plot_type)
            self.analyse_mat(shuffled_both_ori, self.both_name + 'Shuffled', plot_type)
            if self.group_name == 'position':
                self.analyse_mat(c_wise, self.c_wise_name, plot_type)
                self.analyse_mat(shuffled_c_wise, self.c_wise_name + 'Shuffled', plot_type)
                self.analyse_mat(c_c_wise, self.c_c_wise_name, plot_type)
                self.analyse_mat(shuffled_c_c_wise, self.c_c_wise_name + 'Shuffled', plot_type)
        
        if self.activity_type == 'spiking':
            if do_spiking_ratio:
                self.normalise_by_bsl_freq_and_analyse(bothOri, self.both_name, plot_type)
                if self.group_name == 'position':
                    self.normalise_by_bsl_freq_and_analyse(c_wise, self.c_wise_name, plot_type)
                    self.normalise_by_bsl_freq_and_analyse(c_c_wise, self.c_c_wise_name, plot_type)
        
    def analyse_mat(self, mat, name, plot_type):
        """
        Performs anova on the array 'mat' and saves plot (of type plot_type)
        and csv of array
        """
        if plot_type == 'polar': self.plot_polar(np.nan_to_num(mat), name)  # WARNING rescues NaNs
        elif plot_type == 'scatter': self.plot_scatter(np.nan_to_num(mat), name)
        else:
            raise NotImplementedError('Unrecognised plot type: {}'.format(plot_type))
        self._simple_anova(mat, name)
        self.write_mat(mat, name)
                                
    def plot_scatter(self, mat, name):
        """
        Plots array 'mat' as a scatter plot
        """
        plot = ScatterPlot(mat, self.bins, dest=name)
        plot.plot(self.ext)

    def plot_polar(self, mat, name):
        """
        Plots array mat as a polar plot
        """
        do_offset = self.activity_type.lower() == 'vm'
        polar = PolarPlot(mat, self.bins, doOffset=do_offset, dest=name)
        polar.plot(self.ext)
        
    def write_mat(self, mat, dest):
        """
        Writes array mat (up to 3 dimensions) to csv using dest as output path
        The third dimension if present appears separated by '-' * nColumns
        """
        dest = dest if dest.endswith('.csv') else dest+'.csv'
        with open(dest, 'w', newline='') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if mat.ndim == 1:
                writer.writerow(mat)
                return
            elif mat.ndim == 2:
                layers = [mat]
            elif mat.ndim == 3:
                layers = [mat[:, :, z] for z in range(mat.shape[2])]
            else:
                raise NotImplementedError('Number of dimension {} is not implemented'.format(mat.ndim))
            for layer in layers:
                for y in range(layer.shape[1]):
                    writer.writerow(layer[:, y])
                writer.writerow(['-']*mat.shape[0])  # separate layers by '----------'

    def normalise_by_bsl_freq_and_analyse(self, mat, name, plot_type):  # FIXME: lacks documentation
        normalised_mat = self.normalise_spiking(mat)
        if self.plot_graphs:
            self.analyse_mat(normalised_mat, name + 'Normalised', plot_type)

    def normalise_spiking(self, mat):  # FIXME: lacks documentation
        normalised_spiking = np.zeros(mat.shape, dtype=np.float64)
        warnings.filterwarnings('ignore')
        normalised_spiking /= 0  # Force to NaN
        warnings.filterwarnings('error')

        for i in range(mat.shape[1]):
            diff = mat[:, i] - self.exp.bsl_spiking_freq[i]
            try:
                normalised_spiking[:, i] = diff / self.exp.bsl_spiking_freq[i]
            except RuntimeWarning:  # Remains NaN
                if __debug__:
                    print('Baseline spiking is {} for trial {}. Values: {}'.
                          format(self.exp.bsl_spiking_freq[i], i, normalised_spiking))
        return normalised_spiking

    def _simple_anova(self, mat, title):
        """
        Performs a one way anova on the matrix mat (2D) and prints the result with title as prefix
        """
        pass
        #        if mat.ndim == 3: # flatten
        #            layers = [mat[:,:,i] for i in range(mat.shape[2])]
        #            mat = np.concatenate(layers, axis=1)
        #        args = [np.squeeze(a) for a in np.vsplit(mat, mat.shape[0])] # Rows have repeats
        #        if __debug__:
        #            print("Number of groups: {}, number of repeats: {}".format(len(args), len(args[0])))
        #        f, p = oneWayAnova(*args)
        ##        p = float('NaN')
        #        color = "\033[1;31m" if p < 0.05 else "\033[0m"
        #        color = color if not isnan(p) else "\033[7;33m"
        #        result = "\t{}, P value = {}{:.3f}\033[0m".format(title, color, p)
        #        print(result)
        #        with open(self.stats_path, 'a') as outFile:
        #            outFile.write(result + '\n')
