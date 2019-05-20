import os
import shutil
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime as dt

from configobj import ConfigObj

from analysis.event_detection.calcium_trace_handling import load_fluo_profiles


class Recording(object):
    """
    exampleOriginalPath = "...\(20160108_18_57_12\)__spont_100x100ym_256x256pxl/\(20160108_18_57_12\)__spont_100x100ym_256x256pxl.raw"
    """
    def __init__(self, src_dir):
        """
        :param str src_dir: The source directory where the ini file of that recording resides
        """
        self.dir = src_dir
        self.checkIsValid()
        self.ini = self.getIniFilePath()
        self.config = ConfigObj(unrepr=True)
        self.vars = self._parseVariables()
        self.header = None
        self.greenChannel = None
        self.redChannel = None
        self.recNb = None
        if 'date' in self.vars:  # Data already parsed (i.e. Folders already moved)
            self.hasMoved = True
            self.dateTime = self._parseDateTime()
        else:   
            self.hasMoved = False
            self.dateTime = self.getDateTime()
            self.dateToVarIni()
            self.vars = self._parseVariables()
            self._getDepth()
            self.writeIni()
            self._renameFiles()
            self.ini = self.getIniFilePath()
        self.rawImgFilePath = self._getImgFilePath()
        self.derotatedImgFilesPaths = self._getDerotatedImgFilesPaths()
        self.registeredImgFilesPaths = self._getRegisteredImgFilesPaths()

    def __repr__(self):
        return self.ini

    def get_cell_data(self, cell_idx, processing_type):
        """

        .. warning:
            Ignores neuropile

        :param int cell_idx:
        :param str processing_type:
        :return:
        """
        return self.process(cell_idx, processing_type)  # REFACTOR: move to data_processor

    def get_green_trial(self, cell_idx):
        if self.greenChannel.ndim == 3:
            trial = self.greenChannel[:, cell_idx, 0]
        elif self.greenChannel.ndim == 2:
            trial = self.greenChannel[:, cell_idx]
        else:
            raise ValueError("Should have cell or neuropile (1 or 2 options, found {})"
                             .format(self.greenChannel.shape[1]))
        return trial

    def get_red_trial(self, cell_idx):
        if self.redChannel.ndim == 3:
            trial = self.redChannel[:, cell_idx, 0]
        elif self.redChannel.ndim == 2:
            trial = self.redChannel[:, cell_idx]
        else:
            raise ValueError("Should have cell or neuropile (1 or 2 options, found {})"
                             .format(self.redChannel.shape[1]))
        return trial

    def process(self, cell_idx, processing_type='delta_f'):
        processing_functions = {
            'raw': self._get_raw,
            'delta_f': self._get_delta_f,
            'delta_f_ratio': self._get_delta_f_ratio
        }
        return processing_functions[processing_type](cell_idx)

    def _get_raw(self, cell_idx):
        return deepcopy(self.get_green_trial(cell_idx))

    def delta_f(self, f):
        """
        Standard deltaF/F

        :param np.array f:
        """
        f0 = f.mean()
        return (f-f0) / f0
        # return f / f0  # TODO: Add equation to documentation

    def _get_delta_f(self, cell_idx):
        trial = self.get_green_trial(cell_idx)
        return self.delta_f(trial)

    def _get_delta_f_ratio(self, cell_idx):
        green = self.delta_f(self.get_green_trial(cell_idx))
        red = self.delta_f(self.get_red_trial(cell_idx))
        return green / red
        
    def checkIsValid(self):
        """
        Check if the recording has both a valid ini file and a source image
        """
        files = os.listdir(self.dir)
        has_ini = [f for f in files if f.endswith('variables.ini')]
        has_raw = [f for f in files if f.endswith('.raw')]
        if not (has_ini and has_raw):
            raise RuntimeError("Recording directory {} is not valid (does not have a metadata or image file)."
                               .format(self.dir))

    def getProfiles(self):
        """
        :returns: the header, green, red fluorescence profiles of the different cells in the recording extracted from the .csv files
        """
        profiles_paths = self._getProfilesPaths()
        for path in profiles_paths:
            if path.endswith('1.csv'):
                header, greens = load_fluo_profiles(path)
            elif path.endswith('2.csv'):
                _, reds = load_fluo_profiles(path)
        if 'header' in locals():
            self.header = header
        if 'greens' in locals():
            self.greenChannel = greens
        if 'reds' in locals():
            self.redChannel = reds
    
    def getNFrames(self):
        """
        Number of frames in the image stack (time points)
        """
        return self.vars['frame.count']
        
    def getNBslFrames(self):
        """
        """
        n_bsl_frames = int(self.vars['frame.count'] / 4.0)  # TODO: assert always true
        return n_bsl_frames

    def writeIni(self):
        """
        Write the ini files from self.vars
        """
        self.config.filename = self.getIniFilePath()
        self.config['mic'] = self.vars
        self.config.write()  # TODO: see if UnicodeEncodeError check needed

    def _getDepth(self):
        """
        Should only be run before move
        """
        depth = (self.getDepthDir()).strip(" /um")  # compute from ../../
        try:
            depth = int(depth)
        except ValueError:
            print("expected int got {}".format(depth))
            raise
        self.vars['depth'] = depth
    
    def _genericGetFilesPaths(self, suffixes):
        img_file_paths = []
        for filename in os.listdir(self.dir):
            for suffix in suffixes:
                if filename.endswith(suffix):
                    img_file_paths.append(os.path.join(self.dir, filename))
                    break  # Make sure we do not add twice
        return img_file_paths
        
    def _getProfilesPaths(self):
        return self._genericGetFilesPaths( ('_turboReg1.csv', '_turboReg2.csv') )

    def _getDerotatedImgFilesPaths(self):
        return self._genericGetFilesPaths( ('_derotated1.tif', '_derotated2.tif') )

    def _getRegisteredImgFilesPaths(self):
        return self._genericGetFilesPaths( ('_turboReg1.tif', '_turboReg2.tif') )

    def _getImgFilePath(self):
        for filename in os.listdir(self.dir):
            if filename.endswith('image.raw'):
                return os.path.join(self.dir, filename)
        raise IOError("No raw image found for {}".format(self))

    def isRec(self):
        return self.vars['x.pixels'] != self.vars['y.pixels']

    def isHighRes(self):
        return not(self.isRec()) and not(self.isZStack())

    def isZStack(self):
        return True if self.vars['run.z.stack'] else False

    def getRecType(self):
        if self.isHighRes():
            return "highRes"
        elif self.isZStack():
            return "zStack"
        else:
            return "rec"

    def getMaxAngle(self):
        return int(self.vars['rc.maxangle'])
    
    def getIniFilePath(self): 
        files = os.listdir(self.dir)
        ini_file_name = [f for f in files if f.endswith('variables.ini')][0]
        return os.path.join(self.dir, ini_file_name)
    
    def getDateTime(self):
        """
        dateTime from original folder
        Has to be executed before renaming self.dir
        """
        folder_name = os.path.basename(os.path.normpath(self.dir))
        return dt.strptime(folder_name[1:18], "%Y%m%d_%H_%M_%S")

    def _parseDateTime(self):
        """
        dateTime from ini file
        """
        time = dt.strptime(self.vars['time'], '%H:%M:%S').time()
        date = dt.strptime(self.vars['date'], '%Y/%m/%d')
        return dt.combine(date, time)

    def dateToVarIni(self):
        """
        Should be executed before reusing ini file because will strip the end
        """
        with open(self.ini, 'r') as inFile:
            lines = inFile.readlines()
        lines = lines[1:106]  # strip the first line and the binary data at the end
        lines.append('date="{}"\n'.format(self.dateTime.strftime('%Y/%m/%d')))
        lines.append('time="{}"\n'.format(self.dateTime.strftime('%H:%M:%S')))
        with open(self.ini, 'w') as outFile:
            outFile.writelines(lines)
    
    def _parseVariables(self):
        with open(self.ini, 'r') as inFile:
            key_vars = inFile.readlines()
        key_vars = [kv for kv in key_vars if not(kv.startswith('['))]
        vars_dict = OrderedDict()
        for kv in key_vars:
            try:
                k, v = (kv.strip()).split('=')
                k = k.strip()
                v = v.strip("' ")
                vars_dict[k] = self._tryFloat(v.strip('"'))
            except ValueError:
                print(kv)
        return vars_dict
    
    
    def _tryFloat(self, input_str):
        try:
            output = float(input_str)
        except ValueError:
            output = input_str
        return output

    def _renameFiles(self):
        """
        Rename the files to simple names and remove the useless ones
        """
        for f in os.listdir(self.dir):
            f = os.path.join(self.dir, f)
            if f.endswith('protocol.txt'):
                os.remove(f)
            elif f.endswith('macro.txt'):
                os.remove(f)
            elif f.endswith('.bat'):
                os.remove(f)
            elif f.endswith('variables.ini'):
                os.rename(f, f.replace(os.path.basename(f), 'variables.ini'))
            elif f.endswith('.raw'):
                os.rename(f, f.replace(os.path.basename(f), 'image.raw'))

    def getDepthDir(self):
        parent_dir = self.getParentDir()
        return os.path.basename(os.path.normpath(parent_dir))

    def getParentDir(self):
        """
        Returns angle directory
        """
        return os.path.dirname(os.path.normpath(self.dir))
    
    def getBaseDirName(self):
        """
        returns recXX directory
        """
        return os.path.basename(os.path.normpath(self.dir))

    def setRecNb(self, rec_nb):
        if type(rec_nb) == int:
            self.recNb = rec_nb
        else:
            raise ValueError("setRecNb expected an integer, got: {}".format(rec_nb))

    def move(self):
        if self.hasMoved:  # Do Not move twice
            print("Recording {} has already moved. Will not move twice".format(self.dir))
            return
        self.getBaseDirName()
        
        dest_folder = 'maxAngle{}'.format(self.getMaxAngle())
        dest_folder = os.path.join(self.getParentDir(), dest_folder)
        
        new_base_name = "{}{}".format(self.getRecType(), str(self.recNb).zfill(2))

        new_dir_path = os.path.join(dest_folder, new_base_name)
        shutil.move(self.dir, new_dir_path)
        
        self.dir = new_dir_path
        self.ini = self.getIniFilePath()
        self.hasMoved = True


        # def _get_delta_neuropile_ratio(self):  # TODO: see if can merge with above
        #     for i in range(len(self.trials)):  # SK: i has to be a list of integers to iterate over
        #         green = self.trials[i]['green']['deltaNp']
        #         red = self.trials[i]['red']['deltaNp']
        #         self.trials[i]['green']['deltaNpDeltaColor'] = green / red
        #
        # def _get_delta_neuropile(self, color_channel='green'):
        #     if color_channel not in ('green', 'red'):
        #         raise ValueError('Got color channel {} instead of one of ("green", "red")'.format(color_channel))
        #     for angle in self.angles:
        #         for trial, neuropile in zip(self.trials[angle], self.neuropile_trials[angle]):
        #             trial[color_channel]['deltaNp'] = self.delta_neuropile(trial[color_channel]['raw'],
        #                                                                    neuropile[color_channel])
        # def delta_neuropile(self, cell_trial, neuropile_trial):
        #     """
        #     Substract 70% of the neuropile signal
        #     """
        #     cell_delta_f = self.delta_f(cell_trial)
        #     neuropile_delta_f = self.delta_f(neuropile_trial)
        #     return cell_delta_f - (neuropile_delta_f * 0.7)  # WARNING: magic number

