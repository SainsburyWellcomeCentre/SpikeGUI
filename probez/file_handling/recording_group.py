import os

import numpy as np
from file_handling import recording
from file_handling.file_handling_exceptions import InconsistentNChanError


class RecordingGroup(object):
    """

    class for managing recordings and applying operations to multiple recording instances together

    typically, a single experiment would consist of several 2 minute long recordings separated by a few minutes

    example usage:
    >>> root = "./my/path/to/the/directory"
    >>> exp = RecordingGroup(root, trigger_index=384)


    >>> for rec in exp.recordings:
    >>>     data = rec.get_data('ap')  # load some raw data
    >>>     trigger_trace = data[:, exp.trigger_index]
    >>>     plt.plot(trigger_trace)
    >>> plt.show()

    """
    # FIXME: recordings must have a channel number, so self.recordings is all messed up
    # TODO: exp.recordings not as a property, so that _raw_data n_chan can be put in as an argument

    def __init__(self, root, trigger_index=None):
        self.root = root
        self.probe_type = 'Neuropixels 3A'
        self.trigger_index = trigger_index
        self.root = root

    def raw_file_names(self, extension='.ap.bin'):
        """get all file names of a given extension"""
        file_names = []
        for file_name in os.listdir(self.root):
            if file_name[-len(extension):] == extension:
                file_names.append(file_name.split('.')[0])
        return file_names

    @property
    def recordings(self):
        """list of recording objects associated with experiment"""
        recording_list = []
        for name in self.raw_file_names():
            rec = recording.Recording(self.root, self.root, name, self.trigger_index)
            recording_list.append(rec)
        return recording_list

    def common_avg_ref_all(self, n_chan=None, processing_func=None, out_root=None, trigger_idx=(-1,),
                           save_shortened_trigger=None, end_sample=None, start_sample=0):
        """apply common average reference normalisation to all raw files in directory"""

        all_nchans = set([rec.n_chan for rec in self.recordings])

        if len(all_nchans) != 1:
            raise InconsistentNChanError('expected equal channel numbers for all recordings, got {}'.format(all_nchans))

        for rec in self.recordings:
            rec.subtract_common_avg_reference(n_chan=n_chan, start_sample=start_sample, end_sample=end_sample,
                                              save_shortened_trigger=save_shortened_trigger, out_root=out_root,
                                              trigger_idx=trigger_idx, processing_func=processing_func)

    def concatenate_files(self, fpath_out, extension):
        """
        join together all files in directory into one file. NOTE: make sure common_avg_ref_all is done first
        :param string fpath_out:
        :param string extension: need to specify the extension as there may be denoised data and non-denoised data e.g.
        :return:
        """

        fout = open(fpath_out, 'wb')

        for rec in np.sort(self.recordings):
            fin = open(rec.get_path(extension), 'rb')  # FIXME: this is stupid
            while True:
                data = fin.read(2 ** 16)
                if not data:
                    break
                fout.write(data)
            fin.close()
        fout.close()