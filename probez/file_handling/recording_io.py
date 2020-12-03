import math
import os
import struct
from itertools import compress

import numpy as np

from file_handling import binary_classes
from util import detrending


class RecordingIo:

    def __init__(self, path, n_chan):
        self.path = path
        self.root = os.path.split(path)[0]
        self.file_name = os.path.split(path)[1]
        self.name = self.file_name.split('.')[0]
        self.n_chan = n_chan
        self.dtype = 'h'
        self.byte_width = struct.calcsize(self.dtype)
        self.data_point = binary_classes.DataPoint(base_format=self.dtype)
        self.time_point = binary_classes.TimePoint(self.data_point, n_chan=self.n_chan)

    @property
    def _size(self):
        info = os.stat(self.path)
        return info.st_size

    def create_chunks(self, n_samples_total=None, n_samples_to_process=50000):

        if n_samples_total is None:
            n_samples_total = self._size
        if n_samples_to_process > n_samples_total:
            n_samples_to_process = n_samples_total

        chunk = binary_classes.Chunk(self.time_point, n_samples_to_process)  # define normal chunk

        leftover_bytes = n_samples_total % chunk.size  # this is a problem if the last chunk is very small
        leftover_samples = int(leftover_bytes/self.time_point.size)
        last_chunk = binary_classes.Chunk(self.time_point, leftover_samples)  # define final chunk

        print('chunk size is {}, last chunk is {} bytes:'.format(chunk.size, leftover_bytes))
        print('leftover samples = {}'.format(leftover_samples))

        n_chunks = math.ceil(n_samples_total/chunk.size)
        return n_chunks, chunk, last_chunk

    @staticmethod
    def append_chunk(f_out, chunk_out):
        f_out.write(bytes(chunk_out))

    @staticmethod
    def get_next_data_chunk(f_in, chunk_struct):
        """

        :param file f_in:
        :param struct.Struct chunk_struct:
        :return:
        """
        return f_in.read(chunk_struct.size)

    @staticmethod
    def get_data(chunk_in, chunk_struct):
        data = chunk_struct.s.unpack_from(chunk_in)
        n_samples = int(chunk_struct.size/chunk_struct.byte_width/chunk_struct.n_chan)
        reshaped_data = np.array(data).reshape(n_samples, chunk_struct.n_chan)
        return reshaped_data

    def pack_data(self, data, chunk_struct):
        """

        :param data:
        :param chunk_struct:
        :return:
        """
        packable_data = self._make_packable(data, chunk_struct)  # reshape into packable format
        chunk_out = chunk_struct.s.pack(*packable_data)  # pack
        return chunk_out

    @staticmethod
    def _make_packable(data, chunk):
        new_data_length = int(chunk.size/chunk.byte_width)
        data = data.reshape(new_data_length)
        return tuple(data)

    @property
    def data_shape(self):
        if self._size % self.n_chan != 0:
            raise ValueError('size: {} or n_chan: {} incorrect'.format(self._size, self.n_chan))
        n_samples = self._size/self.n_chan/self.byte_width
        n_channels = self.n_chan
        return n_samples, n_channels

    def process_to_file(self, f_in_path, f_out_path, n_chan, channels_to_discard,
                        processing_func=detrending.denoise_detrend, n_samples_to_process=50000,
                        on_data=True, start_sample=0, end_sample=None):

        # TODO: make this work for both chunk and data operations at the same time
        # TODO: make this much cleaner
        # TODO: multiple output files

        start_byte = start_sample * self.time_point.size  # time point is multiple of n_chan
        end_byte = self._size if end_sample is None else end_sample * self.time_point.size

        print(f_in_path, f_out_path)
        with open(f_in_path, 'rb') as f_in:
            f_in.seek(start_byte)
            with open(f_out_path, 'wb') as f_out:
                n_samples_total = end_byte - start_byte
                n_chunks, chunk, last_chunk = self.create_chunks(n_samples_total, n_samples_to_process)

                for i in range(n_chunks):
                    current_chunk_struct = last_chunk if i == n_chunks-1 else chunk

                    print('chunk: {} of {}'.format(i+1, n_chunks))

                    try:
                        chunk_in = self.get_next_data_chunk(f_in, current_chunk_struct)
                    except EOFError:
                        break
                    if processing_func is None:
                        data = self.get_data(chunk_in, current_chunk_struct)
                        chunk_out = self.pack_data(data, current_chunk_struct)
                    elif on_data:
                        data = self.get_data(chunk_in, current_chunk_struct)
                        processed_data = processing_func(data, n_chan)
                        chunk_out = self.pack_data(processed_data, current_chunk_struct)  # pack only works if processing step returns integer values
                    else:
                        print('n_chan_recfile = {}'.format(n_chan))
                        chunk_out, out_channels_bytes = processing_func(chunk_in, n_chan, channels_to_discard)
                        if len(chunk_out) != out_channels_bytes:
                            raise ValueError("Expected to write {} bytes, wrote: {}".format(out_channels_bytes,
                                                                                            len(chunk_out)))
                    self.append_chunk(f_out, chunk_out)

    def make_mask(self, chunk_in, n_chan, channels_to_discard=[]):
        """
        generates a byte mask such that only bytes of channels of interest are marked as True
        :param chunk_in:
        :param n_chan:
        :param channels_to_discard:
        :return:
        """
        mask = []
        byte_width = self.data_point.size
        n_repeats = int(len(chunk_in)/(n_chan*byte_width))
        for i in range(n_chan):
            if i in channels_to_discard:
                mask += [False]*byte_width
            else:
                mask += [True]*byte_width
        return list(np.tile(mask, n_repeats))

    def remove_channels_from_chunk(self, chunk_in, n_chan, channels_to_discard):
        channels_bytes_mask = self.make_mask(chunk_in, n_chan, channels_to_discard)
        n_out_channels_bytes = channels_bytes_mask.count(True)

        chunk_out = list(compress(chunk_in, channels_bytes_mask))  # return only the desired data
        return chunk_out, n_out_channels_bytes


