import struct

"""
classes for handling binary files from probe acquisition without having to think too much about byte number  etc.
"""
# TODO: inherit from struct instead?


class DataPoint(object):
    def __init__(self, base_format='h'):
        """
        one dataPoint should correspond to a single value from a single recording channel.
        i.e. byte_width
        :param base_format:
        """
        self.base_format = base_format
        self.s = struct.Struct("{}".format(self.base_format))
        self.size = self.s.size


class TimePoint(object):
    def __init__(self, data_point, n_chan):
        """
        binary data corresponding to one time point for all channels (i.e. n_channels * byte_width)
        :param data_point:
        :param n_chan:
        """
        self.base_format = data_point.base_format
        self.data_point = data_point
        self.n_chan = n_chan
        self.n_data_points = n_chan
        self.s = struct.Struct("{}{}".format(self.n_data_points, data_point.base_format))
        self.size = self.s.size
        self.byte_width = self.data_point.size


class Chunk(object):
        """
        binary data in units of time points i.e. n_channels * n_samples * byte_width
        :param dataPoint:
        :param n_chan:
        """
        def __init__(self, time_point, n_samples):
            self.base_format = time_point.base_format
            self.time_point = time_point
            self.n_data_points = time_point.n_chan * n_samples
            self.s = struct.Struct("{}{}".format(self.n_data_points, self.base_format))
            self.size = self.s.size
            self.byte_width = time_point.byte_width
            self.n_chan = time_point.n_chan
