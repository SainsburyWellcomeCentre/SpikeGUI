# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:59:10 2014

@author: crousse
"""

import numpy as np
import matplotlib.pyplot as plt

class Trace(np.ndarray):
    """
    An electrophys recording trace, similar to an igor wave
    """
    def __new__(cls, data, metadata, name=''):
        
        assert (len(data)==1), "More than one trace found as argument: {}"\
        .format(data.keys())

        name = data.keys()[0]
        trace = np.asarray(data[name]).view(cls)

        trace.name = name
        trace.xUnits = metadata[0]
        trace.deltaX = metadata[1]
        trace.yUnits = metadata[2]
        if len(metadata)==4:
            trace.startX = metadata[3]
        else:
            trace.startX = 0
        return trace
        
    def __array_finalize__(self, trace):
        if trace is None: return
        self.name = getattr(trace, 'name', None)
        self.xUnits = getattr(trace, 'xUnits', None)
        self.startX = getattr(trace, 'startX', None)
        self.deltaX = getattr(trace, 'deltaX', None)
        self.yUnits = getattr(trace, 'yUnits', None)

    @property
    def x(self):
       return np.array(range(self.startX, len(self))) * self.deltaX
        
    def plot(self, yTrace=None):
        plt.plot(self.x, self)
        plt.xlabel(self.xUnits)
        plt.ylabel(self.yUnits)
