# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:09:50 2014

@author: crousse
"""
from collections import OrderedDict
import matplotlib.pyplot as plt
from trace import Trace


class TracesSet(object):
    """
    A group of traces of the same channel
    Contains group information
    """
    
    def __init__(self, srcTraces, metadata):
        traces = OrderedDict()
        for key in srcTraces.keys():
            traces[key] = Trace({key: srcTraces[key]}, metadata)
        self.traces = traces
    
    def __len__(self):
        return len(self.traces)
        
    def __getitem__(self,  idx):
        if idx not in range(len(self)):
            raise KeyError('Index {} out of range {} to {}'.format(idx,  0,  len(self)))
        key = self.names[idx]
        return self.traces[key]
        
    def __iter__(self):
        return iter(self.traces.values())
    
    @property
    def names(self):
        return self.traces.keys()
    
    @property
    def channel(self):
        return self.traces[0].strip('Record')[0]
        
    def plot(self):
        for trace in self:
            trace.plot()
        plt.show()
