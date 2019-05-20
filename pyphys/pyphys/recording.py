# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:08:22 2014

@author: crousse
"""

from datetime import datetime
from datetime import time
from datetime import date
from collections import OrderedDict
from traceSet import TracesSet
from protocol import Protocol
from note import Note

class Recording(object):
    """
    A complete neuromatic recording.
    Contains a protocol, trace sets and notes
    """
#    PROTOCOLS = ['IVMateo', 'whisk'] # Case sensitive list of all possible protos
    
    def __init__(self, recDict, cell):
        assert (len(recDict)==1), "More than one recording found as argument: {}"\
        .format(recDict.keys())
        
        name = recDict.keys()[0]
        recData = recDict[name]
        
        if not isinstance(name, str): #  WARNING: check for different versions of python
            raise TypeError('name should be a string for {} instead.'.format(type(name)))
        if not name[0].isalpha():
            raise ValueError('name of recording {} should be a string starting with a letter.'.format(name))
        self.name = name
        self.cell = cell # TODO: check if necessary

        self.protocol = Protocol(self._getProtoData(recData), recData['vars'])
        self.traceNames = sorted([key for key in recData.keys() if key.startswith('Record')])
        self.channels = sorted(list(set([traceName[6] for traceName in self.traceNames])))
        self.yLabels = recData['yLabel']
        self.notes = Note(recData['Notes'])
        
        # The actual traces
        self.sets = self._getTraceSets(recData)
        
        recDate = recData['vars']['FileDate'][0]
        recDate = datetime.strptime(recDate,  "%d %b %Y")
        self.date = date(recDate.year,  recDate.month,  recDate.day)
        
        recTime = recData['vars']['FileTime'][0]
        recTime = datetime.strptime(recTime,  "%X")
        self.time = time(recTime.hour, recTime.minute, recTime.second)

    def __getitem__(self,  channel):
        if channel not in self.sets.keys():
            raise KeyError('Channel {} not a valid key for Recording {}'.format(channel,  self.name))
        return self.sets[channel]
    
    def __iter__(self):
        return iter(self.sets.values())
        
    def plot(self,  channel):
        if channel not in self.sets.keys():
            raise KeyError('Channel {} not a valid key for Recording {}'.format(channel,  self.name))
        channel = channel.upper() # TODO: check why doesn't work
        self[channel].plot()
        
    @property
    def nTraces(self):
        return len(self.traceNames)/self.nChannels
        
    @property
    def nChannels(self):
        return len(self.channels)
        
    def _getProtoData(self, recData):
        """
        Extract the protocol from a recording dataset
        """
        protoKeys = [key for key in recData.keys() if key in self.cell.stims]
        if len(protoKeys) == 1:
            return recData[protoKeys[0]]
        else:
            raise ValueError('Should get only one protocol key, got {}: {}'\
            .format(len(protoKeys), protoKeys))
            
    def _getTraceSets(self, recData):
        sets = {}
        xLabel = self.protocol.xLabel
        deltaX = self.protocol.sampleInterval
        for channel, yLabel in zip(self.channels, self.yLabels):
            keys = [key for key in self.traceNames if key[len(self.protocol.prefix)] == channel]
            traceKeys = sorted(keys)
            traces = [recData[key] for key in traceKeys ]
            traceSet = OrderedDict(zip(traceKeys, traces))
            
            metadata = (xLabel, deltaX, yLabel)
            sets[channel] = TracesSet(traceSet, metadata)
        return sets
