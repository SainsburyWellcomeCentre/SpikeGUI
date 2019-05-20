# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:08:42 2014

@author: crousse
"""

from collections import OrderedDict
from board import BoardConf
from traceSet import TracesSet

class Protocol(object):
    """
    An ephys protocol (I/V, ramp, gratings ....)
    """
    def __init__(self, data, recordingMetadata): # data = reconding.data
 
        metadata = data['vars']
        self.name = metadata['StimTag'][0] #
        self.prefix = metadata['WavePrefix'][0]
        self.nmVersion = metadata['Version'][0] #
        self.sampleInterval = metadata['SampleInterval'][0] #
        self.interRepetitionsTime = metadata['InterRepTime'][0] #
        self.interStimTime = metadata['InterStimTime'][0] #
        self.numPulseVar = metadata['NumPulseVar'][0]# the number of variables in the pulse table
        self.duration = metadata['TotalTime'][0]
        self.numStims = metadata['NumStimWaves'][0]
        self.numRepetitions = metadata['NumStimReps'][0]

        # From recording metadata
        self.acqMode = recordingMetadata['AcqMode'][0]
        self.xLabel = recordingMetadata['xLabel'][0]
        self.channels = recordingMetadata['NumChannels'][0]
        
        # Protocol specific. To be set by user
        self.recordingChan = 0
        self.stimChan = 1
        self.injectionChan = 2

        self.boardConf = BoardConf(data['BoardConfigs'])
        
        self.activeIOs = sorted([key.replace('_pulse',  '') for key in data.keys() if key.endswith('pulse')])
        self.pulseTable = self._parseTable(data)
        self.pulseTraces = self._getTraces(data) # TODO: give labels
    
    @property
    def nChannels(self):
        return len(self.channels)
        
    def plot(self,  io):
        if io in self.activeIOs:
            set = self.pulseTraces[io]
            set.plot()
        
    def _parseTable(self, data):
        """
        Returns the table of all pulses reshaped to a matrix of N_PROTO_VARIABLES columns
        """
        pulseTable = []
        for key in self.activeIOs:
            pulseTable.append(zip(*[iter(data[key+'_pulse'])]*self.numPulseVar))
        return pulseTable
            
    def _getTraces(self, data):
        pulses = []
        metadata = (self.xLabel, self.sampleInterval, "") # TODO: get units from boardConf
        traceSets = []
        newSet = None
        for key in sorted(data.keys()):
            if key not in pulses:
                pulses.append(key)
                if newSet is not None:
                    traceSets.append(newSet)
                newSet = OrderedDict() # reset
            for ioKey in self.activeIOs:
                if key.startswith('u'+ioKey):
                    newSet[key] = data[key]
        sets = [TracesSet(traceSet,  metadata) for traceSet in traceSets]
        return sets
