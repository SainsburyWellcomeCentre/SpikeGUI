# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:09:11 2014

@author: crousse
"""
from collections import OrderedDict
from recording import Recording
from mouse import Mouse
from pipette import Pipette

class Cell(object):
    """
    A complete set of recordings/protocols pertaining to a single cell
    """
    def __init__(self, data, cellID):
        self.id = cellID
        self.mouseData = Mouse() # To be filled later manually by user
        self.pipette = Pipette() # To be filled later manually by user
        self.stims = data['Stims'].keys() # TODO: document
        self.recordings = self._parseRec(data)
        self.__attributize()

    def __getitem__(self, key):
        if key not in self.recordings.keys():
            raise KeyError('Key {} not a valid key for cell {}'.format(key,  self.id))
        return self.recordings[key]
    
    def __iter__(self):
        return iter(self.recordings.values())
    
    def __attributize(self):
        for name,  rec in self.recordings.iteritems():
            setattr(self,  '{}'.format(name),  rec)

    def _parseRec(self, data):
        metadataDicts = ['Stims', 'vars', 'Packages', 'Logs', 'nm_folder0']
        recordingsKeys = [ddict for ddict in data if ddict not in metadataDicts]
        for key in recordingsKeys:
            parts = key.split('_')
            if not(parts[0][-1] == self.id):
                recordingsKeys.remove(key)
        recordings = OrderedDict()
        for key in recordingsKeys:
            recordings[key] = Recording({key: data[key]}, self)
        return recordings
