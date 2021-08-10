# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:08:59 2014

@author: crousse
"""

import numpy

class BoardConf(object):
    def __init__(self, boardData):
        adcFactor = numpy.isnan(boardData['ADCboard'])
        self.adcBoard = self._parseBoardItem(boardData['ADCboard'], adcFactor)
        self.adcChan = self._parseBoardItem(boardData['ADCchan'], adcFactor)
        self.adcGain = self._parseBoardItem(boardData['ADCgain'], adcFactor)
        self.adcMode = self._parseBoardItem(boardData['ADCmode'], adcFactor)
        self.adcName = self._parseBoardItem(boardData['ADCname'], adcFactor)
        self.adcScale = self._parseBoardItem(boardData['ADCscale'], adcFactor)
        self.adcTGain = self._parseBoardItem(boardData['ADCtgain'], adcFactor)
        self.adcUnits = self._parseBoardItem(boardData['ADCunits'], adcFactor)
        
        dacFactor = numpy.isnan(boardData['DACboard'])
        self.dacBoard = self._parseBoardItem(boardData['DACboard'], dacFactor)
        self.dacChan = self._parseBoardItem(boardData['DACchan'], dacFactor)
        self.dacName = self._parseBoardItem(boardData['DACname'], dacFactor)
        self.dacScale = self._parseBoardItem(boardData['DACscale'], dacFactor)
        self.dacUnits = self._parseBoardItem(boardData['DACunits'], dacFactor)
        
        ttlFactor = numpy.isnan(boardData['TTLboard'])
        self.ttlBoard = self._parseBoardItem(boardData['TTLboard'], ttlFactor)
        self.ttlChan = self._parseBoardItem(boardData['TTLchan'], ttlFactor)
        self.ttlName = self._parseBoardItem(boardData['TTLname'], ttlFactor)
        self.ttlScale = self._parseBoardItem(boardData['TTLscale'], ttlFactor)
        self.ttlUnits = self._parseBoardItem(boardData['TTLunits'], ttlFactor)
        
    def  _parseBoardItem(self,  item, boolFactor):
        if type(item[0]) == str:
            return [elt[0] for elt in zip(item, boolFactor) if elt[1] == True]
        else:
            return item[boolFactor==True]
