# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:03:19 2014

@author: crousse
"""

class Pipette(object):
    def __init__(self):
        self.resistance = 0
        self.internal = ""
        self.id = 0 # which pipette number
        self.depth = 0.0
        self.hitQuality = 0 # 0 to 5
        self.sealQuality = 0 # 0 to 5
        self.seriesResistance = 0 # different from resistance
        self.vm = 0
        self.pullOff = 0 # 0 to 5
        self.remark = ''
