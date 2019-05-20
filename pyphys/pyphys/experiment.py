# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:20:51 2014

@author: crousse
"""
from pyphys.pyphys import PxpParser
from pyphys.cell import Cell


class Experiment(object):
    
    def __init__(self, path):
        self.data = PxpParser(path).data
        self.cells = [Cell(self.data, cellId) for cellId in self.cellIds]
        self.__attributize()
        
    def __attributize(self):
        for cell in self.cells:
            setattr(self, 'cell{}'.format(cell.id),  cell)

    @property
    def nCells(self):
        return len(self.cellIds)
        
    @property
    def cellIds(self):
        metadataDicts = ['Stims', 'vars', 'Packages', 'Logs', 'nm_folder0']
        recordingsKeys = [ddict for ddict in self.data if ddict not in metadataDicts]
        cellIds = [key.split('_')[0][-1] for key in recordingsKeys]
        cellIds = sorted(list(set(cellIds)))
        return cellIds
