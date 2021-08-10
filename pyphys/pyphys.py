# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:58:27 2014

@author: crousse
"""
from rpy2.robjects.vectors import Matrix, Array, DataFrame, FloatVector, IntVector, StrVector, ListVector
import numpy
from collections import OrderedDict

from rpy2.robjects.packages import importr
igorr = importr('IgorR')


class PxpParser(object):
    """
    Parse an Igor pxp (Neuromatic) into python dictionnaries, lists and arrays
    """
    def __init__(self, pxpPath):
        self.data = self.parsePxP(pxpPath)
        
    def parsePxP(self, path):
        """
        Return a python structure from the pxp path
        """
        exp = igorr.read_pxp(path)
        return self.recursiveRToPy(exp)
        
    def recursiveRToPy(self, data):
        """
        The recursive function to convert from rpy2 objects to native python
        """
        rDictTypes = (DataFrame, ListVector)
        rArrayTypes = (FloatVector, IntVector, Array, Matrix)
        rListTypes = tuple([StrVector])

        if type(data) in rDictTypes:
            return OrderedDict(zip(data.names, [self.recursiveRToPy(elt) for elt in data]))
        elif type(data) in rListTypes:
            return [self.recursiveRToPy(elt) for elt in data]
        elif type(data) in rArrayTypes:
            return numpy.array(data)
        else:
            if hasattr(data, "rclass"): # An unknown r class
                raise KeyError('Could not proceed, type {} is not defined'.format(type(data)))
            else:
                return data # We reached the end of recursion
