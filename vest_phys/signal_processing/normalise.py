import numpy as np


def normalise(srcArray):
    """
    Normalise srcArray from 0 to 1
    """
    outArray = srcArray.copy()
    outArray -= outArray.min()
    outArray /= outArray.max()
    return outArray


def normaliseAroundZero(srcArray, bslEnd=15000):
    """
    Normalise trace by centering average(bsl) on 0 and to max == 1
    """
    outArray = srcArray.copy()
    
    bsl = np.mean(outArray[:bslEnd])
    outArray -= bsl
    outArray /= outArray.max()
    
    return outArray
