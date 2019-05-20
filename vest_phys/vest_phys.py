
"""
Created on Wed May  6 11:44:18 2015

.. moduleauthor:: C.V. Rousseau c.rousseau@ucl.ac.uk

"""
import os
import sys

from matplotlib import pyplot as plt

from experiment import Experiment

def checkFigExtension(extension):
    """
    Check possible figure extensions
    """
    fig = plt.figure()
    formats = fig.canvas.get_supported_filetypes().keys()
    if not(extension in formats):
        raise ValueError("Unsupported extension {}. Supported formats are: {}".format(extension, formats))

if __name__ == "__main__":
    """
    Expects
    args[1]: (str) experiment (.pxp) path
    args[2]: (str) figures file extension
    args[3]: (bool) doSpikingDifference
    args[4]: (bool) doSpikingRatio
    """
    expPath = sys.argv[1]
    if not os.path.exists(expPath):
        raise ValueError("file does not exist")
    extension = sys.argv[2]
    plotOnly = sys.argv[5]
    
#    checkFigExtension(extension)

    os.chdir(os.path.dirname(expPath))
    exp = Experiment(expPath, ext=extension)
    if not plotOnly:
        exp.analyse(do_spiking_difference=bool(sys.argv[3]), do_spiking_ratio=bool(sys.argv[4]))
        exp.write()
    
#class vestPhysTests(object)
#    def testVect():
#    testValues = ((0,  2),  (12,  3),  (5,  1))
#    results = ((), (), ())
#    for valPair, resultPair in zip(testValues, results):
#        assert vect(valPair )== resultPair,  'Problem in testVect'

#    def _testBinVector(self, vect):
#        xData = self.xData
#        bins = self.bins
#        levels = np.digitize(xData, bins)
#        levelTypes = set(levels)
#        startLevelN = 0
#        for levelType in levelTypes:
#            if levelType ==0:
#                raise NotImplementedError
#            elif levelType ==1:
#                startLevelN = len(levels[levels==levelType])
#            if levelType>1:
#                assert  len(levels[levels==levelType]) == startLevelN, \
#                "number of points in bins missmatch at level {}".format(levelType)


