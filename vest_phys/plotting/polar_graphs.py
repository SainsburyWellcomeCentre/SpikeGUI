import warnings
from math import cos, acos, sin, sqrt, radians

import numpy as np
from matplotlib import pyplot as plt

from src.signal_processing.mat_utils import avg, sd

warnings.filterwarnings('error')

class PolarPlot(object):
    
    def __init__(self, data, xData, doOffset=False, dest=None, ext='png', lines=True):
        """
        Initialises a polar plot.
        Expects binned data (shifts by half bin)
        returns the axes of the graph to append traces.
        """
        self.data = data[:]
        if doOffset:
            self.offsetData = self.data - self.data.min()
        self.doOffset = doOffset
        self.xData = xData[:]
        
        self.dest = dest
        self.lines = lines
        self.ext = ext
        
        fig = plt.figure(figsize=(16,16))
        self.ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True) # graph 80% of area start at 10%
        self.ax.set_rmin(3)
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1) # clockwise
#        angles = self.xData[:]
#        delta = angles[1] - angles[0]
#        plotAngles = np.radians(angles + (delta/2.0)) # Add half to shift bin center
#        self.ax.set_thetagrids(angles, labels=angles)
            
    def plot(self, ext=''):
        """
        Plot in gray all vectors from mat.
        Add average in red.
        """
        try:
            self.plotVectors()
            self.plotPolar(avg(self.data), 'red')
        except (UserWarning, RuntimeWarning):
            print('problem with your plot')
        angle, ampl = self.avgVectToAngleAndAmpl()
        self.ax.step((angle, angle), (ampl, 0), color='green', linewidth=2, where='mid') # hack to force to draw a line
        if self.dest:
            if ext:
                plt.savefig('{}.{}'.format(self.dest, ext)); plt.close()
            elif self.ext:
                plt.savefig('{}.{}'.format(self.dest, self.ext)); plt.close()
            else:
                plt.show()
        else:
            plt.show()
            
    def plotPolar(self, data, color):
        plotData = data[:]
        angles = self.xData[:]
        delta = angles[1] - angles[0]
        if (angles[-1] + delta - 360) == angles[0]: # close plot
            angles = np.append(angles, angles[0])
            plotData = np.append(plotData, plotData[0])
        plotAngles = np.radians(angles + (delta/2.0)) # Add half to shift bin center
        if self.lines:
            self.ax.step(plotAngles, plotData, color=color, linewidth=2, where='mid')
        else:
            self.ax.step(plotAngles, plotData, color=color, linestyle='None', marker='o', markersize=2, where='mid')

    def plotVectors(self, color='grey'):
        if self.doOffset:
            mat = self.offsetData
        else:
            mat = self.data
        for i in range(mat.shape[1]):
            if mat.ndim == 3:
                for j in range(mat.shape[2]):
                    self.plotPolar(mat[:,i,j], color)
            else:
                self.plotPolar(mat[:,i], color)
            
    def avgVectToAngleAndAmpl(self):
        pt0, pt1 = self.getAvgVect()
        x, y = pt1
        ampl = sqrt(x**2 + y**2)
        if ampl == 0:
            return (0,0)
        angle = acos(x/ampl)
        angle = angle if (y >0) else -angle
        return (angle, ampl)
            
    def getAvgVect(self):
        coords = avgDir(self.xData, avg(self.data))
        avgVect = np.array([[0, 0], coords])
        return avgVect    
                
class ScatterPlot(object):

    def __init__(self, data, xData, dest=None, ext='png', lines=True):
        self.data = data[:]
        self.xData = xData[:]
        self.dest = dest
        self.ext = ext
        fig = plt.figure()
        
    def plotRows(self):
        for i in range(self.data.shape[1]):
            if self.data.ndim == 3:
                for j in range(self.data.shape[2]):
                    vect = self.data[:,i,j]
                    plt.step(self.xData, vect, color='grey', where='mid')
            else:
                plt.step(self.xData, self.data[:,i], color='grey', where='mid')

    def plotAvg(self):
        avgData = avg(self.data)
        sdData = sd(self.data)
        plotXData = self.xData.copy()
        plotXData += (plotXData[1]-plotXData[0])/2.0
        plt.step(plotXData, avgData, color='red', linewidth=2, where='mid')
        plt.errorbar(plotXData, avgData, color='red', yerr=sdData, linestyle='')
#        plt.step(self.xData, avgData-sdData, color='pink', linewidth=2)
#        plt.step(self.xData, avgData+sdData, color='pink', linewidth=2)

    def plot(self, ext=''):
#        self.plotRows()
        self.plotAvg()
        if self.dest is not None:
            if ext: plt.savefig('{}.{}'.format(self.dest, ext)); plt.close()
            elif self.ext: plt.savefig('{}.{}'.format(self.dest, self.ext)); plt.close()
            else: raise AttributeError("Missing extension")
        else: plt.show()

def vect(angle, amp):
    """
    Get the vector from 0 from angle and amplitude.
    """
    x = cos(angle) * amp
    y = sin(angle) * amp
    return (x, y)
    
def avgDir(angles, ampls): # FIXME: rename
    rads = [radians(a) for a in angles]
    coords = np.array([vect(angle, ampl) for (angle, ampl) in zip(rads, ampls)])
    avgVectCoords = coords.mean(axis=0)
    return avgVectCoords
