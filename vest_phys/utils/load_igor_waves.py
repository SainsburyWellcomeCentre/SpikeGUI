import numpy as np

def loadIgorCsvWaves(srcPath):
    mat = []
    with open(srcPath, 'r') as inFile:
        data = inFile.readlines()
        data = data[0].split('\r')
    header = ((data[0]).strip()).split(',')
    header = header[1::2]

    for i in range(1, (len(header)*2), 2):
        mat.append(loadCsv(srcPath, i)) # WARNING: Reloads file all the time.

    mat = np.array(mat, dtype=np.float64)
    return header, mat
    
def loadCsv(srcPath, columnIdx):
    """
    Loads igor wave from dataframe of wave.l wave.d stacks
    """
    with open(srcPath, 'r') as inFile:
        data = inFile.readlines()
    data = data[0].split('\r') # FIXME: MacOS specific
    data = data[1:] # skip header
    data = [d.strip() for d in data if d]
    data = [d.split(',') for d in data]
    data = [d[columnIdx] for d in data]
    return np.array(data, dtype=np.float64)    
    
def loadCsvBuiltin(srcPath):
    return np.loadtxt(srcPath)
