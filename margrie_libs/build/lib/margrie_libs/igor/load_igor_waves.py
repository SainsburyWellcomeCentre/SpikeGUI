import numpy as np


def load_igor_csv_waves(src_path):
    mat = []
    with open(src_path, 'r') as inFile:
        data = inFile.readlines()
        data = data[0].split('\r')
    header = ((data[0]).strip()).split(',')
    header = header[1::2]

    for i in range(1, (len(header)*2), 2):
        mat.append(load_csv(src_path, i))  # WARNING: Reloads file all the time.

    mat = np.array(mat, dtype=np.float64)
    return header, mat


def load_csv(src_path, column_idx):
    """
    Loads igor wave from dataframe of wave.l wave.d stacks
    """
    with open(src_path, 'r') as inFile:
        data = inFile.readlines()
    data = data[0].split('\r')  # FIXME: MacOS specific
    data = data[1:]  # skip header
    data = [d.strip() for d in data if d]
    data = [d.split(',') for d in data]
    data = [d[column_idx] for d in data]
    return np.array(data, dtype=np.float64)    


def load_csv_builtin(src_path):
    return np.loadtxt(src_path)
