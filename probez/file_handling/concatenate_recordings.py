import numpy as np


def concatenate_files(fout_path, recordings):
    """join together all files in fout_path into one file"""

    all_nchans = set([rec.n_chan for rec in recordings])

    if len(all_nchans) != 1:
        raise InconsistentNChanError('expected equal channel numbers for all recordings, got {}'.format(all_nchans))

    fout = open(fout_path, 'wb')

    for rec in np.sort(recordings):
        fin = open(rec.path, 'rb')
        while True:
            data = fin.read(2 ** 16)
            if not data:
                break
            fout.write(data)
        fin.close()
    fout.close()


class InconsistentNChanError(Exception):
    pass
