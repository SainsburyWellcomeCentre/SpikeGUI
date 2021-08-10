SORTING_QUALITY_GITHUB_REPO = 'https://github.com/cortex-lab/sortingQuality'


class SpikeStructLoadDataError(Exception):
    pass


class QualityNotLoadedError(Exception):
    def __str__(self):
        return 'to calculate quality measures, we recommend {}' \
               ' just add the quality.mat file to the working directory and retry.'.format(SORTING_QUALITY_GITHUB_REPO)


class SpikeTimeTypeError(Exception):
    pass


class NoTracesInSpikeStructError(SpikeStructLoadDataError):
    def __str__(self):
        return 'cannot find raw traces from which to find best waveform'


class IncorrectNchanTracesStructError(Exception):
    def __init__(self, n_datapoints, n_chan):
        super().__init__()
        self.n_dp = n_datapoints
        self.n_chan = n_chan

    def __str__(self):
        return 'n_chan incorrect, try again n_samples: {} divided ' \
               'by n_chan: {} not integer'.format(self.n_dp, self.n_chan)

