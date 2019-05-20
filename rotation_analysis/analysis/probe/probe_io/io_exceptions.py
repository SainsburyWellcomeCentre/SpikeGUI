class ProbeIoError(Exception):
    pass


class ProbeIoKeyError(ProbeIoError):
    def __init__(self, *args):
        print('The key: {} does not seem to exist in the {}'.format(*args))
    pass


class IgorFileKeyError(ProbeIoKeyError):
    def __init__(self, str):
        super().__init__(str, 'igor data structure')


class BonsaiQueryError(ProbeIoKeyError):
    def __init__(self, str):
        super().__init__(str, 'bonsai data frame')


class PhotodiodeOnNotFoundError(ProbeIoError):
    pass