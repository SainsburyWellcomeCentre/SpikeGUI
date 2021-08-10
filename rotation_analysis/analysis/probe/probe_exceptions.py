class NoKeptTrialsError(Exception):
    def __init__(self):
        print('there are no trials to analyse, please check your condition entries and try again')