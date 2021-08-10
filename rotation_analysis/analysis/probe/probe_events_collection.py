import operator
import warnings

from bisect import bisect_left
from rotation_analysis.analysis.event_detection.events_collection import EventsCollection
import numpy as np
import pandas as pd
from rotation_analysis.analysis.probe.config import FS_probe


class ProbeEventsCollection(EventsCollection):
    def __init__(self, events):
        self.events = events

    def __len__(self):
        return len(self.events)

    def __getitem__(self, item):
        return self.events[item]

    def __sub__(self, scalar):
        return ProbeEventsCollection([e - scalar for e in self])

    def to_df(self, trial_id=0):
        return pd.DataFrame({
            'trial_id': [trial_id] * len(self),
            'start_t': self.start_times,
            # 'rise_t': self.rise_times,
            'peak_t': self.peak_times,
            # 'amplitude': self.amplitudes,
        })

    def extend(self, events_collection):
        if list(events_collection.events):
            self.events.extend(events_collection.events)

    @staticmethod
    def from_concatenation_of_events_collections(events_collections):
        events = []
        for event_collection in events_collections:
            if event_collection is not None:
                events.extend(event_collection.events)
        return ProbeEventsCollection(events)

    def get_sorted(self, comparator=None):
        if comparator is None:
            return sorted(self)
        else:
            return sorted(self, key=operator.attrgetter(comparator))

    def get_spike_times_in_interval(self, start_p, end_p, spike_times):
        """

        :param start_p: start point (n_samples)
        :param end_p: end point (n_samples)
        :param spike_times: the set of spikes from which to get subset from
        :return spike_times: all spike times within a user specified interval
        """
        start_idx = bisect_left(spike_times, start_p)
        end_idx = bisect_left(spike_times, end_p)
        return spike_times[start_idx:end_idx]

    def in_point_range(self, range_start, range_end):
        events_in_point_range = list(self.get_spike_times_in_interval(range_start, range_end, self.events))
        return ProbeEventsCollection(events_in_point_range)

    def in_time_range(self, start_t, end_t):
        start_p, end_p = self.convert_times_to_points(start_t, end_t)
        events_in_point_range = list(self.get_spike_times_in_interval(start_p, end_p, self.events))
        return ProbeEventsCollection(events_in_point_range)

    @staticmethod
    def convert_times_to_points(start_t, end_t):
        warnings.warn('SAMPLING FREQUENCY IS HARD CODED')  # FIXME:
        start_p = start_t * FS_probe
        end_p = end_t * FS_probe
        return start_p, end_p

    def in_unordered_point_range(self, p1, p2):
        start = min(p1, p2)
        end = max(p1, p2)
        events_in_point_range = list(self.get_spike_times_in_interval(start, end, self.events))
        return ProbeEventsCollection(events_in_point_range)

    def in_unordered_time_range(self, t1, t2):
        start_t = min(t1, t2)
        end_t = max(t1, t2)
        start_p, end_p = self.convert_times_to_points(start_t, end_t)
        events_in_point_range = list(self.get_spike_times_in_interval(start_p, end_p, self.events))
        return ProbeEventsCollection(events_in_point_range)

    @property
    def peak_positions(self):
        return np.array(self.events, dtype=np.float64)

    @property
    def peak_times(self):
        return np.array(self.events, dtype=np.float64)

    @property
    def start_times(self):
        return np.array(self.events, dtype=np.float64)

    def get_events_point_params(self):
        peaks_pos = self.events
        return peaks_pos
