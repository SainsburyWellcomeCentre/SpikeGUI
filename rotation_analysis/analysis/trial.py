import numpy as np
from margrie_libs.margrie_libs.signal_processing.measures import get_sd

from rotation_analysis.analysis.event_detection.event_detection import detect_trace
from rotation_analysis.analysis.event_detection.events import Event
from rotation_analysis.analysis.event_detection.events_collection import EventsCollection
from rotation_analysis.analysis.stimulus import CalciumRotationStimulus


class Trial(object):
    def __init__(self, block, idx, rec, condition, use_bsl_2):
        self.noises = None
        self.processed_trace = None
        self.raw_trace = None
        self.events = EventsCollection([])
        self.block = block
        self.idx = idx
        self.stimulus = self.get_stimulus(rec, condition, use_bsl_2)
        self.set_traces(rec)
        self.keep = True

    @staticmethod
    def get_stimulus(rec, condition, use_bsl_2):
        return CalciumRotationStimulus(rec, condition, use_bsl_2=use_bsl_2)

    def matches_attributes(self, matching_dictionary):
        """
        For each attribute in attributes_dict, checks whether the trial has the attribute
        if so, it will return False if the attribute does not match (any attribute not matching means False)
        otherwise, it will check if self.stimulus has the attribute.

        If not: returns False, of yes returns False if does not match. (any attribute not matching means False)

        :param matching_dictionary:
        :return:
        """
        for k, v in matching_dictionary.items():
            if not hasattr(self, k):
                if not self.stimulus.matches_attributes({k: v}):
                    return False
            elif getattr(self, k) != v:
                return False
        return True

    def get_weighted_amplitude(self, period_name, constraining_period_name=None):
        ranges = self.stimulus.get_ranges_by_type(period_name, constraining_period_name)
        duration = self.stimulus.get_ranges_duration(ranges)
        print("Period {}, ranges: {}".format(period_name, ranges))
        events = self.get_events_in_period(period_name, constraining_period_name)
        weighted_amplitude = events.total_amplitude() / duration
        weighted_amplitude = weighted_amplitude if not np.isnan(weighted_amplitude) else 0
        return weighted_amplitude

    def get_frequency(self, period_name, constraining_period_name=None):
        events_in_period = self.get_events_in_period(period_name, constraining_period_name)
        ranges = self.stimulus.get_ranges_by_type(period_name, constraining_period_name)
        duration = self.stimulus.get_ranges_duration(ranges)
        return float(len(events_in_period)) / duration

    def extract_period(self, period_name, constraining_period=None):
        extracted_period = np.zeros(0, dtype=np.float64)
        for rng in self.stimulus.get_ranges_by_type(period_name, constraining_period):
            start, end = rng
            extracted_period = np.hstack((extracted_period, self.processed_trace[start:end]))
        return extracted_period

    def get_events_in_period(self, period_name, constraining_period_name, relative=False):
        """

        :param str period_name:
        :param str constraining_period_name: determines the duration of the period specified by period_name
        :return:
        """

        ranges = self.stimulus.get_ranges_by_type(period_name, constraining_period_name)

        trial_events_in_period = EventsCollection([])
        for j, rng in enumerate(ranges):
            start, end = rng
            events_in_range = self.events.in_point_range(start, end)
            if events_in_range and j > 0:
                end_of_previous_range = ranges[j - 1][1]
                shift = start - end_of_previous_range
                shifted_events = events_in_range - shift  # Shift to the end of the previous range
                if not shifted_events:
                    print("Attempted shift by {} failed".format(shift))
                    raise ValueError(
                        "Should still be events for cell {}, period {}, trial {}, part {}".
                        format(self.block, period_name, self.idx, j))
                events_in_range = shifted_events
            trial_events_in_period.extend(events_in_range)
            if relative:
                trial_events_in_period -= start

            return trial_events_in_period

    def set_traces(self, rec):
        rec.getProfiles()
        self.raw_trace = rec.get_cell_data(self.block.cell.id, 'raw')
        self.processed_trace = rec.get_cell_data(self.block.cell.id, 'delta_f')

    def reset_detection(self):
        self.events = EventsCollection([])
        self.noises = []

    def detect(self, params, processed):
        """

        :param bool processed: Whether to work on the raw or processed trace
        :return:
        """
        trial = self.processed_trace if processed else self.raw_trace
        events_params, noises = self._detect_events(trial, params)
        self.events = (EventsCollection(  # WARNING: should store in different one for raw or processed
            [Event(*params, self.stimulus.sampling_interval) for params in zip(*events_params)]
        ))
        self.noises = noises  # TEST: check that correct

    def _detect_events(self, trace, params):  # WARNING events are detected in pnts
        """

        :param trace:
        :param params:
        :return:
        """
        default_result = ([],)*4
        try:
            results = detect_trace(trace,
                                   params.threshold,
                                   params.n_pnts_bsl, params.n_pnts_peak, params.n_pnts_rise_t,
                                   params.n_pnts_for_peak_detection,
                                   params.n_sds,
                                   params.n_pnts_high_pass_filter, params.median_kernel_size)

            _, _, events_pos, peaks_pos, half_rises, peak_ampls = results
            events_params = (events_pos, peaks_pos, half_rises, peak_ampls)
        except StopIteration:
            events_params = default_result
        if [e is False for e in events_params].count(True) == 4:
            events_params = default_result
        noise = get_sd(trace, params.n_pnts_high_pass_filter)
        return events_params, noise
