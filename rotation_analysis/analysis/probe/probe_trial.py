from rotation_analysis.analysis.trial import Trial
from rotation_analysis.analysis.probe.probe_events_collection import ProbeEventsCollection
from rotation_analysis.analysis.probe.probe_stimulus import ProbeRotationStimulus, StaticStimulus


class ProbeTrial(Trial):
    def __init__(self, block, idx, use_bsl_2, spike_struct, bonsai_io, igor_io, trigger_trace_io):
        self.block = block
        self.idx = idx

        self.bonsai_io = bonsai_io
        self.igor_io = igor_io
        self.trigger_trace_io = trigger_trace_io
        self.waveform = igor_io.get_waveform(self.idx)

        loc = trigger_trace_io.get_corrected_trigger(self.idx)

        self.loc = loc
        self.stimulus = self.get_stimulus(self.loc, use_bsl_2, bonsai_io, igor_io)
        self.keep = True

        self.start = self.loc - self.stimulus.n_samples_before_trigger
        self.end = self.loc + len(self.stimulus.cmd)
        self.events = self.get_events(spike_struct)
        self.angle = bonsai_io.get_condition(self.idx)

    def matches_attributes(self, matching_dictionary):
        """
        For each attribute in attributes_dict, checks whether the trial has the attribute
        if so, it will return False if the attribute does not match (any attribute not matching means False)
        otherwise, it will check if self.stimulus has the attribute.
        If not: returns False, of yes returns False if does not match. (any attribute not matching means False)

        :param matching_dictionary:
        :return:
        """
        print('checking probe trial attributes')
        for k, v in matching_dictionary.items():
            if not hasattr(self, k):
                if not self.stimulus.matches_attributes({k: v}):
                    return False
            elif getattr(self, k) != v:
                return False
        return True

    @property
    def condition(self):
        all_condition_params = self.bonsai_io.get_datapoint(self.idx)
        all_condition_params_list = []
        for key, value in all_condition_params.items():
            if key == 'Timestamp':
                continue
            if 'nan' in str(value):
                continue

            all_condition_params_list.append(str(value))
        return '_'.join(all_condition_params_list)

    def get_events(self, spike_struct):
        spike_times = spike_struct.cluster_spike_times_in_interval(self.block.cell.id, self.start, self.end)
        events = ProbeEventsCollection((spike_times - self.start))
        # if self.flip:  # FIXME:
        #     events -= len(self.stimulus.cmd)

        return events

    def get_stimulus(self, loc, use_bsl_2, bonsai_io, igor_io):

        waveform = igor_io.get_waveform(self.idx)

        if (self.waveform == 0).all():
            return StaticStimulus(self.idx, loc, waveform, bonsai_io, igor_io)

        return ProbeRotationStimulus(self.idx, loc, waveform, bonsai_io, igor_io, use_bsl_2)

    @property
    def flip(self):

        if (self.waveform == 0).all():
            return False

        peaks = ProbeRotationStimulus.get_sine_peaks(self.waveform)

        return self.waveform[peaks[0]] > 0

    def bonsai_data(self):
        return self.bonsai_io.get_datapoint(self.idx)

    # @property
    # def waveform(self):
    #     return self.rec.igorIO.get_waveform(self.idx)

    def set_traces(self, rec):
        pass

    def reset_detection(self):
        pass

    def detect(self, params, processed):
        pass

    def _detect_events(self, trace, params):
        pass
