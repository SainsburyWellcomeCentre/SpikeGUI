
from util import probe_histology


histology_boundaries_170316a = {
    'surface_upper': 1602.3,
    'L2/3': 1425.5,
    'L4': 1253.8,
    'L5': 1101.2,
    'L6': 882.5,
    'WM': 573.8,
    'Sb': 427,
    'probe_tip': 0
}

# dark_stimuli = sp.stimuli[-7:]
# light_stimuli = sp.stimuli[1:7]


histology_boundaries_170315b = {
    'surface_upper': 1699.5,
    'L2/3': 1547.1,
    'L4': 1311.6,
    'L5': 1127.4,
    'L6': 912.7,
    'WM': 705.7,
    'Sb': 407.0,
    'surface_lower': 40.7,
    'probe_tip': 0
}

# dark_segments = sp.stimuli[0:8]
# vis_stim_ani_move_segments = sp.stimuli[8:16]



total_probe_insert_distance_170316a = 1750
tip_to_chan_0_distance_npix3A_opt1 = 137

boundaries = probe_histology.layer_boundaries_relative_to_channel_0_dict(histology_boundaries_170316a,
                                                                         total_probe_insert_distance_170316a,
                                                                         tip_to_chan_0_distance_npix3A_opt1)
