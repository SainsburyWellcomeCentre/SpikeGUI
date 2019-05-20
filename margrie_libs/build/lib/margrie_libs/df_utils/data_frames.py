import numpy as np
import pandas as pd

"""because I can't be bothered to look up how to make data frames all the time"""


def create_data_frame(keys, data):

    df_keys = np.array([keys])

    df = pd.DataFrame(data=df_keys[1:, 1:],  # convert to empty dataframe
                      index=df_keys[1:, 0],
                      columns=df_keys[0, 1:])

    for key, item in zip(keys, data):
        df[key] = item

    return df


def add_to_table(df, data_entry):
    current_index = df.index.max()+1  # the index of the new line to be added
    if np.isnan(current_index):
        current_index = 0

    df.loc[current_index] = data_entry



#
#
# def create_data(cluster_ids, cluster_channels, stat_test_func=stats.wilcoxon()):
#     for cluster, depth in zip(cluster_ids, cluster_channels):
#         segment_dicts = sp.window_rasters_all_segments([1], sp.segments[0:8], relative=True)
#         interleaved_eventplot_raster = sp.chronological_trials(segment_dicts)
#
#         # for a given cluster, get the number of events in each trial
#
#         bsl_event_counts = sp.event_count(interleaved_eventplot_raster['baseline_trials'])
#         cw_event_counts = sp.event_count(interleaved_eventplot_raster['clockwise_trials'])
#         acw_event_counts = sp.event_count(interleaved_eventplot_raster['anticlockwise_trials'])
#
#         # divide by the duration of the section being considered to get firing rate in hz
#
#         bsl_hz = sp.avg_firing_rates_hz(bsl_event_counts, 2)
#         cw_hz = sp.avg_firing_rates_hz(cw_event_counts, 3.354)
#         acw_hz = sp.avg_firing_rates_hz(acw_event_counts, 3.354)
#
#         # calculate a p_value
#         cw_p = stat_test_func(bsl_hz, cw_hz)
#         acw_p = stat_test_func(bsl_hz, acw_hz)
#
#         cw_data = ['m161122a', int(cluster), depth, 'cw', cw_p[0], cw_p[1]]
#         acw_data = ['m161122a', int(cluster), depth, 'acw', acw_p[0], acw_p[1]]
#
#         return cw_data, acw_data
#
#
# def add_to_df(df, cw_data, acw_data):
#     """
#     >>> cw_data = ['m161122a', 'Layer 6', int(cluster), depth, 'cw', cw_p[0], cw_p[1]]
#     >>> acw_data = ['m161122a', 'Layer 6', int(cluster), depth, 'acw', acw_p[0], acw_p[1]]
#     :param df:
#     :param cw_data:
#     :param acw_data:
#     :return:
#     """
#
#     current_index = df.index.max()+1  # the index of the new line to be added
#     if np.isnan(current_index):
#         current_index = 0
#
#     df.loc[current_index] = cw_data
#     df.loc[current_index+1] = acw_data
#     df['cluster_id'] = df['cluster_id'].astype(int)
