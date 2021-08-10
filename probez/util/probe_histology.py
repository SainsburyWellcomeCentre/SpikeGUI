import numpy as np
import operator


def layer_boundaries_relative_to_channel_0_dict(layer_boundaries_dict,
                                                probe_insert_distance,
                                                tip_to_0th_channel_distance):
    """

    :param dict layer_boundaries_dict: the distance from the deepest point of a track to each layer
    :param int probe_insert_distance: the distance the probe was inserted (based on experiment, not histology)
    :param tip_to_0th_channel_distance: the distance from the tip of the probe to the first recording site
    :return:
    """
    track_length = max(layer_boundaries_dict.values()) - min(layer_boundaries_dict.values())
    scale_factor = probe_insert_distance/track_length

    for item in layer_boundaries_dict.items():
        layer_boundaries_dict[item[0]] *= scale_factor
        layer_boundaries_dict[item[0]] -= tip_to_0th_channel_distance

    return layer_boundaries_dict


def get_clusters_in_boundaries(lower_bound, upper_bound, cluster_ids, cluster_depths):

    cluster_id_array = np.array(cluster_ids)
    depth_array = np.array(cluster_depths)
    idx = np.logical_and(depth_array > lower_bound/10, depth_array < upper_bound/10)  # depth is 10x channel number

    clusters_in_bounds = cluster_id_array[idx]
    cluster_depths_in_bounds = depth_array[idx]

    return clusters_in_bounds, cluster_depths_in_bounds


def get_layer_from_depth(depth, layer_boundary_dict):
    sorted_boundaries = sorted(layer_boundary_dict.items(), key=operator.itemgetter(1))
    for lower, upper in zip(sorted_boundaries[:-1], sorted_boundaries[1:]):
        if lower[1] < depth < upper[1]:
            return upper[0]
    return 'no layer'
