import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns
from sorting_quality import load_quality_measures

group_dict_for_quality = {
    'noise': 0,
    'MUA': 1,
    'good': 2,
    'unsorted': 3

}


class Quality(object):
    def __init__(self, path):
        self.groups, self.contamination_rates, self.isi_violations, \
                                               self.unit_qualities = load_quality_measures.load_from_matlab(path)

    def get_cluster_quality(self, cluster_id, spike_clusters):

        cluster_ids_raw = np.unique(spike_clusters)  # because the order of clusters isn't known and we need idx
        cid = np.where(cluster_ids_raw == cluster_id)[0][0]
        group = self.groups[cid]
        contamination_rate = self.contamination_rates[cid][0]
        isi_violation = self.isi_violations[cid][0]
        unit_quality = self.unit_qualities[cid][0]

        return group, contamination_rate, isi_violation, unit_quality


def plot_cluster_quality_by_group(cluster_groups, sorting_quality_parameter):
    """
    visualise any sorting quality parameter according to the classification clusters of each group
    groups 0:

    :param cluster_groups:
    :param sorting_quality_parameter:
    :return:
    """

    fig = plt.figure()
    for group in np.unique(cluster_groups)[0:3]:
        in_this_group = cluster_groups == group
        these_unit_qualities = sorting_quality_parameter[in_this_group]
        group_mean = np.mean(these_unit_qualities)
        plt.scatter(np.ones_like(these_unit_qualities)*group, these_unit_qualities, color='k', alpha=0.5)
        plt.scatter(group, group_mean, s=50)
    return fig


def plot_cluster_quality(cluster_id, spike_clusters, isi_violations, contamination_rates, unit_qualities, color,
                         zorder, s, edge=False):
    """

    :param cluster_id:
    :param spike_clusters:
    :param isi_violations:
    :param contamination_rates:
    :param unit_qualities:
    :param color:
    :param zorder:
    :param s:
    :param edge:
    :return:
    """
    cluster_ids_raw = np.unique(spike_clusters)
    cid = np.where(cluster_ids_raw == cluster_id)
    contamination_rate = contamination_rates[cid]
    isi_violation = isi_violations[cid]
    unit_qualities = unit_qualities[cid]

    cmap, norm = get_colormaps(color, 0, 100)
    if edge:
        plt.scatter(contamination_rate, isi_violation,
                    cmap=cmap, norm=norm, c=unit_qualities, zorder=zorder, s=s, edgecolor='k')
    else:
        plt.scatter(contamination_rate, isi_violation,
                    cmap=cmap, norm=norm, c=unit_qualities, zorder=zorder, s=s)
    plt.xlabel('contamination rate')
    plt.ylabel('isi violations')


def quality_box_plot(df):
    y_params = ['unit_quality', 'contamination_rate', 'isi_violations']
    fig = plt.figure()

    for i, y_param in enumerate(y_params):
        fig.add_subplot(2, 2, i+1)
        sns.set(style="ticks", palette="deep", color_codes=True)

        sns.boxplot(x="group", y=y_param, data=df,
                    whis=np.inf, color="c")

        # Add in points to show each observation
        sns.stripplot(x="group", y=y_param, data=df,
                      jitter=True, size=3, color=".3", linewidth=0)


def plot_all_quality():
    """plot all quality visualisations in a single figure"""
    pass


def filter_by_thresholds(df, isi_threshold, contamination_rate_threshold, unit_quality_threshold):

    cr_thresh_df = df[df['contamination_rate'] < contamination_rate_threshold]
    cr_isi_thresh_df = cr_thresh_df[cr_thresh_df['isi_violation'] < isi_threshold]
    all_thresh_df = cr_isi_thresh_df[cr_isi_thresh_df['unit_quality'] > unit_quality_threshold]

    return all_thresh_df


def plot_quality_df(df):

    plt.subplot(1, 3, 1)
    sns.regplot(df['isi_violation'], df['unit_quality'], fit_reg=False)

    plt.subplot(1, 3, 2)
    sns.regplot(df['isi_violation'], df['contamination_rate'], fit_reg=False)

    plt.subplot(1, 3, 3)
    sns.regplot(df['contamination_rate'], df['unit_quality'], fit_reg=False)


def filter_by_layer(df, layer):
    return df[df['layer'] == layer]


def filter_df_by(df, label, value):
    return df[df[label] == value]


def get_proportion_active(df, layer):
    """

    :param df:
    :param layer:
    :return:
    """

    # TODO: make filter work with different conditions

    layer_df = filter_df_by(df, 'layer', layer)
    rwvs_df = filter_df_by(layer_df, 'exp_condition', 'rotate_w_vis_stim')
    dark_df = filter_df_by(layer_df, 'exp_condition', 'dark')

    # make a data frame of only those that are modulated in visual condition
    rwvs_mod_df = rwvs_df[rwvs_df['p_value'] < 0.05]

    # get the (unique) cluster ids for these
    cids_vismod = np.unique(rwvs_mod_df['cluster_id'])
    n_vismod_cids = len(cids_vismod)

    # make a df of all dark trials from visually modulated data frame
    dark_vismod_df = dark_df[dark_df['cluster_id'].isin(cids_vismod)]

    # make a data frame of only those that are modulated in dark condition
    darkmod_vismod_df = dark_vismod_df[dark_vismod_df['p_value'] < 0.05]

    # get the (unique) cluster ids for these
    cids_darkmod_vismod = np.unique(darkmod_vismod_df['cluster_id'])
    n_darkmod_vismod_cids = len(cids_darkmod_vismod)

    proportion = n_darkmod_vismod_cids/n_vismod_cids if n_vismod_cids > 0 else np.nan

    print('{} dark modulated clusters, of the {} visually modulated clusters in layer {}. {} %'.format(n_darkmod_vismod_cids, n_vismod_cids, layer, proportion))

    return n_darkmod_vismod_cids, n_vismod_cids


def get_colormaps(palette, min_val, max_val):
    from matplotlib.colors import BoundaryNorm
    # define the colormap
    cmap = plt.get_cmap(palette)

    # extract all colors from the Reds map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize and forcing 0 to be part of the colorbar!
    bounds = np.arange(min_val, max_val, 1)
    idx = np.searchsorted(bounds, 0)
    bounds = np.insert(bounds, idx, 0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm