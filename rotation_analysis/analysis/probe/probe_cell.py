import warnings

from rotation_analysis.analysis.cell import Cell
from rotation_analysis.analysis.probe.probe_block import ProbeBlock

from probez.spike_handling import cluster


class ProbeCell(Cell, cluster.Cluster):  #TODO: dirty multi-inheritance, probably not a good long term strategy
    CLUSTER_TYPES = ['Good', 'Multi_unit']

    def __init__(self, exp_id, src_dir, depth, cell_idx, extension, use_bsl_2,
                 spike_io, bonsai_io, igor_io, trigger_trace_io):

        self.spike_struct = spike_io

        self.analysed_metrics = ('frequency',)
        self.dir = src_dir
        self.ext = extension

        self.id = cell_idx
        self.exp_id = exp_id
        self.depth = self.spike_struct.get_cluster_channel_from_avg_waveforms(self.id)

        self.cell_type = 'Pyramid'  # TODO: config
        self.skip = False  # FIXME: use

        self.angles = bonsai_io.ordered_conditions  # TODO: FIXME
        self.block = ProbeBlock(self, use_bsl_2, self.spike_struct, bonsai_io, igor_io, trigger_trace_io)

        self.main_dir = None


def get_all_clusters(exp_id, sp, src_dir, bonsai_io, igor_io, trigger_trace_io, cluster_ids=None):
    warnings.warn('This can become very memory intensive for large numbers of cells with many trials. Use the iterator '
                  'version instead')

    clusters = []
    for i, cid in enumerate(cluster_ids):
        c = ProbeCell(src_dir=src_dir, exp_id=exp_id, depth=0, cell_idx=cid, extension='eps', use_bsl_2=False,
                      spike_io=sp, bonsai_io=bonsai_io, igor_io=igor_io, trigger_trace_io=trigger_trace_io)
        clusters.append(c)

    return clusters


def get_cluster(exp_id, sp, src_dir, bonsai_io, igor_io, trigger_trace_io, cluster_ids=None):

    if cluster_ids is None:
        cluster_ids = sp.good_cluster_ids

    for cid in cluster_ids:
        c = ProbeCell(src_dir=src_dir, exp_id=exp_id, depth=0, cell_idx=cid, extension='eps', use_bsl_2=False,
                      spike_io=sp, bonsai_io=bonsai_io, igor_io=igor_io, trigger_trace_io=trigger_trace_io)
        yield c
