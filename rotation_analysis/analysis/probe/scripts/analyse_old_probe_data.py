from margrie_libs.signal_processing.list_utils import flatten as flatten_list
from spike_handling import spike_io

from analysis.probe import probe_cell
from analysis import field_of_view
from analysis.probe.trigger_handling.triggers_old_data import Triggers
from analysis.probe.ctxt_manager_decorator import temp_setattr;


def main():

    # ps = ProbeRotationStimulus('/home/slenzi/spine_shares/buffer/sepi_rotation/CA23_3/', 1/30000)
    # ps.get_counter_clock_wise_ranges()

    src_dir = '/home/slenzi/spine_shares/buffer/sepi_rotation/CA23_3/'
    traces_path = '/home/slenzi/spine_shares/buffer/sepi_rotation/CA23_3/dataMerged.bin'

    cmd_path = src_dir + 'cmd/'
    trigger_path = cmd_path + 'trigger_indices.mat'
    conditions_path = cmd_path + 'condition_indices.txt'

    def load_trigger_list(path):
        import scipy.io
        trigger_list = []
        for inds in scipy.io.loadmat(path)['indsTrig'][0]:
            trigger_list.extend(inds)
        return flatten_list(trigger_list)

    trigger_list = load_trigger_list(trigger_path)
    trigger_subset = trigger_list[12:22]
    flip_idx = [False for t in trigger_subset]

    # for i in range(0,10,2):
    #     flip_idx[i] = False

    recordings = Triggers(src_dir, trigger_subset, ['0']*len(trigger_subset), flip_idx=flip_idx)

    cells = []
    sp = spike_io.SpikeIo(src_dir, traces_path, 385)

    for i, cid in enumerate([2243]):  #2243 # 4177
        c = probe_cell.ProbeCell(src_dir=src_dir, exp_id='CA23_3', depth=0, cell_idx=cid, recordings=recordings,
                                 extension='eps', use_bsl_2=False, spike_io=sp)
        cells.append(c)

    spiking_tables = []
    all_events = []
    all_backwards_events = []

    stimulus = c.block.trials[0].stimulus

    with temp_setattr(c.block, 'current_condition', {'keep': True, 'angle': '180'}):
        ec = c.block.kept_events_collections
        spiking_table, levels, cmd = c.block.get_freqs_from_timetable('acceleration')
    #     plt.imshow(spiking_table.T)
    #     for e, t in zip(ec, flip_idx):
    #         if t:
    #             all_events.append(e.events)
    #         else:
    #             all_backwards_events.append(e.events)
        # st_arr = np.array(spiking_table)
        # st_mean = np.mean(spiking_table, axis=0)
        # spiking_tables.append(spiking_table)
        # [plt.hist(e.events, histtype='step') for e in ec]

    fov = field_of_view.FieldOfView(cells, None, src_dir, 100)

    #fov.plot_spiking_heatmaps('180')
    fov.analyse(True, False, 'eps')


if __name__ == '__main__':
    main()
