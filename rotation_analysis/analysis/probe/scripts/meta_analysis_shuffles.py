import os

import numpy as np

from analysis.probe.probe_cell import ProbeCell
from analysis.probe.ctxt_manager_decorator import temp_setattr
from spike_handling import spike_io
from analysis.probe.trigger_handling import trigger_handler
import pandas as pd

def main():
    src_dir = '/home/slenzi/Desktop/CA242_B/'
    traces_path = '/home/slenzi/Desktop/sepi_subset_probe_data/CA242_B_g0_t0_shortened_sample_22000000_to_168000000.imec.ap.bin'
    trigger_path = '/home/slenzi/Desktop/CA242_B/full_clean_trigger.npy'

    stimuli_path = '/home/slenzi/Desktop/sepi_calcium_data/Sepi_stimuli_probe/Stimuli_igor/CA_242_B_randwaves_clean'
    timestamps_path = '/home/slenzi/Desktop/sepi_calcium_data/Sepi_stimuli_probe/Bonsai_timestamps/CA_242_B_copy.csv'

    cells = []
    sp = spike_io.SpikeIo(src_dir, traces_path, 385)
    exp_id = 'CA242_B'
    conditions = ['CA_242_B_90_black', 'CA_242_B_90_uniform', 'CA_242_B_90_landmarkleft']

    failed_trials_242_B = None
    failed_trials_probe = [64, 65, 66, 67, 68, 69, 70]
    START_SAMPLE = 22000000

    triggers_from_probe = get_triggers_from_probe(trigger_path, START_SAMPLE)

    real_triggers = get_real_triggers(failed_trials_242_B, failed_trials_probe, stimuli_path, timestamps_path, triggers_from_probe)

    for condition in conditions:
        all_cell_dfs = []
        for c in get_cluster(exp_id, real_triggers, sp, src_dir):

            print('condition_{}_cluster_{}'.format(condition, c.id))

            with temp_setattr(c.block, 'current_condition', {'keep': True, 'condition': condition}):

                for ref_var in ['position', 'distance', 'velocity', 'acceleration']:
                    print('computing shuffles for {}'.format(ref_var))
                    c.block.get_freqs_from_timetable(ref_var)

            c.block.shuffles_results['cid'] = pd.Series(c.id, index=c.block.shuffles_results.index)
            all_cell_dfs.append(c.block.shuffles_results)

        df_all_cell_dfs = pd.concat(all_cell_dfs)
        save_path = os.path.join(src_dir, '{}_shuffle_stats.csv'.format(condition))

        df_all_cell_dfs.to_csv(save_path)


def get_cluster(exp_id, real_triggers, sp, src_dir):
    for i, cid in enumerate(sp.good_clusters):  # 2243 # 4177
        c = ProbeCell(src_dir=src_dir, exp_id=exp_id, depth=0, cell_idx=cid, recordings=real_triggers,
                      extension='eps', use_bsl_2=False, spike_io=sp)
        yield c


def get_real_triggers(failed_trials_242_B, failed_trials_probe, stimuli_path, timestamps_path, triggers_from_probe):
    th = trigger_handler.TriggerIo(stimuli_path, timestamps_path, failed_trials_242_B)
    triggers_from_igor = th.igor_triggers()
    real_triggers = []
    for i, trigger_loc in enumerate(triggers_from_probe):
        if i not in failed_trials_probe:
            real_trigger = triggers_from_igor.pop(0)
            real_trigger.loc = trigger_loc
            real_triggers.append(real_trigger)
    real_triggers = trigger_handler.TriggerGroup(real_triggers)
    return real_triggers


def get_triggers_from_probe(full_probe_trigger_path, start_sample):
    full_probe_trigger = np.load(full_probe_trigger_path)
    triggers_242 = np.where(np.diff(full_probe_trigger) > 0)[
                       0] - start_sample  # FIXME: because discarded data means offset
    triggers_from_probe = triggers_242
    return triggers_from_probe


if __name__ == '__main__':
    main()
