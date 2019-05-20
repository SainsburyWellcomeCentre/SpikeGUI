import os
import sys
import pandas as pd
import numpy as np
from margrie_libs.stats import stats as margrie_stats
import collections

condition_pairs = (
  ('bsl_short', 'spin'),
  ('bsl2', 'spin'),
  ('bsl_short', 'bsl2'),
  ('bsl_short', 'c_wise'),
  ('bsl_short', 'c_c_wise'),
  ('c_wise', 'c_c_wise')
)

#
# paths = [  # FIXME: os.listdir()
#     '/home/slenzi/Desktop/exp_CA242_B_depth_137_cell_107_keep_True_angle_CA_242_B_90_uniform_trials.csv',
# ]


def main():
    do_cell_stats('/home/slenzi/Desktop/data/CA242_B_90_landmarkleft/')


def do_cell_stats(base_dir):
    fnames = os.listdir(base_dir)
    # base_dir = os.path.dirname(path)
    for j, fname in enumerate(fnames):

        #fname = os.path.splitext(os.path.basename(path))[0]
        path = os.path.join(base_dir, fname)
        c0_data = pd.read_csv(path)
        out_dict = collections.OrderedDict()

        cid = fname.split('_')[6]
        cdepth = fname.split('_')[4]
        out_dict['cid'] = cid
        out_dict['depth'] = cdepth

        for c in c0_data.keys()[1:]:
            out_dict[c] = np.nanmean(c0_data[c])

        for i, (c1, c2) in enumerate(condition_pairs):
            col1 = '{}_{}'.format(c1, 'frequency')
            col2 = '{}_{}'.format(c2, 'frequency')
            comparison_name = '{}_vs_{}'.format(col1, col2)
            # print(comparison_name)
            p_val = margrie_stats.wilcoxon(c0_data[col1], c0_data[col2])
            # print(p_val)

            out_dict[comparison_name] = p_val

        df = pd.DataFrame(out_dict, index=[0])
        if j == 0:
            new_df = df
        else:
            new_df = pd.concat([new_df, df])

    out_file_path = os.path.join(base_dir, 'all_stats.csv')
    new_df.to_csv(out_file_path)


if __name__ == '__main__':
    main()