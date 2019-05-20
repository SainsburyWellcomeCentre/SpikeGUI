# module import
import numpy as np
import analysis_guis.common_func as cf

#
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_stim_phase_timing(rot_info, w_form, tt_io, igor_io, t_type='Black', p_type='Full'):
    '''

    :param w_form:
    :param ind:
    :return:
    '''

    #
    is_full = p_type == 'Full'
    t_type = rot_info['trial_type'][np.array([x != 'UniformDrifting' for x in rot_info['trial_type']])]
    # t_type = ['MotorDrifting']
    # t_type = ['Black']
    # t_type = ['Uniform']
    # t_type = ['LandmarkLeft']
    # t_type = ['LandmarkRight']

    #
    if is_full:
        fig, n_plot = plt.figure(figsize=(16, 8), dpi=100), len(t_type)
        ax = np.empty(n_plot, dtype=object)
        for i_plot in range(n_plot):
            ax[i_plot] = fig.add_subplot(n_plot, 1, i_plot+1)


    #
    for jtt, tt in enumerate(t_type):
        #
        print('Outputting Graph For: "{0}"'.format(tt))

        # retrives the trial type index array
        itt = list(rot_info['trial_type']).index(tt)
        idxtt = rot_info['ind_trial'][itt]

        #
        col = 'krgb'
        w_form_tt, wfp = w_form[itt], rot_info['wfm_para'][tt]
        t_stim = [tt_io.get_corrected_trigger(idx, verbose=False) for idx in idxtt]

        if p_type == 'Full':
            #
            Ttt = np.arange(t_stim[0], t_stim[-1] + len(w_form_tt[-1])) / 30000
            Ntt = len(Ttt)

            y_plt = np.empty((Ntt,4), dtype=object)
            y_plt[:, 0] = 0.0

            #
            for iwf, wf in enumerate(w_form_tt):
                # sets the plot values
                ind_stim = np.arange(t_stim[iwf], t_stim[iwf]+len(wf)) - t_stim[0]
                y_plt[ind_stim, 0] = wf

                #
                t_phase = np.floor(wfp[iwf]['tPeriod'] / 2)
                t_end = wfp[iwf]['tSS0'] + wfp[iwf]['tPeriod']

                #
                if tt == 'MotorDrifting':
                    iCC = 2 - (wfp[iwf]['yDir'] > 0)
                else:
                    iCC = 2 - (wfp[iwf]['yDir'] < 0)

                # sets the indices of the phases
                iCCW, i_phs = 3 - iCC, np.empty(3, dtype=object)
                i_phs[0] = np.arange(wfp[iwf]['tBLF'] - t_phase, wfp[iwf]['tBLF']).astype(int)
                i_phs[iCC] = np.arange(wfp[iwf]['tSS0'], wfp[iwf]['tSS0'] + t_phase).astype(int)
                i_phs[iCCW] = np.arange(t_end - t_phase, t_end).astype(int)

                #
                for ind_phs in range(len(i_phs)):
                    ii = ind_stim[i_phs[ind_phs]]
                    y_plt[ii, ind_phs + 1] = wf[i_phs[ind_phs]]


            # plots the time values
            h_plt = []
            for i in range(np.size(y_plt, axis=1)):
                h_nw = ax[jtt].plot(Ttt, y_plt[:, i], col[i])
                if i > 0:
                    h_plt.append(h_nw)

            # sets the
            if jtt == 0:
                ax[jtt].legend(cf.flat_list(list(h_plt)), ['Baseline', 'Clockwise', 'Counter-Clockwise'], loc=2)

            # sets the axis properties
            ax[jtt].set_title('Trial Type = {0}'.format(tt))
            ax[jtt].set_xlim(Ttt[0], Ttt[-1])
            ax[jtt].set_ylabel('Angle (deg)')
            ax[jtt].set_xlabel('Experiment Time (s)')
        else:
            a = 1


    # shows the final graph
    plt.tight_layout()
    plt.show()
    a = 1