import os

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

from rpy2.robjects import pandas2ri


from rpy2.robjects.packages import importr

from margrie_libs.signal_processing import mat_utils


stats = importr('stats')
pandas2ri.activate()

N_BINS_PER_SECOND_PSTH = 4


class ExperimentPlotter(object):
    def __init__(self, exp):
        """

        :param Experiment exp:
        """
        self.exp = exp
        self.bsl_psth_x = None
        self.bsl_psth_y = None

        self.psth_y = None
        self.psth_x = None

    def __setup_main_fig(self):
        gs = gridspec.GridSpec(3, 2, width_ratios=[2, 7])  # WARNING: compute from durations
        f = plt.figure()
        f.suptitle("Cell: {},  type: {},  layer: {}".
                   format(self.exp.exp_id, self.exp.cell_type, self.exp.layer), fontsize=18)

        ax1 = f.add_subplot(gs[0])
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2 = f.add_subplot(gs[1], sharey=ax1)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax3 = f.add_subplot(gs[2], sharex=ax1)
        plt.setp(ax3.get_xticklabels(), visible=False)
        ax4 = f.add_subplot(gs[3], sharex=ax2, sharey=ax3)
        plt.setp(ax4.get_yticklabels(), visible=False)
        plt.setp(ax4.get_xticklabels(), visible=False)
        ax5 = f.add_subplot(gs[4], sharex=ax1)
        ax6 = f.add_subplot(gs[5], sharex=ax2, sharey=ax5)
        plt.setp(ax6.get_yticklabels(), visible=False)

        return f, ax1, ax2, ax3, ax4, ax5, ax6

    def plot(self, savefig=False):
        fig, ax1, ax2, ax3, ax4, ax5, ax6 = self.__setup_main_fig()
        y_lim_range = 20

        # BASELINE
        # Cut and average baseline too to ensure same number of trials are averaged
        cmd_bsl, vel_bsl, acc_bsl = self.exp.get_command_plot_baselines()
        bsl_x_axis = np.linspace(0, self.exp.bsl_plot_segment_duration, self.exp.bsl_plot_segment_length)

        ax1.set_title('Commands baseline')
        ax1.plot(bsl_x_axis, cmd_bsl, label='Pos')
        ax1.plot(bsl_x_axis, vel_bsl, label='Vel')
        ax1.plot(bsl_x_axis, acc_bsl, label='Acc')
        ax1.legend()

        ax3.set_title('Data baseline')
        bsl_mean = self.exp.bsl_clipped_baselined_mean
        bsl_mean_mean = bsl_mean.mean()
        ax3.plot(mat_utils.decimate_x(bsl_x_axis), mat_utils.decimate(bsl_mean), lw=0.25)
        ax3.fill_between(mat_utils.decimate_x(bsl_x_axis),
                         bsl_mean_mean,
                         mat_utils.decimate(bsl_mean),
                         lw=0.25,
                         facecolor='blue', edgecolor='blue',
                         zorder=(len(self.exp._raw_data) * 2),  # TODO: should be raw_clipped_baselined_data
                         interpolate=True
                         )  # Filled curve plots
        ax3.set_ylim((bsl_mean_mean - y_lim_range, bsl_mean_mean + y_lim_range))
        ax3.set_xlim((0, self.exp.bsl_plot_segment_duration))

        # CYCLES  # TODO: cycles repeats baseline, should merge into function
        cmd_segment, vel_segment, acc_segment = self.exp.get_command_plot_segments()
        cycles_x_axis = np.linspace(0, self.exp.segment_duration, self.exp.segment_length)

        ax2.set_title('Commands')
        ax2.plot(cycles_x_axis, cmd_segment, label='Pos')
        ax2.plot(cycles_x_axis, vel_segment, label='Vel')
        ax2.plot(cycles_x_axis, acc_segment, label='Acc')

        data_segment_mean = self.exp.data_plot_segment
        ax4.set_title('Data')
        ax4.plot(mat_utils.decimate_x(cycles_x_axis), mat_utils.decimate(data_segment_mean), lw=0.25)
        ax4.fill_between(mat_utils.decimate_x(cycles_x_axis),
                         bsl_mean_mean,
                         mat_utils.decimate(data_segment_mean),
                         lw=0.25,
                         facecolor='blue', edgecolor='blue',
                         zorder=(len(self.exp._raw_data) * 2),  # TODO: should be raw_clipped_baselined_data
                         interpolate=True
                         )
        ax4.set_xlim((0, self.exp.segment_duration))  # Clip axis to max

        self._plot_rasters(ax5, ax6)

        if savefig:
            fig_path = os.path.join(self.exp.dir, '{}_Summary.{}'.format(self.exp.name, self.exp.ext))
            plt.savefig(fig_path)
            plt.close(fig)
        else:
            plt.show()
        self.plot_hist()
        self.plot_vm_vs_vel()

    def _plot_rasters(self, bsl_axis, data_segment_axis):
        # RASTERS
        bsl_rasters, rasters, raster_starts = self.exp.get_rasters()
        rasters = np.array([r - s for r, s in zip(rasters, raster_starts)])  # absolute to relative start times

        # Plot rasters
        for i, (bsl_rstr, rstr) in enumerate(zip(bsl_rasters, rasters)):
            self._plot_raster_segment(bsl_rstr, bsl_axis, i)
            self._plot_raster_segment(rstr, data_segment_axis, i)

        # PSTHS
        # group rasters
        bsl_raster = self.exp._concatenate_rasters(bsl_rasters)
        raster = self.exp._concatenate_rasters(rasters)
        # --- BSL
        psth_bin_width = 1 / N_BINS_PER_SECOND_PSTH
        bins = np.arange(0, round(self.exp.baseline_duration) + psth_bin_width, psth_bin_width)
        psth_y, psth_x = np.histogram(bsl_raster, bins)
        psth_y = np.hstack((psth_y, np.array(psth_y[-1])))
        bsl_axis.step(psth_x, psth_y, where='post', color='b')
        self.bsl_psth_y = psth_y
        self.bsl_psth_x = psth_x

        # --- DATA
        bins = np.arange(0, round(self.exp.segment_duration) + psth_bin_width, psth_bin_width)
        psth_y, psth_x = np.histogram(raster, bins)

        psth_y = np.hstack((psth_y, np.array(psth_y[-1])))  # WARNING: do after dsi
        data_segment_axis.step(psth_x, psth_y, where='post', color='b')

        self.psth_y = psth_y
        self.psth_x = psth_x

    def _plot_raster_segment(self, raster, subplt, index):
        """
        raster and segment_start_time expected in seconds

        :param raster: raster in seconds
        :param subplt:
        :param index: y value of the plot for that raster
        :return:
        """
        y_vals = np.ones(raster.size) * index
        x_axis = raster.copy()
        for x, y in zip(x_axis, y_vals):
            subplt.plot((x, x), (y, y+1), '-', c='#6D6D6D')
        return x_axis

    def plot_hist(self):
        f2 = plt.figure()
        # self.hist(self.exp.bsl_clipped_baselined_mean)
        # self.hist(self.exp.clock_wise_clipped_baselined_mean)
        # self.hist(self.exp.c_clock_wise_clipped_baselined_mean)
        bsl = self.exp.bsl_clipped_baselined_mean
        cw = self.exp.clock_wise_clipped_baselined_mean
        ccw = self.exp.c_clock_wise_clipped_baselined_mean

        # bsl, cw, ccw = self.exp.get_pooled_vms_bsl_cw_ccw()
        # bsl = np.random.choice(_bsl, 10000)
        # cw = np.random.choice(_bsl, 10000)
        # ccw = np.random.choice(_bsl, 10000)
        # df = pd.DataFrame.from_dict({
        #     'bsl': bsl,
        #     'cw': cw,
        #     'ccw': ccw},
        #     orient='index').transpose()
        # df.to_csv(os.path.join(self.exp.dir, "{}_distribs.csv".format(self.exp.name)))
        bins = np.linspace(-10, 10, 1000)
        if cw.size != ccw.size:
            raise ValueError("Number of points differ for cw({}) and ccw({}) for cell: {}"
                             .format(cw.size, ccw.size, self.exp.name))
        # print("Number of points: bsl:{}, cw:{}, ccw:{}".format(bsl.size, cw.size, ccw.size))
        # print("Shapes: bsl:{}, cw:{}, ccw:{}".format(bsl.shape, cw.shape, ccw.shape))

        # _, cw_vs_bsl_p = scipy.stats.mannwhitneyu(bsl, cw); print(cw_vs_bsl_p)
        # _, ccw_vs_bsl_p = scipy.stats.mannwhitneyu(bsl, ccw); print(ccw_vs_bsl_p)
        plt.suptitle("Distribution of Vm for cell: {}".format(self.exp.name))
        bsl_counts, bsl_bins, _ = plt.hist(bsl, bins, histtype='step', color='blue', label='baseline')
        cw_counts, cw_bins, _ = plt.hist(cw, bins, histtype='step', color='red', label='clockwise')
        ccw_counts, ccw_bins, _ = plt.hist(ccw, bins, histtype='step', color='green', label='counter_clockwise')
        plt.clf()
        cw_counts *= (bsl_counts.max() / cw_counts.max())
        ccw_counts *= (bsl_counts.max() / ccw_counts.max())
        plt.step(bsl_bins[:-1], bsl_counts, label='baseline')
        plt.step(cw_bins[:-1], cw_counts, label='clockwise')
        plt.step(ccw_bins[:-1], ccw_counts, label='counter_clockwise')
        plt.xlabel("Vm (mV)")
        plt.ylabel("Counts (N)")
        # plt.figtext(0.12, 0.8, "Clockwise p value: {}".format(cw_vs_bsl_p))
        # plt.figtext(0.12, 0.75, "Counter clockwise p value: {}".format(ccw_vs_bsl_p))
        plt.legend()

        fig_path = os.path.join(self.exp.dir, '{}_Distributions.{}'.format(self.exp.name, self.exp.ext))
        plt.savefig(fig_path)
        plt.close(f2)

        # w, cw_vs_bsl_p = scipy.stats.ranksums(bsl, cw)
        # print(stats.t_test(bsl, cw, **{'var.equal': False,
        #                                 'paired': False}))
        # print(stats.wilcox_test(bsl, ccw))  #, **{'var.equal': False,
                                        # 'paired': False}))
        #       fml = Formula("frame$vm ~ frame$velocity")
        # # fml.environment['frame'] = frame

    def hist(self, data):
        y, x = np.histogram(data, 100)
        y = np.hstack((y, np.array(y[-1])))
        plt.step(x, y, where='post')

    def plot_cumsum(self, data):
        cs = np.cumsum(data)
        plt.plot(cs)

    def plot_vm_vs_vel(self):
        velocities, matrix = self.exp.resampler._get_velocity_matrix()
        velocity_matrix_average, velocity_matrix_sd, velocity_matrix_se, bin_edges = \
            self.exp.resampler._average_velocity_matrix(velocities, matrix)

        accelerations, acc_mat = self.exp.resampler._get_acceleration_matrix()
        acc_mat_average, acc_mat_sd, acc_mat_se, acc_bin_edges = \
            self.exp.resampler._average_acceleration_matrix(accelerations, acc_mat)

        f = plt.figure()
        f.suptitle("{}_{}_{}".format(self.exp.name, self.exp.cell_type, self.exp.layer))
        # bin_width = bin_edges[1] - bin_edges[0]
        # plt.errorbar(bin_edges[:-1] + bin_width/2, velocity_matrix_average, velocity_matrix_sd, linestyle='None')

        csv_path = os.path.join(self.exp.dir, '{}_vm_vs_velocity.{}'.format(self.exp.name, 'csv'))
        sep = ','
        with open(csv_path, 'w') as out_file:
            out_file.write("{1}{0}{2}{0}{3}\n".format(sep, "bin start", "average", "sd"))
            for i in range(len(velocity_matrix_average)):
                out_file.write("{1}{0}{2}{0}{3}\n"
                               .format(sep, bin_edges[i], velocity_matrix_average[i], velocity_matrix_sd[i]))
        # velocity_matrix_average = np.hstack((velocity_matrix_average, velocity_matrix_average[-1]))
        # plt.step(bin_edges, velocity_matrix_average, where='post')
        # velocities = np.linspace(-80, 79, 160)  # FIXME: hard coded + overwrite
        plt.plot(velocities, velocity_matrix_average, color='blue')
        plt.plot(velocities, velocity_matrix_average - velocity_matrix_sd, color='gray', linewidth=.25)
        plt.plot(velocities, velocity_matrix_average + velocity_matrix_sd, color='gray', linewidth=.25)
        plt.plot(velocities, velocity_matrix_average - velocity_matrix_se, color='red', linewidth=.25)
        plt.plot(velocities, velocity_matrix_average + velocity_matrix_se, color='red', linewidth=.25)
        plt.plot(velocities, np.zeros(velocity_matrix_average.size), color='black', linewidth=.25)

        fig_path = os.path.join(self.exp.dir, '{}_vm_vs_velocity.{}'.format(self.exp.name, self.exp.ext))
        plt.savefig(fig_path)
        plt.close(f)


        # def plotCrossCor(self):
        #     f1 = plt.figure('XCor')
        #     f1.suptitle("Cell: {}, type: {}, layer: {}".format(self.exp_id, self.cell_type, self.layer))
        #     f2 = plt.figure('shuffledXCor')
        #     f2.suptitle("Cell: {}, type: {}, layer: {}".format(self.exp_id, self.cell_type, self.layer))
        #     f3 = plt.figure('PowerSpectra')
        #     f3.suptitle("Cell: {}, type: {}, layer: {}".format(self.exp_id, self.cell_type, self.layer))
        #
        #     cmd_segment = mat_utils.cutAndAvgSine(self.cmd, self.cmd)
        #     vel_segment = mat_utils.cutAndAvgSine(self.cmd, self.velocity)
        #     #        directionSegment = vel_segment / abs(vel_segment)
        #     n_pnts = cmd_segment.shape[0]
        #     x_cor_x_axis = np.linspace(-n_pnts * self.sampling, n_pnts * self.sampling,
        #                                n_pnts)  # FIXME: check why not nPts*2
        #     vel_x_cors = []
        #     vel_x_cors_shuffled = []
        #     all_clipped_segments = []
        #     power_spectra = []
        #     for rawTrace in self.raw_clipped_data:
        #         cutTraces = mat_utils.cutAndGetMultiple(self.cmd, rawTrace)
        #         for trace in cutTraces:
        #             #                posXCor = cross_correlation.normalised_periodic_cross_cor(cmd_segment, trace)
        #             #                plt.plot(x_cor_x_axis[:-1], posXCor, color='b', alpha=0.5) # FIXME: should not have to fix index
        #             plt.figure(f1.number)
        #             vel_x_cor = cross_correlation.normalised_periodic_cross_cor(trace, vel_segment)
        #             plt.plot(x_cor_x_axis[:-1], vel_x_cor, color='g', alpha=0.5)
        #
        #             #                directionXCor = cross_correlation.normalised_periodic_cross_cor(trace, directionSegment)
        #             #                plt.plot(x_cor_x_axis[:-1], directionXCor, color='purple', alpha=0.25)
        #
        #             plt.figure(f2.number)
        #             vel_x_cor_shuffled = cross_correlation.normalised_periodic_cross_cor_shuffled(trace, vel_segment)
        #             plt.plot(x_cor_x_axis[:-1], vel_x_cor_shuffled, color='g', alpha=0.5)
        #
        #             plt.figure(f3.number)
        #             f, power_spectrum = periodogram(trace, 1 / self.sampling, scaling='spectrum')
        #             plt.step(f, power_spectrum)
        #
        #             vel_x_cors.append(vel_x_cor)
        #             vel_x_cors_shuffled.append(vel_x_cor_shuffled)
        #             all_clipped_segments.append(trace)
        #             power_spectra.append(power_spectrum)
        #
        #     plt.figure(f1.number)
        #     vel_x_cor_mean = np.mean(np.array(vel_x_cors), 0)
        #     plt.plot(x_cor_x_axis[:-1], vel_x_cor_mean, color='b')
        #     mean_trace = np.mean(np.array(all_clipped_segments), 0)
        #     mean_vel_x_cor = cross_correlation.normalised_periodic_cross_cor(mean_trace, vel_segment)
        #     plt.plot(x_cor_x_axis[:-1], mean_vel_x_cor, color='purple')
        #     plt.ylim((-1, 1))
        #
        #     plt.figure(f2.number)
        #     vel_x_cor_shuffled_mean = np.mean(np.array(vel_x_cors_shuffled), 0)
        #     plt.plot(x_cor_x_axis[:-1], vel_x_cor_shuffled_mean, color='b')
        #     plt.ylim((-1, 1))
        #
        #     plt.figure(f1.number)
        #     plt.savefig(os.path.join(self.dir, '{}_CrossCorVel.eps'.format(self.exp_id)))
        #     plt.close(f1)
        #
        #     plt.figure(f2.number)
        #     plt.savefig(os.path.join(self.dir, "{}_CrossCorVelShuffled.eps".format(self.exp_id)))
        #     plt.close(f2)
        #
        #     plt.figure(f3.number)
        #     plt.xlim((0, 20))
        #     plt.savefig(os.path.join(self.dir, '{}_powerSpectrum.eps'.format(self.exp_id)))
        #     plt.close(f3)