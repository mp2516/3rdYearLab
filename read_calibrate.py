import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import scipy
from operator import itemgetter
from tqdm import trange
from itertools import *


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * (sigma ** 2)))


def lorentzian(x, a, x0, gamma):
    return ((a / np.pi) * 0.5 * gamma) / ((x - x0)**2 + (0.5*gamma)**2)


def compton_formula(x):
    return 662 / (1 + (662/511)*(1 - np.cos(x * np.pi / 180)))


def compton_error_formula(x):
    return compton_formula(x)**2 * (np.sin(x) / 511) * (np.pi / 90)


def error_function(x, a, b):
   return a*scipy.special.erf(x)+b


def chi_squared(num_bins, fitted, observed, observed_error):
    chi_squared = []
    for num in range(num_bins):
        chi_squared.append(((observed[num] - fitted[num]) / observed_error[num])**2)
    return np.sum(chi_squared)


def sigmoid(x, a, b, c, d):
    y = a / (b + np.exp(-d*(x-c)))
    return y


class VerifyingCompton:
    def __init__(self):
        self.num_bins = 2 ** 7 # 7
        self.cut_off = 0 # 65
        self.bin_width = 2048 / self.num_bins
        self.element_to_E = {'am241': 59.6,
                             'cd109': 88,
                             'ba133': 80,
                             'co57': 122,
                             'cs137': 662,
                             'mn54': 834.827,
                             'na22': 511.0034}
        self.element_to_energy_error = {'am241': 0,
                             'cd109': 0,
                            'ba133': 0,
                             'co57': 0,
                             'cs137': 0,
                             'mn54': 0,
                             'na22': 0}
        self.file_name_calibration = 'calibration_120s_4_900V_2048.txt'
        self.elements_calibration = ['am241', 'cd109', 'ba133', 'co57', 'cs137', 'mn54', 'na22']
        self.elements = ['cs137', 'am241', 'na22']
        self.energy = [self.element_to_E[element] for element in self.elements]
        self.energy_error = [self.element_to_energy_error[element] for element in self.elements]
        self.channels = np.arange(np.ceil(self.bin_width/2), 2049, self.bin_width)
        self.angles = [0, 20, 40, 60, 80, 100, 120, 90, 70, 30, 50, 110]
        self.diameters = [255, 118, 191, 381, 507, 763, 100]  # in units of 10^-4 m
        self.plots = True
        self.signal_background_plots = False
        self.calibration_plots = False
        self.print_results = True
        self.gaussian_error = False
        self.gaussian_sensitivity = False
        self.compton_plots = True

    def bin_data(self, column_data):
        """
        Bins the data from 2048 channels into a smaller number given by the number of bins
        :param column_data: The data to bin (len(column_data) = 2048))
        :return: column_data_binned: The binned data (len(column_data_binned) = self.num_bins)
        """
        column_data_binned = []
        bin_elem, bin_count = 0, 0
        for elem in column_data:
            bin_elem += elem
            bin_count += 1
            if bin_count == self.bin_width:
                column_data_binned.append(bin_elem)
                bin_count, bin_elem = 0, 0
        return column_data_binned

    def remove_background(self, data_binned, background_index, signal_index, cut_off=0):
        """
        Subtracts the background signal from the signal to ensure that the peaks that are fitted are exclusively due to
        the resultant_signal. The error for the resultant_signal is the addition of the signal error and background
        error added in quadrature. As they both follow poisson distributions, this is n**0.5.
        :param data_binned: The binned data, the columns are the background and signal.
        :param background_index: The column index in which the background count is stored
        :param signal_index: The column index in which the signal count is stored
        :return: resultant_signal = signal - background
                 resultant_signal_error = (signal_error + background_error) ** 0.5
        """
        resultant_signal, resultant_signal_error = [], []
        for data_num, background in enumerate(data_binned[background_index]):
            if data_num >= cut_off:
                resultant_signal.append(float(data_binned[signal_index][data_num] - background))
                resultant_signal_error.append(np.sqrt(float(data_binned[signal_index][data_num] + background)))
        return resultant_signal, resultant_signal_error

    def open_file(self, file_name):
        """
        Opens the requested file_name. Scans the .txt file.

        1) The first five lines are information about how the data was collected (0-4)
        MIN=0	0	0	0
        MAX=5	1000	1585	0.5
        SCALE=1	500	500	0.1
        DEC=0	0	0	4
        DEF="Energy" E / s	"Zählrate" N_1	"Zählrate" N_2	"Frequency" f / Hz

        2) The rest of the file (2048 rows) is composed of columns of integers with columns of NAN values. The NAN
        values are discarded and the positions of the integers are worked out each time. The last value is normally
        very large as it includes all counts above that energy too, we set this to zero to avoid an anomalous data
        point.
        NAN	0	0	NAN
        NAN	1	3	NAN
        NAN	1	2	NAN
        NAN	2	2	NAN
        NAN	4	4	NAN
        NAN	4	1	NAN
        ... ... ... ...
        NAN 50  40  NAN

        :param file_name: The name of the file to open.
        :return: data: the list of counts of the columns of the file
        """
        with open('ComptonScattering/Data/' + file_name, 'r') as x:
            for line_num, line in enumerate(x):
                if 4 < line_num < 2050:
                    digit_values, split_digit_values = [], []
                    # a list of all the character indexes where there are numbers, i.e. not NANS
                    for char_num, char in enumerate(line):
                        if char.isdigit():
                            digit_values.append(char_num)
                    # split the digit_values into separate lists based on discontinuities
                    for k, g in groupby(enumerate(digit_values), lambda y: y[0] - y[1]):
                        split_digit_values.append(list(map(itemgetter(1), g)))

                    if line_num == 5:
                        data = [[0 for _ in range(len(digit_values))] for __ in range(2049)]
                    # iterate column by column and append to the correct column
                    for column_num, column_index in enumerate(split_digit_values):

                        # print("The range of {} : {}".format(min(column_index),
                        #                                         len(line) + 1 + max(column_index) - len(first_line)))
                        count = int(line[min(column_index):
                                         1 + max(column_index)])
                        # data[column_num].insert(line_num - 5, count)
                        data[column_num].append(count)
                elif line_num == 2050:
                    for column_num in range(len(digit_values)):
                        data[column_num].append(0)
            return data

    def fit_gaussian(self, xaxis, dependent, dependent_error, initial_guess, print_note, plot_title, xlabel, ylabel, bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), calibration=False, num_bins=2**7):
        """
        Fitting a gaussian function to the data. Also plots the data and the fit.
        :param xaxis: The range over which the gaussian is being fitted, normally channels or energy.
        :param dependent: The count rate
        :param dependent_error: The count rate error
        :param initial_guess: The best guess for the location of the mean of the gaussian. If this number is incorrectly
            guessed the fit may be applied to the wrong peak.
        :param print_note: The header for the print statement
        :param plot_title: The title of the plot
        :param xlabel: The x_1-axis label
        :param ylabel: The y-axis label
        :return: popt_elem: The parameters of the fitted gaussian, the data is stored like [height, mean, sd] so for
            most purposes the 2nd and 3rd element are the important ones.
        """
        if self.gaussian_error and not calibration:
            popt_elem, pcov_elem = curve_fit(gaussian, xaxis, dependent, p0=initial_guess,
                                             sigma=dependent_error, absolute_sigma=True, bounds=bounds, method='lm')
            chisquare = chi_squared(num_bins, gaussian(xaxis, *popt_elem), dependent, dependent_error)
            reduced_chisquare = chisquare / (num_bins - 3)
        else:
            popt_elem, pcov_elem = curve_fit(gaussian, xaxis, dependent, p0=initial_guess, bounds=bounds)
            reduced_chisquare = 0
        perr_elem = np.sqrt(np.diag(pcov_elem))

        if self.print_results:
            print("\n{}"
                  "\nHeight {} +/ {}"
                  "\nMean {} +/- {}"
                  "\nSigma {} +/- {}"
                  "\nChi-squared {}".format(print_note, popt_elem[0], perr_elem[0],
                                             popt_elem[1], perr_elem[1],
                                             popt_elem[2], perr_elem[2],
                                             reduced_chisquare))
        if self.plots:
            fig, ax = plt.subplots()
            ax.errorbar(xaxis, dependent, yerr=dependent_error, fmt='d', elinewidth=0.2, color='black',
                        markersize=2)
            ax.plot(xaxis, gaussian(xaxis, *popt_elem), 'ro:', label='fit', markersize=0.01)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(plot_title)
            # plt.legend(["\nHeight: {} $\pm$ {}"
            #             "\nMean: {} $\pm$ {} keV"
            #             "\nSigma: {} $\pm$ {} keV"
            #            "\nReduced $\chi^2$: {}".format(round(popt_elem[0], 1), round(perr_elem[0], 1),
            #                                                round(popt_elem[1], 1), round(perr_elem[1], 1),
            #                                                round(popt_elem[2], 1), round(perr_elem[2], 1),
            #                                                round(reduced_chisquare, 4))],
            #            frameon=False,
            #            handlelength=0)
            plt.legend(["\nHeight = {} $\pm$ {}"
                        "\nMean = {} $\pm$ {} keV"
                        "\nSigma = {} $\pm$ {} keV".format(round(popt_elem[0], 1), round(perr_elem[0], 1),
                                                    round(popt_elem[1], 1), round(perr_elem[1], 1),
                                                    round(popt_elem[2], 1), round(perr_elem[2], 1))],
                        frameon=False, handlelength=0)
        return popt_elem, perr_elem

    def fit_linear(self, independent, dependent, dependent_error, independent_error, xaxis, xlabel, ylabel, xlim, ylim, print_note):
        """
        Fitting a line y = mx + c to the dependent and independent data. Also plots the data and the line.
        :param independent: The x_1-axis
        :param dependent: The collected data
        :param dependent_error: The error on the collected data (usually associated with binning the data)
        :param independent_error: The error on the x_1-axis points
        :param xaxis: A larger range than that of the max(dependent) - min(dependent) to draw the line on
        :param xlabel: The x_1-axis label for the graph
        :param ylabel: The y-axis label for the graph
        :param xlim: The x_1-axis upper limit for the graph
        :param ylim: The y-axis upper limit for the graph
        :param print_note: The header for the print statement that outputs the gradient and y-intercept along with the
            fitting error.
        :return: p[0]: The gradient of the fitted straight line
        """
        p, pcov = curve_fit(lambda t, a, b: a * t + b,
                            independent, dependent, p0=(1, 0),
                            sigma=dependent_error, absolute_sigma=True)
        straight_line = [value*p[0] + p[1] for value in independent]
        chisquare = chi_squared(len(dependent), straight_line, dependent, dependent_error)
        print(chisquare)
        reduced_chisquare = chisquare / (len(dependent) - 2)
        perr = np.sqrt(np.diag(pcov))
        if self.print_results:
            print("\n{}"
                  "\nGradient {} +/- {}"
                  "\nOffset {} +/- {}".format(print_note, p[0], perr[0], p[1], perr[1]))

        if self.plots:
            fig, ax = plt.subplots()
            y_error = np.sqrt((perr[0]*p[0])**2 + perr[1]**2)
            y1 = p[0] * xaxis + (p[1] + y_error)
            y2 = p[0] * xaxis + (p[1] - y_error)
            # ax.plot(xaxis, xaxis, color='green')
            ax.plot(xaxis, p[0] * xaxis + p[1], linestyle='--', color='red')
            #ax.plot(xaxis, y1, color='black')
            #ax.plot(xaxis, y2, color='black')
            # for i, txt in enumerate(list(self.element_to_E.keys())):
            #     ax.annotate(txt, (independent[i], dependent[i]))
            ax.errorbar(independent, dependent, yerr=dependent_error, xerr=independent_error, fmt='o', elinewidth=0.6,
                        color='black', markersize=4)
            plt.legend(["\nGradient: {} +/- {}"
                        "\nOffset: {} +/- {}"
                       "\nReduced $\chi^2$: {}".format(round(p[0], 2), round(perr[0], 2), round(p[1], 2), round(perr[1], 2), round(reduced_chisquare, 2))],
                        frameon=False,
                       handlelength=0)
            ax.fill_between(xaxis, y1, y2, color='grey', alpha=0.2)
            #plt.title("Standard Deviation vs $\sqrt{Energy}$")
            #plt.title("Measured Energy vs Compton Formula")
            plt.title("Calibration for all Available Sources")
            ax.set_xlim(0, xlim)
            ax.set_ylim(0, ylim)

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        return p[0], p[1]

    def calibration_all(self):
        gaus_mean, gaus_error, gaus_error_fit = [], [], []
        data_binned, data_binned_error = [], []
        for element_name, element_energy in self.element_to_E.items():
            file_name = 'Calibration/11_01_2019/' + element_name + '_callibration_120s_6_902V_2048_withlead.txt'
            for elem_num, elem_data in enumerate(self.open_file(file_name)):
                if all(t == 0 for t in elem_data):
                    continue
                elem_data_binned = self.bin_data(elem_data)
                elem_data_binned_error = [np.sqrt(binned_count) for binned_count in elem_data_binned]
            if len(elem_data_binned) != self.num_bins:
                elem_data_binned.append(0)
                elem_data_binned_error.append(0)
            p0 = [max(elem_data_binned), (element_energy * 2.35) + 40, 10]
            bounds = ([1, element_energy * 2.35 - 50, 1],
                      [max(elem_data_binned) + 50, element_energy * 2.35 + 90, 100])

            if not self.calibration_plots:
                # bit of a hacky way to fix the plotting of calibration plots
                self.plots = False

            popt_elem, perr_elem = self.fit_gaussian(self.channels, elem_data_binned, elem_data_binned_error,
                                                     initial_guess=p0,
                                                     print_note='CALIBRATION ' + element_name,
                                                     plot_title=element_name, xlabel="Channels",
                                                     ylabel="Counts", bounds=bounds, calibration=True)
            gaus_mean.append(popt_elem[1])
            gaus_error.append(popt_elem[2] / 2)
            gaus_error_fit.append(perr_elem[2])
            data_binned.append(elem_data_binned)
            data_binned_error.append(elem_data_binned_error)

        gradient, offset = self.fit_linear(list(self.element_to_E.values()), gaus_mean, gaus_error, list(self.element_to_energy_error.values()), np.arange(0, 1000, 0.1),
                                       "Energy / keV", "Channels", 1000, 2000, print_note='CALIBRATION RESULTS')

        gaus_error = [error / gradient for error in gaus_error]
        sqrt_e = [np.sqrt(energy) for energy in list(self.element_to_E.values())]

        gradient, offset = self.fit_linear(sqrt_e, gaus_error, gaus_error_fit,
                                           list(self.element_to_energy_error.values()), np.arange(0, 30, 0.01),
                                           "$\sqrt{Energy}  /  \sqrt{keV}$", "Standard Deviation / keV", 30, 40, print_note='Resolution constant')
        # plt.show()

        if not self.calibration_plots:
            self.plots = True

        # for j in range(len(gaus_mean)):
        #     element_name = self.elements_calibration[j]
        #     element_energy = self.element_to_E[element_name]
        #     low_threshold_data = int((gaus_mean[j] - gaus_error[j] * 2) / self.bin_width)
        #     if element_name == 'mn54':
        #         high_threshold_data = self.num_bins
        #     else:
        #         high_threshold_data = int((gaus_mean[j] + gaus_error[j] * 2) / self.bin_width)
        #     data_binned_cropped = data_binned[j][low_threshold_data:high_threshold_data]
        #     data_binned_error_cropped = data_binned_error[j][low_threshold_data:high_threshold_data]
        #     p0 = [max(data_binned_cropped), (element_energy * 2.35) + 40 / self.bin_width, 10]
        #     popt_elem, perr_elem = self.fit_gaussian(self.channels[low_threshold_data:high_threshold_data],
        #                                              data_binned_cropped,
        #                                              data_binned_error_cropped,
        #                                              initial_guess=p0, print_note='CALIBRATION ' + element_name,
        #                                              plot_title=element_name, xlabel="Channels", ylabel="Counts",
        #                                              calibration=False,
        #                                              num_bins=high_threshold_data-low_threshold_data)
        plt.show()
        return


    def calibration(self, date):
        """
        Calibrates the energy to channels scale by comparing the known energies of emission with the measured
        channel number.
        :param date: The date the measurement was taken, e.g. 17_01_2019. The MCA has to be calibrated daily
        as air pressure, temperature and humidity changes
        :return: returns the channels_to_energy variable
        """
        file_name = 'Calibration/' + date + '/cs137_am241_na22_calibration_120s_4_900V_2048.txt'
        gaus_mean, gaus_error = [], []
        for elem_num, elem_data in enumerate(self.open_file(file_name)):
            if all(t == 0 for t in elem_data):
                continue
            elem_data_binned = self.bin_data(elem_data)
            elem_data_binned_error = [np.sqrt(binned_count) for binned_count in elem_data_binned]

            p0 = [max(elem_data_binned), (self.energy[elem_num] * 2.4), 20]
            bounds = ([1, self.energy[elem_num] * 2.4 - 50, 1],
                      [max(elem_data_binned) + 50, self.energy[elem_num] * 2.4 + 50, 100])

            if not self.calibration_plots:
                # bit of a hacky way to fix the plotting of calibration plots
                self.plots = False

            popt_elem, perr_elem = self.fit_gaussian(self.channels, elem_data_binned, elem_data_binned_error,
                                            initial_guess=p0, print_note='CALIBRATION ' + str(self.elements[elem_num]),
                                            plot_title=str(self.elements[elem_num]),
                                          xlabel="Channels",
                                          ylabel="Counts", calibration=True)
            gaus_mean.append(popt_elem[1])
            gaus_error.append(popt_elem[2])

        gradient, offset = self.fit_linear(self.energy, gaus_mean, gaus_error, self.energy_error, self.channels, "Energy", "Steps",
                        1000, 2000, print_note='CALIBRATION RESULTS')

        if not self.calibration_plots:
            self.plots = True

        return gradient, offset

    def check_all_calibrations(self):
        all_grads, all_offsets, grad_error, offset_error = [], [], [], []
        for date in ['17_01_2019', '18_01_2019', '21_01_2019_afternoon', '21_01_2019_morning', '24_01_2019', '25_01_2019', '28_01_2019_afternoon', '28_01_2019_morning']:
            gradient, offset = self.calibration(date)
            all_grads.append(gradient)
            all_offsets.append(offset)
            grad_error.append(0)
            offset_error.append(0)

        fig, ax = plt.subplots()
        ax.errorbar(range(1, 9), all_grads, yerr=grad_error, xerr=np.zeros(8), fmt='o', elinewidth=0.6,
                    color='black', markersize=4)

        ax.set_xlim(0, 9)
        ax.set_ylim(2, 3)

        plt.xlabel("Number of Sessions")
        plt.ylabel("Calibration")
        plt.show()

    def compton_scattering(self):
        """
        Takes the different data sets for various angles of compton scattering. Each file had slight differences that
        has to be taken account of. The first three data points (0, 20, 40) are taken on a different day to
        (80, 100, 120) and (60) so must be given a different calibration.
        Additionally for 0 the columns in the data set are swapped round.
        60 was retaken due to a poor initial data set.
        120 was taken for 20 mins as opposed to 15 mins for the rest of the data due to a low count rate.
        The function will output all the fitted graphs and the final 'Compton Formula' vs 'Measured Energy' graph.
        :return: None
        """
        expected_scattered_energy, expected_scattered_energy_error = [], []
        scattered_energy, scattered_energy_error = [], []
        fitting_mean, fitting_mean_error, fitting_height = [], [], []
        resultant_signal_all, resultant_signal_error_all = [], []
        for angle_num, angle in enumerate(self.angles):
            expected_scattered_energy.append(compton_formula(angle))
            expected_scattered_energy_error.append(compton_error_formula(angle))

            # set the day on which the data acquisition occurred
            if angle in [0, 20, 40]:
                date = '17_01_2019'
            elif angle in [80, 100, 120]:
                date = '18_01_2019'
            elif angle in [60]:
                date = '21_01_2019_morning'
            elif angle in [90]:
                date = '24_01_2019'
            elif angle in [45]:
                date = '21_01_2019_afternoon'
            elif angle in [70]:
                date = '28_01_2019_morning'
            elif angle in [10, 30, 50, 110]:
                date = '28_01_2019_afternoon'

            # set the time for which the data was taken, in mins
            if angle == 120:
                time = '20'
            else:
                time = '15'

            # if the data was taken on a different day, there is a need to recalibrate
            print("\nCALIBRATION")
            channel_energy_gradient, channel_energy_offset = self.calibration(date)

            # rescale the channels on the xaxis based on the calibration energy
            xaxis_energy = [((channel - channel_energy_offset)/channel_energy_gradient).item() for channel in self.channels]

            if angle in [10, 30, 50, 110]:
                file_name = 'Scattering/' + date + '/cs137_scattering_' + time + 'min_8_900V_' + str(angle) + '.txt'
            else:
                file_name = 'Scattering/' + date + '/cs137_compscat_' + time + 'min_7_900V_' + str(angle) + '.txt'
            data = self.open_file(file_name)

            # bin the data
            data_binned = []
            for column_num, column_data in enumerate(data):
                if all(t == 0 for t in column_data) or len(column_data) != 2048:
                    continue
                column_data_binned = self.bin_data(column_data)

                data_binned.append(column_data_binned)

            # set the signal in the data to subtract the background rate
            if angle in [0, 10]:
                back_index, sig_index = 0, 1
            else:
                back_index, sig_index = 1, 0
            resultant_signal, resultant_signal_error = self.remove_background(data_binned, back_index, sig_index)

            if self.signal_background_plots:
                popt_elem, perr_elem = self.fit_gaussian(xaxis=xaxis_energy, dependent=data_binned[1],
                                              dependent_error=[elem**0.5 for elem in data_binned[1]],
                                              initial_guess=[max(data_binned[1]), compton_formula(angle), 100],
                                              print_note='DATA COLLECTION ' + str(angle), plot_title=(
                                'Background vs Energy for Cs-137 with Al Target at ' + r'$\theta = $' + str(
                            angle) + '$^\circ$'), xlabel="Energy / keV", ylabel="Count")

                popt_elem, perr_elem = self.fit_gaussian(xaxis=xaxis_energy, dependent=data_binned[0],
                                              dependent_error=[elem ** 0.5 for elem in data_binned[0]],
                                              initial_guess=[max(data_binned[0]), compton_formula(angle), 100],
                                              print_note='DATA COLLECTION ' + str(angle), plot_title=(
                            'Signal vs Energy for Cs-137 with Al Target at ' + r'$\theta = $' + str(
                        angle) + '$^\circ$'), xlabel="Energy / keV", ylabel="Count")

            bounds = ([1, compton_formula(angle) - 50, 1], [max(resultant_signal) + 20, compton_formula(angle) + 50, 100])
            if self.gaussian_sensitivity:
                gaussian_height = []
                for perc_height in np.arange(0, 2.0, 0.01):
                    initial_guess = [max(resultant_signal) * perc_height, compton_formula(angle), 20]
                    popt_elem, perr_elem = self.fit_gaussian(xaxis=xaxis_energy, dependent=resultant_signal,
                                                  dependent_error=resultant_signal_error,
                                                  initial_guess=initial_guess,
                                                  print_note='DATA COLLECTION ' + str(angle), plot_title=(
                                    'Adjusted Count vs Energy for Cs-137 with Al Target at ' + r'$\theta = $' + str(
                                angle) + '$^\circ$'), xlabel="Energy / keV", ylabel="Count")
                    gaussian_height.append(popt_elem[0] / max(resultant_signal))
                fitting_height.append(gaussian_height)
                gaussian_mean, gaussian_mean_error = [], []
                for offset in np.arange(-100, 100, 0.2):
                    initial_guess = [max(resultant_signal), compton_formula(angle) + offset, 20]
                    popt_elem, perr_elem = self.fit_gaussian(xaxis=xaxis_energy, dependent=resultant_signal,
                                                  dependent_error=resultant_signal_error, initial_guess=initial_guess,
                                                  print_note='DATA COLLECTION ' + str(angle), plot_title=(
                                'Adjusted Count vs Energy for Cs-137 with Al Target at ' + r'$\theta = $' + str(
                            angle) + '$^\circ$'), xlabel="Energy / keV", ylabel="Count")
                    gaussian_mean.append(popt_elem[1])
                    gaussian_mean_error.append(popt_elem[2])
                fitting_mean.append(gaussian_mean)
                fitting_mean_error.append(gaussian_mean_error)

            # fit the resultant signal (minus background) to a gaussian
            if self.compton_plots:
                popt_elem, perr_elem = self.fit_gaussian(xaxis=xaxis_energy,
                                              dependent=resultant_signal,
                                              dependent_error=resultant_signal_error,
                                              initial_guess=[max(resultant_signal), compton_formula(angle), 20],
                                              print_note='DATA COLLECTION ' + str(angle),
                                              plot_title=('Adjusted Count vs Energy for Cs-137 with Al Target at '
                                                          + r'$\theta = $' + str(angle) + '$^\circ$'),
                                              xlabel="Energy / keV",
                                              ylabel="Count")
                scattered_energy.append(popt_elem[1])
                scattered_energy_error.append(popt_elem[2] / 2)

            resultant_signal_all.append(resultant_signal)
            resultant_signal_error_all.append(resultant_signal_error)

        if self.gaussian_sensitivity:
            fig_mean, ax_mean = plt.subplots()
            xaxis = np.arange(-100, 100, 0.2)
            color = ['blue', 'red', 'green', 'lightgreen', 'orange', 'purple', 'grey']
            for num, fitting_mean_angle in enumerate(fitting_mean):
                ax_mean.errorbar(xaxis, fitting_mean_angle, yerr=fitting_mean_error[num], fmt='d', elinewidth=0.2, markersize=2, color=color[num])
                ax_mean.plot(xaxis, fitting_mean_angle, color=color[num])
            plt.legend(self.angles)
            ax_mean.set_xlim(-100, 100)
            ax_mean.set_ylim(0, 1000)

            plt.xlabel("Offset")
            plt.ylabel("Fitted Gaussian Mean")
            plt.title("The Fitted Gaussian Mean against the Offset in Mean of the Initial Guess")

            fig_height, ax_height = plt.subplots()
            xaxis = np.arange(0, 2, 0.01)
            color = ['blue', 'red', 'green', 'lightgreen', 'orange', 'purple', 'grey']
            for num, fitting_height_angle in enumerate(fitting_height):
                ax_height.plot(xaxis, fitting_height_angle, color=color[num])
            plt.legend(self.angles)
            ax_height.set_xlim(0, 2)
            ax_height.set_ylim(0, 1)

            plt.xlabel("Percentage change")
            plt.ylabel("Fitted Gaussian Height")
            plt.title("The Fitted Gaussian Height against the Percentage Change of the Initial Guess")

        # The straight line fit for the data, taking into account the error on the dependent variable
        if self.compton_plots:
            self.fit_linear(independent=expected_scattered_energy,
                            dependent=scattered_energy,
                            dependent_error=scattered_energy_error,
                            independent_error=expected_scattered_energy_error,
                            xaxis=np.arange(0, 700, 1.0),
                            xlabel="Compton Formula / keV", ylabel="Measured Energy / keV",
                            xlim=700, ylim=700,
                            print_note='OVERALL RESULTS')

        for j in range(len(scattered_energy)):
            angle = self.angles[j]
            print(scattered_energy[j])
            print(scattered_energy_error[j])
            print(len(resultant_signal_all[j]))
            low_threshold_data = int(((scattered_energy[j] - scattered_energy_error[j] * 6) * channel_energy_gradient
                                      - channel_energy_offset) / self.bin_width)
            high_threshold_data = int(((scattered_energy[j] + scattered_energy_error[j] * 12) * channel_energy_gradient
                                       - channel_energy_offset) / self.bin_width)
            print(high_threshold_data)
            print(low_threshold_data)
            data_binned_cropped = resultant_signal_all[j][low_threshold_data:high_threshold_data]
            data_binned_error_cropped = resultant_signal_error_all[j][low_threshold_data:high_threshold_data]
            print(data_binned_cropped)
            print(data_binned_error_cropped)
            print(len(data_binned_cropped))
            print(len(data_binned_error_cropped))
            p0 = [max(data_binned_cropped), scattered_energy[j], 10]
            popt_elem, perr_elem = self.fit_gaussian(xaxis_energy[low_threshold_data:high_threshold_data],
                                                     data_binned_cropped,
                                                     data_binned_error_cropped,
                                                     initial_guess=p0, print_note='CROPPED DATA',
                                                     plot_title='Adjusted Count vs Energy for Cs-137 with Al Target at '+ r'$\theta = $' + str(angle) + '$^\circ$', xlabel="Energy / keV", ylabel="Count",
                                                     calibration=False,
                                                     num_bins=high_threshold_data-low_threshold_data)
        plt.show()

    def finite_size_scattering(self):
        gaussian_params, gaussian_params_error = [], []
        background_count = []
        diam_to_colour = {118: 'bo:', 255: 'ro:', 191: 'go:', 381: 'co:', 507: 'mo:', 763: 'yo:', 100: 'ko:'}
        for diam_num, diam in enumerate(self.diameters):
            if diam == 118:
                continue
            elif diam == 255:
                date = '21_01_2019_afternoon'
                time = '23'
                index_to_diam = {1: 118, 0: 255}
                index_to_colour = {1: 'bo:', 0: 'ro:'}
            elif diam == 191:
                date = '24_01_2019'
                time = '23'
            elif diam in [381, 507, 763]:
                date = '25_01_2019'
                time = '23'
            elif diam == 100:
                date = '28_01_2019_morning'
                time = '23'

            channel_energy_gradient, channel_energy_offset = self.calibration(date)

            # rescale the channels on the xaxis based on the calibration energy
            xaxis_energy = [((channel - channel_energy_offset)/channel_energy_gradient).item() for channel in self.channels]
            xaxis_energy_cut = xaxis_energy[self.cut_off:]

            file_name = 'Scattering/' + date + '/cs137_scattering_' + time + 'min_8_900V_45_d' + str(diam) + '.txt'
            data = self.open_file(file_name)

            # bin the data
            data_binned = []
            for column_num, column_data in enumerate(data):
                if all(t == 0 for t in column_data) or len(column_data) not in [2047, 2048, 2049]:
                    continue
                column_data_binned = self.bin_data(column_data)
                data_binned.append(column_data_binned)

            # for i in [0, 1, 2]:
            #     self.fit_gaussian(xaxis=xaxis_energy, dependent=data_binned[i],
            #                                   dependent_error=[elem ** 0.5 for elem in data_binned[i]],
            #                                   initial_guess=[max(data_binned[i]), compton_formula(45), 100],
            #                                   print_note='DATA COLLECTION ' + str(45), plot_title=(
            #                 'Background vs Energy for Cs-137 with Al Target at ' + r'$\theta = $' + str(45) + '$^\circ$'),
            #                                   xlabel="Energy / keV", ylabel="Count")

            if diam == 255:
                back_index = 2
                for sig_index in [0, 1]:
                    resultant_signal, resultant_signal_error = [], []
                    for data_num, background in enumerate(data_binned[back_index]):
                        if data_num >= self.cut_off:
                            resultant_signal.append(float(data_binned[sig_index][data_num] - background))
                            resultant_signal_error.append(np.sqrt(float(data_binned[sig_index][data_num] + background)))

                    popt_elem, perr_elem = self.fit_gaussian(xaxis=xaxis_energy_cut, dependent=resultant_signal,
                                                  dependent_error=resultant_signal_error,
                                                  initial_guess=[max(resultant_signal), compton_formula(45), 100],
                                                  print_note='DATA COLLECTION ' + str(index_to_diam[sig_index]) + 'cm',
                                                  plot_title=('Count vs Energy for Cs-137 with ' +
                                                              str(index_to_diam[sig_index]) + r'$\times 10^{-4}$m' +
                                                              ' Al Target at ' + r'$\theta = 45^\circ$'),
                                                  xlabel="Energy / keV",
                                                  ylabel="Count", num_bins=self.num_bins - self.cut_off)
                    gaussian_params.append(popt_elem)
                    gaussian_params_error.append(perr_elem)
            elif diam == 191:
                resultant_signal, resultant_signal_error = [], []
                for data_num, background in enumerate(data_binned[1]):
                    if data_num >= self.cut_off:
                        resultant_signal.append(float(data_binned[0][data_num] - (69/49)*background))
                        resultant_signal_error.append(np.sqrt(float(data_binned[0][data_num] + background)))


                popt_elem, perr_elem = self.fit_gaussian(xaxis=xaxis_energy_cut, dependent=resultant_signal,
                                              dependent_error=resultant_signal_error,
                                              initial_guess=[max(resultant_signal), compton_formula(45), 100],
                                              print_note='DATA COLLECTION ' + str(diam) + 'cm',
                                              plot_title=('Count vs Energy for Cs-137 with ' +
                                                          str(diam) + r'$\times 10^{-4}$m' + 'Al Target at ' + r'$\theta = 45^\circ$'),
                                              xlabel="Energy / keV", ylabel="Count", num_bins=self.num_bins - self.cut_off)
                gaussian_params.append(popt_elem)
                gaussian_params_error.append(perr_elem)
            elif diam in [381, 507, 763, 100]:
                resultant_signal, resultant_signal_error = [], []
                if diam in [381, 763, 100]:
                    background_count = data_binned[1]
                    if diam == 381:
                        initial_guess = [400, 500, 50]
                        bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
                    elif diam == 763:
                        initial_guess = [250, 515, 50]
                        # bounds = ([200, 475, 20], [np.inf, np.inf, 70])
                        bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
                    else:
                        initial_guess = [200, 500, 50]
                        bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
                else:
                    data_binned[0].append(0)
                    initial_guess = [250, 520, 50]
                    bounds = ([200, 465, 20], [np.inf, np.inf, 65])
                    #bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

                for data_num, background in enumerate(background_count):
                    if data_num >= self.cut_off:
                        resultant_signal.append(float(data_binned[0][data_num] - background))
                        resultant_signal_error.append(np.sqrt(float(data_binned[0][data_num] + background)))

                popt_elem, perr_elem = self.fit_gaussian(xaxis=xaxis_energy_cut, dependent=resultant_signal,
                                              dependent_error=resultant_signal_error,
                                              bounds=bounds,
                                              initial_guess=initial_guess,
                                              print_note='DATA COLLECTION ' + str(diam) + 'cm', plot_title=(
                                'Count vs Energy for Cs-137 with ' + str(
                            diam) + r'$\times 10^{-4}$m' + ' Al Target at ' + r'$\theta = 45^\circ$'), xlabel="Energy / keV", ylabel="Count", num_bins=self.num_bins - self.cut_off)

                gaussian_params.append(popt_elem)
                gaussian_params_error.append(perr_elem)

        for label_num, label in enumerate(['Count Rate', 'Mean', 'Standard Deviation']):
            if label == "Count Rate":
                gaussian_param = [params[0] * params[2] for params in gaussian_params]
                gaussian_param_error = [param_error[0] + param_error[2] for param_error in gaussian_params_error]
            else:
                gaussian_param = [params[label_num] for params in gaussian_params]
                gaussian_param_error = [param_error[label_num] for param_error in gaussian_params_error]

            # p, pcov = curve_fit(lambda t, a, b: a * t + b, self.diameters, gaussian_sd, p0=(1, 0))
            # perr = np.sqrt(np.diag(pcov))
            # print("\n{}"
            #       "\nGradient {} +/- {}"
            #       "\nOffset {} +/- {}".format("OVERALL RESULTS", p[0], perr[0], p[1], perr[1]))

            fig, ax = plt.subplots()
            for diam_num, diam in enumerate(self.diameters):
                if label == 'Count Rate':
                    ax.plot(xaxis_energy,
                            gaussian(xaxis_energy, 1, gaussian_params[diam_num][1], gaussian_params[diam_num][2]),
                            diam_to_colour[diam], label=diam, markersize=0.01)
                    ax.set_ylim(0, 1)
                elif label == 'Standard Deviation':
                    ax.plot(xaxis_energy,
                            gaussian(xaxis_energy, 1, compton_formula(45), gaussian_params[diam_num][2]),
                            diam_to_colour[diam], label=diam, markersize=0.01)
                    ax.set_ylim(0, 1)
                else:
                    ax.plot(xaxis_energy,
                            gaussian(xaxis_energy, *gaussian_params[diam_num]),
                            diam_to_colour[diam], label=diam, markersize=0.01)
                    ax.set_ylim(0, 450)
            ax.set_xlim(0, 800)
            ax.axvline(x=compton_formula(45), color='black', linestyle='--')
            ax.legend(loc='best')
            ax.set_xlabel("Energy / keV")
            ax.set_ylabel("Count Rate")

            fig_gaus, ax_gaus = plt.subplots()
            ax_gaus.errorbar(self.diameters, gaussian_param, yerr=gaussian_param_error, fmt='d', elinewidth=0.5, capsize=4,
                                color='black', markersize=1)
            ax_gaus.scatter(self.diameters, gaussian_param, label='Experimental Data', color='black')

            ax_gaus.set_xlabel("Diameter of Scattering Target / $10^{-4}$m")
            ax_gaus.set_xlim(0, 800)
            if label == 'Standard Deviation':
                ax_gaus.set_ylim(0, 100)
                ax_gaus.scatter(sorted(self.diameters),
                                [32.083333333333336, 32.5, 35.41666666666667, 39.16666666666667, 48.75, 60.0,
                                 82.91666666666667], label='Model Data', marker='d', color='red')
                ax_gaus.errorbar(x=sorted(self.diameters),
                                y=[32.083333333333336, 32.5, 35.41666666666667, 39.16666666666667, 48.75, 60.0,
                                 82.91666666666667],
                                yerr=[[1.2738853503184693, 1.698513800424628, 1.698513800424628, 1.2738853503184728, 0.8492569002123176, 0.8492569002123176, 0.8492569002123105],
[1.698513800424628, 1.698513800424628, 1.2738853503184657, 1.2738853503184728, 0.8492569002123105, 0.42462845010615524, 0.8492569002123247]], fmt='d', elinewidth=0.6, capsize=4,
                                color='black', markersize=1)
                ax_gaus.legend(loc='best')
                # popt_error, pcov_error = curve_fit(sigmoid, self.diameters, gaussian_param)
                # ax_gaus.plot(np.arange(0, 800, 0.1), sigmoid(np.arange(0, 800, 0.1), *popt_error))
            elif label == 'Count Rate':
                ax_gaus.set_ylim(0, 30000)
            else:
                ax_gaus.set_ylim(460, 510)
                p, pcov = curve_fit(lambda t, a, b: a * t + b, self.diameters, gaussian_param, p0=(1, 0))
                perr = np.sqrt(np.diag(pcov))
                xaxis = np.arange(0, 800, 0.01)
                y1 = (p[0] + perr[0]) * xaxis + (p[1] + perr[1])
                y2 = (p[0] - perr[0]) * xaxis + (p[1] - perr[1])
                ax_gaus.plot(xaxis, y1, color='green')
                ax_gaus.plot(xaxis, y2, color='green')
                ax_gaus.fill_between(xaxis, y1, y2, color='grey', alpha=0.5)
                ax_gaus.axhline(y=compton_formula(45), color='black', linestyle='--')
                #ax_gaus.axhline(y=0, color='grey', linestyle=':')
            #ax_gaus.set_ylabel(label)
            ax_gaus.set_ylabel("Standard Deviation / kev")
            #ax_gaus.set_title("The " + label + " of the Fitting Gaussian against Scattering Target Diameters")
            ax_gaus.set_title("Standard Deviation vs Target Diameter")

        plt.show()


    def pure_spectrum(self):
        file_name = 'Scattering/24_01_2019/cs137_pure_10min_7_900V_0.txt'
        date = '24_01_2019'
        data = self.open_file(file_name)
        fig, ax = plt.subplots()
        resultant_signal, resultant_signal_error = self.remove_background(data, 0, 1)
        channel_energy_gradient, channel_energy_offset = self.calibration(date)
        xaxis_energy = [((channel - channel_energy_offset) / channel_energy_gradient).item() for channel in
                        np.arange(1, 2049, 1)]
        ax.errorbar(xaxis_energy, resultant_signal, yerr=resultant_signal_error, fmt='d', elinewidth=0.2, color='black', markersize=2)
        ax.set_xlabel("Energy / keV")
        ax.set_ylabel("Count")
        ax.set_title("Spectrum for Cs-137")
        #ax.plot(xaxis, gaussian(xaxis, *popt_elem), 'ro:', label='fit', markersize=0.01)
        plt.show()





