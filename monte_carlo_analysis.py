import pickle
import numpy as np
import matplotlib.pyplot as plt
from ComptonScattering.monte_carlo_sim import start, stop, step

import numpy as np

from ComptonScattering.read_calibrate import gaussian, compton_formula

def fwhm(data, max_value, max_value_index):
    low = 0
    for num, i in enumerate(data):
        if i > max_value / 2:
            if low == 0:
                low = num
        elif num > max_value_index:
            high = num
            break
    return (high - low) / 2.4

standard_deviation = np.sqrt(compton_formula(45)) * 1.2642621944641599 - 4.024042562124008
energy_range = np.arange(-350, 350, 1)
# print(energy_range)

gaussian_detector_function = [gaussian(x=energy, a=1, x0=0, sigma=standard_deviation) for energy in energy_range]
# print(gaussian_detector_function)
# plt.plot(energy_range, gaussian_detector_function)

all_histogram_data = pickle.load(open('histogram_data_rtmax', 'rb'))
xaxis = np.arange(0, 700, 1)
# colours = ['black', 'dimgray', 'dimgrey', 'gray', 'grey', 'darkgray', 'darkgrey', 'silver', 'lightgray', 'lightgrey']
fig, ax = plt.subplots()
plt.xlabel("Energy / keV")
plt.ylabel("Count")
colours = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:pink', 'tab:olive', 'crimson']
diam = [100, 118, 191, 255, 381, 507, 763]
mean_energy, max_count, sd = [], [], []
for i in range(int((stop-start)/step)):
    convolution = np.convolve(gaussian_detector_function, all_histogram_data[i], 'same')
    mean_energy.append(np.argmax(convolution))
    sd.append(fwhm(convolution, max(convolution), mean_energy[-1]))
    max_count.append(max(convolution) * sd[-1])
    #plt.plot(xaxis, all_histogram_data[i], linestyle='-', color=colours[i], label=str(diam[i])+' x $10^{-4}$ m', alpha=0.8)
    plt.plot(xaxis, np.convolve(gaussian_detector_function, all_histogram_data[i], 'same'), linestyle='-', color=colours[i], label=str(diam[i])+' x $10^{-4}$ m', alpha=0.8)
plt.legend(loc='best')

for param in [mean_energy, max_count, sd]:
    fig_gaus, ax_gaus = plt.subplots()
    ax_gaus.scatter(diam, param)


# all_histogram_data[i]

plt.show()