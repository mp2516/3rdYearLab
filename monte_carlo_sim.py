import numpy as np
import matplotlib.pyplot as plt
import pickle
from .read_calibrate import compton_formula, gaussian
from tqdm import trange

all_histogram_data = []
# r_tmax = 125
r_dmax = 97.5
r_bmax = 550
detector_distance = 1800
source_distance = 1550
source_angle = 45
num_samples = 1000000
num_sims = 1
S = np.array([-source_distance * np.sin(source_angle*np.pi/180), 0, -source_distance * np.cos(source_angle*np.pi/180)])
start = 0
stop = 7
step = 1

standard_deviation = np.sqrt(compton_formula(45)) * 1.2642621944641599 - 4.024042562124008
standard_deviation_error = np.sqrt(compton_formula(45) * 0.10050524780381109**2 + 0.8067533289528244**2)
energy_range = np.arange(-350, 350, 1)

gaussian_detector_function_high = [gaussian(x=energy, a=1, x0=0, sigma=standard_deviation + standard_deviation_error)




                                   for energy in energy_range]
gaussian_detector_function_low = [gaussian(x=energy, a=1, x0=0, sigma=standard_deviation - standard_deviation_error)
                                  for energy in energy_range]
gaussian_detector_function = [gaussian(x=energy, a=1, x0=0, sigma=standard_deviation)
                              for energy in energy_range]



def cal_sd(data, max_value, max_value_index):
    low = 0
    for num, i in enumerate(data):
        if i > max_value / 2:
            if low == 0:
                low = num
        elif num > max_value_index:
            high = num
            break
    return (high - low) / 2.355

mean_energy_all, max_count_all, sd_all = [], [], []
mean_energy_err, max_count_err, sd_err = [], [], []
sd_error_upper_all, sd_error_lower_all = [], []
j=0
fig, ax = plt.subplots()
for r_tmax in [100, 118, 191, 255, 381, 507, 763]:
    r_tmax /= 2
    # S = np.array([-source_distance * np.cos(source_angle * np.pi / 180), 0,
    #               -source_distance * np.sin(source_angle * np.pi / 180)])
    # all distances in ccm
    xaxis = np.arange(0, 700, 1)
    mean_energy = []
    max_count = []
    sd = []
    sd_error_upper, sd_error_lower = [], []
    for sim_num in trange(num_sims):
        if r_tmax == 100:
            y_1 = list(np.random.uniform(low=-357, high=357, size=num_samples))
        else:
            y_1 = list(np.random.uniform(low=-450, high=r_bmax, size=num_samples))
        z_1 = list(np.random.uniform(low=-r_tmax, high=r_tmax, size=num_samples))

        x_1 = []
        for z_elem in z_1:
            x_max = np.sqrt(r_tmax**2 - z_elem**2)
            x_1.append(float(np.random.uniform(low=-x_max, high=x_max, size=1)))

        x_2 = [detector_distance] * num_samples
        y_2 = list(np.random.uniform(low=-r_dmax, high=r_dmax, size=num_samples))

        z_2 = []
        for y_elem in y_2:
            z_max = np.sqrt(r_dmax**2 - y_elem**2)
            z_2.append(float(np.random.uniform(low=-z_max, high=z_max, size=1)))

        I_1 = []
        for num in range(num_samples):
            I_1.append(np.asarray([x_1[num], y_1[num], z_1[num]]))

        I_2 = []
        for num in range(num_samples):
            I_2.append(np.asarray([x_2[num], y_2[num], z_2[num]]))

        # P_1 = [I_1_elem - S for I_1_elem in I_1]
        P_1, P_2 = [], []
        for num in range(num_samples):
            P_1.append(I_1[num] - S)
            P_2.append(I_2[num] - I_1[num])


        theta = []
        for i in range(num_samples):
            if source_angle > 90:
                theta.append(
                    (np.arccos(np.dot(P_1[i], P_2[i])
                               / (np.linalg.norm(P_1[i]) * np.linalg.norm(P_2[i]))) * (180 / np.pi)))
            else:
                theta.append(np.arccos(np.dot(P_1[i], P_2[i]) / (np.linalg.norm(P_1[i]) * np.linalg.norm(P_2[i]))) * (
                            180 / np.pi))
        energies = []
        for angle in theta:
            energies.append(compton_formula(angle))

        histogram_data, bin_edges = np.histogram(energies, bins=xaxis)
        histogram_data = np.append(histogram_data, 0)
        all_histogram_data.append(histogram_data)

        # colours = ['black', 'dimgray', 'dimgrey', 'gray', 'grey', 'darkgray', 'darkgrey', 'silver', 'lightgray', 'lightgrey']
        # fig, ax = plt.subplots()
        # plt.xlabel("Energy / keV")
        # plt.ylabel("Count")
        colours = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:pink', 'tab:olive', 'crimson']
        convolution_high = np.convolve(gaussian_detector_function_high, histogram_data, 'same')
        convolution_low = np.convolve(gaussian_detector_function_low, histogram_data, 'same')
        convolution = np.convolve(gaussian_detector_function, histogram_data, 'same')
        mean_energy.append(np.argmax(convolution))
        sd.append(cal_sd(convolution, max(convolution), mean_energy[-1]))
        sd_error_upper.append(cal_sd(convolution_high, max(convolution_high), mean_energy[-1]) - sd[-1])
        sd_error_lower.append(sd[-1] - cal_sd(convolution_low, max(convolution_low), mean_energy[-1]))
        max_count.append(max(convolution) * sd[-1])
        # plt.plot(xaxis, all_histogram_data[i], linestyle='-', color=colours[i], label=str(diam[i])+' x $10^{-4}$ m', alpha=0.8)

        #
        # for param in [mean_energy, max_count, sd]:
        #     fig_gaus, ax_gaus = plt.subplots()
        #     ax_gaus.scatter(diam, param)
    ax.plot(xaxis, histogram_data, linestyle='-', color=colours[j],
             label=str(r_tmax) + ' x $10^{-4}$ m', alpha=0.8)
    j += 1
    ax.legend(loc='best')
    max_count_err.append(np.std(max_count))
    mean_energy_err.append(np.std(mean_energy))
    sd_err.append(np.std(sd))
    sd_error_upper_all.append(np.average(sd_error_upper))
    sd_error_lower_all.append(np.average(sd_error_lower))
    max_count_all.append(np.average(max_count))
    mean_energy_err.append(np.average(mean_energy))
    sd_all.append(np.average(sd))
ax.legend(loc='best')
ax.set_ylabel("Count")
ax.set_xlabel("Energy / keV")
ax.set_title("Distribution of Energy due to Scattering Target")
plt.show()

# pickle.dump(all_histogram_data, open('histogram_data_rtmax', 'wb'))

# fig, ax = plt.subplots()
# plt.xlabel("Energy / keV")
# plt.ylabel("Count")
# #colours = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:pink', 'tab:olive', 'crimson']
# for i in range(int((stop-start)/step)):
#     plt.plot(xaxis, all_histogram_data[i], linestyle='-', color=colours[i], alpha=0.8)
# plt.show()

print(sd_all)
print(sd_err)
print(sd_error_lower_all)
print(sd_error_upper_all)


#print(dtype(x_1))
#print(x_1.dtype)
# print(x_1)
# print(y_1)
# print(z_1)
# print(x_2)
# print(y_2)
# print(z_2)
