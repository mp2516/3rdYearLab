from .read_calibrate import compton_formula
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from .read_calibrate import lorentzian, gaussian
from scipy.optimize import curve_fit

max_radius = 125
d = 3000
s = 3000
detector_radius = 0
num_samples = 1000000
#TODO: Measure these distances

def y(radius, angle):
    return np.sqrt(s**2 + radius**2 - 2 * s * radius * np.cos(2.5*np.pi - angle))


def a(radius, angle):
    return np.sqrt(radius**2 + d**2 - 2 * radius * d * np.cos(angle))


def b(radius, angle):
    return np.sqrt(detector_radius**2 + a(radius, angle)**2 - 2 * a(radius, angle) * detector_radius * np.cos(np.pi/2 - theta_a(radius, angle)))


def theta_a(radius, angle):
    return np.arcsin((radius * np.sin(angle)) / a(radius, angle))


def theta(radius, angle):
    return np.arcsin((detector_radius * np.sin(np.pi/2 - theta_a(radius, angle))) / b(radius, angle))


def theta_s(radius, angle):
    return np.arcsin((s * np.sin(1.25*np.pi - angle)) / y(radius, angle))


def phi(radius, angle):
    return angle - theta(radius, angle) + theta_a(radius, angle) - theta_s(radius, angle)

# radius = 0.01
# angle = np.pi / 2

# print ("theta_s {}".format(theta_s(radius, angle)))
# print ("theta {}".format(theta(radius, angle)))
# print ("theta_a {}".format(theta_a(radius, angle)))
# print ("y {}".format(y(radius, angle)))
# print ("a {}".format(a(radius, angle)))
# print("b {}".format(b(radius, angle)))
# print("phi {}".format(phi(radius, angle)))

radii = []


phi_values = []
angle = np.random.uniform(low=0, high=np.pi/2, size=num_samples)
radius = np.random.uniform(low=0, high=max_radius, size=num_samples)
for num in trange(num_samples):
    phi_values.append((phi(radius[num], angle[num])*180) / np.pi)

phi_values_reflected = []
for phi_num in trange(len(phi_values)):
    phi_values.append(((phi_values[phi_num] - 45) * -1) + 45)

print(phi_values_reflected)
print(phi_values)
# phi_values.append([phi_value_reflected for phi_value_reflected in phi_values_reflected])
print(len(phi_values))
# phi_values.append(phi_values)

energies = []
for angle in phi_values:
    energies.append(compton_formula(angle))

# bins = np.linspace(-1, 360, 1)
xaxis = np.arange(470, 490, 0.2)

#plt.hist(phi_values, bins=1000)
histogram_data, bin_edges = np.histogram(energies, bins=xaxis)

#plt.hist(energies, bins=1000)
#plt.show()
histogram_data=np.append(histogram_data, 0)
print(len(histogram_data))
print(len(xaxis))

popt_elem, pcov_elem = curve_fit(lorentzian, xaxis, histogram_data, p0=(50, 460, 10))
perr_elem = np.sqrt(np.diag(pcov_elem))
popt_elem_gaus, pcov_elem_gaus = curve_fit(gaussian, xaxis, histogram_data, p0=(50, 460, 10))

print("\nHeight {} +/ {}"
          "\nMean {} +/- {}"
          "\nSigma {} +/- {}".format(popt_elem[0], perr_elem[0], popt_elem[1], perr_elem[1], popt_elem[2],
                                    perr_elem[2]))
fig, ax = plt.subplots()
plt.xlabel("Angle")
plt.ylabel("Count")
# plt.title(plot_title)
ax.legend(["\nHeight = {} $\pm$ {}"
            "\nMean = {} $\pm$ {} keV"
            "\nSigma = {} $\pm$ {} keV".format(round(popt_elem[0], 1), round(perr_elem[0], 1), round(popt_elem[1], 1),
                                           round(perr_elem[1], 1), round(popt_elem[2], 1), round(perr_elem[2], 1))])
plt.plot(xaxis, histogram_data)
ax.plot(xaxis, lorentzian(xaxis, *popt_elem), 'ro:', label='fit', markersize=0.01)
ax.plot(xaxis, gaussian(xaxis, *popt_elem_gaus), 'go:', label='fit', markersize=0.01)
plt.show()
