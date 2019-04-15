from ComptonScattering.read_calibrate import gaussian, compton_formula
import numpy as np

standard_deviation = compton_formula(45) * 2.9128283653924534 - 9.271293032027

gaussian_detector_function = gaussian(x, 1, 0, standard_deviation)

np.convolve(gaussian_detector_function, all_histogram_data[i])

