from .read_calibrate import compton_formula
import numpy as np
import matplotlib.pyplot as plt

def plot():
    x_axis = np.arange(-180, 180, 0.1)
    compton = []
    for i in x_axis:
        compton.append(compton_formula(i))

    plt.plot(x_axis, compton)
    plt.show()