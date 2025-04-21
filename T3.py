import matplotlib.pyplot as plt
import sys
from typing import Callable
import numpy as np
from data import load_data
import math

if len(sys.argv) < 2:
    print("Usage: python T1.py <filename.mat>")
    sys.exit(1)

x = load_data(sys.argv[1])
N = len(x)

def generate_gaussian(mean: float, variance: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(x: np.ndarray) -> np.ndarray:
        return 1/(np.sqrt(2*np.pi*variance))*np.exp(-(mean-x)**2/(2*variance))

    return f


def generate_chi(k: int) -> Callable[[np.ndarray], np.ndarray]:
    def f(x: np.ndarray) -> np.ndarray:
        return 1/(2**(k/2) * math.gamma(k/2)) * x**(k/2 - 1) * np.exp(-x/2)

    return f

# Variance
var_real = np.var(x.real)
var_imag = np.var(x.imag)

var = (var_real + var_imag) / 2

# Histogram
hist_num_bins = 300
hist_range = (0, 30)

hist_bin_size = (hist_range[1] - hist_range[0]) / hist_num_bins

# Chi Squared Distribution
x_dist = np.linspace(*hist_range)
y_dist = N * hist_bin_size * generate_chi(2)(x_dist)

# Squaring
x_normalized = 2 * np.abs(x**2) / var

plt.clf()
plt.title("Signal Under $H_0$")
plt.xlabel("Amplitude")
plt.ylabel("Count")
plt.hist(x_normalized, bins=hist_num_bins, range=hist_range, label="Histogram")
plt.plot(x_dist, y_dist, label="Chi Squared Distribution")
plt.legend()
plt.grid()

plt.savefig("histogram.png")

# plt.plot(x, y.real)
# plt.title("Signal")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")

