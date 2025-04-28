from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Callable
import numpy as np
from data import load_data

# Change to dataset of choice
S = load_data("T1_data_Sk_Gaussian")
N = len(S)

def generate_signal(S: np.ndarray, N: int) -> Callable[[np.ndarray], np.ndarray]:
    def s(n: np.ndarray) -> np.ndarray:
        return 1/(np.sqrt(N))*sum(S[k]*np.exp(2j*np.pi*n*k/N) for k in range(0, N))

    return s

x_signal = np.arange(0, 1024, 1)
s = generate_signal(S, N)
y_signal = s(x_signal)

# Expected value
mean_real = np.mean(y_signal.real)
mean_imag = np.mean(y_signal.imag)
print("Expected value:", mean_real + 1j*mean_imag)

# Correlation
corr = np.mean(y_signal.real * y_signal.imag)
print("Correlation:", corr)

# Variance
var_real = np.var(y_signal.real)
var_imag = np.var(y_signal.imag)

# Histogram
hist_num_bins = 25
hist_range = (-2.5, 2.5)

hist_bin_size = (hist_range[1] - hist_range[0]) / hist_num_bins

# Gaussian Distribution
x_dist = np.linspace(-2.5, 2.5)
y_dist = N * hist_bin_size * norm(0, np.sqrt((var_real + var_imag)/2)).pdf(x_dist)

plt.clf()
plt.title("Real Part of Gaussian Signal")
plt.xlabel("Amplitude")
plt.ylabel("Count")
plt.hist(y_signal.real, bins=hist_num_bins, range=hist_range, label="Histogram")
plt.plot(x_dist, y_dist, label="Gaussian Distribution")
plt.legend()
plt.grid()

plt.savefig("histogram_real.png")

plt.clf()
plt.title("Imaginary Part of Gaussian Signal")
plt.xlabel("Amplitude")
plt.ylabel("Count")
plt.hist(y_signal.imag, bins=hist_num_bins, range=hist_range, label="Histogram")
plt.plot(x_dist, y_dist, label="Gaussian Distribution")
plt.legend()
plt.grid()

plt.savefig("histogram_imag.png")
