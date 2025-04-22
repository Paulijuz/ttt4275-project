import matplotlib.pyplot as plt
import numpy as np
from data import load_data
from scipy.stats import chi2

def calucalte_variance(x: np.ndarray) -> np.floating:
    real_varaiance = np.var(x.real)
    imag_varaiance = np.var(x.imag)

    return real_varaiance + imag_varaiance

data_s = load_data("T3_data_sigma_s")
data_w = load_data("T3_data_sigma_w")

signal_variance = calucalte_variance(data_s)
noise_variance = calucalte_variance(data_w)

def create_histogram(name: str, title: str, x: np.ndarray, var: np.floating, num_bins = 200, range = (0, 20)):
    N = len(x)
    bin_size = (range[1] - range[0]) / num_bins

    # Chi Squared Distribution
    x_dist = np.linspace(*range)
    y_dist = N * bin_size * chi2.pdf(x_dist, 2)

    # Normalization
    x_normalized = 2 * np.abs(x)**2 / var

    plt.clf()
    plt.title(title)
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    plt.hist(x_normalized, bins=num_bins, range=range, label="Histogram")
    plt.plot(x_dist, y_dist, label="Chi Squared Distribution")
    plt.legend()
    plt.grid()

    plt.savefig(f"{name}.png")

if __name__ == "__main__":
    print("Variance of signal:", signal_variance)
    print("Variance of noise:", noise_variance)

    x_h0 = load_data("T3_data_x_H0")
    x_h1 = load_data("T3_data_x_H1")

    create_histogram("histogram_h0", "Signal under $H_0$", x_h0, noise_variance)
    create_histogram("histogram_h1", "Signal under $H_1$", x_h1, noise_variance + signal_variance)