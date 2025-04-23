from cProfile import label
from data import load_data
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Number of samples
    K = 256

    # Desired probability of false alarm
    P_fa = 0.1

    # Variances of noise and signal
    noise_variance = 1
    signal_variance = 5

    # Create the approximate distributions for the test statistics
    h0_dist = norm(K*(noise_variance), K*(noise_variance)**2)
    h1_dist = norm(K*(noise_variance + signal_variance), K*(noise_variance + signal_variance)**2)

    # Calculate the threshold from desired probability of false alarm
    threshold = h0_dist.ppf(1-P_fa)

    # Load the data
    data = load_data("T8_numerical_experiment")

    # Calculate the test statistic
    z = np.sum(np.abs(data)**2, axis=0)

    # Split the data into two groups, one for each hypothesis
    z_h0 = np.copy(z)
    z_h1 = z

    detected = z > threshold

    z_h0[detected] = None
    z_h1[np.logical_not(detected)] = None

    # Plot the results
    plt.title("Result of Numerical Experiment")
    plt.axhline(threshold, color="grey", linestyle="--", label="$\\lambda'$")
    plt.plot(z_h1, "gx", label="PU Detected")
    plt.plot(z_h0, "rx", label="PU Not Detected")
    plt.legend(loc="center left")
    plt.grid()
    plt.savefig("numerical_experiment.png")
