from data import load_data
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def run_numerical_experiment(P_fa: float = 0.1, K: int = 256):
    # Variances of noise and signal
    noise_variance = 1
    signal_variance = 5

    # Create the approximate distributions for the test statistics
    mu_0 = K*(noise_variance)
    mu_1 = K*(noise_variance + signal_variance)

    sigma_0 = np.sqrt(K*(noise_variance)**2)
    sigma_1 = np.sqrt(K*(noise_variance + signal_variance)**2)

    h0_dist = norm(mu_0, sigma_0)
    h1_dist = norm(mu_1, sigma_1)

    # Calculate the threshold from desired probability of false alarm
    threshold = h0_dist.ppf(1-P_fa)

    # Load the data
    data = load_data("T8_numerical_experiment")

    # Calculate the test statistic
    z = np.sum(np.abs(data[:K, :])**2, axis=0)

    # Split the data into two groups, one for each hypothesis
    z_h0 = np.copy(z)
    z_h1 = np.copy(z)

    detected = z > threshold

    z_h0[detected] = None
    z_h1[np.logical_not(detected)] = None

    # Plot the results
    plt.figure(dpi=200)

    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_ylim(2e2, 2e3)

    plt.title(f"Detection Result with $P_{{fa}} = {P_fa}$")
    plt.axhline(threshold, color="grey", linestyle="--", label="$\\lambda'$")
    plt.plot(z_h1, "go", label="PU Detected")
    plt.plot(z_h0, "rx", label="PU Not Detected")
    plt.ylabel("Test Statistic")
    plt.xlabel("Time Step")
    plt.legend(loc="center left")
    plt.grid()
    plt.savefig(f"numerical_experiment_P_fa={P_fa}.png")

    for i in range(len(z)):
        if detected[i]:
            print(f"{i:>2}: True")
        else:
            print(f"{i:>2}: False")
    
    print("Number detected:", np.sum(detected))
    print("Number mpt detected:", np.sum(np.logical_not(detected)))

if __name__ == "__main__":
    for P_fa in [0.1, 0.01]:
        run_numerical_experiment(P_fa)