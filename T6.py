from scipy.stats import norm, gamma
import numpy as np
import matplotlib.pyplot as plt

def create_characteristics_plot(K: int = 1):
    # Variances of noise and signal
    noise_variance = 1
    signal_variance = 1

    # Create the distributions for the test statistics
    h0_dist = gamma(a=K, scale=noise_variance)
    h1_dist = gamma(a=K, scale=noise_variance + signal_variance)

    # Create the approximate distributions for the test statistics
    mu_0 = K*(noise_variance)
    mu_1 = K*(noise_variance + signal_variance)

    sigma_0 = np.sqrt(K*(noise_variance)**2)
    sigma_1 = np.sqrt(K*(noise_variance + signal_variance)**2)

    h0_dist_approx = norm(mu_0, sigma_0)
    h1_dist_approx = norm(mu_1, sigma_1)

    # Plot P_fa and P_d as a function of the threashold for both the actual and approximate distributions
    median_fa = h0_dist.ppf(0.5)
    median_d = h1_dist.ppf(0.5)

    margin_fa = 4*np.sqrt(K)
    margin_d = 8*np.sqrt(K)

    x0_fa = median_fa - margin_fa
    x1_fa = median_fa + margin_fa
    x0_d = median_d - margin_d
    x1_d = median_d + margin_d

    x0 = min(x0_fa, x0_d)
    x1 = max(x1_fa, x1_d)

    x_fa = np.linspace(x0, x1, 1024)
    x_d = np.linspace(x0, x1, 1024)

    p_fa = 1-h0_dist.cdf(x_fa)
    p_d = 1-h1_dist.cdf(x_d)

    p_fa_approx = 1-h0_dist_approx.cdf(x_fa)
    p_d_approx = 1-h1_dist_approx.cdf(x_d)

    plt.clf()

    plt.subplot(2, 1, 1)
    plt.title(f"$P_{{fa}}$ and $P_d$ as Functions of $\\lambda$ with $K$={K}")
    plt.plot(x_fa, p_fa,              label="$P_{fa}$ (actual)")
    plt.plot(x_fa, p_fa_approx, "--", label="$P_{fa}$ (approximate)")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(x_d, p_d,              label="$P_d$ (actual)")
    plt.plot(x_d, p_d_approx, "--", label="$P_d$ (approximate)")
    plt.legend()
    plt.grid()

    plt.savefig(f"p_fa_p_d_k={K}.png")

if __name__ == "__main__":
    for i in range(0, 4):
        K = 2**(2*i)
        create_characteristics_plot(K)