from scipy.stats import chi2
import matplotlib.pyplot as plt
import numpy as np
from T3 import signal_variance, noise_variance

def calculate_characteristics(K: int) -> tuple[np.ndarray, np.ndarray]:
    h0_dist = chi2(2*K, scale=noise_variance/2)
    h1_dist = chi2(2*K, scale=(noise_variance + signal_variance)/2)

    h0_mean = h0_dist.mean()
    h1_mean = h1_dist.mean()

    lower = (min(h0_mean, h1_mean) - 100)
    upper = (max(h0_mean, h1_mean) + 100)

    l = np.linspace(lower, upper, 1000)

    p_fa = 1-h0_dist.cdf(l)
    p_d = 1-h1_dist.cdf(l)

    return p_fa, p_d

if __name__ == "__main__":
    plt.plot([0, 1], "--", color="grey", label="$P_{fa}$ = $P_{d}$")

    for i in range(0, 6):
        k = 1 << i

        p_fa, p_d = calculate_characteristics(k)

        plt.plot(p_fa, p_d, label=f"$k={k}$")

    plt.title("ROCs for Different Number of Samples ($k$)")
    plt.xlabel("$P_{fa}$")
    plt.ylabel("$P_d$")
    plt.legend()
    plt.grid()
    plt.savefig("roc.png")