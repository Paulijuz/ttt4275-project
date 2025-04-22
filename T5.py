from cProfile import label
from scipy.stats import gamma, chi2
import matplotlib.pyplot as plt
import numpy as np
from T3 import signal_variance, noise_variance

def calculate_characteristics(k: int) -> tuple[np.ndarray, np.ndarray]:
    h0_dist = gamma(a=k, scale=noise_variance)
    h1_dist = gamma(a=k, scale=noise_variance + signal_variance)

    lower = (min(h0_dist.mean(), h1_dist.mean()) - 100)
    upper = (max(h0_dist.mean(), h1_dist.mean()) + 100)

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