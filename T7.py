from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Desired probability of false alarm and detection
    P_fa = 0.1
    P_d = 0.9

    # Variances of noise and signal
    noise_variance = 1
    signal_variance = 1

    # Number of samples
    K = ((noise_variance*norm.ppf(1-P_fa) - (noise_variance + signal_variance)*norm.ppf(1-P_d))/signal_variance)**2
    print("Theoretic number of samples:", K)
    K = np.ceil(K)
    print("Practical number of samples:", K)

    # Create the approximate distributions for the test statistics
    mu_0 = K*(noise_variance)
    mu_1 = K*(noise_variance + signal_variance)

    sigma_0 = np.sqrt(K*(noise_variance)**2)
    sigma_1 = np.sqrt(K*(noise_variance + signal_variance)**2)

    h0_dist = norm(mu_0, sigma_0)
    h1_dist = norm(mu_1, sigma_1)

    l_fa = h0_dist.ppf(1-P_fa)
    l_d = h1_dist.ppf(1-P_d)

    l2_fa = sigma_0*norm.ppf(1-P_fa) + mu_0
    l2_d = sigma_1*norm.ppf(1-P_d) + mu_1
    
    print(l_fa, l_d)