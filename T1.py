import matplotlib.pyplot as plt
import sys
from typing import Callable
import numpy as np
from data import load_data

if len(sys.argv) < 2:
    print("Usage: python script.py <filename.mat>")
    sys.exit(1)

S = load_data(sys.argv[1])

def generate_signal(S: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    N = len(S)

    def s(n: np.ndarray) -> np.ndarray:
        return 1/(np.sqrt(N))*sum(S[k]*np.exp(2j*np.pi*n*k/N) for k in range(0, N))

    return s

x = np.arange(0, 1024, 1)
s = generate_signal(np.array([0, 1]))
y = s(x)

plt.title("Signal Histograms")
plt.subplot(2, 1, 1)
plt.xlabel("Amplitude")
plt.ylabel("Frequency")
plt.hist(y.real, bins=100)
plt.grid()

plt.subplot(2, 1, 2)
plt.xlabel("Amplitude")
plt.ylabel("Frequency")
plt.hist(y.imag, bins=100)
plt.grid()

# Expected value
print("Expected value:")
print("\tReal:", np.mean(y.real))
print("\tImaginary:", np.mean(y.imag))

# Correlation
print("Covariance:", np.mean(y.real * y.imag))

# plt.plot(x, y.real)
# plt.title("Signal")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
plt.tight_layout()

plt.show()

