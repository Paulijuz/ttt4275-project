from data import load_data
import numpy as np
import matplotlib.pyplot as plt 

data = load_data("T8_numerical_experiment")

data = np.transpose(data)

z = np.array([np.sum(np.abs(samples)**2) for samples in data])

plt.plot(z, "x")
plt.savefig("a.png")
