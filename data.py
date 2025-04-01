import scipy.io
import numpy as np

def load_data(filename: str) -> np.ndarray:
    mat_data = scipy.io.loadmat(filename)

    for name, value in mat_data.items():
        if name.startswith('__'):
            continue

        return value.flatten()
    
    raise ValueError(f"No valid data found in the file \"{filename}\".")
