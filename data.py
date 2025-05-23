import os
from typing import Literal
import scipy.io
import numpy as np

DataName = Literal[
    "T1_data_Sk_BPSK",
    "T1_data_Sk_Gaussian",
    "T3_data_sigma_s",
    "T3_data_sigma_w",
    "T3_data_x_H0",
    "T3_data_x_H1",
    "T8_numerical_experiment"
]

script_folder = os.path.dirname(__file__)
data_folder = os.path.join(script_folder, 'data/')

def load_data(filename: DataName, squeeze: bool = True) -> np.ndarray:
    file_path = data_folder + filename + ".mat"
    mat_data = scipy.io.loadmat(file_path)

    for name, value in mat_data.items():
        if name.startswith("__"):
            continue

        return np.squeeze(value) if squeeze else value
    
    raise ValueError(f"No valid data found in the file \"{filename}\".")
