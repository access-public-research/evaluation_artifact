import numpy as np


def confidence_bins(confidence, num_bins=10):
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    return np.digitize(confidence, bins) - 1
