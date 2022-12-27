import numpy as np


def average_power_decibel(signal: np.ndarray) -> float:
    return 10.0 * np.log10(average_power(signal))


def average_power(signal: np.ndarray) -> float:
    return np.average(np.abs(signal) ** 2)
