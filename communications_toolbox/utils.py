import numpy as np


def normalize_signal(
    signal: np.ndarray, target_power_decibel: float
) -> np.ndarray:
    target_power_linear = 10.0 ** (target_power_decibel / 20.0)
    signal_norm = signal / (average_power(signal) ** 0.5)

    return target_power_linear * signal_norm


def average_power_decibel(signal: np.ndarray) -> float:
    return 10.0 * np.log10(average_power(signal))


def average_power(signal: np.ndarray) -> float:
    return np.average(np.abs(signal) ** 2)
