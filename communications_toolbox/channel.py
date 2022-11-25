import numpy as np


def phase_delay_signal(signal: np.ndarray) -> np.ndarray:
    return NotImplemented


def frequency_shift_signal(signal: np.ndarray) -> np.ndarray:
    return NotImplemented


def unity_noise(size: int) -> np.ndarray:
    return (np.random.randn(size) + 1j * np.random.randn(size)) / np.sqrt(2)
