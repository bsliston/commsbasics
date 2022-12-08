import numpy as np


def phase_delay_signal(signal: np.ndarray, phase_delay: float) -> np.ndarray:
    delay = np.exp(1j * phase_delay)
    return signal * delay


def frequency_shift_signal(
    signal: np.ndarray, sample_rate_hz: float, frequency_shift_hz: float
) -> np.ndarray:
    sample_period = 1.0 / sample_rate_hz
    time = np.arange(0, sample_period * len(signal), sample_period)
    return signal * np.exp(1j * 2 * np.pi * frequency_shift_hz * time)


def unity_noise(size: int) -> np.ndarray:
    return (np.random.randn(size) + 1j * np.random.randn(size)) / np.sqrt(2)
