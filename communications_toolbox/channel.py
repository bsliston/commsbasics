import numpy as np


def phase_delay_signal(
    signal: np.ndarray, phase_delay: float, shift_real_only: bool = False
) -> np.ndarray:
    if shift_real_only:
        delay = np.cos(phase_delay) + 1j * 0.0
    else:
        delay = np.cos(phase_delay) + 1j * np.sin(phase_delay)
    return signal * delay


def frequency_shift_signal(
    signal: np.ndarray, sample_rate_hz: float, frequency_shift_hz: float
) -> np.ndarray:
    sample_period = 1.0 / sample_rate_hz
    time = np.arange(0, sample_period * len(signal), sample_period)
    return signal * np.exp(1j * 2 * np.pi * frequency_shift_hz * time)


def unity_noise(size: int) -> np.ndarray:
    return (np.random.randn(size) + 1j * np.random.randn(size)) / np.sqrt(2)
