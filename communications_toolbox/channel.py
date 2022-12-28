import numpy as np


def phase_delay_signal(signal: np.ndarray, phase_delay: float) -> np.ndarray:
    N = 21  # number of taps
    n = np.arange(-N // 2, N // 2)  # ...-3,-2,-1,0,1,2,3...
    h = np.sinc(n - phase_delay)  # calc filter taps
    h *= np.hamming(
        N
    )  # window the filter to make sure it decays to 0 on both sides
    h /= np.sum(
        h
    )  # normalize to get unity gain, we don't want to change the amplitude/power
    return np.convolve(signal, h)  # apply filter


def frequency_shift_signal(
    signal: np.ndarray, sample_rate_hz: float, frequency_shift_hz: float
) -> np.ndarray:
    sample_period = 1.0 / sample_rate_hz
    time = np.arange(0, sample_period * len(signal), sample_period)
    return signal * np.exp(1j * 2 * np.pi * frequency_shift_hz * time)


def unity_noise(size: int) -> np.ndarray:
    return (np.random.randn(size) + 1j * np.random.randn(size)) / np.sqrt(2)
