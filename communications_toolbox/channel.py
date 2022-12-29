import numpy as np


def phase_delay_signal(
    signal: np.ndarray, phase_delay: float, num_filter_samples: int = 101
) -> np.ndarray:
    # To phase delay the signal, a Sinc function and Hamming window are used
    # with configured phase delay offset to offset samples of the signal. The
    # hamming window convolved over the Sinc filter is to force the edges of
    # the filter to zero.
    filter_samples = np.arange(
        -num_filter_samples // 2, num_filter_samples // 2
    )
    filter = np.sinc(filter_samples - phase_delay)
    filter_with_hamming = filter * np.hamming(num_filter_samples)

    # The filter is normalized to not change the amplitude of the convolved
    # signal.
    filter_with_hamming /= np.sum(filter_with_hamming)
    return np.convolve(signal, filter)


def frequency_shift_signal(
    signal: np.ndarray, sample_rate_hz: float, frequency_shift_hz: float
) -> np.ndarray:
    sample_period = 1.0 / sample_rate_hz
    time = np.arange(0, sample_period * len(signal), sample_period)
    return signal * np.exp(1j * 2 * np.pi * frequency_shift_hz * time)


def unity_noise(size: int) -> np.ndarray:
    return (np.random.randn(size) + 1j * np.random.randn(size)) / np.sqrt(2)
