import numpy as np
from communications_toolbox.metrics import average_power


def normalize_signal(
    signal: np.ndarray, target_power_decibel: float
) -> np.ndarray:
    target_power_linear = 10.0 ** (target_power_decibel / 20.0)
    signal_norm = signal / (average_power(signal) ** 0.5)

    return target_power_linear * signal_norm


def short_time_fourier_transform(
    signal: np.ndarray, frequency_length: int
) -> np.ndarray:
    time_length = signal.size // frequency_length
    signal_reshaped = signal[: time_length * frequency_length].reshape(
        time_length, frequency_length
    )
    sxx = np.fft.fftshift(np.fft.fft(signal_reshaped, axis=1))
    return 10.0 * np.log10(np.abs(sxx))
