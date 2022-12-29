import numpy as np


def coarse_frequency_correction(
    signal: np.ndarray, symbol_rate: int, sample_rate_hz: float
) -> np.ndarray:
    """Performs coarse frequency correction.

    Args:
        signal: Signal to shift in frequency.
        symbol_rate: Symbol rate of signal.
        sample_rate_hz: Sample rate of signal.

    Returns:
        Coarse frequency shifted and corrected signal.
    """
    clustered_symbol_signal = signal**symbol_rate
    signal_power_spectral_density = np.abs(
        np.fft.fftshift(np.fft.fft(clustered_symbol_signal))
    )
    signal_spectral_density_frequency_hz = np.linspace(
        -sample_rate_hz / 2.0,
        sample_rate_hz / 2.0,
        len(signal_power_spectral_density),
    )

    frequency_offset_hz = (
        signal_spectral_density_frequency_hz[
            np.argmax(signal_power_spectral_density)
        ]
        / symbol_rate
    )

    sample_period_sec = 1.0 / sample_rate_hz
    time = np.arange(0, sample_period_sec * len(signal), sample_period_sec)
    return signal * np.exp(1j * 2 * np.pi * -frequency_offset_hz * time)
