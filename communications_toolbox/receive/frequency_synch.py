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


def fine_frequency_correction(
    received_signal: np.ndarray,
    sample_rate_hz: float = 1e6,
    phase: float = 0.0,
    freq: float = 0.0,
) -> np.ndarray:
    num_samples = len(received_signal)

    # These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
    alpha = 0.005
    beta = 0.001
    out = np.zeros(num_samples, dtype=np.complex)
    freq_log = []
    ii = 0
    for i in range(num_samples):
        out[i] = received_signal[i] * np.exp(
            -1j * phase
        )  # adjust the input sample by the inverse of the estimated phase offset
        error = np.real(out[i]) * np.imag(
            out[i]
        )  # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)

        # Advance the loop (recalc phase and freq offset)
        freq += beta * error
        freq_log.append(
            freq * sample_rate_hz / (2 * np.pi) / 8
        )  # convert from angular velocity to Hz for logging
        phase += freq + (alpha * error)

        # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
        while phase >= 2 * np.pi:
            phase -= 2 * np.pi
        while phase < 0:
            phase += 2 * np.pi
        ii += 1

    return out, phase, freq
