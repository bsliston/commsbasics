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


def costas_loop(
    received_signal: np.ndarray,
    error_function,
    alpha: float = 0.005,
    beta: float = 0.001,
) -> np.ndarray:
    phase: float = 0.0
    freq: float = 0.0

    num_samples = len(received_signal)
    out = np.zeros(num_samples, dtype=complex)
    for i in range(num_samples):
        out[i] = received_signal[i] * np.exp(-1j * phase)

        # error = np.real(out[i]) * np.imag(out[i])
        error = error_function(out[i])

        freq += beta * error
        phase += freq + (alpha * error)

    return out


def costas_loop_bpsk(
    received_signal: np.ndarray,
    alpha: float = 0.005,
    beta: float = 0.001,
):
    return costas_loop(received_signal, _costas_loop_bpsk_error, alpha, beta)


def costas_loop_qpsk(
    received_signal: np.ndarray,
    alpha: float = 0.005,
    beta: float = 0.001,
):
    return costas_loop(received_signal, _costas_loop_qpsk_error, alpha, beta)


def _costas_loop_bpsk_error(sample: complex) -> float:
    return np.real(sample) * np.imag(sample)


def _costas_loop_qpsk_error(sample: complex) -> float:
    if sample.real > 0:
        a = 1.0
    else:
        a = -1.0
    if sample.imag > 0:
        b = 1.0
    else:
        b = -1.0
    return a * sample.imag - b * sample.real
