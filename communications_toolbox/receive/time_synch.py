from typing import Tuple
import numpy as np
from scipy.signal import resample_poly


def mueller_time_synch(
    signal: np.ndarray,
    samples_per_symbol: int,
    update_rate: float = 0.55,
    mu_offset: float = 0.0,
    upsample: int = 64,
) -> Tuple[np.ndarray, float]:
    signal = resample_poly(signal, upsample, 1)
    samples_per_symbol *= upsample

    signal_synch = np.zeros_like(signal)
    output_stream = np.zeros_like(signal)

    in_idx: int = 0
    out_idx: int = 2
    while out_idx < len(signal) and in_idx + samples_per_symbol < len(signal):
        mu_offset = mu_offset - np.floor(mu_offset)

        signal_synch[out_idx] = signal[in_idx + int(mu_offset)]
        output_stream[out_idx] = int(
            np.real(signal_synch[out_idx]) > 0
        ) + 1j * int(np.imag(signal_synch[out_idx]) > 0)

        stream_symmetric_difference = (
            output_stream[out_idx] - output_stream[out_idx - 2]
        ) * np.conj(signal_synch[out_idx - 1])
        signal_symmetric_difference = (
            signal_synch[out_idx] - signal_synch[out_idx - 2]
        ) * np.conj(output_stream[out_idx - 1])

        sample_difference = np.real(
            signal_symmetric_difference - stream_symmetric_difference
        )
        mu_offset += samples_per_symbol + update_rate * sample_difference

        in_idx += int(np.floor(mu_offset))
        out_idx += 1

    return signal_synch[2:out_idx], mu_offset
