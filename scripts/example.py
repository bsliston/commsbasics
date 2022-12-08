import numpy as np
from communications_toolbox.transmit.data import random_data
from communications_toolbox.transmit.modulation import Modulation, BPSK
from communications_toolbox.transmit.pulse_shaping import raised_cosine_filter
from communications_toolbox.channel import (
    unity_noise,
    phase_delay_signal,
    frequency_shift_signal,
)
from communications_toolbox.receive.time_synch import mueller_time_synch
from communications_toolbox.utils import (
    average_power_decibel,
    normalize_signal,
)

import pdb


def main():
    sample_rate_hz: float = 250e3
    samples_per_symbol: int = 16

    transmit_signal = generate_transmit_signal(
        samples_per_symbol=samples_per_symbol
    )
    channel_signal = apply_channel_effects(transmit_signal, sample_rate_hz)

    signal_corrected, mu = mueller_time_synch(
        channel_signal, samples_per_symbol
    )
    print(mu)

    import matplotlib.pyplot as plt

    plt.plot(signal_corrected.real[-100:])
    plt.show()

    plt.scatter(
        signal_corrected.real[100:-100],
        signal_corrected.imag[100:-100],
    )
    plt.xlim(
        [
            -np.abs(signal_corrected.real).max(),
            np.abs(signal_corrected.real).max(),
        ]
    )
    plt.ylim(
        [
            -np.abs(signal_corrected.real).max(),
            np.abs(signal_corrected.real).max(),
        ]
    )
    plt.grid()

    plt.show()


def generate_transmit_signal(
    modulation: Modulation = BPSK(),
    signal_noise_ratio_decibel: float = 20.0,
    num_bits: int = 512,
    samples_per_symbol: int = 8,
    num_taps: int = 100,
    beta: float = 0.3,
) -> np.ndarray:
    data = random_data(num_bits)
    signal = modulation.modulate_data(data, samples_per_symbol)
    signal_filter = raised_cosine_filter(num_taps, beta, samples_per_symbol)
    signal_shaped = np.convolve(signal, signal_filter)
    signal_shaped_norm = normalize_signal(
        signal_shaped, signal_noise_ratio_decibel
    )

    return signal_shaped_norm


def apply_channel_effects(
    signal: np.ndarray,
    sample_rate_hz: float,
    phase_delay: float = 0.0,  # 2 * np.pi * 0.05,
    frequency_shift_hz: float = 0,
):
    noise = unity_noise(len(signal))
    signal_plus_noise = signal + noise
    signal_plus_noise_delayed = phase_delay_signal(
        signal_plus_noise, phase_delay
    )
    return frequency_shift_signal(
        signal_plus_noise_delayed, sample_rate_hz, frequency_shift_hz
    )


if __name__ == "__main__":
    main()
