import numpy as np
from communications_toolbox.transmit.data import random_data
from communications_toolbox.transmit.modulation import BPSK
from communications_toolbox.transmit.pulse_shaping import raised_cosine_filter
from communications_toolbox.channel import unity_noise
from communications_toolbox.utils import (
    average_power_decibel,
    normalize_signal,
)


def main():
    freq_sampling_hz: float = 250e3
    signal_noise_ratio_decibel: float = 10.0
    num_bits: int = 1024
    samples_per_symbol: int = 16
    num_taps: int = 100
    beta: float = 0.05

    modulation = BPSK()

    data = random_data(num_bits)
    signal = modulation.modulate_data(data, samples_per_symbol)
    filter = raised_cosine_filter(num_taps, beta, samples_per_symbol)
    signal_shaped = np.convolve(signal, filter)

    signal_shaped_norm = normalize_signal(
        signal_shaped, signal_noise_ratio_decibel
    )

    noise = unity_noise(len(signal_shaped))
    signal_plus_noise = signal_shaped_norm + noise


if __name__ == "__main__":
    main()
