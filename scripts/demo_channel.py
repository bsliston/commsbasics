"""This script encapsulates calls to emulate a basic communications toolbox.

This script is purposed for demonstration of existing modules and tools that
exist within this project.
"""
import numpy as np
from typing import Tuple, Callable
from scipy.signal import correlation_lags
from communications_toolbox.transmit.data import random_data
from communications_toolbox.transmit.modulation import Modulation, BPSK
from communications_toolbox.transmit.pulse_shaping import raised_cosine_filter
from communications_toolbox.channel import (
    unity_noise,
    phase_delay_signal,
    frequency_shift_signal,
)
from communications_toolbox.receive.time_synch import (
    mueller_time_synch,
    align_signals,
)
from communications_toolbox.transformation import normalize_signal
from communications_toolbox.plot import ScatterPlotter, HeatmapPlotter


import pdb
import matplotlib.pyplot as plt

SignalTransform = Callable[[np.ndarray], np.ndarray]


class DataModulator:
    def __init__(
        self,
        modulation: Modulation,
        samples_per_symbol: int,
        signal_noise_ratio_decibel: float,
        signal_filter: np.ndarray,
        data: np.ndarray,
    ) -> None:
        self._modulation = modulation
        self._samples_per_symbol = samples_per_symbol
        self._signal_noise_ratio_decibel = signal_noise_ratio_decibel
        self._signal_filter = signal_filter
        self._data = data

    def generate_signal_data(self) -> np.ndarray:
        signal = self._modulation.modulate_data(
            self._data, self._samples_per_symbol
        )
        signal_shaped = np.convolve(signal, self._signal_filter)
        return normalize_signal(
            signal_shaped,
            self._signal_noise_ratio_decibel,
        )

    def demodulate_signal_data(self, signal: np.ndarray) -> float:
        demodulated_data = self._modulation.demodulate_signal(signal)
        num_error_bits, num_bits_compared = bit_error(
            self._data, demodulated_data
        )
        return num_error_bits / float(num_bits_compared)


class ChannelEffects:
    def __init__(
        self,
        sample_rate_hz: float,
        phase_delay: float,
        frequency_shift_hz: float,
    ) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._phase_delay = phase_delay
        self._frequency_shift_hz = frequency_shift_hz

    def transform(self, signal: np.ndarray) -> np.ndarray:
        noise = unity_noise(len(signal))
        signal_plus_noise = signal + noise
        signal_plus_noise_delayed = phase_delay_signal(
            signal_plus_noise,
            self._phase_delay,
        )
        return frequency_shift_signal(
            signal_plus_noise_delayed,
            self._sample_rate_hz,
            self._frequency_shift_hz,
        )


class CommunicationPlayback:
    """Communication playback for emulating basic digital communication.

    Communication playback for emulating basic digital communication
    modulation, channel, and demodulation.
    """

    def __init__(
        self,
        data_modulator: DataModulator,
        channel_effects: ChannelEffects,
        samples_per_symbol: int,
        time_synch_mu: float,
    ) -> None:
        """Initializes communication playback.

        Args:
            data_modulator:
            channel_effects:
            samples_per_symbol:
            time_synch_mu:
        """
        self._data_modulator = data_modulator
        self._channel_effects = channel_effects
        self._samples_per_symbol = samples_per_symbol
        self._time_synch_mu = time_synch_mu

        self.plotter = ScatterPlotter(
            "",
            "",
            "",
            (-2.0, 2.0),
            (-2.0, 2.0),
        )
        self.plotter2 = ScatterPlotter(
            "",
            "",
            "",
            (-2.0, 2.0),
            (-2.0, 2.0),
        )

    def step(self) -> None:
        """_summary_

        Returns:
            _type_: _description_
        """
        signal = self._data_modulator.generate_signal_data()
        effect_signal = self._channel_effects.transform(signal)

        _, effect_signal_aligned = align_signals(signal, effect_signal)
        signal_corrected, self._time_synch_mu = mueller_time_synch(
            effect_signal_aligned,
            self._samples_per_symbol,
        )

        bit_error_rate = self._data_modulator.demodulate_signal_data(
            signal_corrected
        )

        plot_effect_signal = normalize_signal(effect_signal, 0.0)
        plot_effect_signal = np.array(
            [
                plot_effect_signal[:: self._samples_per_symbol].real,
                plot_effect_signal[:: self._samples_per_symbol].imag,
            ]
        ).T
        self.plotter.update(plot_effect_signal)

        plot_corrected_signal = normalize_signal(signal_corrected, 0.0)
        plot_corrected_signal = np.array(
            [
                plot_corrected_signal.real,
                plot_corrected_signal.imag,
            ]
        ).T
        self.plotter2.update(plot_corrected_signal)

        print(bit_error_rate)


def bit_error(
    data: np.ndarray, demodulated_data: np.ndarray
) -> Tuple[int, int]:
    data_corrected, demodulated_data_corrected = align_signals(
        data, demodulated_data
    )
    data_bit_difference = np.abs(data_corrected - demodulated_data_corrected)
    num_bit_errors = int(data_bit_difference.sum())

    return num_bit_errors, len(data_bit_difference)


def main():
    sample_rate_hz: float = 100e3
    samples_per_symbol: int = 16
    num_bits: int = 128
    num_taps: int = 100
    beta: float = 0.3

    signal_noise_ratio_decibel: float = 15.0
    phase_delay = 2.0
    frequency_shift_hz = 0 * (np.random.random() * 2 - 0.5)

    # Setup data generation and demodulation processing.
    modulation = BPSK()
    data = random_data(num_bits)
    signal_filter = raised_cosine_filter(num_taps, beta, samples_per_symbol)
    data_modulator = DataModulator(
        modulation,
        samples_per_symbol,
        signal_noise_ratio_decibel,
        signal_filter,
        data,
    )

    # Setup channel effects processsing.
    channel_effects = ChannelEffects(
        sample_rate_hz, phase_delay, frequency_shift_hz
    )
    playback = CommunicationPlayback(
        data_modulator, channel_effects, samples_per_symbol, 0.0
    )

    while True:
        playback.step()


if __name__ == "__main__":
    main()
