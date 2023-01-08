"""This script encapsulates calls to emulate a basic communications channel.

This script is purposed for demonstration of existing modules and tools that
exist within this project.
"""
from typing import Tuple, Callable
import numpy as np
from communications_toolbox.transmit.data import random_data
from communications_toolbox.transmit.modulation import Modulation, BPSK
from communications_toolbox.transmit.pulse_shaping import raised_cosine_filter
from communications_toolbox.channel import (
    unity_noise,
    phase_delay_signal,
    frequency_shift_signal,
)
from communications_toolbox.receive.frequency_synch import (
    coarse_frequency_correction,
    fine_frequency_correction,
)
from communications_toolbox.receive.time_synch import (
    mueller_time_synch,
    align_signals,
)
from communications_toolbox.transformation import normalize_signal
from communications_toolbox.plot import ScatterPlotter, HeatmapPlotter


SignalTransform = Callable[[np.ndarray], np.ndarray]

import pdb


class DataModulator:
    """Data modulator for generating and demodulating signals."""

    def __init__(
        self,
        modulation: Modulation,
        signal_power_decibel: float,
        signal_filter: np.ndarray,
        data: np.ndarray,
    ) -> None:
        """Initializes data modulator.

        Args:
            modulation: Signal modulation type for modulating and demodulating
                signals.
            signal_power_decibel: Signal power in decibel to normalize signal
                generated to.
            signal_filter: Signal shaping filter.
            data: Data to modulate and demodulate.
        """
        self._modulation = modulation
        self._signal_power_decibel = signal_power_decibel
        self._signal_filter = signal_filter
        self._data = data

    def generate_signal_data(self) -> np.ndarray:
        """Modulates data into signal.

        Returns:
            Modulated signal.
        """
        signal = self._modulation.modulate_data(self._data)
        signal_shaped = np.convolve(signal, self._signal_filter)
        return normalize_signal(
            signal_shaped,
            self._signal_power_decibel,
        )

    def demodulate_signal_data(self, signal: np.ndarray) -> float:
        """Demodulates signal into modulated data and calculates error rate.

        Args:
            signal: Signal to demodulate.

        Returns:
            Bit error rate.
        """
        demodulated_data = self._modulation.demodulate_signal(signal)
        num_error_bits, num_bits_compared = bit_error(
            self._data, demodulated_data
        )
        return num_error_bits / float(num_bits_compared)


class ChannelEffects:
    """Channel effects tranformer for basic channel effects.

    Channel effects transformer for adding basic channel effects to
    parameterized signals.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        phase_delay: float,
        frequency_shift_hz: float,
    ) -> None:
        """Initializes channel effects transformer.

        Args:
            sample_rate_hz: Sample rate of signal.
            phase_delay: Phase delay in number of samples to delay signal by.
            frequency_shift_hz: Frequency to shift signal in hertz.
        """
        self._sample_rate_hz = sample_rate_hz
        self._phase_delay = phase_delay
        self._frequency_shift_hz = frequency_shift_hz

    def add_noise(self, signal) -> np.ndarray:
        """Adds unity noise to signal.

        Args:
            signal: Signal to add noise to.

        Returns:
            Signal with unity noise.
        """
        noise = unity_noise(len(signal))
        return signal + noise

    def delay_signal(self, signal: np.ndarray) -> np.ndarray:
        """Phase delays signal.

        Args:
            signal: Signal to phase delay

        Returns:
            Phase delayed signal.
        """
        return phase_delay_signal(signal, self._phase_delay)

    def frequency_shift_signal(self, signal: np.ndarray) -> np.ndarray:
        """Frequency shift signal.

        Args:
            signal: Signal to frequency shift.

        Returns:
            Frequency shifted signal.
        """
        return frequency_shift_signal(
            signal,
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
        time_synch_mu: float = 0.0,
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

        (
            self._noise_scatter,
            self._noise_delayed_scatter,
            self._noise_delayed_shifted_scatter,
            self._corrected_scatter,
        ) = self._init_scatter_plotters()

        self._phase = 0.0
        self._freq = 0.0

    def step(self) -> float:
        """Steps in communication playback generation.

        Steps in communication playback generation through generating signals,
        adding channel effects, and recovering and demodulating the effected
        signal.

        Returns:
            The demodulated bit error rate.
        """
        signal = self._data_modulator.generate_signal_data()

        # Add channel effects to signal.
        noise_signal = self._channel_effects.add_noise(signal)
        noise_delayed_signal = self._channel_effects.delay_signal(noise_signal)
        noise_delayed_shifted_signal = (
            self._channel_effects.frequency_shift_signal(noise_delayed_signal)
        )

        # Correct signal for alignment, frequency offset, and time
        # synchronization
        frequency_corrected_signal = coarse_frequency_correction(
            noise_delayed_shifted_signal, 2, 1e6
        )
        effect_signal_aligned, self._time_synch_mu = mueller_time_synch(
            frequency_corrected_signal,
            self._samples_per_symbol,
        )
        signal_corrected, self._phase, self._freq = fine_frequency_correction(
            effect_signal_aligned, 1e6
        )

        # Update plots for generated, effected, and recovered signals.
        self._plot_complex_scatter(
            normalize_signal(noise_signal, 0.0),
            self._noise_scatter,
            self._samples_per_symbol,
        )
        self._plot_complex_scatter(
            normalize_signal(noise_delayed_signal, 0.0),
            self._noise_delayed_scatter,
            self._samples_per_symbol,
        )
        self._plot_complex_scatter(
            normalize_signal(noise_delayed_shifted_signal, 0.0),
            self._noise_delayed_shifted_scatter,
            self._samples_per_symbol,
        )
        self._plot_complex_scatter(
            normalize_signal(signal_corrected, 0.0),
            self._corrected_scatter,
        )

        return self._data_modulator.demodulate_signal_data(signal_corrected)

    def _plot_complex_scatter(
        self, signal: np.ndarray, plotter: ScatterPlotter, skip: int = 1
    ) -> None:
        scatter_signal = np.array([signal[::skip].real, signal[::skip].imag]).T
        plotter.update(scatter_signal)

    def _plot_waterfall(self) -> None:
        return NotImplemented

    def _init_scatter_plotters(
        self, axis_limit: tuple = (-2.0, 2.0)
    ) -> Tuple[ScatterPlotter, ScatterPlotter, ScatterPlotter, ScatterPlotter]:
        """Initializes signal scatter plots.

        Args:
            axis_limit: Scatter plot axis limits for all initialized scatter
                plots.

        Returns:
            Signal plus noise scatter plot.
            Phase delayed signal plus noise scatter plot.
            Frequency shifted, phase delayed signal plus noise scatter plot.
            Corrected signal scatter plot.
        """
        noise_scatter = ScatterPlotter(
            "Signal Plus Noise",
            "In-Phase",
            "Quadrature-Phase",
            axis_limit,
            axis_limit,
        )
        noise_delayed_scatter = ScatterPlotter(
            "Phase Delayed Signal Plus Noise",
            "In-Phase",
            "Quadrature-Phase",
            axis_limit,
            axis_limit,
        )
        noise_delayed_shifted_scatter = ScatterPlotter(
            "Frequency Shifted, Phase Delayed Signal Plus Noise",
            "In-Phase",
            "Quadrature-Phase",
            axis_limit,
            axis_limit,
        )
        corrected_scatter = ScatterPlotter(
            "Corrected Signal",
            "In-Phase",
            "Quadrature-Phase",
            axis_limit,
            axis_limit,
        )

        return (
            noise_scatter,
            noise_delayed_scatter,
            noise_delayed_shifted_scatter,
            corrected_scatter,
        )


def bit_error(
    data: np.ndarray, demodulated_data: np.ndarray
) -> Tuple[int, int]:
    data_corrected, demodulated_data_corrected = align_signals(
        data, demodulated_data
    )
    data_bit_difference = np.abs(data_corrected - demodulated_data_corrected)
    num_bit_errors = int(data_bit_difference.sum())

    return num_bit_errors, len(data_bit_difference)


def main() -> None:
    sample_rate_hz: float = 1.0e6
    samples_per_symbol: int = 8
    num_bits: int = 2500
    num_taps: int = 101
    beta: float = 0.35

    signal_noise_ratio_decibel: float = 20.0
    phase_delay = 0.4  # * np.random.random()
    frequency_shift_hz = 1e3  # * (np.random.random() * 2 - 0.5)

    # Setup data generation and demodulation processing.
    modulation = BPSK(samples_per_symbol)
    data = random_data(num_bits)
    signal_filter = raised_cosine_filter(num_taps, beta, samples_per_symbol)
    data_modulator = DataModulator(
        modulation,
        signal_noise_ratio_decibel,
        signal_filter,
        data,
    )

    # Setup channel effects processsing.
    channel_effects = ChannelEffects(
        sample_rate_hz, phase_delay, frequency_shift_hz
    )
    playback = CommunicationPlayback(
        data_modulator, channel_effects, samples_per_symbol
    )

    while True:
        print(f"Bit Error Rate: {playback.step()}")


if __name__ == "__main__":
    main()
