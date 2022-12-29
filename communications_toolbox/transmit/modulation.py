import numpy as np
import abc
from typing import Tuple, Mapping

DataSequence = Tuple[int, ...]
InphaseQuadratureCoordinates = Tuple[float, float]
BitToSymbolMapping = Mapping[DataSequence, InphaseQuadratureCoordinates]
SymbolToBitMapping = Mapping[InphaseQuadratureCoordinates, DataSequence]


class Modulation:
    @property
    @abc.abstractmethod
    def bits_per_symbol(self) -> int:
        return NotImplemented

    @abc.abstractmethod
    def modulate_data(self, data: np.ndarray) -> np.ndarray:
        return NotImplemented

    @abc.abstractmethod
    def demodulate_signal(self, signal: np.ndarray) -> np.ndarray:
        return NotImplemented


class BPSK(Modulation):
    def __init__(self, samples_per_symbol: int):
        self._samples_per_symbol = samples_per_symbol

        self._bit_to_symbol: BitToSymbolMapping = {}
        self._symbols: InphaseQuadratureCoordinates = ((-1.0, 0.0), (1.0, 0.0))
        self._bit_to_symbol[(0,)] = self._symbols[0]
        self._bit_to_symbol[(1,)] = self._symbols[1]
        self._symbol_to_bit: SymbolToBitMapping = _reverse_mapping(
            self._bit_to_symbol
        )

        self._bits_per_symbol: int = max(
            [len(bit) for bit in self._bit_to_symbol]
        )

    @property
    def bits_per_symbol(self) -> int:
        return self._bits_per_symbol

    def modulate_data(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        num_symbols = len(data) // self._bits_per_symbol
        modulated_signal = np.zeros(
            (num_symbols * self._samples_per_symbol), dtype=np.complex64
        )
        for seq_idx in range(num_symbols):
            data_seq_start = seq_idx * self.bits_per_symbol
            data_seq_stop = (seq_idx + 1) * self.bits_per_symbol
            data_seq = data[data_seq_start:data_seq_stop]

            mouldated_signal_index = seq_idx * self._samples_per_symbol
            modulated_signal[mouldated_signal_index] = self._modulate(
                tuple(data_seq)
            )

        return modulated_signal

    def demodulate_signal(self, signal: np.ndarray) -> np.ndarray:
        demodulated_data = np.zeros(len(signal))
        for signal_idx, signal_symbol in enumerate(signal):
            data_start_idx = signal_idx * self._bits_per_symbol
            data_stop_idx = (signal_idx + 1) * self._bits_per_symbol
            demodulated_data[data_start_idx:data_stop_idx] = self._demodulate(
                signal_symbol
            )

        return demodulated_data

    def _modulate(self, data: DataSequence) -> complex:
        symbol = self._bit_to_symbol[data]
        return _inphase_quadrature_to_complex(symbol)

    def _demodulate(self, signal: complex) -> int:
        nearest_symbol = self._signal_nearest_symbol_mapping(signal)
        return self._symbol_to_bit[nearest_symbol]

    def _signal_nearest_symbol_mapping(self, signal: complex) -> complex:
        signal_symbol_distances = np.zeros(len(self._symbols))
        for symbol_idx, symbol in enumerate(self._symbols):
            complex_symbol = _inphase_quadrature_to_complex(symbol)
            signal_symbol_distances[symbol_idx] = abs(signal - complex_symbol)

        closest_symbol_idx = np.argmin(signal_symbol_distances)
        return self._symbols[closest_symbol_idx]


def _inphase_quadrature_to_complex(
    coords: InphaseQuadratureCoordinates,
) -> complex:
    return coords[0] + 1j * coords[1]


def _reverse_mapping(map: dict):
    return {v: k for k, v in map.items()}
