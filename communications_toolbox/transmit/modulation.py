import numpy as np
import abc
from itertools import product
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
    def modulate_data(self, data: np.ndarray, samples_per_symbol: int):
        return NotImplemented

    @abc.abstractmethod
    def modulate(self, data: DataSequence) -> complex:
        return NotImplemented

    @abc.abstractmethod
    def demodulate(self, signal: complex) -> DataSequence:
        return NotImplemented


class BPSK(Modulation):
    def __init__(self):
        self._bit_to_symbol: BitToSymbolMapping = {}
        self._bit_to_symbol[(0,)] = (-1.0, 0.0)
        self._bit_to_symbol[(1,)] = (1.0, 0.0)
        self._symbol_to_bit: SymbolToBitMapping = _reverse_mapping(
            self._bit_to_symbol
        )

        self._bits_per_symbol: int = 1

    @property
    def bits_per_symbol(self) -> int:
        return self._bits_per_symbol

    def modulate_data(
        self,
        data: np.ndarray,
        samples_per_symbol: int,
    ) -> np.ndarray:
        num_symbols = len(data) // self._bits_per_symbol
        modulated_signal = np.zeros(
            (num_symbols * samples_per_symbol), dtype=np.complex64
        )
        for seq_i in range(num_symbols):
            data_seq_start = seq_i * self.bits_per_symbol
            data_seq_stop = (seq_i + 1) * self.bits_per_symbol
            data_seq = data[data_seq_start:data_seq_stop]

            mouldated_signal_index = seq_i * samples_per_symbol
            modulated_signal[mouldated_signal_index] = self.modulate(
                tuple(data_seq)
            )

        return modulated_signal

    def modulate(self, data: DataSequence) -> complex:
        symbol = self._bit_to_symbol[data]
        return _inphase_quadrature_to_complex(symbol)

    def demodulate(self, signal: complex) -> DataSequence:
        symbol = _complex_to_inphase_quadrature(signal)
        return self._symbol_to_bit[symbol]


def _inphase_quadrature_to_complex(
    coords: InphaseQuadratureCoordinates,
) -> complex:
    return coords[0] + 1j * coords[1]


def _complex_to_inphase_quadrature(
    signal: complex,
) -> InphaseQuadratureCoordinates:
    return (signal.real, signal.imag)


def _reverse_mapping(map: dict):
    return {v: k for k, v in map.items()}
