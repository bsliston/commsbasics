import numpy as np
from typing import Tuple, Mapping

from communications_toolbox.modulation.modulation import Modulation

DataSequence = Tuple[int, ...]
InphaseQuadratureCoordinates = Tuple[float, float]
BitToSymbolMapping = Mapping[DataSequence, InphaseQuadratureCoordinates]
SymbolToBitMapping = Mapping[InphaseQuadratureCoordinates, DataSequence]


class QAM(Modulation):
    def __init__(
        self,
        samples_per_symbol: int,
        symbols: InphaseQuadratureCoordinates,
        bit_to_symbol: BitToSymbolMapping,
    ):
        self._samples_per_symbol = samples_per_symbol
        self._symbols = symbols
        self._bit_to_symbol = bit_to_symbol
        self._symbol_to_bit: SymbolToBitMapping = _reverse_mapping(
            self._bit_to_symbol
        )

        self._bits_per_symbol: int = max(
            [len(bit) for bit in self._bit_to_symbol]
        )
        self._num_symbols = len(self._bit_to_symbol)

    @property
    def bits_per_symbol(self) -> int:
        return self._bits_per_symbol

    @property
    def num_symbols(self) -> int:
        return self._num_symbols

    def modulate_data(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        num_symbols = len(data) // self._bits_per_symbol
        modulated_signal = np.zeros(
            (num_symbols * self._samples_per_symbol), dtype=np.complex64
        )
        for seq_idx in range(num_symbols):
            data_seq_start = seq_idx * self._bits_per_symbol
            data_seq_stop = (seq_idx + 1) * self._bits_per_symbol
            data_seq = data[data_seq_start:data_seq_stop]

            mouldated_signal_index = seq_idx * self._samples_per_symbol
            modulated_signal[mouldated_signal_index] = self._modulate(
                tuple(data_seq)
            )

        return modulated_signal

    def demodulate_signal(self, signal: np.ndarray) -> np.ndarray:
        demodulated_data = np.zeros(len(signal) * self._bits_per_symbol)
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


class BPSK(QAM):
    def __init__(self, samples_per_symbol: int):
        symbols: InphaseQuadratureCoordinates = ((-1.0, 0.0), (1.0, 0.0))
        bit_to_symbol: BitToSymbolMapping = {}
        bit_to_symbol[(0,)] = symbols[0]
        bit_to_symbol[(1,)] = symbols[1]
        super(BPSK, self).__init__(samples_per_symbol, symbols, bit_to_symbol)


class QPSK(QAM):
    def __init__(self, samples_per_symbol: int):
        symbols: InphaseQuadratureCoordinates = (
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
        )
        bit_to_symbol: BitToSymbolMapping = {}
        bit_to_symbol[(0, 0)] = symbols[0]
        bit_to_symbol[(0, 1)] = symbols[1]
        bit_to_symbol[(1, 0)] = symbols[2]
        bit_to_symbol[(1, 1)] = symbols[3]
        super(QPSK, self).__init__(samples_per_symbol, symbols, bit_to_symbol)


def _inphase_quadrature_to_complex(
    coords: InphaseQuadratureCoordinates,
) -> complex:
    return coords[0] + 1j * coords[1]


def _reverse_mapping(data: Mapping) -> Mapping:
    return {v: k for k, v in data.items()}
