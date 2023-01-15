import numpy as np
import abc


class Modulation:
    @property
    @abc.abstractmethod
    def bits_per_symbol(self) -> int:
        return NotImplemented

    @property
    @abc.abstractmethod
    def num_symbols(self) -> int:
        return NotImplemented

    @abc.abstractmethod
    def modulate_data(self, data: np.ndarray) -> np.ndarray:
        return NotImplemented

    @abc.abstractmethod
    def demodulate_signal(self, signal: np.ndarray) -> np.ndarray:
        return NotImplemented
