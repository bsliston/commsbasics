import numpy as np
from communications_toolbox.metrics import average_power


def normalize_signal(
    signal: np.ndarray, target_power_decibel: float
) -> np.ndarray:
    target_power_linear = 10.0 ** (target_power_decibel / 20.0)
    signal_norm = signal / (average_power(signal) ** 0.5)

    return target_power_linear * signal_norm
