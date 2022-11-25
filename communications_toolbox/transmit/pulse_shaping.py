import numpy as np


def raised_cosine_filter(
    num_taps: int, beta: float, samples_per_symbol: int
) -> np.ndarray:
    times = np.arange(-num_taps // 2, (num_taps // 2) + 1)
    filter = np.zeros(len(times))
    for time_i, time in enumerate(times):
        if abs(time) == samples_per_symbol / (2.0 * beta):
            filter[time_i] = (
                np.sinc(1.0 / (2 * beta)) * np.pi / (4.0 * samples_per_symbol)
            )
        else:
            filter[time_i] = (
                (1 / samples_per_symbol)
                * np.sinc(time / samples_per_symbol)
                * np.cos(np.pi * beta * time / samples_per_symbol)
                / (1 - (2 * beta * time / samples_per_symbol) ** 2)
            )

    return filter
