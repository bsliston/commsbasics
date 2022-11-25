"""These functions house tools for data generation.

The functions implemented within this script encapsulate tools for data 
generation and management for basic communication proof of concept development.
"""

import numpy as np


def random_data(num_bits: int) -> np.ndarray:
    """Generates random data bits of either 0 or 1.

    Args:
        num_bits: Number of bits.

    Returns:
        Random data bits.
    """
    return np.random.randint(2, size=num_bits)
