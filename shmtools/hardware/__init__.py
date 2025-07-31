"""
Hardware integration module for data acquisition.

This module provides interfaces for various data acquisition systems
including National Instruments and generic DAQ devices.
"""

from .signal_generation import band_lim_white_noise_shm

# TODO: Implement hardware interfaces
# from .daq import *
# from .ni_interface import *

__all__ = [
    "band_lim_white_noise_shm",
    # "acquire_data",
    # "ni_multiplexed_acquisition",
    # "excite_and_acquire",
]
