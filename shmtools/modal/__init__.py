"""
Modal analysis module for SHMTools.

This module provides functions for modal parameter identification
and structural dynamics analysis.
"""

from .modal_analysis import frf_shm, rpfit_shm

__all__ = [
    "frf_shm",
    "rpfit_shm",
]
