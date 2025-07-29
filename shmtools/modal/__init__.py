"""
Modal analysis module for SHMTools.

This module provides functions for modal parameter identification,
structural dynamics analysis, and optimal sensor placement.
"""

from .modal_analysis import frf_shm, rpfit_shm
from .osp import (
    response_interp_shm,
    add_resp_2_geom_shm,
    osp_fisher_info_eiv_shm,
    get_sensor_layout_shm,
    osp_max_norm_shm,
    node_element_plot_shm,
    plot_sensors_shm,
)

__all__ = [
    # Modal analysis
    "frf_shm",
    "rpfit_shm",
    # Optimal sensor placement
    "response_interp_shm",
    "add_resp_2_geom_shm", 
    "osp_fisher_info_eiv_shm",
    "get_sensor_layout_shm",
    "osp_max_norm_shm",
    "node_element_plot_shm",
    "plot_sensors_shm",
]
