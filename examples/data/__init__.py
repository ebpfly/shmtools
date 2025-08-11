"""
SHMTools example data import functions and loaders.

This module provides MATLAB-compatible import functions for loading
the standard SHMTools example datasets.
"""

from .data_imports import (
    import_3story_structure_shm,
    import_cbm_data_shm,
    import_active_sense1_shm,
    import_sensor_diagnostic_shm,
    import_modal_osp_shm,
)

from .data_loaders import (
    load_3story_data,
    load_cbm_data,
    load_sensor_diagnostic_data,
    load_modal_osp_data,
    load_active_sense_data,
)

__all__ = [
    # Import functions (MATLAB-compatible)
    "import_3story_structure_shm",
    "import_cbm_data_shm",
    "import_active_sense1_shm",
    "import_sensor_diagnostic_shm",
    "import_modal_osp_shm",
    # Load functions (dictionary format)
    "load_3story_data",
    "load_cbm_data",
    "load_sensor_diagnostic_data",
    "load_modal_osp_data",
    "load_active_sense_data",
]