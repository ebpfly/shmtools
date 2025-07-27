"""
Utility functions for SHMTools.

This module provides general utility functions for data management,
file I/O, and other common operations.
"""

# MATLAB-compatible data import functions (exact replicas)
from .data_io import (
    import_3StoryStructure_shm,
    import_CBMData_shm,
    import_ActiveSense1_shm,
    import_SensorDiagnostic_shm,
    import_ModalOSP_shm,
)

# Legacy functions for backward compatibility
try:
    from .data_io import import_cbm_data, load_mat_file, save_results
except ImportError:
    pass

# Modern convenience functions (keep for now but deprecate)
from .data_loading import (
    load_3story_data,
    load_sensor_diagnostic_data,
    load_cbm_data,
    load_active_sensing_data,
    load_modal_osp_data,
    get_available_datasets,
    check_data_availability,
)

__all__ = [
    # MATLAB-compatible functions (primary)
    "import_3StoryStructure_shm",
    "import_CBMData_shm",
    "import_ActiveSense1_shm",
    "import_SensorDiagnostic_shm",
    "import_ModalOSP_shm",
    # Legacy/convenience functions
    "load_3story_data",
    "load_sensor_diagnostic_data",
    "load_cbm_data",
    "load_active_sensing_data",
    "load_modal_osp_data",
    "get_available_datasets",
    "check_data_availability",
]
