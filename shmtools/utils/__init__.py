"""
Utility functions for SHMTools.

This module provides general utility functions for data management,
file I/O, and other common operations.

Note: Data import functions have been moved to examples.data with
proper GUI metadata formatting. Use:
    from examples.data import load_3story_data, import_3story_structure_shm
"""

# Data segmentation utilities
from .data_segmentation import (
    segment_time_series,
    prepare_train_test_split,
)

# Spatial analysis utilities
from .spatial_analysis import (
    compute_channel_wise_damage_indicators,
    plot_damage_indicators,
    analyze_damage_localization,
    compare_ar_arx_localization,
)

__all__ = [
    # Data segmentation utilities
    "segment_time_series",
    "prepare_train_test_split",
    # Spatial analysis utilities
    "compute_channel_wise_damage_indicators",
    "plot_damage_indicators",
    "analyze_damage_localization",
    "compare_ar_arx_localization",
]
