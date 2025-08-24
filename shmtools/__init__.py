"""
SHMTools: Python-based Structural Health Monitoring Toolkit

A comprehensive library for structural health monitoring, signal processing,
and damage detection algorithms. Converted from the original MATLAB SHMTools
library developed by Los Alamos National Laboratory.
"""

__version__ = "0.1.0"
__author__ = "SHMTools Development Team"

# Import main modules for easy access
from . import core
from . import features
from . import classification
from . import modal
from . import active_sensing
from . import hardware
from . import plotting
from . import utils
from . import sensor_diagnostics

# Core signal processing functions
from .core.spectral import psd_welch_shm, stft_shm, cwt_analysis_shm
from .core.signal_filtering import (
    residual_signal_shm,
    difference_signal_shm,
    bandpass_condition_signal_shm,
    gear_mesh_filter_shm,
    envelope_signal_shm,
)
from .core.signal_processing import ars_tach_shm, fir1_shm
from .core.statistics import (
    fm0_shm,
    fm4_shm,
    peak_factor_shm,
    impulse_factor_shm,
    clearance_factor_shm,
    shape_factor_shm,
    compute_damage_features_shm,
)
from .core.preprocessing import scale_min_max_shm

# Feature extraction functions
from .features.time_series import ar_model_shm, ar_model_order_shm
from .features.condition_based_monitoring import time_sync_avg_shm

# Classification and outlier detection functions
from .classification.outlier_detection import (
    learn_mahalanobis_shm,
    score_mahalanobis_shm,
    learn_svd_shm,
    score_svd_shm,
    learn_pca_shm,
    score_pca_shm,
    roc_shm,
    learn_factor_analysis_shm,
    score_factor_analysis_shm,
)
from .classification.nonparametric import (
    gaussian_kernel_shm,
    epanechnikov_kernel_shm,
    uniform_kernel_shm,
    quartic_kernel_shm,
    triangle_kernel_shm,
    triweight_kernel_shm,
    cosine_kernel_shm,
    learn_kernel_density_shm,
    score_kernel_density_shm,
)
from .classification.semiparametric import (
    k_medians_shm,
    learn_gmm_shm,
    score_gmm_shm,
    learn_gmm_semiparametric_model_shm,
    score_gmm_semiparametric_model_shm,
)

# NLPCA functions (optional - requires TensorFlow)
try:
    from .classification.nlpca import learn_nlpca_shm, score_nlpca_shm
except ImportError:
    pass  # TensorFlow not available

# Modal analysis functions
from .modal.modal_analysis import frf_shm, rpfit_shm

# Active sensing functions
from .active_sensing.matched_filter import (
    coherent_matched_filter_shm,
    incoherent_matched_filter_shm,
)
from .active_sensing.utilities import (
    extract_subsets_shm,
    flex_logic_filter_shm,
    sum_mult_dims_shm,
    estimate_group_velocity_shm,
)
from .active_sensing.geometry import (
    propagation_dist_2_points_shm,
    distance_2_index_shm,
    build_contained_grid_shm,
    sensor_pair_line_of_sight_shm,
    fill_2d_map_shm,
)

# Plotting functions
from .plotting.spectral_plots import plot_psd_shm, plot_spectrogram_shm, plotPSD_shm

# Sensor diagnostics functions
from .sensor_diagnostics.sensor_diagnostics import (
    sd_feature_shm,
    sd_autoclassify_shm,
    sd_plot_shm,
)

# Data import functions
from examples.data.data_imports import (
    import_3story_structure_shm,
    import_cbm_data_shm,
    import_active_sense1_shm,
    import_sensor_diagnostic_shm,
    import_modal_osp_shm,
)

# MATLAB-compatible aliases for data import functions
import_ModalOSP_shm = import_modal_osp_shm
import_3StoryStructure_shm = import_3story_structure_shm
import_CBMData_shm = import_cbm_data_shm
import_ActiveSense1_shm = import_active_sense1_shm
import_SensorDiagnostic_shm = import_sensor_diagnostic_shm

# Introspection capabilities moved to JupyterLab extension

# JupyterLab extension is installed as a direct dependency


__all__ = [
    # Modules
    "core",
    "features",
    "classification",
    "modal",
    "active_sensing",
    "hardware",
    "plotting",
    "utils",
    "sensor_diagnostics",
    # Core functions
    "psd_welch_shm",
    "stft_shm",
    "cwt_analysis_shm",
    "residual_signal_shm",
    "difference_signal_shm",
    "bandpass_condition_signal_shm",
    "gear_mesh_filter_shm",
    "envelope_signal_shm",
    "ars_tach_shm",
    "fir1_shm",
    "fm0_shm",
    "fm4_shm",
    "peak_factor_shm",
    "impulse_factor_shm",
    "clearance_factor_shm",
    "shape_factor_shm",
    "compute_damage_features_shm",
    "scale_min_max_shm",
    # Feature extraction
    "ar_model_shm",
    "ar_model_order_shm",
    "time_sync_avg_shm",
    # Classification
    "learn_mahalanobis_shm",
    "score_mahalanobis_shm",
    "learn_svd_shm",
    "score_svd_shm",
    "learn_pca_shm",
    "score_pca_shm",
    "roc_shm",
    "learn_factor_analysis_shm",
    "score_factor_analysis_shm",
    # Nonparametric
    "gaussian_kernel_shm",
    "epanechnikov_kernel_shm",
    "uniform_kernel_shm",
    "quartic_kernel_shm",
    "triangle_kernel_shm",
    "triweight_kernel_shm",
    "cosine_kernel_shm",
    "learn_kernel_density_shm",
    "score_kernel_density_shm",
    # Semi-parametric
    "k_medians_shm",
    "learn_gmm_shm",
    "score_gmm_shm",
    "learn_gmm_semiparametric_model_shm",
    "score_gmm_semiparametric_model_shm",
    # Modal analysis
    "frf_shm",
    "rpfit_shm",
    # Active sensing
    "coherent_matched_filter_shm",
    "incoherent_matched_filter_shm",
    "extract_subsets_shm",
    "flex_logic_filter_shm",
    "sum_mult_dims_shm",
    "estimate_group_velocity_shm",
    "propagation_dist_2_points_shm",
    "distance_2_index_shm",
    "build_contained_grid_shm",
    "sensor_pair_line_of_sight_shm",
    "fill_2d_map_shm",
    # Plotting
    "plot_psd_shm",
    "plot_spectrogram_shm",
    "plotPSD_shm",
    # Sensor diagnostics
    "sd_feature_shm",
    "sd_autoclassify_shm",
    "sd_plot_shm",
    # Data import
    "import_3story_structure_shm",
    "import_cbm_data_shm",
    "import_active_sense1_shm",
    "import_sensor_diagnostic_shm",
    "import_modal_osp_shm",
    # MATLAB-compatible aliases
    "import_ModalOSP_shm",
    "import_3StoryStructure_shm",
    "import_CBMData_shm",
    "import_ActiveSense1_shm",
    "import_SensorDiagnostic_shm",
    # Utilities
    "gui",
]
