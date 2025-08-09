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
from .utils.data_io import (
    import_3StoryStructure_shm,
    import_CBMData_shm,
    import_ActiveSense1_shm,
    import_SensorDiagnostic_shm,
    import_ModalOSP_shm,
)

# Load introspection capabilities for Jupyter notebooks
try:
    from . import introspection
except ImportError:
    pass  # Introspection not available

# JupyterLab extension is installed as a direct dependency


def gui():
    """Load SHM Function Selector GUI for Jupyter notebooks."""
    try:
        from IPython.display import Javascript, display

        js = """
document.querySelectorAll('[id*="shm-dropdown"]').forEach(el => el.remove());

const funcs = {
    'Spectral Analysis': [
        {name: 'PSD Welch', code: 'psd_matrix, frequencies, one_sided = shmtools.psd_welch_shm(X=data, fs=fs)'},
        {name: 'STFT', code: 'f, t, Zxx = shmtools.stft_shm(x=signal, fs=fs)'},
        {name: 'CWT Analysis', code: 'coeffs, freqs = shmtools.cwt_analysis_shm(x=signal, scales=scales)'}
    ],
    'Time Series Modeling': [
        {name: 'AR Model', code: 'ar_params_fv, rms_fv, ar_params, residuals, prediction = shmtools.ar_model_shm(X=data, ar_order=15)'},
        {name: 'AR Model Order', code: 'mean_order, orders, model = shmtools.ar_model_order_shm(X=data, method="PAF")'}
    ],
    'Outlier Detection': [
        {name: 'Learn PCA', code: 'model = shmtools.learn_pca_shm(X=features, per_var=0.95)'},
        {name: 'Score PCA', code: 'scores, residuals = shmtools.score_pca_shm(Y=test_features, model=model)'},
        {name: 'Learn Mahalanobis', code: 'model = shmtools.learn_mahalanobis_shm(X=features)'},
        {name: 'Score Mahalanobis', code: 'scores = shmtools.score_mahalanobis_shm(Y=test_features, model=model)'}
    ],
    'Signal Filtering': [
        {name: 'Residual Signal', code: 'residual = shmtools.residual_signal_shm(x=signal, fs=fs, shaft_freq=freq)'},
        {name: 'Envelope Signal', code: 'envelope = shmtools.envelope_signal_shm(x=signal)'}
    ],
    'Statistics': [
        {name: 'Peak Factor', code: 'pf = shmtools.peak_factor_shm(x=signal)'},
        {name: 'Crest Factor', code: 'cf = shmtools.crest_factor_shm(x=signal)'}
    ]
};

const panel = document.createElement('div');
panel.id = 'shm-dropdown-panel';
panel.style.cssText = 'position:fixed; top:60px; right:20px; background:white; border:2px solid #007bff; border-radius:8px; box-shadow:0 4px 20px rgba(0,0,0,0.15); z-index:1000; width:320px; font-family:Arial,sans-serif;';

const header = document.createElement('div');
header.innerHTML = '🔧 SHMTools Functions';
header.style.cssText = 'background:#007bff; color:white; padding:12px 16px; font-weight:bold; font-size:14px; border-radius:6px 6px 0 0; cursor:move;';

let isDragging = false, dragOffset = {x:0, y:0};
header.onmousedown = e => {
    isDragging = true;
    dragOffset.x = e.clientX - panel.offsetLeft;
    dragOffset.y = e.clientY - panel.offsetTop;
};
document.onmousemove = e => {
    if (isDragging) {
        panel.style.left = (e.clientX - dragOffset.x) + 'px';
        panel.style.top = (e.clientY - dragOffset.y) + 'px';
        panel.style.right = 'auto';
    }
};
document.onmouseup = () => isDragging = false;

panel.appendChild(header);

const content = document.createElement('div');
content.style.cssText = 'max-height:400px; overflow-y:auto;';

Object.keys(funcs).forEach(category => {
    const catHeader = document.createElement('div');
    catHeader.textContent = category;
    catHeader.style.cssText = 'padding:10px 16px; background:#f8f9fa; font-weight:bold; color:#495057; font-size:12px; border-bottom:1px solid #dee2e6;';
    content.appendChild(catHeader);
    
    funcs[category].forEach(func => {
        const item = document.createElement('div');
        item.textContent = func.name;
        item.style.cssText = 'padding:10px 16px; cursor:pointer; border-bottom:1px solid #f1f3f4; transition:background-color 0.2s;';
        
        item.onmouseover = () => item.style.backgroundColor = '#e3f2fd';
        item.onmouseout = () => item.style.backgroundColor = 'transparent';
        item.onclick = () => {
            navigator.clipboard.writeText(func.code);
            console.log('Copied to clipboard:', func.code);
        };
        
        content.appendChild(item);
    });
});

panel.appendChild(content);
document.body.appendChild(panel);
console.log('✅ SHMTools Extension loaded - click functions to copy code');
        """

        display(Javascript(js))
        print(
            "🔧 SHMTools Function Selector loaded! Look for blue panel in top-right corner."
        )
        print("💡 Click any function to copy code to clipboard, then paste into cell.")

    except ImportError:
        print("❌ Jupyter GUI only available in Jupyter notebooks")


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
    "import_3StoryStructure_shm",
    "import_CBMData_shm",
    "import_ActiveSense1_shm",
    "import_SensorDiagnostic_shm",
    "import_ModalOSP_shm",
    # Utilities
    "gui",
]
