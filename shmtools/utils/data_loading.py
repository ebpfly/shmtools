"""
Data loading utilities for SHMTools example datasets.

This module provides consistent interfaces for loading the MATLAB .mat files
used in the original SHMTools examples.
"""

import scipy.io
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def get_data_dir() -> Path:
    """
    Get the standard data directory with robust path resolution.
    
    This function handles multiple execution contexts including:
    - Jupyter notebooks in different directories
    - Python scripts run from various locations  
    - Different working directories
    - Development and production environments
    
    Returns
    -------
    Path
        Path to the examples/data directory.
        
    Raises
    ------
    FileNotFoundError
        If the data directory cannot be located.
    """
    # Get current working directory and this file's location
    current_dir = Path.cwd()
    this_file = Path(__file__).resolve()
    
    # Calculate project root from this file location
    # shmtools/utils/data_loading.py -> shmtools-python/
    project_root = this_file.parent.parent.parent
    
    # Define possible data directory locations in order of preference
    possible_paths = [
        # From current working directory (most common notebook case)
        current_dir / "examples" / "data",           # From project root
        current_dir / "data",                        # From examples directory  
        current_dir.parent / "data",                 # From examples/notebooks/
        current_dir.parent.parent / "data",          # From examples/notebooks/basic/
        current_dir.parent.parent.parent / "data",   # From examples/notebooks/basic/subdir/
        
        # Relative to this file (most reliable)
        project_root / "examples" / "data",
        
        # Additional fallback paths for different execution contexts
        Path.cwd().resolve() / "shmtools-python" / "examples" / "data",
        Path.home() / "repo" / "shm" / "shmtools-python" / "examples" / "data",  # Common dev location
        
        # Check if we're in a subdirectory of the project
        *[p / "examples" / "data" for p in Path.cwd().resolve().parents if (p / "shmtools").exists()],
    ]
    
    # Find the first valid data directory
    for path in possible_paths:
        if path.exists() and path.is_dir():
            # Verify it contains expected data files
            expected_files = ["data3SS.mat", "dataSensorDiagnostic.mat", "data_CBM.mat"]
            if any((path / f).exists() for f in expected_files):
                return path.resolve()
    
    # If not found, return the expected location for better error messages
    return (project_root / "examples" / "data").resolve()


def load_3story_data() -> Dict[str, Any]:
    """
    Load 3-story structure dataset.
    
    This is the primary dataset used by most SHMTools examples, containing
    acceleration measurements from a base-excited 3-story structure with
    various damage conditions.
    
    Returns
    -------
    data : dict
        Dictionary with keys:
        - 'dataset': (8192, 5, 170) acceleration data array
        - 'fs': sampling frequency (Hz)
        - 'channels': list of channel names
        - 'conditions': array of condition numbers (1-170)
        - 'damage_states': array mapping each condition to damage state (1-17)
        - 'description': dataset description string
        
    Raises
    ------
    FileNotFoundError
        If data3SS.mat is not found in the data directory.
        
    Examples
    --------
    >>> data = load_3story_data()
    >>> signals = data['dataset'][:, 1:, :]  # Channels 2-5 only
    >>> baseline = signals[:, :, :90]        # Undamaged conditions  
    >>> damaged = signals[:, :, 90:]         # Damaged conditions
    >>> fs = data['fs']                      # Sampling frequency
    
    Load specific damage states:
    
    >>> damage_states = data['damage_states']
    >>> state_1_indices = np.where(damage_states == 1)[0]  # Undamaged
    >>> state_10_indices = np.where(damage_states == 10)[0]  # First damage level
    """
    data_dir = get_data_dir()
    data_path = data_dir / "data3SS.mat"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}.\n"
            f"Please download data3SS.mat and place it in {data_dir}/\n"
            f"See {data_dir}/README.md for download instructions."
        )
    
    # Load MATLAB file
    try:
        mat_data = scipy.io.loadmat(str(data_path))
    except Exception as e:
        raise IOError(f"Failed to load {data_path}: {e}")
    
    if 'dataset' not in mat_data:
        raise ValueError(f"Expected 'dataset' variable in {data_path}")
    
    dataset = mat_data['dataset']  # Shape should be: (8192, 5, 170)
    
    if dataset.shape != (8192, 5, 170):
        raise ValueError(
            f"Unexpected dataset shape {dataset.shape}. "
            f"Expected (8192, 5, 170) for data3SS.mat"
        )
    
    return {
        'dataset': dataset,
        'fs': 2000.0,  # Sampling frequency from original documentation
        'channels': ['Force', 'Ch2', 'Ch3', 'Ch4', 'Ch5'],
        'conditions': np.arange(1, 171),  # State conditions 1-170
        'damage_states': np.repeat(np.arange(1, 18), 10),  # 17 states, 10 tests each
        'description': "3-story structure base excitation data (LANL)"
    }


def load_sensor_diagnostic_data() -> Dict[str, Any]:
    """
    Load sensor diagnostic dataset.
    
    Contains piezoelectric sensor impedance measurements for sensor
    health monitoring analysis.
    
    Returns
    -------
    data : dict
        Dictionary containing the loaded MATLAB variables.
        
    Raises
    ------
    FileNotFoundError
        If dataSensorDiagnostic.mat is not found.
    """
    data_dir = get_data_dir()
    data_path = data_dir / "dataSensorDiagnostic.mat"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}.\n"
            f"Please download dataSensorDiagnostic.mat and place it in {data_dir}/\n"
            f"See {data_dir}/README.md for download instructions."
        )
    
    try:
        return dict(scipy.io.loadmat(str(data_path)))
    except Exception as e:
        raise IOError(f"Failed to load {data_path}: {e}")


def load_cbm_data() -> Dict[str, Any]:
    """
    Load condition-based monitoring dataset.
    
    Contains rotating machinery vibration data for condition-based
    monitoring analysis (bearing and gearbox examples).
    
    Returns
    -------
    data : dict
        Dictionary containing the loaded MATLAB variables.
        
    Raises
    ------
    FileNotFoundError
        If data_CBM.mat is not found.
    """
    data_dir = get_data_dir()
    data_path = data_dir / "data_CBM.mat"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}.\n"
            f"Please download data_CBM.mat and place it in {data_dir}/\n"
            f"See {data_dir}/README.md for download instructions."
        )
    
    try:
        return dict(scipy.io.loadmat(str(data_path)))
    except Exception as e:
        raise IOError(f"Failed to load {data_path}: {e}")


def load_active_sensing_data() -> Dict[str, Any]:
    """
    Load active sensing dataset.
    
    Contains guided wave ultrasonic measurements for active sensing
    feature extraction analysis.
    
    Returns
    -------
    data : dict
        Dictionary containing the loaded MATLAB variables.
        
    Raises
    ------
    FileNotFoundError
        If data_example_ActiveSense.mat is not found.
    """
    data_dir = get_data_dir()
    data_path = data_dir / "data_example_ActiveSense.mat"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}.\n"
            f"Please download data_example_ActiveSense.mat and place it in {data_dir}/\n"
            f"See {data_dir}/README.md for download instructions."
        )
    
    try:
        return dict(scipy.io.loadmat(str(data_path)))
    except Exception as e:
        raise IOError(f"Failed to load {data_path}: {e}")


def load_modal_osp_data() -> Dict[str, Any]:
    """
    Load modal analysis and optimal sensor placement dataset.
    
    Contains modal analysis data for optimal sensor placement
    and modal feature extraction examples.
    
    Returns
    -------
    data : dict
        Dictionary containing the loaded MATLAB variables.
        
    Raises
    ------
    FileNotFoundError
        If data_OSPExampleModal.mat is not found.
    """
    data_dir = get_data_dir()
    data_path = data_dir / "data_OSPExampleModal.mat"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}.\n"
            f"Please download data_OSPExampleModal.mat and place it in {data_dir}/\n"
            f"See {data_dir}/README.md for download instructions."
        )
    
    try:
        return dict(scipy.io.loadmat(str(data_path)))
    except Exception as e:
        raise IOError(f"Failed to load {data_path}: {e}")


def get_available_datasets() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available datasets.
    
    Returns
    -------
    datasets : dict
        Dictionary mapping dataset names to information dictionaries.
        Each info dict contains: 'file', 'size_mb', 'description', 'examples'.
    """
    data_dir = get_data_dir()
    
    datasets = {
        'data3ss': {
            'file': 'data3SS.mat',
            'size_mb': 25,
            'description': '3-story structure base excitation (primary dataset)',
            'examples': ['PCA', 'Mahalanobis', 'SVD', 'NLPCA', 'Factor Analysis'],
            'available': (data_dir / 'data3SS.mat').exists()
        },
        'cbm': {
            'file': 'data_CBM.mat', 
            'size_mb': 54,
            'description': 'Condition-based monitoring (rotating machinery)',
            'examples': ['CBM Bearing Analysis', 'CBM Gearbox Analysis'],
            'available': (data_dir / 'data_CBM.mat').exists()
        },
        'active_sensing': {
            'file': 'data_example_ActiveSense.mat',
            'size_mb': 32, 
            'description': 'Guided wave ultrasonic measurements',
            'examples': ['Active Sensing Feature Extraction'],
            'available': (data_dir / 'data_example_ActiveSense.mat').exists()
        },
        'sensor_diagnostic': {
            'file': 'dataSensorDiagnostic.mat',
            'size_mb': 0.06,
            'description': 'Piezoelectric sensor impedance measurements', 
            'examples': ['Sensor Diagnostics'],
            'available': (data_dir / 'dataSensorDiagnostic.mat').exists()
        },
        'modal_osp': {
            'file': 'data_OSPExampleModal.mat',
            'size_mb': 0.05,
            'description': 'Modal analysis and optimal sensor placement',
            'examples': ['Modal Features', 'Optimal Sensor Placement'],
            'available': (data_dir / 'data_OSPExampleModal.mat').exists()
        }
    }
    
    return datasets


def check_data_availability() -> None:
    """
    Check which datasets are available and print status.
    
    Useful for verifying data setup before running examples.
    """
    datasets = get_available_datasets()
    data_dir = get_data_dir()
    
    print(f"Data directory: {data_dir}")
    print(f"Directory exists: {data_dir.exists()}")
    print()
    
    total_available = 0
    total_size = 0
    
    for name, info in datasets.items():
        status = "✓ Available" if info['available'] else "✗ Missing"
        print(f"{info['file']:30} ({info['size_mb']:5.1f} MB) - {status}")
        
        if info['available']:
            total_available += 1
            total_size += info['size_mb']
    
    print()
    print(f"Available: {total_available}/{len(datasets)} datasets ({total_size:.1f} MB total)")
    
    if total_available < len(datasets):
        print()
        print("Missing datasets can be downloaded and placed in:")
        print(f"  {data_dir}")
        print("See README.md in that directory for instructions.")


# Notebook-specific convenience functions
def setup_notebook_environment():
    """
    Setup imports and paths for Jupyter notebooks.
    
    This function handles the path setup that's repeated in every notebook,
    making it easy to drop into notebook cells.
    
    Returns
    -------
    dict
        Dictionary with commonly used imports and data loading functions.
        
    Examples
    --------
    In a notebook cell:
    
    >>> from shmtools.utils.data_loading import setup_notebook_environment
    >>> nb = setup_notebook_environment()
    >>> data = nb['load_3story_data']()
    >>> plt = nb['plt']
    >>> np = nb['np']
    """
    import sys
    from pathlib import Path
    
    # Setup path to find shmtools (same logic as in notebooks)
    current_dir = Path.cwd()
    
    # Try different relative paths to find shmtools  
    possible_paths = [
        current_dir,                        # From project root
        current_dir.parent,                 # From examples/
        current_dir.parent.parent,          # From examples/notebooks/
        current_dir.parent.parent.parent,   # From examples/notebooks/basic/
    ]
    
    # Also try relative to this file
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent.parent
    possible_paths.append(project_root)
    
    shmtools_found = False
    for path in possible_paths:
        if (path / 'shmtools').exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            shmtools_found = True
            print(f"Found shmtools at: {path}")
            break
    
    if not shmtools_found:
        print("Warning: Could not find shmtools module")
    
    # Import common packages
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Set up plotting defaults (from notebooks)
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # Return convenience dictionary
    return {
        'np': np,
        'plt': plt,
        'Path': Path,
        'load_3story_data': load_3story_data,
        'load_sensor_diagnostic_data': load_sensor_diagnostic_data,
        'load_cbm_data': load_cbm_data,
        'load_active_sensing_data': load_active_sensing_data,
        'load_modal_osp_data': load_modal_osp_data,
        'check_data_availability': check_data_availability,
        'get_data_dir': get_data_dir,
    }


def load_example_data(example_type: str) -> Dict[str, Any]:
    """
    Load data for specific example types with convenience wrappers.
    
    Parameters
    ----------
    example_type : str
        Type of example: 'pca', 'mahalanobis', 'svd', 'factor_analysis', 
        'ar_model_order', 'cbm', 'active_sensing', 'sensor_diagnostic', 'modal'
    
    Returns
    -------
    dict
        Loaded and preprocessed data ready for the specific example.
        
    Examples
    --------
    >>> # For PCA/Mahalanobis/SVD outlier detection examples
    >>> data = load_example_data('pca')
    >>> signals = data['signals']  # Shape: (8192, 4, 170) - channels 2-5 only
    >>> fs = data['fs']
    >>> damage_states = data['damage_states']
    
    >>> # For condition-based monitoring
    >>> data = load_example_data('cbm')
    >>> bearing_data = data['bearing_data']
    """
    if example_type in ['pca', 'mahalanobis', 'svd', 'factor_analysis', 'nlpca', 'ar_model_order']:
        # For outlier detection examples - preprocess 3-story data
        data_dict = load_3story_data()
        dataset = data_dict['dataset']
        
        # Extract channels 2-5 (indices 1-4) as done in notebooks
        signals = dataset[:, 1:5, :]  # Shape: (8192, 4, 170)
        
        return {
            'dataset': dataset,              # Original full dataset 
            'signals': signals,              # Channels 2-5 only (ready for analysis)
            'fs': data_dict['fs'],
            'channels': ['Ch2', 'Ch3', 'Ch4', 'Ch5'],  # Channel 2-5 labels
            'all_channels': data_dict['channels'],      # All channel labels
            'conditions': data_dict['conditions'],
            'damage_states': data_dict['damage_states'],
            'description': data_dict['description'],
            't': signals.shape[0],           # Time points
            'm': signals.shape[1],           # Channels (4)
            'n': signals.shape[2],           # Conditions (170)
        }
        
    elif example_type == 'cbm':
        return load_cbm_data()
        
    elif example_type == 'active_sensing':
        return load_active_sensing_data()
        
    elif example_type == 'sensor_diagnostic':
        return load_sensor_diagnostic_data()
        
    elif example_type == 'modal':
        return load_modal_osp_data()
        
    else:
        raise ValueError(f"Unknown example type: {example_type}. "
                        f"Supported types: pca, mahalanobis, svd, factor_analysis, "
                        f"nlpca, ar_model_order, cbm, active_sensing, sensor_diagnostic, modal")