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
    Get the standard data directory.
    
    Returns
    -------
    Path
        Path to the examples/data directory.
        
    Raises
    ------
    FileNotFoundError
        If the data directory cannot be located.
    """
    # Try to find data directory relative to current working directory
    current = Path.cwd()
    
    # Check common locations
    possible_paths = [
        current / "examples" / "data",  # From project root
        current / "data",  # From examples directory
        current.parent / "data",  # From examples subdirectory
    ]
    
    # Try to find relative to this file
    this_file = Path(__file__).parent
    project_root = this_file.parent.parent  # shmtools/utils -> shmtools-python
    possible_paths.append(project_root / "examples" / "data")
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return path
    
    # If not found, return the expected location for better error messages
    return project_root / "examples" / "data"


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