"""
Data input/output utilities for SHMTools.

This module provides functions for loading various data formats commonly
used in structural health monitoring applications.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available - .mat file loading disabled")


def import_cbm_data(filename: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Import condition-based monitoring dataset.
    
    Python equivalent of MATLAB's import_CBMData_shm function. Loads
    multi-channel time series data for condition-based monitoring analysis.
    
    .. meta::
        :category: Data Import
        :matlab_equivalent: import_CBMData_shm
        :complexity: Basic
        :data_type: Time Series
        :output_type: Multi-channel Data
    
    Parameters
    ----------
    filename : str, optional
        Path to the CBM data file. If None, loads default example data.
        
        .. gui::
            :widget: file_upload
            :formats: [".mat"]
            :description: "CBM dataset file"
    
    Returns
    -------
    dataset : ndarray, shape (n_samples, n_channels, n_instances)
        Multi-channel time series data. Channel 1 is typically tachometer,
        Channel 2 is accelerometer.
    damage_states : ndarray, shape (n_instances,)
        Damage state labels (0=baseline, 1=damaged, etc.).
    state_list : ndarray, shape (n_instances,)
        Detailed state condition numbers.
    fs : float
        Sampling frequency in Hz.
        
    Raises
    ------
    ImportError
        If scipy is not available for .mat file loading.
    FileNotFoundError
        If specified file cannot be found.
        
    Examples
    --------
    Load default CBM dataset:
    
    >>> from shmtools.utils import import_cbm_data
    >>> dataset, damage_states, state_list, fs = import_cbm_data()
    >>> print(f"Data shape: {dataset.shape}")
    >>> print(f"Sampling frequency: {fs} Hz")
    >>> print(f"Unique states: {np.unique(state_list)}")
    
    Load specific file:
    
    >>> dataset, damage_states, state_list, fs = import_cbm_data("my_data.mat")
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for .mat file loading. Install with: pip install scipy")
    
    if filename is None:
        # Generate synthetic CBM data for testing
        return _generate_synthetic_cbm_data()
    
    try:
        # Load MATLAB .mat file
        mat_data = loadmat(filename)
        
        # Extract data based on typical CBM data structure
        # Adjust these keys based on actual .mat file structure
        dataset = mat_data.get('dataset', mat_data.get('data'))
        damage_states = mat_data.get('damageStates', mat_data.get('labels'))
        state_list = mat_data.get('stateList', mat_data.get('states'))
        fs = float(mat_data.get('Fs', mat_data.get('fs', 1000.0)))
        
        if dataset is None:
            raise ValueError("Could not find dataset in .mat file")
        
        # Ensure proper array shapes
        if dataset.ndim == 2:
            dataset = dataset[:, :, np.newaxis]
        
        if damage_states is not None:
            damage_states = np.squeeze(damage_states)
        else:
            damage_states = np.zeros(dataset.shape[2])
            
        if state_list is not None:
            state_list = np.squeeze(state_list)
        else:
            state_list = np.arange(dataset.shape[2]) + 1
        
        return dataset, damage_states, state_list, fs
        
    except Exception as e:
        raise FileNotFoundError(f"Could not load CBM data from {filename}: {e}")


def _generate_synthetic_cbm_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate synthetic condition-based monitoring data for testing.
    
    Returns
    -------
    dataset : ndarray
        Synthetic multi-channel time series.
    damage_states : ndarray
        Binary damage labels.
    state_list : ndarray
        State condition numbers.
    fs : float
        Sampling frequency.
    """
    # Simulation parameters
    fs = 25600.0  # Sampling frequency
    n_samples = 8192  # Samples per instance
    n_baseline = 10   # Baseline instances
    n_damaged = 10    # Damaged instances
    n_instances = n_baseline + n_damaged
    
    # Time vector
    t = np.arange(n_samples) / fs
    
    # Gear parameters
    main_shaft_freq = 30.0  # Hz (1800 RPM)
    gear_ratio = 3.71
    gear_freq = main_shaft_freq * gear_ratio
    n_gear_teeth = 27
    gear_mesh_freq = gear_freq * n_gear_teeth
    
    # Initialize data arrays
    dataset = np.zeros((n_samples, 2, n_instances))  # [time, channels, instances]
    damage_states = np.zeros(n_instances)
    state_list = np.zeros(n_instances)
    
    np.random.seed(42)  # For reproducible results
    
    for i in range(n_instances):
        is_damaged = i >= n_baseline
        
        # Channel 1: Tachometer signal (pulse per revolution)
        tach_freq = main_shaft_freq + np.random.normal(0, 0.5)  # Small speed variation
        tach_signal = 0.5 * np.sin(2 * np.pi * tach_freq * t)
        
        # Add tachometer pulses
        pulse_times = np.arange(0, t[-1], 1/tach_freq)
        for pulse_time in pulse_times:
            if pulse_time < t[-1]:
                pulse_idx = int(pulse_time * fs)
                if pulse_idx < n_samples:
                    tach_signal[pulse_idx:pulse_idx+10] += 2.0
        
        dataset[:, 0, i] = tach_signal + 0.1 * np.random.randn(n_samples)
        
        # Channel 2: Accelerometer signal
        # Base vibration components
        accel_signal = (0.1 * np.sin(2 * np.pi * main_shaft_freq * t) +
                       0.2 * np.sin(2 * np.pi * gear_freq * t) +
                       0.3 * np.sin(2 * np.pi * gear_mesh_freq * t))
        
        # Add gear mesh harmonics
        for h in range(2, 6):
            accel_signal += 0.1/h * np.sin(2 * np.pi * h * gear_mesh_freq * t)
        
        if is_damaged:
            # Add damage characteristics
            # Increased sidebands around gear mesh frequency
            sideband_freq = 1.0  # Hz
            accel_signal += (0.05 * np.sin(2 * np.pi * (gear_mesh_freq + sideband_freq) * t) +
                           0.05 * np.sin(2 * np.pi * (gear_mesh_freq - sideband_freq) * t))
            
            # Add random impulses (tooth wear/damage)
            n_impulses = 5
            impulse_times = np.random.uniform(0, t[-1], n_impulses)
            for imp_time in impulse_times:
                imp_idx = int(imp_time * fs)
                if imp_idx < n_samples - 50:
                    # Exponentially decaying impulse
                    imp_duration = 50
                    imp_t = np.arange(imp_duration) / fs
                    impulse = 0.5 * np.exp(-imp_t * 200) * np.sin(2 * np.pi * 2000 * imp_t)
                    accel_signal[imp_idx:imp_idx+imp_duration] += impulse
            
            # Increase overall noise level
            accel_signal += 0.05 * np.random.randn(n_samples)
            
            damage_states[i] = 1
            state_list[i] = 3  # Damaged state
        else:
            # Baseline condition
            accel_signal += 0.02 * np.random.randn(n_samples)
            damage_states[i] = 0
            state_list[i] = 1  # Baseline state
        
        dataset[:, 1, i] = accel_signal
    
    return dataset, damage_states, state_list, fs


def load_mat_file(filename: str) -> Dict[str, Any]:
    """
    Load MATLAB .mat file with error handling.
    
    Parameters
    ----------
    filename : str
        Path to .mat file.
        
    Returns
    -------
    data : dict
        Dictionary containing .mat file contents.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for .mat file loading")
    
    try:
        return loadmat(filename)
    except Exception as e:
        raise FileNotFoundError(f"Could not load .mat file {filename}: {e}")


def save_results(data: Dict[str, Any], filename: str) -> None:
    """
    Save analysis results to file.
    
    Parameters
    ----------
    data : dict
        Dictionary containing results to save.
    filename : str
        Output filename (.npz or .mat).
    """
    if filename.endswith('.npz'):
        np.savez(filename, **data)
    elif filename.endswith('.mat') and HAS_SCIPY:
        from scipy.io import savemat
        savemat(filename, data)
    else:
        raise ValueError("Unsupported file format. Use .npz or .mat")