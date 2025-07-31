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
        current_dir / "examples" / "data",  # From project root
        current_dir / "data",  # From examples directory
        current_dir.parent / "data",  # From examples/notebooks/
        current_dir.parent.parent / "data",  # From examples/notebooks/basic/
        current_dir.parent.parent.parent
        / "data",  # From examples/notebooks/basic/subdir/
        # Relative to this file (most reliable)
        project_root / "examples" / "data",
        # Additional fallback paths for different execution contexts
        Path.cwd().resolve() / "shmtools-python" / "examples" / "data",
        Path.home()
        / "repo"
        / "shm"
        / "shmtools-python"
        / "examples"
        / "data",  # Common dev location
        # Check if we're in a subdirectory of the project
        *[
            p / "examples" / "data"
            for p in Path.cwd().resolve().parents
            if (p / "shmtools").exists()
        ],
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

    Primary dataset for SHMTools containing acceleration measurements from a 
    base-excited 3-story aluminum frame structure with simulated damage conditions.

    Dataset Structure
    -----------------
    - **Format**: (8192 time points, 5 channels, 170 test conditions)
    - **Sampling**: 2000 Hz downsampled from 2560 Hz (25.6s duration)
    - **Excitation**: Band-limited random (20-150 Hz), 2.6V RMS
    - **Tests**: 17 damage states × 10 tests each = 170 total conditions

    Physical Structure
    ------------------
    - **Frame**: Aluminum columns (17.7×2.5×0.6 cm) and plates (30.5×30.5×2.5 cm)  
    - **Design**: 4-column frame per floor, essentially 4-DOF system
    - **Sliding Rails**: Constrained motion in x-direction only
    - **Damage Mechanism**: Suspended center column with adjustable bumper contact
    - **Base**: Aluminum plate (76.2×30.5×2.5 cm) on rigid foam isolation

    Instrumentation
    ---------------  
    - **Channel 1**: Load cell (2.2 mV/N) - shaker input force
    - **Channels 2-5**: Accelerometers (1000 mV/g) - floor centerline responses
    - **DAQ**: National Instruments PXI system with ICP conditioning
    - **Shaker**: Electrodynamic with Techron 5530 amplifier

    Damage States
    -------------
    **Undamaged Baseline (States 1-9)**:
    - State 1: Pure baseline condition
    - State 2: +1.2kg mass at base (19% floor mass)
    - State 3: +1.2kg mass on 1st floor  
    - States 4-9: 87.5% stiffness reduction in various columns

    **Damaged Conditions (States 10-17)**:
    - States 10-14: Nonlinear bumper gaps (0.20, 0.15, 0.13, 0.10, 0.05 mm)
    - States 15-17: Combined damage + mass/stiffness variations

    Returns
    -------
    data : dict
        Dictionary with keys:
        - 'dataset': (8192, 5, 170) acceleration data array
        - 'fs': sampling frequency (Hz)
        - 'channels': list of channel names
        - 'conditions': array of condition numbers (1-170)
        - 'damage_states': array mapping each condition to damage state (1-17)
        - 'state_descriptions': dictionary mapping states to descriptions
        - 'description': dataset description string

    Raises
    ------
    FileNotFoundError
        If data3SS.mat is not found in the data directory.

    Examples
    --------
    >>> data = load_3story_data()
    >>> signals = data['dataset'][:, 1:, :]  # Channels 2-5 only (accelerometers)
    >>> baseline = signals[:, :, :90]        # Undamaged conditions (states 1-9)
    >>> damaged = signals[:, :, 90:]         # Damaged conditions (states 10-17)
    >>> fs = data['fs']                      # 2000.0 Hz sampling frequency

    Access specific damage states:

    >>> damage_states = data['damage_states'] 
    >>> state_1_indices = np.where(damage_states == 1)[0]   # Pure baseline
    >>> state_10_indices = np.where(damage_states == 10)[0] # 0.20mm gap damage
    >>> descriptions = data['state_descriptions']           # Human-readable labels

    Extract time series for specific conditions:

    >>> condition_1_data = signals[:, :, 0:10]    # State 1, all 10 tests
    >>> condition_10_data = signals[:, :, 90:100] # State 10, all 10 tests

    Notes
    -----
    This is the benchmark dataset for structural health monitoring algorithm 
    comparisons. Natural frequency variations are ±5 Hz for operational/
    environmental states. Data collection at Los Alamos National Laboratory.

    References: Figueiredo et al. (2009) LA-14393 report.
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

    if "dataset" not in mat_data:
        raise ValueError(f"Expected 'dataset' variable in {data_path}")

    dataset = mat_data["dataset"]  # Shape should be: (8192, 5, 170)

    if dataset.shape != (8192, 5, 170):
        raise ValueError(
            f"Unexpected dataset shape {dataset.shape}. "
            f"Expected (8192, 5, 170) for data3SS.mat"
        )

    # Detailed state descriptions from original MATLAB documentation
    state_descriptions = {
        1: "Undamaged - Baseline condition",
        2: "Undamaged - Mass = 1.2 kg at the base", 
        3: "Undamaged - Mass = 1.2 kg on the 1st floor",
        4: "Undamaged - 87.5% stiffness reduction in column 1BD",
        5: "Undamaged - 87.5% stiffness reduction in columns 1AD and 1BD",
        6: "Undamaged - 87.5% stiffness reduction in column 2BD",
        7: "Undamaged - 87.5% stiffness reduction in columns 2AD and 2BD", 
        8: "Undamaged - 87.5% stiffness reduction in column 3BD",
        9: "Undamaged - 87.5% stiffness reduction in columns 3AD and 3BD",
        10: "Damaged - Gap = 0.20 mm",
        11: "Damaged - Gap = 0.15 mm",
        12: "Damaged - Gap = 0.13 mm", 
        13: "Damaged - Gap = 0.10 mm",
        14: "Damaged - Gap = 0.05 mm",
        15: "Damaged - Gap = 0.20 mm and mass = 1.2 kg at the base",
        16: "Damaged - Gap = 0.20 mm and mass = 1.2 kg on the 1st floor",
        17: "Damaged - Gap = 0.10 mm and mass = 1.2 kg on the 1st floor"
    }

    return {
        "dataset": dataset,
        "fs": 2000.0,  # Sampling frequency from original documentation
        "channels": ["Force", "Ch2", "Ch3", "Ch4", "Ch5"],
        "conditions": np.arange(1, 171),  # State conditions 1-170
        "damage_states": np.repeat(np.arange(1, 18), 10),  # 17 states, 10 tests each
        "state_descriptions": state_descriptions,
        "description": "3-story structure base excitation data (LANL)",
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

    Contains rotating machinery vibration data collected from the SpectraQuest 
    Magnum Machinery Fault Simulator for bearing and gearbox fault analysis.

    Dataset Structure
    -----------------
    - **Channels**: 4 (Tachometer, Gearbox Accel., Top Bearing Accel., Side Bearing Accel.)
    - **Sampling**: 2048 Hz, 5 seconds per instance
    - **Conditions**: 6 fault states with 64 tests each
    - **Shaft Speed**: ~1000 rpm (nominally constant)
    - **Format**: dataset(1:10240, channels, instances) concatenated format

    Damage States
    -------------
    - **State 1-2**: Baseline (ball bearings, healthy gearbox)
    - **State 3**: Main shaft ball roller spin fault
    - **State 4-5**: Baseline (fluid bearings, healthy gearbox)  
    - **State 6**: Gearbox worn tooth fault

    Test Setup
    ----------
    - **Main Shaft**: 3/4" diameter steel, 28.5" center-to-center bearing support
    - **Gearbox**: Hub City M2, 1.5:1 ratio, 18/27 teeth (pinion/gear)
    - **Belt Drive**: ~1:3.71 ratio, 13" span, 3.7 lbs tension
    - **Magnetic Brake**: 1.9 lbs-in torsional load on pinion shaft
    - **Bearing Fault Frequencies**:
      - Cage Speed: 3.048 × Shaft Frequency
      - Outer Race: 3.048 × Shaft Frequency  
      - Inner Race: 4.95 × Shaft Frequency
      - Ball Spin: 1.992 × Shaft Frequency

    Returns
    -------
    data : dict
        Dictionary containing:
        - 'dataset': Raw vibration measurements (time, channels, instances)
        - Additional MATLAB variables from original file

    Raises
    ------
    FileNotFoundError
        If data_CBM.mat is not found.

    Examples
    --------
    >>> data = load_cbm_data()
    >>> vibration = data['dataset']  # Shape: (10240, 4, instances)
    >>> tach_channel = vibration[:, 0, :]      # Tachometer data
    >>> gearbox_accel = vibration[:, 1, :]     # Gearbox accelerometer
    >>> bearing_accels = vibration[:, 2:4, :]  # Bearing accelerometers

    Notes
    -----
    Data collected at Los Alamos National Laboratory using SpectraQuest 
    fault simulator. Useful for condition-based monitoring algorithm 
    development and bearing/gearbox fault detection research.
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
        mat_data = dict(scipy.io.loadmat(str(data_path)))
        
        # Add metadata for easier use
        mat_data.update({
            'fs': 2048.0,  # Sampling frequency (Hz)
            'duration': 5.0,  # Duration per instance (seconds)
            'shaft_speed_rpm': 1000.0,  # Nominal shaft speed
            'channels': ['Tachometer', 'Gearbox_Accel', 'Top_Bearing_Accel', 'Side_Bearing_Accel'],
            'fault_states': {
                1: 'Baseline 1 (ball bearings, healthy)',
                2: 'Baseline 2 (ball bearings, healthy)', 
                3: 'Ball roller spin fault',
                4: 'Baseline 1 (fluid bearings, healthy)',
                5: 'Baseline 2 (fluid bearings, healthy)',
                6: 'Gearbox worn tooth fault'
            },
            'description': 'SpectraQuest Magnum fault simulator CBM data (LANL)'
        })
        
        return mat_data
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
        "data3ss": {
            "file": "data3SS.mat",
            "size_mb": 25,
            "description": "3-story structure base excitation (primary dataset)",
            "examples": ["PCA", "Mahalanobis", "SVD", "NLPCA", "Factor Analysis"],
            "available": (data_dir / "data3SS.mat").exists(),
        },
        "cbm": {
            "file": "data_CBM.mat",
            "size_mb": 54,
            "description": "Condition-based monitoring (rotating machinery)",
            "examples": ["CBM Bearing Analysis", "CBM Gearbox Analysis"],
            "available": (data_dir / "data_CBM.mat").exists(),
        },
        "active_sensing": {
            "file": "data_example_ActiveSense.mat",
            "size_mb": 32,
            "description": "Guided wave ultrasonic measurements",
            "examples": ["Active Sensing Feature Extraction"],
            "available": (data_dir / "data_example_ActiveSense.mat").exists(),
        },
        "sensor_diagnostic": {
            "file": "dataSensorDiagnostic.mat",
            "size_mb": 0.06,
            "description": "Piezoelectric sensor impedance measurements",
            "examples": ["Sensor Diagnostics"],
            "available": (data_dir / "dataSensorDiagnostic.mat").exists(),
        },
        "modal_osp": {
            "file": "data_OSPExampleModal.mat",
            "size_mb": 0.05,
            "description": "Modal analysis and optimal sensor placement",
            "examples": ["Modal Features", "Optimal Sensor Placement"],
            "available": (data_dir / "data_OSPExampleModal.mat").exists(),
        },
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
        status = "✓ Available" if info["available"] else "✗ Missing"
        print(f"{info['file']:30} ({info['size_mb']:5.1f} MB) - {status}")

        if info["available"]:
            total_available += 1
            total_size += info["size_mb"]

    print()
    print(
        f"Available: {total_available}/{len(datasets)} datasets ({total_size:.1f} MB total)"
    )

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
        current_dir,  # From project root
        current_dir.parent,  # From examples/
        current_dir.parent.parent,  # From examples/notebooks/
        current_dir.parent.parent.parent,  # From examples/notebooks/basic/
    ]

    # Also try relative to this file
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent.parent
    possible_paths.append(project_root)

    shmtools_found = False
    for path in possible_paths:
        if (path / "shmtools").exists():
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
    plt.style.use("default")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 10

    # Return convenience dictionary
    return {
        "np": np,
        "plt": plt,
        "Path": Path,
        "load_3story_data": load_3story_data,
        "load_sensor_diagnostic_data": load_sensor_diagnostic_data,
        "load_cbm_data": load_cbm_data,
        "load_active_sensing_data": load_active_sensing_data,
        "load_modal_osp_data": load_modal_osp_data,
        "check_data_availability": check_data_availability,
        "get_data_dir": get_data_dir,
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
    if example_type in [
        "pca",
        "mahalanobis",
        "svd",
        "factor_analysis",
        "nlpca",
        "ar_model_order",
    ]:
        # For outlier detection examples - preprocess 3-story data
        data_dict = load_3story_data()
        dataset = data_dict["dataset"]

        # Extract channels 2-5 (indices 1-4) as done in notebooks
        signals = dataset[:, 1:5, :]  # Shape: (8192, 4, 170)

        return {
            "dataset": dataset,  # Original full dataset
            "signals": signals,  # Channels 2-5 only (ready for analysis)
            "fs": data_dict["fs"],
            "channels": ["Ch2", "Ch3", "Ch4", "Ch5"],  # Channel 2-5 labels
            "all_channels": data_dict["channels"],  # All channel labels
            "conditions": data_dict["conditions"],
            "damage_states": data_dict["damage_states"],
            "state_descriptions": data_dict["state_descriptions"],
            "description": data_dict["description"],
            "t": signals.shape[0],  # Time points
            "m": signals.shape[1],  # Channels (4)
            "n": signals.shape[2],  # Conditions (170)
        }

    elif example_type == "cbm":
        return load_cbm_data()

    elif example_type == "active_sensing":
        return load_active_sensing_data()

    elif example_type == "sensor_diagnostic":
        return load_sensor_diagnostic_data()

    elif example_type == "modal":
        return load_modal_osp_data()

    else:
        raise ValueError(
            f"Unknown example type: {example_type}. "
            f"Supported types: pca, mahalanobis, svd, factor_analysis, "
            f"nlpca, ar_model_order, cbm, active_sensing, sensor_diagnostic, modal"
        )


def validate_dataset_integrity() -> Dict[str, Dict[str, Any]]:
    """
    Validate the integrity and structure of all available datasets.
    
    Checks file existence, loads each dataset, validates expected structure,
    and reports any issues found.
    
    Returns
    -------
    dict
        Validation results for each dataset with status, errors, and metadata.
        
    Examples
    --------
    >>> results = validate_dataset_integrity()
    >>> for dataset, info in results.items():
    ...     print(f"{dataset}: {'✓' if info['valid'] else '✗'}")
    """
    datasets_info = get_available_datasets()
    validation_results = {}
    
    for dataset_name, info in datasets_info.items():
        result = {
            'available': info['available'],
            'valid': False,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        if not info['available']:
            result['errors'].append(f"File {info['file']} not found")
            validation_results[dataset_name] = result
            continue
            
        try:
            # Load each dataset and validate structure
            if dataset_name == 'data3ss':
                data = load_3story_data()
                expected_shape = (8192, 5, 170)
                if data['dataset'].shape != expected_shape:
                    result['errors'].append(f"Expected shape {expected_shape}, got {data['dataset'].shape}")
                else:
                    result['valid'] = True
                    result['metadata'] = {
                        'shape': data['dataset'].shape,
                        'fs': data['fs'],
                        'num_states': len(data['state_descriptions']),
                        'channels': data['channels']
                    }
                    
            elif dataset_name == 'cbm':
                data = load_cbm_data()
                if 'dataset' in data:
                    result['valid'] = True
                    result['metadata'] = {
                        'shape': data['dataset'].shape if 'dataset' in data else 'unknown',
                        'fs': data.get('fs', 'unknown'),
                        'fault_states': len(data.get('fault_states', {})),
                        'channels': data.get('channels', [])
                    }
                else:
                    result['warnings'].append("No 'dataset' variable found in CBM data")
                    result['valid'] = True  # Still valid, just different structure
                    
            else:
                # For other datasets, just try to load
                if dataset_name == 'active_sensing':
                    data = load_active_sensing_data()
                elif dataset_name == 'sensor_diagnostic':
                    data = load_sensor_diagnostic_data()
                elif dataset_name == 'modal_osp':
                    data = load_modal_osp_data()
                
                result['valid'] = True
                result['metadata'] = {
                    'variables': list(data.keys()),
                    'estimated_size': sum(v.nbytes if hasattr(v, 'nbytes') else 0 
                                        for v in data.values() if hasattr(v, 'nbytes'))
                }
                
        except Exception as e:
            result['errors'].append(f"Loading error: {str(e)}")
            
        validation_results[dataset_name] = result
        
    return validation_results


def print_dataset_summary():
    """
    Print a comprehensive summary of all available datasets.
    
    Includes dataset structure, physical setup, and usage information
    for quick reference.
    """
    print("=" * 80)
    print("SHMTools Dataset Summary")
    print("=" * 80)
    
    validation = validate_dataset_integrity()
    
    for dataset_name, result in validation.items():
        info = get_available_datasets()[dataset_name]
        status = "✓ VALID" if result['valid'] else "✗ INVALID"
        
        print(f"\n{info['file']} ({info['size_mb']} MB) - {status}")
        print("-" * 60)
        print(f"Description: {info['description']}")
        print(f"Examples: {', '.join(info['examples'])}")
        
        if result['metadata']:
            print("Structure:")
            for key, value in result['metadata'].items():
                print(f"  {key}: {value}")
        
        if result['errors']:
            print("Errors:")
            for error in result['errors']:
                print(f"  - {error}")
                
        if result['warnings']:
            print("Warnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
    
    print("\n" + "=" * 80)
    
    # Summary stats
    total_datasets = len(validation)
    valid_datasets = sum(1 for r in validation.values() if r['valid'])
    available_datasets = sum(1 for r in validation.values() if r['available'])
    
    print(f"Summary: {valid_datasets}/{available_datasets} valid, {available_datasets}/{total_datasets} available")
    
    if available_datasets < total_datasets:
        print("\nMissing datasets can be downloaded and placed in:")
        print(f"  {get_data_dir()}")
        print("See README.md in that directory for instructions.")
