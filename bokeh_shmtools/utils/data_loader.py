"""
Data loading utilities for the GUI.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, Any, Optional


def load_3story_data() -> Optional[Dict[str, Any]]:
    """
    Load 3-story structure data matching original MATLAB format exactly.

    Returns exactly what MATLAB data3SS.mat contains for perfect compatibility.

    Returns
    -------
    data : dict or None
        Dictionary with original MATLAB variables, or None if loading fails.
    """
    try:
        # Look for data file in examples/data/
        data_path = Path("examples/data/data3SS.mat")
        if not data_path.exists():
            # Alternative path from GUI directory
            data_path = Path("../examples/data/data3SS.mat")
        if not data_path.exists():
            print(f"❌ Data file not found: data3SS.mat")
            return None

        # Load the MATLAB file
        mat_data = sio.loadmat(str(data_path))

        # Extract MATLAB variables exactly as they appear
        dataset = mat_data["dataset"]  # Shape: (8192, 5, 170)
        states = mat_data["states"]  # Shape: (1, 170) - actual state numbers

        print(f"✅ Loaded 3-story structure data (MATLAB format):")
        print(f"   dataset: {dataset.shape}")
        print(f"   states: {states.shape}")

        # Return the dataset directly as main output (for modular compatibility)
        # This matches MATLAB behavior where load functions return the main data array
        return dataset  # (8192, 5, 170) - TIME x CHANNELS x INSTANCES

    except Exception as e:
        print(f"❌ Error loading 3-story data: {e}")
        return None


def get_available_datasets() -> Dict[str, str]:
    """
    Get list of available datasets for GUI.

    Returns
    -------
    datasets : dict
        Mapping of dataset names to descriptions.
    """
    return {
        "3story": "3-Story Structure Dataset (170 conditions)",
        "synthetic": "Synthetic Test Data (for debugging)",
    }


def create_synthetic_data() -> Dict[str, Any]:
    """
    Create synthetic data for testing workflows.

    Returns
    -------
    data : dict
        Synthetic dataset for testing.
    """
    # Create synthetic time series
    fs = 1000.0
    t = np.linspace(0, 8.192, 8192)

    # Baseline signal: damped oscillation + noise
    baseline_data = np.exp(-0.1 * t) * np.sin(
        2 * np.pi * 50 * t
    ) + 0.1 * np.random.randn(len(t))

    # Multiple conditions for PCA training
    baseline_conditions = []
    for i in range(20):
        # Add slight variations
        condition = baseline_data + 0.05 * np.random.randn(len(t))
        baseline_conditions.append(condition)

    return {
        "baseline_data": baseline_data,
        "baseline_conditions": baseline_conditions,
        "fs": fs,
        "channels": 1,
        "conditions": 20,
    }
