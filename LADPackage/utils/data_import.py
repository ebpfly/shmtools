"""
LADPackage data import functions.

These functions provide LADPackage-compatible data loading that matches
the original MATLAB interface for seamless conversion of demo scripts.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Tuple, List, Union, Optional

# Add project root to path to access examples.data
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.data.data_imports import import_3story_structure_shm


def import_3story_structure_sub_floors(floor_numbers: Optional[Union[List[int], int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LADPackage-compatible version of 3-story structure data import.
    
    This function mimics the LADPackage import_3StoryStructure_subFloors.m
    function for compatibility with LADPackage demo scripts.
    
    .. meta::
        :category: Data Import - LADPackage
        :matlab_equivalent: import_3StoryStructure_subFloors
        :complexity: Basic
        :data_type: Time Series
        :output_type: Dataset
        :display_name: Import 3 Story Structure Dataset
        :verbose_call: [Dataset, Damage States, List of States] = Import 3 Story Structure Dataset (Floor Numbers)
    
    Parameters
    ----------
    floor_numbers : list of int, int, or None, optional
        Which floors/channels to include. Uses MATLAB 1-based indexing:
        - None or [] : All channels [1, 2, 3, 4, 5] (default)
        - int : Single channel (e.g., 5 for channel 5)  
        - list : Multiple channels (e.g., [2, 3, 4] for channels 2-4)
        
        .. gui::
            :widget: array_input
            :description: Floor numbers to import (1=Force, 2-5=Accelerometers)
            :default: [1, 2, 3, 4, 5]
        
    Returns
    -------
    dataset : ndarray, shape (time, channels, instances)
        Time series data array with selected channels.
        
    damage_states : ndarray, shape (instances, 1)
        Binary damage labels (0=undamaged, 1=damaged).
        
    state_list : ndarray, shape (1, instances)
        State identifiers for each instance (1-17).
        
    Notes
    -----
    This function provides exact compatibility with the LADPackage MATLAB function
    import_3StoryStructure_subFloors.m. Key features:
    
    - **MATLAB indexing**: Uses 1-based floor numbering (1=Force, 2-5=Accel)
    - **Binary damage states**: 90 undamaged (0) + 80 damaged (1) instances  
    - **State list**: Damage state IDs (1-17) for each of 170 instances
    - **Default behavior**: Empty input [] selects all 5 channels
    
    The 3-story structure dataset contains:
    - **170 instances**: 17 damage states × 10 repetitions each
    - **5 channels**: Force input + 4 floor accelerometers
    - **8192 time points**: Sampled at 2000 Hz (4.096 seconds)
    
    Damage States:
    - States 1-9: Various undamaged configurations (mass, stiffness changes)
    - States 10-17: Progressive damage scenarios (gap formation)
    
    Examples
    --------
    Import all channels (LADPackage default):
    
    >>> dataset, damage_states, state_list = import_3story_structure_sub_floors([])
    >>> print(f"Dataset shape: {dataset.shape}")  # (8192, 5, 170)
    >>> print(f"Channels: Force + 4 accelerometers")
    
    Import only accelerometer channels:
    
    >>> dataset, damage_states, state_list = import_3story_structure_sub_floors([2, 3, 4, 5])
    >>> print(f"Dataset shape: {dataset.shape}")  # (8192, 4, 170)
    >>> print(f"Accelerometer channels only")
    
    Import single channel:
    
    >>> dataset, damage_states, state_list = import_3story_structure_sub_floors(5)
    >>> print(f"Dataset shape: {dataset.shape}")  # (8192, 1, 170)
    >>> print(f"Channel 5 (top floor accelerometer)")
    
    Check damage distribution:
    
    >>> print(f"Undamaged instances: {np.sum(damage_states == 0)}")  # 90
    >>> print(f"Damaged instances: {np.sum(damage_states == 1)}")    # 80
    >>> print(f"Unique states: {np.unique(state_list)}")             # [1, 2, ..., 17]
    
    References
    ----------
    .. [1] LADPackage Documentation, Los Alamos National Laboratory
    .. [2] Figueiredo, E., et al. (2009). "Structural Health Monitoring Algorithm 
           Comparisons using Standard Data Sets." LA-14393.
    """
    # Load full 3-story data
    dataset, damage_states_binary, state_list_matlab = import_3story_structure_shm()
    
    # Create damage_states_array for compatibility (1-17 state IDs)
    damage_states_array = np.repeat(np.arange(1, 18), 10)  # 17 states, 10 tests each
    
    # Handle floor_numbers parameter (MATLAB compatibility)
    if floor_numbers is None or (isinstance(floor_numbers, list) and len(floor_numbers) == 0):
        # Default: all channels [1, 2, 3, 4, 5] in MATLAB indexing
        floor_numbers = [1, 2, 3, 4, 5]
    elif isinstance(floor_numbers, int):
        # Single channel
        floor_numbers = [floor_numbers]
    
    # Convert MATLAB 1-based indexing to Python 0-based
    channel_indices = [f - 1 for f in floor_numbers]
    
    # Validate channel indices
    max_channels = dataset.shape[1]
    for idx in channel_indices:
        if idx < 0 or idx >= max_channels:
            raise ValueError(f"Invalid channel index {idx+1}. Valid range: 1-{max_channels}")
    
    # Select specified channels
    dataset = dataset[:, channel_indices, :]
    
    # Use the state list from import function (already in correct format)
    state_list = state_list_matlab  
    
    # Use the damage states from import function (already in correct format)
    damage_states = damage_states_binary
    
    return dataset, damage_states, state_list