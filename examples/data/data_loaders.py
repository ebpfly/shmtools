"""
Simplified data loader functions that wrap the import functions.

These provide a simpler interface that returns dictionaries with metadata,
similar to the load functions in shmtools.utils.data_loading.
"""

import numpy as np
from typing import Dict, Any
from .data_imports import (
    import_3story_structure_shm,
    import_cbm_data_shm,
    import_active_sense1_shm,
    import_sensor_diagnostic_shm,
    import_modal_osp_shm,
)


def load_3story_data() -> Dict[str, Any]:
    """
    Load 3-story structure dataset with metadata.
    
    Returns
    -------
    data : dict
        Dictionary with keys:
        - 'dataset': (8192, 5, 170) acceleration data array
        - 'damage_states': (170, 1) binary damage labels
        - 'state_list': (1, 170) state identifiers
        - 'fs': sampling frequency (2000 Hz)
        - 'channels': list of channel names
        - 'conditions': array of condition numbers (1-170)
        - 'state_descriptions': dictionary mapping states to descriptions
        - 'description': dataset description string
    """
    dataset, damage_states, state_list = import_3story_structure_shm()
    
    # Create detailed state descriptions
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
        "damage_states": damage_states,
        "state_list": state_list,
        "fs": 2000.0,  # Sampling frequency
        "channels": ["Force", "Ch2", "Ch3", "Ch4", "Ch5"],
        "conditions": np.arange(1, 171),
        "damage_states_array": np.repeat(np.arange(1, 18), 10),  # 17 states, 10 tests each
        "state_descriptions": state_descriptions,
        "description": "3-story structure base excitation data (LANL)",
    }


def load_cbm_data() -> Dict[str, Any]:
    """
    Load condition-based monitoring dataset with metadata.
    
    Returns
    -------
    data : dict
        Dictionary with keys:
        - 'dataset': vibration measurements
        - 'damage_states': damage state labels
        - 'state_list': state identifiers
        - 'fs': sampling frequency
        - 'description': dataset description
    """
    dataset, damage_states, state_list, fs = import_cbm_data_shm()
    
    return {
        "dataset": dataset,
        "damageStates": damage_states,  # Keep original naming for compatibility
        "stateList": state_list,
        "Fs": fs,  # Keep original naming
        "fs": fs,  # Also provide lowercase version
        "channels": ["Tachometer", "Gearbox Accel", "Top Bearing Accel", "Side Bearing Accel"],
        "description": "Condition-based monitoring data from rotating machinery",
    }


def load_sensor_diagnostic_data() -> Dict[str, Any]:
    """
    Load sensor diagnostic dataset with metadata.
    
    Returns
    -------
    data : dict
        Dictionary with keys:
        - 'healthy_data': healthy sensor measurements
        - 'example_sensor_data': mixed health state measurements
        - 'sd_ex': healthy data (original MATLAB name)
        - 'sd_ex_broken': example data (original MATLAB name)
        - 'description': dataset description
    """
    healthy_data, example_sensor_data = import_sensor_diagnostic_shm()
    
    return {
        "healthy_data": healthy_data,
        "example_sensor_data": example_sensor_data,
        "sd_ex": healthy_data,  # Keep original MATLAB names for compatibility
        "sd_ex_broken": example_sensor_data,
        "description": "Piezoelectric sensor impedance measurements for health monitoring",
    }


def load_modal_osp_data() -> Dict[str, Any]:
    """
    Load modal optimal sensor placement dataset with metadata.
    
    Returns
    -------
    data : dict
        Dictionary with keys:
        - 'node_layout': node information
        - 'elements': element connectivity
        - 'mode_shapes': modal displacement vectors
        - 'resp_dof': DOF definitions
        - 'nodeLayout': (original MATLAB name)
        - 'modeShapes': (original MATLAB name)
        - 'respDOF': (original MATLAB name)
        - 'description': dataset description
    """
    node_layout, elements, mode_shapes, resp_dof = import_modal_osp_shm()
    
    return {
        "node_layout": node_layout,
        "elements": elements,
        "mode_shapes": mode_shapes,
        "resp_dof": resp_dof,
        # Keep original MATLAB names for compatibility
        "nodeLayout": node_layout,
        "modeShapes": mode_shapes,
        "respDOF": resp_dof,
        "description": "Modal analysis data for optimal sensor placement studies",
    }


def load_active_sense_data() -> Dict[str, Any]:
    """
    Load active sensing dataset with metadata.
    
    Returns
    -------
    data : dict
        Dictionary with all active sensing data components
    """
    (waveform_base, waveform_test, sensor_layout, pair_list,
     border_struct, sample_rate, actuation_waveform, damage_location) = import_active_sense1_shm()
    
    return {
        "waveform_base": waveform_base,
        "waveform_test": waveform_test,
        "sensor_layout": sensor_layout,
        "pair_list": pair_list,
        "border_struct": border_struct,
        "sample_rate": sample_rate,
        "actuation_waveform": actuation_waveform,
        "damage_location": damage_location,
        # Also provide camelCase versions for compatibility
        "waveformBase": waveform_base,
        "waveformTest": waveform_test,
        "sensorLayout": sensor_layout,
        "pairList": pair_list,
        "borderStruct": border_struct,
        "sampleRate": sample_rate,
        "actuationWaveform": actuation_waveform,
        "damageLocation": damage_location,
        "description": "Active sensing guided wave data from plate structure",
    }


def import_3story_structure_sub_floors(floor_numbers=None):
    """
    LADPackage-compatible version of 3-story structure data import.
    
    This function mimics the LADPackage import_3StoryStructure_subFloors.m
    function for compatibility with LADPackage demo scripts.
    
    Parameters
    ----------
    floor_numbers : list or int, optional
        Which floors/channels to include. Default is [1, 2, 3, 4, 5] (all channels).
        
    Returns
    -------
    dataset : ndarray
        Time series data array
    damage_states : ndarray
        Binary damage labels (0=undamaged, 1=damaged)
    state_list : ndarray  
        State identifiers for each instance
    """
    # Load full 3-story data
    data = load_3story_data()
    dataset = data['dataset']
    
    # Handle floor_numbers parameter
    if floor_numbers is None or (isinstance(floor_numbers, list) and len(floor_numbers) == 0):
        floor_numbers = [1, 2, 3, 4, 5]  # All channels (MATLAB 1-based indexing)
    elif isinstance(floor_numbers, int):
        floor_numbers = [floor_numbers]
    
    # Convert MATLAB 1-based indexing to Python 0-based
    channel_indices = [f - 1 for f in floor_numbers]
    
    # Select specified channels
    dataset = dataset[:, channel_indices, :]
    
    # Create state list (transpose to match MATLAB output)
    state_list = data['damage_states_array'].reshape(1, -1)  
    
    # Create binary damage states (first 90 undamaged, rest damaged)
    damage_states = np.concatenate([
        np.zeros(90, dtype=bool),
        np.ones(80, dtype=bool)
    ]).reshape(-1, 1)
    
    return dataset, damage_states, state_list