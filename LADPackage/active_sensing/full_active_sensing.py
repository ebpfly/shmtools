# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Full Active Sensing
#
# ## Introduction
#
# This notebook demonstrates a complete active sensing workflow for structural health monitoring using guided waves. The process includes:
#
# 1. **Data Import**: Loading baseline and test waveforms from piezoelectric sensors
# 2. **Waveform Processing**: Baseline subtraction and matched filtering
# 3. **Arrival Time Filtering**: Maximum likelihood estimation of wave arrival times
# 4. **Geometry Mapping**: Mapping processed signals to structural geometry
# 5. **Visualization**: Creating damage localization images
#
# Active sensing uses guided ultrasonic waves propagating through a structure to detect and locate damage. By comparing baseline (undamaged) and test (potentially damaged) measurements, we can identify changes in wave propagation that indicate structural damage.
#
# **Auto-Generated Script by mFUSE**
#
# Author: Auto-Generated Script by mFUSE  
# Date Created: 3/19/2016  
# Python Conversion: 2025

# %% [markdown]
# ## Setup and Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add project root to path for imports
notebook_dir = Path().resolve()
project_root = notebook_dir.parent.parent
sys.path.insert(0, str(project_root))

print(f"Working directory: {notebook_dir}")
print(f"Project root: {project_root}")

# Import LADPackage active sensing utilities
from LADPackage.active_sensing.active_sensing_utils import (
    import_active_sense_data,
    process_active_sensing_waveforms,
    arrival_filter,
    map_active_sensing_geometry,
    plot_as_result
)

print("\nSuccessfully imported all required functions!")

# %% [markdown]
# ## Step 1: Import Active Sensing Dataset
#
# Import active sensing dataset #1. This dataset contains:
# - Baseline waveforms (undamaged state)
# - Test waveforms (with simulated damage)
# - Sensor layout and positions
# - Structure boundary definitions
# - Excitation waveform used

# %%
# Import the active sensing data
try:
    (waveformBase_1out, waveformTest_1out, sensorLayout_1out, pairList_1out, 
     borderStruct_1out, sampleRate_1out, actuationWaveform_1out, 
     damageLocation_1out) = import_active_sense_data()
    
    print("Dataset loaded successfully!")
    print(f"\nData dimensions:")
    print(f"  Baseline waveforms: {waveformBase_1out.shape}")
    print(f"  Test waveforms: {waveformTest_1out.shape}")
    print(f"  Number of sensors: {sensorLayout_1out.shape[1]}")
    print(f"  Number of sensor pairs: {pairList_1out.shape[1]}")
    print(f"  Sample rate: {sampleRate_1out} Hz")
    print(f"  Damage location: ({damageLocation_1out[0]:.1f}, {damageLocation_1out[1]:.1f}) cm")
    
except FileNotFoundError as e:
    print(f"\n⚠️ Data file not found: {e}")
    print("\nPlease download 'data_example_ActiveSense.mat' and place it in:")
    print("  LADPackage/active_sensing/data/")
    print("  or")
    print("  examples/data/")
    raise

# %% [markdown]
# ## Step 2: Process Active Sensing Waveforms
#
# Process active sensing waveforms by:
# 1. Selecting a subset of sensors for computational efficiency
# 2. Performing baseline subtraction to highlight changes
# 3. Applying matched filtering to enhance signal-to-noise ratio

# %%
# Define sensor subset (use every 5th sensor from 1 to 31)
# Note: Using 1-based indexing for MATLAB compatibility
sensorSubset_1in_2in = np.arange(1, 32, 5)  # 1:5:31 in MATLAB notation

print(f"Selected sensor subset: {sensorSubset_1in_2in}")
print(f"Number of sensors in subset: {len(sensorSubset_1in_2in)}")

# Process the waveforms
(filterResult_3out_2out, layoutSubset_1out_2out, 
 pairSubset_1out_2out) = process_active_sensing_waveforms(
    sensorSubset_1in_2in, 
    sensorLayout_1out, 
    pairList_1out, 
    waveformBase_1out, 
    waveformTest_1out, 
    actuationWaveform_1out
)

print(f"\nProcessing results:")
print(f"  Filtered waveforms shape: {filterResult_3out_2out.shape}")
print(f"  Layout subset shape: {layoutSubset_1out_2out.shape}")
print(f"  Pair subset shape: {pairSubset_1out_2out.shape}")

# %% [markdown]
# ## Step 3: Arrival Filter
#
# Filter guided wave envelopes to identify first arrivals. This step:
# - Removes initial noise by clipping the front of the signal
# - Applies maximum likelihood estimation to detect wave arrivals
# - Compensates for the excitation waveform width

# %%
# Define arrival filter parameters
frontClip_3in = 450  # Samples to skip at beginning
arrivalOffset_3in = 450  # Offset for excitation width

print(f"Arrival filter parameters:")
print(f"  Front clip: {frontClip_3in} samples")
print(f"  Arrival offset: {arrivalOffset_3in} samples")

# Apply arrival filter
filteredWaveforms_3out = arrival_filter(
    filterResult_3out_2out, 
    frontClip_3in, 
    arrivalOffset_3in
)

print(f"\nFiltered waveforms shape: {filteredWaveforms_3out.shape}")
print(f"Min value: {np.min(filteredWaveforms_3out):.2f}")
print(f"Max value: {np.max(filteredWaveforms_3out):.2f}")

# %% [markdown]
# ## Step 4: Map Active Sensing Geometry
#
# Map processed active sensing waveforms to the structural geometry. This creates a 2D image showing damage likelihood at each point based on:
# - Wave propagation velocity
# - Time-of-flight calculations
# - Geometric constraints from structure boundaries

# %%
# Define mapping parameters
velocity_5in_4in = 66000  # Wave velocity in cm/s
distanceAllowance_10in_4in = np.inf  # No distance restriction

print(f"Geometry mapping parameters:")
print(f"  Wave velocity: {velocity_5in_4in} cm/s")
print(f"  Distance allowance: {distanceAllowance_10in_4in}")

# Map to geometry
(xMatrix_3out_4out, yMatrix_3out_4out, combinedMatrix_2out_4out, 
 dataMap2D_13out_4out) = map_active_sensing_geometry(
    velocity_5in_4in,
    None,  # Default subset window
    distanceAllowance_10in_4in,
    borderStruct_1out,
    None,  # Default X spacing
    None,  # Default Y spacing
    sampleRate_1out,
    actuationWaveform_1out,
    filteredWaveforms_3out,
    pairSubset_1out_2out,
    layoutSubset_1out_2out
)

print(f"\nMapping results:")
print(f"  X matrix shape: {xMatrix_3out_4out.shape}")
print(f"  Y matrix shape: {yMatrix_3out_4out.shape}")
print(f"  Data map shape: {dataMap2D_13out_4out.shape}")
print(f"  Non-NaN pixels: {np.sum(~np.isnan(dataMap2D_13out_4out))}")

# %% [markdown]
# ## Step 5: Plot Active Sensing Map
#
# Visualize the mapped active sensing result along with:
# - Structure boundaries
# - Sensor positions
# - Damage likelihood heat map
#
# Areas with higher values (warmer colors) indicate higher likelihood of damage.

# %%
# Create the active sensing visualization
plot_as_result(
    xMatrix_3out_4out, 
    yMatrix_3out_4out, 
    dataMap2D_13out_4out, 
    combinedMatrix_2out_4out, 
    layoutSubset_1out_2out
)

# Add the true damage location if available
if damageLocation_1out is not None and len(damageLocation_1out) == 2:
    plt.plot(damageLocation_1out[0], damageLocation_1out[1], 'k*', 
             markersize=15, markeredgewidth=2, label='True Damage')
    plt.legend()
    plt.show()
    print(f"\nTrue damage location marked with red star at ({damageLocation_1out[0]:.1f}, {damageLocation_1out[1]:.1f}) cm")

# %% [markdown]
# ## Analysis Summary
#
# This notebook demonstrated a complete active sensing workflow for damage detection and localization:
#
# 1. **Data Loading**: Successfully imported baseline and test waveforms from piezoelectric sensors
# 2. **Signal Processing**: Applied baseline subtraction and matched filtering to enhance damage signatures
# 3. **Arrival Detection**: Used maximum likelihood estimation to identify guided wave arrivals
# 4. **Imaging**: Mapped processed signals to structural geometry for damage localization
# 5. **Visualization**: Created heat map showing damage likelihood across the structure
#
# The resulting image shows areas of high signal change that correlate with structural damage. The technique is particularly effective for:
# - Composite structures
# - Metallic plates and shells
# - Complex geometries with known boundaries
#
# ### Key Parameters
# - **Wave Velocity**: 66,000 cm/s (typical for aluminum)
# - **Sensor Subset**: Every 5th sensor used for efficiency
# - **Front Clipping**: 450 samples to remove initial noise
# - **Arrival Offset**: 450 samples to account for excitation width
#
# ### References
# Flynn EB, Todd MD, Wilcox PD, Drinkwater BW, Croxford AJ. Maximum-likelihood estimation of damage location in guided-wave structural health monitoring. Proceedings of the Royal Society A, 2011.

# %%
print("\n" + "="*60)
print("Full Active Sensing Analysis Complete!")
print("="*60)
