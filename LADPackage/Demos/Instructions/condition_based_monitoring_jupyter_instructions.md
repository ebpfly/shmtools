# Condition Based Monitoring with Jupyter Extension

This guide shows how to recreate the mFUSE condition based monitoring demo using the SHMTools Jupyter extension.

**Dataset**: Condition-based monitoring experimental data  
**Goal**: Import CBM data, estimate power spectral density via Welch's method, and plot results for rotating machinery diagnostics.

## Sequence

### Step 1: Import Condition Based Monitoring Data
1. Use the **SHM Function** dropdown in the notebook toolbar
2. Find and select **Import CBM Data** from the dropdown (category: Data Loading - Example Datasets)
3. Function is inserted automatically

**Output variables:**
- Dataset = dataset (TIME, CHANNELS, INSTANCES)  
- Damage States = damage_states (INSTANCES, 1)
- State List = state_list (INSTANCES, 1)
- Sampling Frequency = fs (scalar)

### Step 2: Power Spectral Density Via Welch's Method
1. Use dropdown to find **PSD Welch** (category: Feature Extraction - Spectral Analysis)
2. Right-click on function parameters to link:
   - `data=None` → select `dataset` (Step 1)
   - `window=None` → leave as None (default)
   - `noverlap=None` → leave as None (default)
   - `nfft=None` → leave as None (default)
   - `fs=None` → select `fs` (Step 1)
   - `detrend=None` → leave as None (default)

**Inputs:**
- Time Series Data = dataset (Step 1)
- Window = None (default Hamming)
- Number of Overlapping Points = None (default 50% overlap)
- FFT Length = None (default)
- Sampling Frequency = fs (Step 1)
- Detrend = None (default 'constant')

**Output variables:**
- PSD Matrix = psd_matrix (FREQUENCIES, CHANNELS, INSTANCES)
- Frequency Vector = f (FREQUENCIES, 1)
- Is One Sided = is_one_sided (boolean)

### Step 3: Plot Power Spectral Density  
1. Find **Plot PSD** in dropdown (category: Plotting - Spectral Analysis)
2. Right-click on function parameters to link:
   - `psd_matrix=None` → select `psd_matrix` (Step 2)
   - `channel_index=1` → change to `3`
   - `is_one_sided=True` → select `is_one_sided` (Step 2)
   - `f=None` → select `f` (Step 2)
   - `plot_type=None` → leave as None (default)
   - `units=None` → leave as None (default)
   - `title=None` → leave as None (default)

**Inputs:**
- PSD Matrix = psd_matrix (Step 2)
- Channel Index = 3
- Is One Sided = is_one_sided (Step 2)
- Frequency Vector = f (Step 2)
- Plot Type = None (default 'loglog')
- Units = None (default)
- Title = None (default)

**Output variables:**
- Axes Handle = axes_handle

## How to Use the Extension

1. **Function Selection**: Use the "SHM Function:" dropdown in the notebook toolbar to browse categorized functions
2. **Parameter Linking**: Right-click on function parameters (like `data=None`) to link them to variables in your notebook
3. **Variable Recommendations**: The context menu shows recommended variables based on parameter names and types
4. **Automatic Code**: Functions are inserted as complete, executable Python code

## Quick Reference

**Popular Functions (Keyboard Shortcuts):**
- Ctrl+Shift+F: Open function search
- Ctrl+Shift+P: Show popular functions popup  
- Ctrl+Shift+L: Smart parameter linking

**Context Menu Features:**
- Right-click parameter values to see compatible variables
- Green background = recommended variables
- Gray background = other available variables
- Shows variable types and source cells

## CBM Analysis Notes

**Frequency Domain Analysis:**
- Look for characteristic defect frequencies in bearing fault detection
- Identify gear mesh frequencies and sidebands for gear analysis
- Monitor fundamental frequency amplitudes for imbalance detection
- Analyze harmonic content patterns for misalignment diagnosis

**Welch's Method Benefits:**
- Reduces noise through averaging of overlapped segments
- Provides better frequency resolution than simple FFT
- Windowing reduces spectral leakage effects
- One-sided spectrum for real-valued vibration signals