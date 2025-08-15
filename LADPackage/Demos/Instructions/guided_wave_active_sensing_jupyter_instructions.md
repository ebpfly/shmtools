# Guided Wave Active Sensing with Jupyter Extension

This guide shows how to recreate the mFUSE guided wave active sensing demo using the SHMTools Jupyter extension.

**Dataset**: Active sensing dataset with ultrasonic guided wave measurements  
**Goal**: Import active sensing data, process waveforms, apply arrival filtering, map geometry, and visualize damage localization results.

## Sequence

### Step 1: Import Active Sensing Dataset
1. Use the **SHM Function** dropdown in the notebook toolbar
2. Find and select **Import Active Sense Data** from the dropdown (category: Data Loading - Example Datasets)
3. Right-click on function parameters:
   - `filename=None` → leave as None (defaults to 'data_example_ActiveSense.mat')

**Inputs:**
- Filename = None (defaults to 'data_example_ActiveSense.mat')

**Output variables:**
- Waveform Base = waveform_base (baseline measurements)
- Waveform Test = waveform_test (test measurements with potential damage)
- Sensor Layout = sensor_layout (sensor coordinates and geometry)
- Pair List = pair_list (actuator-sensor pair combinations)
- Border Struct = border_struct (structure boundary definition)
- Sample Rate = sample_rate (sampling frequency)
- Actuation Waveform = actuation_waveform (excitation signal)
- Damage Location = damage_location (known damage coordinates)

### Step 2: Process Active Sensing Waveforms
1. Use dropdown to find **Process Active Sensing Waveforms** (category: Feature Extraction - Active Sensing)
2. Create sensor subset variable: `sensor_subset = list(range(1, 32, 5))` (1:5:31 in MATLAB)
3. Right-click on function parameters to link:
   - `sensor_subset=None` → select `sensor_subset`
   - `sensor_layout=None` → select `sensor_layout` (Step 1)
   - `pair_list=None` → select `pair_list` (Step 1)
   - `waveform_base=None` → select `waveform_base` (Step 1)
   - `waveform_test=None` → select `waveform_test` (Step 1)
   - `matched_waveform=None` → select `actuation_waveform` (Step 1)

**Inputs:**
- Sensor Subset = list(range(1, 32, 5)) (every 5th sensor from 1 to 31)
- Sensor Layout = sensor_layout (Step 1)
- Pair List = pair_list (Step 1)
- Waveform Base = waveform_base (Step 1)
- Waveform Test = waveform_test (Step 1)
- Matched Waveform = actuation_waveform (Step 1)

**Output variables:**
- Filter Result = filter_result (processed waveforms)
- Layout Subset = layout_subset (reduced sensor layout)
- Pair Subset = pair_subset (reduced pair list)

### Step 3: Arrival Filter
1. Find **Arrival Filter** in dropdown (category: Feature Extraction - Active Sensing)
2. Right-click on function parameters to link:
   - `filter_result=None` → select `filter_result` (Step 2)
   - `front_clip=450` → keep default value 450
   - `arrival_offset=450` → keep default value 450

**Inputs:**
- Filter Result = filter_result (Step 2)
- Front Clip = 450 (samples to clip from front)
- Arrival Offset = 450 (offset for arrival time detection)

**Output variables:**
- Filtered Waveforms = filtered_waveforms (envelope-filtered waveforms)

### Step 4: Map Active Sensing Geometry
1. Find **Map Active Sensing Geometry** in dropdown (category: Feature Extraction - Active Sensing)
2. Right-click on function parameters to link:
   - `velocity=66000` → keep default value 66000
   - `subset_window=1` → keep default value 1
   - `distance_allowance=inf` → set to `float('inf')`
   - `struct_cell=None` → select `border_struct` (Step 1)
   - `x_spacing=0.5` → keep default value 0.5
   - `y_spacing=0.5` → keep default value 0.5
   - `sample_rate=None` → select `sample_rate` (Step 1)
   - `offset=None` → select `actuation_waveform` (Step 1)
   - `data=None` → select `filtered_waveforms` (Step 3)
   - `sensor_pair_list=None` → select `pair_subset` (Step 2)
   - `sensor_layout=None` → select `layout_subset` (Step 2)

**Inputs:**
- Velocity = 66000 (guided wave velocity in structure)
- Subset Window = 1 (window size for data extraction)
- Distance Allowance = inf (maximum propagation distance)
- Struct Cell = border_struct (Step 1)
- X Spacing = 0.5 (grid spacing in x direction)
- Y Spacing = 0.5 (grid spacing in y direction)
- Sample Rate = sample_rate (Step 1)
- Offset = actuation_waveform (Step 1)
- Data = filtered_waveforms (Step 3)
- Sensor Pair List = pair_subset (Step 2)
- Sensor Layout = layout_subset (Step 2)

**Output variables:**
- X Matrix = x_matrix (x-coordinates of mapping grid)
- Y Matrix = y_matrix (y-coordinates of mapping grid)
- Combined Matrix = combined_matrix (combined damage indicator map)
- Data Map 2D = data_map_2d (2D damage localization map)

### Step 5: Plot Active Sensing Result
1. Find **Plot AS Result** in dropdown (category: Plotting - Active Sensing)
2. Right-click on function parameters to link:
   - `x_matrix=None` → select `x_matrix` (Step 4)
   - `y_matrix=None` → select `y_matrix` (Step 4)
   - `data_map_2d=None` → select `data_map_2d` (Step 4)
   - `combined_matrix=None` → select `combined_matrix` (Step 4)
   - `layout_subset=None` → select `layout_subset` (Step 2)

**Inputs:**
- X Matrix = x_matrix (Step 4)
- Y Matrix = y_matrix (Step 4)
- Data Map 2D = data_map_2d (Step 4)
- Combined Matrix = combined_matrix (Step 4)
- Layout Subset = layout_subset (Step 2)

**Output variables:**
- Plot handle (visualization of damage localization)

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

## Active Sensing Analysis Notes

**Guided Wave Principles:**
- Ultrasonic waves propagate through structure following specific modes
- Damage creates reflections and mode conversions at defect locations
- Time-of-flight measurements enable triangulation for damage localization
- Baseline subtraction removes environmental and operational variations

**Processing Workflow:**
- **Subset Selection**: Reduces computational load by using strategic sensor combinations
- **Matched Filtering**: Enhances signal-to-noise ratio for weak reflected signals
- **Arrival Filtering**: Isolates first arrival times for accurate time-of-flight measurement
- **Geometric Mapping**: Converts time-of-flight to spatial damage probability maps

**Visualization Features:**
- 2D damage localization map with probability contours
- Sensor layout overlay showing transducer positions
- Combined damage indicator highlighting most probable damage locations
- Border structure visualization for spatial reference