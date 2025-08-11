# Optimal Sensor Placement with Jupyter Extension

This guide shows how to recreate the mFUSE optimal sensor placement demo using the SHMTools Jupyter extension.

**Dataset**: Modal Optimal Sensor Placement Dataset  
**Goal**: Compute optimal 12-sensor arrangements using Fisher Information and Maximum Norm methods, and plot the resulting arrangements on the structure's geometry.

## Sequence

### Step 1: Import Modal Optimal Sensor Placement Dataset
1. Use the **SHM Function** dropdown in the notebook toolbar
2. Find and select **Import Modal OSP** from the dropdown (category: Data Loading - Example Datasets)
3. Function is inserted automatically

**Output variables:**
- Node Layout = node_layout (node coordinates and connectivity)
- Elements = elements (finite element connectivity matrix)
- Mode Shapes = mode_shapes (structural mode shapes matrix)
- Response DOF = resp_dof (degrees of freedom for response measurement)

### Step 2: Plot Nodal Response (Mode 3)
1. Use dropdown to find **Plot Nodal Response** (category: Plotting - Modal Analysis)
2. Right-click on function parameters to link:
   - `node_layout=None` → select `node_layout` (Step 1)
   - `mode_shapes=None` → select `mode_shapes` (Step 1)
   - `resp_dof=None` → select `resp_dof` (Step 1)
   - `elements=None` → select `elements` (Step 1)
   - `mode_number=1` → change to `3`

**Inputs:**
- Node Layout = node_layout (Step 1)
- Mode Shapes = mode_shapes (Step 1)
- Response DOF = resp_dof (Step 1)
- Elements = elements (Step 1)
- Mode Number = 3

**Output variables:**
- Axes Handle = axes_handle_mode3 (plot handle for mode 3 visualization)

### Step 3: Plot Nodal Response (Mode 10)
1. Use dropdown to find **Plot Nodal Response** (category: Plotting - Modal Analysis)
2. Right-click on function parameters to link:
   - `node_layout=None` → select `node_layout` (Step 1)
   - `mode_shapes=None` → select `mode_shapes` (Step 1)
   - `resp_dof=None` → select `resp_dof` (Step 1)
   - `elements=None` → select `elements` (Step 1)
   - `mode_number=1` → change to `10`

**Inputs:**
- Node Layout = node_layout (Step 1)
- Mode Shapes = mode_shapes (Step 1)
- Response DOF = resp_dof (Step 1)
- Elements = elements (Step 1)
- Mode Number = 10

**Output variables:**
- Axes Handle = axes_handle_mode10 (plot handle for mode 10 visualization)

### Step 4: OSP Fisher Information EIV
1. Find **OSP Fisher Info EIV** in dropdown (category: Feature Extraction - Optimal Sensor Placement)
2. Right-click on function parameters to link:
   - `num_sensors=12` → keep default value 12
   - `mode_shapes=None` → select `mode_shapes` (Step 1)
   - `weights=None` → leave as None (default equal weighting)

**Inputs:**
- Number of Sensors = 12
- Mode Shapes = mode_shapes (Step 1)
- Weights = None (default equal mode weighting)

**Output variables:**
- Optimal List = op_list_fisher (optimal sensor DOF locations)
- Determinant Q = det_q (Fisher Information determinant values)

### Step 5: Plot Sensors With Mesh (Fisher Information)
1. Find **Plot Sensors With Mesh** in dropdown (category: Plotting - Modal Analysis)
2. Right-click on function parameters to link:
   - `elements=None` → select `elements` (Step 1)
   - `node_layout=None` → select `node_layout` (Step 1)
   - `op_list=None` → select `op_list_fisher` (Step 4)
   - `resp_dof=None` → select `resp_dof` (Step 1)

**Inputs:**
- Elements = elements (Step 1)
- Node Layout = node_layout (Step 1)
- Optimal List = op_list_fisher (Step 4)
- Response DOF = resp_dof (Step 1)

**Output variables:**
- Axes Handle 1 = axes_handle_fisher_1 (mesh plot handle)
- Axes Handle 2 = axes_handle_fisher_2 (sensor plot handle)
- Sensor Handle = sensor_handle_fisher (sensor markers handle)

### Step 6: OSP Using Maximum Norm
1. Find **OSP Max Norm** in dropdown (category: Feature Extraction - Optimal Sensor Placement)
2. Create weights variable: `weights = list(range(13, 0, -1))` (13:-1:1 in MATLAB)
3. Right-click on function parameters to link:
   - `num_sensors=12` → keep default value 12
   - `mode_shapes=None` → select `mode_shapes` (Step 1)
   - `weights=None` → select `weights`
   - `dueling_distance=20` → keep default value 20
   - `resp_dof=None` → select `resp_dof` (Step 1)
   - `node_layout=None` → select `node_layout` (Step 1)

**Inputs:**
- Number of Sensors = 12
- Mode Shapes = mode_shapes (Step 1)
- Weights = list(range(13, 0, -1)) (linear weighting, modes 1-13)
- Dueling Distance = 20 (minimum sensor separation)
- Response DOF = resp_dof (Step 1)
- Node Layout = node_layout (Step 1)

**Output variables:**
- Optimal List = op_list_maxnorm (optimal sensor DOF locations)

### Step 7: Plot Sensors With Mesh (Maximum Norm)
1. Find **Plot Sensors With Mesh** in dropdown (category: Plotting - Modal Analysis)
2. Right-click on function parameters to link:
   - `elements=None` → select `elements` (Step 1)
   - `node_layout=None` → select `node_layout` (Step 1)
   - `op_list=None` → select `op_list_maxnorm` (Step 6)
   - `resp_dof=None` → select `resp_dof` (Step 1)

**Inputs:**
- Elements = elements (Step 1)
- Node Layout = node_layout (Step 1)
- Optimal List = op_list_maxnorm (Step 6)
- Response DOF = resp_dof (Step 1)

**Output variables:**
- Axes Handle 1 = axes_handle_maxnorm_1 (mesh plot handle)
- Axes Handle 2 = axes_handle_maxnorm_2 (sensor plot handle)
- Sensor Handle = sensor_handle_maxnorm (sensor markers handle)

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

## Optimal Sensor Placement Analysis Notes

**Fisher Information Method:**
- Maximizes determinant of Fisher Information Matrix
- Uses Equivalent Independence method for efficient computation
- Optimal for parameter estimation accuracy
- Results in well-conditioned mode shape matrix

**Maximum Norm Method:**
- Maximizes weighted norm of modal response
- Linear weighting emphasizes lower modes (typically more important)
- Dueling distance ensures minimum sensor separation
- Prevents sensor clustering in high-response regions

**Method Comparison:**
- **Fisher Information**: Better for modal parameter identification
- **Maximum Norm**: Better for overall structural response measurement
- **Dueling**: Practical constraint for sensor installation
- **Mode Weighting**: Emphasizes modes of engineering interest

**Visualization Features:**
- Finite element mesh display with node-element connectivity
- Mode shape visualization with response amplitude coloring
- Optimal sensor locations highlighted on structural geometry
- Comparative visualization of different OSP methods