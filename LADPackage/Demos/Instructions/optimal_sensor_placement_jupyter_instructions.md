# Demo: Optimal Sensor Placement

**Dataset**: FEM-computed mode shapes for representative 3D structure

**Goal**: Compute the 12-sensor optimal arrangements using the Fisher information method and the maximum norm method, plotting the resulting arrangements on the structure's geometry.

## Sequence

### Step 1: Import Modal Optimal Sensor Placement Dataset (Examples\ExampleData)
No Inputs  
**Output**: Nodes = (NODEINFO, NODES, INSTANCES)  
**Output**: Elements = (NODES, ELEMENTS)  
**Output**: Mode Shapes = (DOFS, MODES)  
**Output**: Response DOF = (DOFS, DOFINFO)

### Step 2: Plot Nodal Response (LADPackage\Optimal Sensor Placement\)
**Input**: Geometry_Layout = Nodes (Step 1)  
**Input**: ModeShapes = Mode Shapes (Step 1)  
**Input**: Response DOF = Response DOF (Step 1)  
**Input**: Elements = Elements (Step 1)  
**Input**: Mode Number = 3  
**Output**: Axes Handle

### Step 3: Plot Nodal Response (LADPackage\Optimal Sensor Placement\)
**Input**: Geometry_Layout = Nodes (Step 1)  
**Input**: ModeShapes = Mode Shapes (Step 1)  
**Input**: Response DOF = Response DOF (Step 1)  
**Input**: Elements = Elements (Step 1)  
**Input**: Mode Number = 10  
**Output**: Axes Handle

### Step 4: OSP Fisher Information EIV (Auxiliary\SensorSupport\OptimalSensorPlacement)
**Input**: # Of Sensors = 12  
**Input**: Mode Shapes = Mode Shapes (Step 1)  
**Input**: Covariance Matrix = I (Default)  
**Output**: Optimal List = (SENSORS)  
**Output**: Fisher Determinant = scalar

### Step 5: Plot Sensors With Mesh (LADPackage\Optimal Sensor Placement\)
**Input**: Elements = Elements (Step 1)  
**Input**: Nodes = Nodes (Step 1)  
**Input**: Sensor Indices = Optimal List (Step 4)  
**Input**: Response DOF = Response DOF (Step 1)  
**Output**: Axes Handle

### Step 6: OSP Using Maximum Norm (Auxiliary\SensorSupport\OptimalSensorPlacement)
**Input**: # Of Sensors = 12  
**Input**: Mode Shapes = Mode Shapes (Step 1)  
**Input**: Weights = 13:-1:1  
**Input**: Dueling Distance = 20  
**Input**: Response DOF = Response DOF (Step 1)  
**Input**: Geometry Layout = Nodes (Step 1)  
**Output**: Optimal List = (SENSORS)  
**Output**: Fisher Determinant = scalar

### Step 7: Plot Sensors With Mesh (LADPackage\Optimal Sensor Placement\)
**Input**: Elements = Elements (Step 1)  
**Input**: Nodes = Nodes (Step 1)  
**Input**: Sensor Indices = Optimal List (Step 6)  
**Input**: Response DOF = Response DOF (Step 1)  
**Output**: Axes Handle