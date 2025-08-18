# Demo: Guided Wave Active Sensing

**Dataset**: Active sensing dataset packaged with SHMTools

**Goal**: Localize defect using active sensing waveform measurements

## Sequence

### Step 1: Import Active Sensing Dataset (LADPackage\Active Sensing)
**Input**: File Name: 'data_example_ActiveSense.mat' (Default)  
**Output**: Baseline Waveforms = (TIME, SENSORPAIRS)  
**Output**: Test Waveforms = (TIME, SENSORPAIRS)  
**Output**: Sensor Layout = (SENSORINFO, SENSORS)  
**Output**: Sensor Pair = (SENSORIDS, PAIRS)  
**Output**: Border Structure = (struct)  
**Output**: Sample Rate = (scalar)  
**Output**: Actuation Waveform = (TIME, 1)  
**Output**: Damage Location = (COORDINATES, 1)

### Step 2: Process Active Sensing Waveforms (LADPackage\ Active Sensing\)
**Input**: Sensor Subset List = 1:31  
**Input**: Sensor Layout = Sensor Layout (Step 1)  
**Input**: Sensor Pair List = Sensor Pair List (Step 1)  
**Input**: Baseline Waveforms = Baseline Waveforms (Step 1)  
**Input**: Test Waveforms = Test Waveforms (Step 1)  
**Input**: Excitation Waveform = Actuation Waveform (Step 1)  
**Output**: Filter Result  
**Output**: Layout Subset  
**Output**: Sensor Pair Subset

### Step 3: Arrival Filter (LADPackage\Active Sensing\)
**Input**: Waveforms = Filter Result (Step 2)  
**Input**: Front Clip = 450 (Default)  
**Input**: ArrivalOffset = 450 (Default)  
**Output**: Filtered Waveforms

### Step 4: Map Active Sensing (LADPackage\Active Sensing\)
**Input**: Velocity = 66000 (Default)  
**Input**: Subset Window = 1 (Default)  
**Input**: Distance Allowance = inf (Default)  
**Input**: Geometry = Border Structure (Step 1)  
**Input**: X Spacing = .5 (Default)  
**Input**: Y Spacing = .5 (Default)  
**Input**: Sample Rate = Sample Rate (Step 1)  
**Input**: Actuation Waveform = Actuation Waveforms (Step 1)  
**Input**: Data = Filtered Waveforms (Step 3)  
**Input**: Sensor Pair List = Sensor Pair Subset (Step 2)  
**Input**: Sensor Layout = Layout Subset (Step 2)  
**Output**: X Matrix  
**Output**: Y Matrix  
**Output**: Data Map 2D  
**Output**: Combined Geometry

### Step 5: Plot Active Sensing Map (LADPackage\Active Sensing)
**Input**: X Matrix = X Matrix (Step 4)  
**Input**: Y Matrix = Y Matrix (Step 4)  
**Input**: Data Map 2D = Data Map 2D (Step 4)  
**Input**: Border = Combined Geometry (Step 4)  
**Input**: Sensor Subset Layout = Layout Subset (Step 2)