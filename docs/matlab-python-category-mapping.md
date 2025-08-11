# MATLAB-Python Category Mapping for SHMTools

This document provides a comprehensive mapping between the MATLAB SHMTools function organization and the Python version categories to ensure users familiar with the MATLAB version can easily find functions in the new JupyterLab extension.

## Summary of Required Changes

The Python version needs to align its category structure with the MATLAB organization. Here's the mapping:

## 1. **Auxiliary Functions** → Keep as "Auxiliary"

### MATLAB Structure:
- `Auxiliary/Plotting/` 
- `Auxiliary/SensorSupport/OptimalSensorPlacement/`
- `Auxiliary/SensorSupport/SensorDiagnostic/`

### Python Current Categories:
- ✅ `Auxiliary - Sensor Support` (matches MATLAB structure)
- ❌ `Modal - Visualization` → **CHANGE TO** `Auxiliary - Plotting`
- ❌ `Modal - Optimal Sensor Placement` → **CHANGE TO** `Auxiliary - Sensor Support - Optimal Sensor Placement`

### Recommended Python Categories:
```
Auxiliary - Plotting
Auxiliary - Sensor Support - Optimal Sensor Placement  
Auxiliary - Sensor Support - Sensor Diagnostics
```

## 2. **Data Acquisition** → New top-level category needed

### MATLAB Structure:
- `DataAcquisition/` (signal generation, data splitting)
- `DataAcquisition/Traditional/` 
- `DataAcquisition/NationalInstrumentsHighSpeed/`

### Python Current Categories:
- ❌ `Hardware - Signal Generation` → **CHANGE TO** `Data Acquisition - Signal Generation`
- ❌ Missing hardware DAQ functions

### Recommended Python Categories:
```
Data Acquisition - Signal Generation
Data Acquisition - Traditional DAQ
Data Acquisition - National Instruments
Data Acquisition - Utilities
```

## 3. **Feature Extraction** → Restructure current "Features" and "Core"

### MATLAB Structure:
- `FeatureExtraction/Preprocessing/`
- `FeatureExtraction/Statistics/`
- `FeatureExtraction/SpectralAnalysis/`
- `FeatureExtraction/TimeSeriesModels/`
- `FeatureExtraction/ModalAnalysis/`
- `FeatureExtraction/ConditionBasedMonitoring/`
- `FeatureExtraction/ActiveSensing/`

### Python Current Categories Need Major Reorganization:

#### 3.1 Preprocessing Functions
- ❌ `Core - Preprocessing` → **CHANGE TO** `Feature Extraction - Preprocessing`
- ❌ `Core - Signal Filtering` → **CHANGE TO** `Feature Extraction - Preprocessing`
- ❌ `Core - Signal Processing` → **CHANGE TO** `Feature Extraction - Preprocessing`

#### 3.2 Statistics Functions
- ❌ `Core - Statistics` → **CHANGE TO** `Feature Extraction - Statistics`  
- ❌ `Features - Statistics` → **CHANGE TO** `Feature Extraction - Statistics`
- ❌ `Statistics - Basic Indicators` → **CHANGE TO** `Feature Extraction - Statistics`

#### 3.3 Spectral Analysis Functions
- ❌ `Core - Spectral Analysis` → **CHANGE TO** `Feature Extraction - Spectral Analysis`
- ❌ `Spectral Analysis - Wavelets` → **CHANGE TO** `Feature Extraction - Spectral Analysis`
- ❌ `Spectral Analysis - Time-Frequency` → **CHANGE TO** `Feature Extraction - Spectral Analysis`
- ❌ `Core - Time-Frequency` → **CHANGE TO** `Feature Extraction - Spectral Analysis`

#### 3.4 Time Series Models
- ✅ `Features - Time Series Models` → **KEEP AS** `Feature Extraction - Time Series Models`

#### 3.5 Modal Analysis  
- ❌ `Features - Modal Analysis` → **CHANGE TO** `Feature Extraction - Modal Analysis`

#### 3.6 Condition Based Monitoring
- ✅ `Features - Condition Based Monitoring` → **KEEP AS** `Feature Extraction - Condition Based Monitoring`

#### 3.7 Active Sensing
- ✅ `Active Sensing - Signal Processing` → **CHANGE TO** `Feature Extraction - Active Sensing`
- ✅ `Active Sensing - Geometry` → **CHANGE TO** `Feature Extraction - Active Sensing`  
- ✅ `Active Sensing - Utilities` → **CHANGE TO** `Feature Extraction - Active Sensing`

### Recommended Python Categories:
```
Feature Extraction - Preprocessing
Feature Extraction - Statistics  
Feature Extraction - Spectral Analysis
Feature Extraction - Time Series Models
Feature Extraction - Modal Analysis
Feature Extraction - Condition Based Monitoring
Feature Extraction - Active Sensing
```

## 4. **Feature Classification** → Rename "Classification"

### MATLAB Structure:
- `FeatureClassification/` (core functions)
- `FeatureClassification/OutlierDetection/ParametricDetectors/`
- `FeatureClassification/OutlierDetection/NonParametricDetectors/`
- `FeatureClassification/OutlierDetection/SemiParametricDetectors/`

### Python Current Categories:
- ❌ `Classification - Parametric Detectors` → **CHANGE TO** `Feature Classification - Parametric Detectors`
- ❌ `Classification - Nonparametric Detectors` → **CHANGE TO** `Feature Classification - Non-Parametric Detectors`
- ❌ `Classification - Non-Parametric Detectors` → **CHANGE TO** `Feature Classification - Non-Parametric Detectors`
- ❌ `Classification - Semi-Parametric Detectors` → **CHANGE TO** `Feature Classification - Semi-Parametric Detectors`
- ❌ `Classification - Performance Evaluation` → **CHANGE TO** `Feature Classification - Performance Evaluation`
- ❌ `Classification - High Level Interface` → **CHANGE TO** `Feature Classification - High Level Interface`
- ❌ `Classification - Detector Assembly` → **CHANGE TO** `Feature Classification - Detector Assembly`

### Recommended Python Categories:
```
Feature Classification - Core Functions
Feature Classification - Parametric Detectors
Feature Classification - Non-Parametric Detectors  
Feature Classification - Semi-Parametric Detectors
Feature Classification - Performance Evaluation
Feature Classification - High Level Interface
Feature Classification - Detector Assembly
```

## 5. **Support Categories** → Keep but reorganize

### Plotting Functions
- ❌ `Plotting - Detection Results` → **CHANGE TO** `Auxiliary - Plotting`
- ❌ `Plotting - Spectral` → **CHANGE TO** `Auxiliary - Plotting`  
- ❌ `Plotting - Spectral Analysis` → **CHANGE TO** `Auxiliary - Plotting`
- ❌ `Plotting - Classification` → **CHANGE TO** `Auxiliary - Plotting`
- ❌ `Plotting - Feature Visualization` → **CHANGE TO** `Auxiliary - Plotting`

### Utilities
- ❌ `Utilities - Spatial Analysis` → **CHANGE TO** `Auxiliary - Utilities`
- ❌ `Utilities - Data Processing` → **CHANGE TO** `Auxiliary - Utilities`
- ❌ `Features - Data Management` → **CHANGE TO** `Auxiliary - Utilities`

### Recommended Python Categories:
```
Auxiliary - Plotting
Auxiliary - Utilities  
```

## Complete Target Category Structure

Based on MATLAB organization, the Python version should use these categories:

```
# Primary SHM Workflow Categories (matching MATLAB)
Data Acquisition - Signal Generation
Data Acquisition - Traditional DAQ  
Data Acquisition - National Instruments
Data Acquisition - Utilities

Feature Extraction - Preprocessing
Feature Extraction - Statistics
Feature Extraction - Spectral Analysis
Feature Extraction - Time Series Models
Feature Extraction - Modal Analysis
Feature Extraction - Condition Based Monitoring
Feature Extraction - Active Sensing

Feature Classification - Core Functions
Feature Classification - Parametric Detectors
Feature Classification - Non-Parametric Detectors
Feature Classification - Semi-Parametric Detectors
Feature Classification - Performance Evaluation
Feature Classification - High Level Interface
Feature Classification - Detector Assembly

# Support Categories
Auxiliary - Plotting
Auxiliary - Sensor Support - Optimal Sensor Placement
Auxiliary - Sensor Support - Sensor Diagnostics
Auxiliary - Utilities
```

## Implementation Plan

1. **Update all Python function docstrings** to use the new MATLAB-compatible categories
2. **Test the JupyterLab extension** to ensure the new categories display correctly  
3. **Update documentation** to reflect the new organization
4. **Verify user experience** - MATLAB users should find functions in expected locations

## Benefits of This Alignment

1. **Familiar Navigation**: Users accustomed to MATLAB SHMTools will immediately understand the organization
2. **Logical Workflow**: Categories follow the natural SHM analysis workflow (acquire → extract → classify)
3. **Educational Value**: Matches the pedagogical structure used in MATLAB documentation
4. **Consistency**: Same terminology between MATLAB reference and Python implementation
5. **Scalability**: Room for additional functions within the established hierarchy

## Migration Notes

- **No breaking changes**: Function names and APIs remain unchanged
- **Only metadata changes**: Only docstring `category:` fields need updating  
- **Backward compatibility**: Old category names can be maintained as aliases if needed
- **Progressive migration**: Can be done function by function without impacting existing code

This alignment ensures that the Python SHMTools provides a seamless transition experience for users migrating from the MATLAB version while maintaining the educational and logical structure that makes SHMTools effective for structural health monitoring education and research.