# JupyterLab Extension Testing Guide

**Updated**: July 21, 2025  
**Extension**: SHM Function Selector for JupyterLab  
**Phase**: 3 - Enhanced Context Menu System  

## Overview

The SHM Function Selector has been **migrated to JupyterLab** using modern TypeScript architecture. This guide provides step-by-step instructions for testing the Phase 3 enhanced context menu system in JupyterLab.

## Prerequisites

### System Requirements
- **Node.js**: >= 16.0.0 (we tested with v22.14.0)
- **Python**: >= 3.8 (we tested with Python 3.9)
- **JupyterLab**: >= 4.0.0 (we tested with JupyterLab 4.4.5)
- **npm**: >= 8.0.0 (we tested with npm 10.9.2)

### Verify Prerequisites
```bash
node --version    # Should be >= 16.0.0
python --version  # Should be >= 3.8
jupyter --version # Should show JupyterLab >= 4.0.0
```

## Installation Instructions

### 1. **Activate Virtual Environment**
```bash
cd /Users/eric/repo/shm/shmtools-python
source venv/bin/activate
```

### 2. **Install the Extension in Development Mode**

**Option A: Direct Installation (Recommended)**
```bash
# Navigate to the extension directory
cd shm_function_selector

# Install Python dependencies
pip install -e .

# Install Node.js dependencies
npm install

# Build the extension
npm run build

# Install the extension for JupyterLab
jupyter labextension develop . --overwrite

# Enable the server extension
jupyter server extension enable shm_function_selector
```

**Option B: Alternative Installation**
```bash
# From the shm_function_selector directory
pip install -e .
jupyter labextension install .
jupyter labextension enable shm-function-selector
```

### 3. **Verify Installation**
```bash
# Check if the extension is installed
jupyter labextension list | grep shm-function-selector

# Check server extension
jupyter server extension list | grep shm_function_selector
```

You should see:
- ✅ `shm-function-selector` enabled in labextension list
- ✅ `shm_function_selector` enabled in server extension list

## Testing Phase 3 Features

### 1. **Start JupyterLab**
```bash
# Start JupyterLab (make sure venv is activated)
source venv/bin/activate
jupyter lab
```

### 2. **Create/Open Test Notebook**
1. Create a new notebook or open: `Phase3_Context_Menu_Test.ipynb`
2. The notebook should load with the SHM extension active

### 3. **Test Phase 3 Enhanced Context Menu**

#### **A. Create Test Variables (Run these cells first)**
```python
# Cell 1: Create test variables
import numpy as np
import matplotlib.pyplot as plt

# Data variables (should be recommended for 'data' parameters)
sensor_data = np.random.randn(1000, 4)
acceleration_data = np.random.randn(2000, 3) 
features_matrix = np.random.randn(500, 20)

# Scalar parameters
sampling_freq = 1000.0
ar_order = 15
n_components = 5

# Metadata 
channel_names = ["X", "Y", "Z", "RX"]
sensor_labels = ["Acc_X", "Acc_Y", "Acc_Z"]

# Model objects
trained_model = {
    'type': 'AR',
    'order': 15,
    'coefficients': np.random.randn(15)
}

print("✅ Test variables created")
```

```python  
# Cell 2: Create tuple unpacking variables
def mock_ar_model(data, order=15):
    n_features = data.shape[0] - order
    features = np.random.randn(n_features, order * data.shape[1])
    model = {'order': order, 'coefficients': np.random.randn(order)}
    return features, model

# This creates multiple variables from tuple unpacking
ar_features, ar_model = mock_ar_model(sensor_data, order=15)
pca_scores, pca_loadings = np.random.randn(985, 3), np.random.randn(60, 3)

print("✅ Tuple variables created")
```

#### **B. Test Enhanced Context Menu System**

**Test 1: Basic Parameter Detection**
```python
# Cell 3: Type this line, then RIGHT-CLICK on 'None' after 'data='
result = shmtools.ar_model_shm(data=None, order=15)
```

**Expected Behavior:**
- 🎯 Right-click on `None` should show context menu
- ✅ **Recommended section** with arrays highlighted in green:
  - `sensor_data (array) • Cell 1`  
  - `acceleration_data (array) • Cell 1`
  - `features_matrix (array) • Cell 1`
- ⚠️ **Other variables section** with scalars dimmed:
  - `sampling_freq (float) • Cell 1`
  - `ar_order (int) • Cell 1`

**Test 2: Parameter-Specific Intelligence**  
```python
# Cell 4: Test different parameter types
result = shmtools.pca_shm(
    data=None,        # RIGHT-CLICK: should recommend arrays
    fs=None,          # RIGHT-CLICK: should recommend sampling_freq  
    order=None,       # RIGHT-CLICK: should recommend ar_order, n_components
    channels=None     # RIGHT-CLICK: should recommend channel_names, sensor_labels
)
```

**Test 3: Multi-line Function Calls**
```python
# Cell 5: Multi-line function - Phase 3 handles this
outlier_result = shmtools.mahalanobis_shm(
    features=None,    # RIGHT-CLICK: should recommend feature arrays
    model=None        # RIGHT-CLICK: should recommend trained_model, ar_model
)
```

**Test 4: TODO Comment Detection**
```python
# Cell 6: TODO comments - Phase 3 recognizes these
# features, model = shmtools.ar_model_shm(data, 15)  # TODO: Set proper data
# RIGHT-CLICK anywhere on the TODO line
```

#### **C. Test Parameter Linking**

1. **Right-click** on any `None` parameter
2. **Select** a recommended variable from the green section
3. **Verify** the parameter is replaced precisely:
   - `data=None` becomes `data=sensor_data`
   - TODO comments are automatically removed
   - Code formatting is preserved

### 4. **Test Advanced Features**

#### **Command Palette Integration**
1. Open Command Palette: `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Type "SHM" - you should see:
   - `Show SHM Functions`
   - `Refresh Variables`

#### **Console Logging**
Open browser Developer Tools (`F12`) and check console for:
```
🚀 SHM Function Selector JupyterLab extension activated!
📓 Notebook added, setting up SHM functionality
🎯 Parameter context detected: ...
🧠 Found X recommended, Y other variables
✅ Parameter linked successfully
```

## Expected Phase 3 Enhancements

### ✅ **What Should Work**

1. **🎯 Enhanced Parameter Detection**
   - Character-precise cursor positioning
   - Multi-line function call support
   - Named parameter parsing (`param=value`)
   - TODO comment recognition

2. **🧠 Smart Variable Compatibility**  
   - Arrays recommended for `data` parameters
   - Floats recommended for `fs` parameters
   - Integers recommended for `order` parameters
   - Lists recommended for `channels` parameters

3. **✨ Professional Context Menu**
   - Monospace typography for code consistency
   - Green highlighting for recommended variables
   - Color-coded compatibility indicators
   - Smooth hover effects and animations

4. **🔧 Precise Code Modification**
   - Boundary-accurate parameter replacement
   - Automatic TODO comment cleanup
   - Format-preserving code generation

### ⚠️ **Known Limitations**

- Extension is in development mode - requires rebuild after code changes
- Some TypeScript compilation issues need resolution for full build
- Parameter detection works primarily for SHM function patterns
- Variable type inference is heuristic-based

## Troubleshooting

### **Extension Not Loading**
```bash
# Check installation status
jupyter labextension list
jupyter server extension list

# Rebuild if needed
cd shm_function_selector
npm run build
jupyter lab build

# Restart JupyterLab
jupyter lab
```

### **Context Menu Not Appearing**
1. Ensure you're right-clicking precisely on parameter values like `None`
2. Check browser console (F12) for JavaScript errors
3. Verify the notebook has SHM functions available:
   ```python
   import shmtools
   print(dir(shmtools))
   ```

### **Server Extension Issues**
```bash
# Reinstall server extension
pip install -e shm_function_selector/
jupyter server extension enable shm_function_selector --sys-prefix

# Check server logs
jupyter lab --debug-level=10
```

### **TypeScript/Build Issues**  
```bash
# Clean and rebuild
cd shm_function_selector
npm run clean
npm install  
npm run build
```

## Architecture Overview

The migrated JupyterLab extension uses modern architecture:

### **Frontend (TypeScript)**
- `src/index.ts` - Main plugin registration
- `src/contextMenuManager.ts` - Phase 3 context menu system
- `src/parameterDetector.ts` - Enhanced parameter detection  
- `src/variableTracker.ts` - Smart variable tracking
- `src/shmFunctionManager.ts` - Function discovery and management

### **Backend (Python)**
- `shm_function_selector/handlers.py` - Server API endpoints
- Function discovery from `shmtools` package
- Variable parsing and type inference

### **Styling (CSS)**
- `style/index.css` - Phase 3 enhanced context menu styles
- Professional monospace typography
- Color-coded compatibility indicators

## Success Criteria

### ✅ **Phase 3 Complete When:**
- Context menu appears on right-click of parameters
- Variables are intelligently grouped by compatibility  
- Parameter linking works with precise code modification
- Professional styling with monospace fonts and colors
- Multi-line function calls are handled correctly

### 🚀 **Next Steps**
After successful testing, proceed to **Phase 4: Advanced Features** including:
- Smart defaults from function docstrings
- Parameter validation with real-time feedback
- Function templates with placeholders
- Enhanced type checking with kernel integration

---

**Note**: This is a development version focusing on Phase 3 functionality. The extension demonstrates the enhanced context menu system with intelligent parameter linking capabilities.