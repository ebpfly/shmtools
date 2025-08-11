# Outlier Detection with Jupyter Extension

This guide shows how to recreate the mFUSE outlier detection demo using the SHMTools Jupyter extension.

**Dataset**: Base-excited 3-story structure dataset  
**Goal**: Estimate AR model parameters, then detect outliers using the Mahalanobis distance and plot results.

## Sequence

### Step 1: Import 3 Story Structure Dataset
1. Use the **SHM Function** dropdown in the notebook toolbar
2. Find and select **load_3story_data** from the dropdown
3. Function is inserted automatically

**Output variables:**
- Dataset = data_dict['dataset'] (TIME, CHANNELS, INSTANCES)  
- Damage States = data_dict['damage_states'] (INSTANCES, 1)
- State Names = data_dict['state_names'] (INSTANCES, 1)

### Step 2: AR Model
1. Use dropdown to find **AR Model** (category: Feature Extraction - Time Series Models)
2. Right-click on `data=None` parameter → select `dataset[:, 1:, :]` (exclude force channel)
3. Right-click on `order=15` parameter → keep default or change value

**Inputs:**
- Time Series Data = dataset[:, 1:, :] (channels 2-5 only)
- AR Model Order = 15

**Output variables:**
- AR Parameters Feature Vectors = ar_features (INSTANCES, FEATURES)
- RMS Residuals Feature Vectors = rms_residuals (INSTANCES, FEATURES)
- AR Parameters = ar_parameters (AR_ORDER, CHANNELS, INSTANCES)
- AR Residuals = ar_residuals (TIME, CHANNELS, INSTANCES)  
- AR Prediction = ar_prediction (TIME, CHANNELS, INSTANCES)

### Step 3: Plot Features
1. Find **Plot Features** in dropdown (category: Plotting - Feature Visualization)
2. Right-click on `features=None` → select `ar_features`
3. Right-click on `feature_indices=None` → set to `[0, 1, 2, 3]`
4. Leave other parameters as default

**Inputs:**
- Feature Vectors = ar_features (Step 2)
- Features to Plot = [0, 1, 2, 3]

### Step 4: Learn Score Mahalanobis  
1. Find **Learn Mahalanobis** in dropdown (category: Feature Classification - Parametric Detectors)
2. Right-click on `X=None` → select `ar_features`
3. Create training indices variable: `train_indices = list(range(0, 91, 2))`
4. Find **Score Mahalanobis** and link the model output

**Inputs:**
- Features = ar_features (Step 2)
- Training Indices = list(range(0, 91, 2)) (every other sample 0-90)

**Output variables:**
- Scores = scores (INSTANCES, 1)

### Step 5: Plot Scores
1. Find **Plot Scores** in dropdown (category: Plotting - Classification Results)  
2. Right-click parameters to link:
   - `scores=None` → select `scores`
   - `states=None` → select `damage_states`
   - `flip_signs=False` → change to `True`
   - Leave other parameters as default

**Inputs:**
- Scores = scores (Step 4)
- States = damage_states (Step 1)
- Flip Signs = True

### Step 6: Plot Score Distributions
1. Find **Plot Score Distributions** in dropdown
2. Right-click parameters to link:
   - `scores=None` → select `scores`  
   - `damage_states=None` → select `damage_states`
   - `flip_signs=False` → change to `True`
   - `use_log_scores=False` → change to `True`
   - `smoothing=0.1` → keep default

**Inputs:**
- Scores = scores (Step 4)
- Damage States = damage_states (Step 1)
- Flip Signs = True
- Use Log Scores = True
- Kernel Smoothing Parameter = 0.1

### Step 7: Receiver Operating Characteristic
1. Find **ROC** in dropdown (category: Feature Classification - Performance Metrics)
2. Right-click parameters:
   - `scores=None` → select `scores`
   - `damage_states=None` → select `damage_states`
   - Leave other parameters as default

**Inputs:**
- Scores = scores (Step 4)
- Damage States = damage_states (Step 1)

**Output variables:**
- True Positive Rate = tpr
- False Positive Rate = fpr

### Step 8: Plot ROC Curve  
1. Find **Plot ROC** in dropdown
2. Right-click parameters:
   - `tpr=None` → select `tpr`
   - `fpr=None` → select `fpr`  
   - Leave other parameters as default

**Inputs:**
- True Positive Rate = tpr (Step 7)
- False Positive Rate = fpr (Step 7)

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