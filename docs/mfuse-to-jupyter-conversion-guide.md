# mFUSE to Jupyter Extension Conversion Guide

This document provides instructions for converting mFUSE workflow instructions to equivalent Jupyter extension workflows.

## Overview

mFUSE (the original Java GUI) provided guided workflows through menu navigation. The Jupyter extension replaces this with:
1. **Function dropdown selector** in the notebook toolbar
2. **Right-click parameter linking** for connecting variables
3. **Automatic code generation** with proper imports

## Conversion Process

### Step 1: Extract mFUSE Workflow

First, identify the original mFUSE workflow structure:

1. **Read the original .docx instructions** (use python-docx to extract text)
2. **Identify the sequence of steps** (usually 8-10 steps)
3. **Note function names** and parameter settings
4. **Document input/output relationships** between steps

### Step 2: Map Functions to Python Equivalents

For each mFUSE function, find the corresponding Python function:

1. **Check function metadata** in the Python code for category information:
   ```python
   # Look for meta docstring sections like:
   # .. meta::
   #     :category: Feature Classification - Parametric Detectors
   #     :display_name: Learn Mahalanobis
   ```

2. **Identify category hierarchy** from the meta tags (this determines dropdown location)

3. **Map parameter names** (often identical but check for differences like MATLAB vs Python indexing)

### Step 3: Create Jupyter Extension Instructions

Follow this template structure:

```markdown
# [Workflow Name] with Jupyter Extension

**Dataset**: [Dataset description]
**Goal**: [Analysis objective]

## Sequence

### Step N: [Function Name]
1. Use dropdown to find **[Display Name]** (category: [Category Path])
2. Right-click parameters to link:
   - `param=None` → select `variable_name`
   - `param=default` → change to `new_value`
3. Leave other parameters as default

**Inputs:**
- Parameter = variable_name (Step X)
- Parameter = value

**Output variables:**
- Output Name = variable_name (SHAPE, TYPE)
```

### Step 4: Verify Function Availability

Check that all required functions exist in the Python codebase:

1. **Search for function definitions**:
   ```bash
   grep -r "def function_name_shm" shmtools/
   ```

2. **Verify category metadata** matches the extension's function discovery

3. **Check parameter signatures** match between MATLAB and Python versions

4. **Note any missing functions** that need to be implemented

## Conversion Rules

### Function Names
- MATLAB: `functionName_shm` 
- Python: `function_name_shm` (same name, proper imports handled by extension)

### Parameter Mapping
- **Data arrays**: Usually `data=None` → right-click to link variables
- **Indices**: Convert MATLAB 1-based to Python 0-based (e.g., `1:2:91` → `list(range(0, 91, 2))`)
- **Boolean flags**: Convert MATLAB `true/false` to Python `True/False`
- **Default values**: Check if Python defaults match MATLAB behavior

### Categories (from function metadata)
Common category patterns:
- **Data Loading**: `Data Loading - Example Datasets`
- **Features**: `Feature Extraction - Time Series Models`
- **Classification**: `Feature Classification - Parametric Detectors`
- **Plotting**: `Plotting - Feature Visualization`, `Plotting - Classification Results`
- **Performance**: `Feature Classification - Performance Metrics`

### Variable Naming
Keep variable names consistent with MATLAB versions when possible:
- `dataset` → `dataset`
- `damageStates` → `damage_states` (Python snake_case)
- `arParametersFV` → `ar_features` (more descriptive Python name)

## Template Files

### Input: mFUSE .docx Instructions
```
Demo: Feature Extraction and Outlier Detection
Dataset: Base-excited 3-Story structure dataset
Goal: Estimate AR model parameters, then detect outliers

Sequence
Step 1: Import 3 Story Structure Dataset (LADPackage/Outlier Models)
Input: Floor Numbers = 5 (Default)
Output: Dataset = (TIME, CHANNELS, INSTANCES)
...
```

### Output: Jupyter Extension Instructions
```markdown
# Outlier Detection with Jupyter Extension

**Dataset**: Base-excited 3-story structure dataset
**Goal**: Estimate AR model parameters, then detect outliers

## Sequence

### Step 1: Import 3 Story Structure Dataset
1. Use the **SHM Function** dropdown in the notebook toolbar
2. Find and select **load_3story_data** from the dropdown
3. Function is inserted automatically

**Output variables:**
- Dataset = data_dict['dataset'] (TIME, CHANNELS, INSTANCES)
...
```

## Automation Script

Create a Python script to help with conversions:

```python
def convert_mfuse_instructions(docx_path, output_path):
    """Convert mFUSE .docx instructions to Jupyter extension format."""
    
    # 1. Extract text from .docx
    from docx import Document
    doc = Document(docx_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    
    # 2. Parse workflow steps
    steps = parse_mfuse_steps(text)
    
    # 3. Map to Python functions
    mapped_functions = map_to_python_functions(steps)
    
    # 4. Generate Jupyter instructions
    jupyter_instructions = generate_jupyter_instructions(mapped_functions)
    
    # 5. Write output
    with open(output_path, 'w') as f:
        f.write(jupyter_instructions)
```

## Quality Checklist

For each converted workflow:
- [ ] All mFUSE steps have corresponding Jupyter steps
- [ ] Function categories are correct (match Python metadata)
- [ ] Parameter names match Python function signatures
- [ ] Input/output variable flow is preserved
- [ ] MATLAB-to-Python indexing is converted correctly
- [ ] Step numbering matches original sequence
- [ ] Instructions are concise and actionable

## Examples

Converted workflows available:
- **Outlier Detection**: `examples/notebooks/outlier_detection/jupyter_extension_outlier_detection_instructions.md`

Original mFUSE instructions:
- **Outlier Detection**: `matlab/LADPackage/Demos/Instructions/Outlier Detection.docx`

## Integration with Development Workflow

Add this conversion process to the development workflow:

1. **When implementing new functions**, include proper metadata for categorization
2. **When finding mFUSE workflows**, convert them to Jupyter extension instructions
3. **Store converted instructions** in the appropriate `examples/notebooks/` subdirectory
4. **Update this guide** with new patterns and edge cases discovered during conversion

This ensures all original mFUSE functionality remains accessible through the modern Jupyter interface while maintaining the educational value of the guided workflows.