# Simplified Notebook Import Patterns

## Quick Start (Recommended)

Replace the complex import block in existing notebooks with this simple pattern:

```python
# One-line setup for notebooks
from shmtools.utils.data_loading import setup_notebook_environment, load_example_data

# Get common imports and functions
nb = setup_notebook_environment()
np, plt = nb['np'], nb['plt']

# Load preprocessed data for your example type
data = load_example_data('pca')  # or 'mahalanobis', 'svd', etc.
signals = data['signals']  # Shape: (8192, 4, 170) - channels 2-5 ready
fs = data['fs']
t, m, n = data['t'], data['m'], data['n']
```

## Old vs New Pattern

### Old Pattern (17 lines)
```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add shmtools to path - handle different execution contexts
current_dir = Path.cwd()
notebook_dir = Path(__file__).parent if '__file__' in globals() else current_dir

# Try different relative paths to find shmtools
possible_paths = [
    notebook_dir.parent.parent.parent,  # From examples/notebooks/basic/
    current_dir.parent.parent,          # From examples/notebooks/
    current_dir,                        # From project root
    Path('/Users/eric/repo/shm/shmtools-python')  # Absolute fallback
]

shmtools_found = False
for path in possible_paths:
    if (path / 'shmtools').exists():
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        shmtools_found = True
        print(f"Found shmtools at: {path}")
        break

if not shmtools_found:
    print("Warning: Could not find shmtools module")

from shmtools.utils.data_loading import load_3story_data
from shmtools.features.time_series import ar_model
from shmtools.classification.outlier_detection import learn_pca, score_pca

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load and preprocess data
data_dict = load_3story_data()
dataset = data_dict['dataset']
fs = data_dict['fs']
data = dataset[:, 1:5, :]  # Extract channels 2-5
t, m, n = data.shape
```

### New Pattern (6 lines)
```python
from shmtools.utils.data_loading import setup_notebook_environment, load_example_data
nb = setup_notebook_environment()
np, plt = nb['np'], nb['plt']

data = load_example_data('pca')
signals = data['signals']
fs = data['fs']
t, m, n = data['t'], data['m'], data['n']
```

## Example Type Mapping

Use these example types with `load_example_data()`:

- **Phase 1**: `'pca'` - PCA Outlier Detection
- **Phase 2**: `'mahalanobis'` - Mahalanobis Distance Outlier Detection  
- **Phase 3**: `'svd'` - SVD Outlier Detection
- **Phase 4**: `'factor_analysis'` - Factor Analysis Outlier Detection
- **Phase 5**: `'nlpca'` - Nonlinear PCA Outlier Detection
- **Phase 6**: `'ar_model_order'` - AR Model Order Selection

All these return the same preprocessed 3-story structure data with:
- `signals`: Shape (8192, 4, 170) - channels 2-5 extracted
- `fs`: 2000.0 Hz sampling frequency
- `channels`: ['Ch2', 'Ch3', 'Ch4', 'Ch5']
- `damage_states`: Damage state mapping
- `t`, `m`, `n`: Time points, channels, conditions

## Additional Functions Available

From `setup_notebook_environment()`:
- `nb['load_3story_data']()` - Direct access to 3-story data
- `nb['check_data_availability']()` - Check what datasets are available
- `nb['get_data_dir']()` - Get data directory path

## Function Imports Still Needed

You still need to import the specific SHMTools functions for your example:

```python
# For PCA example
from shmtools.features.time_series import ar_model
from shmtools.classification.outlier_detection import learn_pca, score_pca

# For Mahalanobis example  
from shmtools.features.time_series import ar_model
from shmtools.classification.outlier_detection import learn_mahalanobis, score_mahalanobis
```

This makes notebooks much more concise while maintaining compatibility with the conversion-plan.md approach.