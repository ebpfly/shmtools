# Simplified Notebook Import Patterns

## Quick Start (Recommended)

**IMPORTANT: First install shmtools in your environment:**
```bash
pip install -e .
```

Then replace the complex import block in existing notebooks with this simple pattern:

```python
import numpy as np
import matplotlib.pyplot as plt

# Import shmtools (installed package)
from shmtools.utils.data_loading import load_3story_data
from shmtools.features.time_series import ar_model_shm
from shmtools.classification.outlier_detection import learn_pca_shm, score_pca_shm

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
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

### New Pattern (10 lines)
```python
import numpy as np
import matplotlib.pyplot as plt

# Import shmtools (installed package)
from shmtools.utils.data_loading import load_3story_data
from shmtools.features.time_series import ar_model_shm
from shmtools.classification.outlier_detection import learn_pca_shm, score_pca_shm

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
```

## Data Loading

Load the standard 3-story structure data:

```python
# Load raw data
data_dict = load_3story_data()
dataset = data_dict['dataset']
fs = data_dict['fs']
channels = data_dict['channels']
damage_states = data_dict['damage_states']

# Extract channels 2-5 for most examples
data = dataset[:, 1:5, :]  # Shape: (8192, 4, 170)
t, m, n = data.shape
```

## Common Import Patterns by Example Type

### Outlier Detection Examples
```python
# PCA example
from shmtools.features.time_series import ar_model_shm
from shmtools.classification.outlier_detection import learn_pca_shm, score_pca_shm

# Mahalanobis example  
from shmtools.features.time_series import ar_model_shm
from shmtools.classification.outlier_detection import learn_mahalanobis_shm, score_mahalanobis_shm

# SVD example
from shmtools.features.time_series import ar_model_shm
from shmtools.classification.outlier_detection import learn_svd_shm, score_svd_shm, roc_shm
from shmtools.core.preprocessing import scale_min_max_shm
```

### Active Sensing Examples
```python
from shmtools.utils.data_loading import load_sensor_diagnostic_data, load_active_sensing_data
from shmtools.sensor_diagnostics import sd_feature_shm, sd_autoclassify_shm, sd_plot_shm
from shmtools.active_sensing import coherent_matched_filter_shm, estimate_group_velocity_shm
```

### Time Series Analysis Examples
```python
from shmtools.features.time_series import ar_model_order_shm, ar_model_shm
from shmtools.core.spectral import time_sync_avg_shm, stft_shm
```

## Benefits of This Approach

- **No path searching**: Works from any directory
- **Clean imports**: Only what you need
- **Version controlled**: Uses the installed package version
- **IDE support**: Full autocomplete and type hints
- **Portable**: Works in any Python environment where shmtools is installed

## Batch Update Script

To update all notebooks at once, run from the `shmtools-python/` directory:

```bash
python update_notebook_imports.py
```

This script will automatically find and update the complex import blocks in all example notebooks.