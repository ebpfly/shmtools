# Enhanced plot_psd_shm: 2D Data Support

The `plot_psd_shm` function now automatically handles 2D PSD data with imshow visualization.

## Key Features

### Automatic 2D Detection
When you pass 2D data (frequency × instances), the function automatically:
- Enables colormap visualization using `imshow`
- Creates frequency vs instance plots
- Shows power spectral density variations across time or conditions

### Usage Examples

```python
import numpy as np
from shmtools.plotting import plot_psd_shm

# 2D PSD matrix (129 frequencies, 50 instances)
psd_2d = np.random.randn(129, 50) + 10  # Add offset to avoid log(0)
f = np.linspace(0, 500, 129)  # Frequency vector

# Automatically creates imshow plot
ax = plot_psd_shm(psd_2d, f=f)
```

### Input Dimensions

| Input Shape | Behavior | Visualization |
|-------------|----------|---------------|
| `(n_freqs,)` | 1D PSD | Single line plot |
| `(n_freqs, n_instances)` | 2D PSD | **Auto imshow** |
| `(n_freqs, n_channels, n_instances)` | 3D PSD | Channel selection + colormap |

### Parameters

- **`psd_matrix`**: Now accepts 2D or 3D arrays
- **`channel`**: Only used for 3D data (1-based indexing)
- **`use_colormap`**: Automatically enabled for 2D data with multiple instances
- **`f`**: Frequency vector for proper axis labeling

### Backward Compatibility

All existing functionality is preserved:
- 3D data still works with channel selection
- MATLAB-compatible interface maintained
- All optional parameters work as before

## Implementation Details

The function automatically detects input dimensions and chooses the appropriate visualization:

```python
if psd_matrix.ndim == 2:
    # 2D: (n_freqs, n_instances) - use directly, force colormap for multiple instances
    psd_data = psd_matrix
    if psd_data.shape[1] > 1:
        use_colormap = True  # Automatic imshow for multiple instances
```

This enhancement makes it much easier to visualize time-frequency data, condition-based monitoring results, and other multi-instance spectral analyses.