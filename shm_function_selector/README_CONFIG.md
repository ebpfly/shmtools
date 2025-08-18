# SHM Function Selector Configuration

This document explains how to configure the SHM Function Selector JupyterLab extension to specify where it looks for functions to include.

## Configuration File Location

The configuration is stored in `config.json` in the extension root directory:
```
shm_function_selector/config.json
```

## Configuration Structure

```json
{
  "function_discovery": {
    "enabled": true,
    "discovery_mode": "modules",
    "modules_to_scan": [
      "shmtools.core.spectral",
      "shmtools.features.time_series",
      "..."
    ],
    "include_patterns": ["*_shm", "learn_*", "apply_*", "compute_*"],
    "exclude_patterns": ["_*", "__*", "*_internal", "*_helper"],
    "custom_categories": {
      "shmtools.core": "Core - Signal Processing",
      "shmtools.features": "Feature Extraction"
    }
  },
  "variable_discovery": {
    "enabled": true,
    "include_notebook_variables": true,
    "include_kernel_variables": false,
    "type_inference": {
      "enabled": true,
      "use_annotations": true,
      "infer_from_values": true
    }
  },
  "gui_integration": {
    "show_parameter_hints": true,
    "show_function_descriptions": true,
    "group_by_category": true,
    "max_functions_per_category": 50
  }
}
```

## Configuration Options

### Function Discovery

- **`enabled`** (boolean): Enable/disable function discovery entirely
- **`discovery_mode`** (string): Discovery method ("modules" or "introspection")
- **`modules_to_scan`** (array): List of Python modules to scan for functions
- **`include_patterns`** (array): Glob patterns for function names to include (e.g., "*_shm")
- **`exclude_patterns`** (array): Glob patterns for function names to exclude (e.g., "_*")
- **`custom_categories`** (object): Custom category mappings for module prefixes

### Variable Discovery

- **`enabled`** (boolean): Enable/disable variable discovery
- **`include_notebook_variables`** (boolean): Include variables from notebook cells
- **`include_kernel_variables`** (boolean): Include variables from kernel namespace
- **`type_inference`** (object): Type inference settings

### GUI Integration

- **`show_parameter_hints`** (boolean): Show parameter hints in GUI
- **`show_function_descriptions`** (boolean): Show function descriptions
- **`group_by_category`** (boolean): Group functions by category
- **`max_functions_per_category`** (number): Limit functions per category (0 = unlimited)

## Examples

### Adding a New Module

To include functions from a new module `shmtools.custom.analysis`:

```json
{
  "function_discovery": {
    "modules_to_scan": [
      "shmtools.core.spectral",
      "shmtools.features.time_series",
      "shmtools.custom.analysis"
    ],
    "custom_categories": {
      "shmtools.custom": "Custom Analysis"
    }
  }
}
```

### Including Functions from Examples Folder

To include the data import and loading functions from the `examples` folder:

```json
{
  "function_discovery": {
    "modules_to_scan": [
      "shmtools.core.spectral",
      "shmtools.features.time_series",
      "examples.data"
    ],
    "include_patterns": [
      "*_shm",
      "import_*",
      "load_*"
    ],
    "custom_categories": {
      "examples.data": "📁 Data Import & Loading"
    }
  }
}
```

This will include functions like:
- `import_3story_structure_shm()`
- `import_cbm_data_shm()`
- `load_3story_data()`
- `load_cbm_data()`

You can also include specific submodules:
```json
{
  "function_discovery": {
    "modules_to_scan": [
      "examples.data.data_imports",
      "examples.data.data_loaders"
    ]
  }
}
```

### Filtering Functions

To only include functions ending with "_shm" or starting with "compute_":

```json
{
  "function_discovery": {
    "include_patterns": ["*_shm", "compute_*"],
    "exclude_patterns": ["_internal*", "*_debug"]
  }
}
```

### Limiting Functions per Category

To limit each category to 25 functions maximum:

```json
{
  "gui_integration": {
    "max_functions_per_category": 25
  }
}
```

## Applying Changes

After modifying the configuration:

1. Save the `config.json` file
2. Restart JupyterLab server
3. The changes will be applied automatically

The extension will log configuration loading status in the JupyterLab server logs.

## Default Configuration

If no `config.json` file exists, the extension uses built-in defaults that include all standard SHMTools modules with no filtering applied.

## Troubleshooting

### Configuration Not Loading

- Check that `config.json` is valid JSON syntax
- Verify file permissions are readable
- Check JupyterLab server logs for error messages

### Functions Not Appearing

- Verify module names in `modules_to_scan` are correct
- Check that `include_patterns` are not too restrictive
- Ensure `exclude_patterns` are not filtering out desired functions
- Verify the module is installed and importable

### Too Many Functions

- Use `include_patterns` to filter function names
- Set `max_functions_per_category` to limit per category
- Disable entire modules by removing from `modules_to_scan`