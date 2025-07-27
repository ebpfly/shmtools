# ✅ Dataloader Functions Fixed - Verification Test

## What Was Fixed

1. **Module Exports**: Added dataloader functions to main `shmtools` module
2. **Extension Display**: Fixed `displayName` format for dropdown
3. **Function Access**: All dataloader functions now work via `shmtools.function_name()`

## Quick Verification

Run this in a Jupyter notebook cell:

```python
import shmtools

# Test that functions are available
print("Available dataloader functions:")
functions = [
    'load_3story_data',
    'load_example_data', 
    'setup_notebook_environment',
    'check_data_availability'
]

for func_name in functions:
    available = hasattr(shmtools, func_name)
    print(f"  • shmtools.{func_name}: {'✅' if available else '❌'}")

# Test actual usage
print("\nTesting function calls:")

# Load 3-story data (should work now!)
data = shmtools.load_3story_data()
print(f"✅ shmtools.load_3story_data() - dataset shape: {data['dataset'].shape}")

# Load example data with preprocessing
example_data = shmtools.load_example_data('pca')
print(f"✅ shmtools.load_example_data('pca') - signals shape: {example_data['signals'].shape}")
```

## Expected Output

```
Available dataloader functions:
  • shmtools.load_3story_data: ✅
  • shmtools.load_example_data: ✅
  • shmtools.setup_notebook_environment: ✅
  • shmtools.check_data_availability: ✅

Testing function calls:
✅ shmtools.load_3story_data() - dataset shape: (8192, 5, 170)
✅ shmtools.load_example_data('pca') - signals shape: (8192, 4, 170)
```

## JupyterLab Extension

The dropdown should now show:

**Data - Loading & Setup** (category with proper names)
- ✅ Check Data Availability
- ✅ Get Available Datasets  
- ✅ Get Data Dir
- ✅ Load 3Story Data ← Click this one!
- ✅ Load Active Sensing Data
- ✅ Load Cbm Data
- ✅ Load Example Data ← Very useful!
- ✅ Load Modal Osp Data
- ✅ Load Sensor Diagnostic Data
- ✅ Setup Notebook Environment ← Very useful!

Selecting any function will insert working code like:
```python
# Load 3-story structure dataset.
result = shmtools.load_3story_data()
```

## For Conversion-Plan.md Examples

Now you can easily reproduce notebooks using either:

1. **Extension dropdown**: Select "Load Example Data" → inserts `shmtools.load_example_data()`
2. **Direct import**: Use the simplified pattern from `SIMPLE_IMPORTS.md`

Both approaches now work correctly!