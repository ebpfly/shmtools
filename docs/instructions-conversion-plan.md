### 📋 mFUSE TO JUPYTER CONVERSION

**Goal**: Convert all mFUSE workflow instructions to Jupyter extension equivalents.

**Process**: 
1. Extract text from mFUSE .docx instruction files using python-docx
2. Map MATLAB function calls to Python equivalents using function metadata
3. Create step-by-step Jupyter extension instructions following the dropdown → right-click → parameter linking workflow
4. Follow the structure of the original mFUSE instruction files exactly. Simple, neat, and clean. Do not add additional instructions.
5. Verify all functions exist and have proper category metadata for the function selector

**Status**: 
- ✅ **Conversion Guide**: Complete methodology documented in `docs/mfuse-to-jupyter-conversion-guide.md`
- ✅ **All Workflows Converted**: All 4 mFUSE workflows converted to Jupyter extension instructions:
  - Outlier Detection → `jupyter_extension_outlier_detection_instructions.md`
  - Condition Based Monitoring → `condition_based_monitoring_jupyter_instructions.md`
  - Optimal Sensor Placement → `optimal_sensor_placement_jupyter_instructions.md`
  - Guided Wave Active Sensing → `guided_wave_active_sensing_jupyter_instructions.md`


**Files Locations**:
- Original: `matlab/LADPackage/Demos/Instructions/*.docx`
- Converted: `/Users/eric/repo/shm/LADPackage/Demos/Instructions/*.md`