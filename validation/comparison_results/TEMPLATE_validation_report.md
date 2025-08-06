# Validation Report: [Example Name]

**Date**: [Date]  
**Python Notebook**: `examples/notebooks/[category]/[notebook_name].ipynb`  
**MATLAB Reference**: ExampleUsages.pdf, Pages [X-Y]  

## Structure Validation

### Section Headings
- [ ] Main title matches
- [ ] Section order identical  
- [ ] All major sections present

**MATLAB Sections**:
1. [Section 1 name]
2. [Section 2 name]
3. ...

**Python Sections**:
1. [Section 1 name]
2. [Section 2 name]
3. ...

**Discrepancies**: [None/List any differences]

### Content Flow
- [ ] Code blocks in same sequence
- [ ] Output placement matches
- [ ] Explanatory text equivalent

## Results Validation

### Numerical Results

| Output | MATLAB Value | Python Value | Difference | Within 10%? |
|--------|--------------|--------------|------------|-------------|
| [Metric 1] | X.XXX | X.XXX | X.X% | ✓/✗ |
| [Metric 2] | X.XXX | X.XXX | X.X% | ✓/✗ |

### Visualizations

#### Plot 1: [Description]
- **Plot Type**: [Same/Different]
- **Axis Ranges**: [Match/Different - specify]
- **Legend**: [Match/Different - specify]
- **Visual Match**: [Yes/No - describe differences]

#### Plot 2: [Description]
- **Plot Type**: [Same/Different]
- **Axis Ranges**: [Match/Different - specify]
- **Legend**: [Match/Different - specify]
- **Visual Match**: [Yes/No - describe differences]

### Console Output
- [ ] Format similar
- [ ] Key metrics reported
- [ ] Messages equivalent

**Differences**: [None/List any]

## Issues Found

### Critical Issues (Must Fix)
1. **Issue**: [Description]
   - **Location**: [Line/Section]
   - **Impact**: [How it affects results]
   - **Fix**: [Proposed solution]

### Minor Issues (Should Fix)
1. **Issue**: [Description]
   - **Location**: [Line/Section]
   - **Impact**: [Cosmetic/Minor functional]
   - **Fix**: [Proposed solution]

### Enhancement Opportunities
1. **Enhancement**: [Description]
   - **Rationale**: [Why it would improve the example]

## Required Code Changes

### File: `shmtools/[module]/[file].py`
```python
# Current code
[existing code]

# Should be
[corrected code]
```

### File: `examples/notebooks/[category]/[notebook].ipynb`
```python
# Cell [X] - [Description]
[changes needed]
```

## Validation Summary

**Overall Status**: ☐ Pass / ☐ Fail / ☐ Pass with minor issues

**Ready for Publication**: ☐ Yes / ☐ No - requires fixes

**Notes**: [Any additional observations or recommendations]