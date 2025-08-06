#!/usr/bin/env python3
"""Batch validation helper for comparing Python notebooks against MATLAB examples."""

import json
import numpy as np
from pathlib import Path
import re


def extract_notebook_values(notebook_path):
    """Extract key numerical values and plot info from notebook outputs."""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    values = {
        'outputs': [],
        'plots': [],
        'shape_outputs': []
    }
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'outputs' in cell:
            for output in cell['outputs']:
                if 'text' in output:
                    text = ''.join(output['text'])
                    
                    # Extract shapes
                    shape_matches = re.findall(r'shape:\s*\([\d,\s]+\)', text)
                    values['shape_outputs'].extend(shape_matches)
                    
                    # Extract UCL/threshold values
                    ucl_match = re.search(r'UCL.*?:\s*([-\d.]+)', text)
                    if ucl_match:
                        values['outputs'].append(f"UCL: {ucl_match.group(1)}")
                    
                    # Extract accuracy
                    acc_match = re.search(r'accuracy:\s*([\d.]+%)', text, re.IGNORECASE)
                    if acc_match:
                        values['outputs'].append(f"Accuracy: {acc_match.group(1)}")
                    
                    # Extract AR order
                    ar_match = re.search(r'AR order.*?(\d+)', text)
                    if ar_match:
                        values['outputs'].append(f"AR Order: {ar_match.group(1)}")
        
        # Extract plot info from markdown cells
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'plt.title' in source:
                title_match = re.search(r"plt\.title\(['\"](.*?)['\"]\)", source)
                if title_match:
                    values['plots'].append(title_match.group(1))
    
    return values


def create_validation_template(example_name, python_path, matlab_pages="TBD"):
    """Create a validation report template with extracted values."""
    
    # Extract values from notebook
    values = extract_notebook_values(python_path)
    
    # Get relative path safely
    try:
        rel_path = python_path.relative_to(Path.cwd())
    except ValueError:
        # If not relative to cwd, use the path as-is
        rel_path = python_path
    
    template = f"""# Validation Report: {example_name}

**Date**: 2025-08-06  
**Python Notebook**: `{rel_path}`  
**MATLAB Reference**: ExampleUsages.pdf, Pages {matlab_pages}

## Structure Validation

### Section Headings
- [ ] Main title matches
- [ ] Section order identical  
- [ ] All major sections present

**MATLAB Sections**:
1. [To be verified from PDF]
2. ...

**Python Sections**:
[Extracted from notebook structure]

**Discrepancies**: [To be determined]

### Content Flow
- [ ] Code blocks in same sequence
- [ ] Output placement matches
- [ ] Explanatory text equivalent

## Results Validation

### Numerical Results

| Output | MATLAB Value | Python Value | Difference | Within 10%? |
|--------|--------------|--------------|------------|-------------|"""

    # Add extracted values
    for output in values['outputs']:
        key, value = output.split(': ', 1)
        template += f"\n| {key} | TBD | {value} | TBD | TBD |"
    
    template += f"""

### Visualizations

Number of plots found: {len(values['plots'])}"""

    for i, plot_title in enumerate(values['plots'], 1):
        template += f"""

#### Plot {i}: {plot_title}
- **Plot Type**: [To verify]
- **Axis Ranges**: [To verify]
- **Legend**: [To verify]
- **Visual Match**: [To verify]"""

    template += """

### Console Output
- [ ] Format similar
- [ ] Key metrics reported
- [ ] Messages equivalent

**Differences**: [To be determined]

## Issues Found

### Critical Issues (Must Fix)
[To be filled after comparison]

### Minor Issues (Should Fix)
[To be filled after comparison]

### Enhancement Opportunities
[To be filled after comparison]

## Required Code Changes

[To be filled after identifying issues]

## Validation Summary

**Overall Status**: ☐ Pass / ☐ Fail / ☐ Pass with minor issues

**Ready for Publication**: ☐ Yes / ☐ No - requires fixes

**Notes**: [To be filled after validation]
"""
    
    return template


def main():
    """Generate validation templates for remaining examples."""
    
    # Examples to validate with their MATLAB page numbers (if known)
    examples = [
        ("SVD Outlier Detection", "svd_outlier_detection.ipynb", "38-41"),
        ("Time Synchronous Averaging", "time_synchronous_averaging_demo.ipynb", "TBD"),
        ("Factor Analysis Outlier Detection", "factor_analysis_outlier_detection.ipynb", "TBD"),
        ("NLPCA Outlier Detection", "nlpca_outlier_detection.ipynb", "TBD"),
        ("Modal Analysis Features", "modal_analysis_features_simplified.ipynb", "TBD"),
        ("Active Sensing Feature Extraction", "active_sensing_feature_extraction.ipynb", "TBD"),
    ]
    
    validation_dir = Path("validation/comparison_results")
    validation_dir.mkdir(exist_ok=True)
    
    for example_name, notebook_file, matlab_pages in examples:
        # Find the notebook
        notebook_path = None
        for category in ['basic', 'intermediate', 'advanced']:
            path = Path(f"examples/notebooks/{category}/{notebook_file}")
            if path.exists():
                notebook_path = path
                break
        
        if notebook_path:
            output_file = validation_dir / f"{notebook_path.stem}_validation_report.md"
            if not output_file.exists():
                print(f"Creating template for: {example_name}")
                template = create_validation_template(example_name, notebook_path, matlab_pages)
                output_file.write_text(template)
                print(f"  Saved to: {output_file}")
            else:
                print(f"Skipping {example_name} - report already exists")
        else:
            print(f"Warning: Could not find notebook for {example_name}")


if __name__ == '__main__':
    main()