# Validation Guide

## How to Validate Python Examples Against MATLAB

### Prerequisites
- PDF reader that supports side-by-side viewing (Preview on Mac, Adobe Reader, etc.)
- Text editor for updating validation reports
- Access to both PDF files:
  - `/validation/matlab_reference/ExampleUsages.pdf` - MATLAB reference
  - `/validation/python_pdfs/[example_name].pdf` - Python conversions
  - `/validation/combined_python_examples.pdf` - All Python examples

### Step-by-Step Validation Process

#### 1. Open PDFs Side-by-Side
- Open ExampleUsages.pdf in one window
- Open the corresponding Python PDF in another window
- Use the table of contents or search to find matching examples

#### 2. Structural Comparison

**Check Section Headings:**
- [ ] Main title matches (may have slight wording differences)
- [ ] Major sections are present in both
- [ ] Section order is logical (doesn't need to be identical)

**Check Content Flow:**
- [ ] Introduction/background present
- [ ] Data loading section exists
- [ ] Processing steps in similar order
- [ ] Results visualization included
- [ ] Summary/conclusions provided

#### 3. Code Comparison

**Note:** Python and MATLAB syntax will differ, focus on:
- [ ] Same algorithms implemented
- [ ] Same data processing steps
- [ ] Same feature extraction methods
- [ ] Same statistical calculations

#### 4. Results Comparison

**Numerical Results:**
- Compare key metrics (within 10% tolerance):
  - Model orders, thresholds, coefficients
  - Statistical measures (mean, std, etc.)
  - Performance metrics (accuracy, ROC curves)
- Document exact values in validation report

**Visualizations:**
- [ ] Same types of plots (line, scatter, bar, etc.)
- [ ] Same data being visualized
- [ ] Axis labels convey same information
- [ ] Legend entries comparable
- [ ] Visual patterns/trends match

#### 5. Document Findings

Use the template in `comparison_results/TEMPLATE_validation_report.md`:

1. Copy template to new file: `[example_name]_validation_report.md`
2. Fill in all sections systematically
3. Mark checkboxes as you validate
4. Document specific issues with line/section references
5. Propose fixes for any discrepancies

### Common Acceptable Differences

These differences are OK and don't need fixes:

1. **Syntax Differences**
   - Python uses 0-based indexing vs MATLAB's 1-based
   - Different function names for same operations
   - Object-oriented vs procedural style

2. **Visualization Style**
   - Different default colors
   - Minor formatting differences
   - Interactive features in Python (zoom, pan)

3. **Text/Documentation**
   - More educational content in Python notebooks
   - Additional explanations of algorithms
   - Python-specific implementation notes

### Red Flags - Must Fix

These require immediate attention:

1. **Missing Functionality**
   - MATLAB features not implemented
   - Processing steps skipped
   - Results not computed

2. **Wrong Results**
   - Numerical values differ by >10%
   - Different conclusions reached
   - Incorrect algorithm implementation

3. **Missing Visualizations**
   - Plots shown in MATLAB but not Python
   - Key results not visualized
   - Different data being plotted

### Validation Workflow Example

```bash
# 1. Start with first example
open validation/matlab_reference/ExampleUsages.pdf
open validation/python_pdfs/ar_model_order_selection.pdf

# 2. Create validation report
cp validation/comparison_results/TEMPLATE_validation_report.md \
   validation/comparison_results/ar_model_validation_report.md

# 3. Edit report while comparing PDFs
# Fill in all sections, check boxes, document issues

# 4. Update main checklist
# Edit validation/validation_checklist.md to mark progress

# 5. If issues found, create GitHub issues or fix directly
```

### Tips for Efficient Validation

1. **Use PDF Search**
   - Search for example names, key terms
   - Find corresponding sections quickly

2. **Focus on Key Results**
   - Don't get bogged down in minor differences
   - Prioritize functional correctness

3. **Batch Similar Issues**
   - If same issue appears in multiple examples
   - Document once and reference

4. **Take Screenshots**
   - For significant visual differences
   - Save in comparison_results/screenshots/

### Quality Standards

An example passes validation when:
- ✓ All major sections present
- ✓ Algorithms correctly implemented  
- ✓ Numerical results within tolerance
- ✓ Visualizations convey same information
- ✓ Educational value maintained or improved

Remember: The goal is functional parity, not character-by-character matching. Python examples should be as good or better than MATLAB versions for teaching SHM concepts.