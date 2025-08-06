# Validation Checklist

## Converted Examples Status

| Example | HTML Exists | PDF Generated | Structure Valid | Results Valid | Issues Found | Fixed | Notes |
|---------|-------------|---------------|-----------------|---------------|--------------|-------|-------|
| AR Model Order Selection | ✓ | ✓ | ✓ | ✓ | None | ✓ | PASS - Perfect match |
| Mahalanobis Outlier Detection | ✓ | ✓ | ✓ | ✓ | None | ✓ | PASS - Perfect match |
| PCA Outlier Detection | ✓ | ✓ | ✓ | ✓ | None | ✓ | PASS - Perfect match |
| SVD Outlier Detection | ✓ | ✓ | ✓ | ✓ | None | ✓ | PASS - Perfect match |
| Time Synchronous Averaging | ✓ | ✓ | ✓ | ✓ | None | ✓ | PASS - New Python enhancement |
| Default Detector Usage | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| Factor Analysis Outlier Detection | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| Damage Localization AR/ARX | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| Nonparametric Outlier Detection | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| Active Sensing Feature Extraction | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| NI DAQ Integration | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| Dataset Management | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| NLPCA Outlier Detection | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| Parametric Distribution Outlier | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| Semiparametric Outlier Detection | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| Fast Metric Kernel Density | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| Modal Analysis Features | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| Modal OSP | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| CBM Gear Box Analysis | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| Sensor Diagnostics | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| Custom Detector Assembly | ✓ | ✓ | ☐ | ☐ | - | ☐ | |
| DataLoader Demo | ✓ | ✓ | ☐ | ☐ | - | ☐ | |

## Examples in ExampleUsages.pdf Not Yet Converted

(To be filled after initial PDF review)

1. Example Name - Page #
2. ...

## Common Issues Found

### Structural Issues
- [ ] Issue type 1
- [ ] Issue type 2

### Numerical Discrepancies  
- [ ] Issue type 1
- [ ] Issue type 2

### Visualization Differences
- [ ] Issue type 1
- [ ] Issue type 2

## Validation Process Notes

1. Run PDF conversion: `python scripts/convert_html_to_pdf.py --merge`
2. Open PDFs side by side for comparison
3. Update this checklist with findings
4. Create detailed reports in `validation/comparison_results/`