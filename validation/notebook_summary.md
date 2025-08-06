# Notebook Structure Summary

## time_synchronous_averaging_demo
**Category**: basic
**Title**: Time Synchronous Averaging for Condition-Based Monitoring
**Code Cells**: 6
**Markdown Cells**: 7

### Sections
- Time Synchronous Averaging for Condition-Based Monitoring
  - Background
  - Generate Synthetic Machinery Signal
  - Apply Time Synchronous Averaging
  - Visualize Results
  - Compare Frequency Content
  - Extract Random Component
  - Summary
    - Key Results:
    - Applications:
    - Next Steps:

### Plots (3 total)
1. **Type**: unknown
2. **Type**: unknown
3. **Type**: unknown

### Numerical Outputs (0 captured)

---

## svd_outlier_detection
**Category**: basic
**Title**: Outlier Detection Based on Singular Value Decomposition
**Code Cells**: 11
**Markdown Cells**: 9

### Sections
- Outlier Detection Based on Singular Value Decomposition
  - Introduction
  - Load Raw Data
    - Plot Time History Segments
  - Extraction of Damage-Sensitive Features
    - Prepare Training and Test Data
  - Statistical Modeling for Feature Classification
    - Plot Damage Indicators
  - Receiver Operating Characteristic Curve
  - Summary

### Plots (3 total)
1. **Type**: unknown
2. **Type**: bar
   - Title: Damage Indicators (DIs) from the Test Data
   - Y-axis: DI's Amplitude
3. **Type**: line
   - Title: ROC Curve for the Test Data
   - X-axis: False Alarm - FPR
   - Y-axis: True Detection - TPR

### Numerical Outputs (0 captured)

---

## pca_outlier_detection
**Category**: basic
**Title**: Outlier Detection Based on Principal Component Analysis
**Code Cells**: 9
**Markdown Cells**: 10

### Sections
- Outlier Detection Based on Principal Component Analysis
  - Introduction
  - Load Raw Data
    - Plot Time History from Baseline and Damaged Conditions
  - Extraction of Damage-Sensitive Features
    - Prepare Training and Test Data
    - Plot Test Data Features
  - Statistical Modeling for Feature Classification
  - Outlier Detection
    - Plot Damage Indicators
  - Summary

### Plots (3 total)
1. **Type**: unknown
2. **Type**: line
   - Title: Features from all Sensors in Concatenated Format for the Test Data
   - X-axis: Channel
   - Y-axis: RMSE
3. **Type**: bar
   - Title: Damage Indicators from the Test Data
   - X-axis: State Condition [Undamaged(1-9) and Damaged (10-17)]
   - Y-axis: DI

### Numerical Outputs (0 captured)

---

## mahalanobis_outlier_detection
**Category**: basic
**Title**: Outlier Detection Based on Mahalanobis Distance
**Code Cells**: 9
**Markdown Cells**: 10

### Sections
- Outlier Detection Based on Mahalanobis Distance
  - Introduction
  - Load Raw Data
    - Plot Time History from Baseline and Damaged Conditions
  - Extraction of Damage-Sensitive Features
    - Prepare Training and Test Data
    - Plot Test Data Features
  - Statistical Modeling for Feature Classification
  - Outlier Detection
    - Plot Damage Indicators
  - Summary

### Plots (3 total)
1. **Type**: unknown
2. **Type**: line
   - Title: Feature Vectors Compose of AR Parameters from Channel2-5
   - X-axis: AR Parameters in Concatenated Format
   - Y-axis: Amplitude
3. **Type**: bar
   - Title: Damage Indicators from the Test Data
   - X-axis: State Condition [Undamaged(1-9) and Damaged (10-17)]
   - Y-axis: DI

### Numerical Outputs (0 captured)

---

## default_detector_usage
**Category**: basic
**Title**: Default Detector Usage: High-Level Outlier Detection
**Code Cells**: 15
**Markdown Cells**: 14

### Sections
- Default Detector Usage: High-Level Outlier Detection
  - Overview
  - Setup and Imports
  - Load Raw Data
  - Plot Sample Time Histories
  - Data Segmentation
  - Feature Extraction
  - Prepare Train/Test Split
  - Train Default Outlier Detector
  - Detect Outliers
  - Calculate Performance Metrics
  - ROC Curve Analysis
  - Visualize Score Distributions
  - Confidence Analysis
  - Summary and Conclusions
    - Key Findings
    - Usage Recommendations
    - Next Steps

### Plots (4 total)
1. **Type**: unknown
2. **Type**: line
3. **Type**: unknown
4. **Type**: unknown

### Numerical Outputs (0 captured)

---

## default_detector_usage_backup
**Category**: basic
**Title**: Default Detector Usage: High-Level Outlier Detection
**Code Cells**: 15
**Markdown Cells**: 14

### Sections
- Default Detector Usage: High-Level Outlier Detection
  - Overview
  - Setup and Imports
  - Load Raw Data
  - Plot Sample Time Histories
  - Data Segmentation
  - Feature Extraction
  - Prepare Train/Test Split
  - Train Default Outlier Detector
  - Detect Outliers
  - Calculate Performance Metrics
  - ROC Curve Analysis
  - Visualize Score Distributions
  - Confidence Analysis
  - Summary and Conclusions
    - Key Findings
    - Usage Recommendations
    - Next Steps

### Plots (4 total)
1. **Type**: unknown
2. **Type**: line
3. **Type**: unknown
4. **Type**: unknown

### Numerical Outputs (0 captured)

---

## ar_model_order_selection
**Category**: basic
**Title**: Appropriate Autoregressive Model Order
**Code Cells**: 5
**Markdown Cells**: 7

### Sections
- Appropriate Autoregressive Model Order
  - Introduction
  - Load Raw Data
    - Plot Time History
  - Run Algorithm to find out the Appropriate AR Model Order
    - Plot Results
  - Summary
    - Visualize Model Predictions and Residuals

### Plots (2 total)
1. **Type**: line
   - Title: Acceleration Time History (Channel 5)
   - X-axis: Data Points
   - Y-axis: Acceleration (g)
2. **Type**: line
   - X-axis: AR Order (p)
   - Y-axis: Magnitude

### Numerical Outputs (0 captured)

---

## damage_localization_ar_arx
**Category**: intermediate
**Title**: Damage Localization using AR and ARX Models
**Code Cells**: 20
**Markdown Cells**: 27

### Sections
  - Load Data
  - Setup and Imports
- Damage Localization using AR and ARX Models
  - Overview
    - Key Concepts:
    - Dataset:
- Damage Localization using AR/ARX Models
  - Overview
  - Setup and Imports
  - Load Raw Data
  - Plot Sample Time Histories
  - DLAR: Damage Localization using AR Parameters
    - Extract AR Model Features
    - Visualize AR Parameters
    - Compute Channel-wise Damage Indicators (AR)
    - Plot AR Damage Indicators
  - DLARX: Damage Localization using ARX Parameters
    - Extract ARX Model Features
    - Visualize ARX Parameters
    - Compute Channel-wise Damage Indicators (ARX)
    - Plot ARX Damage Indicators
  - Damage Localization Analysis
    - AR Method Analysis
    - ARX Method Analysis
    - AR vs ARX Comparison
    - Side-by-Side Comparison Plot
    - Sensitivity Improvement Analysis
  - Summary and Conclusions
    - Key Findings
    - Method Comparison
    - Practical Implications
    - Recommendations
  - Part 1: Damage Localization using AR Parameters
  - Part 2: Damage Localization using ARX Parameters
  - Conclusions
    - Key Findings:
    - References

### Plots (5 total)
1. **Type**: unknown
2. **Type**: line
   - X-axis: AR Parameters
   - Y-axis: Amplitude
3. **Type**: line
   - X-axis: ARX Parameters
   - Y-axis: Amplitude
4. **Type**: unknown
5. **Type**: unknown

### Numerical Outputs (0 captured)

---

## factor_analysis_outlier_detection
**Category**: intermediate
**Title**: Outlier Detection Based on Factor Analysis
**Code Cells**: 11
**Markdown Cells**: 10

### Sections
- Outlier Detection Based on Factor Analysis
  - Introduction
  - Load Raw Data
    - Plot Sample Time Histories
  - Extraction of Damage-Sensitive Features
    - Prepare Training and Test Data
  - Statistical Modeling for Feature Classification
    - Plot Damage Indicators
    - Visualize Factor Analysis Components
  - Receiver Operating Characteristic Curve
  - Summary

### Plots (4 total)
1. **Type**: unknown
2. **Type**: bar
   - Title: Damage Indicators (DIs) from Factor Analysis', fontsize=14, fontweight='bold
   - Y-axis: DI's Amplitude
3. **Type**: unknown
4. **Type**: line
   - Title: ROC Curve for Factor Analysis Outlier Detection', fontsize=14, fontweight='bold
   - X-axis: False Alarm Rate (FPR)
   - Y-axis: True Detection Rate (TPR)

### Numerical Outputs (0 captured)

---

## parametric_distribution_outlier_detection
**Category**: intermediate
**Title**: Parametric Distribution Outlier Detection
**Code Cells**: 17
**Markdown Cells**: 21

### Sections
- Parametric Distribution Outlier Detection
  - Introduction
  - Setup and Imports
  - Load Raw Data
  - Extraction of Damage-Sensitive Features
  - Statistical Modeling For Feature Classification
  - Define the Underlying Distribution of the Undamaged Condition
  - Confidence Interval
  - Hypothesis Test
  - Summary and Conclusions
    - Methodology
    - Classification Performance
    - Advantages of Parametric Approach
    - Key Insights

### Plots (6 total)
1. **Type**: line
   - Title: Two Time Histories (State 1 and 10) in Concatenated Format
   - X-axis: Observations
   - Y-axis: Accelerations (g)
2. **Type**: line
   - X-axis: AR Parameters
   - Y-axis: Amplitude
3. **Type**: line
   - Title: Histogram along with Superimposed Idealized Chi-square PDF (Undamaged Condition)
   - X-axis: DI's Amplitude
   - Y-axis: Probability
4. **Type**: line
   - Title: Chi-square CDF (Undamaged Condition)
   - X-axis: DI's Amplitude
   - Y-axis: Probability
5. **Type**: line
   - Title: Damage Indicators for the Test Data
   - X-axis: State Condition\n[Undamaged(1-90) and Damaged (91-170)]
   - Y-axis: DI's Amplitude
6. **Type**: unknown
   - Title: P-values for Hypothesis Testing
   - X-axis: State Condition\n[Undamaged(1-90) and Damaged (91-170)]
   - Y-axis: P-value (log scale)

### Numerical Outputs (0 captured)

---

## fast_metric_kernel_density
**Category**: advanced
**Title**: Fast Metric Kernel Density Estimation for Outlier Detection
**Code Cells**: 12
**Markdown Cells**: 13

### Sections
- Fast Metric Kernel Density Estimation for Outlier Detection
  - Overview
  - Theory
  - Load and Prepare Data
  - Feature Extraction
  - Data Splitting
  - Fast Metric KDE vs Standard KDE
  - Score Test Data
  - Visualize Score Distributions
  - Threshold Selection and Classification
  - Performance Evaluation
  - Compare Different Distance Metrics
  - Visualize Metric Comparison
  - Computational Efficiency Analysis
  - Summary and Conclusions
    - Key Findings:
    - Practical Recommendations:
    - Applications in SHM:

### Plots (4 total)
1. **Type**: unknown
2. **Type**: unknown
3. **Type**: unknown
4. **Type**: unknown
   - Title: Computational Scaling of Fast Metric KDE
   - X-axis: Dataset Size
   - Y-axis: Time (seconds)

### Numerical Outputs (0 captured)

---

## active_sensing_feature_extraction
**Category**: advanced
**Title**: Active Sensing Feature Extraction
**Code Cells**: 22
**Markdown Cells**: 19

### Sections
- Active Sensing Feature Extraction
  - Introduction
  - Configuration Parameters
  - Load Data and DAQ Parameters
  - Collect Border Line Segments into One Array
  - Extract Data for Sensor Subset
  - Build Contained Grid of Points
  - Propagation Distance to Points
  - Propagation Distance to Boundary
  - Line of Sight
  - Distance Compare
  - Estimate Group Velocity
  - Distance 2 Index
  - Difference
  - Incoherent Matched Filter
  - Extract Subset
  - Apply Logic Filters
  - Sum Dimensions
  - Fill 2D Map
  - Plot 2D Map

### Plots (5 total)
1. **Type**: line
   - Title: Structure Layout with Sensors and Damage Location
   - X-axis: in
   - Y-axis: in
2. **Type**: line
   - X-axis: n
   - Y-axis: Volts
3. **Type**: line
   - X-axis: n
   - Y-axis: Volts
4. **Type**: line
   - X-axis: n
   - Y-axis: Volts
5. **Type**: line
   - Title: Active Sensing Damage Detection Map
   - X-axis: in
   - Y-axis: in

### Numerical Outputs (0 captured)

---

## semiparametric_outlier_detection
**Category**: advanced
**Title**: Example Usage: Direct Use of Semi-Parametric Routines
**Code Cells**: 16
**Markdown Cells**: 20

### Sections
- Example Usage: Direct Use of Semi-Parametric Routines
  - Introduction
  - Load data
  - Train a model over the undamaged data
  - Pick a threshold from the training data
  - Test the detector
  - Report the detector's performance

### Plots (1 total)
1. **Type**: line
   - Title: ROC curve
   - X-axis: falsePositives
   - Y-axis: truePositives

### Numerical Outputs (0 captured)

---

## custom_detector_assembly
**Category**: advanced
**Title**: Custom Detector Assembly
**Code Cells**: 10
**Markdown Cells**: 13

### Sections
- Custom Detector Assembly
  - Overview
  - Setup and Imports
  - Load and Prepare Data
  - Explore Available Detectors
  - Example 1: Assemble a Parametric Detector (PCA)
  - Use the Assembled PCA Detector
  - Example 2: Assemble a Non-Parametric Detector (Kernel Density)
  - Example 3: Assemble a Semi-Parametric Detector (GMM)
  - Compare Detector Performance
  - Save and Load Detector Configurations
  - Interactive Assembly Example
- Example of interactive assembly (commented out for notebook execution)
- interactive_detector = assemble_outlier_detector_shm(interactive=True)
  - Visualize Score Distributions
  - Summary

### Plots (2 total)
1. **Type**: line
   - Title: ROC Curves for Assembled Detectors
   - X-axis: False Positive Rate
   - Y-axis: True Positive Rate
2. **Type**: unknown

### Numerical Outputs (0 captured)

---

## nonparametric_outlier_detection
**Category**: advanced
**Title**: Direct Use of Nonparametric Outlier Detection
**Code Cells**: 16
**Markdown Cells**: 6

### Sections
- Direct Use of Nonparametric Outlier Detection
  - Introduction
  - Load data
  - Train a model over the undamaged data
  - Pick a threshold from the training data
  - Test the detector
  - Report the detector's performance

### Plots (1 total)
1. **Type**: line
   - Title: ROC curve
   - X-axis: falsePositives
   - Y-axis: truePositives

### Numerical Outputs (0 captured)

---

## nlpca_outlier_detection
**Category**: advanced
**Title**: Outlier Detection based on Nonlinear Principal Component Analysis
**Code Cells**: 12
**Markdown Cells**: 15

### Sections
- Outlier Detection based on Nonlinear Principal Component Analysis
  - Introduction
    - References:
    - SHMTools functions used:
  - Load Raw Data
  - Extraction of Damage-Sensitive Features
  - Statistical Modeling for Feature Classification
  - Plot Damage Indicators
  - Performance Analysis
  - Summary
    - Key Results:
    - Note:
    - See also:

### Plots (3 total)
1. **Type**: unknown
2. **Type**: line
   - Title: First Four Statistical Moments of the Test Data
   - X-axis: Features
   - Y-axis: Amplitude
3. **Type**: bar
   - Title: Damage Indicators of the Test Data
   - X-axis: State Condition\n[Undamaged(1-9) and Damaged (10-17)]
   - Y-axis: DI's Amplitude

### Numerical Outputs (7 captured)
1. Dataset shape: (8192, 5, 170)
Time points: 8192
Channels: 5
Conditions: 170
2. Statistical moments shape: (170, 4)
Features per channel: 4 (mean, std, skew, kurt)
Instances: 170
3. Training data shape: (81, 4)
Training samples: 81 (9 states × 9 tests)

---

## active_sensing_feature_extraction_backup
**Category**: advanced
**Title**: Active Sensing Feature Extraction
**Code Cells**: 6
**Markdown Cells**: 8

### Sections
- Active Sensing Feature Extraction
  - Introduction
  - Load Active Sensing Data
- Estimate group velocity from baseline measurements
- Use a subset of pairs for velocity estimation (computational efficiency)
- Estimate actuation width from frequency (handle array case)
- For velocity estimation, we need to use only the relevant sensors
- The waveform data has many channels, but we only have a few actual sensor locations
- Estimate velocity
- Use reasonable default if estimation fails
- Visualize velocity estimates
  - Estimate Group Velocity
  - Create Spatial Grid and Calculate Propagation Distances
  - Apply Matched Filtering
  - Feature Extraction and Spatial Mapping
  - Summary

### Plots (4 total)
1. **Type**: unknown
2. **Type**: unknown
3. **Type**: unknown
4. **Type**: unknown

### Numerical Outputs (0 captured)

---

## modal_analysis_features
**Category**: advanced
**Title**: Modal Analysis Feature Extraction for Damage Detection
**Code Cells**: 16
**Markdown Cells**: 21

### Sections
- Modal Analysis Feature Extraction for Damage Detection
  - Background
    - References
  - Introduction
    - References
    - References
  - Load Dataset
  - Step 1: Compute Frequency Response Functions (FRF)
    - Visualize Sample Time Histories
  - Extraction of Damage-Sensitive Features
    - Set Analysis Parameters
    - Compute Frequency Response Functions (FRFs)
    - Visualize Sample FRFs
    - Extract Natural Frequencies
    - Prepare Feature Data
    - Visualize Feature Vectors
  - Statistical Modeling for Feature Classification
    - PCA-Based Outlier Detection
    - Mahalanobis Distance-Based Outlier Detection
    - Visualize Damage Indicators
  - ROC Curve Analysis
  - Analysis and Conclusions
    - Summary and Recommendations
  - Technical Notes
    - Original MATLAB vs. Python Implementation
    - For Complete NLPCA Implementation

### Plots (5 total)
1. **Type**: unknown
2. **Type**: unknown
3. **Type**: line
   - Title: Feature Vectors Composed of Natural Frequencies
   - X-axis: Mode Number
   - Y-axis: Frequency (Hz)
4. **Type**: unknown
5. **Type**: line
   - Title: ROC Curves for Modal Analysis-Based Damage Detection
   - X-axis: False Positive Rate (FPR)
   - Y-axis: True Positive Rate (TPR)

### Numerical Outputs (0 captured)

---

## modal_analysis_features_simplified
**Category**: advanced
**Title**: Modal Analysis and Frequency Response Functions
**Code Cells**: 9
**Markdown Cells**: 11

### Sections
- Modal Analysis and Frequency Response Functions
  - Introduction
  - Load and Examine Data
    - Visualize Sample Time Histories
  - Frequency Response Function (FRF) Computation
    - Visualize FRFs from Different Structural States
  - Natural Frequency Tracking Across All Conditions
    - Analyze Natural Frequency Trends
    - Compare Baseline vs. Later Conditions
  - Conclusions and Summary
  - References

### Plots (4 total)
1. **Type**: unknown
2. **Type**: unknown
3. **Type**: unknown
4. **Type**: unknown

### Numerical Outputs (0 captured)

---

## modal_osp
**Category**: advanced
**Title**: Optimal Sensor Placement Using Modal Analysis Based Approaches
**Code Cells**: 12
**Markdown Cells**: 10

### Sections
- Optimal Sensor Placement Using Modal Analysis Based Approaches
  - Introduction
    - References
    - SHMTools functions demonstrated
  - Setup and Imports
  - Load Example Modal Data
  - Visualization Functions
  - Plot Mode Shapes
  - OSP Fisher Information Matrix, Effective Independence Method
  - OSP Maximum Norm Method
  - Compare Methods
  - Analysis of Results
  - Summary
    - Fisher Information EI Method
    - Maximum Norm Method
    - Key Findings

### Plots (5 total)
1. **Type**: unknown
2. **Type**: unknown
   - Title: Fisher Information Matrix Determinant Convergence
   - X-axis: Iteration
   - Y-axis: det(Q)
3. **Type**: unknown
4. **Type**: unknown
5. **Type**: unknown

### Numerical Outputs (0 captured)

---

## modal_analysis_features_backup
**Category**: advanced
**Title**: Modal Analysis Feature Extraction for Damage Detection
**Code Cells**: 16
**Markdown Cells**: 21

### Sections
- Modal Analysis Feature Extraction for Damage Detection
  - Background
    - References
  - Introduction
    - References
    - References
  - Load Dataset
  - Step 1: Compute Frequency Response Functions (FRF)
    - Visualize Sample Time Histories
  - Extraction of Damage-Sensitive Features
    - Set Analysis Parameters
    - Compute Frequency Response Functions (FRFs)
    - Visualize Sample FRFs
    - Extract Natural Frequencies
    - Prepare Feature Data
    - Visualize Feature Vectors
  - Statistical Modeling for Feature Classification
    - PCA-Based Outlier Detection
    - Mahalanobis Distance-Based Outlier Detection
    - Visualize Damage Indicators
  - ROC Curve Analysis
  - Analysis and Conclusions
    - Summary and Recommendations
  - Technical Notes
    - Original MATLAB vs. Python Implementation
    - For Complete NLPCA Implementation

### Plots (5 total)
1. **Type**: unknown
2. **Type**: unknown
3. **Type**: line
   - Title: Feature Vectors Composed of Natural Frequencies
   - X-axis: Mode Number
   - Y-axis: Frequency (Hz)
4. **Type**: unknown
5. **Type**: line
   - Title: ROC Curves for Modal Analysis-Based Damage Detection
   - X-axis: False Positive Rate (FPR)
   - Y-axis: True Positive Rate (TPR)

### Numerical Outputs (0 captured)

---

