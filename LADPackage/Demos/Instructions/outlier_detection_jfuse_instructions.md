# Demo: Feature Extraction and Outlier Detection

**Dataset**: Base-excited 3-Story structure dataset as described in SHMTools/Documentation/ExampleUsages.pdf

**Goal**: Estimate AR model parameters, then detect outliers using the Mahalanobis distance and plot results. Swap out or add additional feature extraction methods to compare results for different features.

## Sequence

### Step 1: Import 3 Story Structure Dataset (LADPackage/Outlier Models)
**Input**: Floor Numbers = 5 (Default)  
**Output**: Dataset = (TIME, CHANNELS, INSTANCES)  
**Output**: Damage States = (INSTANCES, 1)  
**Output**: List of States = (INSTANCES,1)

### Step 2: AR Model (SHMFunctions/Feature Extraction/Time Series Models)
**Input**: Time Series Data = topFloorChannel (Step 1)  
**Input**: AR Model Order = 15  
**Output**: AR Parameters Feature Vectors = (INSTANCES, FEATURES)  
**Output**: RMS Residuals Feature Vectors = (INSTANCES, FEATURES)  
**Output**: AR Parameters = (ARORDER, CHANNELS, INSTANCES)  
**Output**: AR Residuals = (TIME, CHANNELS, INSTANCES)  
**Output**: AR Prediction = (TIME, CHANNELS, INSTANCES)

### Step 3: Plot Features (SHMFunctions/Feature Extraction)
**Input**: Feature Vectors = AR Parameters Feature Vectors (Step 2)  
**Input**: Instances to Plot = leave as default  
**Input**: Features to Plot = 1:4  
**Input**: Titles for Subplots = default  
**Input**: Y-Axis Labels for Subplots = leave as default  
**Input**: Axes Handle = leave as default  
**Output**: Axes Handle = scalar

### Step 4: Learn Score Mahalanobis (LADPackage/Feature Models)
**Input**: Features = AR Parameters Feature Vectors (Step 2)  
**Input**: Training Indices = 1:2:91  
**Input**: Scoring Indices = leave as default  
**Output**: Scores (INSTANCES, 1)

### Step 5: Plot Scores (SHMFunctions/Feature Classification)
**Input**: Scores = Scores (Step 4)  
**Input**: States = Damage States (Step 1)  
**Input**: State Names = default  
**Input**: Thresholds = default  
**Input**: Flip Signs = true  
**Input**: Use Log Scores = default  
**Input**: Axes Handle = leave as default  
**Output**: Axes Handle = scalar

### Step 6: Plot Score Distributions (SHMFunctions/Feature Classification)
**Input**: Scores = Scores (Step 4)  
**Input**: Damage States = Damage States (Step 1)  
**Input**: State Names = default  
**Input**: Thresholds = default  
**Input**: Flip Signs = true  
**Input**: Use Log Scores = true  
**Input**: Kernel Smoothing Parameter = true  
**Input**: Axes Handle = leave as default  
**Output**: Axes Handle = scalar

### Step 7: Receiver Operating Characteristic (ROC) Curve (SHMFunctions/Feature Classification)
**Input**: Scores = Scores (Step 4)  
**Input**: Damage States = Damage States (Step 1)  
**Input**: # of Points = default  
**Input**: Threshold Type = 'below' (default)  
**Output**: True Positive Rate = (INSTANCES, 1)  
**Output**: False Positive Rate = (INSTANCES, 1)

### Step 8: Plot Receiver Operating Characteristic (ROC) Curve (SHMFunctions/Feature Classification)
**Input**: True Positive Rate = True Positive Rate (Step 7)  
**Input**: False Positive Rate = False Positive Rate (Step 7)  
**Input**: Scaling = default  
**Input**: Axes Handle = default  
**Output**: Axes Handle = scalar