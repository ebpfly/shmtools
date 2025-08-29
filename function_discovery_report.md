# SHM Function Discovery Report
Generated: 2025-08-29 16:45:21

This report shows all functions discovered by the JupyterLab extension's introspection system.

## Auxiliary - Plotting (6 functions)

- `add_resp_2_geom_shm()` - Add response vector to geometry for deformed shape visualization.
  - Display Name: Add Response to Geometry
  - Module: `shmtools.modal`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/modal/osp.py

- `node_element_plot_shm()` - Plot structural geometry with optional deformed shape.
  - Display Name: Node Element Plot
  - Module: `shmtools.modal`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/modal/osp.py

- `plot_scores_shm()` - Plot damage detection scores with threshold and classification results.
  - Display Name: Plot Detection Scores
  - Module: `shmtools.plotting`
  - Parameters: 7
  - File: /Users/eric/repo/shm/shmtools/plotting/spectral_plots.py

- `plot_sensors_shm()` - Plot sensor locations on structure.
  - Display Name: Plot Sensors
  - Module: `shmtools.modal`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/modal/osp.py

- `plot_spectrogram_shm()` - Plot spectrogram with proper formatting.
  - Display Name: Plot Spectrogram
  - Module: `shmtools`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/plotting/spectral_plots.py

- `response_interp_shm()` - Interpolate response from DOF indices to node XYZ coordinates.
  - Display Name: Response Interpolation
  - Module: `shmtools.modal`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/modal/osp.py

## Auxiliary - Sensor Support (3 functions)

- `sd_autoclassify_shm()` - Automatically classify sensor operational status.
  - Display Name: Sensor Diagnostics Auto-Classify
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/sensor_diagnostics/sensor_diagnostics.py

- `sd_feature_shm()` - Extract capacitance values from piezoelectric sensor admittance data.
  - Display Name: Sensor Diagnostics Feature
  - Module: `shmtools`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/sensor_diagnostics/sensor_diagnostics.py

- `sd_plot_shm()` - Plot sensor diagnostic results.
  - Display Name: Sensor Diagnostics Result Plot
  - Module: `shmtools`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/sensor_diagnostics/sensor_diagnostics.py

## Auxiliary - Sensor Support - Optimal Sensor Placement (3 functions)

- `get_sensor_layout_shm()` - Convert optimal DOF indices to sensor XYZ coordinates.
  - Display Name: Get Sensor Layout
  - Module: `shmtools.modal`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/modal/osp.py

- `osp_fisher_info_eiv_shm()` - Optimal sensor placement using Fisher Information Matrix and Effective Independence method.
  - Display Name: OSP Fisher Information EI
  - Module: `shmtools.modal`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/modal/osp.py

- `osp_max_norm_shm()` - Optimal sensor placement using Maximum Norm method.
  - Display Name: OSP Maximum Norm
  - Module: `shmtools.modal`
  - Parameters: 6
  - File: /Users/eric/repo/shm/shmtools/modal/osp.py

## Auxiliary - Utilities (1 functions)

- `split_features_shm()` - Split feature vectors into training and scoring sets.
  - Display Name: Split Features Into Training and Scoring
  - Module: `shmtools.features`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/features/time_series.py

## Classification & Detection (2 functions)

- `load_detector_assembly()` - Load assembled detector configuration from file.
  - Display Name: Load Detector Assembly
  - Module: `shmtools.classification`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/classification/custom_detector_assembly.py

- `save_detector_assembly()` - Save assembled detector configuration to file.
  - Display Name: Save Detector Assembly
  - Module: `shmtools.classification`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/classification/custom_detector_assembly.py

## Classification - Detector Assembly (1 functions)

- `assemble_outlier_detector_shm()` - Assemble custom outlier detector with interactive or programmatic configuration.
  - Display Name: Assemble Outlier Detector
  - Module: `shmtools.classification`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/classification/custom_detector_assembly.py

## Classification - LADPackage Utils (1 functions)

- `learn_score_mahalanobis()` - LADPackage wrapper: Split data, train, and score using Mahalanobis distance.
  - Display Name: Learn Score Mahalanobis
  - Module: `LADPackage.utils`
  - Parameters: 2
  - File: /Users/eric/repo/shm/LADPackage/utils/learn_score_mahalanobis.py

## Core - Signal Processing (12 functions)

- `analytic_signal()` - Compute analytic signal using Hilbert transform.
  - Display Name: Analytic Signal
  - Module: `shmtools.core.preprocessing`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/core/preprocessing.py

- `bandpass_filter()` - Apply bandpass filter to signal.
  - Display Name: Bandpass Filter
  - Module: `shmtools.core.filtering`
  - Parameters: 7
  - File: /Users/eric/repo/shm/shmtools/core/filtering.py

- `crest_factor()` - Compute crest factor (peak-to-RMS ratio).
  - Display Name: Crest Factor
  - Module: `shmtools.core.statistics`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `demean()` - Remove mean from signal.
  - Display Name: Demean
  - Module: `shmtools.core.preprocessing`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/preprocessing.py

- `envelope()` - Compute envelope of signal.
  - Display Name: Envelope
  - Module: `shmtools.core.preprocessing`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/preprocessing.py

- `filter_signal()` - Apply digital filter to signal.
  - Display Name: Filter Signal
  - Module: `shmtools.core.filtering`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/core/filtering.py

- `highpass_filter()` - Apply highpass filter to signal.
  - Display Name: Highpass Filter
  - Module: `shmtools.core.filtering`
  - Parameters: 6
  - File: /Users/eric/repo/shm/shmtools/core/filtering.py

- `lowpass_filter()` - Apply lowpass filter to signal.
  - Display Name: Lowpass Filter
  - Module: `shmtools.core.filtering`
  - Parameters: 6
  - File: /Users/eric/repo/shm/shmtools/core/filtering.py

- `rms()` - Compute root mean square (RMS) value.
  - Display Name: Rms
  - Module: `shmtools.core.statistics`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `scale_min_max()` - Scale signal to specified range.
  - Display Name: Scale Min Max
  - Module: `shmtools.core.preprocessing`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/preprocessing.py

- `statistical_moments()` - Compute statistical moments of a signal.
  - Display Name: Statistical Moments
  - Module: `shmtools.core`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `window_signal()` - Apply window function to signal.
  - Display Name: Window Signal
  - Module: `shmtools.core.preprocessing`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/preprocessing.py

## Data Acquisition - Signal Generation (1 functions)

- `band_lim_white_noise_shm()` - Generate band-limited white noise for structural excitation.
  - Display Name: Band-Limited White Noise
  - Module: `shmtools.hardware`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/hardware/signal_generation.py

## Data Import (5 functions)

- `import_3story_structure_shm()` - Import 3-story structure experimental data.
  - Display Name: Import 3Story Structure
  - Module: `examples.data`
  - Parameters: 0
  - File: /Users/eric/repo/shm/examples/data/data_imports.py

- `import_active_sense1_shm()` - Import active sensing experimental dataset #1.
  - Display Name: Import Active Sense1
  - Module: `examples.data`
  - Parameters: 0
  - File: /Users/eric/repo/shm/examples/data/data_imports.py

- `import_cbm_data_shm()` - Import condition-based monitoring experimental data.
  - Display Name: Import Cbm Data
  - Module: `examples.data`
  - Parameters: 0
  - File: /Users/eric/repo/shm/examples/data/data_imports.py

- `import_modal_osp_shm()` - Import modal analysis data for optimal sensor placement studies.
  - Display Name: Import Modal Osp
  - Module: `examples.data`
  - Parameters: 0
  - File: /Users/eric/repo/shm/examples/data/data_imports.py

- `import_sensor_diagnostic_shm()` - Import sensor diagnostic dataset for health assessment.
  - Display Name: Import Sensor Diagnostic
  - Module: `examples.data`
  - Parameters: 0
  - File: /Users/eric/repo/shm/examples/data/data_imports.py

## Data Import - LADPackage (1 functions)

- `import_3story_structure_sub_floors()` - LADPackage-compatible version of 3-story structure data import.
  - Display Name: Import 3 Story Structure Dataset
  - Module: `LADPackage.utils`
  - Parameters: 1
  - File: /Users/eric/repo/shm/LADPackage/utils/data_import.py

## Feature Classification - High Level Interface (2 functions)

- `detect_outlier_shm()` - Detect outliers in test features using trained models.
  - Display Name: Detect Outlier
  - Module: `shmtools.classification`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/classification/high_level_detection.py

- `train_outlier_detector_shm()` - Train outlier detector using semi-parametric Gaussian mixture model.
  - Display Name: Train Outlier Detector
  - Module: `shmtools.classification`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/classification/high_level_detection.py

## Feature Classification - Non-Parametric Detectors (13 functions)

- `cosine_kernel_shm()` - Kernel weights for the Cosine kernel.
  - Display Name: Cosine Kernel Weights
  - Module: `shmtools`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/classification/nonparametric.py

- `epanechnikov_kernel_shm()` - Kernel weights for the Epanechnikov kernel.
  - Display Name: Epanechnikov Kernel Weights
  - Module: `shmtools`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/classification/nonparametric.py

- `gaussian_kernel_shm()` - Kernel weights for the Gaussian kernel.
  - Display Name: Gaussian Kernel Weights
  - Module: `shmtools`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/classification/nonparametric.py

- `learn_fast_metric_kernel_density_shm()` - Learn fast metric kernel density estimation model.
  - Display Name: Learn Fast Metric Kernel Density
  - Module: `shmtools.classification`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/classification/nonparametric.py

- `learn_kernel_density_shm()` - Learn nonparametric kernel density estimation model.
  - Display Name: Learn Kernel Density Estimation
  - Module: `shmtools`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/classification/nonparametric.py

- `learn_nlpca_shm()` - Learn nonlinear principal component analysis (NLPCA) model.
  - Display Name: Learn NLPCA
  - Module: `shmtools`
  - Parameters: 6
  - File: /Users/eric/repo/shm/shmtools/classification/nlpca.py

- `quartic_kernel_shm()` - Kernel weights for the Quartic (Biweight) kernel.
  - Display Name: Quartic Kernel Weights
  - Module: `shmtools`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/classification/nonparametric.py

- `score_fast_metric_kernel_density_shm()` - Score fast metric kernel density estimation.
  - Display Name: Score Fast Metric Kernel Density
  - Module: `shmtools.classification`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/classification/nonparametric.py

- `score_kernel_density_shm()` - Score nonparametric kernel density estimation.
  - Display Name: Score Kernel Density Estimation
  - Module: `shmtools`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/classification/nonparametric.py

- `score_nlpca_shm()` - Score test data using trained NLPCA model.
  - Display Name: Score NLPCA
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/classification/nlpca.py

- `triangle_kernel_shm()` - Kernel weights for the Triangle kernel.
  - Display Name: Triangle Kernel Weights
  - Module: `shmtools`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/classification/nonparametric.py

- `triweight_kernel_shm()` - Kernel weights for the Triweight kernel.
  - Display Name: Triweight Kernel Weights
  - Module: `shmtools`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/classification/nonparametric.py

- `uniform_kernel_shm()` - Kernel weights for the Uniform kernel.
  - Display Name: Uniform Kernel Weights
  - Module: `shmtools`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/classification/nonparametric.py

## Feature Classification - Parametric Detectors (8 functions)

- `learn_factor_analysis_shm()` - Learn Factor Analysis model for outlier detection.
  - Display Name: Learn Factor Analysis
  - Module: `shmtools`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/classification/outlier_detection.py

- `learn_mahalanobis_shm()` - Learn Mahalanobis distance model from training data.
  - Display Name: Learn Mahalanobis
  - Module: `shmtools`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/classification/outlier_detection.py

- `learn_pca_shm()` - Learn principal component analysis (PCA) for outlier detection.
  - Display Name: Learn Principal Component Analysis
  - Module: `shmtools`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/classification/outlier_detection.py

- `learn_svd_shm()` - Learn SVD-based outlier detection model from training features.
  - Display Name: Learn Singular Value Decomposition
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/classification/outlier_detection.py

- `score_factor_analysis_shm()` - Score features using trained Factor Analysis outlier detection model.
  - Display Name: Score Factor Analysis
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/classification/outlier_detection.py

- `score_mahalanobis_shm()` - Score Mahalanobis distance for outlier detection.
  - Display Name: Score Mahalanobis
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/classification/outlier_detection.py

- `score_pca_shm()` - Score principal component analysis (PCA) for outlier detection.
  - Display Name: Score Principal Component Analysis
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/classification/outlier_detection.py

- `score_svd_shm()` - Score features using trained SVD outlier detection model.
  - Display Name: Score Singular Value Decomposition
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/classification/outlier_detection.py

## Feature Classification - Performance Evaluation (2 functions)

- `roc_shm()` - Receiver operating characteristic (ROC) curve.
  - Display Name: ROC Curve
  - Module: `shmtools`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/classification/outlier_detection.py

- `roc_shm()` - Receiver operating characteristic (ROC) curve.
  - Display Name: Receiver Operating Characteristic
  - Module: `shmtools.classification.nonparametric`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/classification/nonparametric.py

## Feature Classification - Semi-Parametric Detectors (5 functions)

- `k_medians_shm()` - Partition the data using k-medians clustering.
  - Display Name: K Median Clustering
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/classification/semiparametric.py

- `learn_gmm_semiparametric_model_shm()` - Learn GMM semi-parametric density model.
  - Display Name: Learn GMM Semi-Parametric Density Model
  - Module: `shmtools`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/classification/semiparametric.py

- `learn_gmm_shm()` - Learn gaussian mixture model.
  - Display Name: Learn Gaussian Mixture Model
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/classification/semiparametric.py

- `score_gmm_semiparametric_model_shm()` - Score GMM semi-parametric density model.
  - Display Name: Score GMM Semi-Parametric Density Model
  - Module: `shmtools`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/classification/semiparametric.py

- `score_gmm_shm()` - Score gaussian mixture model.
  - Display Name: Score Gaussian Mixture Model
  - Module: `shmtools`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/classification/semiparametric.py

## Feature Extraction (1 functions)

- `arx_model()` - Fit ARX (autoregressive with exogenous input) model.
  - Display Name: Arx Model
  - Module: `shmtools.features.time_series`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/features/time_series.py

## Feature Extraction - Active Sensing (14 functions)

- `build_contained_grid_shm()` - Build grid of points contained within structure borders.
  - Display Name: Build Contained Grid
  - Module: `shmtools`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/active_sensing/geometry.py

- `coherent_matched_filter_shm()` - Coherent matched filter for guided wave analysis.
  - Display Name: Coherent Matched Filter
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/active_sensing/matched_filter.py

- `distance_2_index_shm()` - Convert propagation distances to waveform sample indices.
  - Display Name: Distance to Index
  - Module: `shmtools`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/active_sensing/geometry.py

- `estimate_group_velocity_shm()` - Estimate group velocity from guided wave measurements.
  - Display Name: Estimate Wavespeed
  - Module: `shmtools`
  - Parameters: 6
  - File: /Users/eric/repo/shm/shmtools/active_sensing/utilities.py

- `extract_subsets_shm()` - Extract data subsets from array using start indices and fixed length.
  - Display Name: Extract Subsets
  - Module: `shmtools`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/active_sensing/utilities.py

- `fill_2d_map_shm()` - Fill 2D map from 1D data using boolean mask.
  - Display Name: Fill 2D Map
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/active_sensing/geometry.py

- `flex_logic_filter_shm()` - Apply flexible logical filtering to multi-dimensional data.
  - Display Name: Flexible Logic Filter
  - Module: `shmtools`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/active_sensing/utilities.py

- `get_prop_dist_2_boundary_shm()` - Calculate propagation distance from actuator to boundary to sensor.
  - Display Name: Get Propagation Distance to Boundary
  - Module: `shmtools.active_sensing`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/active_sensing/geometry.py

- `incoherent_matched_filter_shm()` - Incoherent matched filter for guided wave analysis.
  - Display Name: Incoherent Matched Filter
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/active_sensing/matched_filter.py

- `propagation_dist_2_points_shm()` - Calculate propagation distances from sensor pairs to points of interest.
  - Display Name: Propagation Distance to POIs
  - Module: `shmtools`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/active_sensing/geometry.py

- `reduce_2_pair_subset_shm()` - Extract parameter and data subsets based on sensor subset.
  - Display Name: Reduce to Pair Subset
  - Module: `shmtools.active_sensing`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/active_sensing/utilities.py

- `sensor_pair_line_of_sight_shm()` - Determine line-of-sight visibility for sensor pairs to points of interest.
  - Display Name: Sensor Pair Line of Sight
  - Module: `shmtools`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/active_sensing/geometry.py

- `struct_cell_2_mat_shm()` - Convert structure-cell mixture to matrix.
  - Display Name: Combine Structure-Cell
  - Module: `shmtools.active_sensing`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/active_sensing/geometry.py

- `sum_mult_dims_shm()` - Sum array along multiple dimensions.
  - Display Name: Sum Multiple Dimensions
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/active_sensing/utilities.py

## Feature Extraction - Condition Based Monitoring (4 functions)

- `fm0_shm()` - Feature Extraction: Compute FM0 gear damage indicator.
  - Display Name: FM0 Feature
  - Module: `shmtools`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `fm4_shm()` - Feature Extraction: Compute FM4 damage indicator from difference signal.
  - Display Name: FM4 Feature
  - Module: `shmtools`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `timeSyncAvg_shm()` - Time-synchronous average of angularly sampled signals.
  - Display Name: Time Synchronous Average
  - Module: `shmtools.features.condition_based_monitoring`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/features/condition_based_monitoring.py

- `time_sync_avg_shm()` - Time-synchronous average of angularly sampled signals.
  - Display Name: Time Synchronous Average
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/features/condition_based_monitoring.py

## Feature Extraction - Modal Analysis (2 functions)

- `frf_shm()` - Compute frequency response function (FRF) from time domain data.
  - Display Name: Frequency Response Function
  - Module: `shmtools`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/modal/modal_analysis.py

- `rpfit_shm()` - Extract modal parameters using rational polynomial curve-fitting.
  - Display Name: Rational Poly Fit
  - Module: `shmtools`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/modal/modal_analysis.py

## Feature Extraction - Preprocessing (13 functions)

- `analytic_signal_shm()` - Convert signals to their analytic form using Hilbert transform via FFT.
  - Display Name: Analytic Signal
  - Module: `shmtools.core.preprocessing`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/core/preprocessing.py

- `ars_tach_shm()` - Angular resampling using tachometer signal.
  - Display Name: Ars Tach
  - Module: `shmtools`
  - Parameters: 6
  - File: /Users/eric/repo/shm/shmtools/core/signal_processing.py

- `bandpass_condition_signal_shm()` - Extract bandpass filtered signal around fault frequencies.
  - Display Name: Bandpass Condition Signal
  - Module: `shmtools`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/core/signal_filtering.py

- `demean_shm()` - Remove signal mean from signal matrix.
  - Display Name: Demean Signal
  - Module: `shmtools.core`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/core/preprocessing.py

- `difference_signal_shm()` - Compute difference signal between baseline and test conditions.
  - Display Name: Difference Signal
  - Module: `shmtools`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/core/signal_filtering.py

- `envelope_shm()` - Calculate envelope signals from signal matrix.
  - Display Name: Envelope Signal
  - Module: `shmtools.core.preprocessing`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/core/preprocessing.py

- `envelope_signal_shm()` - Compute signal envelope for amplitude modulation analysis.
  - Display Name: Envelope Signal
  - Module: `shmtools`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/core/signal_filtering.py

- `filter_shm()` - Filter signals with FIR filter using FFT convolution.
  - Display Name: Filter Signal
  - Module: `shmtools.core`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/preprocessing.py

- `fir1_shm()` - Design FIR filter using window method.
  - Display Name: Fir1
  - Module: `shmtools`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/core/signal_processing.py

- `gear_mesh_filter_shm()` - Extract gear mesh frequency components and sidebands.
  - Display Name: Gear Mesh Filter
  - Module: `shmtools`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/core/signal_filtering.py

- `residual_signal_shm()` - Extract residual signal by removing shaft harmonics.
  - Display Name: Residual Signal
  - Module: `shmtools`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/core/signal_filtering.py

- `scale_min_max_shm()` - Scale data to a minimum and maximum value.
  - Display Name: Scale Min Max
  - Module: `shmtools`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/core/preprocessing.py

- `window_shm()` - Generate window vector of specified type.
  - Display Name: Window Generator
  - Module: `shmtools.core`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/core/preprocessing.py

## Feature Extraction - Spectral Analysis (8 functions)

- `cwt_analysis_shm()` - Continuous Wavelet Transform analysis.
  - Display Name: Continuous Wavelet Transform
  - Module: `shmtools`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/core/spectral.py

- `cwt_scalogram_shm()` - Compute continuous wavelet scalograms using mirrored Morlet wavelets.
  - Display Name: Continuous Wavelet Scalogram
  - Module: `shmtools.core.spectral`
  - Parameters: 8
  - File: /Users/eric/repo/shm/shmtools/core/spectral.py

- `dwvd_shm()` - Compute discrete Wigner-Ville distributions from signal matrix.
  - Display Name: Discrete Wigner-Ville Distribution
  - Module: `shmtools.core.spectral`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/core/spectral.py

- `hoelder_exp_shm()` - Calculate Hoelder exponent series from time-frequency scalogram matrix.
  - Display Name: Hoelder Exponent
  - Module: `shmtools.core.spectral`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/spectral.py

- `lpc_spectrogram_shm()` - Compute spectrogram using Linear Predictive Coding (LPC) coefficients.
  - Display Name: Linear Predictive Spectrogram
  - Module: `shmtools.core.spectral`
  - Parameters: 6
  - File: /Users/eric/repo/shm/shmtools/core/spectral.py

- `psd_welch_shm()` - Estimate power spectral density via Welch's method.
  - Display Name: Power Spectral Density via Welch's Method
  - Module: `shmtools`
  - Parameters: 6
  - File: /Users/eric/repo/shm/shmtools/core/spectral.py

- `stft_shm()` - Compute Short-Time Fourier Transform (STFT).
  - Display Name: Short-Time Fourier Transform
  - Module: `shmtools`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/core/spectral.py

- `wavelet_shm()` - Generate wavelet of specified type.
  - Display Name: Wavelet Generator
  - Module: `shmtools.core.spectral`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/spectral.py

## Feature Extraction - Statistics (12 functions)

- `clearance_factor_shm()` - Compute clearance factor (peak divided by square of mean square root).
  - Display Name: Clearance Factor
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `compute_damage_features_shm()` - Compute multiple damage-sensitive features from vibration signals.
  - Display Name: Compute Damage Features
  - Module: `shmtools`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `crest_factor_shm()` - Calculate crest factor feature matrix from raw signal matrix.
  - Display Name: Crest Factor Feature
  - Module: `shmtools.core.statistics`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `impulse_factor_shm()` - Compute impulse factor (peak divided by mean absolute value).
  - Display Name: Impulse Factor
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `m6a_shm()` - Calculate M6A feature from difference signal matrix.
  - Display Name: M6A Feature
  - Module: `shmtools.core.statistics`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `m8a_shm()` - Calculate M8A feature from difference signal matrix.
  - Display Name: M8A Feature
  - Module: `shmtools.core.statistics`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `na4m_shm()` - Calculate NA4M damage feature from residual signal matrix.
  - Display Name: NA4M Feature
  - Module: `shmtools.core.statistics`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `nb4m_shm()` - Calculate NB4M feature from band passed mesh signal matrix.
  - Display Name: NB4M Feature
  - Module: `shmtools.core.statistics`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `peak_factor_shm()` - Compute peak factor (maximum value normalized by RMS).
  - Display Name: Peak Factor
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `rms_shm()` - Calculate root mean square (RMS) feature matrix from signal matrix.
  - Display Name: Root Mean Square Feature
  - Module: `shmtools.core`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `shape_factor_shm()` - Compute shape factor (RMS divided by mean absolute value).
  - Display Name: Shape Factor
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

- `stat_moments_shm()` - Calculate first four statistical moments as damage sensitive features.
  - Display Name: Statistical Moments
  - Module: `shmtools.core.statistics`
  - Parameters: 1
  - File: /Users/eric/repo/shm/shmtools/core/statistics.py

## Feature Extraction - Time Series Models (4 functions)

- `ar_model_order_shm()` - Determine appropriate autoregressive model order (MATLAB-compatible).
  - Display Name: AR Model Order
  - Module: `shmtools`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/features/time_series.py

- `ar_model_shm()` - Estimate autoregressive model parameters and compute RMSE.
  - Display Name: AR Model
  - Module: `shmtools`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/features/time_series.py

- `arx_model_shm()` - Estimate AutoRegressive model with eXogenous inputs (ARX) parameters.
  - Display Name: ARX Model
  - Module: `shmtools.features`
  - Parameters: 2
  - File: /Users/eric/repo/shm/shmtools/features/time_series.py

- `eval_arx_model_shm()` - Evaluate ARX model using pre-trained parameters.
  - Display Name: Evaluate ARX Model
  - Module: `shmtools.features`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/features/time_series.py

## Features - Condition Based Monitoring (1 functions)

- `ars_tach_shm()` - Feature Extraction: Resamples signals to angular domain using tachometer.
  - Display Name: Ars Tach
  - Module: `shmtools.core`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/core/cbm_processing.py

## LAD (5 functions)

- `arrival_filter()` - Filter guided wave envelopes to first arrival.
  - Display Name: Arrival Filter
  - Module: `LADPackage.active_sensing.active_sensing_utils`
  - Parameters: 3
  - File: /Users/eric/repo/shm/LADPackage/active_sensing/active_sensing_utils.py

- `import_active_sense_data()` - Import active sensing dataset.
  - Display Name: Import Active Sensing Dataset
  - Module: `LADPackage.active_sensing.active_sensing_utils`
  - Parameters: 1
  - File: /Users/eric/repo/shm/LADPackage/active_sensing/active_sensing_utils.py

- `map_active_sensing_geometry()` - Map processed active sensing waveforms to geometry.
  - Display Name: Map Active Sensing Geometry
  - Module: `LADPackage.active_sensing.active_sensing_utils`
  - Parameters: 11
  - File: /Users/eric/repo/shm/LADPackage/active_sensing/active_sensing_utils.py

- `plot_as_result()` - Plot active sensing map with geometry.
  - Display Name: Plot Active Sensing Map
  - Module: `LADPackage.active_sensing.active_sensing_utils`
  - Parameters: 5
  - File: /Users/eric/repo/shm/LADPackage/active_sensing/active_sensing_utils.py

- `process_active_sensing_waveforms()` - Process active sensing waveforms with baseline subtraction and matched filtering.
  - Display Name: Process Active Sensing Waveforms
  - Module: `LADPackage.active_sensing.active_sensing_utils`
  - Parameters: 6
  - File: /Users/eric/repo/shm/LADPackage/active_sensing/active_sensing_utils.py

## Plotting - Classification (1 functions)

- `plot_roc_shm()` - Plot receiver operating characteristic curve.
  - Display Name: ROC Curve Plot
  - Module: `shmtools.plotting`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/plotting/spectral_plots.py

## Plotting - Feature Visualization (1 functions)

- `plot_features_shm()` - Plot feature vectors as subplots for each feature.
  - Display Name: Plot Features
  - Module: `shmtools.plotting`
  - Parameters: 6
  - File: /Users/eric/repo/shm/shmtools/plotting/spectral_plots.py

## Plotting - Optimal Sensor Placement (2 functions)

- `plot_nodal_response()` - Plot nodal response with element mesh for a specific mode.
  - Display Name: Plot Nodal Response
  - Module: `LADPackage.utils`
  - Parameters: 5
  - File: /Users/eric/repo/shm/LADPackage/utils/osp_plotting.py

- `plot_sensors_with_mesh()` - Plot sensor locations overlaid on structural mesh.
  - Display Name: Plot Sensors with Mesh
  - Module: `LADPackage.utils`
  - Parameters: 4
  - File: /Users/eric/repo/shm/LADPackage/utils/osp_plotting.py

## Plotting - Score Analysis (1 functions)

- `plot_score_distributions_shm()` - Plot distribution of scores using kernel density estimation (KDE).
  - Display Name: Plot Score Distributions
  - Module: `shmtools.plotting`
  - Parameters: 8
  - File: /Users/eric/repo/shm/shmtools/plotting/spectral_plots.py

## Plotting - Spectral (3 functions)

- `plot_psd_shm()` - Plot power spectral density with various visualization options.
  - Display Name: Plot Psd
  - Module: `shmtools`
  - Parameters: 8
  - File: /Users/eric/repo/shm/shmtools/plotting/spectral_plots.py

- `plot_scalogram_shm()` - Create a scalogram plot.
  - Display Name: Scalogram Plot
  - Module: `shmtools.plotting.spectral_plots`
  - Parameters: 6
  - File: /Users/eric/repo/shm/shmtools/plotting/spectral_plots.py

- `plot_time_freq_shm()` - Create a time-frequency plot.
  - Display Name: Time-Frequency Plot
  - Module: `shmtools.plotting.spectral_plots`
  - Parameters: 7
  - File: /Users/eric/repo/shm/shmtools/plotting/spectral_plots.py

## Utilities (4 functions)

- `analyze_damage_localization()` - Analyze damage localization results and provide interpretation.
  - Display Name: Analyze Damage Localization
  - Module: `shmtools.utils`
  - Parameters: 3
  - File: /Users/eric/repo/shm/shmtools/utils/spatial_analysis.py

- `compare_ar_arx_localization()` - Compare damage localization results between AR and ARX methods.
  - Display Name: Compare Ar Arx Localization
  - Module: `shmtools.utils`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/utils/spatial_analysis.py

- `plot_damage_indicators()` - Plot damage indicators for each channel in a subplot layout.
  - Display Name: Plot Damage Indicators
  - Module: `shmtools.utils`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/utils/spatial_analysis.py

- `prepare_train_test_split()` - Prepare train/test split for outlier detection with undamaged/damaged labels.
  - Display Name: Prepare Train Test Split
  - Module: `shmtools.utils`
  - Parameters: 5
  - File: /Users/eric/repo/shm/shmtools/utils/data_segmentation.py

## Utilities - Data Processing (1 functions)

- `segment_time_series()` - Segment long time series into multiple shorter segments.
  - Display Name: Segment Time Series
  - Module: `shmtools.utils`
  - Parameters: 4
  - File: /Users/eric/repo/shm/shmtools/utils/data_segmentation.py

## Utilities - Spatial Analysis (1 functions)

- `compute_channel_wise_damage_indicators()` - Compute damage indicators for each channel separately.
  - Display Name: Channel-wise Damage Indicators
  - Module: `shmtools.utils`
  - Parameters: 6
  - File: /Users/eric/repo/shm/shmtools/utils/spatial_analysis.py

## Summary

- **Total functions discovered:** 144
- **Total categories:** 34
- **Modules scanned:** 3 (shmtools, examples, LADPackage)

### Functions by Category

- **Feature Extraction - Active Sensing:** 14 functions
- **Feature Extraction - Preprocessing:** 13 functions
- **Feature Classification - Non-Parametric Detectors:** 13 functions
- **Feature Extraction - Statistics:** 12 functions
- **Core - Signal Processing:** 12 functions
- **Feature Extraction - Spectral Analysis:** 8 functions
- **Feature Classification - Parametric Detectors:** 8 functions
- **Auxiliary - Plotting:** 6 functions
- **Feature Classification - Semi-Parametric Detectors:** 5 functions
- **Data Import:** 5 functions
- **LAD:** 5 functions
- **Feature Extraction - Time Series Models:** 4 functions
- **Feature Extraction - Condition Based Monitoring:** 4 functions
- **Utilities:** 4 functions
- **Plotting - Spectral:** 3 functions
- **Auxiliary - Sensor Support:** 3 functions
- **Auxiliary - Sensor Support - Optimal Sensor Placement:** 3 functions
- **Feature Extraction - Modal Analysis:** 2 functions
- **Feature Classification - Performance Evaluation:** 2 functions
- **Feature Classification - High Level Interface:** 2 functions
- **Classification & Detection:** 2 functions
- **Plotting - Optimal Sensor Placement:** 2 functions
- **Classification - Detector Assembly:** 1 functions
- **Features - Condition Based Monitoring:** 1 functions
- **Auxiliary - Utilities:** 1 functions
- **Feature Extraction:** 1 functions
- **Data Acquisition - Signal Generation:** 1 functions
- **Plotting - Feature Visualization:** 1 functions
- **Plotting - Classification:** 1 functions
- **Plotting - Score Analysis:** 1 functions
- **Utilities - Spatial Analysis:** 1 functions
- **Utilities - Data Processing:** 1 functions
- **Data Import - LADPackage:** 1 functions
- **Classification - LADPackage Utils:** 1 functions

---
*Report generated automatically by SHM JupyterLab Extension*