"""
High-level outlier detection interfaces for structural health monitoring.

This module provides simplified interfaces for training and using outlier
detectors without needing to understand the underlying algorithms.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from scipy import stats
import pickle
import warnings

from .semiparametric import learn_gmm_semiparametric_model_shm, score_gmm_semiparametric_model_shm


def train_outlier_detector_shm(
    X_good: np.ndarray,
    k: int = 1,
    confidence: float = 0.95,
    model_filename: Optional[str] = None,
    dist_for_scores: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train outlier detector using semi-parametric Gaussian mixture model.
    
    This is the high-level interface for training outlier detectors with
    automatic threshold selection based on statistical distributions.
    
    .. meta::
        :category: Classification - High Level Interface
        :matlab_equivalent: trainOutlierDetector_shm
        :complexity: Basic
        :data_type: Features
        :output_type: Model
        :display_name: Train Outlier Detector
        :verbose_call: [Models] = Train Outlier Detector (Training Features, Number of Clusters, Confidence, Model File Name, Distribution Type)
        
    Parameters
    ----------
    X_good : array_like
        Training features from undamaged/normal conditions.
        Shape: (INSTANCES, FEATURES)
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    k : int, optional
        Number of Gaussian components to learn. Default is 1.
        This will be used to learn a confidence model (the distribution 
        of scores over the training data).
        
        .. gui::
            :widget: number_input
            :min: 1
            :max: 20
            :default: 1
            
    confidence : float, optional
        Confidence level between 0-1. Threshold is picked at this
        percentile over the training data. Default is 0.95.
        
        .. gui::
            :widget: slider
            :min: 0.5
            :max: 0.99
            :step: 0.01
            :default: 0.95
            
    model_filename : str, optional
        File name to save the model. If None, defaults to 'UndamagedModel.pkl'.
        
        .. gui::
            :widget: text_input
            :default: "UndamagedModel.pkl"
            
    dist_for_scores : str, optional
        Distribution type to model score distribution. Should be one of the
        distributions supported by scipy.stats (e.g., 'norm', 'lognorm', 'gamma').
        If None, threshold is picked directly at the confidence percentile
        without distributional assumptions.
        
        .. gui::
            :widget: dropdown
            :options: ["", "norm", "lognorm", "gamma", "exponential", "weibull_min"]
            :default: ""
            
    Returns
    -------
    models : dict
        Dictionary containing:
        - 'dModel': Density model (GMM parameters)
        - 'p_threshold': Threshold value for outlier detection
        - 'cModel': Confidence model for score distribution
        - 'scoreFun': Scoring function reference
        - 'distForScores': Distribution type used for threshold selection
        
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.classification import train_outlier_detector_shm
    >>>
    >>> # Generate training data (normal conditions)
    >>> np.random.seed(42)
    >>> X_train = np.random.randn(100, 5)
    >>>
    >>> # Train detector with default settings
    >>> models = train_outlier_detector_shm(X_train)
    >>>
    >>> # Train with 5 components and normal distribution threshold
    >>> models = train_outlier_detector_shm(X_train, k=5, confidence=0.9, 
    ...                                     dist_for_scores='norm')
    """
    X_good = np.asarray(X_good)
    
    print("\n****************** TRAIN OUTLIER DETECTOR ***************************")
    
    # Set default values
    if model_filename is None:
        model_filename = "UndamagedModel.pkl"
        
    direct_pick = dist_for_scores is None or dist_for_scores == ""
    
    print("Start learning model of undamaged conditions ----")
    
    # Learn a density model using semi-parametric GMM
    d_model = learn_gmm_semiparametric_model_shm(X_good, None, k)
    
    # Define the scoring function
    score_fun = score_gmm_semiparametric_model_shm
    
    # Pick a threshold value
    print(f"Learning threshold at the {confidence*100:.2f} percent cutoff ----")
    
    n_samples = X_good.shape[0]
    
    # Score the training data
    scores = score_fun(X_good, d_model)
    
    if direct_pick:
        # Direct percentile-based threshold selection
        scores_sorted = np.sort(scores)[::-1]  # Sort descending
        cutoff = int(np.round(confidence * n_samples))
        cutoff = min(cutoff, n_samples - 1)  # Ensure valid index
        p_threshold = scores_sorted[cutoff]
    else:
        # Distribution-based threshold selection
        try:
            # Fit the specified distribution to the scores
            dist = getattr(stats, dist_for_scores)
            params = dist.fit(scores)
            
            # Calculate threshold at (1 - confidence) percentile
            # (since lower scores indicate outliers)
            p_threshold = dist.ppf(1 - confidence, *params)
            
        except AttributeError:
            warnings.warn(f"Distribution '{dist_for_scores}' not found in scipy.stats. "
                         "Falling back to direct percentile method.")
            scores_sorted = np.sort(scores)[::-1]
            cutoff = int(np.round(confidence * n_samples))
            cutoff = min(cutoff, n_samples - 1)
            p_threshold = scores_sorted[cutoff]
            dist_for_scores = None
            
    print(f"The threshold picked is {p_threshold:.2f}")
    
    # Learn a confidence model
    print("Learning a confidence model")
    
    if direct_pick:
        # If no distribution specified, learn another GMM over the scores
        c_model = learn_gmm_semiparametric_model_shm(scores.reshape(-1, 1), None, k)
    else:
        # Store the distribution parameters
        c_model = {'distribution': dist_for_scores, 'params': params}
        
    # Save models
    print(f"Saving the models into model file {model_filename}\n")
    
    models = {
        'dModel': d_model,
        'p_threshold': p_threshold,
        'cModel': c_model,
        'scoreFun': score_fun,
        'distForScores': dist_for_scores
    }
    
    # Save to file (using pickle for Python compatibility)
    with open(model_filename, 'wb') as f:
        pickle.dump(models, f)
        
    return models


def detect_outlier_shm(
    X_test: np.ndarray,
    model_file: Optional[str] = None,
    models: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None,
    sensor_codes: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Detect outliers in test features using trained models.
    
    Universal detection interface that works with any trained outlier
    detection model from SHMTools.
    
    .. meta::
        :category: Classification - High Level Interface  
        :matlab_equivalent: detectOutlier_shm
        :complexity: Basic
        :data_type: Features
        :output_type: Classification
        :display_name: Detect Outlier
        :verbose_call: [Results, Confidences, Scores, Threshold] = Detect Outlier (Test Features, Model File Name, Models, Threshold, Sensor Codes)
        
    Parameters
    ----------
    X_test : array_like
        Test features to classify.
        Shape: (INSTANCES, FEATURES)
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    model_file : str, optional
        Name of file containing the density model. Default is 'UndamagedModel.pkl'.
        Only used if models parameter is None.
        
        .. gui::
            :widget: file_input
            :formats: [".pkl", ".mat"]
            :default: "UndamagedModel.pkl"
            
    models : dict, optional
        Density models as generated by train_outlier_detector_shm.
        If provided, model_file is ignored.
        
    threshold : float, optional
        Threshold for flagging outliers. If None, uses threshold from model.
        
        .. gui::
            :widget: number_input
            :allow_none: true
            
    sensor_codes : array_like, optional
        Vector of sensor codes (integers) associated with each example in X_test.
        Default is -1 for each example.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".npy"]
            :optional: true
            
    Returns
    -------
    results : ndarray
        Binary classification results. Shape: (INSTANCES,)
        0 = normal/undamaged, 1 = outlier/damaged
        
    confidences : ndarray  
        Confidence values between 0-1 for each instance. Shape: (INSTANCES,)
        Value c_i indicates that instance i scores less than c_i fraction
        of the training data.
        
    scores : ndarray
        Raw scores for each instance. Shape: (INSTANCES,)
        Lower scores indicate more likely outliers.
        
    threshold : float
        Threshold value used for classification.
        
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.classification import train_outlier_detector_shm, detect_outlier_shm
    >>>
    >>> # Train model
    >>> X_train = np.random.randn(100, 5)
    >>> models = train_outlier_detector_shm(X_train)
    >>>
    >>> # Test on new data
    >>> X_test = np.random.randn(50, 5)
    >>> X_test[40:, :] += 3  # Add outliers
    >>>
    >>> results, confidences, scores, threshold = detect_outlier_shm(X_test, models=models)
    >>> print(f"Detected {np.sum(results)} outliers")
    """
    X_test = np.asarray(X_test)
    
    print("\n****************** DETECT OUTLIER ***************************")
    
    # Load models if not provided
    if models is None:
        if model_file is None:
            model_file = "UndamagedModel.pkl"
            
        try:
            with open(model_file, 'rb') as f:
                models = pickle.load(f)
        except FileNotFoundError:
            print("No model found, exiting ...")
            return np.array([]), np.array([]), np.array([]), 0.0
            
    # Extract model components
    d_model = models['dModel']
    p_threshold = models['p_threshold']
    c_model = models['cModel']
    score_fun = models['scoreFun']
    dist_for_scores = models.get('distForScores', None)
    
    # Override threshold if provided
    if threshold is not None:
        p_threshold = threshold
        
    # Set default sensor codes if not provided
    if sensor_codes is None:
        sensor_codes = -np.ones(X_test.shape[0], dtype=int)
        
    # Score the test data
    score_result = score_fun(X_test, d_model)
    
    # Handle different return types from scoring functions
    if isinstance(score_result, tuple):
        # Some scoring functions return (scores, residuals) or similar
        scores = score_result[0]
    else:
        scores = score_result
    
    # Flag outliers (scores below threshold are outliers)
    results = (scores < p_threshold).astype(int)
    
    # Calculate confidence values
    if c_model is None:
        # For custom detectors without confidence model, use simple normalization
        # Map scores to [0, 1] range where lower scores = higher confidence
        score_min = np.min(scores)
        score_max = np.max(scores)
        if score_max > score_min:
            # Invert so lower scores = higher confidence
            confidences = 1 - (scores - score_min) / (score_max - score_min)
        else:
            confidences = np.ones_like(scores) * 0.5
    elif dist_for_scores is None:
        # Use empirical CDF from GMM confidence model
        # Score each test score against the confidence model
        conf_result = score_fun(scores.reshape(-1, 1), c_model)
        if isinstance(conf_result, tuple):
            conf_scores = conf_result[0]
        else:
            conf_scores = conf_result
        
        # Normalize to [0, 1] range
        conf_min = np.min(conf_scores)
        conf_max = np.max(conf_scores)
        if conf_max > conf_min:
            confidences = (conf_scores - conf_min) / (conf_max - conf_min)
        else:
            confidences = np.ones_like(conf_scores) * 0.5
    else:
        # Use parametric distribution CDF
        dist = getattr(stats, dist_for_scores)
        params = c_model['params']
        confidences = 1 - dist.cdf(scores, *params)  # Higher confidence = more likely outlier
        
    # Summary statistics
    n_outliers = np.sum(results)
    outlier_rate = n_outliers / len(results) * 100
    
    print(f"\nDetection summary:")
    print(f"  Total instances: {len(results)}")
    print(f"  Outliers detected: {n_outliers} ({outlier_rate:.1f}%)")
    print(f"  Threshold used: {p_threshold:.4f}")
    print(f"  Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    
    return results, confidences, scores, p_threshold