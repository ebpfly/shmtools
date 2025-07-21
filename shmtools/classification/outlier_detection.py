"""
Outlier detection algorithms for structural health monitoring.

This module provides various parametric and non-parametric methods
for detecting anomalies in structural response data.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance


def learn_mahalanobis(X: np.ndarray) -> Dict[str, Any]:
    """
    Learn Mahalanobis distance model from training data.
    
    Python equivalent of MATLAB's learnMahalanobis_shm function.
    
    Parameters
    ----------
    X : np.ndarray
        Training data matrix (instances x features).
        
    Returns
    -------
    model : dict
        Dictionary containing:
        - 'mean': Sample mean vector
        - 'cov': Sample covariance matrix
        - 'cov_inv': Inverse covariance matrix
    """
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    
    # Add regularization for numerical stability
    reg_param = 1e-6
    cov_reg = cov + reg_param * np.eye(cov.shape[0])
    cov_inv = linalg.inv(cov_reg)
    
    return {
        'mean': mean,
        'cov': cov_reg,
        'cov_inv': cov_inv
    }


def score_mahalanobis(X: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    """
    Compute Mahalanobis distance scores.
    
    Python equivalent of MATLAB's scoreMahalanobis_shm function.
    
    Parameters
    ----------
    X : np.ndarray
        Test data matrix (instances x features).
    model : dict
        Mahalanobis model from learn_mahalanobis.
        
    Returns
    -------
    scores : np.ndarray
        Mahalanobis distance scores.
    """
    X_centered = X - model['mean']
    scores = np.sum(X_centered @ model['cov_inv'] * X_centered, axis=1)
    return scores


def learn_mahalanobis_shm(X: np.ndarray) -> Dict[str, Any]:
    """
    Learn Mahalanobis distance model from training data.
    
    .. meta::
        :category: Classification - Parametric Detectors
        :matlab_equivalent: learnMahalanobis_shm
        :complexity: Basic
        :data_type: Features
        :output_type: Model
        
    Parameters
    ----------
    X : array_like
        Training features where each row (INSTANCES) is a feature vector.
        Shape: (INSTANCES, FEATURES)
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    Returns
    -------
    model : dict
        Parameters of the model with fields:
        - 'dataMean': Mean vector of the features (1, FEATURES)
        - 'dataCov': Covariance matrix of the features (FEATURES, FEATURES)
        
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.classification import learn_mahalanobis_shm
    >>> 
    >>> # Generate sample training data
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 5)
    >>> 
    >>> # Learn Mahalanobis model
    >>> model = learn_mahalanobis_shm(X)
    >>> print(f"Mean shape: {model['dataMean'].shape}")
    >>> print(f"Covariance shape: {model['dataCov'].shape}")
    """
    X = np.asarray(X)
    
    # Calculate the mean vector of X (following MATLAB exactly)
    data_mean = np.mean(X, axis=0)  # Shape: (FEATURES,) -> reshape to (1, FEATURES)
    data_mean = data_mean.reshape(1, -1)
    
    # Calculate the covariance of X (using ddof=1 to match MATLAB's cov function)
    data_cov = np.cov(X, rowvar=False, ddof=1)
    
    # Add small regularization for numerical stability (lesson learned from Phase 1)
    data_cov = data_cov + 1e-10 * np.eye(data_cov.shape[0])
    
    # Store as a structure array (following MATLAB exactly)
    model = {
        'dataMean': data_mean,
        'dataCov': data_cov
    }
    
    return model


def score_mahalanobis_shm(Y: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    """
    Score Mahalanobis distance for outlier detection.
    
    .. meta::
        :category: Classification - Parametric Detectors
        :matlab_equivalent: scoreMahalanobis_shm
        :complexity: Basic
        :data_type: Features
        :output_type: Scores
        
    Parameters
    ----------
    Y : array_like
        Test features where each row (INSTANCES) is a feature vector.
        Shape: (INSTANCES, FEATURES)
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    model : dict
        Parameters of the model with fields:
        - 'dataMean': Mean vector of the features (1, FEATURES)
        - 'dataCov': Covariance matrix of the features (FEATURES, FEATURES)
        
    Returns
    -------
    scores : ndarray
        Vector of scores (negative Mahalanobis squared distance).
        Shape: (INSTANCES, 1)
        
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.classification import learn_mahalanobis_shm, score_mahalanobis_shm
    >>> 
    >>> # Generate training and test data
    >>> np.random.seed(42)
    >>> X_train = np.random.randn(100, 5)
    >>> X_test = np.random.randn(20, 5)
    >>> 
    >>> # Learn and score
    >>> model = learn_mahalanobis_shm(X_train)
    >>> scores = score_mahalanobis_shm(X_test, model)
    >>> print(f"Scores shape: {scores.shape}")  # (20, 1)
    """
    # Check parameters (following MATLAB exactly)
    if len([Y, model]) < 2:
        raise ValueError('At least two input arguments required.')
    
    Y = np.asarray(Y)
    
    # Set parameters (following MATLAB exactly)
    n = Y.shape[0]
    data_mean = model['dataMean']  # Mean vector stored in model structure
    data_cov = model['dataCov']    # Covariance stored in model structure
    
    scores = np.zeros((n, 1))
    
    # Mahalanobis squared distance (following MATLAB loop exactly)
    for i in range(n):
        # MATLAB: scores(i,1)=(Y(i,:)-dataMean)/dataCov*(Y(i,:)-dataMean)';
        diff = Y[i, :] - data_mean.flatten()  # Ensure broadcast works correctly
        scores[i, 0] = diff @ np.linalg.solve(data_cov, diff)
    
    # Make negative as in MATLAB
    scores = -scores
    
    return scores


def learn_svd_shm(X: np.ndarray, param_stand: bool = True) -> Dict[str, Any]:
    """
    Learn SVD-based outlier detection model from training features.
    
    Computes the singular values of the training data matrix X. In the context of
    damage detection, if the rank of M is r and if a vector from Y comes from a
    damaged condition, the rank of Mc will be equal to r+1. If the rank of M and
    Mc are the same, this algorithm assumes that when Mc is composed by a damaged
    condition, the damage detection is based on changes in magnitude of the
    singular values.
    
    .. meta::
        :category: Classification - Parametric Detectors
        :matlab_equivalent: learnSVD_shm
        :complexity: Basic
        :data_type: Features
        :output_type: Model
        
    Parameters
    ----------
    X : array_like
        Training features where each row is a feature vector.
        Shape: (INSTANCES, FEATURES)
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    param_stand : bool, optional
        Whether standardization should be used (default: True)
        
        .. gui::
            :widget: checkbox
            :default: True
            
    Returns
    -------
    model : dict
        Parameters of the model with fields:
        - 'X': Training data matrix (INSTANCES, FEATURES)
        - 'S': Singular values of X (FEATURES,)
        - 'dataMean': Mean vector of the features (FEATURES,) or None
        - 'dataStd': Standard deviation vector of the features (FEATURES,) or None
        
    References
    ----------
    Ruotolo, R., & Surage, C. (1999). Using SVD to Detect Damage in 
    Structures with Different Operational Conditions. Journal of Sound and
    Vibration, 226(3), 425-439.
    """
    X = np.asarray(X, dtype=np.float64)
    
    if X.ndim != 2:
        raise ValueError("Input X must be 2-dimensional")
        
    n, m = X.shape
    
    if param_stand not in [0, 1, False, True]:
        raise ValueError("Input parameter param_stand should be either 0 or 1")
    
    if param_stand:
        # Standardize features (mean and std vectors stored for future standardization)
        data_mean = np.mean(X, axis=0)
        data_std = np.std(X, axis=0, ddof=0)  # Use ddof=0 to match MATLAB
        
        # Handle zero standard deviation (defensive programming)
        data_std = np.where(data_std == 0, 1.0, data_std)
        
        X_normalized = (X - data_mean) / data_std
    else:
        data_mean = None
        data_std = None
        X_normalized = X.copy()
    
    # Compute singular values of matrix X
    S = np.linalg.svd(X_normalized, compute_uv=False)
    
    # Store as dictionary (matching MATLAB struct)
    model = {
        'X': X_normalized,
        'S': S,
        'dataMean': data_mean,
        'dataStd': data_std
    }
    
    return model


def score_svd_shm(Y: np.ndarray, model: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score features using trained SVD outlier detection model.
    
    Returns scores based on the Euclidean norm of the residuals between the 
    singular vectors S and Sc. S and Sc are estimated from M and Mc, respectively.
    Each state matrix (Mc) is composed by X and one feature vector of Y at a time.
    In the context of damage detection, if the rank of M is r and if a vector from
    Y comes from a damaged condition, the rank of Mc will be equal to r+1.
    
    .. meta::
        :category: Classification - Parametric Detectors
        :matlab_equivalent: scoreSVD_shm
        :complexity: Basic
        :data_type: Features
        :output_type: Scores
        
    Parameters
    ----------
    Y : array_like
        Test features where each row is a feature vector.
        Shape: (INSTANCES, FEATURES)
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    model : dict
        Parameters of the model with fields:
        - 'X': Training data matrix (INSTANCES, FEATURES)
        - 'S': Singular values of X (FEATURES,)
        - 'dataMean': Mean vector or None
        - 'dataStd': Standard deviation vector or None
        
    Returns
    -------
    scores : ndarray
        Vector of damage indicator scores.
        Shape: (INSTANCES,)
        
    residuals : ndarray
        Residuals between singular values.
        Shape: (INSTANCES, FEATURES)
    """
    Y = np.asarray(Y, dtype=np.float64)
    
    if Y.ndim != 2:
        raise ValueError("Input Y must be 2-dimensional")
    
    # Extract model parameters
    M = model['X']
    S = model['S']
    data_mean = model['dataMean']
    data_std = model['dataStd']
    
    n, m = Y.shape
    nt = M.shape[0]
    
    # Standardize variables with mean and standard deviation from training data
    if data_mean is not None:
        Y_normalized = (Y - data_mean) / data_std
    else:
        Y_normalized = Y.copy()
    
    # Initialize singular value matrix
    if m >= nt:
        S_extended = np.zeros(nt + 1)
        S_extended[:len(S)] = S
        Sc = np.zeros((nt + 1, n))
    else:
        S_extended = S
        Sc = np.zeros((m, n))
    
    # Estimate singular values from matrix Mc for each test instance
    for i in range(n):
        # Create augmented matrix Mc = [M; Y[i,:]]
        Mc = np.vstack([M, Y_normalized[i:i+1, :]])
        
        # Compute singular values of Mc
        Sc[:, i] = np.linalg.svd(Mc, compute_uv=False)
    
    # Calculate residuals
    residuals = Sc - S_extended[:, np.newaxis]
    residuals = residuals.T  # Transpose to match MATLAB output shape
    
    # Scores based on Euclidean distance
    scores = np.sqrt(np.sum(residuals**2, axis=1))
    
    # Make negative as in MATLAB
    scores = -scores
    
    return scores, residuals


def mahalanobis_distance(X: np.ndarray, X_train: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Modern Python interface for Mahalanobis distance computation.
    
    Parameters
    ----------
    X : np.ndarray
        Test data matrix.
    X_train : np.ndarray, optional
        Training data. If None, uses X for both training and testing.
        
    Returns
    -------
    distances : np.ndarray
        Mahalanobis distances.
    """
    if X_train is None:
        X_train = X
        
    model = learn_mahalanobis_shm(X_train)
    scores = score_mahalanobis_shm(X, model)
    return -scores.flatten()  # Return positive distances for modern interface


def learn_pca_shm(X: np.ndarray, per_var: float = 0.90, stand: int = 0) -> Dict[str, Any]:
    """
    Learn principal component analysis (PCA) for outlier detection.
    
    .. meta::
        :category: Classification - Parametric Detectors
        :matlab_equivalent: learnPCA_shm  
        :complexity: Basic
        :data_type: Features
        :output_type: Model
        
    Parameters
    ----------
    X : array_like
        Training features where each row (INSTANCES) is a m-dimensional feature vector.
        Shape: (INSTANCES, FEATURES)
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    per_var : float, optional
        Minimal percentage of the variance to explain the variability in the matrix X.
        Default is 0.90.
        
        .. gui::
            :widget: number_input
            :min: 0.01
            :max: 1.0
            :default: 0.90
            :step: 0.01
            
    stand : int, optional
        Standardization flag: 0 - standardization, 1 - non standardization.
        Default is 0.
        
        .. gui::
            :widget: select
            :options: [0, 1]
            :default: 0
            
    Returns
    -------
    model : dict
        Parameters of the model with fields:
        - 'loadings': Loadings matrix where columns are the number of principal components
        - 'data_param': Matrix composed of (1) means and (2) standard deviations
        
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.classification import learn_pca_shm
    >>> 
    >>> # Generate sample training data
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 5)
    >>> 
    >>> # Learn PCA model
    >>> model = learn_pca_shm(X, per_var=0.95)
    >>> print(f"Loadings shape: {model['loadings'].shape}")
    """
    X = np.asarray(X)
    n, m = X.shape
    
    if per_var < 0 or per_var > 1:
        raise ValueError("per_var must be between 0 and 1")
    
    # Standardize to correlation matrix (following MATLAB exactly)
    if stand == 0:
        data_mean = np.mean(X, axis=0)  # Mean vector of the variables
        data_std = np.std(X, axis=0, ddof=1)  # Standard deviation vector (using ddof=1 to match MATLAB)
        
        # Handle zero standard deviation (constant features)
        data_std = np.where(data_std == 0, 1.0, data_std)
        
        data_param = np.vstack([data_mean, data_std])
        
        # Standardize data
        X = (X - data_mean) / data_std
    elif stand == 1:
        data_param = None
    else:
        raise ValueError("Input parameter stand must be either 0 or 1.")
    
    # Covariance/Correlation matrix (using ddof=1 to match MATLAB's cov function)
    R = np.cov(X, rowvar=False, ddof=1)
    
    # Add small regularization for numerical stability
    R = R + 1e-10 * np.eye(R.shape[0])
    
    # Decomposition of R (following MATLAB exactly)
    U, S, V = np.linalg.svd(R)
    
    loadings = V.T  # MATLAB's svd returns V, we need V'
    latents = S
    total_var = np.sum(latents)
    
    # Determine number of principal components
    if per_var == 1.0:
        num_pc = m
    else:
        cumulative_var = np.cumsum(latents) / total_var
        num_pc = np.argmax(cumulative_var >= per_var) + 1
    
    loadings_final = loadings[:, :num_pc]
    
    # Store as a structure array (following MATLAB exactly)
    model = {
        'loadings': loadings_final,
        'data_param': data_param
    }
    
    return model


def score_pca_shm(Y: np.ndarray, model: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Score principal component analysis (PCA) for outlier detection.
    
    .. meta::
        :category: Classification - Parametric Detectors
        :matlab_equivalent: scorePCA_shm
        :complexity: Basic  
        :data_type: Features
        :output_type: Scores
        
    Parameters
    ----------
    Y : array_like
        Test features where each row (INSTANCES) is a feature vector.
        Shape: (INSTANCES, FEATURES)
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    model : dict
        Parameters of the model with fields:
        - 'loadings': Loadings matrix where columns are the number of principal components
        - 'data_param': (1) mean and (2) standard deviation vectors
        
    Returns
    -------
    scores : ndarray
        Vector of scores (negative Euclidean norm of residuals).
    residuals : ndarray
        Residual errors, shape (INSTANCES, FEATURES).
        
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.classification import learn_pca_shm, score_pca_shm
    >>> 
    >>> # Generate training and test data
    >>> np.random.seed(42)
    >>> X_train = np.random.randn(100, 5)
    >>> X_test = np.random.randn(20, 5)
    >>> 
    >>> # Learn and score
    >>> model = learn_pca_shm(X_train)
    >>> scores, residuals = score_pca_shm(X_test, model)
    >>> print(f"Scores shape: {scores.shape}")  # (20,)
    """
    Y = np.asarray(Y)
    n = Y.shape[0]
    
    loadings = model['loadings']
    data_param = model['data_param']
    
    # Standardize variables with mean and standard deviation from the training data
    if data_param is not None:
        data_mean = data_param[0, :]
        data_std = data_param[1, :]
        
        Y = (Y - data_mean) / data_std
    
    # PCA (following MATLAB algorithm exactly)
    scores_pc = Y @ loadings  # Project to principal component space
    Y_prime = scores_pc @ loadings.T  # Reconstruct from principal components
    residuals = Y - Y_prime  # Calculate residuals
    
    # Euclidean norm of the unique factors (following MATLAB exactly)
    scores = np.sqrt(np.sum(residuals**2, axis=1))
    scores = -scores  # Make negative as in MATLAB
    
    return scores, residuals


def learn_pca(X: np.ndarray, n_components: Optional[int] = None) -> Dict[str, Any]:
    """
    Modern Python interface for PCA model learning.
    
    Parameters
    ----------
    X : np.ndarray
        Training data matrix (instances x features).
    n_components : int, optional
        Number of principal components. If None, uses all components.
        
    Returns
    -------
    model : dict
        Dictionary containing PCA model parameters.
    """
    if n_components is None:
        per_var = 1.0
    else:
        # Convert n_components to percentage of variance
        per_var = 0.95  # Default fallback
    
    return learn_pca_shm(X, per_var=per_var, stand=0)


def score_pca(X: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    """
    Modern Python interface for PCA scoring.
    
    Parameters
    ----------
    X : np.ndarray
        Test data matrix.
    model : dict
        PCA model from learn_pca.
        
    Returns
    -------
    scores : np.ndarray
        PCA reconstruction error scores.
    """
    scores, _ = score_pca_shm(X, model)
    return -scores  # Return positive scores for modern interface


def pca_detector(X: np.ndarray, X_train: Optional[np.ndarray] = None, 
                n_components: Optional[int] = None) -> np.ndarray:
    """
    PCA-based outlier detection.
    
    Parameters
    ----------
    X : np.ndarray
        Test data matrix.
    X_train : np.ndarray, optional
        Training data. If None, uses X.
    n_components : int, optional
        Number of principal components.
        
    Returns
    -------
    scores : np.ndarray
        Outlier scores (reconstruction error).
    """
    if X_train is None:
        X_train = X
        
    model = learn_pca(X_train, n_components)
    return score_pca(X, model)


def roc_shm(scores: np.ndarray, damage_states: np.ndarray, 
           num_pts: Optional[int] = None, threshold_type: str = 'below') -> Tuple[np.ndarray, np.ndarray]:
    """
    Receiver operating characteristic (ROC) curve.
    
    Tool to compare and evaluate the performance of classification algorithms.
    Note that the scores should decrease for the damaged instances.
    
    .. meta::
        :category: Classification - Performance Evaluation
        :matlab_equivalent: ROC_shm
        :complexity: Basic
        :data_type: Scores
        :output_type: Performance Metrics
        
    Parameters
    ----------
    scores : array_like
        Vector composed of scores for each instance.
        Shape: (INSTANCES,)
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    damage_states : array_like
        Binary classification vector of known damage states 
        (0-undamaged and 1-damaged) corresponding to vector of scores.
        Shape: (INSTANCES,)
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    num_pts : int, optional
        Number of points to evaluate ROC curve at. If None (recommended),
        each score value from a damaged state is used as a threshold.
        
        .. gui::
            :widget: spinner
            :min: 10
            :max: 1000
            :default: 100
            
    threshold_type : str, optional
        'above' or 'below' to define if scores above or below a given
        threshold should be flagged as damaged (default: 'below')
        
        .. gui::
            :widget: dropdown
            :options: ["below", "above"]
            :default: "below"
            
    Returns
    -------
    TPR : ndarray
        Vector composed of true positive rates.
        Shape: (POINTS,)
        
    FPR : ndarray
        Vector composed of false positive rates.
        Shape: (POINTS,)
        
    References
    ----------
    MATLAB ROC_shm function from SHMTools.
    """
    scores = np.asarray(scores, dtype=np.float64).flatten()
    damage_states = np.asarray(damage_states, dtype=int).flatten()
    
    if len(scores) != len(damage_states):
        raise ValueError("scores and damage_states must have the same length")
    
    if not np.all(np.isin(damage_states, [0, 1])):
        raise ValueError("damage_states must contain only 0 (undamaged) and 1 (damaged)")
    
    # Get thresholds
    if num_pts is None:
        # Use each damaged state score as threshold (default, recommended)
        thresholds = scores[damage_states == 1]
        thresholds = np.unique(thresholds)
    else:
        # Use linearly spaced thresholds
        thresholds = np.linspace(np.min(scores), np.max(scores), num_pts)
    
    # Sort thresholds
    thresholds = np.sort(thresholds)
    
    TPR = np.zeros(len(thresholds))
    FPR = np.zeros(len(thresholds))
    
    # Count actual positives and negatives
    P = np.sum(damage_states == 1)  # Total damaged instances
    N = np.sum(damage_states == 0)  # Total undamaged instances
    
    if P == 0 or N == 0:
        raise ValueError("Need both damaged and undamaged instances for ROC curve")
    
    for i, threshold in enumerate(thresholds):
        if threshold_type == 'below':
            # Scores below threshold are classified as damaged
            predicted_damaged = scores <= threshold
        else:  # 'above'
            # Scores above threshold are classified as damaged
            predicted_damaged = scores >= threshold
        
        # True positives: correctly identified damaged instances
        TP = np.sum((damage_states == 1) & predicted_damaged)
        
        # False positives: undamaged instances incorrectly classified as damaged
        FP = np.sum((damage_states == 0) & predicted_damaged)
        
        # Calculate rates
        TPR[i] = TP / P  # True Positive Rate (Sensitivity)
        FPR[i] = FP / N  # False Positive Rate (1 - Specificity)
    
    return TPR, FPR