"""
LADPackage Mahalanobis distance learning and scoring functions.

This module provides the LADPackage LearnScoreMahalanobis function that
combines feature splitting, model training, and scoring in a single call.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Union, List

# Add project root to path to access shmtools
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from shmtools.features.time_series import split_features_shm
from shmtools.classification.outlier_detection import learn_mahalanobis_shm, score_mahalanobis_shm


def learn_score_mahalanobis(features: np.ndarray, 
                           training_indices: Union[List[int], np.ndarray]) -> np.ndarray:
    """
    LADPackage wrapper: Split data, train, and score using Mahalanobis distance.
    
    This function replicates the LADPackage LearnScoreMahalanobis.m function
    for direct compatibility with LADPackage demo scripts.
    
    .. meta::
        :category: Classification - LADPackage Utils
        :matlab_equivalent: LearnScoreMahalanobis
        :complexity: Intermediate
        :data_type: Features
        :output_type: Scores
        :display_name: Learn Score Mahalanobis
        :verbose_call: [Scores] = Learn Score Mahalanobis (Features, Training Indices)
        
    Parameters
    ----------
    features : array_like, shape (instances, features)
        Feature vectors to be split into training and scoring sets.
        
        .. gui::
            :widget: array_input
            :description: Feature matrix with instances as rows
            
    training_indices : array_like, shape (n_training,)
        List of indices of instances to be used for training. Can use either:
        - Python 0-based indexing: [0, 2, 4, ..., 90] 
        - MATLAB 1-based indexing: [1, 3, 5, ..., 91] (auto-converted)
        
        .. gui::
            :widget: array_input
            :description: Training instance indices
            :default: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91]
            
    Returns
    -------
    scores : ndarray, shape (instances, 1)
        Vector of Mahalanobis distance scores for all instances.
        
    Notes
    -----
    This is a convenience function that performs three operations in sequence:
    
    1. **Split Features**: Divides the feature matrix into training and scoring sets
       using `split_features_shm()`
    2. **Learn Mahalanobis**: Trains a Mahalanobis distance model on training features
       using `learn_mahalanobis_shm()` 
    3. **Score Mahalanobis**: Applies the model to score all features using
       `score_mahalanobis_shm()`
    
    The function scores all features (not just the test set), which is useful
    for visualization and analysis of both training and test performance.
    
    **MATLAB Compatibility**: The function automatically detects and converts
    MATLAB 1-based indexing to Python 0-based indexing. If the minimum index
    is >= 1, it assumes MATLAB indexing and subtracts 1 from all indices.
    
    **Mahalanobis Distance**: The distance is computed as:
    
    .. math::
        D_M(x) = \\sqrt{(x - \\mu)^T \\Sigma^{-1} (x - \\mu)}
    
    where:
    - $\\mu$ is the mean of training features
    - $\\Sigma$ is the covariance matrix of training features  
    - $x$ is a test feature vector
    
    Examples
    --------
    Basic usage with AR model features (Python indexing):
    
    >>> import numpy as np
    >>> from shmtools.features import ar_model_shm
    >>> from LADPackage.utils import learn_score_mahalanobis
    >>> 
    >>> # Generate AR features from time series data
    >>> data = np.random.randn(1000, 4, 170)  # 1000 time points, 4 channels, 170 instances
    >>> ar_features, _ = ar_model_shm(data, 10)  # AR order 10
    >>> 
    >>> # Use every other instance from first 91 for training (Python 0-based)
    >>> training_indices = list(range(0, 91, 2))  # [0, 2, 4, ..., 90]
    >>> 
    >>> # Learn and score
    >>> scores = learn_score_mahalanobis(ar_features, training_indices)
    >>> print(f"Scores shape: {scores.shape}")  # (170, 1)
    
    LADPackage compatibility with MATLAB indexing:
    
    >>> # MATLAB: trainingIndices_1in = 1:2:91
    >>> # Direct use (will be auto-converted to 0-based):
    >>> training_indices_matlab = list(range(1, 92, 2))  # [1, 3, 5, ..., 91]
    >>> scores = learn_score_mahalanobis(ar_features, training_indices_matlab)
    >>> # Function automatically converts to [0, 2, 4, ..., 90]
    
    Typical LADPackage workflow:
    
    >>> # Import 3-story data
    >>> from LADPackage.utils import import_3story_structure_sub_floors
    >>> dataset, damage_states, state_list = import_3story_structure_sub_floors([])
    >>> 
    >>> # Extract AR features  
    >>> ar_features, _ = ar_model_shm(dataset, 10)
    >>> 
    >>> # Train on every other undamaged instance (MATLAB: 1:2:91)
    >>> training_indices = list(range(1, 92, 2))  # MATLAB indexing
    >>> scores = learn_score_mahalanobis(ar_features, training_indices)
    >>> 
    >>> # Analyze results
    >>> undamaged_scores = scores[:90]  # First 90 instances
    >>> damaged_scores = scores[90:]    # Last 80 instances
    >>> print(f"Mean undamaged score: {np.mean(undamaged_scores):.3f}")
    >>> print(f"Mean damaged score: {np.mean(damaged_scores):.3f}")
    
    References
    ----------
    .. [1] LADPackage Documentation, Los Alamos National Laboratory
    .. [2] Mahalanobis, P.C. (1936). "On the generalised distance in statistics"
    .. [3] Worden, K., Manson, G. (2007). "The application of machine learning to 
           structural health monitoring"
    """
    # Ensure inputs are numpy arrays
    features = np.asarray(features)
    training_indices = np.asarray(training_indices)
    
    # Check for MATLAB 1-based indexing and convert to 0-based if needed
    if training_indices.min() >= 1:
        # Assume MATLAB indexing, convert to Python 0-based
        training_indices = training_indices - 1
        
    # Validate training indices
    max_instances = features.shape[0]
    if training_indices.max() >= max_instances:
        raise ValueError(f"Training indices contain values >= {max_instances}. "
                        f"Max valid index for {max_instances} instances is {max_instances-1}")
    if training_indices.min() < 0:
        raise ValueError("Training indices cannot be negative after conversion to 0-based indexing")
    
    # Step 1: Split features into training and scoring sets
    training_features, scoring_features, all_features = split_features_shm(
        features, 
        training_indices=training_indices, 
        scoring_indices=None,  # Use all non-training indices for scoring
        features_to_use=None   # Use all features
    )
    
    # Step 2: Learn Mahalanobis distance model on training data
    model = learn_mahalanobis_shm(training_features)
    
    # Step 3: Score all features (both training and test) 
    # This matches LADPackage behavior of scoring the full feature set
    scores = score_mahalanobis_shm(all_features, model)
    
    # Ensure output is column vector to match MATLAB format
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    
    return scores