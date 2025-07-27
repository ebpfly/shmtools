"""
Nonparametric outlier detection functions.

This module provides kernel density estimation functions for nonparametric
outlier detection using various kernel functions.
"""

import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple
from scipy import stats
import warnings


def gaussian_kernel_shm(delta: np.ndarray) -> np.ndarray:
    """
    Kernel weights for the Gaussian kernel.

    .. meta::
        :category: Classification - Nonparametric Detectors
        :matlab_equivalent: gaussianKernel_shm
        :complexity: Basic
        :data_type: Features
        :output_type: Weights
        :display_name: Gaussian Kernel Weights
        :verbose_call: Weights = Gaussian Kernel Weights (Evaluation Points)

    Parameters
    ----------
    delta : array_like
        Evaluation points of shape (INSTANCES, FEATURES). Each row is
        (x-x_i)/bandwidth where the kernel is centered at x, and x_i
        are the training points.

        .. gui::
            :widget: array_input
            :description: Normalized evaluation points

    Returns
    -------
    W : ndarray
        Evaluated kernel weights of shape (INSTANCES,).

    Notes
    -----
    In one dimension, this is given as K(u) = (1/sqrt(2*pi))*exp(-u^2/2)
    where u = (x - x_i)/h. In higher dimensions we take the product of
    the one dimensional kernel at each coordinate.

    References
    ----------
    http://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    delta = np.asarray(delta)
    n, D = delta.shape

    # Gaussian kernel: (1/(2*pi)^(D/2)) * exp(-0.5 * sum(delta^2))
    W = (1.0 / (2.0 * np.pi) ** (D / 2)) * np.exp(-0.5 * np.sum(delta * delta, axis=1))

    return W


def epanechnikov_kernel_shm(delta: np.ndarray) -> np.ndarray:
    """
    Kernel weights for the Epanechnikov kernel.

    .. meta::
        :category: Classification - Nonparametric Detectors
        :matlab_equivalent: epanechnikovKernel_shm
        :complexity: Basic
        :data_type: Features
        :output_type: Weights
        :display_name: Epanechnikov Kernel Weights
        :verbose_call: Weights = Epanechnikov Kernel Weights (Evaluation Points)

    Parameters
    ----------
    delta : array_like
        Evaluation points of shape (INSTANCES, FEATURES). Each row is
        (x-x_i)/bandwidth where the kernel is centered at x, and x_i
        are the training points.

        .. gui::
            :widget: array_input
            :description: Normalized evaluation points

    Returns
    -------
    W : ndarray
        Evaluated kernel weights of shape (INSTANCES,).

    Notes
    -----
    In one dimension, this is given as K(u) = 3/4 * (1-u^2)_+ where
    u = (x - x_i)/h and (·)_+ denotes the positive part. In higher
    dimensions we take the product of the one dimensional kernel at
    each coordinate.

    References
    ----------
    http://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    delta = np.asarray(delta)
    n, D = delta.shape

    # Epanechnikov kernel: (3/4)^D * prod(1-delta^2) * indicator(|delta| < 1)
    # Check if all coordinates are within unit hypercube
    # MATLAB: min(abs(Delta) < box,[], 2) - all coordinates must be < 1
    within_box = np.all(np.abs(delta) < 1, axis=1)

    # Compute kernel weights - only for points within box
    W = np.zeros(n)
    if np.any(within_box):
        valid_delta = delta[within_box]
        W[within_box] = ((3.0 / 4.0) ** D) * np.prod(1.0 - valid_delta**2, axis=1)

    return W


def uniform_kernel_shm(delta: np.ndarray) -> np.ndarray:
    """
    Kernel weights for the Uniform kernel.

    .. meta::
        :category: Classification - Nonparametric Detectors
        :matlab_equivalent: uniformKernel_shm
        :complexity: Basic
        :data_type: Features
        :output_type: Weights
        :display_name: Uniform Kernel Weights
        :verbose_call: Weights = Uniform Kernel Weights (Evaluation Points)

    Parameters
    ----------
    delta : array_like
        Evaluation points of shape (INSTANCES, FEATURES). Each row is
        (x-x_i)/bandwidth where the kernel is centered at x, and x_i
        are the training points.

        .. gui::
            :widget: array_input
            :description: Normalized evaluation points

    Returns
    -------
    W : ndarray
        Evaluated kernel weights of shape (INSTANCES,).

    Notes
    -----
    In one dimension, this is given as K(u) = 1/2 * I(|u| < 1) where
    u = (x - x_i)/h and I(·) is the indicator function. In higher
    dimensions we take the product of the one dimensional kernel at
    each coordinate.

    References
    ----------
    http://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    delta = np.asarray(delta)
    n, D = delta.shape

    # Uniform kernel: (1/2)^D * indicator(|delta| < 1 for all coordinates)
    within_box = np.all(np.abs(delta) < 1, axis=1)

    W = np.zeros(n)
    W[within_box] = (0.5) ** D

    return W


def quartic_kernel_shm(delta: np.ndarray) -> np.ndarray:
    """
    Kernel weights for the Quartic (Biweight) kernel.

    .. meta::
        :category: Classification - Nonparametric Detectors
        :matlab_equivalent: quarticKernel_shm
        :complexity: Basic
        :data_type: Features
        :output_type: Weights
        :display_name: Quartic Kernel Weights
        :verbose_call: Weights = Quartic Kernel Weights (Evaluation Points)

    Parameters
    ----------
    delta : array_like
        Evaluation points of shape (INSTANCES, FEATURES). Each row is
        (x-x_i)/bandwidth where the kernel is centered at x, and x_i
        are the training points.

        .. gui::
            :widget: array_input
            :description: Normalized evaluation points

    Returns
    -------
    W : ndarray
        Evaluated kernel weights of shape (INSTANCES,).

    Notes
    -----
    In one dimension, this is given as K(u) = 15/16 * (1-u^2)^2 * I(|u| < 1)
    where u = (x - x_i)/h and I(·) is the indicator function. In higher
    dimensions we take the product of the one dimensional kernel at
    each coordinate.

    References
    ----------
    http://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    delta = np.asarray(delta)
    n, D = delta.shape

    # Quartic kernel: (15/16)^D * prod((1-delta^2)^2) * indicator(|delta| < 1)
    within_box = np.all(np.abs(delta) < 1, axis=1)

    W = np.zeros(n)
    if np.any(within_box):
        valid_delta = delta[within_box]
        W[within_box] = ((15.0 / 16.0) ** D) * np.prod(
            (1.0 - valid_delta**2) ** 2, axis=1
        )

    return W


def triangle_kernel_shm(delta: np.ndarray) -> np.ndarray:
    """
    Kernel weights for the Triangle kernel.

    .. meta::
        :category: Classification - Nonparametric Detectors
        :matlab_equivalent: triangleKernel_shm
        :complexity: Basic
        :data_type: Features
        :output_type: Weights
        :display_name: Triangle Kernel Weights
        :verbose_call: Weights = Triangle Kernel Weights (Evaluation Points)

    Parameters
    ----------
    delta : array_like
        Evaluation points of shape (INSTANCES, FEATURES). Each row is
        (x-x_i)/bandwidth where the kernel is centered at x, and x_i
        are the training points.

        .. gui::
            :widget: array_input
            :description: Normalized evaluation points

    Returns
    -------
    W : ndarray
        Evaluated kernel weights of shape (INSTANCES,).

    Notes
    -----
    In one dimension, this is given as K(u) = (1-|u|) * I(|u| < 1)
    where u = (x - x_i)/h and I(·) is the indicator function. In higher
    dimensions we take the product of the one dimensional kernel at
    each coordinate.

    References
    ----------
    http://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    delta = np.asarray(delta)
    n, D = delta.shape

    # Triangle kernel: prod(1-|delta|) * indicator(|delta| < 1)
    within_box = np.all(np.abs(delta) < 1, axis=1)

    W = np.zeros(n)
    if np.any(within_box):
        valid_delta = delta[within_box]
        W[within_box] = np.prod(1.0 - np.abs(valid_delta), axis=1)

    return W


def triweight_kernel_shm(delta: np.ndarray) -> np.ndarray:
    """
    Kernel weights for the Triweight kernel.

    .. meta::
        :category: Classification - Nonparametric Detectors
        :matlab_equivalent: triweightKernel_shm
        :complexity: Basic
        :data_type: Features
        :output_type: Weights
        :display_name: Triweight Kernel Weights
        :verbose_call: Weights = Triweight Kernel Weights (Evaluation Points)

    Parameters
    ----------
    delta : array_like
        Evaluation points of shape (INSTANCES, FEATURES). Each row is
        (x-x_i)/bandwidth where the kernel is centered at x, and x_i
        are the training points.

        .. gui::
            :widget: array_input
            :description: Normalized evaluation points

    Returns
    -------
    W : ndarray
        Evaluated kernel weights of shape (INSTANCES,).

    Notes
    -----
    In one dimension, this is given as K(u) = 35/32 * (1-u^2)^3 * I(|u| < 1)
    where u = (x - x_i)/h and I(·) is the indicator function. In higher
    dimensions we take the product of the one dimensional kernel at
    each coordinate.

    References
    ----------
    http://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    delta = np.asarray(delta)
    n, D = delta.shape

    # Triweight kernel: (35/32)^D * prod((1-delta^2)^3) * indicator(|delta| < 1)
    within_box = np.all(np.abs(delta) < 1, axis=1)

    W = np.zeros(n)
    if np.any(within_box):
        valid_delta = delta[within_box]
        W[within_box] = ((35.0 / 32.0) ** D) * np.prod(
            (1.0 - valid_delta**2) ** 3, axis=1
        )

    return W


def cosine_kernel_shm(delta: np.ndarray) -> np.ndarray:
    """
    Kernel weights for the Cosine kernel.

    .. meta::
        :category: Classification - Nonparametric Detectors
        :matlab_equivalent: cosineKernel_shm
        :complexity: Basic
        :data_type: Features
        :output_type: Weights
        :display_name: Cosine Kernel Weights
        :verbose_call: Weights = Cosine Kernel Weights (Evaluation Points)

    Parameters
    ----------
    delta : array_like
        Evaluation points of shape (INSTANCES, FEATURES). Each row is
        (x-x_i)/bandwidth where the kernel is centered at x, and x_i
        are the training points.

        .. gui::
            :widget: array_input
            :description: Normalized evaluation points

    Returns
    -------
    W : ndarray
        Evaluated kernel weights of shape (INSTANCES,).

    Notes
    -----
    In one dimension, this is given as K(u) = (π/4) * cos(π*u/2) * I(|u| < 1)
    where u = (x - x_i)/h and I(·) is the indicator function. In higher
    dimensions we take the product of the one dimensional kernel at
    each coordinate.

    References
    ----------
    http://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    delta = np.asarray(delta)
    n, D = delta.shape

    # Cosine kernel: (π/4)^D * prod(cos(π*delta/2)) * indicator(|delta| < 1)
    within_box = np.all(np.abs(delta) < 1, axis=1)

    W = np.zeros(n)
    if np.any(within_box):
        valid_delta = delta[within_box]
        W[within_box] = ((np.pi / 4.0) ** D) * np.prod(
            np.cos(np.pi * valid_delta / 2.0), axis=1
        )

    return W


def learn_kernel_density_shm(
    X: np.ndarray,
    H: Optional[np.ndarray] = None,
    kernel_fun: Optional[Callable] = None,
    bs_method: int = 2,
) -> Dict[str, Any]:
    """
    Learn nonparametric kernel density estimation model.

    .. meta::
        :category: Classification - Nonparametric Detectors
        :matlab_equivalent: learnKernelDensity_shm
        :complexity: Advanced
        :data_type: Features
        :output_type: Model
        :display_name: Learn Kernel Density Estimation
        :verbose_call: Model = Learn Kernel Density Estimation (Training Features, Bandwidth Matrix, Kernel Function, Bandwidth Selection Method)

    Parameters
    ----------
    X : array_like
        Training features of shape (INSTANCES, FEATURES) where each row
        is a feature vector.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Training feature matrix

    H : array_like, optional
        Bandwidth matrix of shape (FEATURES, FEATURES), restricted to
        diagonal bandwidth matrices. If None, bandwidth selection is used.

        .. gui::
            :widget: array_input
            :description: Diagonal bandwidth matrix (optional)

    kernel_fun : callable, optional
        Kernel function to use. Default is gaussian_kernel_shm.

        .. gui::
            :widget: select
            :options: ["gaussian_kernel_shm", "epanechnikov_kernel_shm"]
            :default: "gaussian_kernel_shm"

    bs_method : int, optional
        Bandwidth selection method: 1 or 2 (default: 2).
        Method 1 performs cross validation and sets bandwidth as b*std
        in each coordinate. Method 2 sets b to (1/n)^(1/3) to match
        the minimax l2 rate.

        .. gui::
            :widget: select
            :options: [1, 2]
            :default: 2
            :description: Bandwidth selection method

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'Xtrain': Training data
        - 'kernel': Kernel function
        - 'H': Bandwidth matrix

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.classification import learn_kernel_density_shm
    >>>
    >>> # Generate sample training data
    >>> np.random.seed(42)
    >>> X_train = np.random.randn(100, 3)
    >>>
    >>> # Learn kernel density model with automatic bandwidth selection
    >>> model = learn_kernel_density_shm(X_train)
    >>> print("Model trained successfully")
    """
    X = np.asarray(X, dtype=np.float64)

    # Set default kernel function
    if kernel_fun is None:
        kernel_fun = gaussian_kernel_shm

    # Automatic bandwidth selection if H not provided
    if H is None:
        if bs_method == 1:
            H = _bandwidth_select_var(X)
        elif bs_method == 2:
            H = _bandwidth_select_cross_val(X, kernel_fun)
        else:
            raise ValueError("bs_method must be 1 or 2")
    else:
        H = np.asarray(H, dtype=np.float64)

    model = {"Xtrain": X, "kernel": kernel_fun, "H": H}

    return model


def score_kernel_density_shm(
    Y: np.ndarray, model: Dict[str, Any], return_log_densities: bool = True
) -> np.ndarray:
    """
    Score nonparametric kernel density estimation.

    .. meta::
        :category: Classification - Nonparametric Detectors
        :matlab_equivalent: scoreKernelDensity_shm
        :complexity: Advanced
        :data_type: Features
        :output_type: Scores
        :display_name: Score Kernel Density Estimation
        :verbose_call: Density Scores = Score Kernel Density Estimation (Test Features, Model, Use Log Densities)

    Parameters
    ----------
    Y : array_like
        Test features of shape (INSTANCES, FEATURES) where each row
        is a feature vector.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Test feature matrix

    model : dict
        Distribution parameters including training sample, kernel
        function, and bandwidth matrix (assumed to be diagonal).

        .. gui::
            :widget: model_input
            :description: Trained kernel density model

    return_log_densities : bool, optional
        Return log(densities) if True (default), otherwise return
        density values.

        .. gui::
            :widget: checkbox
            :default: true
            :description: Return log densities

    Returns
    -------
    densities : ndarray
        Evaluated log densities of shape (INSTANCES,).

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.classification import learn_kernel_density_shm, score_kernel_density_shm
    >>>
    >>> # Generate sample data
    >>> np.random.seed(42)
    >>> X_train = np.random.randn(100, 3)
    >>> X_test = np.random.randn(20, 3)
    >>>
    >>> # Learn model and score test data
    >>> model = learn_kernel_density_shm(X_train)
    >>> scores = score_kernel_density_shm(X_test, model)
    >>> print(f"Test scores shape: {scores.shape}")
    """
    Y = np.asarray(Y, dtype=np.float64)

    if Y.size == 0 or not model:
        raise ValueError("scoreKernelDensity_shm -> missing arguments")

    Xtrain = model["Xtrain"]
    kernel_fun = model["kernel"]
    H = model["H"]

    # First compute the determinant of H and its inverse
    detH = np.prod(np.diag(H))
    Hinv = np.diag(1.0 / np.diag(H))

    # Now compute the density at each x in Y
    n, D = Y.shape
    m = Xtrain.shape[0]
    densities = np.zeros(n)

    for i in range(n):
        x = Y[i, :]
        # Delta = (repmat(x, m, 1) - Xtrain) * Hinv
        Delta = (np.tile(x, (m, 1)) - Xtrain) @ Hinv
        weights = kernel_fun(Delta)
        densities[i] = (1.0 / (m * detH)) * np.sum(weights)

    # Use the log-likelihoods by default
    if return_log_densities:
        densities = np.log(densities + np.finfo(float).tiny)

    return densities


def roc_shm(
    scores: np.ndarray,
    damage_states: np.ndarray,
    num_pts: Optional[int] = None,
    threshold_type: str = "below",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Receiver operating characteristic (ROC) curve.

    .. meta::
        :category: Classification - Performance Evaluation
        :matlab_equivalent: ROC_shm
        :complexity: Intermediate
        :data_type: Scores
        :output_type: Performance Metrics
        :display_name: Receiver Operating Characteristic
        :verbose_call: [True Positive Rate, False Positive Rate] = Receiver Operating Characteristic (Scores, Damaged States, # of Points, Threshold Type)

    Parameters
    ----------
    scores : array_like
        Vector of scores for each instance of shape (INSTANCES,).

        .. gui::
            :widget: array_input
            :description: Classification scores

    damage_states : array_like
        Binary classification vector of known damage states
        (0-undamaged, 1-damaged) corresponding to scores.

        .. gui::
            :widget: array_input
            :description: True damage labels (0/1)

    num_pts : int, optional
        Number of points to evaluate ROC curve at. If None (default),
        each score value from damaged states is used as a threshold.

        .. gui::
            :widget: number_input
            :min: 10
            :max: 1000
            :description: Number of ROC points (optional)

    threshold_type : str, optional
        'above' or 'below' to define if scores above or below a given
        threshold should be flagged as damaged (default: 'below').

        .. gui::
            :widget: select
            :options: ["below", "above"]
            :default: "below"
            :description: Threshold direction

    Returns
    -------
    TPR : ndarray
        Vector of true positive rates.
    FPR : ndarray
        Vector of false positive rates.

    Notes
    -----
    Tool to compare and evaluate the performance of classification
    algorithms. Note that the scores should decrease for the damaged
    instances when using threshold_type='below'.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.classification import roc_shm
    >>>
    >>> # Generate sample scores and labels
    >>> np.random.seed(42)
    >>> scores = np.concatenate([np.random.randn(50) + 1, np.random.randn(50) - 1])
    >>> labels = np.concatenate([np.zeros(50), np.ones(50)])
    >>>
    >>> # Compute ROC curve
    >>> tpr, fpr = roc_shm(scores, labels, threshold_type='above')
    >>> print(f"ROC computed with {len(tpr)} points")
    """
    scores = np.asarray(scores, dtype=np.float64)
    damage_states = np.asarray(damage_states, dtype=bool)

    # Check parameters
    if len(scores) != len(damage_states):
        raise ValueError("Input arguments must have the same dimensions")

    # Handle threshold type
    if threshold_type in [0, "below"]:
        # Damage is flagged below threshold
        scores = -scores
    elif threshold_type in [1, "above"]:
        # Damage is flagged above threshold
        pass
    else:
        raise ValueError("Invalid option for threshold_type")

    # Sorted scores
    ordered_undam = np.sort(scores[~damage_states])[::-1]  # descending
    ordered_dam = np.sort(scores[damage_states])[::-1]  # descending

    # Set threshold values
    if num_pts is None:
        threshold_values = ordered_dam
    else:
        threshold_values = np.linspace(np.min(scores), np.max(scores), num_pts)

    num_thresholds = len(threshold_values)

    N = len(ordered_undam)  # number of undamaged instances
    P = len(ordered_dam)  # number of damaged instances

    # Calculate rates - following MATLAB exactly
    # MATLAB uses 1-based indexing but the algorithm logic is the same
    ukk = 0  # Index for undamaged sorted scores
    tkk = 0  # Index for damaged sorted scores
    TPR = np.zeros(num_thresholds)  # True positive rate
    FPR = np.zeros(num_thresholds)  # False positive rate

    for j in range(num_thresholds):
        # Count undamaged instances above threshold (false positives)
        while ukk < N and ordered_undam[ukk] > threshold_values[j]:
            FPR[j] += 1
            ukk += 1

        # Count damaged instances above threshold (true positives)
        while tkk < P and ordered_dam[tkk] > threshold_values[j]:
            TPR[j] += 1
            tkk += 1

    # Convert counts to rates using cumulative sum
    TPR = np.cumsum(TPR) / P
    FPR = np.cumsum(FPR) / N

    return TPR, FPR


def _bandwidth_select_var(X_train: np.ndarray) -> np.ndarray:
    """
    Use the variance of the data for bandwidth selection.

    Setting along each direction should be optimal for l2 error (up to a
    constant), assuming true density function is 1-Lipschitz.
    """
    n, D = X_train.shape
    H_vec = np.std(X_train, axis=0)
    H = ((1.0 / n) ** (1.0 / 3.0)) * np.diag(H_vec)
    return H


def _bandwidth_select_cross_val(
    X_train: np.ndarray, kernel_fun: Callable
) -> np.ndarray:
    """
    Use cross validation for bandwidth selection.
    """
    n, D = X_train.shape
    # MATLAB: hVal = power(2, -[-ceil(log2(sqrt(D))):1:floor(log2(n))]);
    start = -np.ceil(np.log2(np.sqrt(D)))
    end = np.floor(np.log2(n))
    h_val = 2.0 ** (-np.arange(start, end + 1))  # +1 to include endpoint like MATLAB
    m = len(h_val)
    H_vec = np.std(X_train, axis=0)

    # Pick 80% for training and test on 20%, do this 5 times
    n1 = int(np.floor(0.8 * n))
    likelihood = np.ones((m, 5))

    for t in range(5):
        I = np.random.permutation(n)
        X1 = X_train[I[:n1], :]
        X2 = X_train[I[n1:], :]

        model = {"Xtrain": X1, "kernel": kernel_fun, "H": None}

        for i in range(m):
            model["H"] = h_val[i] * np.diag(H_vec)
            # MATLAB calls scoreKernelDensity_shm without return_log_densities parameter
            # which defaults to returning log densities (the default behavior)
            Z = score_kernel_density_shm(X2, model) + 1e-10
            likelihood[i, t] = np.sum(Z)

    # Find best bandwidth
    h_i = np.argmax(np.mean(likelihood, axis=1))
    H = h_val[h_i] * np.diag(H_vec)

    return H
