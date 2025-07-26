"""
Semi-parametric outlier detection functions.

This module provides Gaussian Mixture Model (GMM) based functions for
semi-parametric outlier detection using clustering-based partitioning.
"""

import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple
from scipy.linalg import det


def k_medians_shm(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partition the data using k-medians clustering.

    .. meta::
        :category: Classification - Semi-Parametric Detectors
        :matlab_equivalent: kMedians_shm
        :complexity: Advanced
        :data_type: Features
        :output_type: Clustering
        :display_name: K Median Clustering
        :verbose_call: [Partition of X, Cluster Centers] = K Median Clustering (Data Set, Number of Partitions)

    Parameters
    ----------
    X : array_like
        Training features of shape (INSTANCES, FEATURES).

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Data set for clustering

    k : int
        Number of partition cells.

        .. gui::
            :widget: number_input
            :min: 1
            :max: 50
            :default: 5
            :description: Number of partitions

    Returns
    -------
    idx : ndarray
        Vector of cluster membership of shape (INSTANCES,).
    i_c : ndarray
        Vector of cluster centers of shape (k,).

    Notes
    -----
    K-median clustering algorithm that selects cluster centers to minimize
    the total distance from points to their nearest center.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.classification import k_medians_shm
    >>>
    >>> # Generate sample data
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 3)
    >>>
    >>> # Cluster into 3 groups
    >>> idx, centers = k_medians_shm(X, 3)
    >>> print(f"Cluster assignments shape: {idx.shape}")
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]

    if k < 1:
        raise ValueError("k must be at least 1")
    if k > N:
        k = N

    # Compute pairwise squared distances
    distances = np.zeros((N, N))

    for j in range(N - 1):
        # Compute distances from point j to all subsequent points
        z = X[j, :]
        remaining = X[j + 1:, :]

        # Squared Euclidean distances
        sq_diffs = remaining - z
        dist = np.sum(sq_diffs * sq_diffs, axis=1)

        distances[j, j + 1:] = dist
        distances[j + 1:, j] = dist

    # Select k points as cluster centers
    i_c = [np.random.randint(0, N)]  # First center chosen randomly

    if k < 2:
        idx = np.ones(N, dtype=int)
        return idx, np.array(i_c)

    for i in range(1, k):
        # Distances from all points to current centers
        temp_dist = distances[i_c, :]

        if i > 1:
            # Minimum distance to any center
            temp_dist = np.min(temp_dist, axis=0)
        else:
            temp_dist = temp_dist[0, :]

        # Find point farthest from current centers
        max_i = np.argmax(temp_dist)
        i_c.append(max_i)

    # Assign points to closest centers
    temp_dist = distances[i_c, :]
    idx = np.argmin(temp_dist, axis=0) + 1  # MATLAB uses 1-based indexing

    # Convert to 1-based for MATLAB compatibility
    return idx, np.array(i_c) + 1


def learn_gmm_shm(
    X: np.ndarray, idx: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Learn gaussian mixture model.

    .. meta::
        :category: Classification - Semi-Parametric Detectors
        :matlab_equivalent: learnGMM_shm
        :complexity: Advanced
        :data_type: Features
        :output_type: Model
        :display_name: Learn Gaussian Mixture Model
        :verbose_call: Model = Learn Gaussian Mixture Model (Training Features, Clustering Indices)

    Parameters
    ----------
    X : array_like
        Training features of shape (INSTANCES, FEATURES) where each row
        is a feature vector.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Training feature matrix

    idx : array_like, optional
        Vector of cluster membership of shape (INSTANCES,). If not provided,
        a single Gaussian is learned.

        .. gui::
            :widget: array_input
            :description: Cluster membership vector (optional)

    Returns
    -------
    model : dict
        Dictionary containing GMM parameters:
        - 'p': Mixture rates of shape (k,)
        - 'mu': Means of shape (FEATURES, k)
        - 's2': Covariance matrices of shape (FEATURES, FEATURES, k)

    Notes
    -----
    Learn a mixture of gaussians over the given partition idx of the points
    in X. If idx is not passed in, then a single Gaussian is learned.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.classification import learn_gmm_shm
    >>>
    >>> # Generate sample data
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 3)
    >>> idx = np.random.randint(1, 4, 100)  # 3 clusters
    >>>
    >>> # Learn GMM
    >>> model = learn_gmm_shm(X, idx)
    >>> print(f"Number of components: {len(model['p'])}")
    """
    X = np.asarray(X, dtype=np.float64)
    n, D = X.shape

    if idx is None:
        idx = np.ones(n)
    else:
        idx = np.asarray(idx)

    k = len(np.unique(idx))
    threshold = 0.000005  # For matrices too close to singular

    p = np.zeros(k)  # Mixing proportions
    mu = np.zeros((D, k))  # Means
    s2 = np.zeros((D, D, k))  # Covariances

    unique_labels = np.unique(idx)
    for i, label in enumerate(unique_labels):
        # Get points in this cluster
        cluster_mask = idx == label
        n_i = np.sum(cluster_mask)

        if n_i == 0:
            continue

        C = X[cluster_mask, :]

        # Compute mean
        mu_C = np.mean(C, axis=0)

        # Compute covariance
        if n_i > 1:
            centered = C - mu_C
            Cov = (1.0 / n_i) * (centered.T @ centered)
        else:
            # Single point - use small identity matrix
            Cov = threshold * np.eye(D)

        # Fix covariance to prevent singularity
        eigenvals, eigenvecs = np.linalg.eigh(Cov)
        eigenvals = np.maximum(eigenvals, threshold)
        Cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        p[i] = n_i / n
        mu[:, i] = mu_C
        s2[:, :, i] = Cov

    model = {"p": p, "mu": mu, "s2": s2}
    return model


def score_gmm_shm(
    Y: np.ndarray, model: Dict[str, Any], return_log_densities: bool = True
) -> np.ndarray:
    """
    Score gaussian mixture model.

    .. meta::
        :category: Classification - Semi-Parametric Detectors
        :matlab_equivalent: scoreGMM_shm
        :complexity: Advanced
        :data_type: Features
        :output_type: Scores
        :display_name: Score Gaussian Mixture Model
        :verbose_call: Density Scores = Score Gaussian Mixture Model (Test Features, Model, Use Log Densities)

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
        Distribution parameters from learn_gmm_shm containing 'p', 'mu', 's2'.

        .. gui::
            :widget: model_input
            :description: Trained GMM model

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
        Evaluated densities or log densities of shape (INSTANCES,).

    Notes
    -----
    Estimate the density at points in Y given the distribution (model).

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.classification import learn_gmm_shm, score_gmm_shm
    >>>
    >>> # Generate sample data
    >>> np.random.seed(42)
    >>> X_train = np.random.randn(100, 3)
    >>> X_test = np.random.randn(20, 3)
    >>>
    >>> # Learn and score
    >>> model = learn_gmm_shm(X_train)
    >>> scores = score_gmm_shm(X_test, model)
    >>> print(f"Scores shape: {scores.shape}")
    """
    Y = np.asarray(Y, dtype=np.float64)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    p = model["p"]
    mu = model["mu"]
    s2 = model["s2"]

    k = mu.shape[1]  # Number of mixture components
    n, D = Y.shape

    # Compute likelihood for each component
    component_likelihoods = np.zeros((n, k))

    for i in range(k):
        if p[i] > 0:  # Only compute if component has non-zero weight
            # Current component parameters
            mu_i = mu[:, i]
            cov_i = s2[:, :, i]

            # Compute determinant and inverse
            try:
                det_cov = det(cov_i)
                inv_cov = np.linalg.inv(cov_i)

                if det_cov <= 0:
                    # Singular covariance matrix
                    component_likelihoods[:, i] = 0
                    continue

                # Centered data
                centered = Y - mu_i

                # Mahalanobis distance
                mahal_dist = np.sum(
                    (centered @ inv_cov) * centered, axis=1
                )

                # Gaussian density
                normalizer = 1.0 / (
                    (2 * np.pi) ** (D / 2) * np.sqrt(det_cov)
                )
                component_likelihoods[:, i] = (
                    p[i] * normalizer * np.exp(-0.5 * mahal_dist)
                )

            except np.linalg.LinAlgError:
                # Singular matrix
                component_likelihoods[:, i] = 0

    # Sum over mixture components
    densities = np.sum(component_likelihoods, axis=1)

    # Return log densities by default
    if return_log_densities:
        densities = np.log(densities + np.finfo(float).tiny)

    return densities


def learn_gmm_semiparametric_model_shm(
    X: np.ndarray, partition_fun: Callable = None, k: int = 5
) -> Dict[str, Any]:
    """
    Learn GMM semi-parametric density model.

    .. meta::
        :category: Classification - Semi-Parametric Detectors
        :matlab_equivalent: learnGMMSemiParametricModel_shm
        :complexity: Advanced
        :data_type: Features
        :output_type: Model
        :display_name: Learn GMM Semi-Parametric Density Model
        :verbose_call: Density Model = Learn GMM semi-Parametric Density Model (Training Features, Partition Function, k)

    Parameters
    ----------
    X : array_like
        Training features of shape (INSTANCES, FEATURES) where each row
        is a feature vector.

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Training feature matrix

    partition_fun : callable, optional
        Function handle taking parameters (X, k), returning cluster indices.
        Default is k_medians_shm.

        .. gui::
            :widget: select
            :options: ["k_medians_shm", "k_means"]
            :default: "k_medians_shm"
            :description: Partitioning function

    k : int, optional
        Number of partition cells (default: 5).

        .. gui::
            :widget: number_input
            :min: 1
            :max: 20
            :default: 5
            :description: Number of partitions

    Returns
    -------
    model : dict
        Parameters of the GMM model suitable for scoring.

    Notes
    -----
    Serves to assemble a semi-parametric model learning function, using the
    partitioning function provided. The partitioning function is used to
    partition the data, and a Gaussian Mixture Model (GMM) is learned over
    the partition.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.classification import learn_gmm_semiparametric_model_shm
    >>>
    >>> # Generate sample data
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 3)
    >>>
    >>> # Learn semi-parametric model
    >>> model = learn_gmm_semiparametric_model_shm(X, k=3)
    >>> print("Model learned successfully")
    """
    if partition_fun is None:
        partition_fun = k_medians_shm

    # Get cluster assignments
    idx, _ = partition_fun(X, k)

    # Learn GMM on partitioned data
    model = learn_gmm_shm(X, idx)

    return model


def score_gmm_semiparametric_model_shm(
    Y: np.ndarray, model: Dict[str, Any], return_log_densities: bool = True
) -> np.ndarray:
    """
    Score GMM semi-parametric density model.

    .. meta::
        :category: Classification - Semi-Parametric Detectors
        :matlab_equivalent: scoreGMMSemiParametricModel_shm
        :complexity: Advanced
        :data_type: Features
        :output_type: Scores
        :display_name: Score GMM Semi-Parametric Density Model
        :verbose_call: Density Scores = Score GMM Semi-Parametric Density Model (Test Features, Model)

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
        Distribution parameters from learning function.

        .. gui::
            :widget: model_input
            :description: Trained semi-parametric model

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
        Evaluated densities or log densities of shape (INSTANCES,).

    Notes
    -----
    Estimate the density at points in Y given the distribution (model).
    This function just calls score_gmm_shm and exists to maintain the
    structure of learn/score pairs expected by assembly routines.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.classification import (
    ...     learn_gmm_semiparametric_model_shm,
    ...     score_gmm_semiparametric_model_shm
    ... )
    >>>
    >>> # Generate sample data
    >>> np.random.seed(42)
    >>> X_train = np.random.randn(100, 3)
    >>> X_test = np.random.randn(20, 3)
    >>>
    >>> # Learn and score
    >>> model = learn_gmm_semiparametric_model_shm(X_train)
    >>> scores = score_gmm_semiparametric_model_shm(X_test, model)
    >>> print(f"Scores shape: {scores.shape}")
    """
    return score_gmm_shm(Y, model, return_log_densities)
