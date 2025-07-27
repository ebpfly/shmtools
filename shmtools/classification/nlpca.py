"""Nonlinear Principal Component Analysis (NLPCA) using autoassociative neural networks.

This module implements NLPCA for outlier detection in structural health monitoring
using autoencoder neural networks with bottleneck layers.

References
----------
.. [1] Sohn, H., Worden, K., & Farrar, C. R. (2002). Statistical Damage
       Classification under Changing Environmental and Operational Conditions.
       Journal of Intelligent Material Systems and Structures, 13(9), 561-574.
.. [2] Kramer, M. A. (1991). Nonlinear Principal Component Analysis using
       Autoassociative Neural Networks. AIChE Journal, 37(2), 233-243.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn(
        "TensorFlow not available. NLPCA functions will not work. "
        "Install TensorFlow: pip install tensorflow"
    )


def learn_nlpca_shm(
    X: np.ndarray,
    b: int = 1,
    M1: int = 5,
    M2: Optional[int] = None,
    param_ite: int = 50,
    param_per: float = 1e-10,
) -> Dict[str, Any]:
    """Learn nonlinear principal component analysis (NLPCA) model.

    Trains an autoassociative neural network (autoencoder) to learn the
    correlation among features and reveal unobserved variables that drive
    changes in the data. The bottleneck layer represents these correlations.

    .. meta::
        :category: Classification - Non-Parametric Detectors
        :matlab_equivalent: learnNLPCA_shm
        :display_name: Learn NLPCA
        :verbose_call: [Model] = Learn NLPCA (Training Features, # Nodes Bottleneck Layer, # Nodes Mapping Layer, # Nodes de-Mapping Layer, # Iterations, Performance)
        :complexity: Advanced
        :data_type: Features
        :output_type: Model

    Parameters
    ----------
    X : array_like
        Training features where each row is a feature vector.
        Shape: (INSTANCES, FEATURES)

        .. gui::
            :widget: data_select
            :description: Training Features

    b : int, optional
        Number of nodes in the bottleneck layer. Default: 1

        .. gui::
            :widget: number_input
            :min: 1
            :max: 10
            :default: 1
            :description: # Nodes Bottleneck Layer

    M1 : int, optional
        Number of nodes in the mapping layer. Default: 5

        .. gui::
            :widget: number_input
            :min: 1
            :max: 20
            :default: 5
            :description: # Nodes Mapping Layer

    M2 : int, optional
        Number of nodes in the de-mapping layer. If None, uses M1. Default: None

        .. gui::
            :widget: number_input
            :min: 1
            :max: 20
            :default: 5
            :description: # Nodes de-Mapping Layer

    param_ite : int, optional
        Number of training iterations (epochs). Default: 50

        .. gui::
            :widget: number_input
            :min: 10
            :max: 1000
            :default: 50
            :description: # Iterations

    param_per : float, optional
        Training performance goal (MSE threshold). Default: 1e-10

        .. gui::
            :widget: number_input
            :min: 1e-12
            :max: 1e-3
            :default: 1e-10
            :description: Performance

    Returns
    -------
    model : dict
        Trained NLPCA model containing:

        - 'net': Trained neural network model
        - 'b': Number of bottleneck nodes
        - 'M1': Number of mapping layer nodes
        - 'M2': Number of de-mapping layer nodes
        - 'E': Training mean square error
        - 'input_mean': Input data mean for normalization
        - 'input_std': Input data std for normalization

    Notes
    -----
    The network architecture is:
    Input -> Mapping (M1 nodes, tanh) -> Bottleneck (b nodes, linear) ->
    De-mapping (M2 nodes, tanh) -> Output (linear)

    In damage detection context, assuming the network is trained on
    undamaged system features, prediction errors will grow when features
    from potentially damaged systems are input.

    Examples
    --------
    >>> # Train NLPCA on baseline features
    >>> baseline_features = np.random.randn(100, 3)  # 100 samples, 3 features
    >>> model = learn_nlpca_shm(baseline_features, b=2, M1=5, M2=5)
    >>> print(f"Training MSE: {model['E']:.6f}")
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for NLPCA. Install with: pip install tensorflow"
        )

    # Convert to numpy array and validate
    X = np.array(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (INSTANCES, FEATURES)")

    n_instances, n_features = X.shape

    # Set default M2
    if M2 is None:
        M2 = M1

    # Parameter validation (following MATLAB logic)
    if M1 + M2 <= b:
        raise ValueError(
            "Number of nodes in mapping and de-mapping layers must be "
            "higher than number of nodes in bottleneck layer"
        )

    max_nodes = int(n_features * (n_instances - b) / (n_features + b + 1))
    if M1 + M2 > max_nodes:
        raise ValueError(f"For b={b}, the number of nodes M1+M2 must be < {max_nodes}")

    # Normalize input data (important for neural network training)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std = np.where(X_std == 0, 1.0, X_std)  # Avoid division by zero
    X_normalized = (X - X_mean) / X_std

    # Build autoencoder network
    # Architecture: Input -> M1 (tanh) -> b (linear) -> M2 (tanh) -> Output (linear)
    model = keras.Sequential(
        [
            layers.Dense(
                M1, activation="tanh", input_shape=(n_features,), name="mapping"
            ),
            layers.Dense(b, activation="linear", name="bottleneck"),
            layers.Dense(M2, activation="tanh", name="demapping"),
            layers.Dense(n_features, activation="linear", name="output"),
        ]
    )

    # Compile model (using MSE like MATLAB)
    model.compile(
        optimizer="adam",  # Adam is more robust than default SGD
        loss="mse",
        metrics=["mse"],
    )

    # Early stopping callback to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="loss", patience=10, restore_best_weights=True, min_delta=param_per
    )

    # Train the network (X -> X, autoencoder)
    history = model.fit(
        X_normalized,
        X_normalized,
        epochs=param_ite,
        batch_size=min(32, n_instances),
        verbose=0,  # Silent training
        callbacks=[early_stopping],
    )

    # Calculate training error
    predictions = model.predict(X_normalized, verbose=0)
    training_error = np.mean((X_normalized - predictions) ** 2)

    # Return model structure matching MATLAB output
    return {
        "net": model,
        "b": b,
        "M1": M1,
        "M2": M2,
        "E": float(training_error),
        "input_mean": X_mean,
        "input_std": X_std,
        "training_history": history.history,
    }


def score_nlpca_shm(
    Y: np.ndarray, model: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Score test data using trained NLPCA model.

    For each instance of Y, returns a score based on the Euclidean norm
    of the residual errors between target features and network output.

    .. meta::
        :category: Classification - Non-Parametric Detectors
        :matlab_equivalent: scoreNLPCA_shm
        :display_name: Score NLPCA
        :verbose_call: [Scores, Residuals] = Score NLPCA (Test Features, Model)
        :complexity: Basic
        :data_type: Features
        :output_type: Scores

    Parameters
    ----------
    Y : array_like
        Test features where each row is a feature vector.
        Shape: (INSTANCES, FEATURES)

        .. gui::
            :widget: data_select
            :description: Test Features

    model : dict
        Trained NLPCA model from learn_nlpca_shm

        .. gui::
            :widget: data_select
            :description: Model

    Returns
    -------
    scores : ndarray
        Damage indicator scores (negative Euclidean distances).
        Shape: (INSTANCES,)
        Higher (less negative) values indicate more damage.

    residuals : ndarray
        Residual errors between network output and input.
        Shape: (INSTANCES, FEATURES)

    Notes
    -----
    In damage detection context, assuming the network was trained on
    undamaged system features, prediction errors (and thus scores) will
    grow when features from potentially damaged systems are input.

    The scores are negative Euclidean distances, so values closer to zero
    indicate higher reconstruction error (more damage).

    Examples
    --------
    >>> # Score test data using trained model
    >>> test_features = np.random.randn(50, 3)
    >>> scores, residuals = score_nlpca_shm(test_features, model)
    >>> print(f"Score range: {np.min(scores):.3f} to {np.max(scores):.3f}")
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for NLPCA. Install with: pip install tensorflow"
        )

    # Convert to numpy array and validate
    Y = np.array(Y, dtype=np.float32)
    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array with shape (INSTANCES, FEATURES)")

    # Extract model components
    net = model["net"]
    input_mean = model["input_mean"]
    input_std = model["input_std"]

    # Normalize test data using training statistics
    Y_normalized = (Y - input_mean) / input_std

    # Get network predictions
    predictions = net.predict(Y_normalized, verbose=0)

    # Calculate residuals in normalized space
    residuals_normalized = predictions - Y_normalized

    # Convert residuals back to original scale
    residuals = residuals_normalized * input_std

    # Calculate scores as negative Euclidean distance (matching MATLAB)
    scores = -np.sqrt(np.sum(residuals**2, axis=1))

    return scores, residuals
