"""
Time series modeling functions for feature extraction.

This module provides autoregressive (AR) and ARMAX modeling functions
for extracting damage-sensitive features from time series data.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy import linalg


def ar_model_shm(
    X: np.ndarray, ar_order: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate autoregressive model parameters and compute RMSE.

    .. meta::
        :category: Features - Time Series Models
        :matlab_equivalent: arModel_shm
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Features
        :display_name: AR Model
        :verbose_call: [AR Parameters Feature Vectors, RMS Residuals Feature Vectors, AR Parameters, AR Residuals, AR Prediction] = AR Model (Time Series Data, AR Model Order)

    Parameters
    ----------
    X : array_like
        Input time series data of shape (TIME, CHANNELS, INSTANCES).

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]

    ar_order : int, optional
        AR model order. Default is 15.

        .. gui::
            :widget: number_input
            :min: 1
            :max: 50
            :default: 15

    Returns
    -------
    ar_parameters_fv : ndarray
        Feature vectors of AR parameters in concatenated format,
        shape (INSTANCES, FEATURES) where FEATURES = CHANNELS*ar_order.
    rms_residuals_fv : ndarray
        Feature vectors of root mean squared AR residual errors in concatenated format,
        shape (INSTANCES, FEATURES) where FEATURES = CHANNELS.
    ar_parameters : ndarray
        AR parameters of shape (AR_ORDER, CHANNELS, INSTANCES).
    ar_residuals : ndarray
        AR residuals of shape (TIME, CHANNELS, INSTANCES).
    ar_prediction : ndarray
        AR prediction of shape (TIME, CHANNELS, INSTANCES).

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.features import ar_model_shm
    >>>
    >>> # Generate sample data: 100 time points, 2 channels, 3 instances
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 2, 3)
    >>>
    >>> # Fit AR(15) model
    >>> ar_params_fv, rms_fv, ar_params, residuals, prediction = ar_model_shm(X, 15)
    >>> print(f"AR parameters FV shape: {ar_params_fv.shape}")  # (3, 30)
    >>> print(f"RMS residuals FV shape: {rms_fv.shape}")        # (3, 2)
    """
    X = np.asarray(X)

    # Set parameters following MATLAB exactly
    t, m, n = X.shape

    ar_param = np.zeros((ar_order, m))
    ar_prediction = X.copy()
    ar_residuals = np.zeros((t, m, n))
    A = np.zeros((t - ar_order, ar_order))
    rms_residuals_fv = np.zeros((n, m))
    ar_parameters_fv = np.zeros((n, ar_order * m))
    ar_parameters = np.zeros((ar_order, m, n))

    # AR model estimation - following MATLAB algorithm exactly
    for j in range(n):
        for i in range(m):
            # Populate matrix A for AR parameters calculation
            # Following MATLAB exactly: for k=1:arOrder; A(:,k)=X(arOrder+1-k:t-k,i,j); end
            for k in range(ar_order):
                # MATLAB: k goes 1 to arOrder, Python: k goes 0 to arOrder-1
                # MATLAB: A(:,k) = X(arOrder+1-k:t-k,i,j)
                # Python equivalent: A[:, k] = X[arOrder-1-(k):t-1-(k), i, j]
                matlab_k = k + 1  # Convert to MATLAB 1-based indexing
                start_idx = (
                    ar_order + 1 - matlab_k - 1
                )  # Convert to 0-based: arOrder-k-1
                end_idx = (
                    t - matlab_k
                )  # Convert to 0-based: t-k-1, but Python end is exclusive so t-k
                A[:, k] = X[start_idx:end_idx, i, j]

            # Define B vector
            # MATLAB: B(:,:) = X(arOrder+1:t,i,j) -> Python: X[arOrder:t, i, j]
            B = X[ar_order:t, i, j]

            # Calculate AR parameters by solving least squares problem (Ax=B)
            ar_param[:, i] = np.linalg.pinv(A) @ B

            # Predict the measured signals using the defined AR model
            ar_prediction[ar_order:t, i, j] = A @ ar_param[:, i]

            # Calculate the residual errors between predicted and measured signals
            ar_residuals[:, i, j] = ar_prediction[:, i, j] - X[:, i, j]

        ar_parameters[:, :, j] = ar_param

        # Reshape parameters to feature vector (concatenated format)
        # Use Fortran order to match MATLAB's column-wise reshape
        ar_parameters_fv[j, :] = ar_param.reshape(ar_order * m, order="F")

        # Calculate RMS of residuals
        rms_residuals_fv[j, :] = np.sqrt(np.mean(ar_residuals[:, :, j] ** 2, axis=0))

    return (
        ar_parameters_fv,
        rms_residuals_fv,
        ar_parameters,
        ar_residuals,
        ar_prediction,
    )


def arx_model(
    y: np.ndarray, x: Optional[np.ndarray] = None, na: int = 2, nb: int = 2, nk: int = 1
) -> Dict[str, np.ndarray]:
    """
    Fit ARX (autoregressive with exogenous input) model.

    Python equivalent of MATLAB's arxModel function.

    Parameters
    ----------
    y : np.ndarray
        Output time series.
    x : np.ndarray, optional
        Input time series. If None, fits AR model.
    na : int, optional
        Number of AR terms. Default is 2.
    nb : int, optional
        Number of exogenous input terms. Default is 2.
    nk : int, optional
        Input delay (number of samples). Default is 1.

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'a': AR coefficients
        - 'b': Exogenous input coefficients (if x provided)
        - 'variance': Noise variance estimate
    """
    # TODO: Implement ARX parameter estimation
    # This is a placeholder for the actual implementation

    if x is None:
        # Pure AR model - convert to expected format
        if y.ndim == 1:
            y_formatted = y.reshape(-1, 1, 1)
        elif y.ndim == 2:
            y_formatted = y.reshape(y.shape[0], y.shape[1], 1)
        else:
            y_formatted = y

        _, _, ar_params, _, _ = ar_model_shm(y_formatted, na)
        return {
            "a": ar_params[:, :, 0],
            "b": None,
            "variance": np.var(y),  # Placeholder
        }
    else:
        # ARX model with exogenous input
        raise NotImplementedError("ARX model with exogenous input not yet implemented")


def ar_model_order_shm(
    X: np.ndarray, method: str = "PAF", ar_order_max: int = 30, tolerance: float = 0.078
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Determine appropriate autoregressive model order (MATLAB-compatible).

    This function determines the appropriate AutoRegressive (AR) model order
    using one of five available methods for time series analysis.

    .. meta::
        :category: Features - Time Series Models
        :matlab_equivalent: arModelOrder_shm
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Model Parameters
        :display_name: AR Model Order
        :verbose_call: [Mean AR Order, AR Orders, Model] = AR Model Order (Time Series Data, Method, Maximum AR Order, Tolerance)

    Parameters
    ----------
    X : array_like
        Input time series data.
        - 1D: Single time series (TIME,)
        - 2D: Multi-channel (TIME, CHANNELS)
        - 3D: Multi-instance (TIME, CHANNELS, INSTANCES)

        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]

    method : str, optional
        AR order selection method (default: 'PAF').
        - 'PAF': Partial Autocorrelation Function
        - 'AIC': Akaike Information Criterion
        - 'BIC': Bayesian Information Criterion
        - 'RMS': Root Mean Square error
        - 'SVD': Singular Value Decomposition

        .. gui::
            :widget: select
            :options: ["PAF", "AIC", "BIC", "RMS", "SVD"]
            :default: "PAF"

    ar_order_max : int, optional
        Maximum AR order to test (default: 30).

        .. gui::
            :widget: number_input
            :min: 2
            :max: 100
            :default: 30

    tolerance : float, optional
        Tolerance threshold for convergence criteria (default: 0.078).

        .. gui::
            :widget: number_input
            :min: 0.001
            :max: 1.0
            :default: 0.078
            :step: 0.001

    Returns
    -------
    mean_ar_order : ndarray
        Mean AR order for each channel, shape (CHANNELS,).
    ar_orders : ndarray
        AR orders for each instance, shape (CHANNELS, INSTANCES).
    model : dict
        Dictionary containing:
        - 'control_limits': Control limits for selection criterion
        - 'out_data': Criterion values for each tested order
        - 'method': Method used for order selection

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.features import ar_model_order_shm
    >>>
    >>> # Generate test AR(5) process
    >>> np.random.seed(42)
    >>> t = np.arange(1000)
    >>> x = np.zeros(1000)
    >>> for i in range(5, 1000):
    ...     x[i] = 0.5*x[i-1] - 0.3*x[i-2] + 0.1*x[i-3] + 0.01*np.random.randn()
    >>> X = x.reshape(-1, 1, 1)  # (TIME, CHANNELS, INSTANCES)
    >>>
    >>> # Determine optimal AR order
    >>> mean_order, orders, model = ar_model_order_shm(X, method='PAF')
    >>> print(f"Optimal AR order: {mean_order[0]:.0f}")

    References
    ----------
    Original MATLAB implementation from SHMTools (LA-CC-14-046).
    """
    X = np.asarray(X, dtype=np.float64)

    # Handle different input dimensions
    if X.ndim == 1:
        # Single time series: (TIME,) -> (TIME, 1, 1)
        X = X[:, np.newaxis, np.newaxis]
    elif X.ndim == 2:
        # Multi-channel: (TIME, CHANNELS) -> (TIME, CHANNELS, 1)
        X = X[:, :, np.newaxis]
    elif X.ndim != 3:
        raise ValueError("Input X must be 1D, 2D, or 3D array")

    time_points, n_channels, n_instances = X.shape

    # Initialize outputs
    ar_orders = np.zeros((n_channels, n_instances))
    ar_order_list = np.arange(1, ar_order_max + 1)
    out_data = np.zeros(ar_order_max)

    # Process each channel and instance
    for ch in range(n_channels):
        for inst in range(n_instances):
            # Extract single time series
            x = X[:, ch, inst]

            # Find optimal order for this time series using internal method
            optimal_order, criterion_values = _ar_order_selection(
                x, max_order=ar_order_max, method=method.lower(), tolerance=tolerance
            )

            ar_orders[ch, inst] = optimal_order

            # Accumulate criterion values (average across instances)
            if inst == 0:
                out_data = criterion_values
            else:
                out_data += criterion_values

    # Average outputs across instances
    out_data = out_data / n_instances

    # Calculate mean AR order across instances
    mean_ar_order = np.ceil(np.mean(ar_orders, axis=1))

    # Set control limits based on method
    if method.upper() == "PAF":
        # PAF uses 95% confidence bounds: ±2/√N
        n_eff = time_points
        control_limit = [2 / np.sqrt(n_eff), -2 / np.sqrt(n_eff)]
    else:
        # Other methods use data-driven limits
        control_limit = [np.min(out_data), np.max(out_data)]

    # Create output model dictionary
    model = {
        "control_limits": control_limit,
        "out_data": out_data,
        "method": method.upper(),
        "tolerance": tolerance,
        "ar_order_max": ar_order_max,
    }

    return mean_ar_order, ar_orders, model


def _ar_order_selection(
    x: np.ndarray, max_order: int = 30, method: str = "aic", tolerance: float = 0.078
) -> Tuple[int, np.ndarray]:
    """
    Internal function to determine optimal AR model order using various criteria.

    This is a simplified version of the ar_model_order function for use within
    ar_model_order_shm.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    t = len(x)

    if max_order >= t:
        raise ValueError(
            f"Maximum order ({max_order}) must be less than time series length ({t})"
        )

    # Initialize output arrays
    criterion_values = np.zeros(max_order)

    if method.lower() == "paf":
        # Partial Autocorrelation Function method
        ar_param_last = np.zeros(max_order)

        # Compute AR parameters for each order
        for order in range(1, max_order + 1):
            x_formatted = x.reshape(-1, 1, 1)
            _, _, ar_params, _, _ = ar_model_shm(x_formatted, order)
            # Extract last AR parameter
            ar_param_last[order - 1] = ar_params[-1, 0, 0]

        criterion_values = ar_param_last

        # Find optimal order using confidence interval threshold
        # 95% confidence interval for white noise: ±2/sqrt(N)
        k = 2.0 / np.sqrt(t)
        optimal_indices = np.where(np.abs(criterion_values) < k)[0]
        optimal_order = (
            optimal_indices[0] + 1 if len(optimal_indices) > 0 else max_order
        )

    else:
        # Information criteria methods (AIC, BIC, RMS)
        sum_squared_errors = np.zeros(max_order)

        # Compute AR models for all orders
        for order in range(1, max_order + 1):
            x_formatted = x.reshape(-1, 1, 1)
            _, _, _, residuals, _ = ar_model_shm(x_formatted, order)
            # Sum squared errors
            sum_squared_errors[order - 1] = np.sum(residuals**2)

        # Effective sample sizes
        n_effective = t - np.arange(1, max_order + 1)

        if method.lower() == "aic":
            # Akaike Information Criterion
            log_likelihood = -0.5 * t * np.log(sum_squared_errors / t)
            criterion_values = -2 * log_likelihood + 2 * np.arange(1, max_order + 1)
            optimal_order = np.argmin(criterion_values) + 1

        elif method.lower() == "bic":
            # Bayesian Information Criterion
            log_likelihood = -0.5 * t * np.log(sum_squared_errors / t)
            criterion_values = -2 * log_likelihood + np.arange(
                1, max_order + 1
            ) * np.log(t)
            optimal_order = np.argmin(criterion_values) + 1

        elif method.lower() == "rms":
            # Root Mean Square error criterion
            criterion_values = np.sqrt(sum_squared_errors / n_effective)

            # Find first order where RMS improvement is below tolerance
            if len(criterion_values) > 1:
                rms_changes = np.abs(np.diff(criterion_values)) / criterion_values[:-1]
                optimal_indices = np.where(rms_changes < tolerance)[0]
                optimal_order = (
                    optimal_indices[0] + 1 if len(optimal_indices) > 0 else max_order
                )
            else:
                optimal_order = 1

        else:
            raise ValueError(
                f"Unknown method: {method}. Choose from 'aic', 'bic', 'paf', 'rms'"
            )

    return optimal_order, criterion_values
