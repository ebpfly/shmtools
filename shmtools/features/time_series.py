"""
Time series modeling functions for feature extraction.

This module provides autoregressive (AR) and ARMAX modeling functions
for extracting damage-sensitive features from time series data.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy import linalg


def ar_model_shm(X: np.ndarray, ar_order: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate autoregressive model parameters and compute RMSE.
    
    .. meta::
        :category: Features - Time Series Models  
        :matlab_equivalent: arModel_shm
        :complexity: Intermediate
        :data_type: Time Series
        :output_type: Features
        
    Parameters
    ----------
    X : array_like
        Input time series data of shape (TIME, CHANNELS, INSTANCES).
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            
    ar_order : int, optional
        AR model order.
        
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
    >>> # Fit AR(5) model
    >>> ar_params_fv, rms_fv, ar_params, residuals, prediction = ar_model_shm(X, 5)
    >>> print(f"AR parameters FV shape: {ar_params_fv.shape}")  # (3, 10)
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
                start_idx = ar_order + 1 - matlab_k - 1  # Convert to 0-based: arOrder-k-1
                end_idx = t - matlab_k  # Convert to 0-based: t-k-1, but Python end is exclusive so t-k
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
        ar_parameters_fv[j, :] = ar_param.reshape(ar_order * m, order='F')
        
        # Calculate RMS of residuals
        rms_residuals_fv[j, :] = np.sqrt(np.mean(ar_residuals[:, :, j]**2, axis=0))
    
    return ar_parameters_fv, rms_residuals_fv, ar_parameters, ar_residuals, ar_prediction


def ar_model(x: np.ndarray, order: int) -> np.ndarray:
    """
    Modern Python interface for AR model fitting.
    
    Parameters
    ----------
    x : np.ndarray
        Input time series data. If 2D, AR model is fit to each column.
    order : int
        AR model order.
        
    Returns
    -------
    coefficients : np.ndarray
        AR model coefficients.
    """
    # Convert to format expected by ar_model_shm
    if x.ndim == 1:
        x = x.reshape(-1, 1, 1)
    elif x.ndim == 2:
        x = x.reshape(x.shape[0], x.shape[1], 1)
    
    ar_params_fv, _, ar_params, _, _ = ar_model_shm(x, order)
    
    # Return just the parameters in modern format
    return ar_params[:, :, 0]


def arx_model(
    y: np.ndarray, 
    x: Optional[np.ndarray] = None, 
    na: int = 2, 
    nb: int = 2, 
    nk: int = 1
) -> Dict[str, np.ndarray]:
    """
    Fit ARX (autoregressive with exogenous input) model.
    
    Python equivalent of MATLAB's arxModel_shm function.
    
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
        # Pure AR model
        coeffs = ar_model(y, na)
        return {
            'a': coeffs,
            'b': None,
            'variance': np.var(y)  # Placeholder
        }
    else:
        # ARX model - needs full implementation
        raise NotImplementedError("ARX modeling not yet fully implemented")


def ar_model_order(
    x: np.ndarray, 
    max_order: int = 20, 
    criterion: str = "aic"
) -> int:
    """
    Determine optimal AR model order.
    
    Python equivalent of MATLAB's arModelOrder_shm function.
    
    Parameters
    ----------
    x : np.ndarray
        Input time series.
    max_order : int, optional
        Maximum order to test. Default is 20.
    criterion : str, optional
        Information criterion ('aic', 'bic'). Default is 'aic'.
        
    Returns
    -------
    optimal_order : int
        Optimal AR model order.
    """
    # TODO: Implement model order selection using information criteria
    # This is a placeholder for the actual implementation
    
    n = len(x)
    criteria = np.zeros(max_order)
    
    for p in range(1, max_order + 1):
        # Fit AR model of order p
        coeffs = ar_model(x, p)
        
        # Compute residuals (placeholder)
        residual_var = np.var(x)  # This needs proper residual calculation
        
        # Compute information criterion
        if criterion == "aic":
            criteria[p-1] = n * np.log(residual_var) + 2 * p
        elif criterion == "bic":
            criteria[p-1] = n * np.log(residual_var) + p * np.log(n)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    return np.argmin(criteria) + 1