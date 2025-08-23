"""
Condition-based monitoring specific signal processing functions.

These functions implement specialized algorithms for rotating machinery
analysis and order tracking.
"""

import numpy as np
from scipy import interpolate
from typing import Tuple, Optional, Union
from ..core.preprocessing import filter_shm, window_shm
from ..core.signal_processing import fir1_shm


def ars_tach_shm(
    X: np.ndarray,
    nFilter: Optional[int] = None,
    samplesPerRev: Optional[int] = None,
    gearRatio: float = 1.0
) -> Tuple[np.ndarray, int]:
    """
    Feature Extraction: Resamples signals to angular domain using tachometer.
    
    Resamples a matrix of signals individually in the time domain to a
    specified equally spaced angular domain using a tachometer matrix with
    tachometer signals corresponding to each instance. Primarily used for
    order tracking of signals by resampling a signal to an equally sampled
    angular domain, frequency components that may be smeared due to varying
    shaft speed in the time domain have increased resolution by resampling
    to the angular domain.
    
    NOTE: FIRST CHANNEL OF SIGNAL MATRIX X MUST BE THE TACHOMETER SIGNAL.
    
    .. meta::
        :category: Features - Condition Based Monitoring
        :matlab_equivalent: arsTach_shm
        :complexity: Advanced
        :data_type: Time Series
        :output_type: Features
        :verbose_call: [Angular Resampled Signal Matrix, Samples per Revolution] = Angular Resampling from Tachometer (Signal Matrix, Number of Filter Coefficients, Samples Per Revolution, Gear Ratio)
        
    Parameters
    ----------
    X : array_like, shape (TIME, CHANNELS, INSTANCES)
        Matrix of time series data, first channel must be tachometer signal.
        
        .. gui::
            :widget: data_input
            :description: "Multi-channel time series with tachometer as first channel"
            
    nFilter : int, optional
        Number of filter coefficients. Default is 129.
        
        .. gui::
            :widget: number_input
            :min: 64
            :max: 1024
            :default: 129
            
    samplesPerRev : int, optional
        Number of samples per revolution. If None, uses minimum resolution.
        
        .. gui::
            :widget: number_input
            :min: 64
            :max: 2048
            :default: 512
            
    gearRatio : float, optional
        Gear ratio between tach shaft and target shaft. Default is 1.0.
        
        .. gui::
            :widget: number_input
            :min: 0.1
            :max: 10.0
            :default: 1.0
    
    Returns
    -------
    xARSMatrix : ndarray, shape (SAMPLES, CHANNELS, INSTANCES)
        Angular resampled signal matrix, SAMPLES = samplesPerRev * REVOLUTIONS.
        
        .. gui::
            :plot_type: "line"
            :description: "Angular resampled time series"
            
    samplesPerRev : int
        Samples per revolution in xARSMatrix.
        
        .. gui::
            :widget: text_output
            :description: "Effective samples per revolution"
            
    References
    ----------
    [1] Blough, Jason. Adaptive Resampling - Transforming from the Time
        to Angle Domain.
    """
    # Handle default values
    if nFilter is None:
        nFilter = 129
        
    if gearRatio is None:
        gearRatio = 1.0
    
    # Separate Tach and Accel Data
    tach = X[:, 0:1, :]  # First channel is tachometer
    X_signals = X[:, 1:, :]  # Remaining channels are accelerometers
    
    # Get dimension sizes
    nSignal, nChannel, nInstance = X_signals.shape
    
    xARSMatrix = None
    
    # Process each channel and instance
    for iChannel in range(nChannel):
        for iInstance in range(nInstance):
            xARS, actual_samplesPerRev = _ars_tach_single(
                X_signals[:, iChannel, iInstance],
                tach[:, 0, iInstance], 
                nFilter, 
                samplesPerRev, 
                gearRatio
            )
            
            # Allocate memory for feature space on first iteration
            if iChannel == 0 and iInstance == 0:
                nARS = len(xARS)
                xARSMatrix = np.zeros((nARS, nChannel, nInstance))
                samplesPerRev = actual_samplesPerRev
            
            # Crop signals to same number of revolutions
            if len(xARS) > xARSMatrix.shape[0]:
                xARS = xARS[:xARSMatrix.shape[0]]
                xARSMatrix[:, iChannel, iInstance] = xARS
            elif len(xARS) < xARSMatrix.shape[0]:
                # Crop existing matrix to match shorter signal
                xARSMatrix = xARSMatrix[:len(xARS), :, :]
                xARSMatrix[:, iChannel, iInstance] = xARS
            else:
                xARSMatrix[:, iChannel, iInstance] = xARS
    
    return xARSMatrix, samplesPerRev


def _ars_tach_single(
    X: np.ndarray,
    tach: np.ndarray,
    nFilter: int,
    samplesPerRev: Optional[int],
    gearRatio: float
) -> Tuple[np.ndarray, int]:
    """Single channel angular resampling implementation."""
    
    # Decode tachometer signal to binary
    tach_decoded = _decode_tach(tach)
    
    # Find shaft crossings (zero crossings of tach signal)
    iCross = _find_shaft_crossings(tach_decoded)
    
    if len(iCross) < 2:
        raise ValueError("Insufficient tachometer crossings detected")
    
    # Crop signals to known phase locations
    tach_cropped, X_cropped, nX, iT = _crop_signal(X, tach_decoded, iCross)
    
    # Interpolate shaft angle
    phiS, nT = _interp_shaft_angle(iT, nX, gearRatio)
    
    if gearRatio != 1:
        # Crop to integer number of cycles
        nT = int(np.floor(nT))
        valid_indices = phiS <= (nT * 2 * np.pi)
        if np.any(valid_indices):
            last_valid = np.where(valid_indices)[0][-1]
            X_cropped = X_cropped[:last_valid+1]
            phiS = phiS[:last_valid+1]
    
    # Find minimum samples per revolution
    minSPR = _find_min_spr(phiS)
    
    if samplesPerRev is None:
        samplesPerRev = int(np.floor(minSPR))
    
    # Upsample if necessary
    if minSPR < 2 * samplesPerRev:
        phiS, X_cropped, L = _upsample_signal(phiS, X_cropped, minSPR, samplesPerRev)
        X_cropped = _filter_low_signal(X_cropped, L, nFilter)
        
        # Remove filter delay
        nDelay = int(np.floor((nFilter - 1) / 2))
        X_cropped = X_cropped[nDelay:]
        phiS = phiS[:-nDelay]
        
        # Crop to integer number of cycles
        nT = int(np.floor(phiS[-1] / (2 * np.pi)))
        valid_indices = phiS <= (nT * 2 * np.pi)
        if np.any(valid_indices):
            last_valid = np.where(valid_indices)[0][-1]
            X_cropped = X_cropped[:last_valid+1]
            phiS = phiS[:last_valid+1]
    
    # Angular resampling
    phiA, xARS = _angular_resample_signal(phiS, X_cropped, samplesPerRev, nT)
    xARS = _filter_low_ar_signal(xARS, nFilter)
    
    # Remove filter delay
    nDelay = int(np.floor((nFilter - 1) / 2))
    xARS = xARS[nDelay:]
    phiA = phiA[:-nDelay]
    
    # Downsample to desired SPR
    phiA, xARS = _downsample_to_spr(phiA, xARS)
    
    # Crop to integer number of cycles
    cycles_length = int(np.floor(len(xARS) / samplesPerRev) * samplesPerRev)
    xARS = xARS[:cycles_length]
    
    return xARS, samplesPerRev


def _decode_tach(tach: np.ndarray) -> np.ndarray:
    """Convert tach data to binary data set."""
    meanTach = np.mean(tach)
    tachD = np.zeros_like(tach)
    tachD[tach > meanTach] = 1
    return tachD


def _find_shaft_crossings(tach: np.ndarray) -> np.ndarray:
    """Find zero crossings of tach data."""
    # MATLAB: iCross = find(((tach(1:1:end-1)-tach(2:1:end))/2)==(-0.5))+1;
    # This looks for (tach[i] - tach[i+1])/2 = -0.5, meaning tach[i] - tach[i+1] = -1
    # So tach[i] = 0 and tach[i+1] = 1 (rising edge from 0 to 1)
    diff = (tach[:-1] - tach[1:]) / 2.0
    crossings = np.where(diff == -0.5)[0] + 1
    return crossings


def _crop_signal(X: np.ndarray, tach: np.ndarray, iCross: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Crop data to known phase locations and integer number of rotations."""
    start_idx = iCross[0]
    end_idx = iCross[-1]
    
    X_cropped = X[start_idx:end_idx+1]
    tach_cropped = tach[start_idx:end_idx+1]
    nX = len(X_cropped)
    iT = iCross - start_idx
    
    return tach_cropped, X_cropped, nX, iT


def _interp_shaft_angle(iT: np.ndarray, nSignal: int, gearRatio: float) -> Tuple[np.ndarray, float]:
    """Interpolate shaft angle."""
    n = np.linspace(0, nSignal - 1, nSignal)
    nT = len(iT)
    phi = np.arange(nT) * (2 * np.pi)
    
    # Use scipy interpolation
    f = interpolate.interp1d(iT, phi, kind='cubic', bounds_error=False, fill_value='extrapolate')
    phiS = f(n) * gearRatio
    nT_adjusted = nT * gearRatio
    
    return phiS, nT_adjusted


def _find_min_spr(phiS: np.ndarray) -> float:
    """Find minimum SPR resolution without upsampling."""
    phase_diffs = phiS[1:] - phiS[:-1]
    max_phase_diff = np.max(phase_diffs)
    minSPR = np.floor((2 * np.pi) / max_phase_diff)
    return minSPR


def _upsample_signal(phiS: np.ndarray, X: np.ndarray, minSPR: float, samplesPerRev: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Upsample signal according to desired SPR."""
    M = 2  # Upsample ratio to be decimated later
    L = int(np.ceil((M * samplesPerRev) / minSPR))
    
    nSignal = len(phiS)
    n = np.linspace(0, nSignal - 1, nSignal)
    nL = np.linspace(0, nSignal - 1, (nSignal - 1) * L + 1)
    
    # Use scipy interpolation
    f_phi = interpolate.interp1d(n, phiS, kind='cubic', bounds_error=False, fill_value='extrapolate')
    f_x = interpolate.interp1d(n, X, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    phiSL = f_phi(nL)
    xL = f_x(nL)
    
    return phiSL, xL, L


def _filter_low_signal(X: np.ndarray, L: int, nFilter: int) -> np.ndarray:
    """Low pass filter upsampled signal."""
    Wn = (1 / L) / 2
    beta = 4
    # Use scipy directly since we need to pass the window parameters correctly
    from scipy import signal
    g = signal.firwin(nFilter + 1, Wn, window=('kaiser', beta))
    y = filter_shm(X.reshape(-1, 1, 1), g)
    return y.flatten()


def _angular_resample_signal(phiS: np.ndarray, X: np.ndarray, samplesPerRev: int, nT: float) -> Tuple[np.ndarray, np.ndarray]:
    """Angular resample signal to M x desired SPR."""
    M = 2  # Upsample ratio to be decimated later
    phiA = np.linspace(0, (nT - 1) * (2 * np.pi), int((M * samplesPerRev) * (nT - 1) + 1))
    
    # Use scipy interpolation
    f = interpolate.interp1d(phiS, X, kind='cubic', bounds_error=False, fill_value='extrapolate')
    xARS = f(phiA)
    
    return phiA, xARS


def _filter_low_ar_signal(X: np.ndarray, nFilter: int) -> np.ndarray:
    """Low pass filter angular resampled signal."""
    M = 2  # Upsample ratio to be decimated later
    Wn = (1 / M) / 2
    beta = 4
    # Use scipy directly since we need to pass the window parameters correctly
    from scipy import signal
    g = signal.firwin(nFilter + 1, Wn, window=('kaiser', beta))
    y = filter_shm(X.reshape(-1, 1, 1), g)
    return y.flatten()


def _downsample_to_spr(phiA: np.ndarray, xARS: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample to desired SPR."""
    M = 2  # Decimate
    phiA_down = phiA[::M]
    xARS_down = xARS[::M]
    return phiA_down, xARS_down