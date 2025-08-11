"""
Spectral analysis plotting functions for SHMTools.

This module provides specialized plotting functions for frequency domain
analysis results with publication-quality formatting.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple, Any
from matplotlib.axes import Axes
from matplotlib.figure import Figure

try:
    from bokeh.plotting import figure, show
    from bokeh.models import ColorBar, LinearColorMapper
    from bokeh.palettes import Viridis256

    HAS_BOKEH = True
except ImportError:
    HAS_BOKEH = False


def plot_scores_shm(scores: np.ndarray,
                   detected_states: np.ndarray,
                   state_names: List[str],
                   threshold: float,
                   use_bar_chart: bool = True,
                   show_legend: bool = True,
                   ax: Optional[Axes] = None) -> Axes:
    """
    Plot damage detection scores with threshold and classification results.
    
    .. meta::
        :category: Auxiliary - Plotting
        :matlab_equivalent: plotScores_shm
        :complexity: Basic
        :data_type: Scores
        :output_type: Plot
        :interactive_plot: True
        :display_name: Plot Detection Scores
        :verbose_call: [Axes Handle] = Plot Detection Scores (Scores, Detected States, State Names, Threshold, Use Bar Chart, Show Legend, Axes Handle)
    
    Parameters
    ----------
    scores : array_like, shape (n_tests,)
        Detection scores for each test case.
        
        .. gui::
            :widget: array_input
            :description: Detection scores array
            
    detected_states : array_like, shape (n_tests,)
        Binary detection results (0=healthy, 1=damaged).
        
        .. gui::
            :widget: array_input
            :description: Detection results (0/1)
            
    state_names : list of str
        Names for the detection states ['Healthy', 'Damaged'].
        
        .. gui::
            :widget: text_list
            :default: ["Healthy", "Damaged"]
            
    threshold : float
        Detection threshold value.
        
        .. gui::
            :widget: number_input
            :min: 0.0
            :default: 1.0
            
    use_bar_chart : bool, optional
        If True, use bar chart. If False, use line plot. Default is True.
        
        .. gui::
            :widget: checkbox
            :default: true
            
    show_legend : bool, optional
        Whether to show legend. Default is True.
        
        .. gui::
            :widget: checkbox
            :default: true
            
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes handle for the plot.
    
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.plotting import plot_scores_shm
    >>> 
    >>> # Generate example data
    >>> scores = np.array([1.2, 1.5, 1.3, 2.8, 3.2, 4.1])
    >>> detected = np.array([0, 0, 0, 1, 1, 1])
    >>> threshold = 2.0
    >>> 
    >>> # Plot results
    >>> ax = plot_scores_shm(scores, detected, ['Healthy', 'Damaged'], threshold)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    n_tests = len(scores)
    test_indices = np.arange(n_tests)
    
    # Color mapping
    colors = ['green' if state == 0 else 'red' for state in detected_states]
    
    if use_bar_chart:
        # Bar chart
        bars = ax.bar(test_indices, scores, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05*max(scores),
                   f'{score:.2f}', ha='center', va='bottom')
    else:
        # Line plot with markers
        ax.plot(test_indices, scores, 'o-', markersize=8, linewidth=2)
        for i, (score, state) in enumerate(zip(scores, detected_states)):
            color = 'green' if state == 0 else 'red'
            ax.plot(i, score, 'o', color=color, markersize=10, alpha=0.7)
    
    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold = {threshold:.2f}')
    
    # Formatting
    ax.set_xlabel('Test Case')
    ax.set_ylabel('Detection Score')
    ax.set_title('Damage Detection Results')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis ticks
    ax.set_xticks(test_indices)
    ax.set_xticklabels([f'Test {i+1}' for i in test_indices])
    
    # Add legend if requested
    if show_legend:
        # Create custom legend elements
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label=state_names[0]),
            Patch(facecolor='red', alpha=0.7, label=state_names[1]),
            plt.Line2D([0], [0], color='red', linestyle='--', 
                      label=f'Threshold = {threshold:.2f}')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
    
    # Add statistics text box
    n_healthy = np.sum(detected_states == 0)
    n_damaged = np.sum(detected_states == 1)
    
    stats_text = f"""Statistics:
{state_names[0]}: {n_healthy}/{n_tests} ({n_healthy/n_tests*100:.0f}%)
{state_names[1]}: {n_damaged}/{n_tests} ({n_damaged/n_tests*100:.0f}%)"""
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return ax


def plot_psd_shm(
    psd_matrix: np.ndarray,
    channel: int = 1,
    is_one_sided: bool = True,
    f: Optional[np.ndarray] = None,
    use_colormap: bool = False,
    use_subplots: bool = False,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Union[Axes, List[Axes]]:
    """
    Plot power spectral density with various visualization options.

    Python equivalent of MATLAB's plotPSD_shm function. Provides flexible
    PSD visualization with colormap support for multi-instance data.

    .. meta::
        :category: Plotting - Spectral
        :matlab_equivalent: plotPSD_shm
        :complexity: Basic
        :data_type: Frequency Domain
        :output_type: Plot
        :interactive_plot: True

    Parameters
    ----------
    psd_matrix : array_like, shape (n_freqs, n_channels, n_instances)
        Power spectral density matrix from psd_welch.

        .. gui::
            :widget: data_input
            :description: "PSD matrix from spectral analysis"

    channel : int, optional, default=1
        Channel index to plot (1-based indexing to match MATLAB).

        .. gui::
            :widget: numeric_input
            :min: 1
            :max: 10
            :description: "Channel number to plot"

    is_one_sided : bool, optional, default=True
        Whether PSD is one-sided (positive frequencies only).

        .. gui::
            :widget: checkbox
            :description: "One-sided PSD"

    f : array_like, optional
        Frequency vector in Hz. If None, uses normalized frequency.

        .. gui::
            :widget: data_input
            :description: "Frequency vector"

    use_colormap : bool, optional, default=False
        If True, creates colormap plot for multi-instance visualization.

        .. gui::
            :widget: checkbox
            :description: "Use colormap visualization"

    use_subplots : bool, optional, default=False
        If True, creates separate subplots for different views.

        .. gui::
            :widget: checkbox
            :description: "Create multiple subplots"

    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure.

    **kwargs
        Additional keyword arguments passed to matplotlib plotting functions.

    Returns
    -------
    axes : matplotlib.axes.Axes or list of Axes
        Axes object(s) containing the plots.

        .. gui::
            :plot_type: "line"
            :x_axis: "f"
            :y_axis: "psd"
            :log_scale: "y"
            :xlabel: "Frequency (Hz)"
            :ylabel: "PSD (dB)"

    Examples
    --------
    Basic PSD plot:

    >>> import numpy as np
    >>> from shmtools.core import psd_welch
    >>> from shmtools.plotting import plot_psd
    >>>
    >>> # Generate test signal
    >>> fs = 1000
    >>> t = np.linspace(0, 1, fs, endpoint=False)
    >>> x = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(fs)
    >>>
    >>> # Compute and plot PSD
    >>> f, psd = psd_welch(x, fs=fs)
    >>> ax = plot_psd(psd[:, :, np.newaxis], f=f, channel=1)

    Multi-instance colormap visualization:

    >>> # Multiple signal instances
    >>> x_multi = np.random.randn(1000, 1, 20)  # 20 instances
    >>> f, psd = psd_welch(x_multi, fs=fs)
    >>> ax = plot_psd(psd, f=f, use_colormap=True, use_subplots=True)
    """
    # Convert to 0-based indexing
    channel_idx = channel - 1

    # Handle frequency vector
    if f is None:
        n_freqs = psd_matrix.shape[0]
        if is_one_sided:
            f = np.linspace(0, 0.5, n_freqs)
        else:
            f = np.linspace(-0.5, 0.5, n_freqs)

    # Extract channel data
    if psd_matrix.ndim == 3:
        psd_data = psd_matrix[:, channel_idx, :]
    elif psd_matrix.ndim == 2:
        psd_data = psd_matrix[:, channel_idx : channel_idx + 1]
    else:
        psd_data = psd_matrix[:, np.newaxis]

    # Convert to dB
    psd_db = 10 * np.log10(np.maximum(psd_data, 1e-12))

    if use_colormap and psd_data.shape[1] > 1:
        return _plot_psd_colormap(f, psd_db, use_subplots, ax, **kwargs)
    else:
        return _plot_psd_lines(f, psd_db, ax, **kwargs)


def _plot_psd_colormap(
    f: np.ndarray, psd_db: np.ndarray, use_subplots: bool, ax: Optional[Axes], **kwargs
) -> List[Axes]:
    """
    Create colormap visualization of PSD data.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    psd_db : ndarray
        PSD data in dB.
    use_subplots : bool
        Whether to create multiple subplots.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    axes : list of Axes
        List of axes objects.
    """
    if use_subplots:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        ax1, ax2 = axes
    else:
        if ax is None:
            fig, ax1 = plt.subplots(figsize=(10, 6))
        else:
            ax1 = ax
        ax2 = None

    # Main colormap plot
    extent = [0, psd_db.shape[1], f[0], f[-1]]
    im = ax1.imshow(
        psd_db, aspect="auto", origin="lower", extent=extent, cmap="viridis", **kwargs
    )

    ax1.set_xlabel("Instance Number")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("PSD Matrix (dB)")

    # Add colorbar
    plt.colorbar(im, ax=ax1, label="PSD (dB)")

    # Optional second subplot showing average PSD
    if ax2 is not None:
        mean_psd = np.mean(psd_db, axis=1)
        std_psd = np.std(psd_db, axis=1)

        ax2.plot(f, mean_psd, "b-", linewidth=2, label="Mean")
        ax2.fill_between(
            f,
            mean_psd - std_psd,
            mean_psd + std_psd,
            alpha=0.3,
            color="blue",
            label="±1 STD",
        )

        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("PSD (dB)")
        ax2.set_title("Average PSD with Standard Deviation")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    return [ax1, ax2] if ax2 is not None else [ax1]


def _plot_psd_lines(
    f: np.ndarray, psd_db: np.ndarray, ax: Optional[Axes], **kwargs
) -> Axes:
    """
    Create line plot visualization of PSD data.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    psd_db : ndarray
        PSD data in dB.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    ax : Axes
        Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each instance
    for i in range(psd_db.shape[1]):
        if psd_db.shape[1] == 1:
            label = "PSD"
            color = "blue"
        else:
            label = f"Instance {i+1}"
            color = None

        ax.plot(f, psd_db[:, i], label=label, color=color, **kwargs)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title("Power Spectral Density")
    ax.grid(True, alpha=0.3)

    if psd_db.shape[1] > 1 and psd_db.shape[1] <= 10:
        ax.legend()

    return ax


def plot_spectrogram_shm(
    f: np.ndarray, t: np.ndarray, Sxx: np.ndarray, ax: Optional[Axes] = None, **kwargs
) -> Axes:
    """
    Plot spectrogram with proper formatting.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    t : ndarray
        Time vector.
    Sxx : ndarray
        Spectrogram matrix.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    ax : Axes
        Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Convert to dB
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-12))

    # Create spectrogram plot
    im = ax.pcolormesh(t, f, Sxx_db, cmap="viridis", **kwargs)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Power (dB)")

    return ax


def plotPSD_shm(
    psd_matrix: np.ndarray,
    channel_index: int,
    is1sided: bool,
    freq_vector: np.ndarray,
    use_db_scale: bool = True,
    plot_average: bool = True,
    axes_handle: Optional[Axes] = None,
) -> Axes:
    """
    MATLAB-compatible PSD plotting function.

    Plot power spectral density matrix using MATLAB-style interface.
    
    .. meta::
        :category: Plotting - Spectral Analysis
        :matlab_equivalent: plotPSD_shm
        :complexity: Basic
        :data_type: PSD Matrix
        :output_type: Plot

    Parameters
    ----------
    psd_matrix : ndarray, shape (n_freqs, n_channels, n_instances)
        Power spectral density matrix from psd_welch_shm.
    channel_index : int
        Channel to plot (1-based indexing like MATLAB).
    is1sided : bool
        Whether PSD is one-sided.
    freq_vector : ndarray
        Frequency vector in Hz.
    use_db_scale : bool, optional
        Whether to use dB scale. Default is True.
    plot_average : bool, optional
        Whether to plot average over instances. Default is True.
    axes_handle : Axes, optional
        Existing axes handle.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object containing the plot.
    """
    if axes_handle is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax = axes_handle
    
    # Convert to 0-based indexing
    channel_idx = channel_index - 1
    
    if channel_idx >= psd_matrix.shape[1]:
        raise ValueError(f"Channel index {channel_index} exceeds available channels ({psd_matrix.shape[1]})")
    
    # Get channel data
    if psd_matrix.ndim == 3:
        channel_data = psd_matrix[:, channel_idx, :]
    else:
        channel_data = psd_matrix[:, channel_idx:channel_idx+1]
    
    if plot_average and channel_data.ndim == 2:
        # Average over instances
        psd_plot = np.mean(channel_data, axis=1)
    else:
        # Plot individual instances or single instance
        if channel_data.ndim == 2:
            psd_plot = channel_data
        else:
            psd_plot = channel_data
    
    # Convert to dB if requested
    if use_db_scale:
        psd_plot = 10 * np.log10(np.maximum(psd_plot, 1e-12))
        ylabel = "Power (dB)"
    else:
        ylabel = "Power"
    
    # Plot
    if psd_plot.ndim == 1:
        ax.plot(freq_vector, psd_plot)
    else:
        # Plot multiple instances
        for i in range(psd_plot.shape[1]):
            ax.plot(freq_vector, psd_plot[:, i], alpha=0.7)
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    
    return ax


def plot_time_freq(
    tf_matrix: np.ndarray,
    f: Optional[np.ndarray] = None,
    t: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    title: str = "Time-Frequency Analysis",
    **kwargs,
) -> Axes:
    """
    General time-frequency plotting function.

    Python equivalent of MATLAB's plotTimeFreq_shm function.

    Parameters
    ----------
    tf_matrix : ndarray
        Time-frequency matrix.
    f : ndarray, optional
        Frequency vector.
    t : ndarray, optional
        Time vector.
    ax : Axes, optional
        Existing axes.
    title : str
        Plot title.

    Returns
    -------
    ax : Axes
        Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Handle default frequency and time vectors
    if f is None:
        f = np.arange(tf_matrix.shape[0])
    if t is None:
        t = np.arange(tf_matrix.shape[1])

    # Convert to dB if values are large
    if np.max(tf_matrix) > 100:
        tf_db = 10 * np.log10(np.maximum(np.abs(tf_matrix), 1e-12))
        label = "Power (dB)"
    else:
        tf_db = np.abs(tf_matrix)
        label = "Magnitude"

    # Create plot
    im = ax.pcolormesh(t, f, tf_db, cmap="viridis", **kwargs)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)

    # Add colorbar
    plt.colorbar(im, ax=ax, label=label)

    return ax


def plot_roc_shm(
    tpr: np.ndarray,
    fpr: np.ndarray,
    scaling: str = 'linear',
    ax: Optional[Axes] = None
) -> Axes:
    """
    Plot receiver operating characteristic curve.
    
    .. meta::
        :category: Plotting - Classification
        :matlab_equivalent: plotROC_shm
        :complexity: Basic
        :data_type: Performance Metrics
        :output_type: Plot
        :display_name: ROC Curve Plot
        :verbose_call: [Axes Handle] = Plot Receiver Operating Characteristic Curve (True Positive Rate, False Positive Rate, Scaling, Axes Handle)
        
    Plot receiver operating characteristic curve(s). A curve is plotted
    for each column of TPR and FPR.
    
    Parameters
    ----------
    tpr : ndarray, shape (points, curves)
        Matrix composed of true positive rates between 0 and 1.
        
        .. gui::
            :widget: data_input
            :description: True positive rates for ROC curve
            
    fpr : ndarray, shape (points, curves)
        Matrix composed of false positive rates between 0 and 1.
        
        .. gui::
            :widget: data_input
            :description: False positive rates for ROC curve
            
    scaling : str, optional
        Axis scaling. Options: 'linear', 'logx', 'logy', 'logxy', 'normal'
        Default is 'linear'.
        
        .. gui::
            :widget: select
            :options: ["linear", "logx", "logy", "logxy", "normal"]
            :default: "linear"
            :description: Axis scaling type
            
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes handle. If None, creates new figure and axes.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes handle for the plot.
        
        .. gui::
            :plot_type: "line"
            :x_axis: "fpr"
            :y_axis: "tpr"
            :xlabel: "False Positive Rate"
            :ylabel: "True Positive Rate"
            
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.plotting.spectral_plots import plot_roc_shm
    >>> 
    >>> # Generate example ROC data
    >>> n_points = 100
    >>> fpr = np.linspace(0, 1, n_points)
    >>> # Perfect classifier
    >>> tpr_perfect = np.concatenate([np.zeros(50), np.ones(50)])
    >>> # Random classifier
    >>> tpr_random = fpr
    >>> # Good classifier
    >>> tpr_good = np.sqrt(fpr)
    >>> 
    >>> # Combine into matrix
    >>> tpr_matrix = np.column_stack([tpr_perfect, tpr_random, tpr_good])
    >>> fpr_matrix = np.column_stack([fpr, fpr, fpr])
    >>> 
    >>> # Plot ROC curves
    >>> ax = plot_roc_shm(tpr_matrix, fpr_matrix)
    >>> ax.legend(['Perfect', 'Random', 'Good'])
    
    References
    ----------
    [1] Fawcett, T. (2006). An introduction to ROC analysis. Pattern 
    recognition letters, 27(8), 861-874.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Ensure arrays are 2D
    if tpr.ndim == 1:
        tpr = tpr.reshape(-1, 1)
    if fpr.ndim == 1:
        fpr = fpr.reshape(-1, 1)
    
    # Plot each curve
    n_curves = tpr.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_curves))
    
    for i in range(n_curves):
        ax.plot(fpr[:, i], tpr[:, i], color=colors[i], linewidth=2)
    
    # Add diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    # Set scaling
    if scaling == 'logx':
        ax.set_xscale('log')
    elif scaling == 'logy':
        ax.set_yscale('log')  
    elif scaling == 'logxy':
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif scaling == 'normal':
        # Normal probability paper scale - approximation
        ax.set_xscale('logit')
        ax.set_yscale('logit')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return ax


def plot_time_freq_shm(
    time_freq_matrix: np.ndarray,
    channel_index: Optional[int] = None,
    instance: Optional[int] = None,
    time_vector: Optional[np.ndarray] = None,
    freq_vector: Optional[np.ndarray] = None,
    use_db_scale: Optional[bool] = None,
    ax: Optional[Axes] = None
) -> Axes:
    """
    Create a time-frequency plot.
    
    .. meta::
        :category: Plotting - Spectral
        :matlab_equivalent: plotTimeFreq_shm
        :complexity: Basic
        :data_type: Time-Frequency
        :output_type: Plot
        :display_name: Time-Frequency Plot
        :verbose_call: [Axes Handle] = Plot Time-Frequency (Time-Frequency Data, Channel Index, Instance Index, Time Vector, Frequency Vector, Use dB Magnitude, Axes Handle)
        
    Create a time-frequency plot from output of lpcSpectrogram_shm,
    stft_shm or dwvd_shm.
    
    Parameters
    ----------
    time_freq_matrix : ndarray, shape (freq, time, channels, instances)
        Time-frequency data matrix.
        
        .. gui::
            :widget: data_input
            :description: Time-frequency matrix from spectral analysis
            
    channel_index : int, optional
        Channel to plot (1-based indexing). Default is 1.
        
        .. gui::
            :widget: number_input
            :min: 1
            :description: Channel index to plot
            
    instance : int, optional
        Instance to plot (1-based indexing). Default is 1.
        
        .. gui::
            :widget: number_input
            :min: 1
            :description: Instance index to plot
            
    time_vector : ndarray, optional
        Sampling times corresponding to time_freq_matrix.
        If None, uses sample indices.
        
        .. gui::
            :widget: array_input
            :description: Time vector (seconds)
            
    freq_vector : ndarray, optional
        Frequency values corresponding to time_freq_matrix.
        If None, uses normalized frequency.
        
        .. gui::
            :widget: array_input
            :description: Frequency vector (Hz)
            
    use_db_scale : bool, optional
        Use dB for magnitude scale instead of linear. Default is True.
        
        .. gui::
            :widget: checkbox
            :default: True
            :description: Use dB scale for magnitude
            
    ax : matplotlib.axes.Axes, optional
        Handle for plotting axes. If None, creates new figure and axes.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes handle for the plot.
        
        .. gui::
            :plot_type: "heatmap"
            :x_axis: "time_vector"
            :y_axis: "freq_vector"
            :xlabel: "Time (s)"
            :ylabel: "Frequency (Hz)"
            
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.spectral import dwvd_shm
    >>> from shmtools.plotting.spectral_plots import plot_time_freq_shm
    >>> 
    >>> # Generate chirp signal
    >>> fs = 1000
    >>> t = np.linspace(0, 1, fs)
    >>> signal = np.sin(2*np.pi*(50 + 50*t)*t)  # 50-100 Hz chirp
    >>> X = signal.reshape(-1, 1, 1)
    >>> 
    >>> # Compute time-frequency representation
    >>> tf_matrix, f, t = dwvd_shm(X, fs=fs)
    >>> 
    >>> # Plot time-frequency
    >>> ax = plot_time_freq_shm(tf_matrix, time_vector=t, freq_vector=f)
    
    See Also
    --------
    dwvd_shm : Discrete Wigner-Ville Distribution
    lpc_spectrogram_shm : Linear Predictive Coding spectrogram
    stft_shm : Short-Time Fourier Transform
    """
    # Set defaults
    if channel_index is None:
        channel_index = 1
    if instance is None:
        instance = 1
    if use_db_scale is None:
        use_db_scale = True
    
    # Convert to 0-based indexing
    channel_idx = channel_index - 1
    instance_idx = instance - 1
    
    # Extract data for specified channel and instance
    if time_freq_matrix.ndim == 4:
        data = time_freq_matrix[:, :, channel_idx, instance_idx]
    elif time_freq_matrix.ndim == 3:
        data = time_freq_matrix[:, :, min(channel_idx, time_freq_matrix.shape[2]-1)]
    else:
        data = time_freq_matrix
    
    # Handle time and frequency vectors
    use_samples = False
    if time_vector is None:
        time_vector = np.arange(data.shape[1])
        use_samples = True
        
    if freq_vector is None:
        freq_vector = np.linspace(0, 0.5, data.shape[0])  # Normalized frequency
        freq_label = 'Normalized Frequency'
    else:
        freq_label = 'Frequency (Hz)'
    
    # Convert to magnitude and apply scaling
    magnitude = np.abs(data)
    if use_db_scale:
        plot_data = 10 * np.log10(np.maximum(magnitude, 1e-12))
        cbar_label = 'Magnitude (dB)'
    else:
        plot_data = magnitude
        cbar_label = 'Magnitude'
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create time-frequency plot using imshow for better compatibility
    extent = [time_vector[0], time_vector[-1], freq_vector[0], freq_vector[-1]]
    im = ax.imshow(plot_data, aspect='auto', origin='lower', extent=extent, 
                   cmap='viridis', interpolation='nearest')
    
    # Set labels
    if use_samples:
        ax.set_xlabel('Time (samples)')
    else:
        ax.set_xlabel('Time (s)')
    ax.set_ylabel(freq_label)
    ax.set_title(f'Time-Frequency Analysis - Channel {channel_index}, Instance {instance}')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label=cbar_label)
    
    return ax


def plot_scalogram_shm(
    scalo_matrix: np.ndarray,
    channel_index: Optional[int] = None,
    instance: Optional[int] = None,
    time_vector: Optional[np.ndarray] = None,
    freq_vector: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None
) -> Axes:
    """
    Create a scalogram plot.
    
    .. meta::
        :category: Plotting - Spectral
        :matlab_equivalent: plotScalogram_shm
        :complexity: Basic
        :data_type: Time-Frequency
        :output_type: Plot
        :display_name: Scalogram Plot
        :verbose_call: [Axes Handle] = Plot Scalogram (Scalogram Data, Channel Index, Instance Index, Time Vector, Frequency Vector, Axes Handle)
        
    Create a time-frequency plot from output of cwt_scalogram_shm.
    
    Parameters
    ----------
    scalo_matrix : ndarray, shape (n_scale, time, channels, instances)
        Scalogram data matrix.
        
        .. gui::
            :widget: data_input
            :description: Scalogram matrix from CWT analysis
            
    channel_index : int, optional
        Channel to plot (1-based indexing). Default is 1.
        
        .. gui::
            :widget: number_input
            :min: 1
            :description: Channel index to plot
            
    instance : int, optional
        Instance to plot (1-based indexing). Default is 1.
        
        .. gui::
            :widget: number_input
            :min: 1
            :description: Instance index to plot
            
    time_vector : ndarray, optional
        Sampling times corresponding to scalo_matrix.
        If None, uses sample indices.
        
        .. gui::
            :widget: array_input
            :description: Time vector (seconds)
            
    freq_vector : ndarray, optional
        Frequency vector corresponding to scales in scalo_matrix.
        If None, uses scale indices.
        
        .. gui::
            :widget: array_input
            :description: Frequency vector (Hz)
            
    ax : matplotlib.axes.Axes, optional
        Handle for plotting axes. If None, creates new figure and axes.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes handle for the plot.
        
        .. gui::
            :plot_type: "heatmap"
            :x_axis: "time_vector"
            :y_axis: "freq_vector"
            :xlabel: "Time (s)"
            :ylabel: "Frequency (Hz)"
            
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.core.spectral import cwt_scalogram_shm
    >>> from shmtools.plotting.spectral_plots import plot_scalogram_shm
    >>> 
    >>> # Generate test signal with frequency variation
    >>> fs = 1000
    >>> t = np.linspace(0, 2, 2*fs)
    >>> # Chirp signal from 10 to 100 Hz
    >>> signal = np.sin(2*np.pi*(10 + 45*t)*t)
    >>> X = signal.reshape(-1, 1, 1)
    >>> 
    >>> # Compute scalogram
    >>> scalo, f, t_scalo = cwt_scalogram_shm(X, fs=fs, f_min=5, f_max=150, n_scale=64)
    >>> 
    >>> # Plot scalogram
    >>> ax = plot_scalogram_shm(scalo, time_vector=t_scalo, freq_vector=f)
    
    See Also
    --------
    cwt_scalogram_shm : Continuous Wavelet Transform scalogram
    """
    # Set defaults
    if channel_index is None:
        channel_index = 1
    if instance is None:
        instance = 1
    
    # Convert to 0-based indexing
    channel_idx = channel_index - 1
    instance_idx = instance - 1
    
    # Extract data for specified channel and instance
    if scalo_matrix.ndim == 4:
        data = scalo_matrix[:, :, channel_idx, instance_idx]
    elif scalo_matrix.ndim == 3:
        data = scalo_matrix[:, :, min(channel_idx, scalo_matrix.shape[2]-1)]
    else:
        data = scalo_matrix
    
    # Handle time and frequency vectors
    if time_vector is None:
        time_vector = np.arange(data.shape[1])
        time_label = 'Time (samples)'
    else:
        time_label = 'Time (s)'
        
    if freq_vector is None:
        freq_vector = np.arange(data.shape[0])
        freq_label = 'Scale'
    else:
        freq_label = 'Frequency (Hz)'
    
    # Convert to dB scale
    plot_data = 10 * np.log10(np.maximum(data, 1e-12))
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scalogram plot using imshow for better compatibility
    extent = [time_vector[0], time_vector[-1], freq_vector[0], freq_vector[-1]]
    im = ax.imshow(plot_data, aspect='auto', origin='lower', extent=extent, 
                   cmap='viridis', interpolation='nearest')
    
    # Set labels and title
    ax.set_xlabel(time_label)
    ax.set_ylabel(freq_label)
    ax.set_title(f'Scalogram - Channel {channel_index}, Instance {instance}')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Power (dB)')
    
    return ax


def plot_features_shm(
    feature_vectors: np.ndarray,
    instance_indices: Optional[np.ndarray] = None,
    feature_indices: Optional[np.ndarray] = None,
    subplot_titles: Optional[List[str]] = None,
    subplot_ylabels: Optional[List[str]] = None,
    axes_handle=None
):
    """
    Plot feature vectors as subplots for each feature.
    
    .. meta::
        :category: Plotting - Feature Visualization
        :matlab_equivalent: plotFeatures_shm
        :complexity: Basic
        :data_type: Features
        :output_type: Figure
        :display_name: Plot Features
        :verbose_call: [Axes Handle] = Plot Features (Feature Vectors, Instances to Plot, Features to Plot, Titles for Subplots, Y-Axis Labels for Subplots, Axes Handle)
        
    Plot feature vectors or a set of samples from feature vectors. Each
    feature uses its own subplot. This is useful for visualizing the
    distribution and characteristics of damage-sensitive features across
    different structural states or conditions.
    
    Parameters
    ----------
    feature_vectors : ndarray, shape (instances, features)
        Feature matrix where each row is a feature vector from one instance.
        
        .. gui::
            :widget: file_upload
            :formats: [".csv", ".mat", ".npy"]
            :description: Feature vectors to visualize
            
    instance_indices : ndarray, optional
        List of indices of instances to be plotted. If None, all instances
        are plotted.
        
        .. gui::
            :widget: array_input
            :description: Subset of instances to plot
            
    feature_indices : ndarray, optional  
        List of indices of features to be plotted. If None, all features
        are plotted. Maximum of 20 features recommended for readability.
        
        .. gui::
            :widget: array_input
            :description: Subset of features to plot
            
    subplot_titles : list of str, optional
        Titles for each subplot. Should match the order of feature_indices.
        If None, defaults to "Feature #".
        
        .. gui::
            :widget: text_array
            :description: Custom titles for each feature subplot
            
    subplot_ylabels : list of str, optional
        Y-axis labels for each subplot. Should match the order of feature_indices.
        
        .. gui::
            :widget: text_array
            :description: Custom y-axis labels for each feature subplot
            
    axes_handle : matplotlib axes, optional
        Existing axes to plot on. If None, creates new figure.
    
    Returns
    -------
    axes_handle : matplotlib axes
        Handle to the created or modified axes.
        
    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.plotting.spectral_plots import plot_features_shm
    >>> 
    >>> # Generate sample feature vectors
    >>> np.random.seed(42)
    >>> baseline_features = np.random.normal(0, 1, (50, 4))  # 50 baseline instances, 4 features
    >>> damage_features = np.random.normal(1, 1.2, (20, 4))  # 20 damage instances, 4 features
    >>> features = np.vstack([baseline_features, damage_features])
    >>> 
    >>> # Plot all features
    >>> feature_names = ['RMS', 'Kurtosis', 'Crest Factor', 'Peak Factor']
    >>> plot_features_shm(features, subplot_titles=feature_names)
    >>> 
    >>> # Plot subset of features for specific instances
    >>> plot_features_shm(features, 
    >>>                   instance_indices=np.arange(0, 70, 5),  # Every 5th instance
    >>>                   feature_indices=[0, 2],  # Only RMS and Crest Factor
    >>>                   subplot_titles=['RMS', 'Crest Factor'])
    
    References
    ----------
    This function replicates the plotting behavior of plotFeatures_shm.m
    from the MATLAB SHMTools library for visualizing feature distributions.
    """
    import matplotlib.pyplot as plt
    
    # Handle default parameters
    n_instances, n_features = feature_vectors.shape
    
    if instance_indices is None:
        instance_indices = np.arange(n_instances)
    
    if feature_indices is None:  
        feature_indices = np.arange(min(n_features, 20))  # Limit to 20 features max
    
    n_selected_features = len(feature_indices)
    
    if subplot_titles is None:
        subplot_titles = [f'Feature {i+1}' for i in feature_indices]
    
    if subplot_ylabels is None:
        subplot_ylabels = [None] * n_selected_features
    
    # Create figure if no axes provided
    if axes_handle is None:
        # Determine subplot layout
        n_cols = min(4, n_selected_features)  # Max 4 columns
        n_rows = int(np.ceil(n_selected_features / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        
        # Handle single subplot case
        if n_selected_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()
    else:
        axes = [axes_handle]
    
    # Plot each selected feature
    for i, feat_idx in enumerate(feature_indices):
        if i >= len(axes):
            break
            
        ax = axes[i] if len(axes) > 1 else axes[0]
        
        # Extract feature values for selected instances
        feature_values = feature_vectors[instance_indices, feat_idx]
        
        # Create simple line plot
        ax.plot(instance_indices, feature_values, 'b.-', markersize=4, linewidth=1)
        ax.grid(True, alpha=0.3)
        
        # Set labels and title
        ax.set_xlabel('Instance Index')
        if subplot_ylabels[i] is not None:
            ax.set_ylabel(subplot_ylabels[i])
        else:
            ax.set_ylabel('Feature Value')
        
        ax.set_title(subplot_titles[i])
        
        # Auto-scale with some padding
        y_range = np.ptp(feature_values)
        if y_range > 0:
            y_margin = y_range * 0.1
            ax.set_ylim(np.min(feature_values) - y_margin, 
                       np.max(feature_values) + y_margin)
    
    # Hide unused subplots
    if axes_handle is None and n_selected_features < len(axes):
        for j in range(n_selected_features, len(axes)):
            axes[j].set_visible(False)
    
    plt.tight_layout()
    
    return axes if len(axes) > 1 else axes[0]
