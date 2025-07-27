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
    f: np.ndarray,
    is_one_sided: bool = True,
    channel: int = 1,
    **kwargs,
) -> Axes:
    """
    MATLAB-compatible PSD plotting function.

    Plot power spectral density matrix using MATLAB-style interface.

    Parameters
    ----------
    psd_matrix : ndarray, shape (n_freqs, n_channels, n_instances)
        Power spectral density matrix from psd_welch_shm.
    f : ndarray
        Frequency vector in Hz.
    is_one_sided : bool, optional
        Whether PSD is one-sided. Default is True.
    channel : int, optional
        Channel to plot (1-based indexing). Default is 1.
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object containing the plot.
    """
    # Use the existing plot_psd function with MATLAB-style interface
    return plot_psd_shm(
        psd_matrix=psd_matrix, f=f, is_one_sided=is_one_sided, channel=channel, **kwargs
    )


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
