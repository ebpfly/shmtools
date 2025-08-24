"""
Spatial damage analysis utilities for structural health monitoring.

This module provides functions for analyzing damage indicators across
sensor arrays to localize damage in structures.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import matplotlib.pyplot as plt
from ..classification import learn_mahalanobis_shm, score_mahalanobis_shm


def compute_channel_wise_damage_indicators(
    features: np.ndarray,
    states: np.ndarray,
    undamaged_states: List[int],
    n_channels: int,
    features_per_channel: int,
    method: str = "mahalanobis",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Compute damage indicators for each channel separately.

    This function analyzes each sensor channel independently to identify
    which channels show the strongest response to damage, helping to
    localize damage in the structure.

    .. meta::
        :category: Utilities - Spatial Analysis
        :complexity: Intermediate
        :data_type: Features
        :output_type: Damage Indicators
        :display_name: Channel-wise Damage Indicators

    Parameters
    ----------
    features : array_like
        Feature matrix where features from different channels are concatenated.
        Shape: (INSTANCES, TOTAL_FEATURES)

    states : array_like
        State labels for each instance. Shape: (INSTANCES,)

    undamaged_states : list
        List of state values representing undamaged conditions.

    n_channels : int
        Number of sensor channels.

    features_per_channel : int
        Number of features per channel.

    method : str, optional
        Detection method to use. Currently supports 'mahalanobis'.
        Default is 'mahalanobis'.

        .. gui::
            :widget: dropdown
            :options: ["mahalanobis"]
            :default: "mahalanobis"

    Returns
    -------
    damage_indicators : ndarray
        Damage indicators for each state and channel.
        Shape: (N_STATES, N_CHANNELS)

    models : list
        List of trained models for each channel.

    Examples
    --------
    >>> import numpy as np
    >>> from shmtools.utils import compute_channel_wise_damage_indicators
    >>>
    >>> # Create sample features (170 instances, 60 features = 4 channels × 15 features)
    >>> features = np.random.randn(170, 60)
    >>> states = np.repeat(np.arange(1, 18), 10)  # 17 states, 10 instances each
    >>> undamaged_states = list(range(1, 10))
    >>>
    >>> dis, models = compute_channel_wise_damage_indicators(
    ...     features, states, undamaged_states, n_channels=4, features_per_channel=15
    ... )
    >>> print(f"Damage indicators shape: {dis.shape}")
    """
    features = np.asarray(features)
    states = np.asarray(states)

    # Get unique states and sort them
    unique_states = np.unique(states)
    n_states = len(unique_states)

    # Initialize damage indicators array
    damage_indicators = np.zeros((n_states, n_channels))
    models = []

    # Process each channel separately
    for channel in range(n_channels):

        # Extract features for this channel
        start_idx = channel * features_per_channel
        end_idx = start_idx + features_per_channel
        channel_features = features[:, start_idx:end_idx]

        # Prepare training data: use first 9 instances from each undamaged state
        # This follows the MATLAB example pattern
        train_data_list = []
        for state in undamaged_states:
            state_mask = states == state
            state_instances = np.where(state_mask)[0]

            # Take first 9 instances from this state (MATLAB: j*10-9:j*10-1)
            if len(state_instances) >= 9:
                train_indices = state_instances[:9]
                train_data_list.append(channel_features[train_indices])

        # Combine training data
        if train_data_list:
            train_features = np.vstack(train_data_list)
        else:
            raise ValueError(f"No training data available for channel {channel+1}")

        # Train model
        if method == "mahalanobis":
            model = learn_mahalanobis_shm(train_features)
        else:
            raise ValueError(f"Unknown method: {method}")

        models.append(model)

        # Score each state (using 10th instance from each state)
        for i, state in enumerate(unique_states):
            state_mask = states == state
            state_instances = np.where(state_mask)[0]

            # Use the 10th instance (index 9) if available, otherwise use the last
            if len(state_instances) >= 10:
                test_idx = state_instances[9]  # MATLAB: 10*(1:17)
            else:
                test_idx = state_instances[-1]

            test_features = channel_features[test_idx : test_idx + 1]  # Keep 2D

            # Score using trained model
            if method == "mahalanobis":
                score = score_mahalanobis_shm(test_features, model)[0]
            else:
                raise ValueError(f"Unknown method: {method}")

            # Store damage indicator (negative score to match MATLAB convention)
            damage_indicators[i, channel] = -score

    return damage_indicators, models


def plot_damage_indicators(
    damage_indicators: np.ndarray,
    channel_names: Optional[List[str]] = None,
    state_labels: Optional[np.ndarray] = None,
    undamaged_states: Optional[List[int]] = None,
    title: str = "Channel-wise Damage Indicators",
) -> None:
    """
    Plot damage indicators for each channel in a subplot layout.

    Parameters
    ----------
    damage_indicators : array_like
        Damage indicators matrix, shape (N_STATES, N_CHANNELS).

    channel_names : list, optional
        Names for each channel. If None, uses "Channel X".

    state_labels : array_like, optional
        Labels for each state. If None, uses state numbers.

    undamaged_states : list, optional
        List of undamaged state indices for color coding.

    title : str, optional
        Overall plot title.

    Examples
    --------
    >>> plot_damage_indicators(
    ...     damage_indicators,
    ...     channel_names=['Channel 2', 'Channel 3', 'Channel 4', 'Channel 5'],
    ...     undamaged_states=list(range(9))
    ... )
    """
    n_states, n_channels = damage_indicators.shape

    # Set default values
    if channel_names is None:
        channel_names = [f"Channel {i+2}" for i in range(n_channels)]

    if state_labels is None:
        state_labels = np.arange(1, n_states + 1)

    if undamaged_states is None:
        undamaged_states = []

    # Create subplot layout
    n_rows = 2
    n_cols = (n_channels + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    if n_channels == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    # Plot each channel
    for i in range(n_channels):
        ax = axes[i]

        # Create bar colors
        colors = ["k" if j in undamaged_states else "r" for j in range(n_states)]

        # Plot bars
        bars = ax.bar(range(n_states), damage_indicators[:, i], color=colors)

        # Format subplot
        ax.set_title(channel_names[i])
        ax.set_xlim(-0.5, n_states - 0.5)
        ax.set_xticks(range(n_states))
        ax.set_xticklabels(state_labels)
        ax.grid(True, alpha=0.3)

        # Labels only on bottom and left subplots
        if i >= n_channels - n_cols:
            ax.set_xlabel("State Condition")
        if i % n_cols == 0:
            ax.set_ylabel("DI")

    # Hide unused subplots
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def analyze_damage_localization(
    damage_indicators: np.ndarray,
    channel_names: Optional[List[str]] = None,
    undamaged_states: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Analyze damage localization results and provide interpretation.

    Parameters
    ----------
    damage_indicators : array_like
        Damage indicators matrix, shape (N_STATES, N_CHANNELS).

    channel_names : list, optional
        Names for each channel.

    undamaged_states : list, optional
        List of undamaged state indices.

    Returns
    -------
    analysis : dict
        Dictionary containing analysis results:
        - 'channel_sensitivity': Mean damage indicator for damaged states per channel
        - 'damage_ranking': Channels ranked by damage sensitivity
        - 'interpretation': Text interpretation of results

    Examples
    --------
    >>> analysis = analyze_damage_localization(damage_indicators,
    ...                                       channel_names=['Ch2', 'Ch3', 'Ch4', 'Ch5'])
    >>> print(analysis['interpretation'])
    """
    n_states, n_channels = damage_indicators.shape

    # Set defaults
    if channel_names is None:
        channel_names = [f"Channel {i+2}" for i in range(n_channels)]

    if undamaged_states is None:
        undamaged_states = []

    # Identify damaged states
    damaged_states = [i for i in range(n_states) if i not in undamaged_states]

    if not damaged_states:
        return {
            "channel_sensitivity": np.zeros(n_channels),
            "damage_ranking": list(range(n_channels)),
            "interpretation": "No damaged states identified.",
        }

    # Calculate mean damage indicators for damaged states
    damaged_dis = damage_indicators[damaged_states, :]
    channel_sensitivity = np.mean(damaged_dis, axis=0)

    # Rank channels by sensitivity
    sensitivity_order = np.argsort(channel_sensitivity)[::-1]  # Descending order

    # Create interpretation
    most_sensitive = channel_names[sensitivity_order[0]]
    least_sensitive = channel_names[sensitivity_order[-1]]

    interpretation = (
        f"Damage localization analysis:\\n"
        f"- Most sensitive channel: {most_sensitive} "
        f"(DI = {channel_sensitivity[sensitivity_order[0]]:.2f})\\n"
        f"- Least sensitive channel: {least_sensitive} "
        f"(DI = {channel_sensitivity[sensitivity_order[-1]]:.2f})\\n"
        f"- Channel ranking (most to least sensitive): "
        f"{', '.join([channel_names[i] for i in sensitivity_order])}\\n\\n"
        f"Interpretation: Damage is likely located closest to the most sensitive "
        f"channels ({', '.join([channel_names[i] for i in sensitivity_order[:2]])})."
    )

    return {
        "channel_sensitivity": channel_sensitivity,
        "damage_ranking": sensitivity_order,
        "interpretation": interpretation,
    }


def compare_ar_arx_localization(
    ar_damage_indicators: np.ndarray,
    arx_damage_indicators: np.ndarray,
    channel_names: Optional[List[str]] = None,
    undamaged_states: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compare damage localization results between AR and ARX methods.

    Parameters
    ----------
    ar_damage_indicators : array_like
        Damage indicators from AR model analysis.

    arx_damage_indicators : array_like
        Damage indicators from ARX model analysis.

    channel_names : list, optional
        Names for each channel.

    undamaged_states : list, optional
        List of undamaged state indices.

    Returns
    -------
    comparison : dict
        Dictionary containing comparison results and interpretation.

    Examples
    --------
    >>> comparison = compare_ar_arx_localization(ar_dis, arx_dis)
    >>> print(comparison['summary'])
    """
    # Analyze each method separately
    ar_analysis = analyze_damage_localization(
        ar_damage_indicators, channel_names, undamaged_states
    )
    arx_analysis = analyze_damage_localization(
        arx_damage_indicators, channel_names, undamaged_states
    )

    # Compare sensitivities
    ar_sensitivity = ar_analysis["channel_sensitivity"]
    arx_sensitivity = arx_analysis["channel_sensitivity"]

    # Calculate improvement in discrimination
    sensitivity_ratio = arx_sensitivity / (
        ar_sensitivity + 1e-10
    )  # Avoid division by zero

    if channel_names is None:
        channel_names = [f"Channel {i+2}" for i in range(len(ar_sensitivity))]

    # Create comparison summary
    best_ar_channel = channel_names[ar_analysis["damage_ranking"][0]]
    best_arx_channel = channel_names[arx_analysis["damage_ranking"][0]]

    summary = (
        f"AR vs ARX Damage Localization Comparison:\\n\\n"
        f"AR Method Results:\\n"
        f"- Best channel: {best_ar_channel}\\n"
        f"- Channel ranking: {', '.join([channel_names[i] for i in ar_analysis['damage_ranking']])}\\n\\n"
        f"ARX Method Results:\\n"
        f"- Best channel: {best_arx_channel}\\n"
        f"- Channel ranking: {', '.join([channel_names[i] for i in arx_analysis['damage_ranking']])}\\n\\n"
        f"Improvement Analysis:\\n"
        f"- ARX provides {'better' if np.mean(sensitivity_ratio) > 1.1 else 'similar'} "
        f"damage discrimination\\n"
        f"- Average sensitivity improvement: {np.mean(sensitivity_ratio):.2f}×\\n"
        f"- ARX benefits from input-output relationships for damage localization"
    )

    return {
        "ar_analysis": ar_analysis,
        "arx_analysis": arx_analysis,
        "sensitivity_ratio": sensitivity_ratio,
        "summary": summary,
    }
