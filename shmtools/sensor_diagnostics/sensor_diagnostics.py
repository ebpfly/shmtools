"""Piezoelectric sensor diagnostics functions for structural health monitoring.

This module provides functions for automatic detection of sensor failures
(fractures and debonding) in piezoelectric active-sensor arrays using
electrical admittance measurements.

References
----------
.. [1] Overly, T.G., Park, G., Farinholt, K.M., Farrar, C.R. "Piezoelectric
       Active-Sensor Diagnostics and Validation Using Instantaneous Baseline
       Data," IEEE Sensors Journal, in press.
.. [2] Park, G., Farrar, C.R., Rutherford, C.A., Robertson, A.N., 2006,
       "Piezoelectric Active Sensor Self-diagnostics using Electrical Admittance
       Measurements," ASME Journal of Vibration and Acoustics, 128(4), 469-476.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional


def sd_feature_shm(admittance_data: np.ndarray) -> np.ndarray:
    """Extract capacitance values from piezoelectric sensor admittance data.

    Extracts the capacitance of an array of piezoelectric active-sensors by
    fitting a linear polynomial to the imaginary part of the electrical
    admittance data measured at a relatively low-frequency range (<20 kHz).

    .. meta::
        :category: Auxiliary - Sensor Support
        :matlab_equivalent: sdFeature_shm
        :display_name: Sensor Diagnostics Feature
        :verbose_call: [Sensor Capacitance] = Sensor Diagnostics Feature (Admittance Data)
        :complexity: Basic
        :data_type: Frequency Response
        :output_type: Features

    Parameters
    ----------
    admittance_data : array_like
        Matrix of imaginary electrical admittance data.
        Shape: (FREQUENCY, SENSORS)

        - First column: frequency range in Hz
        - Remaining columns: admittance data from each sensor

        .. gui::
            :widget: file_upload
            :formats: [".mat", ".csv", ".npy"]
            :description: Admittance Data

    Returns
    -------
    capacitance : ndarray
        Vector of capacitance values (slope of imaginary admittance).
        Shape: (SENSORS-1,)
        Units: Siemens/Hz (multiply by 1e9 for nF)

    Notes
    -----
    The capacitance is extracted as the slope of a linear fit to the
    imaginary part of the admittance vs. frequency. This value is
    temperature sensitive, so the algorithm uses an array of sensors
    to establish an instantaneous baseline.

    Examples
    --------
    >>> # Load admittance data with 12 sensors
    >>> data = load_sensor_diagnostic_data()
    >>> admittance = data['sd_ex_broken']
    >>> capacitance = sd_feature_shm(admittance)
    >>> print(f"Capacitance values (nF): {capacitance * 1e9}")
    """
    # Get number of columns (frequency + sensors)
    n_cols = admittance_data.shape[1]

    if n_cols <= 1:
        raise ValueError("Input must have at least 2 columns (frequency + 1 sensor)")

    # Extract frequency vector
    frequency = admittance_data[:, 0]

    # Initialize output array
    n_sensors = n_cols - 1
    capacitance = np.zeros(n_sensors)

    # Fit linear polynomial to each sensor's admittance data
    for i in range(n_sensors):
        sensor_data = admittance_data[:, i + 1]  # Skip frequency column
        # polyfit returns [slope, intercept], we only need slope
        coeffs = np.polyfit(frequency, sensor_data, 1)
        capacitance[i] = coeffs[0]  # Slope = capacitance

    return capacitance


def sd_autoclassify_shm(
    capacitance: np.ndarray, threshold: float = 0.02
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Automatically classify sensor operational status.

    Classifies piezoelectric sensors as healthy, de-bonded, or broken/fractured
    based on their capacitance values using an instantaneous baseline approach.

    .. meta::
        :category: Auxiliary - Sensor Support
        :matlab_equivalent: sdAutoclassify_shm
        :display_name: Sensor Diagnostics Auto-Classify
        :verbose_call: [Sensor Status, Data for Plotting] = Sensor Diagnostics Auto-Classify (slope, Threshold Limit)
        :complexity: Intermediate
        :data_type: Features
        :output_type: Classification

    Parameters
    ----------
    capacitance : array_like
        Vector of capacitance values from sd_feature_shm.
        Shape: (SENSORS,)

        .. gui::
            :widget: data_select
            :description: slope

    threshold : float, optional
        Threshold limit for sensor failure detection.
        Default: 0.02 (2% variation allowed among healthy sensors)

        .. gui::
            :widget: number_input
            :min: 0.001
            :max: 0.1
            :step: 0.001
            :default: 0.02
            :description: Threshold Limit

    Returns
    -------
    sensor_status : ndarray
        Sensor information matrix.
        Shape: (SENSORS, 3)

        - Column 0: Sensor ID number (1-based)
        - Column 1: Status (0=healthy, 1=de-bonded, 2=broken/fractured)
        - Column 2: Capacitance value in nF

    data_for_plotting : dict
        Data structure for visualization:

        - 'A': Processing matrix with deviation information
        - 'pos': Number of healthy sensors
        - 'ave': Average capacitance of healthy sensors
        - 'list': [broken_list, debonded_list] sensor indices

    Notes
    -----
    Assumptions:
    1. All sensors are the same size/material
    2. Maximum number of unhealthy sensors < 50% of total

    The algorithm iteratively removes sensors that contribute most to
    the standard deviation until a stable baseline is established.
    """
    n_sensors = len(capacitance)

    # Initialize processing matrix
    A = np.zeros((n_sensors, 5))
    A[:, 4] = capacitance  # Store original capacitance values

    # Iterative outlier removal to find healthy sensor baseline
    remaining_cap = capacitance.copy()
    dev_prev = np.std(remaining_cap)

    for round_num in range(1, n_sensors):
        if len(remaining_cap) <= n_sensors // 2:
            break

        # Find sensor that contributes most to standard deviation
        max_delta_dev = 0
        max_idx = -1
        current_std = np.std(remaining_cap)

        for i in range(len(remaining_cap)):
            # Test removing this sensor
            temp = np.delete(remaining_cap, i)
            new_std = np.std(temp)
            delta_dev = dev_prev - new_std

            if delta_dev > max_delta_dev:
                max_delta_dev = delta_dev
                max_idx = i

        if max_delta_dev > 0:
            # Find original sensor index
            sensor_idx = -1
            count = 0
            for j in range(n_sensors):
                if A[j, 0] == 0:  # Not yet removed
                    if count == max_idx:
                        sensor_idx = j
                        break
                    count += 1

            # Mark sensor as removed
            A[sensor_idx, 0] = round_num
            A[sensor_idx, 1] = max_delta_dev
            A[sensor_idx, 2] = current_std
            A[sensor_idx, 3] = sensor_idx + 1  # 1-based sensor ID

            # Remove from remaining set
            remaining_cap = np.delete(remaining_cap, max_idx)
            dev_prev = np.std(remaining_cap)

    # Mark remaining sensors as healthy (last round)
    for i in range(n_sensors):
        if A[i, 0] == 0:
            A[i, 0] = n_sensors
            A[i, 3] = i + 1  # 1-based sensor ID

    # Determine cutoff between healthy and faulty sensors
    # Sort by standard deviation contribution
    sorted_idx = np.argsort(A[:, 2])
    A_sorted = A[sorted_idx]

    # Find optimal cutoff point
    cutoff_pos = n_sensors
    if n_sensors > 2:
        # Look for biggest gap in deviation curve
        max_diff = 0
        for m in range(n_sensors - 1, n_sensors // 2 - 1, -1):
            # Expected linear decrease
            y = (A_sorted[-1, 2] / (n_sensors - 1)) * m - (
                A_sorted[-1, 2] / (n_sensors - 1)
            )
            diff = y - A_sorted[m, 2]

            if diff > max_diff:
                max_diff = diff
                cutoff_pos = m + 1
            elif diff < 0:
                # All sensors are healthy
                cutoff_pos = n_sensors
                break

    # Calculate average capacitance of healthy sensors
    healthy_mask = A_sorted[cutoff_pos:, 3].astype(int) - 1
    ave_healthy = np.mean(capacitance[healthy_mask])

    # Classify faulty sensors
    sensor_status = np.zeros((n_sensors, 3))
    sensor_status[:, 0] = np.arange(1, n_sensors + 1)  # Sensor IDs
    sensor_status[:, 2] = capacitance * 1e9  # Convert to nF

    broken_list = []
    debonded_list = []

    if cutoff_pos < n_sensors:
        # Check each potentially faulty sensor
        for i in range(cutoff_pos):
            sensor_idx = int(A_sorted[i, 3]) - 1
            cap_value = capacitance[sensor_idx]

            # Debonded: capacitance higher than average
            if (
                cap_value > ave_healthy
                and (cap_value - ave_healthy) / ave_healthy > threshold
            ):
                sensor_status[sensor_idx, 1] = 1
                debonded_list.append(sensor_idx + 1)
            # Broken: capacitance lower than average
            elif (
                cap_value < ave_healthy
                and (ave_healthy - cap_value) / ave_healthy > threshold
            ):
                sensor_status[sensor_idx, 1] = 2
                broken_list.append(sensor_idx + 1)

    # Prepare plotting data
    data_for_plotting = {
        "A": A,
        "pos": len(healthy_mask),
        "ave": ave_healthy,
        "list": [broken_list, debonded_list],
    }

    return sensor_status, data_for_plotting


def sd_plot_shm(data_for_plotting: Dict[str, Any]) -> None:
    """Plot sensor diagnostic results.

    Creates two figures showing the classification process and sensor status.

    .. meta::
        :category: Auxiliary - Sensor Support
        :matlab_equivalent: sdPlot_shm
        :display_name: Sensor Diagnostics Result Plot
        :verbose_call: Sensor Diagnostics Result Plot (Data for Plotting)
        :complexity: Basic
        :data_type: Plotting Data
        :output_type: Visualization

    Parameters
    ----------
    data_for_plotting : dict
        Output from sd_autoclassify_shm containing:

        - 'A': Processing matrix
        - 'pos': Number of healthy sensors
        - 'ave': Average healthy capacitance
        - 'list': [broken_list, debonded_list]

        .. gui::
            :widget: data_select
            :description: Data for Plotting

    Returns
    -------
    None
        Displays two matplotlib figures.

    Notes
    -----
    Figure 1: Shows the iterative classification process with standard
    deviation contributions.

    Figure 2: Bar chart showing percent deviation from healthy baseline
    for each sensor (blue=healthy, red=broken, magenta=debonded).
    """
    A = data_for_plotting["A"]
    pos = data_for_plotting["pos"]
    ave = data_for_plotting["ave"]
    broken_list, debonded_list = data_for_plotting["list"]

    n_sensors = len(A)

    # Figure 1: Classification process
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Sort by standard deviation
    sorted_idx = np.argsort(A[:, 2])
    A_sorted = A[sorted_idx]

    # Create color gradient (red to green)
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, n_sensors))

    # Plot points in reverse order (highest std first)
    x_values = np.arange(1, n_sensors + 1)
    for i in range(n_sensors):
        sensor_id = int(A_sorted[n_sensors - 1 - i, 3])
        ax1.scatter(
            i + 1,
            A_sorted[n_sensors - 1 - i, 2],
            c=[colors[i]],
            s=50,
            label=f"Sensor {sensor_id}",
        )

    # Draw baseline trend line
    ax1.plot([1, n_sensors], [A_sorted[-1, 2], 0], "k:", linewidth=1)

    # Draw cutoff line if sensors were classified as faulty
    if pos < n_sensors:
        ax1.axvline(x=n_sensors - pos + 0.5, color="k", linestyle=":", linewidth=1)

        # Add arrows to indicate healthy/faulty regions
        ax1.annotate(
            "",
            xy=(n_sensors - pos, ax1.get_ylim()[0]),
            xytext=(1, ax1.get_ylim()[0]),
            arrowprops=dict(arrowstyle="<-", color="red", lw=2),
        )
        ax1.annotate(
            "",
            xy=(n_sensors - pos + 1, ax1.get_ylim()[0]),
            xytext=(n_sensors, ax1.get_ylim()[0]),
            arrowprops=dict(arrowstyle="<-", color="green", lw=2),
        )

        ax1.text(
            n_sensors // 4,
            ax1.get_ylim()[0] * 0.1,
            "Faulty",
            color="red",
            ha="center",
            fontsize=12,
        )
        ax1.text(
            3 * n_sensors // 4,
            ax1.get_ylim()[0] * 0.1,
            "Healthy",
            color="green",
            ha="center",
            fontsize=12,
        )

    ax1.set_xlabel("Sensor Number", fontsize=12)
    ax1.set_ylabel("Standard Deviation (\u03c3)", fontsize=12)
    ax1.set_title("Sensor Diagnostic Classification Process", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n_sensors + 1)

    # Set x-tick labels to show sensor IDs
    xticks = (
        [""]
        + [str(int(A_sorted[n_sensors - i - 1, 3])) for i in range(n_sensors)]
        + [""]
    )
    ax1.set_xticks(range(n_sensors + 2))
    ax1.set_xticklabels(xticks)

    # Figure 2: Sensor status bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    sensor_ids = np.arange(1, n_sensors + 1)
    deviations = (A[:, 4] - ave) / ave * 100  # Percent deviation

    # Plot all sensors as healthy (blue) first
    bars = ax2.bar(sensor_ids, deviations, color="blue", label="Healthy sensors")

    # Recolor broken sensors (red)
    for sensor_id in broken_list:
        bars[sensor_id - 1].set_color("red")

    # Recolor debonded sensors (magenta)
    for sensor_id in debonded_list:
        bars[sensor_id - 1].set_color("magenta")

    # Create legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color="blue", label="Healthy sensors")
    ]
    if broken_list:
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, color="red", label="Broken sensors")
        )
    if debonded_list:
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, color="magenta", label="De-bonded sensors")
        )

    ax2.legend(handles=legend_elements, loc="best")

    ax2.set_xlabel("Sensor Number", fontsize=12)
    ax2.set_ylabel("Percent Deviation of Sensor Status (%)", fontsize=12)
    ax2.set_title("Sensor Status Classification Results", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, n_sensors + 1)

    # Adjust y-limits for better visualization
    if pos == n_sensors:  # All healthy
        ax2.set_ylim(-20, 20)

    plt.tight_layout()
    plt.show()
