"""
Results viewer panel for displaying analysis outputs.

This panel provides interactive visualization of workflow results
including plots, data tables, and export capabilities.
"""

from bokeh.models import (
    Div,
    Tabs,
    TabPanel,
    Button,
    Select,
    Column,
    Row,
    DataTable,
    TableColumn,
    ColumnDataSource,
)
from bokeh.plotting import figure
from bokeh.layouts import column, row
import numpy as np


class ResultsViewerPanel:
    """
    Panel for viewing and interacting with analysis results.

    This panel provides interactive plots and data views for
    workflow outputs, with export and analysis capabilities.
    """

    def __init__(self):
        """Initialize the results viewer panel."""
        self.current_results = None
        self.workflow_results = {}  # Store all workflow step results
        self.panel = self._create_panel()

    def _create_panel(self):
        """
        Create the results viewer panel components.

        Returns
        -------
        panel : bokeh.models.Panel
            Results viewer panel.
        """
        # Panel title
        title = Div(text="<h3>Results Viewer</h3>", width=800)

        # Results tabs
        plot_tab = self._create_plot_tab()
        data_tab = self._create_data_tab()
        export_tab = self._create_export_tab()

        results_tabs = Tabs(tabs=[plot_tab, data_tab, export_tab])

        # Store references
        self.results_tabs = results_tabs

        # Create panel layout
        panel_content = column(
            title, results_tabs, width=850, sizing_mode="stretch_width"
        )

        return panel_content  # Return the layout directly instead of wrapping in Panel

    def _create_plot_tab(self):
        """Create the interactive plots tab."""
        # Plot type selector - will be populated with workflow results
        plot_select = Select(
            title="Select Result to View:", value="", options=[], width=300
        )
        plot_select.on_change("value", self._on_result_select)

        # Create example plot
        plot = figure(
            title="Analysis Results",
            x_axis_label="Time (s)",
            y_axis_label="Amplitude",
            width=700,
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )

        # Add example data
        t = np.linspace(0, 1, 1000)
        y = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(1000)
        plot.line(t, y, line_width=2, color="blue", alpha=0.8)

        # Plot controls
        controls = row(plot_select)
        plot_content = column(controls, plot)

        # Store references
        self.plot = plot
        self.plot_select = plot_select

        return TabPanel(child=plot_content, title="Plots")

    def _create_data_tab(self):
        """Create the data table tab."""
        # Create example data table
        data = {
            "time": np.linspace(0, 1, 100),
            "signal": np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100)),
            "feature": np.random.randn(100),
        }
        source = ColumnDataSource(data)

        columns = [
            TableColumn(field="time", title="Time (s)", width=100),
            TableColumn(field="signal", title="Signal", width=100),
            TableColumn(field="feature", title="Feature", width=100),
        ]

        data_table = DataTable(
            source=source, columns=columns, width=700, height=400, selectable=True
        )

        # Data controls
        download_btn = Button(label="Download CSV", button_type="primary", width=150)
        download_btn.on_click(self._download_data)

        data_content = column(download_btn, data_table)

        # Store references
        self.data_table = data_table

        return TabPanel(child=data_content, title="Data")

    def _create_export_tab(self):
        """Create the export/report tab."""
        # Export options
        export_format = Select(
            title="Export Format:",
            value="PNG",
            options=["PNG", "SVG", "PDF", "HTML"],
            width=200,
        )

        export_btn = Button(label="Export Plot", button_type="success", width=150)
        export_btn.on_click(self._export_plot)

        # Report generation
        report_btn = Button(label="Generate Report", button_type="primary", width=150)
        report_btn.on_click(self._generate_report)

        export_content = column(
            Div(text="<h4>Export Options</h4>"),
            export_format,
            row(export_btn, report_btn),
            Div(text="<h4>Analysis Summary</h4>"),
            Div(
                text="<p>Workflow completed successfully. Results available above.</p>"
            ),
        )

        return TabPanel(child=export_content, title="Export")

    def update_results(self, results):
        """
        Update the results viewer with new analysis results.

        Parameters
        ----------
        results : dict
            Analysis results dictionary from workflow execution.
        """
        self.current_results = results

        # Clear workflow results for new execution
        self.workflow_results = {}

        # Store workflow results for visualization
        if "step_results" in results:
            for i, step_result in enumerate(results["step_results"]):
                if step_result.get("success") and "outputs" in step_result:
                    # Store each step's outputs
                    for output_name, output_value in step_result["outputs"].items():
                        self.workflow_results[output_name] = output_value
                        print(f"Stored result: {output_name}")

        # Update the plot selector with available results
        if self.workflow_results:
            self.plot_select.options = list(self.workflow_results.keys())
            self.plot_select.value = list(self.workflow_results.keys())[
                -1
            ]  # Select latest result

            # Visualize the selected result
            self._visualize_selected_result()

    def _auto_visualize(self, output_name: str, output_value):
        """
        Automatically generate appropriate visualization based on output type.

        Parameters
        ----------
        output_name : str
            Name of the output variable.
        output_value : any
            The output data to visualize.
        """
        # Clear existing plot
        self.plot.renderers = []

        # Debug output
        print(f"\nVisualizing {output_name}:")
        print(f"  Type: {type(output_value)}")
        if hasattr(output_value, "shape"):
            print(f"  Shape: {output_value.shape}")
        elif isinstance(output_value, (tuple, list)):
            print(f"  Length: {len(output_value)}")
            for i, item in enumerate(output_value[:3]):  # Show first 3 items
                if hasattr(item, "shape"):
                    print(f"  Item {i}: {type(item)} shape={item.shape}")
                else:
                    print(f"  Item {i}: {type(item)}")

        if hasattr(output_value, "shape"):
            # Handle numpy arrays based on shape
            shape = output_value.shape

            if len(shape) == 1:
                # 1D array - plot as time series or coefficients
                self._plot_1d_array(output_name, output_value)
            elif len(shape) == 2:
                # 2D array - could be multi-channel time series or feature matrix
                if shape[0] > shape[1] and shape[0] > 100:
                    # Likely time series (time x channels)
                    self._plot_multi_channel_time_series(output_name, output_value)
                else:
                    # Likely feature matrix or coefficients
                    self._plot_2d_heatmap(output_name, output_value)
            elif len(shape) == 3:
                # 3D array - full dataset (time x channels x conditions)
                self._plot_3d_dataset_summary(output_name, output_value)

        elif isinstance(output_value, dict):
            # Handle dictionary outputs (like PCA models)
            self._plot_dict_summary(output_name, output_value)

        elif isinstance(output_value, (list, tuple)):
            # Handle tuple/list outputs
            if "ar_model" in output_name.lower() and len(output_value) >= 3:
                # AR model specific output format
                # Item 2 is the AR parameters: (order, channels, conditions)
                if (
                    hasattr(output_value[2], "shape")
                    and len(output_value[2].shape) == 3
                ):
                    ar_params = output_value[2]  # Shape: (15, 5, 170)
                    # Show AR coefficients for first channel, first condition
                    coeffs = ar_params[:, 0, 0]  # First channel, first condition
                    self._plot_ar_coefficients(output_name, coeffs)
                else:
                    # Fallback to first element
                    self._auto_visualize(f"{output_name}[0]", output_value[0])
            elif len(output_value) == 2:
                # Could be (freq, psd) or (coeffs, errors) etc.
                if hasattr(output_value[0], "shape") and hasattr(
                    output_value[1], "shape"
                ):
                    # Both are arrays - check dimensionality
                    if output_value[0].ndim == 1 and output_value[1].ndim == 1:
                        # Two 1D arrays - might be frequency response
                        if (
                            "psd" in output_name.lower()
                            or "freq" in output_name.lower()
                        ):
                            self._plot_frequency_response(
                                output_name, output_value[0], output_value[1]
                            )
                        else:
                            # AR model output or similar - plot first array as coefficients
                            self._plot_1d_array(
                                f"{output_name} (coefficients)", output_value[0]
                            )
                    else:
                        # Multi-dimensional - visualize first element
                        self._auto_visualize(f"{output_name}[0]", output_value[0])
                elif hasattr(output_value[0], "shape"):
                    # First element is array - visualize it
                    self._auto_visualize(
                        f"{output_name} (primary output)", output_value[0]
                    )
            else:
                # Multiple outputs - show summary
                self._display_text_summary(
                    output_name, f"Tuple/List with {len(output_value)} elements"
                )

        else:
            # Default text display for other types
            self._display_text_summary(output_name, output_value)

    def _on_result_select(self, attr, old, new):
        """Handle result selection change."""
        if new and new in self.workflow_results:
            print(f"Displaying result: {new}")
            self._visualize_selected_result()

    def _visualize_selected_result(self):
        """Visualize the currently selected result."""
        if self.plot_select.value and self.plot_select.value in self.workflow_results:
            output_name = self.plot_select.value
            output_value = self.workflow_results[output_name]
            self._auto_visualize(output_name, output_value)

    def _plot_time_series(self):
        """Plot time series data."""
        # Clear existing plot
        self.plot.renderers = []

        # Add time series data
        t = np.linspace(0, 1, 1000)
        y = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(1000)
        self.plot.line(t, y, line_width=2, color="blue", alpha=0.8)

        self.plot.title.text = "Time Series"
        self.plot.xaxis.axis_label = "Time (s)"
        self.plot.yaxis.axis_label = "Amplitude"

    def _plot_psd(self):
        """Plot power spectral density."""
        self.plot.renderers = []

        # Example PSD data
        f = np.linspace(0, 50, 1000)
        psd = 1 / (1 + (f / 5) ** 2) + np.random.randn(1000) * 0.01
        self.plot.line(f, psd, line_width=2, color="red", alpha=0.8)

        self.plot.title.text = "Power Spectral Density"
        self.plot.xaxis.axis_label = "Frequency (Hz)"
        self.plot.yaxis.axis_label = "PSD"

    def _plot_spectrogram(self):
        """Plot spectrogram."""
        # TODO: Implement spectrogram plotting
        print("Plotting spectrogram...")

    def _plot_outlier_scores(self):
        """Plot outlier detection scores."""
        self.plot.renderers = []

        # Example outlier scores
        x = np.arange(100)
        scores = np.random.exponential(1, 100)
        scores[50:55] = scores[50:55] * 3  # Add some outliers

        self.plot.scatter(x, scores, size=6, color="green", alpha=0.7)
        self.plot.line(
            [0, 100],
            [3, 3],
            line_width=2,
            color="red",
            line_dash="dashed",
            legend_label="Threshold",
        )

        self.plot.title.text = "Outlier Detection Scores"
        self.plot.xaxis.axis_label = "Sample"
        self.plot.yaxis.axis_label = "Outlier Score"

    def _download_data(self):
        """Download current data as CSV."""
        print("Downloading data as CSV...")
        # TODO: Implement CSV download

    def _export_plot(self):
        """Export current plot."""
        print("Exporting plot...")
        # TODO: Implement plot export

    def _generate_report(self):
        """Generate analysis report."""
        print("Generating analysis report...")
        # TODO: Implement report generation

    def _plot_1d_array(self, name: str, data: np.ndarray):
        """Plot 1D array as line plot or bar chart."""
        self.plot.renderers = []

        if "coeff" in name.lower() or "ar" in name.lower():
            # AR coefficients - use bar chart
            x = np.arange(len(data))
            self.plot.vbar(x=x, top=data, width=0.8, color="navy", alpha=0.8)
            self.plot.title.text = f"AR Coefficients: {name}"
            self.plot.xaxis.axis_label = "Coefficient Index"
            self.plot.yaxis.axis_label = "Value"
        else:
            # Time series or other 1D data
            x = np.arange(len(data))
            self.plot.line(x, data, line_width=2, color="blue", alpha=0.8)
            self.plot.title.text = f"1D Array: {name}"
            self.plot.xaxis.axis_label = "Index"
            self.plot.yaxis.axis_label = "Value"

    def _plot_multi_channel_time_series(self, name: str, data: np.ndarray):
        """Plot multi-channel time series."""
        self.plot.renderers = []

        colors = ["blue", "red", "green", "orange", "purple"]
        time_points = np.arange(data.shape[0])

        for ch in range(min(data.shape[1], 5)):  # Limit to 5 channels
            self.plot.line(
                time_points,
                data[:, ch],
                line_width=2,
                color=colors[ch % len(colors)],
                alpha=0.8,
                legend_label=f"Ch {ch+1}",
            )

        self.plot.title.text = f"Multi-Channel Time Series: {name}"
        self.plot.xaxis.axis_label = "Time Points"
        self.plot.yaxis.axis_label = "Amplitude"
        self.plot.legend.location = "top_right"

    def _plot_3d_dataset_summary(self, name: str, data: np.ndarray):
        """Plot summary of 3D dataset."""
        self.plot.renderers = []

        # Show RMS of each condition across channels
        rms_per_condition = np.sqrt(np.mean(data**2, axis=(0, 1)))
        x = np.arange(len(rms_per_condition))

        self.plot.scatter(x, rms_per_condition, size=6, color="blue", alpha=0.7)
        self.plot.line(x, rms_per_condition, line_width=1, color="blue", alpha=0.5)

        self.plot.title.text = f"Dataset Summary (RMS per condition): {name}"
        self.plot.xaxis.axis_label = "Condition Index"
        self.plot.yaxis.axis_label = "RMS Value"

    def _plot_2d_heatmap(self, name: str, data: np.ndarray):
        """Plot 2D array as heatmap."""
        # For now, show summary statistics
        self.plot.renderers = []

        # Plot column means
        col_means = np.mean(data, axis=0)
        x = np.arange(len(col_means))
        self.plot.line(x, col_means, line_width=2, color="red", alpha=0.8)

        self.plot.title.text = f"2D Array Column Means: {name}"
        self.plot.xaxis.axis_label = "Column Index"
        self.plot.yaxis.axis_label = "Mean Value"

    def _plot_frequency_response(
        self, name: str, freqs: np.ndarray, response: np.ndarray
    ):
        """Plot frequency response (PSD, FFT, etc.)."""
        self.plot.renderers = []

        self.plot.line(freqs, response, line_width=2, color="red", alpha=0.8)
        self.plot.title.text = f"Frequency Response: {name}"
        self.plot.xaxis.axis_label = "Frequency (Hz)"
        self.plot.yaxis.axis_label = "Magnitude"

        # Log scale for y-axis if appropriate
        if np.max(response) / np.min(response[response > 0]) > 100:
            self.plot.y_scale = "log"

    def _plot_dict_summary(self, name: str, data: dict):
        """Display dictionary summary."""
        # For now, show text summary
        self._display_text_summary(name, f"Dictionary with keys: {list(data.keys())}")

    def _display_text_summary(self, name: str, data):
        """Display text summary for non-plottable data."""
        self.plot.renderers = []

        # Add text annotation
        from bokeh.models import Label

        text = f"{name}:\n{str(data)[:200]}..."  # Limit text length
        label = Label(x=10, y=10, text=text, text_font_size="12pt")
        self.plot.add_layout(label)

    def _plot_ar_coefficients(self, name: str, coeffs: np.ndarray):
        """Plot AR coefficients as a bar chart."""
        self.plot.renderers = []

        x = np.arange(len(coeffs))
        self.plot.vbar(x=x, top=coeffs, width=0.8, color="navy", alpha=0.8)

        self.plot.title.text = f"AR Coefficients (Ch1, Condition1): {name}"
        self.plot.xaxis.axis_label = "Coefficient Index"
        self.plot.yaxis.axis_label = "Value"

        # Add grid for better readability
        self.plot.grid.grid_line_alpha = 0.3
