"""
Function library panel for browsing available SHMTools functions.

This panel provides a tree-like interface for discovering and selecting
functions from the SHMTools library, similar to the left panel in mFUSE.
"""

from bokeh.models import (
    Div, Select, Button, Column, DataTable, TableColumn,
    ColumnDataSource
)
from bokeh.layouts import column
import shmtools
from bokeh_shmtools.utils.docstring_parser import parse_shmtools_docstring


class FunctionLibraryPanel:
    """
    Panel for browsing and selecting SHMTools functions.
    
    This panel replicates the function library browser from mFUSE,
    allowing users to explore available functions by category.
    """
    
    def __init__(self):
        """Initialize the function library panel."""
        self.panel = self._create_panel()
        self.selected_function = None
        
    def _create_panel(self):
        """
        Create the function library panel components.
        
        Returns
        -------
        panel : bokeh.models.Panel
            Function library panel.
        """
        # Panel title
        title = Div(text="<h3>Function Library</h3>", width=250)
        
        # Category selector
        categories = [
            "All Functions",
            "Core - Spectral Analysis", 
            "Core - Filtering",
            "Core - Statistics",
            "Features - Time Series",
            "Classification - Outlier Detection",
            "Modal Analysis",
            "Active Sensing",
            "Hardware",
        ]
        
        category_select = Select(
            title="Category:",
            value="All Functions",
            options=categories,
            sizing_mode="stretch_width"
        )
        category_select.on_change("value", self._on_category_change)
        
        # Function list
        functions_data = self._get_functions_data("All Functions")
        functions_source = ColumnDataSource(functions_data)
        
        columns = [
            TableColumn(field="name", title="Function", width=180),
            TableColumn(field="description", title="Description", width=200)
        ]
        
        functions_table = DataTable(
            source=functions_source,
            columns=columns,
            height=300,
            selectable=True,
            sizing_mode="stretch_width"
        )
        functions_table.source.selected.on_change("indices", self._on_function_select)
        
        # Add to workflow button
        add_button = Button(
            label="Add to Workflow",
            button_type="primary",
            disabled=True,
            sizing_mode="stretch_width"
        )
        add_button.on_click(self._add_to_workflow)
        
        # Store references
        self.category_select = category_select
        self.functions_table = functions_table
        self.add_button = add_button
        
        # Create panel layout
        panel_content = column(
            title,
            category_select,
            functions_table,
            add_button,
            min_width=300,
            sizing_mode="stretch_both"
        )
        
        return panel_content  # Return the layout directly instead of wrapping in Panel
    
    def _get_functions_data(self, category):
        """
        Get function data for the specified category.
        
        Parameters
        ----------
        category : str
            Function category name.
            
        Returns
        -------
        data : dict
            Dictionary with function names and descriptions.
        """
        # Discover functions from shmtools modules
        discovered_functions = self._discover_shmtools_functions()
        
        if category == "All Functions":
            # Combine all categories
            all_functions = []
            for cat_functions in discovered_functions.values():
                all_functions.extend(cat_functions)
            functions = all_functions
        else:
            functions = discovered_functions.get(category, [])
        
        return {
            "name": [f["name"] for f in functions],
            "description": [f["description"] for f in functions]
        }
    
    def _discover_shmtools_functions(self):
        """
        Dynamically discover all SHMTools functions with their metadata.
        
        Returns
        -------
        functions : dict
            Dictionary mapping categories to lists of function info.
        """
        functions_by_category = {}
        
        # Discover functions from core modules
        modules_to_scan = [
            ('shmtools.core.spectral', shmtools.core.spectral),
            ('shmtools.core.filtering', shmtools.core.filtering), 
            ('shmtools.core.statistics', shmtools.core.statistics),
            ('shmtools.core.preprocessing', shmtools.core.preprocessing),
            ('shmtools.features.time_series', shmtools.features.time_series),
            ('shmtools.classification.outlier_detection', shmtools.classification.outlier_detection),
        ]
        
        for module_name, module in modules_to_scan:
            try:
                for attr_name in dir(module):
                    if not attr_name.startswith('_'):
                        attr = getattr(module, attr_name)
                        if callable(attr):
                            # Parse docstring for metadata
                            metadata = parse_shmtools_docstring(attr)
                            if metadata:
                                category = metadata.category
                                if category not in functions_by_category:
                                    functions_by_category[category] = []
                                
                                functions_by_category[category].append({
                                    "name": metadata.name,
                                    "description": metadata.brief_description,
                                    "metadata": metadata
                                })
            except Exception as e:
                print(f"Warning: Failed to scan module {module_name}: {e}")
        
        # Add fallback functions if discovery fails
        if not functions_by_category:
            functions_by_category = {
                "Data Loading": [
                    {"name": "load_3story_data", "description": "Load 3-story structure dataset"},
                    {"name": "create_synthetic_data", "description": "Create synthetic test data"},
                ],
                "Core - Spectral Analysis": [
                    {"name": "psd_welch", "description": "Power spectral density via Welch's method"},
                    {"name": "stft", "description": "Short-time Fourier transform"},
                ],
                "Core - Statistics": [
                    {"name": "rms", "description": "Root mean square value"},
                    {"name": "crest_factor", "description": "Crest factor (peak-to-RMS ratio)"},
                    {"name": "statistical_moments", "description": "Statistical moments calculation"},
                ],
                "Features - Time Series": [
                    {"name": "ar_model", "description": "Autoregressive model fitting"},
                ],
                "Classification - Outlier Detection": [
                    {"name": "learn_pca", "description": "Train PCA outlier detection model"},
                    {"name": "score_pca", "description": "Score data with PCA model"},
                    {"name": "mahalanobis_distance", "description": "Mahalanobis distance outlier detection"},
                ],
            }
        
        # Always ensure we have some functions for "All Functions"
        all_functions_exist = any(functions_by_category.values())
        if not all_functions_exist:
            functions_by_category["Core - Spectral Analysis"] = [
                {"name": "psd_welch", "description": "Power spectral density via Welch's method"},
                {"name": "rms", "description": "Root mean square value"},
                {"name": "ar_model", "description": "Autoregressive model fitting"},
            ]
        
        return functions_by_category
    
    def _on_category_change(self, attr, old, new):
        """Handle category selection change."""
        functions_data = self._get_functions_data(new)
        self.functions_table.source.data = functions_data
        self.add_button.disabled = True
        self.selected_function = None
    
    def _on_function_select(self, attr, old, new):
        """Handle function selection."""
        if new:
            idx = new[0]
            function_name = self.functions_table.source.data["name"][idx]
            self.selected_function = function_name
            self.add_button.disabled = False
        else:
            self.selected_function = None
            self.add_button.disabled = True
    
    def _add_to_workflow(self):
        """Add selected function to workflow."""
        if self.selected_function:
            print(f"Adding {self.selected_function} to workflow")
            # Note: Actual integration is handled in app.py _setup_panel_communication()