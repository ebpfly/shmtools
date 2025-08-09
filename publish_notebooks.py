#!/usr/bin/env python3
"""
Comprehensive notebook publication script for shmtools example notebooks.

This script finds, executes, and converts all example notebooks to HTML for
publication and documentation purposes. It handles various execution contexts
and provides detailed reporting.
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError


class NotebookPublisher:
    """
    Robust notebook execution and HTML publication system.
    
    Handles execution, conversion, and organization of published notebooks.
    """
    
    def __init__(
        self, 
        timeout: int = 900,  # 15 minutes for complex notebooks
        kernel_name: str = "python3",
        output_dir: str = "published_notebooks"
    ):
        """
        Initialize the notebook publisher.
        
        Parameters
        ----------
        timeout : int, default=900
            Maximum time in seconds to wait for notebook execution
        kernel_name : str, default="python3"
            Jupyter kernel to use for execution
        output_dir : str, default="published_notebooks"
            Directory to store published HTML files
        """
        self.timeout = timeout
        self.kernel_name = kernel_name
        self.output_dir = Path(output_dir)
        
        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "basic").mkdir(exist_ok=True)
        (self.output_dir / "intermediate").mkdir(exist_ok=True)
        (self.output_dir / "advanced").mkdir(exist_ok=True)
        (self.output_dir / "specialized").mkdir(exist_ok=True)
        (self.output_dir / "other").mkdir(exist_ok=True)
        (self.output_dir / "assets").mkdir(exist_ok=True)
        
        # Set up execution processor
        self.executor = ExecutePreprocessor(
            timeout=timeout,
            kernel_name=kernel_name,
            allow_errors=False
        )
        
        # Set up HTML exporter with custom configuration
        self.html_exporter = HTMLExporter()
        self.html_exporter.template_name = 'classic'
        self.html_exporter.exclude_input_prompt = False
        self.html_exporter.exclude_output_prompt = False
        
    def format_display_name(self, filename: str) -> str:
        """
        Convert filename to proper display name with correct acronym capitalization.
        
        Parameters
        ----------
        filename : str
            Filename (without extension) to convert
            
        Returns
        -------
        display_name : str
            Properly formatted display name
        """
        # Common technical acronyms that should stay uppercase
        acronyms = {
            'ar': 'AR',
            'pca': 'PCA', 
            'svd': 'SVD',
            'roc': 'ROC',
            'cbm': 'CBM',
            'nlpca': 'NLPCA',
            'gmm': 'GMM',
            'frf': 'FRF',
            'osp': 'OSP',
            'ml': 'ML',
            'ai': 'AI',
            'gui': 'GUI',
            'api': 'API',
            'fft': 'FFT',
            'dft': 'DFT',
            'stft': 'STFT',
            'cwt': 'CWT',
            'ica': 'ICA',
            'rms': 'RMS',
            'shm': 'SHM'
        }
        
        # Replace underscores with spaces and split into words
        words = filename.replace('_', ' ').split()
        
        # Process each word
        formatted_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower in acronyms:
                formatted_words.append(acronyms[word_lower])
            else:
                formatted_words.append(word.capitalize())
        
        return ' '.join(formatted_words)
        
    def find_notebooks(
        self, 
        examples_dir: Path,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, List[Path]]:
        """
        Find and categorize all notebooks in the examples directory.
        
        Parameters
        ----------
        examples_dir : Path
            Path to examples directory
        exclude_patterns : list of str, optional
            Patterns to exclude
            
        Returns
        -------
        categorized_notebooks : dict
            Dictionary mapping categories to lists of notebook paths
        """
        if exclude_patterns is None:
            exclude_patterns = [
                "*checkpoint*",
                "*debug*",
                "*Debug*",
                "*test*",
                "*Test*"
            ]
        
        # Find all notebooks
        all_notebooks = list(examples_dir.rglob("*.ipynb"))
        
        # Filter out excluded patterns
        filtered_notebooks = []
        for notebook in all_notebooks:
            exclude = False
            for exclude_pattern in exclude_patterns:
                if notebook.match(exclude_pattern):
                    exclude = True
                    break
            if not exclude:
                filtered_notebooks.append(notebook)
        
        # Categorize by directory
        categories = {
            'basic': [],
            'intermediate': [],
            'advanced': [],
            'specialized': [],
            'other': []
        }
        
        for notebook in filtered_notebooks:
            parent_dir = notebook.parent.name.lower()
            if parent_dir in categories:
                categories[parent_dir].append(notebook)
            else:
                categories['other'].append(notebook)
        
        # Sort within each category
        for category in categories:
            categories[category].sort()
        
        return categories
    
    def execute_notebook(
        self, 
        notebook_path: Path,
        working_dir: Optional[Path] = None
    ) -> Tuple[bool, nbformat.NotebookNode, Dict[str, Any]]:
        """
        Execute a notebook and return results.
        
        Parameters
        ----------
        notebook_path : Path
            Path to notebook file
        working_dir : Path, optional
            Working directory for execution
            
        Returns
        -------
        success : bool
            True if execution succeeded
        executed_notebook : NotebookNode
            The executed notebook
        metadata : dict
            Execution metadata
        """
        if working_dir is None:
            working_dir = notebook_path.parent
            
        metadata = {
            'notebook_path': str(notebook_path),
            'working_dir': str(working_dir),
            'start_time': time.time(),
            'end_time': None,
            'execution_time': None,
            'total_cells': 0,
            'executed_cells': 0,
            'error_cells': 0,
            'errors': [],
            'kernel_name': self.kernel_name
        }
        
        try:
            print(f"    Executing notebook: {notebook_path.name}")
            
            # Read the notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
                
            metadata['total_cells'] = len(notebook.cells)
            
            # Set up execution environment
            self.executor.cwd = str(working_dir)
            
            # Execute the notebook
            executed_notebook, resources = self.executor.preprocess(
                notebook, {'metadata': {'path': str(working_dir)}}
            )
            
            # Collect execution statistics
            for i, cell in enumerate(executed_notebook.cells):
                if cell.cell_type == 'code':
                    metadata['executed_cells'] += 1
                    
                    # Check for execution errors
                    if 'outputs' in cell:
                        for output in cell.outputs:
                            if output.output_type == 'error':
                                metadata['error_cells'] += 1
                                metadata['errors'].append({
                                    'cell_index': i,
                                    'error_name': output.ename,
                                    'error_value': output.evalue,
                                    'traceback': output.traceback
                                })
            
            metadata['end_time'] = time.time()
            metadata['execution_time'] = metadata['end_time'] - metadata['start_time']
            
            success = metadata['error_cells'] == 0
            print(f"      OK Executed in {metadata['execution_time']:.1f}s ({metadata['executed_cells']} cells)")
            
            return success, executed_notebook, metadata
            
        except CellExecutionError as e:
            metadata['end_time'] = time.time()
            metadata['execution_time'] = metadata['end_time'] - metadata['start_time']
            metadata['errors'].append({
                'type': 'CellExecutionError',
                'error_name': e.__class__.__name__,
                'error_value': str(e),
                'traceback': getattr(e, 'traceback', [])
            })
            print(f"      X Execution failed: {e}")
            return False, notebook, metadata
            
        except Exception as e:
            metadata['end_time'] = time.time()
            metadata['execution_time'] = metadata['end_time'] - metadata['start_time']
            metadata['errors'].append({
                'type': 'GeneralError',
                'error_name': e.__class__.__name__,
                'error_value': str(e),
                'traceback': []
            })
            print(f"      X Execution failed: {e}")
            return False, notebook, metadata
    
    def convert_to_html(
        self, 
        notebook: nbformat.NotebookNode,
        output_path: Path,
        notebook_path: Path
    ) -> bool:
        """
        Convert executed notebook to HTML.
        
        Parameters
        ----------
        notebook : NotebookNode
            The executed notebook
        output_path : Path
            Path for output HTML file
        notebook_path : Path
            Original notebook path (for title)
            
        Returns
        -------
        success : bool
            True if conversion succeeded
        """
        try:
            # Add metadata to notebook for better HTML rendering
            if 'metadata' not in notebook:
                notebook.metadata = {}
                
            notebook.metadata.update({
                'title': self.format_display_name(notebook_path.stem),
                'authors': [{'name': 'SHMTools Development Team'}],
                'generated_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source_notebook': str(notebook_path)
            })
            
            # Convert to HTML
            (body, resources) = self.html_exporter.from_notebook_node(notebook)
            
            # Enhance HTML with custom styling and navigation
            enhanced_html = self.enhance_html(body, notebook_path)
            
            # Write HTML file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_html)
                
            print(f"      OK HTML saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"      X HTML conversion failed: {e}")
            return False
    
    def enhance_html(self, html_body: str, notebook_path: Path) -> str:
        """
        Enhance HTML with custom styling and navigation.
        
        Parameters
        ----------
        html_body : str
            Original HTML content
        notebook_path : Path
            Original notebook path
            
        Returns
        -------
        enhanced_html : str
            Enhanced HTML with styling and navigation
        """
        # Custom CSS for better presentation
        custom_css = """
        <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .notebook-header {
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .notebook-title { 
            font-size: 2.5em; 
            margin: 0 0 10px 0;
            font-weight: 300;
        }
        .notebook-subtitle { 
            font-size: 1.2em; 
            opacity: 0.9;
            margin: 0;
        }
        .navigation {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .navigation a {
            text-decoration: none;
            color: #007bff;
            margin-right: 20px;
            font-weight: 500;
        }
        .navigation a:hover { color: #0056b3; }
        .cell { 
            background: white;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .code_cell { border-left: 4px solid #007bff; }
        .text_cell { border-left: 4px solid #28a745; }
        .output_area { 
            background-color: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }
        .prompt { 
            color: #6c757d;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
        }
        pre { 
            background-color: #f8f9fa !important;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
        }
        .highlight { background-color: #fff3cd; }
        .footer {
            margin-top: 50px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            text-align: center;
            color: #6c757d;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        """
        
        # Create header
        title = self.format_display_name(notebook_path.stem)
        category = notebook_path.parent.name.title()
        
        header = f"""
        <div class="notebook-header">
            <h1 class="notebook-title">{title}</h1>
            <p class="notebook-subtitle">SHMTools Example - {category} Level</p>
        </div>
        """
        
        # Create navigation
        navigation = """
        <div class="navigation">
            <a href="../index.html">← Back to Index</a>
            <a href="https://github.com/shmtools/shmtools-python">GitHub Repository</a>
            <a href="https://shmtools.readthedocs.io">Documentation</a>
        </div>
        """
        
        # Create footer
        footer = f"""
        <div class="footer">
            <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} from <code>{notebook_path.name}</code></p>
            <p>Part of the SHMTools Python Package - Structural Health Monitoring Toolkit</p>
        </div>
        """
        
        # Combine all parts
        enhanced_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>{title} - SHMTools</title>
            {custom_css}
        </head>
        <body>
            {header}
            {navigation}
            {html_body}
            {footer}
        </body>
        </html>
        """
        
        return enhanced_html
    
    def create_master_html(
        self,
        categorized_notebooks: Dict[str, List[Path]],
        publication_results: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Create a master HTML file with embedded navigation and all notebooks.
        
        Parameters
        ----------
        categorized_notebooks : dict
            Dictionary of categorized notebooks
        publication_results : dict
            Results from notebook publication
        """
        master_path = self.output_dir / "master.html"
        
        # Count statistics
        total_notebooks = sum(len(notebooks) for notebooks in categorized_notebooks.values())
        successful_publications = sum(
            1 for results in publication_results.values() 
            if results.get('html_success', False)
        )
        
        # Category descriptions
        category_descriptions = {
            'basic': 'Fundamental outlier detection and time series analysis methods',
            'intermediate': 'More complex analysis techniques and statistical methods',
            'advanced': 'Specialized algorithms and computationally intensive methods',
            'specialized': 'Domain-specific applications and advanced techniques',
            'other': 'Utility notebooks and demonstrations'
        }
        
        # Create master HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>SHMTools Example Notebooks - Interactive Documentation</title>
            <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                height: 100vh;
                overflow: hidden;
                background: #f8f9fa;
            }}
            
            .container {{
                display: flex;
                height: 100vh;
            }}
            
            .sidebar {{
                width: 300px;
                background: #ffffff;
                border-right: 2px solid #dee2e6;
                overflow-y: auto;
                box-shadow: 2px 0 4px rgba(0,0,0,0.1);
                z-index: 100;
            }}
            
            .sidebar-header {{
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white;
                padding: 20px;
                text-align: center;
                position: sticky;
                top: 0;
                z-index: 101;
            }}
            
            .sidebar-title {{
                font-size: 1.4em;
                font-weight: 600;
                margin-bottom: 5px;
            }}
            
            .sidebar-subtitle {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            
            .stats {{
                background: #f8f9fa;
                padding: 15px 20px;
                border-bottom: 1px solid #dee2e6;
                font-size: 0.85em;
                color: #6c757d;
            }}
            
            .stats-item {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
            }}
            
            .stats-item:last-child {{
                margin-bottom: 0;
            }}
            
            .nav-section {{
                border-bottom: 1px solid #f1f3f4;
            }}
            
            .nav-header {{
                background: #f8f9fa;
                padding: 15px 20px;
                font-weight: 600;
                color: #495057;
                cursor: pointer;
                border-bottom: 1px solid #dee2e6;
                transition: background-color 0.2s;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .nav-header:hover {{
                background: #e9ecef;
            }}
            
            .nav-header.active {{
                background: #e3f2fd;
                color: #1976d2;
            }}
            
            .nav-toggle {{
                font-size: 0.8em;
                transition: transform 0.2s;
            }}
            
            .nav-toggle.expanded {{
                transform: rotate(90deg);
            }}
            
            .nav-description {{
                font-size: 0.8em;
                color: #6c757d;
                font-weight: normal;
                margin-top: 3px;
            }}
            
            .nav-items {{
                display: none;
                background: white;
            }}
            
            .nav-items.expanded {{
                display: block;
            }}
            
            .nav-item {{
                padding: 12px 20px 12px 40px;
                cursor: pointer;
                border-bottom: 1px solid #f8f9fa;
                transition: all 0.2s;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .nav-item:hover {{
                background: #f8f9fa;
                padding-left: 45px;
            }}
            
            .nav-item.active {{
                background: #e3f2fd;
                color: #1976d2;
                font-weight: 500;
                border-left: 3px solid #2196f3;
            }}
            
            .nav-item.failed {{
                color: #dc3545;
                opacity: 0.7;
            }}
            
            .nav-item.failed:hover {{
                background: #f8d7da;
            }}
            
            .nav-status {{
                font-size: 0.7em;
                padding: 2px 6px;
                border-radius: 10px;
                font-weight: 500;
            }}
            
            .status-success {{
                background: #d4edda;
                color: #155724;
            }}
            
            .status-failed {{
                background: #f8d7da;
                color: #721c24;
            }}
            
            .main-content {{
                flex: 1;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }}
            
            .content-header {{
                background: white;
                padding: 20px 30px;
                border-bottom: 1px solid #dee2e6;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                z-index: 50;
            }}
            
            .content-title {{
                font-size: 1.8em;
                font-weight: 300;
                margin-bottom: 5px;
                color: #495057;
            }}
            
            .content-subtitle {{
                color: #6c757d;
                font-size: 1em;
            }}
            
            .content-body {{
                flex: 1;
                overflow-y: auto;
                padding: 0;
                background: white;
            }}
            
            .notebook-content {{
                display: none;
                height: 100%;
            }}
            
            .notebook-content.active {{
                display: block;
            }}
            
            .notebook-iframe {{
                width: 100%;
                height: 100%;
                border: none;
                background: white;
            }}
            
            .welcome-screen {{
                padding: 40px;
                text-align: center;
                color: #6c757d;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }}
            
            .welcome-icon {{
                font-size: 4em;
                margin-bottom: 20px;
                color: #007bff;
            }}
            
            .welcome-title {{
                font-size: 2em;
                margin-bottom: 15px;
                color: #495057;
            }}
            
            .welcome-text {{
                font-size: 1.1em;
                line-height: 1.6;
                max-width: 500px;
            }}
            
            .loading-screen {{
                display: none;
                padding: 40px;
                text-align: center;
                color: #6c757d;
                height: 100%;
                justify-content: center;
                align-items: center;
                flex-direction: column;
            }}
            
            .loading-spinner {{
                width: 40px;
                height: 40px;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #007bff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 20px;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            /* Responsive design */
            @media (max-width: 768px) {{
                .sidebar {{
                    width: 250px;
                }}
                
                .content-header {{
                    padding: 15px 20px;
                }}
                
                .content-title {{
                    font-size: 1.4em;
                }}
            }}
            
            /* Dark mode support */
            @media (prefers-color-scheme: dark) {{
                body {{
                    background: #1a1a1a;
                    color: #ffffff;
                }}
                
                .sidebar {{
                    background: #2d2d2d;
                    border-right-color: #404040;
                }}
                
                .main-content {{
                    background: #1a1a1a;
                }}
                
                .content-header {{
                    background: #2d2d2d;
                    border-bottom-color: #404040;
                }}
            }}
            </style>
        </head>
        <body>
            <div class="container">
                <!-- Sidebar Navigation -->
                <div class="sidebar">
                    <div class="sidebar-header">
                        <div class="sidebar-title">SHMTools Examples</div>
                        <div class="sidebar-subtitle">Interactive Documentation</div>
                    </div>
                    
                    <div class="stats">
                        <div class="stats-item">
                            <span>Total Notebooks:</span>
                            <span>{total_notebooks}</span>
                        </div>
                        <div class="stats-item">
                            <span>Successfully Published:</span>
                            <span>{successful_publications}</span>
                        </div>
                        <div class="stats-item">
                            <span>Categories:</span>
                            <span>{len([cat for cat, notebooks in categorized_notebooks.items() if notebooks])}</span>
                        </div>
                        <div class="stats-item">
                            <span>Generated:</span>
                            <span>{time.strftime('%Y-%m-%d')}</span>
                        </div>
                    </div>
                    
                    <div class="navigation">
        """
        
        # Add navigation for each category
        for category, notebooks in categorized_notebooks.items():
            if not notebooks:
                continue
                
            description = category_descriptions.get(category, 'Additional examples and utilities')
            
            html_content += f"""
                        <div class="nav-section">
                            <div class="nav-header" onclick="toggleSection('{category}')">
                                <div>
                                    <div>{category.title()} Level</div>
                                    <div class="nav-description">{description}</div>
                                </div>
                                <span class="nav-toggle" id="toggle-{category}">▶</span>
                            </div>
                            <div class="nav-items" id="items-{category}">
            """
            
            for notebook_path in notebooks:
                notebook_name = self.format_display_name(notebook_path.stem)
                notebook_id = f"{category}_{notebook_path.stem}"
                
                # Check if publication was successful
                result = publication_results.get(str(notebook_path), {})
                if result.get('html_success', False):
                    status_class = "status-success"
                    status_text = "OK"
                    item_class = "nav-item"
                    relative_path = f"{category}/{notebook_path.stem}.html"
                else:
                    status_class = "status-failed"
                    status_text = "X"
                    item_class = "nav-item failed"
                    relative_path = None
                
                onclick_action = f"loadNotebook('{notebook_id}', '{relative_path}', '{notebook_name}', '{category.title()}')" if relative_path else ""
                
                html_content += f"""
                                <div class="{item_class}" id="nav-{notebook_id}" onclick="{onclick_action}">
                                    <span>{notebook_name}</span>
                                    <span class="nav-status {status_class}">{status_text}</span>
                                </div>
                """
            
            html_content += """
                            </div>
                        </div>
            """
        
        # Add main content area and JavaScript
        html_content += f"""
                    </div>
                </div>
                
                <!-- Main Content Area -->
                <div class="main-content">
                    <div class="content-header">
                        <div class="content-title" id="content-title">Welcome to SHMTools Examples</div>
                        <div class="content-subtitle" id="content-subtitle">Select a notebook from the sidebar to begin exploring</div>
                    </div>
                    
                    <div class="content-body">
                        <!-- Welcome Screen -->
                        <div class="welcome-screen" id="welcome-screen">
                            <div class="welcome-icon">📊</div>
                            <div class="welcome-title">SHMTools Example Notebooks</div>
                            <div class="welcome-text">
                                Explore comprehensive examples of structural health monitoring techniques. 
                                Select any notebook from the sidebar to view interactive examples with 
                                complete code, explanations, and visualizations.
                            </div>
                        </div>
                        
                        <!-- Loading Screen -->
                        <div class="loading-screen" id="loading-screen">
                            <div class="loading-spinner"></div>
                            <div>Loading notebook...</div>
                        </div>
                        
                        <!-- Notebook Content Container -->
                        <div id="notebook-container">
                            <!-- Individual notebook content will be loaded here -->
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
            // Global state
            let currentNotebook = null;
            let loadedNotebooks = new Map();
            
            // Initialize the interface
            document.addEventListener('DOMContentLoaded', function() {{
                // Expand the first category by default
                const firstCategory = document.querySelector('.nav-header');
                if (firstCategory) {{
                    firstCategory.click();
                }}
            }});
            
            function toggleSection(category) {{
                const items = document.getElementById('items-' + category);
                const toggle = document.getElementById('toggle-' + category);
                const header = document.querySelector('.nav-header[onclick*="' + category + '"]');
                
                if (items.classList.contains('expanded')) {{
                    items.classList.remove('expanded');
                    toggle.classList.remove('expanded');
                    toggle.textContent = '▶';
                    header.classList.remove('active');
                }} else {{
                    items.classList.add('expanded');
                    toggle.classList.add('expanded');
                    toggle.textContent = '▼';
                    header.classList.add('active');
                }}
            }}
            
            async function loadNotebook(notebookId, relativePath, notebookName, category) {{
                if (!relativePath) {{
                    alert('This notebook failed to publish and cannot be displayed.');
                    return;
                }}
                
                // Update active navigation item
                document.querySelectorAll('.nav-item').forEach(item => {{
                    item.classList.remove('active');
                }});
                document.getElementById('nav-' + notebookId).classList.add('active');
                
                // Update header
                document.getElementById('content-title').textContent = notebookName;
                document.getElementById('content-subtitle').textContent = category + ' Level Example';
                
                // Show loading screen
                document.getElementById('welcome-screen').style.display = 'none';
                document.getElementById('loading-screen').style.display = 'flex';
                
                try {{
                    // Check if already loaded
                    if (loadedNotebooks.has(notebookId)) {{
                        showNotebook(notebookId);
                        return;
                    }}
                    
                    // Fetch and process the notebook HTML
                    const response = await fetch(relativePath);
                    if (!response.ok) {{
                        throw new Error('Failed to load notebook');
                    }}
                    
                    let html = await response.text();
                    
                    // Extract the main content (remove navigation and headers we added)
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    
                    // Remove our custom navigation and headers
                    const elementsToRemove = [
                        '.notebook-header',
                        '.navigation', 
                        '.footer'
                    ];
                    
                    elementsToRemove.forEach(selector => {{
                        const elements = doc.querySelectorAll(selector);
                        elements.forEach(el => el.remove());
                    }});
                    
                    // Get the cleaned content
                    const content = doc.body.innerHTML;
                    
                    // Create notebook container
                    const notebookDiv = document.createElement('div');
                    notebookDiv.className = 'notebook-content';
                    notebookDiv.id = 'notebook-' + notebookId;
                    notebookDiv.innerHTML = content;
                    
                    // Add to container
                    document.getElementById('notebook-container').appendChild(notebookDiv);
                    
                    // Cache the loaded notebook
                    loadedNotebooks.set(notebookId, true);
                    
                    // Show the notebook
                    showNotebook(notebookId);
                    
                }} catch (error) {{
                    console.error('Error loading notebook:', error);
                    console.error('Attempted to load path:', relativePath);
                    console.error('Full error details:', error.message);
                    document.getElementById('loading-screen').style.display = 'none';
                    document.getElementById('content-title').textContent = 'Error Loading Notebook';
                    document.getElementById('content-subtitle').textContent = 'Failed to load: ' + notebookName;
                    
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'welcome-screen';
                    errorDiv.innerHTML = `
                        <div class="welcome-icon">⚠️</div>
                        <div class="welcome-title">Failed to Load Notebook</div>
                        <div class="welcome-text">
                            Could not load the notebook "${{notebookName}}".<br><br>
                            <strong>Most likely cause:</strong> You're opening this file directly in your browser (file:// protocol), which blocks loading other HTML files for security reasons.<br><br>
                            <strong>Solution:</strong><br>
                            1. Open a terminal in the published_notebooks directory<br>
                            2. Run: <code>python -m http.server 8000</code><br>
                            3. Open: <a href="http://localhost:8000/master.html">http://localhost:8000/master.html</a><br><br>
                            <strong>Technical details:</strong><br>
                            Attempted path: ${{relativePath}}<br>
                            Error: ${{error.message}}
                        </div>
                    `;
                    
                    document.getElementById('notebook-container').appendChild(errorDiv);
                }}
            }}
            
            function showNotebook(notebookId) {{
                // Hide loading screen
                document.getElementById('loading-screen').style.display = 'none';
                
                // Hide all notebook content
                document.querySelectorAll('.notebook-content').forEach(content => {{
                    content.classList.remove('active');
                }});
                
                // Show selected notebook
                const targetNotebook = document.getElementById('notebook-' + notebookId);
                if (targetNotebook) {{
                    targetNotebook.classList.add('active');
                    currentNotebook = notebookId;
                    
                    // Scroll to top
                    document.querySelector('.content-body').scrollTop = 0;
                }}
            }}
            
            // Keyboard shortcuts
            document.addEventListener('keydown', function(e) {{
                if (e.ctrlKey || e.metaKey) {{
                    switch(e.key) {{
                        case 'k':
                            e.preventDefault();
                            // Focus on first navigation item
                            const firstNav = document.querySelector('.nav-item:not(.failed)');
                            if (firstNav) firstNav.focus();
                            break;
                    }}
                }}
            }});
            
            // Add search functionality (future enhancement)
            function searchNotebooks(query) {{
                // Implementation for searching through notebooks
                console.log('Search functionality to be implemented:', query);
            }}
            </script>
        </body>
        </html>
        """
        
        # Write master file
        with open(master_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"\nOK Master HTML file created: {master_path}")

    def create_index_page(
        self, 
        categorized_notebooks: Dict[str, List[Path]],
        publication_results: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Create an index page linking to all published notebooks.
        
        Parameters
        ----------
        categorized_notebooks : dict
            Dictionary of categorized notebooks
        publication_results : dict
            Results from notebook publication
        """
        index_path = self.output_dir / "index.html"
        
        # Count statistics
        total_notebooks = sum(len(notebooks) for notebooks in categorized_notebooks.values())
        successful_publications = sum(
            1 for results in publication_results.values() 
            if results.get('html_success', False)
        )
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>SHMTools Example Notebooks</title>
            <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .header {{
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white;
                padding: 40px;
                border-radius: 8px;
                margin-bottom: 30px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .stats {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .category {{
                background: white;
                margin: 20px 0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .category-header {{
                background: #f8f9fa;
                padding: 20px;
                border-bottom: 1px solid #dee2e6;
            }}
            .category-title {{ 
                margin: 0;
                color: #495057;
                font-size: 1.5em;
            }}
            .category-description {{ 
                margin: 5px 0 0 0;
                color: #6c757d;
            }}
            .notebook-list {{
                padding: 20px;
            }}
            .notebook-item {{
                padding: 15px;
                border-bottom: 1px solid #f8f9fa;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .notebook-item:last-child {{ border-bottom: none; }}
            .notebook-name {{ 
                font-weight: 500;
                color: #007bff;
                text-decoration: none;
                flex-grow: 1;
            }}
            .notebook-name:hover {{ color: #0056b3; }}
            .notebook-status {{
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8em;
                font-weight: 500;
                margin-left: 15px;
            }}
            .status-success {{
                background: #d4edda;
                color: #155724;
            }}
            .status-failed {{
                background: #f8d7da;
                color: #721c24;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
            }}
            .stat-card {{
                text-align: center;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #007bff;
            }}
            .stat-label {{
                color: #6c757d;
                text-transform: uppercase;
                font-size: 0.9em;
                letter-spacing: 1px;
            }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SHMTools Example Notebooks</h1>
                <p>Interactive examples demonstrating structural health monitoring techniques</p>
            </div>
            
            <div class="stats">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{total_notebooks}</div>
                        <div class="stat-label">Total Notebooks</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{successful_publications}</div>
                        <div class="stat-label">Successfully Published</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(categorized_notebooks)}</div>
                        <div class="stat-label">Categories</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{time.strftime('%Y-%m-%d')}</div>
                        <div class="stat-label">Generated</div>
                    </div>
                </div>
            </div>
        """
        
        # Category descriptions
        category_descriptions = {
            'basic': 'Fundamental outlier detection and time series analysis methods',
            'intermediate': 'More complex analysis techniques and statistical methods',
            'advanced': 'Specialized algorithms and computationally intensive methods',
            'specialized': 'Domain-specific applications and advanced techniques',
            'other': 'Utility notebooks and demonstrations'
        }
        
        # Add each category
        for category, notebooks in categorized_notebooks.items():
            if not notebooks:
                continue
                
            description = category_descriptions.get(category, 'Additional examples and utilities')
            
            html_content += f"""
            <div class="category">
                <div class="category-header">
                    <h2 class="category-title">{category.title()} Level</h2>
                    <p class="category-description">{description}</p>
                </div>
                <div class="notebook-list">
            """
            
            for notebook_path in notebooks:
                notebook_name = self.format_display_name(notebook_path.stem)
                relative_path = f"{category}/{notebook_path.stem}.html"
                
                # Check if publication was successful
                result = publication_results.get(str(notebook_path), {})
                if result.get('html_success', False):
                    status_class = "status-success"
                    status_text = "Published"
                    link_html = f'<a href="{relative_path}" class="notebook-name">{notebook_name}</a>'
                else:
                    status_class = "status-failed"
                    status_text = "Failed"
                    link_html = f'<span class="notebook-name" style="color: #6c757d;">{notebook_name}</span>'
                
                html_content += f"""
                <div class="notebook-item">
                    {link_html}
                    <span class="notebook-status {status_class}">{status_text}</span>
                </div>
                """
            
            html_content += """
                </div>
            </div>
            """
        
        html_content += """
            <div style="margin-top: 50px; padding: 20px; background: white; border-radius: 8px; text-align: center; color: #6c757d; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p>Generated from Jupyter notebooks in the SHMTools Python package</p>
                <p><a href="https://github.com/shmtools/shmtools-python">GitHub Repository</a> | 
                   <a href="https://shmtools.readthedocs.io">Documentation</a></p>
            </div>
        </body>
        </html>
        """
        
        # Write index file
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"\nOK Index page created: {index_path}")
    
    def publish_notebooks(
        self, 
        examples_dir: Path,
        skip_execution_errors: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Main method to find, execute, and publish all notebooks.
        
        Parameters
        ----------
        examples_dir : Path
            Path to examples directory
        skip_execution_errors : bool, default=True
            If True, publish notebooks even if execution fails
            
        Returns
        -------
        results : dict
            Publication results for each notebook
        """
        print("SHMTools Notebook Publisher")
        print("=" * 50)
        
        # Find notebooks
        print("\n1. Discovering notebooks...")
        categorized_notebooks = self.find_notebooks(examples_dir)
        
        total_notebooks = sum(len(notebooks) for notebooks in categorized_notebooks.values())
        print(f"   Found {total_notebooks} notebooks across {len(categorized_notebooks)} categories")
        
        for category, notebooks in categorized_notebooks.items():
            if notebooks:
                print(f"   {category.title()}: {len(notebooks)} notebooks")
        
        # Execute and publish each notebook
        print("\n2. Executing and publishing notebooks...")
        publication_results = {}
        
        for category, notebooks in categorized_notebooks.items():
            if not notebooks:
                continue
                
            print(f"\n  Processing {category.title()} notebooks:")
            
            for notebook_path in notebooks:
                result = {
                    'category': category,
                    'execution_success': False,
                    'html_success': False,
                    'execution_time': None,
                    'errors': []
                }
                
                try:
                    # Execute notebook
                    exec_success, executed_notebook, exec_metadata = self.execute_notebook(notebook_path)
                    
                    result['execution_success'] = exec_success
                    result['execution_time'] = exec_metadata.get('execution_time')
                    result['errors'] = exec_metadata.get('errors', [])
                    
                    # Convert to HTML (even if execution failed, if skip_execution_errors is True)
                    if exec_success or skip_execution_errors:
                        output_path = self.output_dir / category / f"{notebook_path.stem}.html"
                        html_success = self.convert_to_html(executed_notebook, output_path, notebook_path)
                        result['html_success'] = html_success
                        
                        if html_success:
                            result['output_path'] = str(output_path)
                    
                except Exception as e:
                    print(f"      X Failed to process {notebook_path.name}: {e}")
                    result['errors'].append({
                        'type': 'ProcessingError',
                        'error_name': e.__class__.__name__,
                        'error_value': str(e)
                    })
                
                publication_results[str(notebook_path)] = result
        
        # Create index page
        print("\n3. Creating index page...")
        self.create_index_page(categorized_notebooks, publication_results)
        
        # Create master HTML page with embedded navigation
        print("4. Creating master HTML page...")
        self.create_master_html(categorized_notebooks, publication_results)
        
        # Summary report
        print("\n" + "=" * 50)
        print("PUBLICATION SUMMARY")
        print("=" * 50)
        
        total_processed = len(publication_results)
        successful_executions = sum(1 for r in publication_results.values() if r['execution_success'])
        successful_publications = sum(1 for r in publication_results.values() if r['html_success'])
        
        print(f"Total notebooks processed: {total_processed}")
        print(f"Successful executions: {successful_executions}")
        print(f"Successful publications: {successful_publications}")
        print(f"Output directory: {self.output_dir.absolute()}")
        
        # Show failed notebooks
        failed_notebooks = [
            path for path, result in publication_results.items() 
            if not result['html_success']
        ]
        
        if failed_notebooks:
            print(f"\nFailed to publish ({len(failed_notebooks)}):")
            for path in failed_notebooks:
                notebook_name = Path(path).name
                result = publication_results[path]
                if result['errors']:
                    error_msg = result['errors'][0].get('error_value', 'Unknown error')
                    print(f"  X {notebook_name}: {error_msg}")
                else:
                    print(f"  X {notebook_name}: Unknown error")
        
        return publication_results


def main():
    """Main entry point for the notebook publisher."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Find, execute, and publish all shmtools example notebooks to HTML"
    )
    parser.add_argument(
        "--examples-dir", 
        type=Path,
        default=Path("examples/notebooks"),
        help="Path to examples notebooks directory (default: examples/notebooks)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path, 
        default=Path("published_notebooks"),
        help="Output directory for published HTML files (default: published_notebooks)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout in seconds for notebook execution (default: 900)"
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Publish notebooks even if execution fails"
    )
    parser.add_argument(
        "--clean",
        action="store_true", 
        help="Clean output directory before publishing"
    )
    
    args = parser.parse_args()
    
    # Validate examples directory
    if not args.examples_dir.exists():
        print(f"Error: Examples directory not found: {args.examples_dir}")
        print("Make sure you're running from the shmtools-python directory")
        return 1
    
    # Clean output directory if requested
    if args.clean and args.output_dir.exists():
        print(f"Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    # Create publisher and run
    publisher = NotebookPublisher(
        timeout=args.timeout,
        output_dir=str(args.output_dir)
    )
    
    try:
        results = publisher.publish_notebooks(
            args.examples_dir,
            skip_execution_errors=args.skip_errors
        )
        
        # Success if at least some notebooks were published
        successful_publications = sum(1 for r in results.values() if r['html_success'])
        
        if successful_publications > 0:
            print(f"\nSUCCESS: Published {successful_publications} notebooks!")
            print(f"\nViewing Options:")
            print(f"   - Quick start: cd {args.output_dir} && python start_server.py")
            print(f"   - Manual server: cd {args.output_dir} && python -m http.server 8000")
            print(f"   - Simple index: Open {args.output_dir / 'index.html'} directly")
            print(f"\nInteractive master documentation available at:")
            print(f"   http://localhost:8000/master.html")
            print(f"\nNote: The master.html requires a web server to load notebooks properly")
            return 0
        else:
            print("\nERROR: No notebooks were successfully published")
            return 1
            
    except Exception as e:
        print(f"\nERROR: Publication failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())