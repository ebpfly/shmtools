"""
SHM Function Selector - JupyterLab Extension

This package provides introspection and extension installation utilities
for the SHM Function Selector JupyterLab extension.
"""

__version__ = "0.1.0"

# Import utilities for direct access
from .introspection import discover_functions_locally, summarize_discovered_parameters
from .jupyter_extension_installer import install_extension, uninstall_extension

# Server extension hooks for JupyterLab
def _jupyter_labextension_paths():
    """Return metadata for the JupyterLab extension."""
    return [{
        "src": "shm_function_selector/labextension",
        "dest": "shm-function-selector"
    }]


def _jupyter_server_extension_points():
    """Return metadata for the Jupyter server extension."""
    return [{
        "module": "shm_function_selector"
    }]


def _load_jupyter_server_extension(server_app):
    """Load the Jupyter server extension."""
    from .shm_function_selector.handlers import setup_handlers
    web_app = server_app.web_app
    setup_handlers(web_app)
    server_app.log.info("SHM Function Selector server extension loaded")

__all__ = [
    "discover_functions_locally", 
    "summarize_discovered_parameters",
    "install_extension",
    "uninstall_extension",
    "_jupyter_labextension_paths",
    "_jupyter_server_extension_points", 
    "_load_jupyter_server_extension"
]