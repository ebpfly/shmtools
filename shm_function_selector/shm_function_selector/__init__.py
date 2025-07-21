"""A JupyterLab extension providing SHM function selector and parameter linking."""

from ._version import __version__


def _jupyter_labextension_paths():
    """Return metadata for the JupyterLab extension."""
    return [{
        "src": "labextension",
        "dest": "shm-function-selector"
    }]


def _jupyter_server_extension_points():
    """Return metadata for the Jupyter server extension."""
    return [{
        "module": "shm_function_selector"
    }]


def _load_jupyter_server_extension(server_app):
    """Load the Jupyter server extension."""
    from .handlers import setup_handlers
    web_app = server_app.web_app
    setup_handlers(web_app)
    server_app.log.info("SHM Function Selector server extension loaded")
