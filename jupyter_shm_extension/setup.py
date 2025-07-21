#!/usr/bin/env python
"""Setup for SHM Jupyter Extension"""

from setuptools import setup, find_packages
import os
import json

# Get the current directory
here = os.path.abspath(os.path.dirname(__file__))

# Read version from package.json if it exists, otherwise use default
version = "0.2.0"

setup(
    name="jupyter_shm_extension",
    version=version,
    description="Jupyter Notebook extension for SHM function selection and parameter linking",
    long_description="A Jupyter Notebook extension that provides dropdown function selection and right-click parameter linking for SHM analysis workflows.",
    author="SHMTools Team", 
    author_email="",
    url="",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'jupyter_shm_extension': [
            'static/*'
        ]
    },
    data_files=[
        ('share/jupyter/nbextensions/shm_function_selector', [
            'static/main.js',
            'static/main.css'
        ]),
        ('etc/jupyter/nbconfig/notebook.d', ['shm_function_selector.json']),
        ('etc/jupyter/jupyter_server_config.d', ['jupyter-config/jupyter_server_config.d/jupyter_shm_extension.json'])
    ],
    install_requires=[
        'notebook>=4.0',
        'tornado>=4.0',
    ],
    python_requires='>=3.6',
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'jupyter-shm-extension = jupyter_shm_extension:main',
        ],
    },
    classifiers=[
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)