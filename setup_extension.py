"""
Setup script for SHM Function Selector Jupyter extension
"""

from setuptools import setup, find_packages

setup(
    name='jupyter_shm_extension',
    version='0.1.0',
    description='Jupyter Notebook extension for SHM function selection and parameter linking',
    author='SHMTools Team',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'jupyter_shm_extension': [
            'static/*'
        ]
    },
    install_requires=[
        'notebook',
        'tornado',
    ],
    zip_safe=False
)