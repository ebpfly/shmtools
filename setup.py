#!/usr/bin/env python
"""Setup script for SHMTools Python package."""

from setuptools import setup, find_packages
import os

# Read the README file for long description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f 
                   if line.strip() and not line.startswith('#')]

# Read dev requirements
with open(os.path.join(this_directory, 'requirements-dev.txt'), encoding='utf-8') as f:
    dev_requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith('#')]

# Read advanced requirements
with open(os.path.join(this_directory, 'requirements-advanced.txt'), encoding='utf-8') as f:
    advanced_requirements = [line.strip() for line in f 
                            if line.strip() and not line.startswith('#')]

# Read hardware requirements
with open(os.path.join(this_directory, 'requirements-hardware.txt'), encoding='utf-8') as f:
    hardware_requirements = [line.strip() for line in f 
                            if line.strip() and not line.startswith('#')]

setup(
    name='shmtools',
    version='0.1.0',
    description='Python-based Structural Health Monitoring Toolkit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='SHMTools Development Team',
    url='https://github.com/ebpfly/shm',
    license='BSD-3-Clause',
    packages=find_packages(include=['shmtools', 'shmtools.*']),
    python_requires='>=3.10',
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements,
        'advanced': advanced_requirements,
        'hardware': hardware_requirements,
        'all': dev_requirements + advanced_requirements + hardware_requirements,
    },
    # entry_points={
    #     'console_scripts': [
    #         'shmtools-gui=bokeh_shmtools.app:main',  # Archived - bokeh GUI on pause
    #     ],
    # },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    keywords='structural-health-monitoring signal-processing machine-learning modal-analysis',
    include_package_data=True,
    zip_safe=False,
)