# !/usr/bin/env python
######################################################################
# \file setup.py
#######################################################################
from setuptools import setup, find_packages


install_requires = [
    "numpy",
    'plotly'
    ]

setup(
    name="human2robot",
    version=1.0,
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)