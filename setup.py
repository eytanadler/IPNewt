#!/usr/bin/env python
"""
@File    :   setup.py
@Time    :   2021/11/13
@Desc    :   Install setup for the IPNewt package
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import sys
from setuptools import setup

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================

__version__ = "0.0.1"

setup(
    name="IPNewt",
    version=__version__,
    description="IPNewt framework",
    long_description="""IPNewt is a toolbox for solving box-constrained
    nonlinear problems using pseduo-transient continuation coupled
    with interior penalty methods.
    """,
    author="Eytan Adler, Andrew Lamkin",
    author_email="eytana@umich.edu, lamkina@umich.edu",
    license="MIT License",
    packages=["ipnewt", "ipnewt.newton", "ipnewt.linear", "ipnewt.line_search", "ipnewt.model"],
    python_requires=">=3.6",
    install_requires=["numpy", "scipy",],
)
