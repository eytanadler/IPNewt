#!/usr/bin/env python
"""
@File    :   linear_system.py
@Time    :   2021/11/13
@Desc    :   Class definition for the linear system
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import scipy

# ==============================================================================
# Extension modules
# ==============================================================================


class LinearSystem(object):
    def __init__(self):
        """
        Initialize all attributed
        """
        self._problem = None
        self.jacobian = None
        self.states = None
        self.residuals = None
        self.options = {}

    def _check_options(self):
        pass

    def setup(self):
        # Any setup operations go here
        pass

    def add_pseudo_transient(self):
        """
        Add pseudo transient component to Jacobian.
        """
        pass

    def add_jacobian_penalty(self):
        """
        Add penalty to Jacobian.
        """
        pass

    def add_residual_penalty(self):
        """
        Add penalty to residual vector.
        """
        pass
