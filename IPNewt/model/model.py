#!/usr/bin/env python
"""
@File    :   model.py
@Time    :   2021/11/13
@Desc    :   Class definition for the top level Model class
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
from newton.newton import NewtonSolver
from linear.linear_solver import ScipySolver


class Model(object):
    def __init__(self):
        """
        Initialize all attributed
        """
        self.options = {}
        self.nonlinear_solver = NewtonSolver()
        self.nonlinear_solver.linear_solver = ScipySolver()

    def _check_options(self):
        pass

    def setup(self):
        # Any setup operations go here
        pass

    def run(self):
        """
        Drives the residuals in the model to zero using the nonlinear solver.
        """
        pass

    def compute_residuals(self):
        """
        Compute the residuals of the nonlinear equations in the model.
        """
        pass

    def compute_partials(self):
        """
        Compute the residuals of the nonlinear equations in the model.
        """
        pass
