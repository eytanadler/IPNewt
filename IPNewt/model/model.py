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


class Model(object):
    def __init__(self):
        """
        Initialize all attributed
        """
        self.options = {}
        self.residual = None
        self.states = None
        self.jacobian = None

    def _check_options(self):
        pass

    def _compute_jacobian(self):
        """
        Computes the partial derivatives and sets up the Jacobian matrix.
        """
        pass

    def compute_residuals(self):
        """
        USER DEFINED

        Computes the residuals of the model (which all equal zero when the
        model is solved). Sets self.residual to a numpy vector.
        """
        pass

    def compute_partials(self):
        """
        USER DEFINED

        Compute the partial derivatives of the residuals of the nonlinear
        equations in the model.
        """
        pass
