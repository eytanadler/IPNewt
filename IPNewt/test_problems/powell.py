#!/usr/bin/env python
"""
@File    :   powell.py
@Time    :   2021/11/13
@Desc    :   Model for the Powell test problem
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
from IPNewt.api import Model


class Powell(Model):
    def compute_residuals(self, u, res):
        """
        Compute the residuals of the nonlinear equations in the model.
        """
        res[0] = 1e4 * u[0] * u[1] - 1
        res[1] = np.exp(-u[0]) + np.exp(-u[1]) - 1.0001

    def compute_jacobian(self, u, J):
        """
        Compute the residuals of the nonlinear equations in the model.
        """
        J[0, 0] = 1e4 * u[1]  # dr0/du0
        J[0, 1] = 1e4 * u[0]  # dr0/du1

        J[1, 0] = -np.exp(-u[0])  # dr1/du0
        J[1, 1] = -np.exp(-u[1])  # dr1/du1
