#!/usr/bin/env python
"""
@File    :   level_hard.py
@Time    :   2021/11/13
@Desc    :   Model for the Bounded And Multimodal Function test problem
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
from ipnewt.api import Model


class BAMF(Model):
    """
    Defines the Bounded And Multimodal Function test problem (one we made up).
    """

    def __init__(self):
        # Call the base class init with the correct number of states and bounds
        super().__init__(2, lower=[-np.inf, 0.])

    def compute_residuals(self, u, res):
        """
        Compute the residuals of the nonlinear equations in the model.
        """
        res[0] = u[0] + u[1] + (np.sin(u[0]) + np.sin(u[1]))**2
        res[1] = u[0] + np.log(u[1] + 1)

    def compute_jacobian(self, u, J):
        """
        Compute the residuals of the nonlinear equations in the model.
        """
        J[0, 0] = 1 + 2 * np.sin(u[0]) * np.cos(u[0])  # dr0/du0
        J[0, 1] = 1 + 2 * np.sin(u[1]) * np.cos(u[1])  # dr0/du1

        J[1, 0] = 1  # dr1/du0
        J[1, 1] = 1 / (u[1] + 1)  # dr1/du1
