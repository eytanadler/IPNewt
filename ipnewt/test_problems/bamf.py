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

    Options
    =======
    lower : float
        Lower bound on all the states (default 0)
    upper : float
        Upper bound on all the states (default infinity)
    """

    def __init__(self, options={}):
        # Set options defaults
        opt_defaults = {"lower": 0.0, "upper": 20.}
        for opt in opt_defaults.keys():
            if opt not in options.keys():
                options[opt] = opt_defaults[opt]

        # Call the base class init with the correct number of states and bounds
        super().__init__(2, lower=options["lower"], upper=options["upper"], options=options)

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
