#!/usr/bin/env python
"""
@File    :   h-equation.py
@Time    :   2021/12/3
@Desc    :   Model for the Chandrasekhar's H-equation test problem
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


class HEquation(Model):
    """
    Defines the Chandrasekhar H-equation test problem. Can optionally change the
    bound locations for the two states.

    Options
    =======
    n_states : int
        Number of states to use in the problem (default 2)
    lower : float
        Lower bound on all the states (default 0)
    upper : float
        Upper bound on all the states (default infinity)
    """
    def __init__(self, options={}):
        # Set options defaults
        opt_defaults = {
            "n_states": 2,
            "lower": 0.,
            "upper": np.inf
        }
        for opt in opt_defaults.keys():
            if opt not in options.keys():
                options[opt] = opt_defaults[opt]

        # Call the base class init with the correct number of states and bounds
        super().__init__(options["n_states"], lower=options["lower"], upper=options["upper"],
                         options=options)

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
