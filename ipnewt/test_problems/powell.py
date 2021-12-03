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
from ipnewt.api import Model


class Powell(Model):
    """
    Defines the Powell test problem. Can optionally change the
    bound locations for the two states.

    Options
    =======
    one_lower : float
        Lower bound on the first state (default -10)
    two_lower : float
        Lower bound on the second state (default -10)
    one_upper : float
        Upper bound on the first state (default 15)
    two_upper : float
        Upper bound on the second state (default 15)
    """
    def __init__(self, options={}):
        # Set options defaults
        opt_defaults = {
            "one_lower": -10.,
            "two_lower": -10.,
            "one_upper": 15.,
            "two_upper": 15.
        }
        for opt in opt_defaults.keys():
            if opt not in options.keys():
                options[opt] = opt_defaults[opt]

        # Call the base class init with the correct number of states and bounds
        super().__init__(2, lower=[options["one_lower"], options["two_lower"]],
                         upper=[options["one_upper"], options["two_upper"]])

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
