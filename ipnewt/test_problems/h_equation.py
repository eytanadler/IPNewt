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
from copy import copy

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
    c : float
        Constant in the H-equation between 0 and 1 (default 0.5)
    """

    def __init__(self, options={}):
        # Set options defaults
        opt_defaults = {"n_states": 2, "lower": 0.0, "upper": np.inf, "c": 0.5}
        for opt in opt_defaults.keys():
            if opt not in options.keys():
                options[opt] = opt_defaults[opt]

        # Call the base class init with the correct number of states and bounds
        super().__init__(options["n_states"], lower=options["lower"], upper=options["upper"], options=options)

    def compute_residuals(self, u, res):
        """
        Compute the residuals of the nonlinear equations in the model.
        """
        n = self.options["n_states"]
        c = self.options["c"]

        mu = (np.arange(1, n + 1) - 0.5) / n
        for i in range(n):
            res[i] = u[i] - 1 / (1 - c / (2 * n) * np.sum(mu[i] * u / (mu[i] + mu)))

    def compute_jacobian(self, u, J):
        """
        Compute the residuals of the nonlinear equations in the model.
        """
        n = self.options["n_states"]
        c = self.options["c"]

        mu = (np.arange(1, n + 1) - 0.5) / n
        for i in range(n):
            denom = 1 - c / (2 * n) * np.sum(mu[i] * u / (mu[i] + mu))
            for j in range(n):
                J[i, j] = copy(denom) ** -2
                J[i, j] *= -c / (2 * n) * mu[i] / (mu[i] + mu[j])
                if i == j:
                    J[i, j] += 1
