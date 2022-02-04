#!/usr/bin/env python
"""
@File    :   linear_res_min.py
@Time    :   2022/02/02
@Desc    :   Linear solver that minimizes the L2 norm of the linearized residuals within the bounds
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import copy

# ==============================================================================
# External Python modules
# ==============================================================================
from scipy.optimize import lsq_linear
from scipy.linalg import lu_factor, lu_solve

# ==============================================================================
# Extension modules
# ==============================================================================
from ipnewt.linear.linear_system import LinearSystem

class MinLinResLinearSystem(LinearSystem):
    """
    Solves for the states that minimize the L2 norm of the linear system
    residual, subject to the states being within the bounds.
    """
    def __init__(self, options={}):
        """
        Valid LinearSystem options:
            "lu switch" : (default 100) number of model function evaluations after which to switch to a pure linear solver
        """
        super().__init__(options)

        # Set option defaults if not already defined
        opt_defaults = {"lu switch": 100}
        for opt in opt_defaults.keys():
            if opt not in self.options.keys():
                self.options[opt] = opt_defaults[opt]

    def solve(self):
        """Solve the linear system by either minimizing the L2 norm of the linear
        system residual within the bounds or solving the linear system directly.
        Depends on if the number of function evaluations is less or greater than
        the "lu switch" option.
        """
        # Need to flip the residuals to negative
        A = self.jacobian
        b = -self.residuals

        # Shift the bounds to bound the step (delta x) as opposed to the absolute states
        step_lower = self.model.lower - self.model.states
        step_upper = self.model.upper - self.model.states

        if self.model.func_calls < self.options["lu switch"]:
            opt_result = lsq_linear(A, b, bounds=(step_lower, step_upper), verbose=2)
            self.du = opt_result["x"]
        else:
            self.du = lu_solve(lu_factor(self.jacobian), b)
