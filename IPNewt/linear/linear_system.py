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
from scipy.linalg import lu_factor, lu_solve

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
        self.linear_solver = None
        self.model = None
        self.residuals = None
        self.mu = None
        self.tau = None
        self.options = {}

    def _check_options(self):
        pass

    def setup(self):
        # Any setup operations go here
        pass

    def update_pt_jacobian(self):
        """
        Add pseudo transient component to Jacobian.
        """
        self.jacobian += np.identity(self.jacobian.shape[0], dtype=float) * (1 / self.tau)

    def update_penalty_jacobian(self):
        """
        Add penalty to Jacobian.
        """
        # Get the finite bound masks from the model
        lb_mask = self.model.lower_finite_mask
        ub_mask = self.model.upper_finite_mask

        # Get the bounds from the model
        ub = self.model.upper_bounds
        lb = self.model.lower_bounds

        # Get the states from the model
        u = self.model.states

        # Initialize the penalty derivative array
        dp_du = np.zeros(len(u))

        # Compute the denominator of the penalty derivative terms
        # associated with the lower and upper bounds
        t_lower = u[lb_mask] - lb[lb_mask]
        t_upper = ub[ub_mask] - u[lb_mask]

        # Only compute the derivative if finite bounds exist
        if t_lower.size > 0:
            dp_du[lb_mask] = -self.mu / (t_lower + 1e-10)

        if t_upper.size > 0:
            dp_du[ub_mask] = -self.mu / (t_upper + 1e-10)

        # Add the penalty jacobian to the problem jacobian
        self.jacobian += np.diag(dp_du)

    def update_penalty_residual(self):
        """
        Add penalty to residual vector.
        """
        # Get the finite bound masks from the model
        lb_mask = self.model.lower_finite_mask
        ub_mask = self.model.upper_finite_mask

        # Get the bounds from the model
        ub = self.model.upper
        lb = self.model.lower

        # Get the states from the model
        u = self.model.states

        # Get the residuals from the model
        self.residuals = self.model.residuals.copy()

        # Compute the penalized residual for each state that has a
        # corresponding finite bound
        self.residuals[lb_mask] = self.model.residuals[lb_mask] - np.sum(self.mu * np.log(u[lb_mask] - lb[lb_mask]))
        self.residuals[ub_mask] = self.model.residuals[ub_mask] - np.sum(self.mu * np.log(ub[ub_mask] - u[ub_mask]))

    def factorize(self):
        pass

    def solve(self):
        pass


class LULinearSystem(LinearSystem):
    def __init__(self):
        super().__init__()
        self.lu = None

    def check_options(self):
        pass

    def factorize(self):
        """
        Peform LU factorization and store the LU matrix operator.
        """
        self.lu = lu_factor(self.jacobian)

    def solve(self):
        """Solve the linear system using LU factorization.

        Returns
        -------
        ndarray
            The newton update vector
        """
        b = self.model.residuals
        return lu_solve(self.lu, b)
