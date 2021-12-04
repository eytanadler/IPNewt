#!/usr/bin/env python
"""
@File    :   linear_system.py
@Time    :   2021/11/13
@Desc    :   Class definition for the linear system
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import copy

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from scipy.linalg import lu_factor, lu_solve

# ==============================================================================
# Extension modules
# ==============================================================================


class LinearSystem(object):
    def __init__(self, options={}):
        """
        Valid LinearSystem options:
            "pseudo transient" : if True (default), add the pseudo transient term to the Jacobian
            "jacboain penalty" : if True (default), add logarithmic penalty to Jacobian
            "residual penalty" : if True (default), add logarithmic penalty to residual vector
        """
        self.jacobian = None
        self.model = None
        self.residuals = None
        self.mu_lower = None
        self.mu_upper = None
        self.tau = None
        self.du = None
        self.options = copy.deepcopy(options)

        # Set option defaults if not already defined
        opt_defaults = {"pseudo transient": True, "jacobian penalty": True, "residual penalty": True}
        for opt in opt_defaults.keys():
            if opt not in self.options.keys():
                self.options[opt] = opt_defaults[opt]

    def _check_options(self):
        pass

    def update(self):
        """
        Update the Jacobian and residuals with the model's Jacobian and residuals.
        """
        # Get the residuals and Jacobian from the model
        self.residuals = self.model.residuals.copy()
        self.jacobian = self.model.jacobian.copy()

        # Add extra terms if the options say so
        if self.options["pseudo transient"]:
            self.update_pseudo_transient_jacobian()

        if self.options["jacobian penalty"]:
            self.update_penalty_jacobian()

        if self.options["residual penalty"]:
            self.update_penalty_residual()

    def update_pseudo_transient_jacobian(self):
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
        ub = self.model.upper
        lb = self.model.lower

        # Get the states from the model
        u = self.model.states

        # Initialize the penalty derivative array
        dp_du = np.zeros(len(u))

        # Compute the denominator of the penalty derivative terms
        # associated with the lower and upper bounds
        t_lower = u[lb_mask] - lb[lb_mask]
        t_upper = ub[ub_mask] - u[ub_mask]

        # Only compute the derivative if finite bounds exist
        if t_lower.size > 0:
            dp_du[lb_mask] = -self.mu_lower / (t_lower + 1e-10)

        if t_upper.size > 0:
            dp_du[ub_mask] = -self.mu_upper / (t_upper + 1e-10)

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

        # Compute the penalized residual for each state that has a
        # corresponding finite bound
        self.residuals[lb_mask] -= self.mu_lower * np.log(u[lb_mask] - lb[lb_mask])
        self.residuals[ub_mask] -= self.mu_upper * np.log(ub[ub_mask] - u[ub_mask])

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
        # Need to flip the residuals to negative
        b = -self.model.residuals
        self.du = lu_solve(self.lu, b)
