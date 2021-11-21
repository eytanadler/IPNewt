#!/usr/bin/env python
"""
@File    :   newton.py
@Time    :   2021/11/13
@Desc    :   Class definition for the Newton solver
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


class NewtonSolver(object):
    def __init__(self):
        """
        Initialize all attributed
        """
        self.model = None
        self._iter_count = 0
        self.options = {}
        self.linesearch = None
        self.linear_system = None

    def _check_options(self):
        pass

    def setup(self):
        # Any setup operations go here
        pass

    def _start_solver(self):
        """
        Starts the solver by running the model and setting up the
        Jacobian.
        """
        # We can include error handling inside the function that runs
        # the model.
        self.model.run()
        self.linear_system.update()

    def _objective(self):
        return np.linalg.norm(self.model.residuals)

    def solve(self):
        # Get the options
        options = self.options
        max_iter = options["maxiter"]
        atol = options["atol"]
        rtol = options["rtol"]

        # Set the converged flag
        converged = False

        # Get the model
        model = self.model

        # Compute the initial residual norm
        phi0 = self._objective()

        # Start the solver by running the model and updating the
        # Jacobian
        self._start_solver()

        while self._iter_count <= max_iter:
            # Logic for a single Newton iteration
            if self._iter_count > 0:
                model.run()
                self.linear_system.update()

            # Solve the linear system
            self.linear_system.factorize()
            du = self.linear_system.solve()

            # Run the linesearch
            self.linesearch.solve(model.states, du)

            phi = self._objective()

            # Print the solver info
            print(f"| NL Newton: {self._iter_count} {phi}")

            # Check the convergence tolerances
            if phi < atol:
                converged = True
                break

            if phi / phi0 < rtol:
                converged = True
                break

        if converged:
            print(f"| NL Newton converged in {self._iter_count} iterations")
        else:
            print("| NL Newton failed to converge to the requested tolerance")
