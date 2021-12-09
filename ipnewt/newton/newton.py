#!/usr/bin/env python
"""
@File    :   newton.py
@Time    :   2021/11/13
@Desc    :   Class definition for the Newton solver
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import copy

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================


class NewtonSolver(object):
    def __init__(self, options={}):
        """
        Valid Newton Solver Options:
            "atol": float (default=1e-6), the absolute convergence tolerance
            "rtol": float (default=1e-99), the relative convergence tolerance
            "maxiter": int (default=10), maximum iterations in Newton solver
            "beta": float (default=10.0), geometric penalty multiplier
            "rho": float (default=0.5), constant penalty scaling term
            "mu": float (default=1e-10), initial penalty parameter
            "mu max": float (default=1e6), maximum penalty parameter
            "tau": float (default=0.1), initial psuedo transient time step
            "tau max": float (default=1e20), maximum psuedo transient time step
            "gamma": float (default=2.0), pseudo transient time step geometric multiplier
            "pseudo transient" : if True (default), add the pseudo transient term to the Jacobian
            "interior penalty" : if True (default), add logarithmic penalty to Jacobian
            "residual penalty" : if False (default), add logarithmic penalty to residual vector
            "iprint": int (default=2), Newton solver print level
                        0 = print nothing
                        1 = print convergence message
                        2 = print iteration history
        """
        self.model = None
        self._iter_count = 0
        self.options = copy.deepcopy(options)
        self.linesearch = None
        self.linear_system = None
        self.mu_lower = None
        self.mu_upper = None
        self.data = {"atol": [], "rtol": [], "mu lower": [], "mu upper": [], "tau": [], "states": []}

        # Set options defaults
        opt_defaults = {
            "pseudo transient": True,
            "interior penalty": True,
            "residual penalty": False,
            "atol": 1e-6,
            "rtol": 1e-99,
            "maxiter": 10,
            "beta": 10.0,
            "rho": 0.5,
            "mu": 1e-10,
            "mu max": 1e6,
            "tau": 0.1,
            "tau max": 1e20,
            "gamma": 2.0,
            "iprint": 2,
        }
        for opt in opt_defaults.keys():
            if opt not in self.options.keys():
                self.options[opt] = opt_defaults[opt]

        self.data["options"] = self.options

    def _check_options(self):
        pass

    def _check_states(self, raise_error=False):
        """Checks that the states are all within the bounds.
        Optionally will raise an error if they are not. Otherwise,
        returns true if all the states are valid and false otherwise.

        Parameters
        ----------
        raise_error : bool, optional
            Raise an error if states not all within bounds, by default False

        Returns
        -------
        bool
            True if all states are within bounds, False otherwise
        """
        u = self.model.states
        lb = self.model.lower
        ub = self.model.upper

        feasible = np.all(np.logical_and(lb < u, u < ub))

        if raise_error and not feasible:
            infeas_states = np.where(np.logical_or(lb > u, u > ub))[0]
            raise ValueError(f"State(s) {', '.join(str(s) for s in infeas_states)} are not within the specified bounds")

        return feasible

    def setup(self):
        # Get the finite bound masks from the model
        lb_mask = self.model.lower_finite_mask
        ub_mask = self.model.upper_finite_mask

        # Set the initial mu for the linear system and line search
        n_states = len(self.model.states)
        if self.options["interior penalty"]:
            self.linear_system.mu_lower = np.full(np.sum(lb_mask), self.options["mu"])
            self.linear_system.mu_upper = np.full(np.sum(ub_mask), self.options["mu"])
            if self.linesearch:
                self.linesearch.mu_lower = np.full(np.sum(lb_mask), self.options["mu"])
                self.linesearch.mu_upper = np.full(np.sum(ub_mask), self.options["mu"])
            self.mu_lower = np.full(np.sum(lb_mask), self.options["mu"])
            self.mu_upper = np.full(np.sum(ub_mask), self.options["mu"])

        # Set the initial time step for the linear system
        if self.options["pseudo transient"]:
            self.linear_system.tau = self.options["tau"]

        # Set options for the linear system
        self.linear_system.options["jacobian penalty"] = self.options["interior penalty"]
        self.linear_system.options["pseudo transient"] = self.options["pseudo transient"]
        self.linear_system.options["residual penalty"] = self.options["residual penalty"]

        # Set the linear system model
        self.linear_system.model = self.model

        # Set options and model for the linesearch
        if self.linesearch:
            self.linesearch.options["residual penalty"] = self.options["interior penalty"]
            self.linesearch.model = self.model

        self.data["lower bounds"] = self.model.lower
        self.data["upper bounds"] = self.model.upper

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

    def _update_penalty(self):
        beta = self.options["beta"]
        rho = self.options["rho"]

        # Use u and du to compute the full step pure Newton wanted to take
        u = self.data["states"][-2]
        du = self.du_newton  # use the step from Newton as opposed to with the
        # bounds enforcement handed by the linesearch

        lb_mask = self.model.lower_finite_mask
        ub_mask = self.model.upper_finite_mask

        lb = self.model.lower
        ub = self.model.upper

        # Initialize d_alpha to zeros
        # We only want to store and calculate d_alpha for states that
        # have bounds
        d_alpha_lower = np.zeros(np.count_nonzero(lb_mask))
        d_alpha_upper = np.zeros(np.count_nonzero(ub_mask))

        # Compute d_alpha for all states with finite bounds
        t_lower = lb[lb_mask] - (u[lb_mask] + du[lb_mask])
        t_upper = (u[ub_mask] + du[ub_mask]) - ub[ub_mask]

        # d_alpha > 0 means that the state has violated a bound
        # d_alpha < 0 means that the state has not violated a bound
        d_alpha_lower = t_lower / np.abs(du[lb_mask])
        d_alpha_upper = t_upper / np.abs(du[ub_mask])

        # We want to set all values of d_alpha < 0 to 0 so that the
        # penalty logic and formula won't do anything for those terms
        d_alpha_lower = np.where(d_alpha_lower < 0, 0, d_alpha_lower)
        d_alpha_upper = np.where(d_alpha_upper < 0, 0, d_alpha_upper)

        # Now compute the penalty update
        if d_alpha_lower.size > 0:
            self.mu_lower *= beta * d_alpha_lower + rho

        if d_alpha_upper.size > 0:
            self.mu_upper *= beta * d_alpha_upper + rho

        if np.any(self.mu_lower > self.options["mu max"]):
            self.mu_lower[:] = self.options["mu max"]
            if self.options["iprint"] > 1:
                print("Warning: Maximum penalty value reached.")

        if np.any(self.mu_upper > self.options["mu max"]):
            self.mu_lower[:] = self.options["mu max"]
            if self.options["iprint"] > 1:
                print("Warning: Maximum penalty value reached.")

    def solve(self):
        if self.options["iprint"] > 0:
            print(
                "\n    ____________________   __              _____ \n"
                "    ____  _/__  __ \__  | / /_______      ___  /_\n"
                "    __  / __  /_/ /_   |/ /_  _ \_ | /| / /  __/ \n"
                "    __/ /  _  ____/_  /|  / /  __/_ |/ |/ // /_  \n"
                "    /___/  /_/     /_/ |_/  \___/____/|__/ \__/  \n"
                "                                                 \n"
            )

        # Check that the states all start within the bounds
        self._check_states(raise_error=True)

        # Get the options
        options = self.options
        max_iter = options["maxiter"]
        atol = options["atol"]
        rtol = options["rtol"]

        # Set the converged flag
        converged = False

        # Get the model
        model = self.model

        # Start the solver by running the model and updating the
        # Jacobian
        self._start_solver()

        # Compute the initial residual norm
        phi0 = self._objective()
        self.data["atol"].append(phi0)
        self.data["rtol"].append(1.0)
        if self.options["interior penalty"]:
            self.data["mu lower"].append(self.mu_lower.copy())
            self.data["mu upper"].append(self.mu_upper.copy())
        if self.options["pseudo transient"]:
            self.data["tau"].append(self.linear_system.tau)
        self.data["states"].append(self.model.states)

        # Print the solver info
        if self.options["iprint"] > 1:
            print(f"| NL Newton: {self._iter_count} {phi0}")

        while self._iter_count <= max_iter:
            # Logic for a single Newton iteration
            if self._iter_count > 0:
                # If interior penalty method is turned on, update the penalty parameters
                if self.options["interior penalty"]:
                    self._update_penalty()
                    self.linear_system.mu_lower = self.mu_lower
                    self.linear_system.mu_upper = self.mu_upper
                    if self.linesearch:
                        self.linesearch.mu_lower = self.mu_lower
                        self.linesearch.mu_upper = self.mu_upper

                # Geometrically update the pseduo transient term
                if self.options["pseudo transient"]:
                    self.linear_system.tau *= self.options["gamma"]
                    if self.linear_system.tau > self.options["tau max"]:
                        self.linear_system.tau = self.options["tau max"]
                        if self.options["iprint"] > 1:
                            print("Warning: Maximum pseudo transient time step reached.")

                # Run the model and update the linear system
                model.run()
                self.linear_system.update()

            # Solve the linear system
            self.linear_system.factorize()
            self.linear_system.solve()

            # If the adaptive interior penalty is used, need to store the step
            # from the Newton solver to use when computing the penalty update
            if self.options["interior penalty"]:
                self.du_newton = np.copy(self.linear_system.du)

            # Run the linesearch
            if self.linesearch:
                self.linesearch.solve(self.linear_system.du)
            else:
                self.model.states = self.model.states + self.linear_system.du
                self.model.run()

            phi = self._objective()

            self._iter_count += 1

            # Print the solver info
            if self.options["iprint"] > 1:
                print(f"| NL Newton: {self._iter_count} {phi} {phi/phi0}")

            self.data["atol"].append(phi)
            self.data["rtol"].append(phi / phi0)
            if self.options["interior penalty"]:
                self.data["mu lower"].append(self.mu_lower.copy())
                self.data["mu upper"].append(self.mu_upper.copy())
            if self.options["pseudo transient"]:
                self.data["tau"].append(self.linear_system.tau)
            self.data["states"].append(self.model.states)

            # Check the convergence tolerances
            if phi < atol:
                converged = True
                break

            if phi / phi0 < rtol:
                converged = True
                break

        if self.options["iprint"] > 0:
            if converged:
                print(f"| NL Newton converged in {self._iter_count} iterations")
            else:
                print("| NL Newton failed to converge to the requested tolerance")
