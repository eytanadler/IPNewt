#!/usr/bin/env python
"""
@File    :   line_search.py
@Time    :   2021/11/13
@Desc    :   Class definition for the line search
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


class LineSearch(object):
    def __init__(self, options={}):
        """
        Valid linesearch options:
            "alpha": float (default=1.0), initial lineserach step length
            "alpha max: float(default=2.0), initial max linesearch step length for forward tracking mode
            "maxiter": int (default=3), maximum linesearch iterations
            "residual penalty": True (default), add logarithmic penalty to residual vector
            "iprint": int (default=2), linesearch print level
                        0 = print nothing
                        1 = print convergence message
                        2 = print iteration history
        """
        self.model = None
        self.mu_lower = None
        self.mu_upper = None
        self._iter_count = 0
        self.options = copy.deepcopy(options)
        self.data = {"data": []}

        # Set options defaults
        opt_defaults = {
            "alpha": 1.0,
            "maxiter": 2,
            "residual penalty": True,
            "alpha max": 10.0,
            "iprint": 2,
        }
        for opt in opt_defaults.keys():
            if opt not in self.options.keys():
                self.options[opt] = opt_defaults[opt]

    def _check_options(self):
        pass

    def setup(self):
        # Any setup operations go here
        pass

    def _enforce_bounds(self, step, alpha):
        """Enforces bounds on states along the newton step direction

        Parameters
        ----------
        step : array
            Newton step vector
        alpha : float
            Step length

        Returns
        -------
        bool
            True if the step was limited by the bounds enforcement, False otherwise
        """
        lb = self.model.lower
        ub = self.model.upper

        return _enforce_bounds_vector(self.model.states, step, alpha, lb, ub)

    def _objective(self):
        """Computes the objective function for the linesearch.  If the
        linesearch uses the penalized residual, this function will
        compute the residual before the objective.

        Returns
        -------
        float
            L2 norm of the residual vector
        """
        if self.options["residual penalty"]:
            u = self.model.states
            lb = self.model.lower
            ub = self.model.upper
            lb_mask = self.model.lower_finite_mask
            ub_mask = self.model.upper_finite_mask

            penalty = np.zeros(u.size)

            t_lower = u[lb_mask] - lb[lb_mask]
            t_upper = ub[ub_mask] - u[ub_mask]

            if t_lower.size > 0:
                penalty[lb_mask] += np.sum(self.mu_lower * -np.log(t_lower + 1e-10))

            if t_upper.size > 0:
                penalty[ub_mask] += np.sum(self.mu_upper * -np.log(t_upper + 1e-10))

            residuals = self.model.residuals + penalty
        else:
            residuals = self.model.residuals

        return np.linalg.norm(residuals)

    def _update_states(self, alpha, du):
        self.model.states = self.model.states + (alpha * du)

    def solve(self):
        pass


class AdaptiveLineSearch(LineSearch):
    def __init__(self, options={}):
        """
        Valid backtracking linesearch options:
            "c": float (default = 0.1), armijo-goldstein curvature parameter
            "rho": float (default=0.5), geometric multiplier for the step length
            "FT_factor": factor by which to multiply alpha when forward tracking
        """
        super().__init__(options)
        self._phi0 = None
        self._dir_derivative = None
        self.alpha = None

        # Set options defaults
        opt_defaults = {"c": 0.1, "rho": 0.5, "FT_factor": 2.0}

        for opt in opt_defaults.keys():
            if opt not in self.options.keys():
                self.options[opt] = opt_defaults[opt]

        self.data["options"] = self.options

    def _check_options(self):
        pass

    def setup(self):
        pass

    def _start_solver(self, du, recorder):
        """Initial iteration of the linesearch.  This method enforces
        bounds on the first step and returns the objective function
        value.

        Parameters
        ----------
        du : array
            Newton step vector

        Returns
        -------
        float
            Objective function value after bounds enforcement
        """
        self._iter_count = 0
        self.alpha = self.options["alpha"]
        phi0 = self._objective()
        recorder["atol"].append(phi0)
        recorder["alpha"].append(self.alpha)
        if self.options["iprint"] > 1:
            print(f"    + Init LS: {self._iter_count} {phi0} 0.0")
        use_fwd_track = False
        if phi0 == 0.0:
            phi0 = 1.0

        self._phi0 = phi0

        # From the definition of Newton's method, one full step should
        # drive the linearized residuals to zero, hence the directional
        # derivative is equal to the initial function value
        self._dir_derivative = -phi0

        self._update_states(self.alpha, du)

        step_limited = self._enforce_bounds(du, self.alpha)

        self.model.run()
        phi = self._objective()
        self._iter_count += 1

        recorder["atol"].append(phi)
        recorder["alpha"].append(self.alpha)

        if phi < self._phi0 and phi < 1.0:
            use_fwd_track = True

        # Prevent forward tracking linesearch when the step goes right up to a bound
        if step_limited:
            use_fwd_track = False

        if self.options["iprint"] > 1:
            print(f"    + {'FT' if use_fwd_track else 'AG'} LS: {self._iter_count} {phi} {self.alpha}")

        return phi, use_fwd_track

    def _stopping_criteria(self, fval):
        """Armijo-Goldstein criteria for terminating the linesearch

        Parameters
        ----------
        fval : float
            Current objective function value

        Returns
        -------
        bool
            Result of the criteria check
        """
        fval0 = self._phi0
        df_dalpha = self._dir_derivative
        c1 = self.options["c"]
        alpha = self.alpha

        return fval0 + (1 - c1) * alpha * df_dalpha <= fval <= fval0 + c1 * alpha * df_dalpha

    def _forward_track(self, du, phi, recorder):
        recorder["mode"] = "fwd"
        phi1 = phi

        alpha_max = self.options["alpha max"]

        # Decrease alpha max to go exactly to the bounds if it would otherwise violate them
        # TODO: this is not the smartest way of doing this, make it better bro
        # This is elite, pogchamp
        du_bounded = np.copy(du)
        _enforce_bounds_vector(
            self.model.states + alpha_max * du, du_bounded, alpha_max, self.model.lower, self.model.upper
        )
        alpha_max *= np.linalg.norm(du_bounded) / np.linalg.norm(du)

        while self.alpha < alpha_max:
            # Forward track to the next alpha
            new_alpha = self.options["FT_factor"] * self.alpha
            self._update_states(new_alpha - self.alpha, du)
            self.model.run()
            phi2 = self._objective()
            self._iter_count += 1
            recorder["atol"].append(phi2)
            recorder["alpha"].append(new_alpha)

            if self.options["iprint"] > 1:
                print(f"    + FT LS: {self._iter_count} {phi2} {new_alpha}")

            if phi2 >= phi1:
                self._iter_count += 1
                self._update_states(self.alpha - new_alpha, du)
                self.model.run()
                recorder["atol"].append(phi1)
                recorder["alpha"].append(self.alpha)
                if self.options["iprint"] > 1:
                    print(f"    + FT LS: {self._iter_count} {phi1} {self.alpha}")
                break
            else:
                self.alpha = new_alpha

    def _back_track(self, du, phi, maxiter, recorder):
        rho = self.options["rho"]

        while self._iter_count < maxiter and (not self._stopping_criteria(phi)):
            # Geometrically decrease the step length
            alpha_old = self.alpha
            self.alpha *= rho
            self._update_states(self.alpha - alpha_old, du)

            # update the model
            self.model.run()
            self._iter_count += 1

            # compute the objective
            phi = self._objective()

            recorder["atol"].append(phi)
            recorder["alpha"].append(self.alpha)

            if self.options["iprint"] > 1:
                print(f"    + AG LS: {self._iter_count} {phi} {self.alpha}")

    def solve(self, du):
        """Solve method for the linesearch

        Parameters
        ----------
        du : array
            Newton step vector
        """
        recorder = {"atol": [], "alpha": [], "mode": "ag"}
        phi, use_fwd_track = self._start_solver(du, recorder)

        if use_fwd_track:
            self._forward_track(du, phi, recorder)
            if self.options["iprint"] == 1:
                print(f"    + FT LS done in {self._iter_count} iterations with phi = {phi},  alpha = {self.alpha}")
        else:
            self._back_track(du, phi, self.options["maxiter"], recorder)
            if self.options["iprint"] == 1:
                print(f"    + AG LS done in {self._iter_count} iterations with phi = {phi},  alpha = {self.alpha}")

        self.data["data"].append(recorder)


class BoundsEnforceLineSearch(LineSearch):
    """
    A linesearch that performs vector bounds enforcement and that's it.
    """

    def __init__(self, options={}):
        """
        Valid bounds enforce linesearch options:
            None
        """
        super().__init__(options)
        self._phi0 = None
        self._dir_derivative = None
        self.alpha = None

        # Set options defaults
        opt_defaults = {}

        for opt in opt_defaults.keys():
            if opt not in self.options.keys():
                self.options[opt] = opt_defaults[opt]

        self.data["options"] = self.options

    def solve(self, du):
        """Solve method for the bounds enforce linesearch

        Parameters
        ----------
        du : array
            Newton step vector
        """
        if self.options["iprint"] > 1:
            du_orig = np.copy(du)

        self.alpha = self.options["alpha"]
        recorder = {"alpha": [self.options["alpha"]]}
        self._update_states(self.alpha, du)
        step_limited = self._enforce_bounds(du, self.alpha)
        self.model.run()

        if step_limited and self.options["iprint"] > 1:
            print(f"    + BE LS limited step by {(1 - np.linalg.norm(du)/np.linalg.norm(du_orig))*100}%")

        recorder["alpha"].append(self.alpha)
        self.data["data"].append(recorder)


class IPLineSearch(LineSearch):
    def __init__(self, options={}):
        """
        Valid backtracking linesearch options:
            "beta": float (default = 2.0), bracketing expansion factor
            "rho": float (default = 0.5), Illinois algorithm contraction factor
            "root_method": string (default = "illinois"), Name of the root finding algorithm
        """
        super().__init__(options)
        self._phi0 = None
        self._dir_derivative = None
        self.alpha = None

        # Set options defaults
        opt_defaults = {"beta": 2.0, "rho": 0.5, "root_method": "illinois"}

        for opt in opt_defaults.keys():
            if opt not in self.options.keys():
                self.options[opt] = opt_defaults[opt]

        self.data["options"] = self.options

    def _objective(self, du):
        """Computes the objective function for the linesearch.  If the
        linesearch uses the penalized residual, this function will
        compute the residual before the objective.

        Returns
        -------
        float
            Inner product between the Newton step and residual vectors.
        float
            L2 norm of the residual vector
        """
        if self.options["residual penalty"]:
            u = self.model.states
            lb = self.model.lower
            ub = self.model.upper
            lb_mask = self.model.lower_finite_mask
            ub_mask = self.model.upper_finite_mask

            penalty = np.zeros(u.size)

            t_lower = u[lb_mask] - lb[lb_mask]
            t_upper = ub[ub_mask] - u[ub_mask]

            if t_lower.size > 0:
                penalty[lb_mask] += np.sum(self.mu_lower * -np.log(t_lower + 1e-10))

            if t_upper.size > 0:
                penalty[ub_mask] += np.sum(self.mu_upper * -np.log(t_upper + 1e-10))

            residuals = self.model.residuals + penalty
        else:
            residuals = self.model.residuals

        return np.dot(du, residuals), np.linalg.norm(residuals)

    def _start_solver(self, du, recorder):
        """Initial iteration of the linesearch.  This method enforces
        bounds on the first step and returns the objective function
        value.

        Parameters
        ----------
        du : array
            Newton step vector

        Returns
        -------
        float
            Objective function value after bounds enforcement
        """
        self._iter_count = 0
        s_b = 0.0  # Lower bracket step length
        s_a = self.options["alpha"]  # Upper bracket step length
        self.g_0, self._phi0 = self._objective(du)

        # Record initial iteration
        recorder["atol"].append(self._phi0)
        recorder["alpha"].append(s_a)

        # Print iteration
        if self.options["iprint"] > 1:
            print(f"    + Init LS: {self._iter_count} {self._phi0} {s_a}")

        # Move the states the upper bracket step
        self._update_states(s_a, du)

        self._enforce_bounds(du, s_a)
        self.model.run()

        g_a, phi = self._objective(du)
        self._iter_count += 1

        recorder["atol"].append(phi)
        recorder["alpha"].append(s_a)

        # Construct the initial brackets
        self.s_ab = [s_b, s_a]
        self.g_ab = [self.g_0, g_a]

        return phi

    def _bracketing(self, du, recorder):
        alpha_max = self.options["alpha max"]
        du_bounded = np.copy(du)
        _enforce_bounds_vector(
            self.model.states + alpha_max * du, du_bounded, alpha_max, self.model.lower, self.model.upper
        )
        s_max = alpha_max * np.linalg.norm(du_bounded) / np.linalg.norm(du)

        beta = self.options["beta"]
        maxiter = self.options["maxiter"]

        while self.s_ab[1] < s_max and self._iter_count < maxiter and np.sign(self.g_ab[1]) * np.sign(self.g_ab[0]) > 0:
            # Set the lower bracket equal to the upper bracket
            self.s_ab[0], self.g_ab[0] = self.s_ab[1], self.g_ab[1]

            # Update the upper brack step
            self.s_ab[1] *= beta
            s_rel = self.s_ab[1] - self.s_ab[0]

            # Move the relative step between the bracket steps
            self._update_states(s_rel, du)

            # Enforce the bounds at the upper bracket step
            self._enforce_bounds(du, self.s_ab[1])

            self.model.run()

            self.g_ab[1], phi = self._objective(du)
            self._iter_count += 1

            recorder["atol"].append(phi)
            recorder["alpha"].append(self.s_ab[1])

            if self.options["iprint"] > 1:
                print(f"    + IP BRKT LS: {self._iter_count} {phi} {self.s_ab[1]}")

    def _illinois(self, du, recorder):
        maxiter = self.options["maxiter"]
        rho = self.options["rho"]

        # 'g_k' is the value of the objective function at the current
        # iteration.  We will always start the Illinois algorithm
        # at the upper bracket.
        g_k = self.g_ab[1]

        while self._iter_count < maxiter and (
            abs(g_k) > 0.5 * abs(self.g_0) or abs(self.s_ab[0] - self.s_ab[1]) > 0.25 * np.sum(self.s_ab)
        ):
            # Compute new root estimate using the regular-falsi method
            s_k = self.s_ab[1] - self.g_ab[1] * ((self.s_ab[1] - self.s_ab[0]) / (self.g_ab[1] - self.g_ab[0]))

            # Update states using the relative step between the new
            # root guess and the previous root guess
            self._update_states(s_k - self.s_ab[1], du)

            self.model.run()

            g_k, phi = self._objective(du)
            self._iter_count += 1
            recorder["atol"].append(phi)
            recorder["alpha"].append(s_k)

            # If the signs of the current and previous function value
            # are the same, we want to contract the bracket by rho;
            # otherwise, swap the bracket
            if np.sign(g_k) * np.sign(self.g_ab[1]) > 0:
                self.g_ab[0] *= rho
            else:
                self.s_ab[0], self.g_ab[0] = self.s_ab[1], self.g_ab[1]

            # Update the upper bracket
            self.s_ab[1], self.g_ab[1] = s_k, g_k

            if self.options["iprint"] > 1:
                print(f"    + IP ILLNS LS: {self._iter_count} {phi} {s_k}")

    def solve(self, du):
        recorder = {"atol": [], "alpha": []}
        self._start_solver(du, recorder)
        self._bracketing(du, recorder)

        if not np.sign(self.g_ab[1]) * np.sign(self.g_ab[0]) >= 0:
            self._illinois(du, recorder)

        self.data["data"].append(recorder)


# This is a helper function directly from OpenMDAO for enforcing bounds.
# I didn't feel like re-writing this code.
# link: https://github.com/OpenMDAO/OpenMDAO/blob/master/openmdao/solvers/linesearch/backtracking.py
def _enforce_bounds_vector(u, du, alpha, lower_bounds, upper_bounds):
    """
    Enforce lower/upper bounds, backtracking the entire vector together.
    This method modifies both self (u) and step (du) in-place.

    Parameters
    ----------
    u : array
        Output vector.
    du : array
        Newton step; the backtracking is applied to this array in-place.
    alpha : float
        step size.
    lower_bounds : array
        Lower bounds array.
    upper_bounds : array
        Upper bounds array.

    Returns
    -------
    bool
        True if the step was limited by the bounds enforcement, False otherwise
    """
    # The assumption is that alpha * du has been added to self (i.e., u)
    # just prior to this method being called. We are currently in the
    # initialization of a line search, and we're trying to ensure that
    # the u does not violate bounds in the first iteration. If it does,
    # we modify the du vector directly.
    # This is the required change in step size, relative to the du vector.
    d_alpha = 0

    # Find the largest amount a bound is violated
    # where positive means a bound is violated - i.e. the required d_alpha.
    du_arr = du
    mask = du_arr != 0
    if mask.any():
        abs_du_mask = np.abs(du_arr[mask])
        u_mask = u[mask]

        # Check lower bound
        if lower_bounds is not None:
            max_d_alpha = np.amax((lower_bounds[mask] - u_mask) / abs_du_mask)
            if max_d_alpha > d_alpha:
                d_alpha = max_d_alpha

        # Check upper bound
        if upper_bounds is not None:
            max_d_alpha = np.amax((u_mask - upper_bounds[mask]) / abs_du_mask)
            if max_d_alpha > d_alpha:
                d_alpha = max_d_alpha

    # A d_alpha greater than alpha will make it take a step backwards, meaning
    # that it already violated the bounds before the step, which doesn't make sense
    d_alpha = min(d_alpha, alpha)

    if d_alpha > 0:
        # d_alpha will not be negative because it was initialized to be 0
        # and we've only done max operations.
        # d_alpha will not be greater than alpha because the assumption is that
        # the original point was valid - i.e., no bounds were violated.
        # Therefore 0 <= d_alpha <= alpha.

        # We first update u to reflect the required change to du.
        u -= d_alpha * du
        # At this point, we normalize d_alpha by alpha to figure out the relative
        # amount that the du vector has to be reduced, then apply the reduction.
        du *= 1 - d_alpha / alpha

        return True

    return False
