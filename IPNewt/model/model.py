#!/usr/bin/env python
"""
@File    :   model.py
@Time    :   2021/11/13
@Desc    :   Class definition for the top level Model class
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


class Model(object):
    def __init__(self, n_states, lower=None, upper=None):
        """
        Initialize all attributes.

        Inputs
        ------
        n_states : int
            Number of states in the model.
        lower : float or iterable of length n_states (optional)
            Lower bounds on the states. If it is a single number, all states
            will take on that lower bound. If it is an iterable, it must be
            of length n_states and the bounds will be set in that order.
        upper : float or iterable of length n_states (optional)
            Upper bounds on the states. If it is a single number, all states
            will take on that upper bound. If it is an iterable, it must be
            of length n_states and the bounds will be set in that order.
        """
        # Error check number of states
        if not isinstance(n_states, int):
            raise TypeError(f"n_states must be an int, not {type(n_states)}")
        if n_states < 1:
            raise ValueError(f"Model must include at least one state (defined by n_states), not {n_states}")

        self.options = {}
        self.n_states = n_states
        self.residual = np.empty(n_states)
        self.states = np.ones(n_states)  # initialize all states to one
        self.jacobian = None

        # Set the bounds if necessary
        if isinstance(lower, (float, int, complex)):
            self.lower = lower * np.ones(n_states)
        elif isinstance(lower, (list, np.ndarray)):
            if len(lower) != n_states:
                raise ValueError("If the lower bounds are defined as an iterable, \
                                  it must have a length of n_states")
            self.lower = lower.copy()
        else:
            self.lower = -np.inf * np.ones(n_states)

        if isinstance(upper, (float, int, complex)):
            self.upper = upper * np.ones(n_states)
        elif isinstance(upper, (list, np.ndarray)):
            if len(upper) != n_states:
                raise ValueError("If the upper bounds are defined as an iterable, \
                                  it must have a length of n_states")
            self.upper = upper.copy()
        else:
            self.upper = -np.inf * np.ones(n_states)
        
        # Mask for bounds that are defined
        self.lower_finite_mask = np.isfinite(self.lower)
        self.upper_finite_mask = np.isfinite(self.upper)

    def _check_options(self):
        """
        Perform a few useful checks on the model. These include the following:
            - Check that lower bounds are less than upper bounds
        """
        # Check bounds
        for i in range(self.n_states):
            if self.lower[i] > self.upper[i]:
                raise ValueError(f"Lower bound is greater than upper bound on state {i}")

    def _compute_jacobian(self):
        """
        Computes the partial derivatives and sets up the Jacobian matrix.
        """
        pass

    def compute_residuals(self):
        """
        USER DEFINED

        Computes the residuals of the model (which all equal zero when the
        model is solved). Sets self.residual to a numpy vector.
        """
        pass

    def compute_partials(self):
        """
        USER DEFINED

        Compute the partial derivatives of the residuals of the nonlinear
        equations in the model.
        """
        pass
