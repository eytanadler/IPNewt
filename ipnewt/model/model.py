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
import copy

# ==============================================================================
# Extension modules
# ==============================================================================


class Model(object):
    def __init__(self, n_states, lower=None, upper=None, options={}):
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
        options : dict
            Options that could be model-specific. For example, if your model
            wanted some constant you could define that here and access it in
            your compute_residuals and compute_jacobian implementations.
        """
        # Error check number of states
        if not isinstance(n_states, int):
            raise TypeError(f"n_states must be an int, not {type(n_states)}")
        if n_states < 1:
            raise ValueError(f"Model must include at least one state (defined by n_states), not {n_states}")

        self.options = copy.deepcopy(options)
        self.n_states = n_states
        self.residuals = np.empty(n_states)
        self.states = np.ones(n_states)  # initialize all states to one
        self.jacobian = np.empty((n_states, n_states))

        # Set the bounds if necessary
        if isinstance(lower, (float, int, complex)):
            self.lower = lower * np.ones(n_states)
        elif isinstance(lower, (list, np.ndarray)):
            if len(lower) != n_states:
                raise ValueError(
                    "If the lower bounds are defined as an iterable, \
                                  it must have a length of n_states"
                )
            self.lower = np.array(lower)
        else:
            self.lower = -np.inf * np.ones(n_states)

        if isinstance(upper, (float, int, complex)):
            self.upper = upper * np.ones(n_states)
        elif isinstance(upper, (list, np.ndarray)):
            if len(upper) != n_states:
                raise ValueError(
                    "If the upper bounds are defined as an iterable, \
                                  it must have a length of n_states"
                )
            self.upper = np.array(upper)
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

    def run(self):
        """
        Computes the residuals and the Jacobian matrix.
        """
        self._compute_residuals()
        self._compute_jacobian()

    def _compute_residuals(self):
        """
        Internally pass states and residuals to user-defined function.
        """
        self.compute_residuals(self.states, self.residuals)

    def _compute_jacobian(self):
        """
        Internally pass states and jacobian to user-defined function.
        """
        self.compute_jacobian(self.states, self.jacobian)

    def compute_residuals(self, states, residuals):
        """
        USER DEFINED

        Computes the residuals of the model (which all equal zero when the model is
        solved). Sets residuals (numpy vector with a length of the number of states).
        """
        pass

    def compute_jacobian(self, states, jacobian):
        """
        USER DEFINED

        Sets the Jacobian matrix with the partial derivatives
        of the nonlinear equations in the model. The order of the Jacobian is

        dr0/du0     dr0/du1     dr0/du2     ...
        dr1/du0     dr1/du1     ...
        .           .
        .             .
        .               .

        where r represents the residuals and u represents the states. Thus, to set
        the dr0/du1 partial derivative, set jacobian[0, 1].
        """
        pass
