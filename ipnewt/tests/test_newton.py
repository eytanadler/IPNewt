#!/usr/bin/env python
"""
@File    :   test_newton.py
@Time    :   2021/12/4
@Desc    :   Test the Newton solver
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import numpy as np
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================
from ipnewt.api import Powell, NewtonSolver, LULinearSystem, AdaptiveLineSearch


class TestNewton(unittest.TestCase):
    def test_bounds_not_violated(self):
        # Set up problem
        prob = NewtonSolver(options={"maxiter": 100, "tau": 0.01})
        prob.model = Powell(options={"one lower": -10, "two lower": -10, "one upper": 15, "two upper": 15})
        prob.linear_system = LULinearSystem()
        prob.linesearch = AdaptiveLineSearch(options={"alpha max": 3.0})

        # Set the initial state values
        prob.model.states = np.array([14.9, 14.9])

        # Run the problem
        prob.setup()
        prob.solve()

        states_hist = np.array(prob.data["states"])

        # Check that the bounds are never violated during the solve
        self.assertTrue(np.all(-10 <= states_hist[:, 0]))
        self.assertTrue(np.all(-10 <= states_hist[:, 1]))
        self.assertTrue(np.all(states_hist[:, 0] <= 15))
        self.assertTrue(np.all(states_hist[:, 1] <= 15))


if __name__ == "__main__":
    unittest.main()
