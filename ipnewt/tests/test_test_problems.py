#!/usr/bin/env python
"""
@File    :   test_test_problems.py
@Time    :   2021/12/3
@Desc    :   Test the models in the test_problems folder
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
from ipnewt.api import Powell, HEquation, test_utils

class TestPowell(unittest.TestCase):
    def test_residuals_soln(self):
        model = Powell()

        # Test that the residuals are zero at the solution
        u = np.array([1.098e-5, 9.106])
        res = np.zeros(2)
        model.compute_residuals(u, res)

        np.testing.assert_allclose(res, np.zeros(2), atol=1e-3)
    
    def test_residuals_zeros(self):
        model = Powell()

        # Test that the residuals are zero at the solution
        u = np.zeros(2)
        res = np.zeros(2)
        model.compute_residuals(u, res)

        np.testing.assert_allclose(res, np.array([-1, .9999]), atol=1e-3)
    
    def test_derivatives(self):
        model = Powell()

        # Check the derivaties at the solution
        model.states = np.array([1.098e-5, 9.106])
        self.assertTrue(test_utils.check_model_derivatives(model, print_results=False))

        # Check the derivaties away from the solution
        model.states = np.array([14.9, 14.9])
        self.assertTrue(test_utils.check_model_derivatives(model, print_results=False))
    
    def test_bounds(self):
        model = Powell(options={"one lower": -14,
                                "one upper": -13,
                                "two lower": -3,
                                "two upper": 1})
        np.testing.assert_allclose(model.lower, np.array([-14, -3]), atol=1e-15)
        np.testing.assert_allclose(model.upper, np.array([-13, 1]), atol=1e-15)


class TestHEquation(unittest.TestCase):
    def test_residuals(self):
        model = HEquation()

        u = np.array([2., 3.])
        res = np.zeros(2)
        model.compute_residuals(u, res)

        np.testing.assert_allclose(res, np.array([0.72, 1.4]), atol=1e-10)
    
    def test_derivatives(self):
        model = HEquation()

        # Check the derivaties in one spot
        model.states = np.array([1.5, 1.5])
        self.assertTrue(test_utils.check_model_derivatives(model, print_results=False))

        # Check the derivaties in another spot
        model.states = np.array([14.9, 14.9])
        self.assertTrue(test_utils.check_model_derivatives(model, print_results=False))
    
    def test_bounds(self):
        model = HEquation(options={"lower": -14,
                                   "upper": -13})
        np.testing.assert_allclose(model.lower, np.array([-14, -14]), atol=1e-15)
        np.testing.assert_allclose(model.upper, np.array([-13, -13]), atol=1e-15)
    
    def test_higher_dim(self):
        n = 5
        model = HEquation(options={"n_states": n})

        u = np.array([2., 3., 1., 4., 5.])
        res = np.zeros(n)
        model.compute_residuals(u, res)

        np.testing.assert_allclose(res, np.array([0.829268,  1.589065, -0.599238,  2.242402,  3.105548]), atol=1e-6)

        # Check the derivaties in one spot
        model.states = 1.5 * np.ones(n)
        self.assertTrue(test_utils.check_model_derivatives(model, print_results=False))


if __name__=="__main__":
    unittest.main()