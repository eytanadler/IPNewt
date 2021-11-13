#!/usr/bin/env python
"""
@File    :   linear_solver.py
@Time    :   2021/11/13
@Desc    :   Class definition for the linear solver
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import scipy

# ==============================================================================
# Extension modules
# ==============================================================================


class LinearSolver(object):
    def __init__(self):
        """
        Initialize all attributed
        """
        self._problem = None
        self.linear_system = None
        self.options = {}

    def _check_options(self):
        pass

    def setup(self):
        # Any setup operations go here
        pass

    def solve(self):
        # Solve linear system
        pass

class ScipySolver(LinearSolver):
    def __init__(self):
        super().__init__()

    def solve(self):
        """
        Use SciPy's linear solver to solve the system.
        """
        pass

class PETScSolver(LinearSolver):
    def __init__(self):
        super().__init__()

    def solve(self):
        """
        Use PETSc's linear solver to solve the system.
        """
        pass