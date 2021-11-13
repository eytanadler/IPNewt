#!/usr/bin/env python
"""
@File    :   line_search.py
@Time    :   2021/11/13
@Desc    :   Class definition for the line search
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


class LineSearch(object):
    def __init__(self):
        """
        Initialize all attributed
        """
        self._problem = None
        self.residuals = None

    def _check_options(self):
        pass

    def setup(self):
        # Any setup operations go here
        pass

    def _objective(self):
        return np.linalg.norm(self._problem.model.residuals)

    def search(self):
        pass

