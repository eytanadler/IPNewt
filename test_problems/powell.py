#!/usr/bin/env python
"""
@File    :   powell.py
@Time    :   2021/11/13
@Desc    :   Model for the Powell test problem
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
from IPNewt.model.model import Model


class Powell(Model):
    def compute_residuals(self):
        """
        Compute the residuals of the nonlinear equations in the model.
        """
        pass

    def compute_partials(self):
        """
        Compute the residuals of the nonlinear equations in the model.
        """
        pass
