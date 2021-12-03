#!/usr/bin/env python
"""
@File    :   powell.py
@Time    :   2021/12/3
@Desc    :   Run script for the Powell test problem
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import niceplots

# ==============================================================================
# Extension modules
# ==============================================================================
from ipnewt.api import NewtonSolver, LULinearSystem, AdaptiveLineSearch, Powell, viz2D

# ==============================================================================
# External Python modules
# ==============================================================================

# Set up niceplots
dark_mode = True
niceplots.setRCParams(dark_mode=dark_mode, set_dark_background=dark_mode)
plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 14

# Set up problem
prob = NewtonSolver(options={"maxiter": 100, "tau": 100.0})
prob.model = Powell()
prob.linear_system = LULinearSystem()
prob.linesearch = AdaptiveLineSearch(options={"alpha max": 3.0})

# Run the problem
prob.setup()
prob.solve()

# Plot the results
plt.figure()
viz2D.contour(plt.gca(), prob.model, [-11, 16], [-11, 16], levels=100)
viz2D.bounds(plt.gca(), prob.model, [-11, 16], [-11, 16], colors='white', alpha=0.5, zorder=2, linestyles='solid')
plt.gca().set_aspect("equal")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.show()
