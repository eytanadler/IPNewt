#!/usr/bin/env python
"""
@File    :   h-equation.py
@Time    :   2021/12/3
@Desc    :   Run script for the H-equation test problem
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
from ipnewt.api import NewtonSolver, LULinearSystem, AdaptiveLineSearch, HEquation, viz2D

# ==============================================================================
# External Python modules
# ==============================================================================

# Set up niceplots
dark_mode = True
niceplots.setRCParams(dark_mode=dark_mode, set_dark_background=dark_mode)
plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 14

# Set up problem
prob = NewtonSolver(options={"maxiter": 100, "tau": 0.01})
prob.model = HEquation(options={"n_states": 2})
prob.linear_system = LULinearSystem()
prob.linesearch = AdaptiveLineSearch(options={"alpha max": 3.0})

prob.model.states = np.array([9., 4.])

# Run the problem
prob.setup()
prob.solve()

print(f"Solution at {prob.model.states} with residuals of {prob.model.residuals}")

# Plot the results
plt.figure(figsize=[12, 10])
xlim = [0, 10]
ylim = [0, 15]
c = viz2D.contour(plt.gca(), prob.model, xlim, ylim,
                  levels=10**np.linspace(0, 3.176, 100) - 1, cmap='viridis')
plt.colorbar(c)
viz2D.newton_path(plt.gca(), prob.data, c='white')
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.show()
