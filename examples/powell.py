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
import os

# ==============================================================================
# Extension modules
# ==============================================================================
import ipnewt
from ipnewt.api import NewtonSolver, LULinearSystem, AdaptiveLineSearch, Powell, viz2D, vizNewt

# ==============================================================================
# External Python modules
# ==============================================================================

# Set up niceplots
dark_mode = True
niceColors = niceplots.get_niceColors()
niceplots.setRCParams(dark_mode=dark_mode, set_dark_background=dark_mode)
plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 14
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{cmbright}"

save_dir = os.path.join(os.path.split(ipnewt.__path__[0])[0], "examples", "plots")

# Set up problem
prob = NewtonSolver(options={"maxiter": 100, "tau": 1e3, "mu": 1e0})
prob.model = Powell()
prob.linear_system = LULinearSystem()
prob.linesearch = AdaptiveLineSearch(options={"alpha max": 1e6})

# Set the initial state values
prob.model.states = np.array([14.9, 14.9])

# Run the problem
prob.setup()
prob.solve()

print(f"Solution at {prob.model.states} with residuals of {prob.model.residuals}")

# Plot the results
plt.figure(figsize=[12, 10])
xlim = [-11, 16]
ylim = [-11, 16]
c = viz2D.contour(plt.gca(), prob.model, xlim, ylim, n_pts=500, levels=100, cmap="viridis")
plt.colorbar(c)
viz2D.bounds(plt.gca(), prob.model, xlim, ylim, colors="white", alpha=0.5, zorder=2, linestyles="solid")
viz2D.newton_path(plt.gca(), prob.data, c="white")
plt.xlabel(r"$u_1$")
plt.ylabel(r"$u_2$")
plt.savefig(os.path.join(save_dir, "powell_solver_path.pdf"))

# Plot the penalty contour
plt.figure(figsize=[12, 10])
viz2D.contour(plt.gca(), prob.model, xlim, ylim, levels=100, colors="grey", n_pts=500, alpha=0.7, linewidths=0.5)
viz2D.bounds(plt.gca(), prob.model, xlim, ylim, colors="white", alpha=0.5, zorder=2, linestyles="solid")
viz2D.newton_path(plt.gca(), prob.data, zorder=5, c="white")
xlim = [-9.9, 14.9]
ylim = [-9.9, 14.9]
c = viz2D.penalty_contour(plt.gca(), prob.data, prob.model, xlim, ylim, 1, n_pts=500, levels=100, cmap="viridis")
plt.colorbar(c)
plt.xlabel(r"$u_1$")
plt.ylabel(r"$u_2$")
plt.savefig(os.path.join(save_dir, "powell_penalty_contours.pdf"))

# Plot the convergence
fig, axs = plt.subplots(4, 1, figsize=[10, 15])
vizNewt.convergence(axs[0], prob.data, "atol", color=niceColors["Blue"], marker="o")
vizNewt.convergence(axs[1], prob.data, "rtol", color=niceColors["Red"], marker="o")
vizNewt.pseudo_time_step(axs[2], prob.data, marker="o", color=niceColors["Green"])
vizNewt.penalty_parameter(axs[3], prob.data, "mu upper", 1, color=niceColors["Yellow"], marker="o")

plt.savefig(os.path.join(save_dir, "powell_var_hist.pdf"))

# plt.show()
