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
import os

# ==============================================================================
# Extension modules
# ==============================================================================
import ipnewt
from ipnewt.api import NewtonSolver, LULinearSystem, AdaptiveLineSearch, HEquation, viz2D, vizNewt

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

save_dir = os.path.join(os.path.split(ipnewt.__path__[0])[0], 'examples', 'plots')

# Set up problem
n = 3  # number of dimensions
prob = NewtonSolver(options={"maxiter": 100, "tau": 1e-6, "mu": 1., "mu max": 1e100, "rtol": 0.})
prob.model = HEquation(options={"n_states": n,})
prob.linear_system = LULinearSystem()
prob.linesearch = AdaptiveLineSearch(options={"alpha max": 1e2})

if n == 2:
    prob.model.states = np.array([9., 4.])
else:
    np.random.seed(5)
    prob.model.states = np.random.rand(n)*15

# Run the problem
prob.setup()
prob.solve()

print(f"Solution at {prob.model.states} with residuals of {prob.model.residuals}")

# Plot the results
if n == 2:
    plt.figure(figsize=[12, 10])
    xlim = [0, 10]
    ylim = [0, 15]
    c = viz2D.contour(plt.gca(), prob.model, xlim, ylim,
                    levels=10**np.linspace(0, 3.176, 100) - 1, cmap='viridis')
    plt.colorbar(c)
    viz2D.newton_path(plt.gca(), prob.data, c='white')
    plt.xlabel(r"$u_1$")
    plt.ylabel(r"$u_2$")
    plt.savefig(os.path.join(save_dir, "h_equation_solver_path.pdf"))

# Plot the convergence
fig, axs = plt.subplots(4, 1, figsize=[10, 15])
marker = "o"
linewidth=2
if len(prob.data["mu lower"]) > 50:
    marker = None
    linewidth = 0.8
vizNewt.convergence(axs[0], prob.data, "atol", color=niceColors["Blue"], marker=marker, linewidth=linewidth)
vizNewt.convergence(axs[1], prob.data, "rtol", color=niceColors["Red"], marker=marker, linewidth=linewidth)
vizNewt.pseudo_time_step(axs[2], prob.data, marker=marker, linewidth=linewidth, color=niceColors["Green"])
for i in range(n):
    vizNewt.penalty_parameter(axs[3], prob.data, "mu lower", i, color=niceColors["Cyan"], marker=marker, linewidth=linewidth)
    vizNewt.penalty_parameter(axs[3], prob.data, "mu upper", i, color=niceColors["Yellow"], marker=marker, linewidth=linewidth)

plt.savefig(os.path.join(save_dir, "h_equation_var_hist.pdf"))
