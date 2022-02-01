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
from ipnewt.api import (
    NewtonSolver,
    LULinearSystem,
    AdaptiveLineSearch,
    Powell,
    viz2D,
    vizNewt,
    IPLineSearch,
    BracketingLineSearch,
)  # noqa

# Set up niceplots
dark_mode = True
niceColors = niceplots.get_niceColors()
if dark_mode:
    niceColors["base"] = "#ffffffff"
else:
    niceColors["base"] = "#000000ff"
niceplots.setRCParams(dark_mode=dark_mode, set_dark_background=dark_mode)
plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 14
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{cmbright}"

save_dir = os.path.join(os.path.split(ipnewt.__path__[0])[0], "examples", "plots")

# Set up problem
prob = NewtonSolver(
    options={"maxiter": 100, "tau": 1e-3, "mu": 1e1, "gamma": 2.0, "pt_adapt": "LS", "tau max": 1e100, "atol": 1e-8,}
)
prob.model = Powell()
prob.linear_system = LULinearSystem()
# prob.linesearch = IPLineSearch(options={"iprint": 2, "alpha max": 100.0, "beta": 2.0, "maxiter": 5})
# prob.linesearch = AdaptiveLineSearch(options={"iprint": 2, "alpha max": 100.0, "rho": 0.7, "FT_factor": 2.0})
prob.linesearch = BracketingLineSearch(options={"maxiter": 5, "beta": 2.0})

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
cbar = plt.colorbar(c)
cbar.ax.set_ylabel(r"$\lVert r \rVert_2$", rotation="horizontal", horizontalalignment="left")
viz2D.contour(plt.gca(), prob.model, xlim, ylim, i_res=0, n_pts=200, levels=[0], colors="red")
viz2D.contour(plt.gca(), prob.model, xlim, ylim, i_res=1, n_pts=200, levels=[0], colors="green")
viz2D.bounds(plt.gca(), prob.model, xlim, ylim, colors=niceColors["base"], alpha=0.5, zorder=2, linestyles="solid")
viz2D.newton_path(plt.gca(), prob.data, c=niceColors["base"])
plt.scatter(1.09815933e-05, 9.10614674, zorder=5, marker="*", s=300, c=niceColors["Yellow"])
# plt.plot(prob.data["states"][12][0], prob.data["states"][12][1], "-o", color=niceColors["Red"])
# viz2D.newton_soln_viz(plt.gca(), prob.model, prob.data, xlim, iter=12, penalty=True, pt=False)
plt.xlabel(r"$u_1$")
plt.ylabel(r"$u_2$", rotation="horizontal", horizontalalignment="right")
plt.savefig(os.path.join(save_dir, "powell_solver_path.pdf"))

# Plot the penalty contour
plt.figure(figsize=[12, 10])
viz2D.contour(plt.gca(), prob.model, xlim, ylim, levels=100, colors="grey", n_pts=500, alpha=0.7, linewidths=0.5)
viz2D.bounds(plt.gca(), prob.model, xlim, ylim, colors=niceColors["base"], alpha=0.5, zorder=2, linestyles="solid")
viz2D.newton_path(plt.gca(), prob.data, zorder=5, c=niceColors["base"])
xlim = [-9.9, 14.9]
ylim = [-9.9, 14.9]
c = viz2D.penalty_contour(plt.gca(), prob.data, prob.model, xlim, ylim, 1, n_pts=500, levels=100, cmap="viridis")
plt.colorbar(c)
plt.xlabel(r"$u_1$")
plt.ylabel(r"$u_2$")
plt.savefig(os.path.join(save_dir, "powell_penalty_contours.pdf"))

# Plot the history
fig, axs = plt.subplots(6, 1, figsize=[10, 18], sharex="row")
vizNewt.convergence(axs[0], prob.data, "atol", color=niceColors["Blue"], marker="o")
vizNewt.convergence(axs[1], prob.data, "rtol", color=niceColors["Red"], marker="o")
vizNewt.pseudo_time_step(axs[2], prob.data, marker="o", color=niceColors["Green"])
vizNewt.penalty_parameter(axs[3], prob.data, "mu lower", 0, color=niceColors["Yellow"], marker="o")
vizNewt.penalty_parameter(axs[3], prob.data, "mu upper", 0, color=niceColors["Blue"], marker="o")
vizNewt.penalty_parameter(axs[3], prob.data, "mu lower", 1, color=niceColors["Yellow"], marker="o")
vizNewt.penalty_parameter(axs[3], prob.data, "mu upper", 1, color=niceColors["Blue"], marker="o")
vizNewt.condition_number(axs[4], prob.data, penalty=False, pt=False, color=niceColors["Yellow"], marker="o")
vizNewt.condition_number(axs[4], prob.data, penalty=True, pt=False, color=niceColors["Orange"], marker="o")
vizNewt.condition_number(axs[4], prob.data, penalty=False, pt=True, color=niceColors["Green"], marker="o")
vizNewt.condition_number(axs[4], prob.data, penalty=True, pt=True, color=niceColors["Blue"], marker="o")
vizNewt.alpha(axs[5], prob.data, color=niceColors["Cyan"], marker="o")
axs[4].legend([r"J", r"$J + J_\text{penalty}$", r"$J + J_\text{pt}$", r"$J + J_\text{pt} + J_\text{penalty}$"])
axs[4].set_ylabel(r"$\kappa$", rotation="horizontal", horizontalalignment="right")
axs[4].set_xlabel("Iterations")
plt.savefig(os.path.join(save_dir, "powell_var_hist.pdf"))

# --- Plot the linesearch quality ---
# fig, axs = plt.subplots(2, 1, figsize=[10, 12])
# vizNewt.ls_ag_frequency(axs[0], prob.data, 0.1)
# vizNewt.ls_rtol_frequency(axs[1], prob.data)
# plt.savefig(os.path.join(save_dir, "ls_quality.pdf"))


# # Plot the solution lines
# plt.figure(figsize=[12, 10])
# xlim = [-10, 10]
# ylim = [-10, 10]
# viz2D.contour(plt.gca(), prob.model, xlim, ylim, levels=100, colors="grey", n_pts=500, alpha=0.7, linewidths=0.5)
# viz2D.newton_soln_viz(plt.gca(), prob.model, prob.data, xlim, iter=12, penalty=True, pt=False)

# plt.savefig(os.path.join(save_dir, "powell_solution_lines.pdf"))

# plt.show()
