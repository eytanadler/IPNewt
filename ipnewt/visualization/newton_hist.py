#!/usr/bin/env python
"""
@File    :   objective.py
@Time    :   2021/12/03
@Desc    :   None
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


def convergence(ax, data, var, **kwargs):
    """Plots the absolute or relative convergence tolerance for the
    Newton solver.

    Additional keyword arguments for matplotlib's contour
    function can be added to this function.

    Parameters
    ----------
    ax : matplotlib axis object
        Axis on which to plot the convergence
    data : dict
        NewtonSolver data attribute.  The convergence information is
        extracted from here.
    var : string
        Either "atol" or "rtol" depending on if you want to plot
        absolute or relative convergence respectively.
    """
    num_iters = len(data["atol"])
    if var.lower() == "atol":
        var_data = data["atol"]
        ax.set_ylabel(r"$\lVert r^k(u) \rVert_2$", rotation="horizontal", horizontalalignment="right")

    elif var.lower() == "rtol":
        var_data = data["rtol"]
        ax.set_ylabel(
            r"$\frac{\lVert r^k(u) \rVert_2}{\lVert r^0(u) \rVert_2}$",
            rotation="horizontal",
            horizontalalignment="right",
        )

    ax.plot(np.arange(0, num_iters, 1), var_data, **kwargs)
    ax.set_yscale("log")


def penalty_parameter(ax, data, var, idx, **kwargs):
    """Plots the penalty parameter across all iteratons for the state
    corresponding to the specified index.

    Additional keyword arguments for matplotlib's contour
    function can be added to this function.

    Parameters
    ----------
    ax : matplotlib axis object
        Axis on which to plot the penalty parameter
    data : dict
        NewtonSolver data attribute.  The penalty information is
        extracted from here.
    var : string
        Either "mu lower" or "mu upper" depending on which penalty
        you want to plot.
    idx : int
        The index of the state that corresponds to the desired penalty
        you want to plot.
    """
    num_iters = len(data["atol"])
    n_states = len(data["states"][0])

    if var.lower() == "mu lower":
        var_data = np.array(data["mu lower"]).reshape((num_iters, n_states))
        ax.set_ylabel(r"$\underline{\mu}$", rotation="horizontal", horizontalalignment="right")

    elif var.lower() == "mu upper":
        var_data = np.array(data["mu upper"]).reshape((num_iters, n_states))
        ax.set_ylabel(r"$\overline{\mu}$", rotation="horizontal", horizontalalignment="right")

    ax.plot(np.arange(0, num_iters, 1), var_data[:, idx], **kwargs)
    ax.set_yscale("log")


def pseudo_time_step(ax, data, **kwargs):
    """Plots the pseudo-transient time step versus iterations.

    Additional keyword arguments for matplotlib's contour
    function can be added to this function.

    Parameters
    ----------
    ax : matplotlib axis object
        The axis on which to plot the time step.
    data : dict
        NewtonSolver data attribute.  The time step information is
        extracted from here.
    """
    num_iters = len(data["atol"])

    ax.plot(np.arange(0, num_iters, 1), data["tau"], **kwargs)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Delta\tau^k$", rotation="horizontal", horizontalalignment="right")


def condition_number(ax, data, penalty=False, pt=False, **kwargs):
    len_data = len(data["linear_data"])
    J = [data["linear_data"][i]["jacobian"] for i in range(len_data)]
    if penalty:
        for i in range(len_data):
            J[i] += np.diag(data["linear_data"][i]["penalty_vector"])
    if pt:
        for i in range(len_data):
            J[i] += np.diag(1 / data["linear_data"][i]["pt_vector"])

    cond_nums = list(map(np.linalg.cond, J))

    ax.semilogy(np.arange(0, len_data, 1), cond_nums, **kwargs)


def step_size(ax, data, **kwargs):
    len_data = len(data["linear_data"])
    step_size = np.zeros(len_data - 1)
    for i in range(len_data - 1):
        step_size[i] = np.linalg.norm(data["states"][i + 1] - data["states"][i])

    ax.semilogy(np.arange(1, len_data, 1), step_size, **kwargs)


def alpha(ax, data, **kwargs):
    len_data = len(data["atol"])
    alpha_data = [ls_dict["alpha"][-1] for ls_dict in data["linesearch_data"]["data"]]

    ax.semilogy(np.arange(1, len_data, 1), alpha_data, **kwargs)
    ax.set_ylabel(r"$\alpha$", rotation="horizontal", horizontalalignment="right")


def ls_ag_frequency(ax, data, c, **kwargs):
    phi0 = np.array([ls_dict["atol"][0] for ls_dict in data["linesearch_data"]["data"]])
    phiEnd = np.array([ls_dict["atol"][-1] for ls_dict in data["linesearch_data"]["data"]])
    alpha = np.array([ls_dict["alpha"][-1] for ls_dict in data["linesearch_data"]["data"]])

    df_dalpha = -phi0

    gs_lower = phi0 + (1 - c) * alpha * df_dalpha
    gs_upper = phi0 + c * alpha * df_dalpha

    inside_mask = np.logical_and(phiEnd > gs_lower, phiEnd < gs_upper)

    num_inside = np.count_nonzero(inside_mask)
    num_outside = len(inside_mask) - num_inside

    ax.barh(0, num_inside, align="center")
    ax.barh(1, num_outside, align="center")
    ax.set_yticks([0, 1], labels=["Succesfull", "Unsuccessful"])
    ax.set_xlabel("Line Search Performance")


def ls_rtol_frequency(ax, data, **kwargs):
    # NOTE: this assumes that the full Newton step (alpha = 1) is evaluated first
    phi0 = np.array([ls_dict["atol"][0] for ls_dict in data["linesearch_data"]["data"]])
    phiInit = np.array([ls_dict["atol"][1] for ls_dict in data["linesearch_data"]["data"]])
    phiEnd = np.array([ls_dict["atol"][-1] for ls_dict in data["linesearch_data"]["data"]])

    phiRel = (phiInit - phi0) / (phiEnd - phi0)

    ax.hist(phiRel, bins=20)
    ax.set_ylabel("Frequency", rotation="horizontal", horizontalalignment="right")
