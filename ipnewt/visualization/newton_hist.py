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
    ax.set_xlabel("Iterations")


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
    ax.set_xlabel("Iterations")


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
    ax.set_xlabel("Iterations")
