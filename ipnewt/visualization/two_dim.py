#!/usr/bin/env python
"""
@File    :   two_dim_viz.py
@Time    :   2021/12/3
@Desc    :   2D visualization tools
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


def contour(ax, model, xlim, ylim, i_res=None, n_pts=100, **kwargs):
    """Plot the contours of a 2D problem.

    Additional keyword arguments for matplotlib's contour
    function can be added to this function.

    Parameters
    ----------
    ax : matplotlib axis object
        Axis on which to plot contours.
    model : ipnewt model
        The model to use to compute the contour values to plot. Uses the 2-norm
        of the residual vector by default.
    xlim : two-element iterable
        Lower and upper bounds to plot contours along the x-axis.
    ylim : two-element iterable
        Lower and upper bounds to plot contours along the y-axis.
    i_res : int, optional
        Index of the residual to plot, if None will plot the residual norm (default None)
    n_pts : int, optional
        Number of points in each direction at which to evaluate the plotted function.

    Returns
    -------
    matplotlib QuadContourSet
        Useful to make colorbar, can be ignored
    """
    if i_res is not None and (i_res > 1 or i_res < 0):
        raise ValueError(f"i_res must be either None (plot 2-norm), 0, or 1, not {i_res}")

    # Generate a grid on which to evaluate the residual norm
    x, y = np.meshgrid(np.linspace(*xlim, n_pts), np.linspace(*ylim, n_pts), indexing="xy")
    val = np.zeros(x.shape)

    # States and residuals
    u = np.zeros(2)
    res = np.zeros(2)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            u[0] = x[i, j]
            u[1] = y[i, j]

            # Compute the residuals at the current point
            model.compute_residuals(u, res)

            if i_res is None:
                val[i, j] = np.linalg.norm(res)
            else:
                val[i, j] = res[i_res]

    return ax.contour(x, y, val, **kwargs)


def penalty_contour(ax, data, model, xlim, ylim, idx, n_pts=100, **kwargs):
    """Plot the contours of a 2D problem with an interior penalty
    contribution.

    Additional keyword arguments for matplotlib's contour
    function can be added to this function.

    Parameters
    ----------
    ax : matplotlib axis object
        Axis on which to plot contours.
    data: dict
        NewtonSovler data attribute.  Constains the penalty parameter
        data for computing the penalty function.
    model : ipnewt model
        The model to use to compute the contour values to plot. Uses the 2-norm
        of the residual vector by default.
    xlim : two-element iterable
        Lower and upper bounds to plot contours along the x-axis.
    ylim : two-element iterable
        Lower and upper bounds to plot contours along the y-axis.
    idx : int
        Iteration index for plotting the penalty contour.
    n_pts : int, optional
        Number of points in each direction at which to evaluate the plotted function.

    Returns
    -------
    matplotlib QuadContourSet
        Useful to make colorbar, can be ignored
    """
    # Generate a grid on which to evaluate the residual norm
    x, y = np.meshgrid(np.linspace(*xlim, n_pts), np.linspace(*ylim, n_pts), indexing="xy")
    norms = np.zeros(x.shape)

    # States and residuals
    u = np.zeros(2)
    res = np.zeros(2)

    # Get the bounds and finite masks from the model
    lb = model.lower
    ub = model.upper
    lb_mask = model.lower_finite_mask
    ub_mask = model.upper_finite_mask

    # Get the penalty parameters at iteration idx
    mu_lower = data["mu lower"][idx]
    mu_upper = data["mu upper"][idx]

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            u[0] = x[i, j]
            u[1] = y[i, j]

            # Compute the residuals at the current point
            model.compute_residuals(u, res)

            # Compute the penalty terms
            penalty = np.zeros(u.size)

            t_lower = u[lb_mask] - lb[lb_mask]
            t_upper = ub[ub_mask] - u[ub_mask]

            if t_lower.size > 0:
                penalty[lb_mask] += np.sum(mu_lower * -np.log(t_lower + 1e-10))

            if t_upper.size > 0:
                penalty[ub_mask] += np.sum(mu_upper * -np.log(t_upper + 1e-10))

            # Add the penalty terms to the residual
            res = res + penalty

            norms[i, j] = np.linalg.norm(res)

    return ax.contour(x, y, norms, **kwargs)


def bounds(ax, model, xlim, ylim, **kwargs):
    """Shade bounds on problem.

    Additional keyword arguments for matplotlib's contour
    and contourf functions can be added to this function.

    Parameters
    ----------
    ax : matplotlib axis object
        Axis on which to plot shaded bounds.
    model : ipnewt model
        Model from which to extract bounds to plot.
    xlim : two-element iterable
        Lower and upper bounds to plot contours along the x-axis.
    ylim : two-element iterable
        Lower and upper bounds to plot contours along the y-axis.
    """
    x, y = np.meshgrid(np.linspace(*xlim, 100), np.linspace(*ylim, 100), indexing="ij")

    # Plot lower bounds
    if np.isfinite(model.lower[0]):
        ax.contourf(x, y, x, levels=[-np.inf, model.lower[0]], **kwargs)
        ax.contour(x, y, x, levels=[model.lower[0]], **kwargs)
    if np.isfinite(model.lower[1]):
        ax.contourf(x, y, y, levels=[-np.inf, model.lower[1]], **kwargs)
        ax.contour(x, y, y, levels=[model.lower[1]], **kwargs)

    # Plot upper bounds
    if np.isfinite(model.upper[0]):
        ax.contourf(x, y, x, levels=[model.upper[0], np.inf], **kwargs)
        ax.contour(x, y, x, levels=[model.upper[0]], **kwargs)
    if np.isfinite(model.upper[1]):
        ax.contourf(x, y, y, levels=[model.upper[1], np.inf], **kwargs)
        ax.contour(x, y, y, levels=[model.upper[1]], **kwargs)


def newton_path(ax, data, **kwargs):
    """Plot the Newton solver's path.

    Additional keyword arguments for matplotlib's plot
    function can be added to this function.

    Parameters
    ----------
    ax : matplotlib axis object
        Axis on which to plot the solver path.
    data : dict
        NewtonSolver data attribute. The path and convergence
        information is extracted from here.
    """
    states = np.array(data["states"])
    ax.plot(states[:, 0], states[:, 1], "-o", **kwargs)


def newton_soln_viz(ax, model, data, xlim, iter=0, penalty=False, pt=False, **kwargs):
    """Plots the lines that are the intersection
    with zero of the two planes of a 2D Newton step (each
    plane is the linearized residual around the current states).

    Parameters
    ----------
    ax : matplotlib axis object
        Axis on which to plot the solver path.
    model : ipnewt model
        Model used to compute residuals and derivatives.
    data : dict
        NewtonSolver data attribute. The path and convergence
        information is extracted from here.
    xlim : two-element iterable
        Lower and upper bounds to plot lines along the x-axis.
    iter : int, optional
        Iteration of solve from which to plot lines.
    """
    # Compute residual and Jacobian at desired point
    u = data["states"][iter]
    res = np.zeros(2)
    J = np.zeros((2, 2))
    model.compute_residuals(u, res)
    model.compute_jacobian(u, J)
    if penalty:
        dp_du = data["linear_data"][iter]["penalty_vector"]
        J += np.diag(dp_du)

    if pt:
        tau = data["linear_data"][iter]["pt_vector"]
        J += np.diag(tau)

    u_1 = np.linspace(*xlim, 2)

    u_2_res_1 = J[0, 0] / J[0, 1] * (u[0] - u_1) + u[1] - res[0] / J[0, 1]
    u_2_res_2 = J[1, 0] / J[1, 1] * (u[0] - u_1) + u[1] - res[1] / J[1, 1]

    ymin = min(np.minimum(u_2_res_1, u_2_res_2))
    ymax = max(np.maximum(u_2_res_1, u_2_res_2))

    u1_grid, u2_grid = np.meshgrid(np.linspace(*xlim, 100), np.linspace(ymin, ymax, 100))
    lin_res_1 = J[0, 0] * (u1_grid - u[0]) + J[0, 1] * (u2_grid - u[1]) + res[0]
    lin_res_2 = J[1, 0] * (u1_grid - u[0]) + J[1, 1] * (u2_grid - u[1]) + res[1]
    lin_res_norm = np.sqrt(lin_res_1 ** 2 + lin_res_2 ** 2)
    ax.contour(u1_grid, u2_grid, lin_res_norm, 100)

    # print(J[0, 0] / J[0, 1])
    # print(J[1, 0] / J[1, 1])
    # print(u_2_res_2)

    du = np.linalg.solve(J, -res)
    u = u + du
    ax.scatter(*u)

    ax.plot(u_1, u_2_res_1, "--", **kwargs)
    ax.plot(u_1, u_2_res_2, **kwargs)

    ax.set_ylim([-10, 15])
