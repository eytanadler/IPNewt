#!/usr/bin/env python
"""
@File    :   two_dim_viz.py
@Time    :   2021/12/3
@Desc    :   2D visualization tools
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================


def contour(ax, model, xlim, ylim, **kwargs):
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
    
    Returns
    -------
    matplotlib QuadContourSet
        Useful to make colorbar, can be ignored
    """
    # Generate a grid on which to evaluate the residual norm
    x, y = np.meshgrid(np.linspace(*xlim, 100), np.linspace(*ylim, 100),
                       indexing='xy')
    norms = np.zeros(x.shape)

    # States and residuals
    u = np.zeros(2)
    res = np.zeros(2)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            u[0] = x[i, j]
            u[1] = y[i, j]

            # Compute the residuals at the current point
            model.compute_residuals(u, res)

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
    x, y = np.meshgrid(np.linspace(*xlim, 100), np.linspace(*ylim, 100),
                       indexing='ij')

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
    ax.plot(states[:, 0], states[:, 1], '-o', **kwargs)
