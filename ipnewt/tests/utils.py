#!/usr/bin/env python
"""
@File    :   utils.py
@Time    :   2021/12/3
@Desc    :   Utilities that are useful for testing
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================


def check_model_derivatives(model, use_complex=True, step_size=None, print_results=True, atol=1e-10, rtol=1e-10):
    """Use complex step (or finite difference) to check the derivatives of a model.

    Parameters
    ----------
    model : ipnewt model
        Model to test derivatives on.
    use_complex : bool, optional
        Use complex step (forward finite difference if not), by default True
    step_size : float, optional
        Step size for finite difference, by default 1e-100 for complex and 1e-6 for fd
    print_results : bool, optional
        Print results of derivative check, by default True
    atol : float, optional
        Absolute tolerance all derivatives must meet
    rtol : float, optional
        Relative tolerance all derivatives must meet

    Returns
    -------
    bool
        Returns False if at least one entry does not meet both atol and rtol
    """
    # Set default step size
    if step_size is None:
        step_size = 1e-6
        if use_complex:
            step_size = 1e-100

    n_states = model.n_states
    dtype = float
    if use_complex:
        dtype = complex
    u = np.array(model.states, dtype=dtype)

    # Get the Jacobian computed by the model
    J_model = np.zeros((n_states, n_states), dtype=dtype)
    model.compute_jacobian(u, J_model)

    # Build the Jacobian to check the model
    J_check = np.zeros((n_states, n_states))
    res_pert = np.zeros(n_states, dtype=dtype)
    res_orig = np.zeros(n_states, dtype=dtype)
    model.compute_residuals(u, res_orig)

    for i in range(n_states):
        # Perturb the ith state
        if use_complex:
            u[i] += step_size * 1j
        else:
            u[i] += step_size

        # Evaluate the new residuals
        model.compute_residuals(u, res_pert)

        # Compute the ith column of the Jacobian
        if use_complex:
            J_check[:, i] = np.imag(res_pert) / step_size
        else:
            J_check[:, i] = (res_pert - res_orig) / step_size

        # Unperturb the ith state
        if use_complex:
            u[i] -= step_size * 1j
        else:
            u[i] -= step_size

    # Compute tolerances
    abs_diff = J_model - J_check
    rel_diff = (J_model - J_check) / J_check

    # Print results if desired
    if print_results:
        print("==================== Checking derivatives ====================")
        print(
            f"Using {'complex step' if use_complex else 'forward finite difference'} with a step size of {step_size:e}"
        )

        print(f"\nJacobian returned by model:")
        print(J_model)

        print(f"\nJacobian with finite different checks:")
        print(J_check)

        print(f"\nAbsolute difference:")
        print(abs_diff)

        print(f"\nRelative difference:")
        print(rel_diff)

    # Check the tolerances
    results = np.logical_and(abs_diff > atol, rel_diff > rtol)
    return not results.any()
