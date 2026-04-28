"""Free energy and stability routines for regular-solution mixtures."""

import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=())
def free_energy_density(phi, chi, chi_s):
    """Compute the mean-field regular-solution free-energy density.

    Parameters
    ----------
    phi : array, shape (N_com,) or (N_com, ...)
        Volume fractions of each solute species.  The first axis indexes
        components; any remaining axes are spatial.
    chi : array, shape (N_com, N_com)
        Symmetric pairwise interaction matrix (chi_ij).
    chi_s : array, shape (N_com,)
        Component-solvent interaction parameters (chi_is).

    Returns
    -------
    f : scalar or array matching spatial dimensions
        Free-energy density at each spatial point.
    """
    phi_safe = jnp.maximum(phi, 1e-30)
    phi_s = jnp.maximum(1.0 - jnp.sum(phi, axis=0), 1e-30)

    entropic_solute = jnp.sum(phi_safe * jnp.log(phi_safe), axis=0)
    entropic_solvent = phi_s * jnp.log(phi_s)

    interaction = 0.5 * jnp.einsum("i...,ij,j...->...", phi, chi, phi)

    solvent_interaction = jnp.sum(
        chi_s[:, None, None] * phi * phi_s[None, ...] if phi.ndim > 1
        else chi_s * phi * phi_s,
        axis=0,
    )

    return entropic_solute + interaction + entropic_solvent + solvent_interaction


def chemical_potential_bulk(phi, chi, chi_s):
    """Bulk chemical potential mu_i = d(f)/d(phi_i).

    Parameters
    ----------
    phi : array, shape (N_com,) or (N_com, Nx, Ny)
    chi : array, shape (N_com, N_com)
    chi_s : array, shape (N_com,)

    Returns
    -------
    mu : same shape as phi
    """
    phi_safe = jnp.maximum(phi, 1e-30)
    phi_s = jnp.maximum(1.0 - jnp.sum(phi, axis=0), 1e-30)

    log_term = jnp.log(phi_safe) - jnp.log(phi_s)

    if phi.ndim == 1:
        chi_term = chi @ phi
    else:
        chi_term = jnp.einsum("ij,j...->i...", chi, phi)

    mu = log_term + chi_term
    return mu


def compute_jacobian(n_com, beta, chi, chi_s, r):
    """Compute the Hessian/Jacobian matrix of the free energy at the
    equimolar uniform state.

    Parameters
    ----------
    n_com : int
        Number of solute components.
    beta : float
        Total solute volume fraction.
    chi : array, shape (n_com, n_com)
        Interaction matrix.
    chi_s : array, shape (n_com,) or (n_com, n_com)
        Component-solvent interactions.  If 1-D, broadcast to full matrix.
    r : array, shape (n_com,)
        Polymer lengths (degree of polymerization) per component.

    Returns
    -------
    J : array, shape (n_com, n_com)
    """
    n_com = int(n_com)
    n_com_f = float(n_com)

    diag = jnp.diag(n_com_f / (beta * jnp.asarray(r).ravel()))
    solvent_term = 1.0 / (1.0 - beta) * jnp.ones((n_com, n_com))

    chi_s = jnp.asarray(chi_s)
    chi_s_mat = chi_s if chi_s.ndim == 2 else jnp.broadcast_to(chi_s[:, None], (n_com, n_com))
    J = chi + diag + solvent_term - chi_s_mat - chi_s_mat.T
    return J


@jax.jit
def stability_analysis(J):
    """Eigenvalue decomposition of the Jacobian.

    Returns
    -------
    eigenvalues : sorted array (ascending)
    eigenvectors : columns are eigenvectors, ordered to match eigenvalues
    n_unstable : int, number of negative eigenvalues
    """
    w, v = jnp.linalg.eigh(J)
    n_unstable = jnp.sum(w < 0)
    return w, v, n_unstable
