"""Utilities for random matrices, initial conditions, and plotting."""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from jax_phase_separation.solver import SimulationParams


def generate_chi_matrix(n_com, chi_mean, chi_std, key, alpha_ensemble=False):
    """Generate a random symmetric interaction matrix.

    Parameters
    ----------
    n_com : int
        Number of solute components.
    chi_mean : float
        Mean of the chi distribution.
    chi_std : float
        Standard deviation.  If ``alpha_ensemble`` is True this is the
        proportionality constant alpha, and the actual std is
        alpha * sqrt(n_com).
    key : jax.random.PRNGKey
    alpha_ensemble : bool
        If True, interpret ``chi_std`` as alpha and set
        sigma = alpha * sqrt(n_com).

    Returns
    -------
    chi : jnp.ndarray (n_com, n_com)
    """
    if alpha_ensemble:
        sigma = chi_std * jnp.sqrt(float(n_com))
    else:
        sigma = chi_std

    noise = sigma * jax.random.normal(key, (n_com, n_com))
    noise_sym = (noise + noise.T) / jnp.sqrt(2.0)
    chi = noise_sym + chi_mean
    chi = chi.at[jnp.diag_indices(n_com)].set(0.0)
    return chi


def generate_B_matrix(n_com, b_mean, b_std, key, normalize="1/sqrt(N)", zero_diag=True):
    """Generate a random asymmetric coupling matrix.

    Parameters
    ----------
    n_com : int
    b_mean : float
        Mean of the B distribution.
    b_std : float
        Standard deviation before normalisation.
    key : jax.random.PRNGKey
    normalize : {"1/sqrt(N)", "none"}
        ``"1/sqrt(N)"`` scales ``b_std`` by ``1/sqrt(n_com)`` so the
        spectral radius stays O(1) as N grows (random-matrix convention).
    zero_diag : bool
        If True, set diagonal to zero so the coupling is a pure
        cross-species effect.

    Returns
    -------
    B : jnp.ndarray (n_com, n_com)
    """
    if normalize == "1/sqrt(N)":
        sigma = b_std / jnp.sqrt(float(n_com))
    elif normalize == "none":
        sigma = b_std
    else:
        raise ValueError(f"unknown normalize={normalize!r}")

    B = b_mean + sigma * jax.random.normal(key, (n_com, n_com))
    if zero_diag:
        B = B.at[jnp.diag_indices(n_com)].set(0.0)
    return B


def generate_initial_conditions(n_com, N_grid, beta=None, noise_strength=0.01, key=None):
    """Create equimolar initial volume-fraction fields with small noise.

    Parameters
    ----------
    n_com : int
    N_grid : int
    beta : float, optional
        Total solute volume fraction.  Default N/(N+1).
    noise_strength : float
    key : jax.random.PRNGKey, optional

    Returns
    -------
    c0 : jnp.ndarray (n_com, N_grid, N_grid)
    """
    if beta is None:
        beta = n_com / (n_com + 1.0)
    if key is None:
        key = jax.random.PRNGKey(0)

    phi_mean = beta / n_com
    noise = noise_strength * phi_mean * jax.random.uniform(
        key, (n_com, N_grid, N_grid), minval=-1.0, maxval=1.0
    )
    # Keep the total solute fraction fixed.
    noise = noise - noise.mean(axis=0, keepdims=True)
    c0 = phi_mean + noise
    return c0


def build_params(chi, n_com, beta=None, lmbda=0.01, dt=5e-6, kappa_mag=1.0,
                 kon=0.0, r_mu=1.0, B=None, reactions=None, omit_solvent_flux=False,
                 A=None):
    """Build a ``SimulationParams`` object.

    Parameters
    ----------
    chi : jnp.ndarray (n_com, n_com)
    n_com : int
    beta : float, optional
    lmbda : float
    dt : float
    kappa_mag : float
    kon : float
        Legacy uniform production rate.  Ignored when ``reactions`` is given.
    r_mu : float or array-like of shape ``(n_com,)`` or ``(n_com, 1)``
        Per-species inverse diffusion: Fourier transport uses ``-k^2 c_i / r_i``,
        so ``D_i \\propto 1/r_i``. A scalar broadcasts to all components.
    B : jnp.ndarray (n_com, n_com), optional
        Nonreciprocal coupling matrix.  ``None`` disables (zeros).
    reactions : ReactionSystem, optional
        General reaction network.  When provided, replaces the legacy
        ``kon``/``koff`` source term.
    omit_solvent_flux : bool
        If True, omit the solvent-gradient cross-flux (classical Fickian
        diffusion only, plus any CH terms from ``chi``/``lmbda``).
    A : float, optional
        Implicit biharmonic stabilisation.  If omitted, uses
        ``max(|chi|_\\mathrm{max}, |B|_\\mathrm{max}) * lmbda`` when ``B`` is set.
        Large ``|B|`` inflates this and can overdamp spatial structure in active
        runs; pass ``A=float(jnp.abs(chi).max()) * lmbda`` to keep the
        equilibrium scaling only.

    Returns
    -------
    SimulationParams
    """
    if beta is None:
        beta = n_com / (n_com + 1.0)

    kappa = jnp.eye(n_com) * kappa_mag
    chi_s = jnp.zeros((n_com, n_com))
    r_arr = jnp.asarray(r_mu)
    if r_arr.ndim == 0:
        r = jnp.ones((n_com, 1), dtype=r_arr.dtype) * r_arr
    else:
        r = jnp.reshape(r_arr, (n_com, 1))
        if int(r.shape[0]) != int(n_com):
            raise ValueError(
                f"r_mu must be scalar or length-{n_com} vector; got shape {r_arr.shape}"
            )
    koff = kon * n_com / beta
    if A is not None:
        A = float(A)
    else:
        A_from_chi = float(jnp.abs(chi).max()) * lmbda
        if B is not None:
            A_from_B = float(jnp.abs(jnp.asarray(B)).max()) * lmbda
            A = max(A_from_chi, A_from_B)
        else:
            A = A_from_chi

    return SimulationParams(
        chi=chi,
        kappa=kappa,
        chi_s=chi_s,
        r=r,
        lmbda=lmbda,
        dt=dt,
        kon=kon,
        koff=koff,
        A=A,
        B=B,
        reactions=reactions,
        omit_solvent_flux=omit_solvent_flux,
    )


_CMAPS = [
    "Greens", "Oranges", "Blues", "Reds", "Greys",
    "Purples", "PuRd", "BuGn", "YlGn", "YlOrBr",
    "Greens", "Oranges", "Blues", "Reds", "Greys",
    "Purples", "PuRd", "BuGn", "YlGn", "YlOrBr",
]


def plot_volume_fractions(c, ncols=4, vmin=0.0, vmax=0.75, figsize=None):
    """Plot heatmaps of all component volume fractions.

    Parameters
    ----------
    c : array (N_com, Nx, Ny)
    ncols : int
    vmin, vmax : float
    figsize : tuple, optional

    Returns
    -------
    fig, axes
    """
    c = np.asarray(c)
    n_com = c.shape[0]
    nrows = int(np.ceil(n_com / ncols))
    if figsize is None:
        figsize = (3 * ncols, 3 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if idx < n_com:
            cmap = _CMAPS[idx % len(_CMAPS)]
            im = ax.imshow(c[idx], cmap=cmap, vmin=vmin, vmax=vmax,
                           origin="lower", interpolation="nearest")
            ax.set_title(f"$\\phi_{{{idx}}}$", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")
    fig.tight_layout()
    return fig, axes


def plot_phase_map(labels, n_phases, ax=None):
    """Plot a spatial map of assigned phases.

    Parameters
    ----------
    labels : array (Nx, Ny) int
    n_phases : int
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    cmap = plt.get_cmap("ocean", n_phases)
    im = ax.imshow(labels, cmap=cmap, origin="lower", interpolation="nearest")
    plt.colorbar(im, ax=ax, ticks=range(n_phases), label="phase")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def plot_partition_ratios(partitions, ax=None):
    """Bar-chart of partition ratios per phase.

    Parameters
    ----------
    partitions : array (n_phases, N_com)
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    partitions = np.asarray(partitions)
    n_phases, n_com = partitions.shape
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, n_com * 0.5), 4))
    else:
        fig = ax.figure

    x = np.arange(n_com)
    width = 0.8 / n_phases
    for g in range(n_phases):
        ax.bar(x + g * width, partitions[g], width, label=f"phase {g + 1}")
    ax.axhline(1.0, ls="--", color="k", lw=0.8)
    ax.set_xlabel("component")
    ax.set_ylabel("partition ratio")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    return fig, ax
