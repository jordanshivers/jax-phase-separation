"""Random-matrix theory predictions for the number of coexisting phases.

Uses the Wigner semicircle CDF approach from Shrinivas & Brenner 2021,
with optional Tracy-Widom edge corrections.
"""

import numpy as np
from scipy import special


def _wigner_cdf(x, R):
    """CDF of the Wigner semicircle distribution on [-R, R].

    F(x) = 0.5 + x*sqrt(R^2 - x^2) / (pi*R^2) + arcsin(x/R)/pi
    """
    x = np.clip(x, -R, R)
    return 0.5 + x * np.sqrt(R ** 2 - x ** 2) / (np.pi * R ** 2) + np.arcsin(x / R) / np.pi


def n_phases_linear(N, sigma, beta=None):
    """Linearised prediction for number of coexisting phases.

    N_phases ~ (N-1)/2 * (1 - sqrt(N)/(2*sigma*beta)) + 1

    Parameters
    ----------
    N : int or array
        Number of solute components.
    sigma : float or array
        Standard deviation of chi distribution.
    beta : float or array, optional
        Total solute volume fraction.  Default equimolar: N/(N+1).

    Returns
    -------
    n_ph : float or array
    """
    N = np.asarray(N, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if beta is None:
        beta = N / (N + 1.0)
    beta = np.asarray(beta, dtype=float)

    n_ph = (N - 1.0) / 2.0 * (1.0 - np.sqrt(N) / (2.0 * sigma * beta)) + 1.0
    return np.maximum(n_ph, 1.0)


def n_phases_wigner(N, sigma, beta=None):
    """Full Wigner semicircle CDF prediction.

    <N_ph> = N * F_{2*sigma*sqrt(N)}(lambda <= -N/beta) + 1

    Parameters
    ----------
    N, sigma, beta : same as ``n_phases_linear``

    Returns
    -------
    n_ph : float or array
    """
    N = np.asarray(N, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if beta is None:
        beta = N / (N + 1.0)
    beta = np.asarray(beta, dtype=float)

    R = 2.0 * sigma * np.sqrt(N)
    threshold = -N / beta

    n_ph = np.where(
        np.abs(threshold) <= R,
        N * _wigner_cdf(threshold, R) + 1.0,
        np.where(threshold < -R, 1.0, N + 1.0),
    )
    return np.maximum(n_ph, 1.0)


def n_phases_turnover_linear(N, sigma, koff, kappa_eff=0.01, beta=None):
    """Linearised prediction for number of phases with active turnover.

    From the dispersion relation with first-order degradation, the most
    unstable wavevector gives growth rate (beta/(4*N*kappa_eff))*lam^2 - koff,
    so the instability threshold for chi eigenvalues shifts by
    -2*sqrt(N*kappa_eff*koff/beta).  Substituting into the linearised CDF:

        n_ph(k) = (N-1)/2 * (1 - (N/beta + 2*sqrt(N*kappa_eff*koff/beta))
                                   / (2*sigma*sqrt(N))) + 1

    Parameters
    ----------
    N : int
    sigma : float
        Per-entry standard deviation of the chi matrix.
    koff : float or array
        Degradation rate.
    kappa_eff : float
        Effective surface tension = kappa_mag * lmbda.
    beta : float, optional

    Returns
    -------
    n_ph : float or array
    """
    N = np.asarray(N, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    koff = np.asarray(koff, dtype=float)
    if beta is None:
        beta = N / (N + 1.0)
    beta = np.asarray(beta, dtype=float)

    R = 2.0 * sigma * np.sqrt(N)
    threshold_abs = N / beta + 2.0 * np.sqrt(N * kappa_eff * koff / beta)
    n_ph = (N - 1.0) / 2.0 * (1.0 - threshold_abs / R) + 1.0
    return np.maximum(n_ph, 1.0)


def n_phases_turnover_wigner(N, sigma, koff, kappa_eff=0.01, beta=None):
    """Nonlinear Wigner-CDF prediction for phases under active turnover.

    Maximising the growth rate sigma_i(q) = -(beta/N)*q^2*(lam_i + kappa_eff*q^2) - koff
    over q gives the instability condition lam_i^2 > 4*N*kappa_eff*koff/beta.
    The threshold for chi eigenvalues therefore shifts by
    -2*sqrt(N*kappa_eff*koff/beta):

        n_ph(k) = N * F_{2*sigma*sqrt(N)}(-N/beta - 2*sqrt(N*kappa_eff*koff/beta)) + 1

    Parameters
    ----------
    N, sigma, koff, kappa_eff, beta : see ``n_phases_turnover_linear``

    Returns
    -------
    n_ph : float or array
    """
    N = np.asarray(N, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    koff = np.asarray(koff, dtype=float)
    if beta is None:
        beta = N / (N + 1.0)
    beta = np.asarray(beta, dtype=float)

    R = 2.0 * sigma * np.sqrt(N)
    threshold = -N / beta - 2.0 * np.sqrt(N * kappa_eff * koff / beta)

    n_ph = np.where(
        np.abs(threshold) <= R,
        N * _wigner_cdf(threshold, R) + 1.0,
        np.where(threshold < -R, 1.0, N + 1.0),
    )
    return np.maximum(n_ph, 1.0)


def n_phases_tracy_widom(N, sigma, beta=None):
    """Wigner prediction with Tracy-Widom edge correction.

    The extreme eigenvalue of a random matrix fluctuates around the
    semicircle edge by ~ N^{-1/6}.  This shifts the effective edge
    inward, reducing the number of unstable modes at small N.

    Parameters
    ----------
    N, sigma, beta : same as above

    Returns
    -------
    n_ph : float or array
    """
    N = np.asarray(N, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if beta is None:
        beta = N / (N + 1.0)
    beta = np.asarray(beta, dtype=float)

    tw_mean = -1.2065
    R_effective = 2.0 * sigma * np.sqrt(N) + tw_mean * sigma * N ** (1.0 / 6.0)
    threshold = -N / beta

    R_effective = np.maximum(R_effective, 0.0)
    n_ph = np.where(
        (R_effective > 0) & (np.abs(threshold) <= R_effective),
        N * _wigner_cdf(threshold, np.maximum(R_effective, 1e-10)) + 1.0,
        np.where(threshold < -R_effective, N + 1.0, 1.0),
    )
    return np.maximum(n_ph, 1.0)


def optimal_N_components(sigma):
    """Optimal number of components maximising phases in fixed-sigma ensemble.

    N_opt ~ (4/3 * sigma)^2 = 16/9 * sigma^2

    Parameters
    ----------
    sigma : float

    Returns
    -------
    N_opt : float
    """
    return 16.0 / 9.0 * sigma ** 2


def wigner_semicircle_pdf(x, R):
    """Probability density of the Wigner semicircle distribution on [-R, R].

    p(x) = 2/(pi*R^2) * sqrt(R^2 - x^2)   for |x| <= R
    """
    x = np.asarray(x, dtype=float)
    R = float(R)
    out = np.zeros_like(x)
    mask = np.abs(x) <= R
    out[mask] = 2.0 / (np.pi * R ** 2) * np.sqrt(R ** 2 - x[mask] ** 2)
    return out
