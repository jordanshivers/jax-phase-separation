"""IMEX FFT solver for differentiable multicomponent phase-field dynamics."""

from __future__ import annotations

import warnings
from functools import partial
from typing import NamedTuple, Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax_phase_separation.reactions import (
    ReactionSystem,
    empty_reaction_system,
    apply_reactions,
)

_WARNED_SIMULATE_PROGRESS = False
_WARNED_SNAPSHOTS_PROGRESS = False


class SimulationParams(NamedTuple):
    """JAX-traceable solver parameters."""
    chi: jnp.ndarray          # (N_com, N_com) interaction matrix
    kappa: jnp.ndarray        # (N_com, N_com) surface tension matrix
    chi_s: jnp.ndarray        # (N_com, N_com) component-solvent interactions
    r: jnp.ndarray            # (N_com, 1) polymer lengths
    lmbda: float              # surface tension coefficient
    dt: float                 # time step
    kon: float                # production rate (legacy linear turnover)
    koff: float               # degradation rate (legacy linear turnover)
    A: float                  # implicit stabilisation magnitude
    B: jnp.ndarray = None     # (N_com, N_com) nonreciprocal coupling; zeros disables
    reactions: Optional[ReactionSystem] = None   # general reaction network (overrides kon/koff)
    # Drop solvent-gradient flux for Fickian reaction-diffusion examples.
    omit_solvent_flux: bool = False


def make_wavenumbers(N: int, dx: float = None):
    """Build 2-D wavenumber arrays for a periodic NxN grid."""
    if dx is None:
        dx = 1.0 / N
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(N, dx)
    kx = k[:, None]
    ky = k[None, :]
    k2 = kx ** 2 + ky ** 2
    k4 = kx ** 4 + ky ** 4
    return dict(kx=kx, ky=ky, k2=k2, k4=k4, kxj=kx * 1j, kyj=ky * 1j)


def _phi_activation(x, activation: str):
    """Apply the selected nonreciprocal activation."""
    if activation == "identity":
        return x
    if activation == "tanh":
        return jnp.tanh(x)
    raise ValueError(f"unknown activation={activation!r}")


def _add_B_term(muHat, c, cHat, B, activation: str, N_com: int, N: int):
    """Add the nonreciprocal B * phi(c) contribution to muHat."""
    if activation == "identity":
        return muHat + (B @ cHat.reshape(N_com, -1)).reshape(N_com, N, N)
    phi_c = _phi_activation(c, activation)
    B_phi = (B @ phi_c.reshape(N_com, -1)).reshape(N_com, N, N)
    return muHat + jnp.fft.fftn(B_phi, axes=(-2, -1))


def _cal_NHat_simple(
    c, cHat, chi, kappa, chi_s, r, lmbda, A, B, wn, activation, omit_solvent_flux: bool
):
    """Explicit RHS with mobility M_i = M * phi_i."""
    N_com = c.shape[0]
    N = c.shape[1]
    k2 = wn["k2"]
    k4 = wn["k4"]
    kxj = wn["kxj"]
    kyj = wn["kyj"]

    cs = jnp.maximum(1.0 - jnp.sum(c, axis=0, keepdims=True), 1e-30)
    cs = jnp.broadcast_to(cs, c.shape)
    csHat = jnp.fft.fftn(cs, axes=(-2, -1))

    r3d = r.reshape(-1, 1, 1)
    NDiff = -k2[None, :, :] * cHat / r3d

    NImp = A * k4[None, :, :] * cHat

    kappa_term = cHat * k2[None, :, :] * lmbda
    cHat_flat = cHat.reshape(N_com, -1)
    muHat = (chi @ cHat_flat).reshape(N_com, N, N)
    muHat = muHat + (kappa @ kappa_term.reshape(N_com, -1)).reshape(N_com, N, N)
    muHat = muHat - (chi_s @ cHat_flat).reshape(N_com, N, N)
    chi_s_diag = jnp.diag(chi_s[:, 0])
    muHat = muHat + (chi_s_diag @ csHat.reshape(N_com, -1)).reshape(N_com, N, N)
    muHat = _add_B_term(muHat, c, cHat, B, activation, N_com, N)

    gradMuX = jnp.fft.ifftn(kxj[None, :, :] * muHat, axes=(-2, -1)).real
    gradMuY = jnp.fft.ifftn(kyj[None, :, :] * muHat, axes=(-2, -1)).real

    gradcsX = jnp.fft.ifftn(kxj[None, :, :] * csHat, axes=(-2, -1)).real
    gradcsY = jnp.fft.ifftn(kyj[None, :, :] * csHat, axes=(-2, -1)).real

    JsX = c * gradcsX / cs
    JsY = c * gradcsY / cs
    JsXHat = jnp.fft.fftn(JsX, axes=(-2, -1))
    JsYHat = jnp.fft.fftn(JsY, axes=(-2, -1))
    NsFlux = kxj[None, :, :] * JsXHat + kyj[None, :, :] * JsYHat

    JX = c * gradMuX
    JY = c * gradMuY
    JXHat = jnp.fft.fftn(JX, axes=(-2, -1))
    JYHat = jnp.fft.fftn(JY, axes=(-2, -1))
    NFlux = kxj[None, :, :] * JXHat + kyj[None, :, :] * JYHat

    core = NFlux + NDiff + NImp
    return core if omit_solvent_flux else core - NsFlux


def _cal_NHat_mobility(
    c, cHat, chi, kappa, chi_s, r, lmbda, A, B, wn, activation, omit_solvent_flux: bool
):
    """Explicit RHS with mobility M_i = M * phi_i * (1-phi_i)."""
    N_com = c.shape[0]
    N = c.shape[1]
    k2 = wn["k2"]
    k4 = wn["k4"]
    kxj = wn["kxj"]
    kyj = wn["kyj"]

    cs = jnp.maximum(1.0 - jnp.sum(c, axis=0, keepdims=True), 1e-30)
    cs = jnp.broadcast_to(cs, c.shape)
    csHat = jnp.fft.fftn(cs, axes=(-2, -1))

    gradcX = jnp.fft.ifftn(kxj[None, :, :] * cHat, axes=(-2, -1)).real
    gradcY = jnp.fft.ifftn(kyj[None, :, :] * cHat, axes=(-2, -1)).real
    JDX = gradcX * (1.0 - c)
    JDY = gradcY * (1.0 - c)
    JXDHat = jnp.fft.fftn(JDX, axes=(-2, -1))
    JYDHat = jnp.fft.fftn(JDY, axes=(-2, -1))
    NDiff = kxj[None, :, :] * JXDHat + kyj[None, :, :] * JYDHat

    NImp = A * k4[None, :, :] * cHat

    kappa_term = cHat * k2[None, :, :] * lmbda
    cHat_flat = cHat.reshape(N_com, -1)
    muHat = (chi @ cHat_flat).reshape(N_com, N, N)
    muHat = muHat + (kappa @ kappa_term.reshape(N_com, -1)).reshape(N_com, N, N)
    muHat = muHat - (chi_s @ cHat_flat).reshape(N_com, N, N)
    chi_s_diag = jnp.diag(chi_s[:, 0])
    muHat = muHat + (chi_s_diag @ csHat.reshape(N_com, -1)).reshape(N_com, N, N)
    muHat = _add_B_term(muHat, c, cHat, B, activation, N_com, N)

    gradMuX = jnp.fft.ifftn(kxj[None, :, :] * muHat, axes=(-2, -1)).real
    gradMuY = jnp.fft.ifftn(kyj[None, :, :] * muHat, axes=(-2, -1)).real

    gradcsX = jnp.fft.ifftn(kxj[None, :, :] * csHat, axes=(-2, -1)).real
    gradcsY = jnp.fft.ifftn(kyj[None, :, :] * csHat, axes=(-2, -1)).real

    c1mc = c * (1.0 - c)
    JsX = c1mc * gradcsX / cs
    JsY = c1mc * gradcsY / cs
    JsXHat = jnp.fft.fftn(JsX, axes=(-2, -1))
    JsYHat = jnp.fft.fftn(JsY, axes=(-2, -1))
    NsFlux = kxj[None, :, :] * JsXHat + kyj[None, :, :] * JsYHat

    JX = c1mc * gradMuX
    JY = c1mc * gradMuY
    JXHat = jnp.fft.fftn(JX, axes=(-2, -1))
    JYHat = jnp.fft.fftn(JY, axes=(-2, -1))
    NFlux = kxj[None, :, :] * JXHat + kyj[None, :, :] * JYHat

    core = NFlux + NDiff + NImp
    return core if omit_solvent_flux else core - NsFlux


def _make_step_fn_arrays(
    chi,
    kappa,
    chi_s,
    r,
    lmbda,
    A,
    dt,
    kon,
    koff,
    B,
    reactions,
    wn: dict,
    Ainv,
    mobility_flag: bool,
    activation: str,
    rate_law: str,
    use_reactions: bool,
    omit_solvent_flux: bool,
):
    """Return the scan step closed over static solver choices."""
    cal_NHat = _cal_NHat_mobility if mobility_flag else _cal_NHat_simple

    def step(carry, _):
        c, cHat = carry
        ncHat = cal_NHat(
            c, cHat, chi, kappa, chi_s, r, lmbda, A, B, wn, activation, omit_solvent_flux
        )
        cHat_new = (cHat + dt * ncHat) * Ainv[None, :, :]
        c_new = jnp.fft.ifftn(cHat_new, axes=(-2, -1)).real
        c_new = 1e-6 * jax.nn.softplus(c_new / 1e-6)
        if use_reactions:
            source = apply_reactions(c_new, reactions, rate_law, chi, chi_s)
            c_new = c_new + source * dt
        else:
            c_new = c_new + (kon - koff * c_new) * dt
        cHat_new = jnp.fft.fftn(c_new, axes=(-2, -1))
        return (c_new, cHat_new), None

    return step


def _resolve_B(params: SimulationParams):
    """Return a concrete (N_com, N_com) B matrix; zeros if ``params.B is None``."""
    n_com = params.chi.shape[0]
    if params.B is None:
        return jnp.zeros((n_com, n_com))
    return jnp.asarray(params.B)


def _resolve_reactions(params: SimulationParams):
    """Return (reactions pytree, use_reactions flag)."""
    n_com = params.chi.shape[0]
    if params.reactions is None:
        return empty_reaction_system(n_com), False
    return params.reactions, True


def _make_step_fn(
    params: SimulationParams,
    wn: dict,
    Ainv,
    mobility_flag: bool,
    activation: str = "identity",
    rate_law: str = "mass_action",
):
    """Return a scan step for custom notebook scans."""
    reactions, use_reactions = _resolve_reactions(params)
    return _make_step_fn_arrays(
        params.chi,
        params.kappa,
        params.chi_s,
        params.r,
        params.lmbda,
        params.A,
        params.dt,
        jnp.asarray(params.kon),
        jnp.asarray(params.koff),
        _resolve_B(params),
        reactions,
        wn,
        Ainv,
        mobility_flag,
        activation,
        rate_law,
        use_reactions,
        params.omit_solvent_flux,
    )


@partial(
    jax.jit,
    static_argnames=(
        "N_grid", "n_steps", "mobility_flag",
        "activation", "rate_law", "use_reactions", "omit_solvent_flux",
    ),
)
def _simulate_jitted(
    c0,
    chi,
    kappa,
    chi_s,
    r,
    lmbda,
    dt,
    A,
    kon,
    koff,
    B,
    reactions,
    N_grid: int,
    n_steps: int,
    mobility_flag: bool,
    activation: str,
    rate_law: str,
    use_reactions: bool,
    omit_solvent_flux: bool,
):
    wn = make_wavenumbers(N_grid)
    Ainv = 1.0 / (1.0 + A * wn["k4"] * dt)
    cHat0 = jnp.fft.fftn(c0, axes=(-2, -1))
    step_fn = _make_step_fn_arrays(
        chi, kappa, chi_s, r, lmbda, A, dt, kon, koff, B, reactions,
        wn, Ainv, mobility_flag, activation, rate_law, use_reactions,
        omit_solvent_flux,
    )
    (c_final, _), _ = jax.lax.scan(step_fn, (c0, cHat0), None, length=n_steps)
    return c_final


@partial(
    jax.jit,
    static_argnames=(
        "N_grid", "n_steps", "save_every", "mobility_flag",
        "activation", "rate_law", "use_reactions", "omit_solvent_flux",
    ),
)
def _simulate_with_snapshots_jitted(
    c0,
    chi,
    kappa,
    chi_s,
    r,
    lmbda,
    dt,
    A,
    kon,
    koff,
    B,
    reactions,
    N_grid: int,
    n_steps: int,
    save_every: int,
    mobility_flag: bool,
    activation: str,
    rate_law: str,
    use_reactions: bool,
    omit_solvent_flux: bool,
):
    wn = make_wavenumbers(N_grid)
    Ainv = 1.0 / (1.0 + A * wn["k4"] * dt)
    cHat0 = jnp.fft.fftn(c0, axes=(-2, -1))
    inner_step = _make_step_fn_arrays(
        chi, kappa, chi_s, r, lmbda, A, dt, kon, koff, B, reactions,
        wn, Ainv, mobility_flag, activation, rate_law, use_reactions,
        omit_solvent_flux,
    )
    n_snapshots = n_steps // save_every

    def outer_step(carry, x):
        (c, cHat), _ = jax.lax.scan(inner_step, carry, None, length=save_every)
        return (c, cHat), c

    (c_final, _), snapshots = jax.lax.scan(
        outer_step, (c0, cHat0), jnp.arange(n_snapshots), length=n_snapshots
    )
    return c_final, snapshots


def _params_to_jitted_arrays(params: SimulationParams):
    """Pack scalar fields as arrays for ``_simulate_*_jitted``."""
    return (
        jnp.asarray(params.lmbda),
        jnp.asarray(params.dt),
        jnp.asarray(params.A),
        jnp.asarray(params.kon),
        jnp.asarray(params.koff),
    )


def simulate(
    c0,
    params: SimulationParams,
    N_grid: int,
    n_steps: int,
    mobility_flag: bool = False,
    activation: str = "identity",
    rate_law: str = "mass_action",
    progress_bar: bool = False,
):
    """Run the phase-field simulation and return the final state.

    Parameters
    ----------
    c0 : array (N_com, N_grid, N_grid)
        Initial volume-fraction field.
    params : SimulationParams
    N_grid : int
    n_steps : int
    mobility_flag : bool
        If True use M_i = M*phi_i*(1-phi_i); otherwise M_i = M*phi_i.
    activation : {"identity", "tanh"}
        Activation applied to c before the nonreciprocal coupling term
        ``B * phi(c)``. Ignored when ``params.B`` is None or zero.
    rate_law : {"mass_action", "thermodynamic"}
        Rate law for the general reaction system. Ignored when
        ``params.reactions`` is None (legacy ``kon``/``koff`` path).
    progress_bar : bool
        Ignored by the jitted implementation; kept for API compatibility.

    Returns
    -------
    c_final : array (N_com, N_grid, N_grid)
    """
    global _WARNED_SIMULATE_PROGRESS
    if progress_bar and not _WARNED_SIMULATE_PROGRESS:
        _WARNED_SIMULATE_PROGRESS = True
        warnings.warn(
            "simulate(..., progress_bar=True) is ignored; the solver uses a "
            "jitted forward pass. Wrap the call with tqdm manually if needed.",
            UserWarning,
            stacklevel=2,
        )
    lmbda, dt, A, kon, koff = _params_to_jitted_arrays(params)
    B = _resolve_B(params)
    reactions, use_reactions = _resolve_reactions(params)
    return _simulate_jitted(
        c0,
        params.chi,
        params.kappa,
        params.chi_s,
        params.r,
        lmbda,
        dt,
        A,
        kon,
        koff,
        B,
        reactions,
        N_grid,
        n_steps,
        mobility_flag,
        activation,
        rate_law,
        use_reactions,
        params.omit_solvent_flux,
    )


def simulate_with_snapshots(
    c0,
    params: SimulationParams,
    N_grid: int,
    n_steps: int,
    save_every: int = 1,
    mobility_flag: bool = False,
    activation: str = "identity",
    rate_law: str = "mass_action",
    progress_bar: bool = False,
    include_initial: bool = False,
    mode: str = "stream",
):
    """Run simulation and return snapshots of the concentration field.

    Only every ``save_every``-th step is stored.

    Parameters
    ----------
    c0 : array (N_com, N_grid, N_grid)
    params : SimulationParams
    N_grid : int
    n_steps : int
    save_every : int
    mobility_flag : bool
    activation : {"identity", "tanh"}
        See :func:`simulate`.
    rate_law : {"mass_action", "thermodynamic"}
        See :func:`simulate`.
    progress_bar : bool
        Show chunk progress in ``mode="stream"``. Ignored in ``mode="jitted"``.
    include_initial : bool
        If True, prepend ``c0`` as the first snapshot (physical time 0).
        Remaining slices are at ``save_every``, ``2 * save_every``, … steps.
    mode : {"stream", "jitted"}
        ``"stream"`` uses Python orchestration over JIT-compiled chunks and is
        friendlier for notebooks and large snapshot sets. ``"jitted"`` uses one
        nested ``jax.lax.scan`` and is better for fully compiled runs.

    Returns
    -------
    c_final : array (N_com, N_grid, N_grid)
    snapshots : ndarray (n_snapshots, N_com, N_grid, N_grid)
        Returned as a NumPy array in host (CPU) memory.
    """
    global _WARNED_SNAPSHOTS_PROGRESS
    if save_every <= 0:
        raise ValueError(f"save_every must be positive; got {save_every}")

    n_snapshots = n_steps // save_every
    if n_snapshots < 1:
        raise ValueError(
            f"n_steps must be at least save_every; got n_steps={n_steps}, "
            f"save_every={save_every}"
        )

    mode = mode.lower()
    if mode not in {"stream", "jitted"}:
        raise ValueError(f"mode must be 'stream' or 'jitted'; got {mode!r}")

    if mode == "jitted":
        if progress_bar and not _WARNED_SNAPSHOTS_PROGRESS:
            _WARNED_SNAPSHOTS_PROGRESS = True
            warnings.warn(
                "simulate_with_snapshots(..., progress_bar=True, mode='jitted') "
                "is ignored; use mode='stream' for progress.",
                UserWarning,
                stacklevel=2,
            )
        lmbda, dt, A, kon, koff = _params_to_jitted_arrays(params)
        B = _resolve_B(params)
        reactions, use_reactions = _resolve_reactions(params)
        c_final, stack = _simulate_with_snapshots_jitted(
            c0,
            params.chi,
            params.kappa,
            params.chi_s,
            params.r,
            lmbda,
            dt,
            A,
            kon,
            koff,
            B,
            reactions,
            N_grid,
            n_steps,
            save_every,
            mobility_flag,
            activation,
            rate_law,
            use_reactions,
            params.omit_solvent_flux,
        )
        if include_initial:
            stack = jnp.concatenate([jnp.asarray(c0)[None, ...], stack], axis=0)
        return c_final, np.asarray(stack)

    iterator = range(n_snapshots)
    if progress_bar:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=n_snapshots)
        except ImportError:
            if not _WARNED_SNAPSHOTS_PROGRESS:
                _WARNED_SNAPSHOTS_PROGRESS = True
                warnings.warn(
                    "simulate_with_snapshots(..., progress_bar=True) requires "
                    "tqdm; continuing without a progress bar.",
                    UserWarning,
                    stacklevel=2,
                )

    snapshots = []
    c = c0

    for _ in iterator:
        c = simulate(
            c, params, N_grid, save_every,
            mobility_flag=mobility_flag,
            activation=activation, rate_law=rate_law,
            progress_bar=False,
        )
        snapshots.append(c)

    stack = jnp.stack(snapshots)
    if include_initial:
        stack = jnp.concatenate([jnp.asarray(c0)[None, ...], stack], axis=0)
    return c, np.asarray(stack)
