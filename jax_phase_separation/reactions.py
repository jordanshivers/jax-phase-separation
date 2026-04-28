"""Reaction-network support for the phase-field solver."""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import jax.numpy as jnp

from jax_phase_separation.free_energy import chemical_potential_bulk


class ReactionSystem(NamedTuple):
    """JAX arrays describing a reaction network."""
    nu_plus:     jnp.ndarray   # (N_com, N_rxn)  reactant stoichiometry (internal)
    nu_minus:    jnp.ndarray   # (N_com, N_rxn)  product  stoichiometry (internal)
    nu_plus_ex:  jnp.ndarray   # (N_ex,  N_rxn)  reactant stoichiometry (external)
    nu_minus_ex: jnp.ndarray   # (N_ex,  N_rxn)  product  stoichiometry (external)
    mu_ex:       jnp.ndarray   # (N_ex,)         chemostat chemical potentials
    k_plus:      jnp.ndarray   # (N_rxn,)        mass-action forward rates
    k_minus:     jnp.ndarray   # (N_rxn,)        mass-action backward rates
    a:           jnp.ndarray   # (N_rxn,)        thermodynamic prefactor


def empty_reaction_system(n_com: int, n_ex: int = 0) -> ReactionSystem:
    """Return an inert reaction system."""
    return ReactionSystem(
        nu_plus=jnp.zeros((n_com, 0)),
        nu_minus=jnp.zeros((n_com, 0)),
        nu_plus_ex=jnp.zeros((n_ex, 0)),
        nu_minus_ex=jnp.zeros((n_ex, 0)),
        mu_ex=jnp.zeros((n_ex,)),
        k_plus=jnp.zeros((0,)),
        k_minus=jnp.zeros((0,)),
        a=jnp.zeros((0,)),
    )


def build_reaction_system(
    reactions: list,
    n_com: int,
    n_ex: int = 0,
    mu_ex=None,
) -> ReactionSystem:
    """Assemble a ``ReactionSystem`` from a list of reaction dicts.

    Each reaction dict:
        {
            "reactants": {species: coef, ...},
            "products":  {species: coef, ...},
            "k_plus":  float,   # forward rate (mass action)
            "k_minus": float,   # backward rate (mass action)
            "a":       float,   # prefactor    (thermodynamic)
        }

    ``species`` may be an ``int`` (index of an internal species, 0..n_com-1)
    or a string ``"Y0"``, ``"Y1"``, ... denoting external (chemostatted)
    species, whose chemical potentials are supplied via ``mu_ex``.

    Missing rate constants default to 1.0 / 0.0 / 1.0.
    """
    n_rxn = len(reactions)
    nu_plus = np.zeros((n_com, n_rxn))
    nu_minus = np.zeros((n_com, n_rxn))
    nu_plus_ex = np.zeros((n_ex, n_rxn))
    nu_minus_ex = np.zeros((n_ex, n_rxn))
    k_plus = np.ones(n_rxn)
    k_minus = np.zeros(n_rxn)
    a = np.ones(n_rxn)

    def _place(sp, coef, internal_mat, external_mat, rho):
        if isinstance(sp, str):
            if not sp.startswith("Y"):
                raise ValueError(f"external species must be 'Y<int>', got {sp!r}")
            idx = int(sp[1:])
            if idx >= external_mat.shape[0]:
                raise IndexError(
                    f"external species index {idx} exceeds n_ex={external_mat.shape[0]}"
                )
            external_mat[idx, rho] = coef
        else:
            if sp < 0 or sp >= internal_mat.shape[0]:
                raise IndexError(f"internal species index {sp} out of range [0,{n_com})")
            internal_mat[sp, rho] = coef

    for rho, rxn in enumerate(reactions):
        for sp, coef in rxn.get("reactants", {}).items():
            _place(sp, coef, nu_plus, nu_plus_ex, rho)
        for sp, coef in rxn.get("products", {}).items():
            _place(sp, coef, nu_minus, nu_minus_ex, rho)
        if "k_plus" in rxn:
            k_plus[rho] = rxn["k_plus"]
        if "k_minus" in rxn:
            k_minus[rho] = rxn["k_minus"]
        if "a" in rxn:
            a[rho] = rxn["a"]

    if mu_ex is None:
        mu_ex = np.zeros(n_ex)
    mu_ex = np.asarray(mu_ex, dtype=np.float32)
    if mu_ex.shape != (n_ex,):
        raise ValueError(f"mu_ex must have shape ({n_ex},); got {mu_ex.shape}")

    return ReactionSystem(
        nu_plus=jnp.asarray(nu_plus),
        nu_minus=jnp.asarray(nu_minus),
        nu_plus_ex=jnp.asarray(nu_plus_ex),
        nu_minus_ex=jnp.asarray(nu_minus_ex),
        mu_ex=jnp.asarray(mu_ex),
        k_plus=jnp.asarray(k_plus),
        k_minus=jnp.asarray(k_minus),
        a=jnp.asarray(a),
    )


def linear_turnover_system(kon: float, koff: float, n_com: int) -> ReactionSystem:
    """Reproduce the legacy ``kon - koff * c`` turnover as a ReactionSystem.

    Builds 2 * n_com reactions:
      production:  Y0 -> X_i     with k_plus = kon
      degradation: X_i -> Y0     with k_plus = koff

    ``Y0`` is a single external buffer species with mu_ex = 0.
    """
    rxns = []
    for i in range(n_com):
        rxns.append({"reactants": {"Y0": 1}, "products":  {i: 1}, "k_plus": kon,  "k_minus": 0.0})
        rxns.append({"reactants": {i: 1},    "products":  {"Y0": 1}, "k_plus": koff, "k_minus": 0.0})
    return build_reaction_system(rxns, n_com=n_com, n_ex=1, mu_ex=jnp.zeros(1))


_SAFE = 1e-30


def apply_reactions(
    c: jnp.ndarray,
    rxn: ReactionSystem,
    rate_law: str = "mass_action",
    chi: Optional[jnp.ndarray] = None,
    chi_s: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute the per-component reaction source term on a grid.

    Parameters
    ----------
    c : array, shape (N_com, Nx, Ny)
    rxn : ReactionSystem
    rate_law : {"mass_action", "thermodynamic"}
    chi, chi_s : arrays, required only for ``"thermodynamic"`` rate law.
        ``chi_s`` must be shape ``(N_com,)`` (1-D, component-solvent).

    Returns
    -------
    source : array, shape (N_com, Nx, Ny)
    """
    n_com = c.shape[0]
    grid_shape = c.shape[1:]
    n_rxn = int(rxn.k_plus.shape[0])   # static

    if n_rxn == 0:
        return jnp.zeros_like(c)

    c_flat = c.reshape(n_com, -1)  # (N_com, Npx)

    if rate_law == "mass_action":
        log_c = jnp.log(jnp.maximum(c_flat, _SAFE))
        log_prod_plus  = jnp.einsum("iR,iP->RP", rxn.nu_plus,  log_c)
        log_prod_minus = jnp.einsum("iR,iP->RP", rxn.nu_minus, log_c)
        j = (rxn.k_plus[:, None]  * jnp.exp(log_prod_plus)
             - rxn.k_minus[:, None] * jnp.exp(log_prod_minus))
    elif rate_law == "thermodynamic":
        if chi is None or chi_s is None:
            raise ValueError("thermodynamic rate law requires `chi` and `chi_s`.")
        chi_s_1d = chi_s if chi_s.ndim == 1 else chi_s[:, 0]
        mu = chemical_potential_bulk(c, chi, chi_s_1d).reshape(n_com, -1)

        exp_plus  = jnp.einsum("iR,iP->RP", rxn.nu_plus,  mu)
        exp_minus = jnp.einsum("iR,iP->RP", rxn.nu_minus, mu)
        if rxn.mu_ex.size > 0:
            ex_plus  = rxn.nu_plus_ex.T  @ rxn.mu_ex
            ex_minus = rxn.nu_minus_ex.T @ rxn.mu_ex
            exp_plus  = exp_plus  + ex_plus[:, None]
            exp_minus = exp_minus + ex_minus[:, None]

        j = rxn.a[:, None] * (jnp.exp(exp_plus) - jnp.exp(exp_minus))
    else:
        raise ValueError(f"unknown rate_law={rate_law!r}")

    S = rxn.nu_minus - rxn.nu_plus
    source = (S @ j).reshape(n_com, *grid_shape)
    return source
