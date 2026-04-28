"""JAX-based differentiable phase-field simulation for multicomponent fluids.

A port of Shrinivas & Brenner (PNAS 2021) to JAX, enabling automatic
differentiation through the full simulation loop.
"""

from jax_phase_separation.free_energy import (
    free_energy_density,
    chemical_potential_bulk,
    compute_jacobian,
    stability_analysis,
)
from jax_phase_separation.solver import (
    SimulationParams,
    make_wavenumbers,
    simulate,
    simulate_with_snapshots,
)
from jax_phase_separation.analysis import (
    count_phases_pca,
    assign_phases_kmeans,
    compute_partition_ratios,
    compute_phase_angles,
    count_enriched_components,
)
from jax_phase_separation.theory import (
    n_phases_linear,
    n_phases_wigner,
    n_phases_turnover_linear,
    n_phases_turnover_wigner,
)
from jax_phase_separation.utils import (
    generate_chi_matrix,
    generate_initial_conditions,
)
