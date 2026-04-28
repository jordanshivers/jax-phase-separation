"""Post-processing analysis for simulated composition fields."""

import numpy as np
from sklearn.cluster import KMeans


def compute_gradient_magnitude(c):
    """Compute the maximum squared gradient across components.

    Parameters
    ----------
    c : array (N_com, Nx, Ny)

    Returns
    -------
    gradient : array (Nx, Ny) -- max over components of |grad phi|^2
    """
    c = np.asarray(c)
    dcdx = np.zeros_like(c)
    dcdy = np.zeros_like(c)
    for n in range(c.shape[0]):
        for i in range(c.shape[1]):
            dcdx[n, i, :] = np.gradient(c[n, i, :])
            dcdy[n, :, i] = np.gradient(c[n, :, i])
    return np.amax(dcdx ** 2 + dcdy ** 2, axis=0)


def count_phases_pca(c, filter_thres=10.0, phase_thresh=9e-3, centered=True):
    """Count the number of distinct phases using PCA on the composition field.

    Parameters
    ----------
    c : array (N_com, Nx, Ny)
        Concentration field (solute components only, or including solvent).
    filter_thres : float
        Gradient-filter threshold factor; points with gradient magnitude
        above max/filter_thres are excluded as interfaces.
    phase_thresh : float
        Eigenvalue threshold for counting significant PCA modes.
    centered : bool
        Whether to centre and normalise before PCA.

    Returns
    -------
    n_phases : int
    eig_vals : array
    eig_vecs : array
    """
    c = np.asarray(c)
    gradient = compute_gradient_magnitude(c)
    cre = c.reshape(c.shape[0], -1)
    mask = gradient.flatten() < gradient.max() / filter_thres + 1e-5
    cre = cre[:, mask].T  # (n_points, N_com)

    if centered:
        mean_vec = np.mean(cre, axis=0)
        std_vec = np.std(cre, axis=0)
        std_vec = np.where(std_vec < 1e-12, 1.0, std_vec)
        q_cent = (cre - mean_vec) / std_vec
    else:
        q_cent = cre

    cov_mat = q_cent.T @ q_cent / (q_cent.shape[0] - 1)
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)

    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    n_phases = int(np.sum(eig_vals > phase_thresh))
    return n_phases, eig_vals, eig_vecs


def assign_phases_kmeans(c, n_phases, random_state=0):
    """Assign each spatial point to one of *n_phases* using K-means.

    Parameters
    ----------
    c : array (N_com, Nx, Ny)
    n_phases : int

    Returns
    -------
    labels : array (Nx, Ny) of int
    centres : array (n_phases, N_com) -- bulk composition of each phase
    """
    c = np.asarray(c)
    N_com, Nx, Ny = c.shape
    X = c.reshape(N_com, -1).T  # (Nx*Ny, N_com)
    n_phases = max(1, int(n_phases))
    km = KMeans(n_clusters=n_phases, random_state=random_state, n_init=10).fit(X)
    labels = km.labels_.reshape(Nx, Ny)

    order = np.argsort(km.cluster_centers_[:, 0])
    remap = np.zeros(n_phases, dtype=int)
    for new, old in enumerate(order):
        remap[old] = new
    labels = remap[labels]
    centres = km.cluster_centers_[order]

    return labels, centres


def compute_partition_ratios(c, labels, n_phases):
    """Compute partition ratio p_i^gamma = <phi_i>_gamma / phi_i^0 for each
    phase gamma and component i.

    Parameters
    ----------
    c : array (N_com, Nx, Ny)
    labels : array (Nx, Ny) int
    n_phases : int

    Returns
    -------
    partitions : array (n_phases, N_com)
    """
    c = np.asarray(c)
    N_com = c.shape[0]
    phi0 = c.mean(axis=(1, 2))  # overall average per component
    partitions = np.zeros((n_phases, N_com))
    for gamma in range(n_phases):
        mask = labels == gamma
        if mask.sum() == 0:
            continue
        for i in range(N_com):
            mean_in_phase = c[i][mask].mean()
            partitions[gamma, i] = mean_in_phase / max(phi0[i], 1e-30)
    return partitions


def compute_phase_angles(centres):
    """Compute pairwise angles (degrees) between phase composition vectors.

    Parameters
    ----------
    centres : array (n_phases, N_com)

    Returns
    -------
    angles : list of floats (all unique pairs)
    """
    n = centres.shape[0]
    angles = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = centres[i], centres[j]
            cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_theta)))
    return angles


def count_enriched_components(partitions, epsilon=0.2):
    """Count the number of enriched components per phase.

    Parameters
    ----------
    partitions : array (n_phases, N_com)
    epsilon : float

    Returns
    -------
    n_enriched : array (n_phases,) of int
    """
    return (partitions > 1.0 + epsilon).sum(axis=1)


def analyse_snapshot(c, filter_thres=10.0, phase_thresh=9e-3, epsilon=0.2):
    """Run the full analysis pipeline on a concentration snapshot.

    Parameters
    ----------
    c : array (N_com, Nx, Ny) -- solute components only

    Returns
    -------
    result : dict with keys 'n_phases', 'labels', 'centres', 'partitions',
             'angles', 'n_enriched', 'eig_vals'
    """
    c = np.asarray(c, dtype=float)
    # Downstream PCA and KMeans require finite values.
    if not np.isfinite(c).all():
        c = np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0)
    c_with_solvent = np.concatenate(
        [c, (1.0 - np.sum(c, axis=0, keepdims=True))],
        axis=0,
    )
    n_phases, eig_vals, _ = count_phases_pca(
        c_with_solvent, filter_thres=filter_thres,
        phase_thresh=phase_thresh, centered=False,
    )
    n_phases = max(n_phases, 1)
    labels, centres = assign_phases_kmeans(c_with_solvent, n_phases)
    partitions = compute_partition_ratios(c_with_solvent, labels, n_phases)
    angles = compute_phase_angles(centres)
    n_enriched = count_enriched_components(partitions, epsilon=epsilon)
    return dict(
        n_phases=n_phases,
        labels=labels,
        centres=centres,
        partitions=partitions,
        angles=angles,
        n_enriched=n_enriched,
        eig_vals=eig_vals,
    )
