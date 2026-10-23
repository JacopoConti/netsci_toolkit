"""Tools for describing degree distributions."""

import numpy as np


def log_binned_distribution(degrees, n_bins=20, k_min=None):
    """Log-binned degree distribution p_k.

    Bins are equally spaced in log scale. Each bin is normalized by the total
    count and its width (density=True), so the values estimate p_k.

    Parameters
    ----------
    degrees : array-like of int
        Degree sequence.
    n_bins : int
        Number of bin edges (n_bins - 1 bins).
    k_min : int, optional
        Smallest degree to include. Defaults to the smallest positive degree.

    Returns
    -------
    centers : numpy.ndarray
        Geometric bin centers.
    pk : numpy.ndarray
        Estimated p_k. Empty bins are dropped.
    """
    degrees = np.asarray(degrees)
    degrees = degrees[degrees > 0]
    k_min = degrees.min() if k_min is None else k_min
    bins = np.logspace(np.log10(k_min), np.log10(degrees.max()), num=n_bins)

    pk, edges = np.histogram(degrees, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = pk > 0
    return centers[mask], pk[mask]
