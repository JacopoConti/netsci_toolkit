"""Null models for networks."""

import networkx as nx


def degree_preserving_null(G, swaps_per_edge=10, seed=None):
    """Randomize a network while keeping every node's degree.

    Uses double edge swaps: (a-b, c-d) -> (a-d, c-b).

    Parameters
    ----------
    G : networkx.Graph
        Undirected input graph (not modified).
    swaps_per_edge : int
        Number of swaps per edge. 10 is usually enough to randomize.
    seed : int, optional
        Random seed.

    Returns
    -------
    networkx.Graph
    """
    G_rand = G.copy()
    n_swaps = swaps_per_edge * G_rand.number_of_edges()
    nx.double_edge_swap(G_rand, nswap=n_swaps, max_tries=n_swaps * 100, seed=seed)
    return G_rand
