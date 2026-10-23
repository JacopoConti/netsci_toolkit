"""Network generators: Barabási-Albert growth, power-law degrees, LFR benchmark."""

import random

import networkx as nx
import numpy as np


def grow_ba_network(N, m, seed=None):
    """Grow a Barabási-Albert network by preferential attachment.

    Starts from a complete graph of m nodes. Each new node connects to m
    distinct existing nodes, chosen with probability proportional to degree.
    Picking uniformly from the list of all edge endpoints (the "targets pool")
    is equivalent to picking node i with probability k_i / sum(k).

    Parameters
    ----------
    N : int
        Final number of nodes.
    m : int
        Number of links added by each new node (also the size of the seed graph).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    networkx.Graph
    """
    rng = random.Random(seed)
    edges = [(i, j) for i in range(m) for j in range(i + 1, m)]
    targets_pool = [node for edge in edges for node in edge]

    for new_node in range(m, N):
        chosen = set()
        while len(chosen) < m:
            chosen.add(rng.choice(targets_pool))
        for target in chosen:
            edges.append((new_node, target))
            targets_pool.extend([new_node, target])

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    return G


def powerlaw_degree_sequence(n, gamma, avg_k, k_max=None, seed=None):
    """Sample a power-law degree sequence P(k) ~ k^(-gamma).

    The sequence is rescaled to the target average degree, capped at k_max,
    and its sum is made even so it can be used in a configuration model.

    Parameters
    ----------
    n : int
        Number of nodes.
    gamma : float
        Power-law exponent (gamma > 1).
    avg_k : float
        Target average degree.
    k_max : int, optional
        Maximum degree. Defaults to n - 1.
    seed : int or numpy.random.Generator, optional
        Random seed or generator.

    Returns
    -------
    numpy.ndarray of int
    """
    rng = np.random.default_rng(seed)
    k_max = n - 1 if k_max is None else k_max

    k = rng.pareto(gamma - 1, n) + 1
    for _ in range(20):  # rescale and cap until the mean settles
        k = np.clip(k * avg_k / k.mean(), 1, k_max)
    k = np.round(k).astype(int)

    if k.sum() % 2 == 1:  # configuration model needs an even sum
        i = np.flatnonzero(k < k_max)[0]
        k[i] += 1
    return k


# --- LFR helpers (internal) -------------------------------------------------

def _split_degrees(k, mu):
    """Step 2: split each degree into internal and external parts."""
    k_in = np.round(k * (1 - mu)).astype(int)
    return k_in, k - k_in


def _community_sizes(n, beta, s_min, s_max, rng):
    """Step 3: power-law community sizes that sum to n."""
    a = 1 - beta
    sizes = []
    while sum(sizes) < n:
        u = rng.random()
        s = (s_min**a + u * (s_max**a - s_min**a)) ** (1 / a)
        sizes.append(int(round(s)))
    sizes[-1] -= sum(sizes) - n  # trim the last one so the total is n
    if sizes[-1] < s_min and len(sizes) > 1:  # merge it if it became too small
        last = sizes.pop()
        sizes[-1] += last
    return sizes


def _assign_communities(k_in, sizes, rng, max_iter=100_000):
    """Step 4: assign nodes to communities large enough for their k_in."""
    members = [[] for _ in sizes]
    homeless = list(range(len(k_in)))
    for _ in range(max_iter):
        if not homeless:
            return members
        node = homeless.pop(rng.integers(len(homeless)))
        c = rng.integers(len(sizes))
        if k_in[node] < sizes[c]:
            members[c].append(node)
            if len(members[c]) > sizes[c]:  # community full: kick someone out
                homeless.append(members[c].pop(rng.integers(len(members[c]))))
        else:
            homeless.append(node)
    raise RuntimeError("Could not assign all nodes; check s_min, s_max and mu.")


def _pair_stubs(stubs, rng):
    """Pair stubs at random (configuration model)."""
    stubs = list(stubs)
    rng.shuffle(stubs)
    return list(zip(stubs[0::2], stubs[1::2]))


def lfr_benchmark(n, gamma, beta, avg_k, k_max, mu, s_min, s_max, seed=None):
    """Generate an LFR benchmark graph (Lancichinetti, Fortunato & Radicchi, 2008).

    Parameters
    ----------
    n : int
        Number of nodes.
    gamma : float
        Degree distribution exponent.
    beta : float
        Community size distribution exponent.
    avg_k : float
        Target average degree.
    k_max : int
        Maximum degree. Must be smaller than s_max.
    mu : float
        Mixing parameter: fraction of each node's edges that leave its community.
    s_min, s_max : int
        Minimum and maximum community size.
    seed : int, optional
        Random seed.

    Returns
    -------
    G : networkx.Graph
        Graph with a "community" node attribute.
    partition : dict
        Planted partition {node: community_id}.
    """
    if k_max >= s_max:
        raise ValueError("s_max must be larger than k_max (see Step 3 of the paper).")
    rng = np.random.default_rng(seed)

    k = powerlaw_degree_sequence(n, gamma, avg_k, k_max, seed=rng)
    k_in, k_out = _split_degrees(k, mu)
    sizes = _community_sizes(n, beta, s_min, s_max, rng)
    while max(sizes) <= k_in.max():  # the biggest hub needs a big enough community
        sizes = _community_sizes(n, beta, s_min, s_max, rng)
    members = _assign_communities(k_in, sizes, rng)

    partition = {node: c for c, nodes in enumerate(members) for node in nodes}
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Internal edges: configuration model inside each community
    for nodes in members:
        stubs = [v for v in nodes for _ in range(k_in[v])]
        G.add_edges_from(_pair_stubs(stubs, rng))

    # External edges: configuration model on the whole graph,
    # re-shuffling stubs that land inside the same community
    stubs = [v for v in range(n) for _ in range(k_out[v])]
    for _ in range(20):
        bad = []
        for u, v in _pair_stubs(stubs, rng):
            if partition[u] != partition[v]:
                G.add_edge(u, v)
            else:
                bad += [u, v]
        stubs = bad
        if len(stubs) < 2:
            break

    G.remove_edges_from(nx.selfloop_edges(G))
    nx.set_node_attributes(G, partition, "community")
    return G, partition
