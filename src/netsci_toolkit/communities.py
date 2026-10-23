"""Community detection and partition comparison."""

import networkx as nx
from sklearn.metrics import normalized_mutual_info_score


def to_graph_tool(G):
    """Convert an undirected networkx graph to graph-tool.

    Requires graph-tool (install with conda: `conda install -c conda-forge graph-tool`).

    Returns
    -------
    g : graph_tool.Graph
    node_list : list
        node_list[i] is the original networkx node of graph-tool vertex i.
    """
    import graph_tool.all as gt

    node_list = list(G.nodes())
    index = {node: i for i, node in enumerate(node_list)}
    g = gt.Graph(directed=False)
    g.add_vertex(len(node_list))
    g.add_edge_list([(index[u], index[v]) for u, v in G.edges()])
    return g, node_list


def louvain_partition(G, seed=None):
    """Louvain modularity maximization.

    Returns
    -------
    partition : dict
        {node: community_id}, with id 0 for the largest community.
    Q : float
        Modularity of the partition.
    """
    communities = nx.community.louvain_communities(G, seed=seed)
    communities = sorted(communities, key=len, reverse=True)
    Q = nx.community.modularity(G, communities)
    partition = {node: i for i, c in enumerate(communities) for node in c}
    return partition, Q


def sbm_partition(G, nested=False, seed=None):
    """Fit a stochastic block model by minimizing the description length.

    Requires graph-tool.

    Parameters
    ----------
    G : networkx.Graph
    nested : bool
        If True, fit a nested (hierarchical) SBM and return its lowest level.
    seed : int, optional
        Seed for graph-tool's random number generator.

    Returns
    -------
    dict
        {node: block_id}
    """
    import graph_tool.all as gt

    g, node_list = to_graph_tool(G)
    if seed is not None:
        gt.seed_rng(seed)
    if nested:
        state = gt.minimize_nested_blockmodel_dl(g)
        blocks = state.get_levels()[0].get_blocks()
    else:
        state = gt.minimize_blockmodel_dl(g)
        blocks = state.get_blocks()
    return {node_list[i]: int(blocks[i]) for i in range(len(node_list))}


def compare_partitions(p1, p2):
    """Normalized mutual information between two partitions.

    Compares how nodes are grouped, not the labels: [0, 0, 1, 1] and
    [1, 1, 0, 0] give NMI = 1. Only nodes present in both partitions are used.

    Parameters
    ----------
    p1, p2 : dict
        Partitions as {node: community_id}.

    Returns
    -------
    float
        NMI in [0, 1].
    """
    nodes = [v for v in p1 if v in p2]
    return normalized_mutual_info_score([p1[v] for v in nodes], [p2[v] for v in nodes])
