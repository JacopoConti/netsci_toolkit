# netsci-toolkit

A small Python toolkit for network science, built from code I wrote in
**NETS/PHYS 7052** (Northeastern University, Fall 2026).

It covers a typical workflow in one place:

1. **Generate** a network (Barabási–Albert growth, power-law degree sequences, LFR benchmark)
2. **Describe** its degree distribution (log binning)
3. **Build a null model** (degree-preserving randomization)
4. **Detect communities** (Louvain, stochastic block model)
5. **Evaluate** the result (normalized mutual information between partitions)

All community functions return partitions in the same format, `{node: community_id}`,
so the output of one step plugs directly into the next.

---

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/JacopoConti/netsci_toolkit.git
cd netsci_toolkit
pip install -e .
```

Dependencies (installed automatically): `numpy`, `networkx`, `scikit-learn`.

**Optional: graph-tool.** `to_graph_tool` and `sbm_partition` need
[graph-tool](https://graph-tool.skewed.de/), which cannot be installed with pip.
Use conda:

```bash
conda install -c conda-forge graph-tool
```

Every other function works without it.

---

## Quick start

```python
import netsci_toolkit as nt

# Generate an LFR benchmark with planted communities
G, planted = nt.lfr_benchmark(n=1000, gamma=2.5, beta=1.5, avg_k=10,
                              k_max=50, mu=0.2, s_min=20, s_max=100, seed=42)

# Detect communities and compare them with the planted ones
found, Q = nt.louvain_partition(G, seed=42)
print(f"Q = {Q:.3f}, NMI = {nt.compare_partitions(planted, found):.3f}")

# Is the modularity meaningful? Compare with a degree-preserving null model
G_null = nt.degree_preserving_null(G, seed=42)
_, Q_null = nt.louvain_partition(G_null, seed=42)
print(f"Q (null model) = {Q_null:.3f}")
```

---

## Package structure

```
netsci-toolkit/
├── src/netsci_toolkit/
│   ├── __init__.py       # exposes all public functions
│   ├── generators.py     # grow_ba_network, powerlaw_degree_sequence, lfr_benchmark
│   ├── degrees.py        # log_binned_distribution
│   ├── nullmodels.py     # degree_preserving_null
│   └── communities.py    # to_graph_tool, louvain_partition, sbm_partition, compare_partitions
├── tests/test_basic.py
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Function reference

### Generators — `generators.py`

#### `grow_ba_network(N, m, seed=None)`
Grows a Barabási–Albert network by preferential attachment. It starts from a
complete graph of `m` nodes; each new node then links to `m` distinct existing
nodes. Targets are drawn uniformly from a list of all edge endpoints (the
"targets pool"): a node with degree k appears k times in that list, so it is
chosen with probability kᵢ / Σk.

| Parameter | Description |
|---|---|
| `N` | final number of nodes |
| `m` | links added by each new node |
| `seed` | random seed |

**Returns** a `networkx.Graph`. Its degree distribution approaches pₖ ~ k⁻³.

```python
G = nt.grow_ba_network(10_000, 4, seed=0)
```

#### `powerlaw_degree_sequence(n, gamma, avg_k, k_max=None, seed=None)`
Samples a degree sequence with P(k) ~ k^(−γ) (Step 1 of the LFR benchmark). The
values are rescaled to the target average degree, capped at `k_max`, and the sum
is made even so the sequence can be used in a configuration model.

| Parameter | Description |
|---|---|
| `n` | number of nodes |
| `gamma` | power-law exponent (> 1) |
| `avg_k` | target average degree ⟨k⟩ |
| `k_max` | maximum degree (default `n - 1`) |
| `seed` | seed or `numpy.random.Generator` |

**Returns** a NumPy array of integers.

```python
k = nt.powerlaw_degree_sequence(1000, gamma=2.5, avg_k=10, k_max=50, seed=1)
G = nx.configuration_model(k)
```

#### `lfr_benchmark(n, gamma, beta, avg_k, k_max, mu, s_min, s_max, seed=None)`
Generates a benchmark graph with planted communities, following
Lancichinetti, Fortunato & Radicchi (2008). It runs the five steps of the paper:

1. sample a power-law degree sequence (exponent γ);
2. split each degree into internal edges kᵢ(1 − µ) and external edges kᵢµ;
3. sample power-law community sizes (exponent β) that sum to n;
4. assign each node to a community large enough for its internal degree;
5. wire internal edges inside each community and external edges between communities.

| Parameter | Description |
|---|---|
| `n` | number of nodes |
| `gamma` | degree exponent |
| `beta` | community-size exponent |
| `avg_k` | average degree |
| `k_max` | maximum degree (must be smaller than `s_max`) |
| `mu` | mixing parameter: fraction of each node's edges leaving its community |
| `s_min`, `s_max` | minimum and maximum community size |
| `seed` | random seed |

**Returns** `(G, partition)`: the graph, with a `"community"` node attribute,
and the planted partition `{node: community_id}`.

Raises `ValueError` if `k_max >= s_max`, and `RuntimeError` if nodes cannot be
assigned to communities (try larger communities or a larger `mu`).

### Degree distributions — `degrees.py`

#### `log_binned_distribution(degrees, n_bins=20, k_min=None)`
Estimates pₖ with logarithmic bins, the standard way to plot heavy-tailed
distributions without noise in the tail. Bins are normalized by count and width,
and their centers are geometric means of the edges.

**Returns** `(centers, pk)`, with empty bins removed.

```python
import matplotlib.pyplot as plt
k, pk = nt.log_binned_distribution([d for _, d in G.degree()])
plt.loglog(k, pk, "o-")
```

### Null models — `nullmodels.py`

#### `degree_preserving_null(G, swaps_per_edge=10, seed=None)`
Randomizes a network with double edge swaps, (a–b, c–d) → (a–d, c–b), so that
every node keeps its degree. Anything that disappears after randomization
(community structure, clustering, degree correlations) was not explained by the
degree sequence alone. The input graph is not modified.

**Returns** a new `networkx.Graph`.

### Communities — `communities.py`

#### `to_graph_tool(G)`
Converts an undirected networkx graph to graph-tool, which needs vertices
labeled 0…n−1. *Requires graph-tool.*

**Returns** `(g, node_list)`, where `node_list[i]` is the original node of
graph-tool vertex `i`. Use it to map results back to the original IDs.

#### `louvain_partition(G, seed=None)`
Louvain modularity maximization (networkx implementation). Communities are
sorted by size, so id `0` is the largest.

**Returns** `(partition, Q)`: `{node: community_id}` and the modularity.

#### `sbm_partition(G, nested=False, seed=None)`
Fits a stochastic block model by minimizing the description length
(graph-tool). With `nested=True` it fits a hierarchical SBM and returns its
lowest level. Unlike modularity, this method does not tend to find communities
in random networks. *Requires graph-tool.*

**Returns** `{node: block_id}`.

#### `compare_partitions(p1, p2)`
Normalized mutual information (NMI) between two partitions. It compares how
nodes are grouped together, not their labels: `[0, 0, 1, 1]` and `[1, 1, 0, 0]`
give NMI = 1. Only nodes present in both partitions are compared.

**Returns** a float between 0 (independent) and 1 (identical).

```python
nt.compare_partitions(planted, found)
```

---

## Tests

```bash
pip install -e ".[test]"
pytest
```

## References

- Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509–512.
- Lancichinetti, A., Fortunato, S., & Radicchi, F. (2008). Benchmark graphs for testing community detection algorithms. *Physical Review E*, 78(4), 046110.
- Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks. *J. Stat. Mech.*, P10008.
- Peixoto, T. P. (2014). Hierarchical block structures and high-resolution model selection in large networks. *Physical Review X*, 4(1), 011047.
- Danon, L., et al. (2005). Comparing community structure identification. *J. Stat. Mech.*, P09008.

## License

MIT — see [LICENSE](LICENSE).
