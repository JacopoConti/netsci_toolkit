import networkx as nx
import netsci_toolkit as nt


def test_ba_network():
    G = nt.grow_ba_network(500, 3, seed=1)
    assert G.number_of_nodes() == 500
    assert min(d for _, d in G.degree()) >= 3


def test_degree_sequence_even():
    k = nt.powerlaw_degree_sequence(1000, 2.5, 10, k_max=50, seed=1)
    assert k.sum() % 2 == 0 and k.max() <= 50


def test_null_keeps_degrees():
    G = nt.grow_ba_network(300, 2, seed=1)
    assert dict(G.degree()) == dict(nt.degree_preserving_null(G, seed=1).degree())


def test_compare_partitions_ignores_labels():
    assert nt.compare_partitions({0: 0, 1: 0, 2: 1, 3: 1}, {0: 1, 1: 1, 2: 0, 3: 0}) == 1.0


def test_lfr_recoverable():
    G, planted = nt.lfr_benchmark(1000, 2.5, 1.5, 10, 50, 0.2, 20, 100, seed=42)
    found, _ = nt.louvain_partition(G, seed=42)
    assert nt.compare_partitions(planted, found) > 0.8
