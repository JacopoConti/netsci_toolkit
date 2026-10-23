"""netsci_toolkit: a small network science toolkit from NETS/PHYS 7052."""

from .generators import grow_ba_network, powerlaw_degree_sequence, lfr_benchmark
from .degrees import log_binned_distribution
from .nullmodels import degree_preserving_null
from .communities import (
    to_graph_tool,
    louvain_partition,
    sbm_partition,
    compare_partitions,
)

__version__ = "0.1.0"

__all__ = [
    "grow_ba_network",
    "powerlaw_degree_sequence",
    "lfr_benchmark",
    "log_binned_distribution",
    "degree_preserving_null",
    "to_graph_tool",
    "louvain_partition",
    "sbm_partition",
    "compare_partitions",
]
