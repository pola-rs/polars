"""Data generation functionality for use in the benchmarking suite."""

from tests.benchmark.data.h2oai import generate_group_by_data
from tests.benchmark.data.pdsh import load_pdsh_table

__all__ = [
    "generate_group_by_data",
    "load_pdsh_table",
]
