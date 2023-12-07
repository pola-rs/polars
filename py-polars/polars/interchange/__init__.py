"""
Module containing the implementation of the Python dataframe interchange protocol.

Details on the protocol:
https://data-apis.org/dataframe-protocol/latest/index.html
"""
from polars.interchange.from_dataframe_ import from_dataframe

__all__ = ["from_dataframe"]
