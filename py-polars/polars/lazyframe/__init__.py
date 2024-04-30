from polars.lazyframe.frame import LazyFrame, _expr_nodes, _ir_nodes
from polars.lazyframe.in_process import InProcessQuery

__all__ = ["LazyFrame", "InProcessQuery", "_ir_nodes", "_expr_nodes"]
