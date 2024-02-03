"""Public functions that provide information about the Polars package or the environment it runs in."""  # noqa: W505
from polars.meta.build_info import build_info
from polars.meta.get_index_type import get_index_type
from polars.meta.show_versions import show_versions
from polars.meta.threadpool_size import threadpool_size

__all__ = [
    "build_info",
    "get_index_type",
    "show_versions",
    "threadpool_size",
]
