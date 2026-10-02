# Type stub of the `polars_iceberg` extension module (`src/python.rs`).
from typing import Any

__version__: str
_polars_io_plugin_ids: tuple[str, ...]

def _capsule(id: str) -> Any: ...
def _capsule_for_testing(name: str) -> Any: ...
