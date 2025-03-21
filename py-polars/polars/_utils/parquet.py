from typing import TYPE_CHECKING, Any, Callable

from polars._typing import ParquetMetadataFn

if TYPE_CHECKING:
    from polars._typing import ParquetMetadataContext


def wrap_parquet_metadata_callback(
    fn: ParquetMetadataFn,
) -> Callable[[Any], list[tuple[str, str]]]:
    def pyo3_compatible_callback(ctx: Any) -> list[tuple[str, str]]:
        ctx_py: ParquetMetadataContext = {
            "key_value_metadata": dict(ctx.key_value_metadata),
            "info": dict(ctx.info),
        }
        return list(fn(ctx_py).items())

    return pyo3_compatible_callback
