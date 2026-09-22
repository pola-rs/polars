from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from polars._utils.construction.dataframe import dataframe_to_pydf
from polars._utils.wrap import wrap_df

with contextlib.suppress(ImportError):
    from polars._plr import PySeries

if TYPE_CHECKING:
    from polars import DataFrame
    from polars._typing import SchemaDefinition, SchemaDict


def is_pycapsule(obj: Any) -> bool:
    """Check if object looks like it supports the PyCapsule interface."""
    return any(
        callable(getattr(obj, attr, None))
        for attr in ("__arrow_c_stream__", "__arrow_c_array__")
    )


def pycapsule_to_frame(
    obj: Any,
    *,
    schema: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
    rechunk: bool = False,
) -> DataFrame:
    """Convert PyCapsule object to DataFrame."""
    if hasattr(obj, "__arrow_c_array__"):
        # This uses the fact that PySeries.from_arrow_c_array will create a
        # struct-typed Series. Then we unpack that to a DataFrame.
        # `struct_unnest` is a single Rust call that releases the GIL; going via
        # `DataFrame.unnest` would run the whole lazy engine while holding it,
        # which serializes construction across Python threads.
        df = wrap_df(PySeries.from_arrow_c_array(obj).struct_unnest())

    elif hasattr(obj, "__arrow_c_stream__"):
        # This uses the fact that PySeries.from_arrow_c_stream will create a
        # struct-typed Series. Then we unpack that to a DataFrame.
        df = wrap_df(PySeries.from_arrow_c_stream(obj).struct_unnest())
    else:
        msg = f"object does not support PyCapsule interface; found {obj!r} "
        raise TypeError(msg)

    if rechunk:
        df = df.rechunk()
    if schema or schema_overrides:
        df = wrap_df(
            dataframe_to_pydf(df, schema=schema, schema_overrides=schema_overrides)
        )
    return df
