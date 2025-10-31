from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from polars._utils.construction.dataframe import dataframe_to_pydf
from polars._utils.wrap import wrap_df, wrap_s
from polars.datatypes import Struct
from polars.exceptions import SchemaError

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
        s = wrap_s(PySeries.from_arrow_c_array(obj))
    elif hasattr(obj, "__arrow_c_stream__"):
        s = wrap_s(PySeries.from_arrow_c_stream(obj))
    else:
        msg = f"object does not support PyCapsule interface; found {obj!r} "
        raise TypeError(msg)

    if isinstance(s.dtype, Struct):
        tmp_col_name = ""
        df = s.to_frame(tmp_col_name).unnest(tmp_col_name)
    else:
        msg = (
            f"Cannot create DataFrame from single column data (got {s.dtype}). "
            f"Use series.to_frame('column_name') or pl.DataFrame({{'col': series}}) instead."
        )
        raise SchemaError(msg)

    if rechunk:
        df = df.rechunk()
    if schema or schema_overrides:
        df = wrap_df(
            dataframe_to_pydf(df, schema=schema, schema_overrides=schema_overrides)
        )
    return df
