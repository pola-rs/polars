from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from polars._utils.construction.dataframe import dataframe_to_pydf
from polars._utils.wrap import wrap_df, wrap_s
from polars.datatypes import Struct

with contextlib.suppress(ImportError):
    from polars._plr import PySeries

if TYPE_CHECKING:
    from polars import DataFrame
    from polars._typing import SchemaDefinition, SchemaDict


def is_pycapsule(obj: Any) -> bool:
    """Check if object supports the PyCapsule interface."""
    return hasattr(obj, "__arrow_c_stream__") or hasattr(obj, "__arrow_c_array__")


def pycapsule_to_frame(
    obj: Any,
    *,
    schema: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
    rechunk: bool = False,
) -> DataFrame:
    """Convert PyCapsule object to DataFrame."""
    if hasattr(obj, "__arrow_c_array__"):
        tmp_col_name = ""
        s = wrap_s(PySeries.from_arrow_c_array(obj))
        # Check if it's actually a struct (can be unnested) or just a regular series
        if isinstance(s.dtype, Struct):
            # Original behavior for struct types (e.g., RecordBatch)
            df = s.to_frame(tmp_col_name).unnest(tmp_col_name)
        else:
            # Direct conversion for regular series types (e.g., Categorical, Int64)
            df = s.to_frame()

    elif hasattr(obj, "__arrow_c_stream__"):
        tmp_col_name = ""
        s = wrap_s(PySeries.from_arrow_c_stream(obj))
        # Check if it's actually a struct (can be unnested) or just a regular series
        if isinstance(s.dtype, Struct):
            # Original behavior for struct types (e.g., RecordBatch)
            df = s.to_frame(tmp_col_name).unnest(tmp_col_name)
        else:
            # Direct conversion for regular series types (e.g., Categorical, Int64)
            df = s.to_frame()
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
