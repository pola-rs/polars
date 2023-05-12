from __future__ import annotations

from typing import TYPE_CHECKING

from polars.datatypes import Utf8

if TYPE_CHECKING:
    from polars import DataFrame, Series


def _to_rust_syntax(df: DataFrame) -> str:
    """Utility to generate the syntax that creates a polars 'DataFrame' in Rust."""
    syntax = "df![\n"

    def format_s(s: Series) -> str:
        if s.null_count() == 0:
            out = str(s.to_list()).replace("'", '"')
            if s.dtype != Utf8:
                out = out.lower()
            return out
        else:
            tmp = "["
            for val in s:
                if val is None:
                    tmp += "None, "
                else:
                    if isinstance(val, str):
                        tmp += f'Some("{val}"), '
                    else:
                        val = str(val).lower()
                        tmp += f"Some({val}), "
            tmp = tmp[:-2] + "]"
            return tmp

    for s in df:
        syntax += f'    "{s.name}" => {format_s(s)},\n'
    syntax += "]"
    return syntax
