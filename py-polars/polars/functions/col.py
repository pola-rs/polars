from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Iterable

from polars.datatypes import is_polars_dtype
from polars.utils._wrap import wrap_expr

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr


if TYPE_CHECKING:
    from polars import Expr
    from polars.type_aliases import PolarsDataType


def col(
    name: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType],
    *more_names: str | PolarsDataType,
) -> Expr:
    """
    Return an expression representing column(s) in a dataframe.

    Parameters
    ----------
    name
        The name or datatype of the column(s) to represent. Accepts regular expression
        input. Regular expressions should start with ``^`` and end with ``$``.
    *more_names
        Additional names or datatypes of columns to represent, specified as positional
        arguments.

    Examples
    --------
    Pass a single column name to represent that column.

    >>> df = pl.DataFrame(
    ...     {
    ...         "ham": [1, 2, 3],
    ...         "hamburger": [11, 22, 33],
    ...         "foo": [3, 2, 1],
    ...         "bar": ["a", "b", "c"],
    ...     }
    ... )
    >>> df.select(pl.col("foo"))
    shape: (3, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 3   │
    │ 2   │
    │ 1   │
    └─────┘

    Use the wildcard ``*`` to represent all columns.

    >>> df.select(pl.col("*"))
    shape: (3, 4)
    ┌─────┬───────────┬─────┬─────┐
    │ ham ┆ hamburger ┆ foo ┆ bar │
    │ --- ┆ ---       ┆ --- ┆ --- │
    │ i64 ┆ i64       ┆ i64 ┆ str │
    ╞═════╪═══════════╪═════╪═════╡
    │ 1   ┆ 11        ┆ 3   ┆ a   │
    │ 2   ┆ 22        ┆ 2   ┆ b   │
    │ 3   ┆ 33        ┆ 1   ┆ c   │
    └─────┴───────────┴─────┴─────┘
    >>> df.select(pl.col("*").exclude("ham"))
    shape: (3, 3)
    ┌───────────┬─────┬─────┐
    │ hamburger ┆ foo ┆ bar │
    │ ---       ┆ --- ┆ --- │
    │ i64       ┆ i64 ┆ str │
    ╞═══════════╪═════╪═════╡
    │ 11        ┆ 3   ┆ a   │
    │ 22        ┆ 2   ┆ b   │
    │ 33        ┆ 1   ┆ c   │
    └───────────┴─────┴─────┘

    Regular expression input is supported.

    >>> df.select(pl.col("^ham.*$"))
    shape: (3, 2)
    ┌─────┬───────────┐
    │ ham ┆ hamburger │
    │ --- ┆ ---       │
    │ i64 ┆ i64       │
    ╞═════╪═══════════╡
    │ 1   ┆ 11        │
    │ 2   ┆ 22        │
    │ 3   ┆ 33        │
    └─────┴───────────┘

    Multiple columns can be represented by passing a list of names.

    >>> df.select(pl.col(["hamburger", "foo"]))
    shape: (3, 2)
    ┌───────────┬─────┐
    │ hamburger ┆ foo │
    │ ---       ┆ --- │
    │ i64       ┆ i64 │
    ╞═══════════╪═════╡
    │ 11        ┆ 3   │
    │ 22        ┆ 2   │
    │ 33        ┆ 1   │
    └───────────┴─────┘

    Or use positional arguments to represent multiple columns in the same way.

    >>> df.select(pl.col("hamburger", "foo"))
    shape: (3, 2)
    ┌───────────┬─────┐
    │ hamburger ┆ foo │
    │ ---       ┆ --- │
    │ i64       ┆ i64 │
    ╞═══════════╪═════╡
    │ 11        ┆ 3   │
    │ 22        ┆ 2   │
    │ 33        ┆ 1   │
    └───────────┴─────┘

    Easily select all columns that match a certain data type by passing that datatype.

    >>> df.select(pl.col(pl.Utf8))
    shape: (3, 1)
    ┌─────┐
    │ bar │
    │ --- │
    │ str │
    ╞═════╡
    │ a   │
    │ b   │
    │ c   │
    └─────┘
    >>> df.select(pl.col(pl.Int64, pl.Float64))
    shape: (3, 3)
    ┌─────┬───────────┬─────┐
    │ ham ┆ hamburger ┆ foo │
    │ --- ┆ ---       ┆ --- │
    │ i64 ┆ i64       ┆ i64 │
    ╞═════╪═══════════╪═════╡
    │ 1   ┆ 11        ┆ 3   │
    │ 2   ┆ 22        ┆ 2   │
    │ 3   ┆ 33        ┆ 1   │
    └─────┴───────────┴─────┘

    """
    if more_names:
        if isinstance(name, str):
            names_str = [name]
            names_str.extend(more_names)  # type: ignore[arg-type]
            return wrap_expr(plr.cols(names_str))
        elif is_polars_dtype(name):
            dtypes = [name]
            dtypes.extend(more_names)
            return wrap_expr(plr.dtype_cols(dtypes))
        else:
            raise TypeError(
                "invalid input for `col`"
                f"\n\nExpected `str` or `DataType`, got {type(name).__name__!r}."
            )

    if isinstance(name, str):
        return wrap_expr(plr.col(name))
    elif is_polars_dtype(name):
        return wrap_expr(plr.dtype_cols([name]))
    elif isinstance(name, Iterable):
        names = list(name)
        if not names:
            return wrap_expr(plr.cols(names))

        item = names[0]
        if isinstance(item, str):
            return wrap_expr(plr.cols(names))
        elif is_polars_dtype(item):
            return wrap_expr(plr.dtype_cols(names))
        else:
            raise TypeError(
                "invalid input for `col`"
                "\n\nExpected iterable of type `str` or `DataType`,"
                f" got iterable of type {type(item).__name__!r}"
            )
    else:
        raise TypeError(
            "invalid input for `col`"
            f"\n\nExpected `str` or `DataType`, got {type(name).__name__!r}"
        )
