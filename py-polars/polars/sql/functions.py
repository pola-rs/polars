from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from polars.dataframe import DataFrame
    from polars.lazyframe import LazyFrame


@overload
def sql(query: str, *, eager: Literal[False] = False) -> LazyFrame: ...


@overload
def sql(query: str, *, eager: Literal[True]) -> DataFrame: ...


def sql(query: str, *, eager: bool = False) -> DataFrame | LazyFrame:
    """
    Execute a SQL query against frames in the global namespace.

    .. versionadded:: 0.20.31

    .. warning::
        This functionality is considered **unstable**, although it is close to
        being considered stable. It may be changed at any point without it being
        considered a breaking change.

    Parameters
    ----------
    query
        SQL query to execute.
    eager
        Automatically collect the result and return a DataFrame instead of a LazyFrame.

    Notes
    -----
    * More control over registration and execution behaviour is available by
      using the :class:`SQLContext` object.

    See Also
    --------
    SQLContext

    Examples
    --------
    >>> lf1 = pl.LazyFrame({"a": [1, 2, 3], "b": [6, 7, 8], "c": ["z", "y", "x"]})
    >>> lf2 = pl.LazyFrame({"a": [3, 2, 1], "d": [125, -654, 888]})

    Query the LazyFrame using SQL:

    >>> lf1.sql("SELECT c, b FROM self WHERE a > 1").collect()
    shape: (2, 2)
    ┌─────┬─────┐
    │ c   ┆ b   │
    │ --- ┆ --- │
    │ str ┆ i64 │
    ╞═════╪═════╡
    │ y   ┆ 7   │
    │ x   ┆ 8   │
    └─────┴─────┘

    Join two LazyFrames:

    >>> pl.sql(
    ...     '''
    ...     SELECT lf1.*, d
    ...     FROM lf1
    ...     INNER JOIN lf2 USING (a)
    ...     WHERE a > 1 AND b < 8
    ...     '''
    ... ).collect()
    shape: (1, 4)
    ┌─────┬─────┬─────┬──────┐
    │ a   ┆ b   ┆ c   ┆ d    │
    │ --- ┆ --- ┆ --- ┆ ---  │
    │ i64 ┆ i64 ┆ str ┆ i64  │
    ╞═════╪═════╪═════╪══════╡
    │ 2   ┆ 7   ┆ y   ┆ -654 │
    └─────┴─────┴─────┴──────┘

    Apply SQL transforms (aliasing "self" to "frame") and subsequently
    filter natively (you can freely mix SQL and native operations):

    >>> pl.sql(
    ...     query='''
    ...         SELECT
    ...             a,
    ...             (a % 2 == 0) AS a_is_even,
    ...             (b::float4 / 2) AS "b/2",
    ...             CONCAT_WS(':', c, c, c) AS c_c_c
    ...         FROM lf1
    ...         ORDER BY a
    ...     ''',
    ... ).filter(~pl.col("c_c_c").str.starts_with("x")).collect()
    shape: (2, 4)
    ┌─────┬───────────┬─────┬───────┐
    │ a   ┆ a_is_even ┆ b/2 ┆ c_c_c │
    │ --- ┆ ---       ┆ --- ┆ ---   │
    │ i64 ┆ bool      ┆ f32 ┆ str   │
    ╞═════╪═══════════╪═════╪═══════╡
    │ 1   ┆ false     ┆ 3.0 ┆ z:z:z │
    │ 2   ┆ true      ┆ 3.5 ┆ y:y:y │
    └─────┴───────────┴─────┴───────┘
    """
    from polars.sql import SQLContext

    with SQLContext(eager=eager, register_globals=True) as ctx:
        return ctx.execute(query)


__all__ = ["sql"]
