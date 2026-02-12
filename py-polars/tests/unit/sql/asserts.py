from __future__ import annotations

import contextlib
import sqlite3
from typing import TYPE_CHECKING, Any, Literal

import pytest

import polars as pl
from polars.datatypes.group import FLOAT_DTYPES, INTEGER_DTYPES
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from polars.type_aliases import PolarsDataType

_POLARS_TO_SQLITE_: dict[PolarsDataType, str] = {
    # SQLite has limited type support (primitive scalar types only)
    **dict.fromkeys(INTEGER_DTYPES, "INTEGER"),
    **dict.fromkeys(FLOAT_DTYPES, "FLOAT"),
    pl.Boolean: "INTEGER",
    pl.String: "TEXT",
}


def _execute_with_sqlite(
    frames: dict[str, pl.DataFrame | pl.LazyFrame],
    query: str,
) -> pl.DataFrame:
    """Execute a SQL query against SQLite, returning a DataFrame."""
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        cursor = conn.cursor()
        for name, df in frames.items():
            if isinstance(df, pl.LazyFrame):
                df = df.collect()

            frame_schema = df.schema
            types = (_POLARS_TO_SQLITE_[frame_schema[col]] for col in df.columns)
            schema = ", ".join(
                f"{col} {tp}" for col, tp in zip(df.columns, types, strict=True)
            )
            cursor.execute(f"CREATE TABLE {name} ({schema})")
            cursor.executemany(
                f"INSERT INTO {name} VALUES ({','.join(['?'] * len(df.columns))})",
                df.iter_rows(),
            )

        conn.commit()
        cursor.execute(query)

        return pl.DataFrame(
            cursor.fetchall(),
            schema=[desc[0] for desc in cursor.description],
            orient="row",
        )


def _execute_with_duckdb(
    frames: dict[str, pl.DataFrame | pl.LazyFrame],
    query: str,
) -> pl.DataFrame:
    """Execute a SQL query against DuckDB, returning a DataFrame."""
    try:
        import duckdb
    except ImportError:
        # if not available locally, skip (will always be run on CI)
        pytest.skip(
            """DuckDB not installed; required for `assert_sql_matches` with "compare_with='duckdb'"."""
        )
    with duckdb.connect(":memory:") as conn:
        for name, df in frames.items():
            conn.register(name, df)
        return conn.execute(query).pl()  # type: ignore[no-any-return]


_COMPARISON_BACKENDS_ = {
    "sqlite": _execute_with_sqlite,
    "duckdb": _execute_with_duckdb,
}


def assert_sql_matches(
    frames: pl.DataFrame | pl.LazyFrame | dict[str, pl.DataFrame | pl.LazyFrame],
    *,
    query: str,
    compare_with: Literal["sqlite", "duckdb"] | Collection[Literal["sqlite", "duckdb"]],
    check_dtypes: bool = False,
    check_row_order: bool = True,
    check_column_names: bool = True,
    expected: pl.DataFrame | dict[str, Sequence[Any]] | None = None,
) -> bool:
    """
    Assert that a Polars SQL query produces the same result as a reference backend.

    This function executes the provided SQL query using both Polars and a reference
    SQL engine (eg: SQLite or DuckDB), then asserts that the results match.

    Parameters
    ----------
    frames
        Mapping of table names to DataFrame or LazyFrame; the query should reference
        the table names as they appear in the dict keys. If passed a single frame,
        "self" is assumed to be the name of the referenced table/frame.
    query
        SQL query string to test, referencing table names from `frames`.
    compare_with
        One or more named SQL engines to use as a reference for comparison.
        - 'sqlite': Use Python's built-in `sqlite3` module.
        - 'duckdb': Use DuckDB (requires `duckdb` to be installed separately).
    check_dtypes
        Require that the comparison frame dtypes match; defaults to False, as different
        backends may use different type systems, and we care about the values.
    check_row_order
        Set False to ignore the row order in the Polars/comparison frame match.
    check_column_names
        Set False to ignore the column names in the Polars/comparison frame match
        (but still compare each column in the same expected order).
    expected
        An optional DataFrame (or dictionary) containing the expected result;
        with this we can confirm both that the result matches the reference
        implementation *and* that those results match expectation.

    Examples
    --------
    >>> import polars as pl
    >>> from tests.unit.sql import assert_sql_matches

    Confirm that a given SQL query against a single frame returns the same
    result values when executed with Polars and executed with SQLite:

    >>> lf = pl.LazyFrame({"lbl": ["xx", "yy", "zz"], "value": [-150, 325, 275]})
    >>> query = "SELECT lbl, value * 2 AS doubled FROM demo WHERE id > 1 ORDER BY lbl"
    >>> assert_sql_matches({"demo": lf}, query=query, compare_with="sqlite")

    Check that a multi-frame JOIN produces the same result as DuckDB:

    >>> users = pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    >>> orders = pl.DataFrame({"user_id": [1, 1, 2], "amount": [100, 200, 150]})
    >>> assert_sql_matches(
    ...     frames={"users": users, "orders": orders},
    ...     query='''
    ...         SELECT u.name, SUM(o.amount) as total
    ...         FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name
    ...     ''',
    ...     compare_with="duckdb",
    ...     check_row_order=False,
    ... )
    """
    if isinstance(frames, (pl.DataFrame, pl.LazyFrame)):
        frames = {"self": frames}

    with pl.SQLContext(frames=frames, eager=True) as ctx:
        polars_result = ctx.execute(query=query, eager=True)

    if isinstance(compare_with, str):
        compare_with = [compare_with]

    for comparison_backend in compare_with:
        if (exec_comparison := _COMPARISON_BACKENDS_.get(comparison_backend)) is None:
            valid_engines = ", ".join(repr(b) for b in sorted(_COMPARISON_BACKENDS_))
            msg = (
                f"invalid `compare_with` value: {comparison_backend!r}; "
                f"expected one of {valid_engines}"
            )
            raise ValueError(msg)

        comparison_result = exec_comparison(frames, query)
        if not check_column_names:
            n_comparison_cols = comparison_result.width
            comparison_result.columns = polars_result.columns[:n_comparison_cols]

        # validate against the reference engine/backend
        assert_frame_equal(
            polars_result,
            comparison_result,
            check_dtypes=check_dtypes,
            check_row_order=check_row_order,
        )

    # confirm that these values are not just consistent
    # but also match a specific/expected result
    if expected is not None:
        if isinstance(expected, dict):
            expected = pl.from_dict(
                data=expected,
                schema=polars_result.schema,
            )

        assert_frame_equal(
            polars_result,
            expected,
            check_dtypes=check_dtypes,
            check_row_order=check_row_order,
        )

    return True


__all__ = ["assert_sql_matches"]
