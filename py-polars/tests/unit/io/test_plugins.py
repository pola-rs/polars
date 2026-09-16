from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Iterator

    from polars._typing import EngineType


# A simple python source. But this can dispatch into a rust IO source as well.
def my_source(
    with_columns: list[str] | None,
    predicate: pl.Expr | None,
    _n_rows: int | None,
    _batch_size: int | None,
) -> Iterator[pl.DataFrame]:
    for i in [1, 2, 3]:
        df = pl.DataFrame({"a": [i], "b": [i]})

        if predicate is not None:
            df = df.filter(predicate)

        if with_columns is not None:
            df = df.select(with_columns)

        yield df


def scan_my_source() -> pl.LazyFrame:
    # schema inference logic
    # TODO: make lazy via callable
    schema = pl.Schema({"a": pl.Int64(), "b": pl.Int64()})

    return register_io_source(my_source, schema=schema)


def test_my_source() -> None:
    assert_frame_equal(
        scan_my_source().collect(), pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    )
    assert_frame_equal(
        scan_my_source().filter(pl.col("b") > 1).collect(),
        pl.DataFrame({"a": [2, 3], "b": [2, 3]}),
    )
    assert_frame_equal(
        scan_my_source().filter(pl.col("b") > 1).select("a").collect(),
        pl.DataFrame({"a": [2, 3]}),
    )
    assert_frame_equal(
        scan_my_source().select("a").collect(), pl.DataFrame({"a": [1, 2, 3]})
    )


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_no_dynamic_predicate_pushdown_28629(engine: EngineType) -> None:
    df = pl.DataFrame({"a": range(10), "b": ["x", "y"] * 5})
    received: list[pl.Expr | None] = []

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        _batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        received.append(predicate)

        out = df
        if predicate is not None:
            out = out.filter(predicate)
        if n_rows is not None:
            out = out.head(n_rows)
        if with_columns is not None:
            out = out.select(with_columns)

        yield out

    def scan() -> pl.LazyFrame:
        return register_io_source(source, schema=df.schema)

    # `sort` + `head` inserts a dynamic predicate, which has no DSL representation
    # and so must not be pushed into the plugin.
    assert_frame_equal(scan().sort("a").head(3).collect(engine=engine), df.head(3))

    # The dynamic predicate is keyed on the same column as the user predicate, so
    # both end up in a single conjunction. Only the dynamic min-term may be dropped.
    assert_frame_equal(
        scan().filter(pl.col("a") > 4).sort("a").head(3).collect(engine=engine),
        df.filter(pl.col("a") > 4).head(3),
    )

    assert received
    assert not any("dynamic_pred" in str(p) for p in received if p is not None)
