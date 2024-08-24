from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from typing import Iterator


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
