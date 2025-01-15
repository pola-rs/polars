"""Tests for ``.list.index_of_in()``."""

import polars as pl
from polars.testing import assert_frame_equal


def test_index_of_in_from_constant() -> None:
    df = pl.DataFrame({"lists": [[3, 1], [2, 4], [5, 3, 1]]})
    assert_frame_equal(
        df.select(pl.col("lists").list.index_of_in(1)),
        pl.DataFrame({"lists": [1, None, 2]}, schema={"lists": pl.get_index_type()}),
    )


def test_index_of_in_from_column() -> None:
    df = pl.DataFrame({"lists": [[3, 1], [2, 4], [5, 3, 1]], "values": [1, 2, 6]})
    assert_frame_equal(
        df.select(pl.col("lists").list.index_of_in(pl.col("values"))),
        pl.DataFrame({"lists": [1, 0, None]}, schema={"lists": pl.get_index_type()}),
    )
