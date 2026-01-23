from pathlib import Path
from typing import cast

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_collect_all_type_coercion_21805() -> None:
    df = pl.LazyFrame({"A": [1.0, 2.0]})
    df = df.with_columns(pl.col("A").shift().fill_null(2))
    assert pl.collect_all([df])[0]["A"].to_list() == [2.0, 1.0]


@pytest.mark.parametrize("optimizations", [pl.QueryOptFlags(), pl.QueryOptFlags.none()])
def test_collect_all(df: pl.DataFrame, optimizations: pl.QueryOptFlags) -> None:
    lf1 = df.lazy().select(pl.col("int").sum())
    lf2 = df.lazy().select((pl.col("floats") * 2).sum())
    out = pl.collect_all([lf1, lf2], optimizations=optimizations)
    assert cast("int", out[0].item()) == 6
    assert cast("float", out[1].item()) == 12.0


def test_collect_all_issue_26097() -> None:
    data = pl.DataFrame({"A": [1]})
    tmp_file = "/tmp/polars-bug-repr.parquet"
    data.write_parquet(tmp_file)

    df = pl.scan_parquet(tmp_file).select([pl.col("A")])

    dummy_df = pl.DataFrame({"v": [1]}).lazy().select(pl.len())
    results = pl.collect_all([dummy_df, df])

    expected = pl.DataFrame({"A": [1]})
    assert_frame_equal(results[1], expected)

    Path.unlink(tmp_file)
