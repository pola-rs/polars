import io
from typing import IO

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_concat_invalid_schema_err_20355() -> None:
    lf1 = pl.LazyFrame({"x": [1], "y": [None]})
    lf2 = pl.LazyFrame({"y": [1]})
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.concat([lf1, lf2]).collect(engine="streaming")


def test_concat_df() -> None:
    df1 = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    df2 = pl.concat([df1, df1], rechunk=True)

    assert df2.shape == (6, 3)
    assert df2.n_chunks() == 1
    assert df2.rows() == df1.rows() + df1.rows()
    assert pl.concat([df1, df1], rechunk=False).n_chunks() == 2

    # concat from generator of frames
    df3 = pl.concat(items=(df1 for _ in range(2)))
    assert_frame_equal(df2, df3)

    # check that df4 is not modified following concat of itself
    df4 = pl.from_records(((1, 2), (1, 2)))
    _ = pl.concat([df4, df4, df4])

    assert df4.shape == (2, 2)
    assert df4.rows() == [(1, 1), (2, 2)]

    # misc error conditions
    with pytest.raises(ValueError):
        _ = pl.concat([])

    with pytest.raises(ValueError):
        pl.concat([df1, df1], how="rubbish")  # type: ignore[arg-type]


def test_concat_to_empty() -> None:
    assert pl.concat([pl.DataFrame([]), pl.DataFrame({"a": [1]})]).to_dict(
        as_series=False
    ) == {"a": [1]}


def test_concat_multiple_parquet_inmem() -> None:
    f = io.BytesIO()
    g = io.BytesIO()

    df1 = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["xyz", "abc", "wow"],
        }
    )
    df2 = pl.DataFrame(
        {
            "a": [5, 6, 7],
            "b": ["a", "few", "entries"],
        }
    )

    dfs = pl.concat([df1, df2])

    df1.write_parquet(f)
    df2.write_parquet(g)

    f.seek(0)
    g.seek(0)

    items: list[IO[bytes]] = [f, g]
    assert_frame_equal(pl.read_parquet(items), dfs)

    f.seek(0)
    g.seek(0)

    assert_frame_equal(pl.read_parquet(items, use_pyarrow=True), dfs)

    f.seek(0)
    g.seek(0)

    fb = f.read()
    gb = g.read()

    assert_frame_equal(pl.read_parquet([fb, gb]), dfs)
    assert_frame_equal(pl.read_parquet([fb, gb], use_pyarrow=True), dfs)


def test_concat_series() -> None:
    s = pl.Series("a", [2, 1, 3])

    assert pl.concat([s, s]).len() == 6
    # check if s remains unchanged
    assert s.len() == 3


def test_concat_null_20501() -> None:
    a = pl.DataFrame({"id": [1], "value": ["foo"]})
    b = pl.DataFrame({"id": [2], "value": [None]})

    assert pl.concat([a.lazy(), b.lazy()]).collect().to_dict(as_series=False) == {
        "id": [1, 2],
        "value": ["foo", None],
    }
