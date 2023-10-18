import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_series_equal


def test_cast_list_array() -> None:
    payload = [[1, 2, 3], [4, 2, 3]]
    s = pl.Series(payload)

    dtype = pl.Array(width=3, inner=pl.Int64)
    out = s.cast(dtype)
    assert out.dtype == dtype
    assert out.to_list() == payload
    assert_series_equal(out.cast(pl.List(pl.Int64)), s)

    # width is incorrect
    with pytest.raises(
        pl.ComputeError,
        match=r"incompatible `offsets` in source `list`",
    ):
        s.cast(pl.Array(width=2, inner=pl.Int64))


def test_array_construction() -> None:
    payload = [[1, 2, 3], [4, 2, 3]]

    dtype = pl.Array(width=3, inner=pl.Int64)
    s = pl.Series(payload, dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == payload

    # inner type
    dtype = pl.Array(2, pl.UInt8)
    payload = [[1, 2], [3, 4]]
    s = pl.Series(payload, dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == payload

    # create using schema
    df = pl.DataFrame(
        schema={
            "a": pl.Array(width=3, inner=pl.Float32),
            "b": pl.Array(width=5, inner=pl.Datetime("ms")),
        }
    )
    assert df.dtypes == [
        pl.Array(width=3, inner=pl.Float32),
        pl.Array(width=5, inner=pl.Datetime("ms")),
    ]
    assert df.rows() == []


def test_array_in_group_by() -> None:
    df = pl.DataFrame(
        [
            pl.Series("id", [1, 2]),
            pl.Series("list", [[1, 2], [5, 5]], dtype=pl.Array(2, pl.UInt8)),
        ]
    )

    assert next(iter(df.group_by("id", maintain_order=True)))[1]["list"].to_list() == [
        [1, 2]
    ]

    df = pl.DataFrame(
        {"a": [[1, 2], [2, 2], [1, 4]], "g": [1, 1, 2]},
        schema={"a": pl.Array(inner=pl.Int64, width=2), "g": pl.Int64},
    )

    out0 = df.group_by("g").agg(pl.col("a")).sort("g")
    out1 = df.set_sorted("g").group_by("g").agg(pl.col("a"))

    for out in [out0, out1]:
        assert out.schema == {
            "g": pl.Int64,
            "a": pl.List(pl.Array(inner=pl.Int64, width=2)),
        }
        assert out.to_dict(False) == {"g": [1, 2], "a": [[[1, 2], [2, 2]], [[1, 4]]]}


def test_array_invalid_operation() -> None:
    s = pl.Series(
        [[1, 2], [8, 9]],
        dtype=pl.Array(width=2, inner=pl.Int32),
    )
    with pytest.raises(
        InvalidOperationError,
        match=r"`sign` operation not supported for dtype `array\[",
    ):
        s.sign()


def test_array_concat() -> None:
    a_df = pl.DataFrame({"a": [[0, 1], [1, 0]]}).select(
        pl.col("a").cast(pl.Array(width=2, inner=pl.Int32))
    )
    b_df = pl.DataFrame({"a": [[1, 1], [0, 0]]}).select(
        pl.col("a").cast(pl.Array(width=2, inner=pl.Int32))
    )
    assert pl.concat([a_df, b_df]).to_dict(False) == {
        "a": [[0, 1], [1, 0], [1, 1], [0, 0]]
    }
