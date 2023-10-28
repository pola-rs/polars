import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_series_equal


def test_cast_list_array() -> None:
    payload = [[1, 2, 3], [4, 2, 3]]
    s = pl.Series(payload)

    dtype = pl.Array(inner=pl.Int64, width=3)
    out = s.cast(dtype)
    assert out.dtype == dtype
    assert out.to_list() == payload
    assert_series_equal(out.cast(pl.List(pl.Int64)), s)

    # width is incorrect
    with pytest.raises(
        pl.ComputeError,
        match=r"incompatible offsets in source list",
    ):
        s.cast(pl.Array(inner=pl.Int64, width=2))


def test_array_construction() -> None:
    payload = [[1, 2, 3], [4, 2, 3]]

    dtype = pl.Array(inner=pl.Int64, width=3)
    s = pl.Series(payload, dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == payload

    # inner type
    dtype = pl.Array(inner=pl.UInt8, width=2)
    payload = [[1, 2], [3, 4]]
    s = pl.Series(payload, dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == payload

    # create using schema
    df = pl.DataFrame(
        schema={
            "a": pl.Array(inner=pl.Float32, width=3),
            "b": pl.Array(inner=pl.Datetime("ms"), width=5),
        }
    )
    assert df.dtypes == [
        pl.Array(inner=pl.Float32, width=3),
        pl.Array(inner=pl.Datetime("ms"), width=5),
    ]
    assert df.rows() == []


def test_array_in_group_by() -> None:
    df = pl.DataFrame(
        [
            pl.Series("id", [1, 2]),
            pl.Series(
                "list", [[1, 2], [5, 5]], dtype=pl.Array(inner=pl.UInt8, width=2)
            ),
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
        dtype=pl.Array(inner=pl.Int32, width=2),
    )
    with pytest.raises(
        InvalidOperationError,
        match=r"`sign` operation not supported for dtype `array\[",
    ):
        s.sign()


def test_array_concat() -> None:
    a_df = pl.DataFrame({"a": [[0, 1], [1, 0]]}).select(
        pl.col("a").cast(pl.Array(inner=pl.Int32, width=2))
    )
    b_df = pl.DataFrame({"a": [[1, 1], [0, 0]]}).select(
        pl.col("a").cast(pl.Array(inner=pl.Int32, width=2))
    )
    assert pl.concat([a_df, b_df]).to_dict(False) == {
        "a": [[0, 1], [1, 0], [1, 1], [0, 0]]
    }


def test_array_equal_and_not_equal() -> None:
    left = pl.Series([[1, 2], [3, 5]], dtype=pl.Array(width=2, inner=pl.Int64))
    right = pl.Series([[1, 2], [3, 1]], dtype=pl.Array(width=2, inner=pl.Int64))
    assert_series_equal(left == right, pl.Series([True, False]))
    assert_series_equal(left.eq_missing(right), pl.Series([True, False]))
    assert_series_equal(left != right, pl.Series([False, True]))
    assert_series_equal(left.ne_missing(right), pl.Series([False, True]))

    left = pl.Series([[1, None], [3, None]], dtype=pl.Array(width=2, inner=pl.Int64))
    right = pl.Series([[1, None], [3, 4]], dtype=pl.Array(width=2, inner=pl.Int64))
    assert_series_equal(left == right, pl.Series([False, False]))
    assert_series_equal(left.eq_missing(right), pl.Series([True, False]))
    assert_series_equal(left != right, pl.Series([True, True]))
    assert_series_equal(left.ne_missing(right), pl.Series([False, True]))


def test_array_init_deprecation() -> None:
    with pytest.deprecated_call():
        pl.Array(2)
    with pytest.deprecated_call():
        pl.Array(2, pl.Utf8)
    with pytest.deprecated_call():
        pl.Array(2, inner=pl.Utf8)
    with pytest.deprecated_call():
        pl.Array(width=2)


def test_array_list_supertype() -> None:
    s1 = pl.Series([[1, 2], [3, 4]], dtype=pl.Array(width=2, inner=pl.Int64))
    s2 = pl.Series([[1.0, 2.0], [3.0, 4.5]], dtype=pl.List(inner=pl.Float64))

    result = s1 == s2

    expected = pl.Series([True, False])
    assert_series_equal(result, expected)
