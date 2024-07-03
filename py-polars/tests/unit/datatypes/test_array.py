import datetime
from datetime import timedelta
from typing import Any

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal


def test_cast_list_array() -> None:
    payload = [[1, 2, 3], [4, 2, 3]]
    s = pl.Series(payload)

    dtype = pl.Array(pl.Int64, 3)
    out = s.cast(dtype)
    assert out.dtype == dtype
    assert out.to_list() == payload
    assert_series_equal(out.cast(pl.List(pl.Int64)), s)

    # width is incorrect
    with pytest.raises(
        ComputeError, match=r"not all elements have the specified width"
    ):
        s.cast(pl.Array(pl.Int64, 2))


def test_array_in_group_by() -> None:
    df = pl.DataFrame(
        [
            pl.Series("id", [1, 2]),
            pl.Series("list", [[1, 2], [5, 5]], dtype=pl.Array(pl.UInt8, 2)),
        ]
    )

    result = next(iter(df.group_by(["id"], maintain_order=True)))[1]["list"]
    assert result.to_list() == [[1, 2]]

    df = pl.DataFrame(
        {"a": [[1, 2], [2, 2], [1, 4]], "g": [1, 1, 2]},
        schema={"a": pl.Array(pl.Int64, 2), "g": pl.Int64},
    )

    out0 = df.group_by("g").agg(pl.col("a")).sort("g")
    out1 = df.set_sorted("g").group_by("g").agg(pl.col("a"))

    for out in [out0, out1]:
        assert out.schema == {
            "g": pl.Int64,
            "a": pl.List(pl.Array(pl.Int64, 2)),
        }
        assert out.to_dict(as_series=False) == {
            "g": [1, 2],
            "a": [[[1, 2], [2, 2]], [[1, 4]]],
        }


def test_array_invalid_operation() -> None:
    s = pl.Series(
        [[1, 2], [8, 9]],
        dtype=pl.Array(pl.Int32, 2),
    )
    with pytest.raises(
        InvalidOperationError,
        match=r"`sign` operation not supported for dtype `array\[",
    ):
        s.sign()


def test_array_concat() -> None:
    a_df = pl.DataFrame({"a": [[0, 1], [1, 0]]}).select(
        pl.col("a").cast(pl.Array(pl.Int32, 2))
    )
    b_df = pl.DataFrame({"a": [[1, 1], [0, 0]]}).select(
        pl.col("a").cast(pl.Array(pl.Int32, 2))
    )
    assert pl.concat([a_df, b_df]).to_dict(as_series=False) == {
        "a": [[0, 1], [1, 0], [1, 1], [0, 0]]
    }


def test_array_equal_and_not_equal() -> None:
    left = pl.Series([[1, 2], [3, 5]], dtype=pl.Array(pl.Int64, 2))
    right = pl.Series([[1, 2], [3, 1]], dtype=pl.Array(pl.Int64, 2))
    assert_series_equal(left == right, pl.Series([True, False]))
    assert_series_equal(left.eq_missing(right), pl.Series([True, False]))
    assert_series_equal(left != right, pl.Series([False, True]))
    assert_series_equal(left.ne_missing(right), pl.Series([False, True]))

    left = pl.Series([[1, None], [3, None]], dtype=pl.Array(pl.Int64, 2))
    right = pl.Series([[1, None], [3, 4]], dtype=pl.Array(pl.Int64, 2))
    assert_series_equal(left == right, pl.Series([True, False]))
    assert_series_equal(left.eq_missing(right), pl.Series([True, False]))
    assert_series_equal(left != right, pl.Series([False, True]))
    assert_series_equal(left.ne_missing(right), pl.Series([False, True]))

    # TODO: test eq_missing with nulled arrays, rather than null elements.


def test_array_list_supertype() -> None:
    s1 = pl.Series([[1, 2], [3, 4]], dtype=pl.Array(pl.Int64, 2))
    s2 = pl.Series([[1.0, 2.0], [3.0, 4.5]], dtype=pl.List(inner=pl.Float64))

    result = s1 == s2

    expected = pl.Series([True, False])
    assert_series_equal(result, expected)


def test_array_in_list() -> None:
    s = pl.Series(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        dtype=pl.List(pl.Array(pl.Int8, 2)),
    )
    assert s.dtype == pl.List(pl.Array(pl.Int8, 2))


def test_array_data_type_equality() -> None:
    assert pl.Array(pl.Int64, 2) == pl.Array
    assert pl.Array(pl.Int64, 2) == pl.Array(pl.Int64, 2)
    assert pl.Array(pl.Int64, 2) != pl.Array(pl.Int64, 3)
    assert pl.Array(pl.Int64, 2) != pl.Array(pl.String, 2)
    assert pl.Array(pl.Int64, 2) != pl.List(pl.Int64)

    assert pl.Array(pl.Int64, (4, 2)) == pl.Array
    assert pl.Array(pl.Array(pl.Int64, 2), 4) == pl.Array(pl.Int64, (4, 2))
    assert pl.Array(pl.Int64, (4, 2)) == pl.Array(pl.Int64, (4, 2))
    assert pl.Array(pl.Int64, (4, 2)) != pl.Array(pl.String, (4, 2))
    assert pl.Array(pl.Int64, (4, 2)) != pl.Array(pl.Int64, 4)
    assert pl.Array(pl.Int64, (4,)) != pl.Array(pl.Int64, (4, 2))


@pytest.mark.parametrize(
    ("data", "inner_type"),
    [
        ([[1, 2], None, [3, None], [None, None]], pl.Int64),
        ([[True, False], None, [True, None], [None, None]], pl.Boolean),
        ([[1.0, 2.0], None, [3.0, None], [None, None]], pl.Float32),
        ([["a", "b"], None, ["c", None], [None, None]], pl.String),
        (
            [
                [datetime.datetime(2021, 1, 1), datetime.datetime(2022, 1, 1, 10, 30)],
                None,
                [datetime.datetime(2023, 12, 25), None],
                [None, None],
            ],
            pl.Datetime,
        ),
        (
            [
                [datetime.date(2021, 1, 1), datetime.date(2022, 1, 15)],
                None,
                [datetime.date(2023, 12, 25), None],
                [None, None],
            ],
            pl.Date,
        ),
        (
            [
                [datetime.timedelta(10), datetime.timedelta(1, 22)],
                None,
                [datetime.timedelta(20), None],
                [None, None],
            ],
            pl.Duration,
        ),
        ([[[1, 2], None], None, [[3], None], [None, None]], pl.List(pl.Int32)),
    ],
)
def test_cast_list_to_array(data: Any, inner_type: pl.DataType) -> None:
    s = pl.Series(data, dtype=pl.List(inner_type))
    s = s.cast(pl.Array(inner_type, 2))
    assert s.dtype == pl.Array(inner_type, shape=2)
    assert s.to_list() == data


@pytest.fixture()
def data_dispersion() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "int": [[1, 2, 3, 4, 5]],
            "float": [[1.0, 2.0, 3.0, 4.0, 5.0]],
            "duration": [[1000, 2000, 3000, 4000, 5000]],
        },
        schema={
            "int": pl.Array(pl.Int64, 5),
            "float": pl.Array(pl.Float64, 5),
            "duration": pl.Array(pl.Duration, 5),
        },
    )


def test_arr_var(data_dispersion: pl.DataFrame) -> None:
    df = data_dispersion

    result = df.select(
        pl.col("int").arr.var().name.suffix("_var"),
        pl.col("float").arr.var().name.suffix("_var"),
        pl.col("duration").arr.var().name.suffix("_var"),
    )

    expected = pl.DataFrame(
        [
            pl.Series("int_var", [2.5], dtype=pl.Float64),
            pl.Series("float_var", [2.5], dtype=pl.Float64),
            pl.Series(
                "duration_var",
                [timedelta(microseconds=2000)],
                dtype=pl.Duration(time_unit="ms"),
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_arr_std(data_dispersion: pl.DataFrame) -> None:
    df = data_dispersion

    result = df.select(
        pl.col("int").arr.std().name.suffix("_std"),
        pl.col("float").arr.std().name.suffix("_std"),
        pl.col("duration").arr.std().name.suffix("_std"),
    )

    expected = pl.DataFrame(
        [
            pl.Series("int_std", [1.5811388300841898], dtype=pl.Float64),
            pl.Series("float_std", [1.5811388300841898], dtype=pl.Float64),
            pl.Series(
                "duration_std",
                [timedelta(microseconds=1581)],
                dtype=pl.Duration(time_unit="us"),
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_arr_median(data_dispersion: pl.DataFrame) -> None:
    df = data_dispersion

    result = df.select(
        pl.col("int").arr.median().name.suffix("_median"),
        pl.col("float").arr.median().name.suffix("_median"),
        pl.col("duration").arr.median().name.suffix("_median"),
    )

    expected = pl.DataFrame(
        [
            pl.Series("int_median", [3.0], dtype=pl.Float64),
            pl.Series("float_median", [3.0], dtype=pl.Float64),
            pl.Series(
                "duration_median",
                [timedelta(microseconds=3000)],
                dtype=pl.Duration(time_unit="us"),
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_array_repeat() -> None:
    dtype = pl.Array(pl.UInt8, shape=1)
    s = pl.repeat([42], n=3, dtype=dtype, eager=True)
    expected = pl.Series("repeat", [[42], [42], [42]], dtype=dtype)
    assert s.dtype == dtype
    assert_series_equal(s, expected)


def test_create_nested_array() -> None:
    data = [[[1, 2], [3]], [[], [4, None]], None]
    s1 = pl.Series(data, dtype=pl.Array(pl.List(pl.Int64), 2))
    assert s1.to_list() == data
    data = [[[1, 2], [3, None]], [[None, None], [4, None]], None]
    s2 = pl.Series(
        [[[1, 2], [3, None]], [[None, None], [4, None]], None],
        dtype=pl.Array(pl.Array(pl.Int64, 2), 2),
    )
    assert s2.to_list() == data


def test_recursive_array_dtype() -> None:
    assert str(pl.Array(pl.Int64, (2, 3))) == "Array(Int64, shape=(2, 3))"
    assert str(pl.Array(pl.Int64, 3)) == "Array(Int64, shape=(3,))"
    dtype = pl.Array(pl.Int64, 3)
    s = pl.Series(np.arange(6).reshape((2, 3)), dtype=dtype)
    assert s.dtype == dtype
    assert s.len() == 2


def test_ndarray_construction() -> None:
    a = np.arange(16, dtype=np.int64).reshape((2, 4, -1))
    s = pl.Series(a)
    assert s.dtype == pl.Array(pl.Int64, (4, 2))
    assert (s.to_numpy() == a).all()


def test_array_width_deprecated() -> None:
    with pytest.deprecated_call():
        dtype = pl.Array(pl.Int8, width=2)
    with pytest.deprecated_call():
        assert dtype.width == 2


def test_array_inner_recursive() -> None:
    shape = (2, 3, 4, 5)
    dtype = pl.Array(int, shape=shape)
    for dim in shape:
        assert dtype.size == dim
        dtype = dtype.inner  # type: ignore[assignment]


def test_array_inner_recursive_python_dtype() -> None:
    dtype = pl.Array(int, shape=(2, 3))
    assert dtype.inner.inner == pl.Int64  # type: ignore[union-attr]


def test_array_missing_shape() -> None:
    with pytest.raises(TypeError):
        pl.Array(pl.Int8)
