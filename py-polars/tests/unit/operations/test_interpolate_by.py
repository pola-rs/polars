from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import column, dataframes

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


@pytest.mark.parametrize(
    "times_dtype",
    [
        pl.Datetime("ms"),
        pl.Datetime("us", "Asia/Kathmandu"),
        pl.Datetime("ns"),
        pl.Date,
        pl.Int64,
        pl.Int32,
        pl.UInt64,
        pl.UInt32,
        pl.Float32,
        pl.Float64,
    ],
)
@pytest.mark.parametrize(
    "values_dtype",
    [
        pl.Float64,
        pl.Float32,
        pl.Int64,
        pl.Int32,
        pl.UInt64,
        pl.UInt32,
    ],
)
def test_interpolate_by(
    values_dtype: PolarsDataType, times_dtype: PolarsDataType
) -> None:
    df = pl.DataFrame(
        {
            "times": [
                1,
                3,
                10,
                11,
                12,
                16,
                21,
                30,
            ],
            "values": [1, None, None, 5, None, None, None, 6],
        },
        schema={"times": times_dtype, "values": values_dtype},
    )
    result = df.select(pl.col("values").interpolate_by("times"))
    expected = pl.DataFrame(
        {
            "values": [
                1.0,
                1.7999999999999998,
                4.6,
                5.0,
                5.052631578947368,
                5.2631578947368425,
                5.526315789473684,
                6.0,
            ]
        }
    )
    if values_dtype == pl.Float32:
        expected = expected.select(pl.col("values").cast(pl.Float32))
    assert_frame_equal(result, expected)
    result = (
        df.sort("times", descending=True)
        .with_columns(pl.col("values").interpolate_by("times"))
        .sort("times")
        .drop("times")
    )
    assert_frame_equal(result, expected)


def test_interpolate_by_leading_nulls() -> None:
    df = pl.DataFrame(
        {
            "times": [
                date(2020, 1, 1),
                date(2020, 1, 1),
                date(2020, 1, 1),
                date(2020, 1, 1),
                date(2020, 1, 3),
                date(2020, 1, 10),
                date(2020, 1, 11),
            ],
            "values": [None, None, None, 1, None, None, 5],
        }
    )
    result = df.select(pl.col("values").interpolate_by("times"))
    expected = pl.DataFrame(
        {"values": [None, None, None, 1.0, 1.7999999999999998, 4.6, 5.0]}
    )
    assert_frame_equal(result, expected)
    result = (
        df.sort("times", descending=True)
        .with_columns(pl.col("values").interpolate_by("times"))
        .sort("times")
        .drop("times")
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("dataset", ["floats", "dates"])
def test_interpolate_by_trailing_nulls(dataset: str) -> None:
    input_data = {
        "dates": pl.DataFrame(
            {
                "times": [
                    date(2020, 1, 1),
                    date(2020, 1, 3),
                    date(2020, 1, 10),
                    date(2020, 1, 11),
                    date(2020, 1, 12),
                    date(2020, 1, 13),
                ],
                "values": [1, None, None, 5, None, None],
            }
        ),
        "floats": pl.DataFrame(
            {
                "times": [0.2, 0.4, 0.5, 0.6, 0.9, 1.1],
                "values": [1, None, None, 5, None, None],
            }
        ),
    }

    expected_data = {
        "dates": pl.DataFrame(
            {"values": [1.0, 1.7999999999999998, 4.6, 5.0, None, None]}
        ),
        "floats": pl.DataFrame({"values": [1.0, 3.0, 4.0, 5.0, None, None]}),
    }

    df = input_data[dataset]
    expected = expected_data[dataset]

    result = df.select(pl.col("values").interpolate_by("times"))

    assert_frame_equal(result, expected)
    result = (
        df.sort("times", descending=True)
        .with_columns(pl.col("values").interpolate_by("times"))
        .sort("times")
        .drop("times")
    )
    assert_frame_equal(result, expected)


@given(data=st.data(), x_dtype=st.sampled_from([pl.Date, pl.Float64]))
def test_interpolate_vs_numpy(data: st.DataObject, x_dtype: pl.DataType) -> None:
    if x_dtype == pl.Float64:
        by_strategy = st.floats(
            min_value=-1e150,
            max_value=1e150,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    else:
        by_strategy = None

    dataframe = (
        data.draw(
            dataframes(
                [
                    column(
                        "ts",
                        dtype=x_dtype,
                        allow_null=False,
                        strategy=by_strategy,
                    ),
                    column(
                        "value",
                        dtype=pl.Float64,
                        allow_null=True,
                    ),
                ],
                min_size=1,
            )
        )
        .sort("ts")
        .fill_nan(None)
        .unique("ts")
    )

    if x_dtype == pl.Float64:
        assume(not dataframe["ts"].is_nan().any())
        assume(not dataframe["ts"].is_null().any())
        assume(not dataframe["ts"].is_in([float("-inf"), float("inf")]).any())

    assume(not dataframe["value"].is_null().all())
    assume(not dataframe["value"].is_in([float("-inf"), float("inf")]).any())

    dataframe = dataframe.sort("ts")

    result = dataframe.select(pl.col("value").interpolate_by("ts"))["value"]

    mask = dataframe["value"].is_not_null()

    np_dtype = "int64" if x_dtype == pl.Date else "float64"
    x = dataframe["ts"].to_numpy().astype(np_dtype)
    xp = dataframe["ts"].filter(mask).to_numpy().astype(np_dtype)
    yp = dataframe["value"].filter(mask).to_numpy().astype("float64")
    interp = np.interp(x, xp, yp)
    # Polars preserves nulls on boundaries, but NumPy doesn't.
    first_non_null = dataframe["value"].is_not_null().arg_max()
    last_non_null = len(dataframe) - dataframe["value"][::-1].is_not_null().arg_max()  # type: ignore[operator]
    interp[:first_non_null] = float("nan")
    interp[last_non_null:] = float("nan")
    expected = dataframe.with_columns(value=pl.Series(interp, nan_to_null=True))[
        "value"
    ]

    assert_series_equal(result, expected)
    result_from_unsorted = (
        dataframe.sort("ts", descending=True)
        .with_columns(pl.col("value").interpolate_by("ts"))
        .sort("ts")["value"]
    )
    assert_series_equal(result_from_unsorted, expected)


def test_interpolate_by_invalid() -> None:
    s = pl.Series([1, None, 3])
    by = pl.Series([1, 2])
    with pytest.raises(InvalidOperationError, match=r"\(3\), got 2"):
        s.interpolate_by(by)

    by = pl.Series([1, None, 3])
    with pytest.raises(
        InvalidOperationError,
        match="null values in `by` column are not yet supported in 'interpolate_by'",
    ):
        s.interpolate_by(by)
