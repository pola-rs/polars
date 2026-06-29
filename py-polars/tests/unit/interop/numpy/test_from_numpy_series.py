from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import polars as pl

if TYPE_CHECKING:
    from polars._typing import TimeUnit


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_from_numpy_timedelta(time_unit: TimeUnit) -> None:
    s = pl.Series(
        "name",
        np.array(
            [timedelta(days=1), timedelta(seconds=1)], dtype=f"timedelta64[{time_unit}]"
        ),
    )
    assert s.dtype == pl.Duration(time_unit)
    assert s.name == "name"
    assert s.dt[0] == timedelta(days=1)
    assert s.dt[1] == timedelta(seconds=1)


def test_from_numpy_records() -> None:
    # numpy arrays in dicts/records
    arr_int = np.array([1, 2, 3], dtype=np.int64)
    arr_float = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    arr_bool = np.array([True, False, True], dtype=np.bool_)

    s = pl.Series(
        name="data",
        values=[{"ints": arr_int, "floats": arr_float, "bools": arr_bool}],
    )
    assert s.dtype == pl.Struct(
        {
            "ints": pl.List(pl.Int64),
            "floats": pl.List(pl.Float64),
            "bools": pl.List(pl.Boolean),
        }
    )
    round_trip_array = s.to_frame().unnest("data").row(0)
    assert_array_equal(
        round_trip_array,
        [arr_int, arr_float, arr_bool],
    )

    data = [
        {"id": 1, "values": np.array([1, 2, 3], dtype=np.int64)},
        {"id": 2, "values": np.array([4, 5, 6], dtype=np.int64)},
        {"id": 3, "values": np.array([7, 8, 9], dtype=np.int64)},
    ]
    s = pl.Series("data", data)
    assert s.dtype == pl.Struct({"id": pl.Int64, "values": pl.List(pl.Int64)})
    assert len(s) == 3


@pytest.mark.parametrize(
    ("numpy_dtype", "polars_dtype"),
    [
        (np.int8, pl.Int8),
        (np.int16, pl.Int16),
        (np.int32, pl.Int32),
        (np.int64, pl.Int64),
        (np.uint8, pl.UInt8),
        (np.uint16, pl.UInt16),
        (np.uint32, pl.UInt32),
        (np.uint64, pl.UInt64),
        (np.float16, pl.Float16),
        (np.float32, pl.Float32),
        (np.float64, pl.Float64),
        (np.bool_, pl.Boolean),
    ],
)
def test_from_numpy_records_2d(
    numpy_dtype: type[np.generic], polars_dtype: pl.DataType
) -> None:
    arr2d = np.array([[0, 1], [2, 3]], dtype=numpy_dtype)
    s = pl.Series("data", [{"id": 1, "values": arr2d}])

    assert s.dtype == pl.Struct(
        {"id": pl.Int64, "values": pl.List(pl.Array(polars_dtype, shape=(2,)))}
    )
    expected_array_values = (
        [[False, True], [True, True]]
        if polars_dtype == pl.Boolean
        else [[0, 1], [2, 3]]  # type: ignore[list-item]
    )
    assert s[0] == {"id": 1, "values": expected_array_values}

    round_trip_array = s.to_numpy()[0][1]
    assert_array_equal(round_trip_array, arr2d)
