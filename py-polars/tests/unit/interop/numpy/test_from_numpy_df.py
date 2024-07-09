from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import PolarsTemporalType


def test_from_numpy() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]])
    df = pl.from_numpy(
        data,
        schema=["a", "b"],
        orient="col",
        schema_overrides={"a": pl.UInt32, "b": pl.UInt32},
    )
    assert df.shape == (3, 2)
    assert df.rows() == [(1, 4), (2, 5), (3, 6)]
    assert df.schema == {"a": pl.UInt32, "b": pl.UInt32}
    data2 = np.array(["foo", "bar"], dtype=object)
    df2 = pl.from_numpy(data2)
    assert df2.shape == (2, 1)
    assert df2.rows() == [("foo",), ("bar",)]
    assert df2.schema == {"column_0": pl.String}
    with pytest.raises(
        ValueError,
        match="cannot create DataFrame from array with more than two dimensions",
    ):
        _ = pl.from_numpy(np.array([[[1]]]))
    with pytest.raises(
        ValueError, match="cannot create DataFrame from zero-dimensional array"
    ):
        _ = pl.from_numpy(np.array(1))


def test_from_numpy_array_value() -> None:
    df = pl.DataFrame({"A": [[2, 3]]})
    assert df.rows() == [([2, 3],)]
    assert df.schema == {"A": pl.List(pl.Int64)}


def test_construct_from_ndarray_value() -> None:
    array_cell = np.array([2, 3])
    df = pl.DataFrame(np.array([[array_cell, 4]], dtype=object))
    assert df.dtypes == [pl.Object, pl.Object]
    to_numpy = df.to_numpy()
    assert to_numpy.shape == (1, 2)
    assert_array_equal(to_numpy[0][0], array_cell)
    assert to_numpy[0][1] == 4


def test_from_numpy_nparray_value() -> None:
    array_cell = np.array([2, 3])
    df = pl.from_numpy(np.array([[array_cell, 4]], dtype=object))
    assert df.dtypes == [pl.Object, pl.Object]
    to_numpy = df.to_numpy()
    assert to_numpy.shape == (1, 2)
    assert_array_equal(to_numpy[0][0], array_cell)
    assert to_numpy[0][1] == 4


def test_from_numpy_structured() -> None:
    test_data = [
        ("Google Pixel 7", 521.90, True),
        ("Apple iPhone 14 Pro", 999.00, True),
        ("Samsung Galaxy S23 Ultra", 1199.99, False),
        ("OnePlus 11", 699.00, True),
    ]
    # create a numpy structured array...
    arr_structured = np.array(
        test_data,
        dtype=np.dtype(
            [
                ("product", "U32"),
                ("price_usd", "float64"),
                ("in_stock", "bool"),
            ]
        ),
    )
    # ...and also establish as a record array view
    arr_records = arr_structured.view(np.recarray)

    # confirm that we can cleanly initialise a DataFrame from both,
    # respecting the native dtypes and any schema overrides, etc.
    for arr in (arr_structured, arr_records):
        df = pl.DataFrame(data=arr).sort(by="price_usd", descending=True)

        assert df.schema == {
            "product": pl.String,
            "price_usd": pl.Float64,
            "in_stock": pl.Boolean,
        }
        assert df.rows() == sorted(test_data, key=lambda row: -row[1])

        for df in (
            pl.DataFrame(
                data=arr, schema=["phone", ("price_usd", pl.Float32), "available"]
            ),
            pl.DataFrame(
                data=arr,
                schema=["phone", "price_usd", "available"],
                schema_overrides={"price_usd": pl.Float32},
            ),
        ):
            assert df.schema == {
                "phone": pl.String,
                "price_usd": pl.Float32,
                "available": pl.Boolean,
            }


def test_from_numpy2() -> None:
    # note: numpy timeunit support is limited to those supported by polars.
    # as a result, datetime64[s] raises
    x = np.asarray(range(100_000, 200_000, 10_000), dtype="datetime64[s]")
    with pytest.raises(ValueError, match="Please cast to the closest supported unit"):
        pl.Series(x)


@pytest.mark.parametrize(
    ("numpy_time_unit", "expected_values", "expected_dtype"),
    [
        ("ns", ["1970-01-02T01:12:34.123456789"], pl.Datetime("ns")),
        ("us", ["1970-01-02T01:12:34.123456"], pl.Datetime("us")),
        ("ms", ["1970-01-02T01:12:34.123"], pl.Datetime("ms")),
        ("D", ["1970-01-02"], pl.Date),
    ],
)
def test_from_numpy_supported_units(
    numpy_time_unit: str,
    expected_values: list[str],
    expected_dtype: PolarsTemporalType,
) -> None:
    values = np.array(
        ["1970-01-02T01:12:34.123456789123456789"],
        dtype=f"datetime64[{numpy_time_unit}]",
    )
    result = pl.from_numpy(values)
    expected = (
        pl.Series("column_0", expected_values).str.strptime(expected_dtype).to_frame()
    )
    assert_frame_equal(result, expected)
