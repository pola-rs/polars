import pytest

import polars as pl
from polars.exceptions import SchemaError
from polars.testing import assert_series_equal


def test_append() -> None:
    a = pl.Series("a", [1, 2])
    b = pl.Series("b", [8, 9, None])

    result = a.append(b)

    expected = pl.Series("a", [1, 2, 8, 9, None])
    assert_series_equal(a, expected)
    assert_series_equal(result, expected)
    assert a.n_chunks() == 2


def test_append_self_3915() -> None:
    a = pl.Series("a", [1, 2])

    a.append(a)

    expected = pl.Series("a", [1, 2, 1, 2])
    assert_series_equal(a, expected)
    assert a.n_chunks() == 2


def test_append_bad_input() -> None:
    a = pl.Series("a", [1, 2])
    b = a.to_frame()

    with pytest.raises(AttributeError):
        a.append(b)  # type: ignore[arg-type]


def test_struct_schema_on_append_extend_3452() -> None:
    housing1_data = [
        {
            "city": "Chicago",
            "address": "100 Main St",
            "price": 250000,
            "nbr_bedrooms": 3,
        },
        {
            "city": "New York",
            "address": "100 First Ave",
            "price": 450000,
            "nbr_bedrooms": 2,
        },
    ]

    housing2_data = [
        {
            "address": "303 Mockingbird Lane",
            "city": "Los Angeles",
            "nbr_bedrooms": 2,
            "price": 450000,
        },
        {
            "address": "404 Moldave Dr",
            "city": "Miami Beach",
            "nbr_bedrooms": 1,
            "price": 250000,
        },
    ]
    housing1, housing2 = pl.Series(housing1_data), pl.Series(housing2_data)
    with pytest.raises(
        SchemaError,
    ):
        housing1.append(housing2)

    with pytest.raises(
        SchemaError,
    ):
        housing1.extend(housing2)


def test_append_null_series() -> None:
    a = pl.Series("a", [1, 2], pl.Int64)
    b = pl.Series("b", [None, None], pl.Null)

    result = a.append(b)

    expected = pl.Series("a", [1, 2, None, None], pl.Int64)
    assert_series_equal(a, expected)
    assert_series_equal(result, expected)
    assert a.n_chunks() == 2
