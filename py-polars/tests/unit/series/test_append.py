import pytest

import polars as pl
from polars.testing import assert_series_equal


def test_append_extend() -> None:
    a = pl.Series("a", [1, 2])
    b = pl.Series("b", [8, 9, None])

    a.append(b, append_chunks=False)
    expected = pl.Series("a", [1, 2, 8, 9, None])

    assert_series_equal(a, expected)
    assert a.n_chunks() == 1


def test_append_self_3915() -> None:
    s = pl.Series("s", [1, 2, 3])
    assert s.append(s).to_list() == [1, 2, 3, 1, 2, 3]


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
        pl.SchemaError,
        match=(
            'cannot append field with name "address" '
            'to struct with field name "city"'
        ),
    ):
        housing1.append(housing2, append_chunks=True)
    with pytest.raises(
        pl.SchemaError,
        match=(
            'cannot extend field with name "address" '
            'to struct with field name "city"'
        ),
    ):
        housing1.append(housing2, append_chunks=False)
