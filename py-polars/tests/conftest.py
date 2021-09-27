import pytest

import polars as pl


@pytest.fixture
def df() -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "bools": [False, True, False],
            "bools_nulls": [None, True, False],
            "int": [1, 2, 3],
            "int_nulls": [1, None, 3],
            "floats": [1.0, 2.0, 3.0],
            "floats_nulls": [1.0, None, 3.0],
            "strings": ["foo", "bar", "ham"],
            "strings_nulls": ["foo", None, "ham"],
            "date32": [1324, 123, 1234],
            "date64": [13241324, 12341256, 12341234],
        }
    )
    return df.with_columns(
        [pl.col("date32").cast(pl.Date32), pl.col("date64").cast(pl.Date64)]
    )


@pytest.fixture
def fruits_cars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B": [5, 4, 3, 2, 1],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        }
    )
